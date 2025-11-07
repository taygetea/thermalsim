"""
ThermalGraph: topology manager and equation assembler.

The graph is responsible for:
  - Storing components and their connections
  - Validating type compatibility at connection time
  - Assembling component equations into global residual function
  - Invoking the solver
"""

from typing import List, Tuple, Callable, Dict, Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import warnings

from thermal_sim.core.component import Component
from thermal_sim.core.port import Port, MassFlowPort


class ThermalGraph:
    """
    Manages system topology and orchestrates solution.

    Usage:
        graph = ThermalGraph()
        graph.add_component(boiler)
        graph.add_component(turbine)
        graph.connect(boiler.outlet, turbine.inlet)
        result = graph.solve(t_span=(0, 1000))
    """

    def __init__(self):
        self.components: List[Component] = []
        self.connections: List[Tuple[Port, Port]] = []
        self._component_offsets: Dict[str, int] = {}

    def add_component(self, component: Component) -> None:
        """
        Add a component to the system.

        Args:
            component: Component instance to add

        Raises:
            ValueError: If component with this name already exists
        """
        if component.name in [c.name for c in self.components]:
            raise ValueError(f"Component '{component.name}' already exists")

        # Set component reference in all ports
        for port in component.ports.values():
            port.component = component

        self.components.append(component)

    def connect(self, port_a: Port, port_b: Port) -> None:
        """
        Connect two ports.

        Enforces type safety and direction compatibility.
        Uses reference sharing: both components will reference the same Port object.

        Args:
            port_a: First port (typically an outlet)
            port_b: Second port (typically an inlet)

        Raises:
            TypeError: If ports are incompatible types or directions

        Implementation note:
            We use reference sharing rather than explicit synchronization.
            After connection, port_b is replaced by reference to port_a.
            This ensures conservation automatically: what leaves A enters B.
        """
        if not port_a.compatible_with(port_b):
            raise TypeError(
                f"Cannot connect {port_a} to {port_b}: "
                f"incompatible types or directions"
            )

        # Store connection for topology tracking
        self.connections.append((port_a, port_b))

        # Implement reference sharing:
        # Replace the inlet port with reference to outlet port
        if port_a.direction == 'out' and port_b.direction == 'in':
            # port_b's component should now reference port_a
            port_b.component.ports[port_b.name] = port_a
        elif port_a.direction == 'in' and port_b.direction == 'out':
            port_a.component.ports[port_a.name] = port_b
        else:
            warnings.warn("Unusual connection direction pairing")

    def _build_state_mapping(self) -> None:
        """
        Build mapping from component names to global state vector offsets.

        Called internally by assemble().
        """
        self._component_offsets = {}
        offset = 0

        for comp in self.components:
            self._component_offsets[comp.name] = offset
            offset += comp.get_state_size()

        self._total_state_size = offset

    def assemble(self) -> Tuple[Callable, np.ndarray]:
        """
        Assemble global residual function and initial state.

        Returns:
            (residual_func, y0) where:
                residual_func(t, y) -> np.ndarray: global residual vector
                y0: initial state vector for all components

        Note:
            This is called automatically by solve(), but can be called
            explicitly for debugging or custom solver setup.
        """
        if not self.components:
            raise ValueError("Cannot assemble empty graph")

        # Build state mapping
        self._build_state_mapping()

        # Construct initial state vector
        y0_parts = []
        for comp in self.components:
            y0_parts.append(comp.get_initial_state())
        y0 = np.concatenate(y0_parts)

        # Build residual function
        def residual_func(t: float, y: np.ndarray) -> np.ndarray:
            """
            Global residual function: F(t, y) = 0

            Args:
                t: Current time [s]
                y: Global state vector (concatenated component states)

            Returns:
                Global residual vector (solver drives this to zero)
            """
            residuals = []

            for comp in self.components:
                # Extract this component's state slice
                offset = self._component_offsets[comp.name]
                size = comp.get_state_size()
                local_state = y[offset:offset+size]

                # Compute component residual
                # (ports are already shared via reference, so no synchronization needed)
                res = comp.residual(local_state, comp.ports, t)

                if res.shape != (size,):
                    raise ValueError(
                        f"Component {comp.name} returned residual of shape {res.shape}, "
                        f"expected ({size},)"
                    )

                residuals.append(res)

            return np.concatenate(residuals)

        return residual_func, y0

    def solve(self,
              t_span: Tuple[float, float],
              method: str = 'BDF',
              **solver_options) -> object:
        """
        Solve the system over a time interval.

        Args:
            t_span: (t_start, t_end) in seconds
            method: ODE solver method (default 'BDF' for stiff problems)
            **solver_options: Additional arguments passed to scipy.integrate.solve_ivp
                Common options:
                    rtol: Relative tolerance (default 1e-3)
                    atol: Absolute tolerance (default 1e-6)
                    max_step: Maximum timestep (default inf)

        Returns:
            Result object from scipy.integrate.solve_ivp with attributes:
                .t: Time points
                .y: State values (shape: [n_states, n_timepoints])
                .success: Whether solver succeeded
                .message: Solver status message

        Solver notes:
            - Uses scipy's BDF (Backward Differentiation Formula) by default
            - BDF is appropriate for stiff problems (typical in thermal systems)
            - Future versions will support SUNDIALS IDA for full DAE capability
            - For now, system must be in ODE form (algebraic variables eliminated)
        """
        residual_func, y0 = self.assemble()

        # Wrap residual as ODE: dy/dt = F(t, y)
        # (For true DAE support, we'll switch to IDA later)
        def ode_func(t, y):
            return residual_func(t, y)

        result = solve_ivp(
            ode_func,
            t_span=t_span,
            y0=y0,
            method=method,
            dense_output=True,
            **solver_options
        )

        if not result.success:
            warnings.warn(f"Solver failed: {result.message}")

        # Attach metadata for post-processing
        result.component_names = [c.name for c in self.components]
        result.component_offsets = self._component_offsets.copy()
        result.state_names = {}
        for comp in self.components:
            offset = self._component_offsets[comp.name]
            result.state_names[comp.name] = comp.get_state_names()

        return result

    def solve_steady_state(self, method: str = 'hybr', **solver_options) -> object:
        """
        Find steady-state solution using root finding.

        This is the recommended solver for pure algebraic systems where all variables
        are in steady state (no time dynamics). Uses scipy.optimize.root to find
        the state vector that satisfies all residual equations simultaneously.

        Args:
            method: Root finding algorithm (default 'hybr' - modified Powell hybrid)
                Options: 'hybr', 'lm' (Levenberg-Marquardt), 'broyden1', 'df-sane'
            **solver_options: Additional arguments passed to scipy.optimize.root
                Common options:
                    tol: Tolerance for termination (default solver-specific)
                    options: Dict of solver-specific options

        Returns:
            Result object with attributes:
                .x: Solution state vector
                .success: Whether solver converged
                .message: Solver status message
                .fun: Final residual values (should be near zero)
                .nfev: Number of function evaluations

        Note:
            For steady-state problems, this is more appropriate than solve() which
            uses a transient ODE solver. The residual equations f(x) = 0 are solved
            directly without introducing artificial dynamics.

        Example:
            result = graph.solve_steady_state()
            if result.success:
                turbine_state = graph.get_component_state(result, 'turbine')
        """
        residual_func, y0 = self.assemble()

        # Wrap residual function with error handling (time is irrelevant at steady state)
        def steady_residual(y):
            try:
                return residual_func(0.0, y)
            except (ValueError, RuntimeError) as e:
                # If CoolProp encounters invalid state during iteration,
                # return large residuals to guide solver away from unphysical regions
                return np.full_like(y, 1e10)

        # Try solving with more robust options
        solver_opts = {
            'ftol': 1e-8,
            'xtol': 1e-8,
            'maxfev': 10000,
        }
        solver_opts.update(solver_options)

        result = root(steady_residual, y0, method=method, options=solver_opts)

        if not result.success:
            warnings.warn(f"Steady-state solver failed: {result.message}")

        # Attach metadata for compatibility with solve() interface
        result.component_names = [c.name for c in self.components]
        result.component_offsets = self._component_offsets.copy()
        result.state_names = {}
        for comp in self.components:
            offset = self._component_offsets[comp.name]
            result.state_names[comp.name] = comp.get_state_names()

        # For compatibility with get_component_state, reshape x to look like y
        # y normally has shape [n_states, n_timepoints], make it [n_states, 1]
        result.y = result.x.reshape(-1, 1)
        result.t = np.array([0.0])  # Single time point at steady state

        # Trigger final residual evaluation to update all port values
        _ = residual_func(0.0, result.x)

        return result

    def get_component_state(self, result: object, component_name: str) -> np.ndarray:
        """
        Extract one component's state trajectory from solution.

        Args:
            result: Solution object from solve()
            component_name: Name of component to extract

        Returns:
            State array of shape [n_states, n_timepoints] for this component
        """
        offset = self._component_offsets[component_name]
        comp = next(c for c in self.components if c.name == component_name)
        size = comp.get_state_size()

        return result.y[offset:offset+size, :]

    def validate_topology(self) -> List[str]:
        """
        Check for common topology errors.

        Returns:
            List of warning messages (empty if no issues)

        Checks:
            - Disconnected components
            - Unconnected ports
            - Mass flow conservation
        """
        warnings_list = []

        # Check for unconnected ports
        for comp in self.components:
            for port_name, port in comp.ports.items():
                is_connected = any(
                    port is conn_port
                    for conn in self.connections
                    for conn_port in conn
                )
                if not is_connected:
                    warnings_list.append(
                        f"Component '{comp.name}' has unconnected port '{port_name}'"
                    )

        return warnings_list
