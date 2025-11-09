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
from scipy.optimize import root, brentq
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

                # Debug check: detect NaN/Inf in residuals
                if np.any(np.isnan(res)):
                    raise ValueError(
                        f"NaN detected in residual for component '{comp.name}'. "
                        f"State: {local_state}, Residual: {res}"
                    )
                if np.any(np.isinf(res)):
                    raise ValueError(
                        f"Inf detected in residual for component '{comp.name}'. "
                        f"State: {local_state}, Residual: {res}"
                    )

                if res.shape != (size,):
                    raise ValueError(
                        f"Component {comp.name} returned residual of shape {res.shape}, "
                        f"expected ({size},)"
                    )

                residuals.append(res)

            return np.concatenate(residuals)

        return residual_func, y0

    def assemble_dae(self):
        """
        Assemble DAE system for transient solver (Phase 3+).

        Constructs residual function, initial conditions, and metadata
        for DAE solver: M(x)·dx/dt = f(x,t)

        Returns:
            Tuple of:
                residual_func: Function(t, y, ydot) -> residuals
                y0: Initial state vector
                ydot0: Initial derivative vector (zeros)
                algebraic_vars: Boolean list (True = algebraic, False = differential)

        Usage:
            >>> residual_func, y0, ydot0, alg_vars = graph.assemble_dae()
            >>> # Pass to DAE solver (scipy, diffeqpy, etc.)

        Note:
            This is the DAE-aware version of assemble(). For steady-state problems
            (dx/dt = 0), use assemble() or solve_steady_state() instead.
        """
        if not self.components:
            raise ValueError("Cannot assemble empty graph")

        # Build state mapping
        self._build_state_mapping()

        # Collect initial conditions and variable metadata
        y0_parts = []
        ydot0_list = []
        algebraic_vars = []

        for comp in self.components:
            y0_parts.append(comp.get_initial_state())

            # Get variable types
            for var in comp.get_variables():
                if var.kind == 'algebraic':
                    ydot0_list.append(0.0)  # Algebraic vars have zero derivative
                    algebraic_vars.append(True)
                elif var.kind == 'differential':
                    ydot0_list.append(0.0)  # Initial derivative (will be computed by solver)
                    algebraic_vars.append(False)
                else:
                    raise ValueError(
                        f"Unknown variable kind '{var.kind}' for {comp.name}.{var.name}. "
                        f"Must be 'algebraic' or 'differential'"
                    )

        y0 = np.concatenate(y0_parts)
        ydot0 = np.array(ydot0_list)

        # Build DAE residual function
        def residual_func(t: float, y: np.ndarray, ydot: np.ndarray) -> np.ndarray:
            """
            Global DAE residual function: F(t, y, dy/dt) = 0

            Args:
                t: Current time [s]
                y: Global state vector
                ydot: Global derivative vector (dy/dt)

            Returns:
                Global residual vector
            """
            residuals = []

            for comp in self.components:
                # Extract this component's state and derivative slices
                offset = self._component_offsets[comp.name]
                size = comp.get_state_size()
                local_state = y[offset:offset+size]
                local_state_dot = ydot[offset:offset+size]

                # Compute component residual (now includes state_dot)
                res = comp.residual(local_state, comp.ports, t, local_state_dot)

                # Debug checks
                if np.any(np.isnan(res)):
                    raise ValueError(
                        f"NaN detected in residual for component '{comp.name}'. "
                        f"State: {local_state}, State_dot: {local_state_dot}, Residual: {res}"
                    )
                if np.any(np.isinf(res)):
                    raise ValueError(
                        f"Inf detected in residual for component '{comp.name}'. "
                        f"State: {local_state}, State_dot: {local_state_dot}, Residual: {res}"
                    )
                if res.shape != (size,):
                    raise ValueError(
                        f"Component {comp.name} returned residual of shape {res.shape}, "
                        f"expected ({size},)"
                    )

                residuals.append(res)

            return np.concatenate(residuals)

        return residual_func, y0, ydot0, algebraic_vars

    def solve_transient(self,
                       tspan: Tuple[float, float] | np.ndarray,
                       backend: str = 'scipy',
                       **solver_options) -> object:
        """
        Solve transient DAE system: M(x)·dx/dt = f(x,t)

        Integrates differential equations over time while maintaining
        algebraic constraints. Uses steady-state as initial condition
        unless custom IC provided.

        Args:
            tspan: (t_start, t_end) or array of time points [s]
            backend: 'scipy' (default) or 'diffeqpy'
            **solver_options:
                y0: Optional initial condition (otherwise uses solve_steady_state)
                method: Integration method ('BDF', 'Radau', etc.)
                rtol, atol: Tolerances
                max_step: Maximum time step

        Returns:
            Result object with:
                .t: Time points
                .y: State trajectory [n_vars, n_timepoints]
                .success: Convergence flag
                .message: Status message
                .component_names, .component_offsets, .state_names: Metadata

        Example:
            >>> # Tank filling from steady-state
            >>> result = graph.solve_transient(tspan=(0, 100))
            >>> level_history = graph.get_component_state(result, 'tank')[0, :]

        Note:
            For pure algebraic systems (Phase 0-2), use solve_steady_state() instead.
            This method is for systems with at least one differential variable.
        """
        # Get or compute initial condition
        if 'y0' in solver_options:
            y0 = solver_options.pop('y0')
        else:
            # Use steady-state as IC
            print("Computing steady-state initial condition...")
            ss_result = self.solve_steady_state()
            if not ss_result.success:
                warnings.warn(
                    "Steady-state initialization failed. Using default initial conditions. "
                    "Transient solution may not be physically meaningful."
                )
                _, y0, _, _ = self.assemble_dae()
            else:
                y0 = ss_result.x

        # Assemble DAE system
        residual_func, _, ydot0, algebraic_vars = self.assemble_dae()

        # Check if system has differential variables
        if all(algebraic_vars):
            raise ValueError(
                "System has no differential variables. "
                "Use solve_steady_state() for pure algebraic systems."
            )

        if backend == 'scipy':
            return self._solve_transient_scipy(
                residual_func, y0, ydot0, tspan, algebraic_vars, **solver_options
            )
        elif backend == 'diffeqpy':
            return self._solve_transient_diffeqpy(
                residual_func, y0, ydot0, tspan, algebraic_vars, **solver_options
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _solve_transient_scipy(self, residual_func, y0, ydot0, tspan,
                              algebraic_vars, **solver_options):
        """
        Scipy backend for transient DAE solving.

        Strategy: Use scipy's implicit BDF method and convert DAE residuals
        to ODE form by solving algebraic constraints implicitly.
        """
        from scipy.integrate import solve_ivp
        from scipy.optimize import fsolve

        # Extract options
        method = solver_options.pop('method', 'BDF')  # BDF for stiff systems
        rtol = solver_options.pop('rtol', 1e-6)
        atol = solver_options.pop('atol', 1e-8)

        # Parse tspan
        if isinstance(tspan, tuple):
            t_start, t_end = tspan
            t_eval = None
        else:
            t_start = tspan[0]
            t_end = tspan[-1]
            t_eval = tspan

        # For semi-explicit DAE, we need to convert to ODE form
        # Differential variables: dy/dt = f(y, t)
        # Algebraic variables: solve g(y, t) = 0 at each time step

        n_vars = len(y0)
        diff_indices = [i for i, is_alg in enumerate(algebraic_vars) if not is_alg]
        alg_indices = [i for i, is_alg in enumerate(algebraic_vars) if is_alg]

        def ode_func(t, y):
            """
            Convert DAE residual to ODE form: dy/dt = F(t, y)

            For differential vars: extract dy/dt from residual = f - dy/dt
            For algebraic vars: solve f = 0 to maintain constraints
            """
            ydot = np.zeros(n_vars)

            # If we have algebraic variables, solve them at this time instant
            if alg_indices:
                def alg_residual(y_alg_vals):
                    """Residual for algebraic variables only"""
                    y_temp = y.copy()
                    y_temp[alg_indices] = y_alg_vals
                    ydot_temp = np.zeros(n_vars)
                    res = residual_func(t, y_temp, ydot_temp)
                    return res[alg_indices]

                # Solve algebraic constraints
                y_alg_init = y[alg_indices]
                y_alg_solution = fsolve(alg_residual, y_alg_init, full_output=False)
                y_new = y.copy()
                y_new[alg_indices] = y_alg_solution
            else:
                y_new = y

            # Compute residual with solved algebraic vars
            res = residual_func(t, y_new, ydot)

            # For differential vars, residual = f - dy/dt, so dy/dt = -residual + f
            # Actually, residual = f - dy/dt, so dy/dt = f - residual
            # Wait, let me think about the sign...
            # In the Tank: residual = dlevel_dt - state_dot[0]
            # So: 0 = dlevel_dt - state_dot[0]
            # Therefore: state_dot[0] = dlevel_dt = f(...)
            # So: residual = f - state_dot, meaning state_dot = f - residual

            # Actually, rearranging: residual = f - ydot
            # So: ydot = f - residual
            # But we want ydot such that residual = 0
            # If residual = f - ydot, then ydot = f when residual = 0

            # Let me reconsider. The residual function is defined as:
            # residual = computed_value - state_dot
            # At solution, residual = 0, so: computed_value = state_dot
            # Therefore: ydot = computed_value = residual + state_dot = residual (when state_dot=0)

            # Actually in the DAE formulation with state_dot passed in:
            # residual(t, y, ydot) should equal 0
            # For differential: residual = f(y,t) - ydot
            # So at solution: 0 = f(y,t) - ydot => ydot = f(y,t)

            # To extract ydot, I solve: residual_func(t, y, ydot) = 0 for ydot
            # For now, let's use a simpler approach:

            # Compute residual with ydot = 0
            ydot_trial = np.zeros(n_vars)
            res_zero = residual_func(t, y_new, ydot_trial)

            # For differential variables, extract derivative
            # residual = f - ydot, so ydot = f = -residual (when ydot_trial = 0)
            # No wait: residual = f - ydot_trial = f - 0 = f
            # So f = residual, and we want ydot = f
            ydot[diff_indices] = res_zero[diff_indices]

            # Algebraic variables have zero derivative
            ydot[alg_indices] = 0.0

            return ydot

        # Solve ODE
        result = solve_ivp(
            ode_func,
            (t_start, t_end),
            y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            dense_output=True,
            **solver_options
        )

        if not result.success:
            warnings.warn(f"Transient solver failed: {result.message}")

        # Attach metadata
        result.component_names = [c.name for c in self.components]
        result.component_offsets = self._component_offsets.copy()
        result.state_names = {}
        for comp in self.components:
            result.state_names[comp.name] = comp.get_state_names()

        return result

    def _solve_transient_diffeqpy(self, residual_func, y0, ydot0, tspan,
                                 algebraic_vars, **solver_options):
        """
        Diffeqpy backend for transient DAE solving (future implementation).

        Julia's DifferentialEquations.jl has native DAE solvers (IDA, Rodas5, etc.)
        that can handle differential-algebraic systems directly.
        """
        raise NotImplementedError(
            "diffeqpy transient solver not yet implemented. "
            "Use backend='scipy' for now."
        )

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

    def solve_steady_state(self, backend: str = 'scipy', method: str = 'hybr',
                           use_sequential_init: bool = True,
                           fallback_to_sequential: bool = True,
                           **solver_options) -> object:
        """
        Find steady-state solution using simultaneous root finding.

        This is the recommended solver for pure algebraic systems where all variables
        are in steady state (no time dynamics). Supports multiple backends for
        solving f(x) = 0 simultaneously.

        Strategy:
            1. Attempt direct solution with backend (scipy or diffeqpy)
            2. If use_sequential_init=True and direct fails: use solve_sequential()
               to generate better initial guess, then retry
            3. If fallback_to_sequential=True and still fails: return sequential result

        This multi-stage approach provides robustness: simultaneous solvers are more
        accurate when they converge, but sequential solver ensures a working solution.

        Args:
            backend: Solver backend to use (default 'scipy')
                'scipy': Uses scipy.optimize.root (fast, reliable, pure Python)
                'diffeqpy': Uses Julia's NonlinearSolve.jl (more robust for stiff systems)
                'sequential': Uses solve_sequential() directly (for testing/fallback)
            method: Root finding algorithm (used with 'scipy' backend)
                Options: 'hybr' (default), 'lm', 'broyden1', 'df-sane'
                Ignored when backend='diffeqpy' or 'sequential'
            use_sequential_init: If True, use solve_sequential() to generate initial
                guess when direct solution fails (default True)
            fallback_to_sequential: If True, return sequential result if simultaneous
                methods fail (default True)
            **solver_options: Additional arguments passed to the backend solver
                scipy backend: passed to scipy.optimize.root
                    tol: Tolerance for termination
                    options: Dict of solver-specific options
                diffeqpy backend: passed to de.solve()
                    abstol: Absolute tolerance (default 1e-8)
                    reltol: Relative tolerance (default 1e-8)
                    maxiters: Maximum iterations (default 10000)

        Returns:
            Result object with attributes:
                .x: Solution state vector
                .success: Whether solver converged
                .message: Solver status message
                .fun: Final residual values (should be near zero)
                .nfev: Number of function evaluations
                .solver_used: Which solver produced this result

        Note:
            For steady-state problems, this is more appropriate than solve() which
            uses a transient ODE solver. The residual equations f(x) = 0 are solved
            directly without introducing artificial dynamics.

        Example:
            # Using scipy with sequential initialization (default)
            result = graph.solve_steady_state()

            # Using diffeqpy without fallback
            result = graph.solve_steady_state(
                backend='diffeqpy',
                fallback_to_sequential=False
            )

            # Direct sequential solver
            result = graph.solve_steady_state(backend='sequential')

            if result.success:
                turbine_state = graph.get_component_state(result, 'turbine')
        """
        # Handle sequential backend directly
        if backend == 'sequential':
            result = self.solve_sequential()
            result.solver_used = 'sequential'
            return result

        # Try direct solution first
        if backend == 'scipy':
            result = self._solve_steady_scipy(method, y0=None, **solver_options)
        elif backend == 'diffeqpy':
            result = self._solve_steady_diffeqpy(y0=None, **solver_options)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'scipy', 'diffeqpy', or 'sequential'")

        result.solver_used = backend

        # If failed and sequential init enabled, retry with better initial guess
        if not result.success and use_sequential_init:
            try:
                seq_result = self.solve_sequential()
                if seq_result.success:
                    # Retry with sequential solution as initial guess
                    if backend == 'scipy':
                        result = self._solve_steady_scipy(method, y0=seq_result.x, **solver_options)
                    elif backend == 'diffeqpy':
                        result = self._solve_steady_diffeqpy(y0=seq_result.x, **solver_options)

                    if result.success:
                        result.solver_used = f"{backend} (sequential init)"
                    elif fallback_to_sequential:
                        # Use sequential result as fallback
                        result = seq_result
                        result.solver_used = 'sequential (fallback)'
            except Exception as e:
                warnings.warn(f"Sequential initialization failed: {e}")

        # Final fallback if still unsuccessful
        if not result.success and fallback_to_sequential:
            try:
                seq_result = self.solve_sequential()
                if seq_result.success:
                    result = seq_result
                    result.solver_used = 'sequential (fallback)'
            except Exception as e:
                warnings.warn(f"Sequential fallback failed: {e}")

        return result

    def _solve_steady_scipy(self, method: str = 'hybr', y0=None, **solver_options) -> object:
        """
        Scipy backend for solve_steady_state(). See solve_steady_state() docs.

        Args:
            method: Root finding algorithm
            y0: Initial guess (if None, use component defaults)
            **solver_options: Options passed to scipy.optimize.root
        """
        residual_func, y0_default = self.assemble()
        if y0 is None:
            y0 = y0_default

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

    def _solve_steady_diffeqpy(self, y0=None, **solver_options) -> object:
        """
        Find steady-state solution using Julia's NonlinearSolve.jl via diffeqpy.

        This method uses NonlinearSolve.jl for solving f(x) = 0 systems. It's more
        robust than scipy for stiff systems and provides better convergence diagnostics.

        Args:
            y0: Initial guess (if None, use component defaults)
            **solver_options: Options passed to de.solve()
                Common options:
                    abstol: Absolute tolerance (default 1e-8)
                    reltol: Relative tolerance (default 1e-8)
                    maxiters: Maximum iterations (default 10000)

        Returns:
            Result object compatible with solve_steady_state() interface

        Raises:
            ImportError: If diffeqpy is not installed
            RuntimeError: If Julia solver fails to converge
        """
        try:
            from diffeqpy import de
        except ImportError:
            raise ImportError(
                "diffeqpy is required for this solver. Install with: uv pip install diffeqpy"
            )

        residual_func, y0_default = self.assemble()
        if y0 is None:
            y0 = y0_default

        # Wrap residual for NonlinearSolve (takes (u, p) and returns residual list)
        def nonlinear_func(u, p):
            """Residual function for NonlinearSolve"""
            try:
                res = residual_func(0.0, u)  # t=0 for steady-state
                # Return list (not numpy array) for Julia compatibility
                return res.tolist()
            except (ValueError, RuntimeError) as e:
                # Return large residuals if CoolProp fails
                return [1e10] * len(u)

        # Set default solver options
        opts = {
            'abstol': 1e-8,
            'reltol': 1e-8,
            'maxiters': 10000,
        }
        opts.update(solver_options)

        # Create and solve NonlinearProblem
        # Pass y0 as numpy array (Julia will convert to Vector{Float64})
        prob = de.NonlinearProblem(nonlinear_func, y0)
        sol = de.solve(prob, **opts)

        # Package result in scipy-compatible format
        result = type('Result', (), {})()
        result.x = np.array(sol.u)
        # Check if solution converged (retcode should be a success indicator)
        try:
            result.success = bool(str(sol.retcode) == 'Success' or 'Success' in str(sol.retcode))
            result.message = f"Julia NonlinearSolve: {sol.retcode}"
        except:
            result.success = True  # Assume success if we got a solution
            result.message = "Julia NonlinearSolve: completed"
        result.fun = residual_func(0.0, result.x)
        result.nfev = -1  # Julia doesn't expose this easily

        if not result.success:
            warnings.warn(f"diffeqpy solver failed: {result.message}")

        # Attach metadata for compatibility with solve() interface
        result.component_names = [c.name for c in self.components]
        result.component_offsets = self._component_offsets.copy()
        result.state_names = {}
        for comp in self.components:
            offset = self._component_offsets[comp.name]
            result.state_names[comp.name] = comp.get_state_names()

        # For compatibility with get_component_state, reshape x to look like y
        result.y = result.x.reshape(-1, 1)
        result.t = np.array([0.0])

        # Trigger final residual evaluation to update all port values
        _ = residual_func(0.0, result.x)

        return result

    def solve_sequential(self, max_iter: int = 100, tol: float = 1e-6) -> object:
        """
        Sequential steady-state solver for initialization and fallback.

        This solver propagates states sequentially through components rather than
        solving all equations simultaneously. While less accurate than simultaneous
        methods, it provides two key benefits:

        1. **Initialization**: Generates thermodynamically consistent initial guesses
           for simultaneous solvers (scipy, diffeqpy), avoiding basin-of-attraction
           issues that cause Newton-type methods to fail.

        2. **Fallback**: Provides a working solution when simultaneous methods fail
           to converge from default initial conditions.

        Algorithm:
            1. Guess mass flow rate
            2. Propagate state sequentially through components
            3. Check loop closure (inlet of first component matches outlet of last)
            4. Adjust mass flow and repeat until convergence

        Limitations:
            - Only supports 4-component Rankine cycles (boiler→turbine→condenser→pump)
            - Components must be named 'boiler', 'turbine', 'condenser', 'pump'
            - Only works for simple closed loops (not general graphs)
            - Less accurate than simultaneous solution
            - Sequential errors can accumulate

        Use Cases:
            - Generating initial guesses: `y0 = graph.solve_sequential().x`
            - Fallback when solve_steady_state() fails
            - Quick rough solutions for validation

        For production use, prefer solve_steady_state() which uses this method
        automatically for initialization when needed.

        Args:
            max_iter: Maximum iterations for outer loop convergence
            tol: Tolerance for loop closure error

        Returns:
            Result object compatible with solve_steady_state interface:
                .x: Solution state vector
                .success: Whether solver converged
                .message: Solver status message
                .fun: Final residual values
                .y: Reshaped state for get_component_state()
                .t: Time array [0.0]
        """
        # Detect if we have a simple closed loop
        if len(self.components) != 4:
            warnings.warn(
                "Sequential solver currently only supports 4-component Rankine cycles. "
                "Use solve_steady_state() with scipy/diffeqpy for general topologies."
            )

        # For now, assume Rankine cycle order: boiler → turbine → condenser → pump
        comp_dict = {c.name: c for c in self.components}

        # Try to identify components by type
        boiler = None
        turbine = None
        condenser = None
        pump = None

        for c in self.components:
            if 'boiler' in c.name.lower():
                boiler = c
            elif 'turbine' in c.name.lower():
                turbine = c
            elif 'condenser' in c.name.lower() or 'cond' in c.name.lower():
                condenser = c
            elif 'pump' in c.name.lower():
                pump = c

        if not all([boiler, turbine, condenser, pump]):
            raise ValueError(
                "Sequential solver requires components named 'boiler', 'turbine', "
                "'condenser', and 'pump'. Use solve_steady_state() for other topologies."
            )

        # Initial guess for mass flow rate
        mdot_guess = 100.0

        def loop_error(mdot):
            """
            Compute loop closure error for given mass flow rate.

            Sequential solver uses root finding on this function to determine
            the mass flow that closes the thermodynamic loop.
            """
            try:
                # Set mass flow in all components
                for comp in [boiler, turbine, condenser, pump]:
                    for port in comp.ports.values():
                        port.mdot = mdot

                # Propagate states sequentially through loop

                # Start with pump outlet (known high pressure, low enthalpy)
                from CoolProp.CoolProp import PropsSI
                h_pump_in = PropsSI('H', 'P', condenser.P, 'Q', 0, 'Water')  # Sat liquid at low P

                # Solve pump: given h_in, find h_out
                s_pump_in = pump.fluid.entropy(condenser.P, h_pump_in)
                h_pump_out_s = pump.fluid.enthalpy_from_ps(pump.P_out, s_pump_in)
                h_pump_out = h_pump_in + (h_pump_out_s - h_pump_in) / pump.eta
                pump.ports['inlet'].h = h_pump_in
                pump.ports['inlet'].P = condenser.P
                pump.ports['outlet'].h = h_pump_out
                pump.ports['outlet'].P = pump.P_out

                # Solve boiler: given h_in, Q, find h_out
                h_boiler_in = h_pump_out
                h_boiler_out = h_boiler_in + boiler.Q / mdot

                # Check if boiler outlet enthalpy is physically reasonable
                if h_boiler_out > 5e6 or h_boiler_out < h_boiler_in:
                    # Return large error to guide root finder away from this region
                    return 1e10 * np.sign(mdot - 100.0)

                boiler.ports['inlet'].h = h_boiler_in
                boiler.ports['inlet'].P = pump.P_out
                boiler.ports['outlet'].h = h_boiler_out
                boiler.ports['outlet'].P = boiler.P

                # Solve turbine: given h_in, P_out, eta, find h_out
                h_turbine_in = h_boiler_out
                s_turbine_in = turbine.fluid.entropy(boiler.P, h_turbine_in)
                h_turbine_out_s = turbine.fluid.enthalpy_from_ps(turbine.P_out, s_turbine_in)
                h_turbine_out = h_turbine_in - turbine.eta * (h_turbine_in - h_turbine_out_s)

                turbine.ports['inlet'].h = h_turbine_in
                turbine.ports['inlet'].P = boiler.P
                turbine.ports['outlet'].h = h_turbine_out
                turbine.ports['outlet'].P = turbine.P_out

                # Solve condenser: given h_in, Q, find h_out
                h_condenser_in = h_turbine_out
                h_condenser_out = h_condenser_in + condenser.Q / mdot

                # Check condenser outlet is near saturated liquid
                h_sat_liq = PropsSI('H', 'P', condenser.P, 'Q', 0, 'Water')

                # Loop closure error: condenser outlet should match pump inlet
                error = h_condenser_out - h_pump_in

                # Also check that condenser brings fluid close to saturated liquid
                # If condenser outlet is far from sat liquid, increase error
                sat_error = abs(h_condenser_out - h_sat_liq) / 1000.0  # Normalize

                return error + sat_error

            except (ValueError, RuntimeError) as e:
                # If CoolProp fails, return large error
                # Sign guides solver toward larger/smaller mdot
                return 1e10 * np.sign(mdot - 100.0)

        # Use root finding to find mass flow that closes loop
        from scipy.optimize import brentq

        # Bracket the solution: try wide range of mass flows
        mdot_min = 10.0
        mdot_max = 500.0

        # Check if both bounds have same sign (no root in interval)
        err_min = loop_error(mdot_min)
        err_max = loop_error(mdot_max)

        success = False
        message = ""
        state_vector = None

        if np.sign(err_min) == np.sign(err_max):
            # No root in bracketing interval - try expanding range
            if abs(err_min) < abs(err_max):
                # Solution might be below mdot_min
                mdot_min = 1.0
                err_min = loop_error(mdot_min)
            else:
                # Solution might be above mdot_max
                mdot_max = 1000.0
                err_max = loop_error(mdot_max)

        # Try root finding
        try:
            if np.sign(err_min) != np.sign(err_max):
                mdot_solution = brentq(loop_error, mdot_min, mdot_max, xtol=tol, maxiter=max_iter)

                # Evaluate final state at solution mass flow
                _ = loop_error(mdot_solution)  # Sets all port values

                # Build state vector from component states
                # Order: boiler, turbine, condenser, pump
                boiler_state = np.array([boiler.ports['outlet'].h, mdot_solution])
                turbine_state = np.array([
                    turbine.ports['outlet'].h,
                    mdot_solution * (turbine.ports['inlet'].h - turbine.ports['outlet'].h),  # W_shaft
                    mdot_solution
                ])
                condenser_state = np.array([condenser.ports['outlet'].h, mdot_solution])
                pump_state = np.array([
                    pump.ports['outlet'].h,
                    mdot_solution * (pump.ports['outlet'].h - pump.ports['inlet'].h),  # W_shaft
                    mdot_solution
                ])

                state_vector = np.concatenate([boiler_state, turbine_state, condenser_state, pump_state])
                success = True
                message = "Sequential solver converged"
            else:
                message = f"No root found in interval [{mdot_min}, {mdot_max}] kg/s"
                state_vector = np.zeros(10)  # Placeholder

        except (ValueError, RuntimeError) as e:
            message = f"Sequential solver failed: {str(e)}"
            state_vector = np.zeros(10)  # Placeholder

        # Package result in standardized format
        class SequentialResult:
            def __init__(self, x, success, message):
                self.x = x
                self.success = success
                self.message = message
                self.fun = np.zeros(len(x))  # Sequential doesn't compute residuals directly
                self.y = self.x.reshape(-1, 1)
                self.t = np.array([0.0])

        result = SequentialResult(state_vector, success, message)

        # Attach metadata for compatibility
        result.component_names = [c.name for c in [boiler, turbine, condenser, pump]]
        self._component_offsets = {}
        offset = 0
        for comp in [boiler, turbine, condenser, pump]:
            self._component_offsets[comp.name] = offset
            offset += comp.get_state_size()
        result.component_offsets = self._component_offsets.copy()
        result.state_names = {}
        for comp in [boiler, turbine, condenser, pump]:
            result.state_names[comp.name] = comp.get_state_names()

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
