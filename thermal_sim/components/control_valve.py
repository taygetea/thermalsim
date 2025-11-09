"""
Control valve component with position dynamics (Phase 3 Milestone 4).

Valve with first-order lag response to control commands.
"""

import numpy as np
from thermal_sim.core.component import Component
from thermal_sim.core.variable import Variable
from thermal_sim.core.port import MassFlowPort, ScalarPort
from thermal_sim.properties.coolprop_wrapper import FluidProperties


class ControlValve(Component):
    """
    Valve with position control and first-order lag dynamics (Phase 3 Milestone 4).

    The valve position responds to a control command with first-order dynamics:
        tau · d(position)/dt = command - position

    Flow through the valve is modulated by position:
        mdot = (Cv_max · position) · sqrt(rho · dP)

    Where position ∈ [0, 1]:
        - 0 = fully closed (no flow)
        - 1 = fully open (maximum Cv)

    State variables:
        - position: Valve position [0, 1] (differential)
        - mdot: Mass flow rate [kg/s] (algebraic)

    Parameters:
        name: Component identifier
        Cv_max: Maximum valve coefficient [kg/(s·√Pa)] when fully open
        tau: Time constant [s] for position response (default 5.0)
        position_initial: Initial valve position [0, 1] (default 0.5)
        fluid: Fluid name for CoolProp (default 'Water')

    Ports:
        inlet (MassFlowPort, in): Upstream fluid connection
        outlet (MassFlowPort, out): Downstream fluid connection
        command (ScalarPort, in): Position command signal [0, 1]

    Example:
        >>> valve = ControlValve('valve', Cv_max=0.02, tau=3.0, position_initial=0.5)
        >>> graph.add_component(valve)
        >>> graph.connect(tank.outlet, valve.inlet)
        >>> graph.connect(pid.output, valve.command)
        >>> result = graph.solve_transient(tspan=(0, 100))
    """

    def __init__(self,
                 name: str,
                 Cv_max: float,
                 tau: float = 5.0,
                 position_initial: float = 0.5,
                 fluid: str = 'Water'):
        super().__init__(name)

        if Cv_max <= 0:
            raise ValueError(f"Cv_max must be positive, got {Cv_max}")
        if tau <= 0:
            raise ValueError(f"Time constant tau must be positive, got {tau}")
        if not (0 <= position_initial <= 1):
            raise ValueError(f"Initial position must be in [0, 1], got {position_initial}")

        self.Cv_max = Cv_max  # Maximum valve coefficient [kg/(s·√Pa)]
        self.tau = tau  # Time constant [s]
        self.position_initial = position_initial  # Initial position [0, 1]
        self.fluid_name = fluid
        self.fluid = FluidProperties(fluid)

        # Create ports
        self.inlet = MassFlowPort('inlet', direction='in')
        self.outlet = MassFlowPort('outlet', direction='out')
        self.command = ScalarPort('command', direction='in')

        self.inlet.component = self
        self.outlet.component = self
        self.command.component = self

        # Initialize port values (will be overridden during solve)
        self.inlet.mdot = 100.0
        self.inlet.h = 419e3  # Water at ~100 C
        self.inlet.P = 2e5

        self.outlet.mdot = 100.0
        self.outlet.h = 419e3
        self.outlet.P = 1e5

        self.command.value = position_initial

        self.ports = {
            'inlet': self.inlet,
            'outlet': self.outlet,
            'command': self.command,
        }

    def get_variables(self):
        return [
            Variable('position', kind='differential', initial=self.position_initial, units='-'),
            Variable('mdot', kind='algebraic', initial=100.0, units='kg/s'),
        ]

    def get_initial_state(self):
        return np.array([self.position_initial, 100.0])

    def residual(self, state, ports, t, state_dot=None):
        position, mdot = state

        # Equation 1: Valve position dynamics (first-order lag)
        # tau · d(position)/dt = command - position
        # Rearranged: d(position)/dt = (command - position) / tau
        command = np.clip(ports['command'].value, 0.0, 1.0)  # Saturate command

        if state_dot is not None:
            # Transient: residual = f(x,t) - dx/dt
            dposition_dt = (command - position) / self.tau
            eq_position = dposition_dt - state_dot[0]
        else:
            # Steady-state: position = command
            eq_position = command - position

        # Equation 2: Flow through valve (modulated by position)
        # mdot = Cv(position) · sqrt(rho · dP)
        # where Cv(position) = Cv_max · position

        # Get fluid density
        h_in = ports['inlet'].h
        P_in = ports['inlet'].P
        P_out = ports['outlet'].P if hasattr(ports['outlet'], 'P') else P_in * 0.9

        # Use inlet conditions for density
        try:
            T_in = self.fluid.temperature(P_in, h_in)
            rho = self.fluid.density(P_in, T_in)
        except Exception:
            # Fallback to nominal conditions if property call fails
            rho = 1000.0  # kg/m³ for water

        # Pressure drop (prevent negative)
        dP = max(P_in - P_out, 0.0)

        # Effective valve coefficient
        Cv_effective = self.Cv_max * max(position, 0.0)  # Prevent negative position

        # Flow equation
        mdot_calc = Cv_effective * np.sqrt(rho * dP)
        eq_mdot = mdot - mdot_calc

        # Update outlet port (isenthalpic expansion)
        ports['outlet'].mdot = mdot
        ports['outlet'].h = h_in  # Isenthalpic
        ports['outlet'].P = P_out

        return np.array([eq_position, eq_mdot])

    def get_position(self) -> float:
        """Get current valve position (for diagnostics)"""
        return self.position_initial  # Would return actual position from solution

    def get_flow(self) -> float:
        """Get current mass flow rate (for diagnostics)"""
        return self.outlet.mdot

    def __repr__(self):
        return (f"ControlValve('{self.name}', Cv_max={self.Cv_max:.3f}, "
                f"tau={self.tau:.1f}s, position={self.position_initial:.2f})")
