"""
Simple storage tank with level dynamics.

This is a simplified tank for demonstrating DAE capabilities (Phase 3 Milestone 2).
Tracks liquid level based on inlet/outlet mass flow.
"""

import numpy as np
from thermal_sim.core.component import Component
from thermal_sim.core.variable import Variable
from thermal_sim.core.port import MassFlowPort
from thermal_sim.properties.coolprop_wrapper import FluidProperties


class SimpleTank(Component):
    """
    Simple storage tank with level dynamics (Phase 3 Milestone 2).

    This component demonstrates differential equations in the simulator.
    It tracks the liquid level based on mass balance.

    Governing equations:
        1. d(level)/dt = (mdot_in - mdot_out) / (rho * A)   [differential]
        2. mdot_out = C_v * sqrt(rho * (P_tank - P_out))    [algebraic]

    State variables:
        - level: Liquid level [m] (differential)
        - mdot_out: Outlet mass flow rate [kg/s] (algebraic)

    Parameters:
        A: Cross-sectional area [m²]
        P: Operating pressure [Pa]
        P_downstream: Downstream pressure [Pa]
        C_v: Valve coefficient [kg/(s·√Pa)]
        level_initial: Initial liquid level [m]
        fluid: Fluid name for CoolProp (default 'Water')

    Example:
        >>> tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)
        >>> # Connect inlet and outlet
        >>> graph.add_component(tank)
        >>> graph.connect(pump.outlet, tank.inlet)
        >>> graph.connect(tank.outlet, next_component.inlet)
        >>> # Solve transient to see level change over time
        >>> result = graph.solve_transient(tspan=(0, 100))
    """

    def __init__(self,
                 name: str,
                 A: float,
                 P: float,
                 P_downstream: float,
                 C_v: float,
                 level_initial: float = 2.0,
                 T: float = 300.0,
                 fluid: str = 'Water'):
        super().__init__(name)

        if A <= 0:
            raise ValueError(f"Area must be positive, got {A}")
        if level_initial <= 0:
            raise ValueError(f"Initial level must be positive, got {level_initial}")
        if C_v <= 0:
            raise ValueError(f"Valve coefficient must be positive, got {C_v}")

        self.A = A  # Cross-sectional area [m²]
        self.P = P  # Tank pressure [Pa]
        self.P_downstream = P_downstream  # Downstream pressure [Pa]
        self.C_v = C_v  # Valve coefficient [kg/(s·√Pa)]
        self.level_initial = level_initial  # Initial level [m]
        self.T = T  # Temperature [K] (assumed constant for now)
        self.fluid_name = fluid
        self.fluid = FluidProperties(fluid)

        # Create ports
        self.inlet = MassFlowPort('inlet', direction='in')
        self.outlet = MassFlowPort('outlet', direction='out')

        # Initialize port values
        rho = self.fluid.density(self.P, self.T)
        h = self.fluid.enthalpy(self.P, self.T)

        self.inlet.P = P
        self.inlet.h = h
        self.inlet.mdot = 100.0

        self.outlet.P = P
        self.outlet.h = h
        self.outlet.mdot = 100.0

        self.ports = {
            'inlet': self.inlet,
            'outlet': self.outlet,
        }

    def get_variables(self):
        return [
            Variable('level', kind='differential', initial=self.level_initial, units='m'),
            Variable('mdot_out', kind='algebraic', initial=100.0, units='kg/s'),
        ]

    def get_initial_state(self):
        return np.array([self.level_initial, 100.0])

    def residual(self, state, ports, t, state_dot=None):
        level, mdot_out = state

        # Get inlet mass flow
        mdot_in = ports['inlet'].mdot

        # Fluid density at tank conditions
        rho = self.fluid.density(self.P, self.T)

        # Equation 1: Mass balance (differential equation)
        # d(level)/dt = (mdot_in - mdot_out) / (rho * A)
        if state_dot is not None:
            # DAE formulation: residual = f(x) - dx/dt
            dlevel_dt = (mdot_in - mdot_out) / (rho * self.A)
            eq_mass_balance = dlevel_dt - state_dot[0]
        else:
            # Steady-state: d(level)/dt = 0
            # So: mdot_in - mdot_out = 0
            eq_mass_balance = mdot_in - mdot_out

        # Equation 2: Outlet flow (algebraic constraint)
        # mdot_out = C_v * sqrt(rho * dP)
        dP = max(self.P - self.P_downstream, 0)  # Prevent negative pressure drop
        mdot_out_calc = self.C_v * np.sqrt(rho * dP)
        eq_outlet_flow = mdot_out - mdot_out_calc

        # Update outlet port
        ports['outlet'].mdot = mdot_out
        ports['outlet'].h = self.fluid.enthalpy(self.P, self.T)
        ports['outlet'].P = self.P

        return np.array([eq_mass_balance, eq_outlet_flow])

    def get_level(self) -> float:
        """Get current tank level (for diagnostics)"""
        # This would be called after solve with the solution state
        # In practice, user would extract from result.y
        return self.level_initial

    def __repr__(self):
        return (f"SimpleTank('{self.name}', A={self.A:.1f} m², "
                f"P={self.P/1e5:.1f} bar, level_init={self.level_initial:.1f} m)")
