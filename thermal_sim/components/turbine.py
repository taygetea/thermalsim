"""
Turbine: converts fluid enthalpy to shaft work via expansion.
"""

import numpy as np
from thermal_sim.core.component import Component
from thermal_sim.core.variable import Variable
from thermal_sim.core.port import MassFlowPort
from thermal_sim.properties.coolprop_wrapper import FluidProperties


class Turbine(Component):
    """
    Expands high-pressure fluid to produce shaft work.

    Governing equations:
        1. h_out = h_in - η * (h_in - h_out_isentropic)
        2. W_shaft = mdot * (h_in - h_out)
        3. P_out = P_set

    State variables:
        - h_out: outlet enthalpy [J/kg]
        - W_shaft: shaft power output [W]
        - mdot: mass flow rate [kg/s]

    Parameters:
        efficiency: Isentropic efficiency (0 < η < 1)
        P_out: Outlet pressure [Pa]
        fluid: Fluid name
    """

    def __init__(self, name: str, efficiency: float, P_out: float, fluid: str = 'Water'):
        super().__init__(name)

        if not 0 < efficiency <= 1:
            raise ValueError(f"Efficiency must be in (0,1], got {efficiency}")

        self.eta = efficiency
        self.P_out = P_out
        self.fluid_name = fluid
        self.fluid = FluidProperties(fluid)

        self.inlet = MassFlowPort('inlet', direction='in')
        self.outlet = MassFlowPort('outlet', direction='out')

        # Initialize with reasonable defaults
        self.inlet.P = 10e6  # High pressure
        self.inlet.h = 3e6   # Superheated steam
        self.inlet.mdot = 100.0
        self.outlet.P = P_out
        self.outlet.h = 2e6
        self.outlet.mdot = 100.0

        self.ports = {
            'inlet': self.inlet,
            'outlet': self.outlet,
        }

    def get_variables(self):
        return [
            Variable('h_out', kind='algebraic', initial=1.9e6, units='J/kg'),  # Two-phase after expansion
            Variable('W_shaft', kind='algebraic', initial=80e6, units='W'),    # ~80 MW for typical cycle
            Variable('mdot', kind='algebraic', initial=100.0, units='kg/s'),
        ]

    def get_initial_state(self):
        return np.array([1.9e6, 80e6, 100.0])

    def residual(self, state, ports, t, state_dot=None):
        h_out, W_shaft, mdot = state

        # Inlet conditions
        h_in = ports['inlet'].h
        P_in = ports['inlet'].P

        # Compute isentropic outlet enthalpy
        s_in = self.fluid.entropy(P_in, h_in)
        h_out_s = self.fluid.enthalpy_from_ps(self.P_out, s_in)

        # Equation 1: Isentropic efficiency relation
        # h_out = h_in - η * (h_in - h_out_s)
        eq_efficiency = h_out - (h_in - self.eta * (h_in - h_out_s))

        # Equation 2: Power extraction
        # W = mdot * (h_in - h_out)
        eq_power = W_shaft - mdot * (h_in - h_out)

        # Equation 3: Outlet pressure
        eq_pressure = ports['outlet'].P - self.P_out

        # Update outlet
        ports['outlet'].mdot = mdot
        ports['outlet'].h = h_out
        ports['outlet'].P = self.P_out

        return np.array([eq_efficiency, eq_power, eq_pressure])
