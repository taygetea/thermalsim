"""
Pump: increases fluid pressure using shaft work.
"""

import numpy as np
from thermal_sim.core.component import Component
from thermal_sim.core.variable import Variable
from thermal_sim.core.port import MassFlowPort
from thermal_sim.properties.coolprop_wrapper import FluidProperties


class Pump(Component):
    """
    Increases pressure of liquid using shaft power.

    Governing equations:
        1. h_out = h_in + (h_out_isentropic - h_in) / η
        2. W_shaft = mdot * (h_out - h_in)
        3. P_out = P_set

    State variables:
        - h_out: outlet enthalpy [J/kg]
        - W_shaft: shaft power input [W]
        - mdot: mass flow rate [kg/s]

    Parameters:
        efficiency: Isentropic efficiency
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
        self.inlet.P = 10e3   # Low pressure
        self.inlet.h = 2e5    # Saturated liquid
        self.inlet.mdot = 100.0
        self.outlet.P = P_out
        self.outlet.h = 2e5
        self.outlet.mdot = 100.0

        self.ports = {
            'inlet': self.inlet,
            'outlet': self.outlet,
        }

    def get_variables(self):
        return [
            Variable('h_out', kind='algebraic', initial=1e5, units='J/kg'),
            Variable('W_shaft', kind='algebraic', initial=1e5, units='W'),
            Variable('mdot', kind='algebraic', initial=100.0, units='kg/s'),
        ]

    def get_initial_state(self):
        return np.array([1e5, 1e5, 100.0])

    def residual(self, state, ports, t):
        h_out, W_shaft, mdot = state

        h_in = ports['inlet'].h
        P_in = ports['inlet'].P

        # Isentropic outlet enthalpy
        s_in = self.fluid.entropy(P_in, h_in)
        h_out_s = self.fluid.enthalpy_from_ps(self.P_out, s_in)

        # Equation 1: Efficiency relation (pump uses more work than isentropic)
        # h_out = h_in + (h_out_s - h_in) / η
        eq_efficiency = h_out - (h_in + (h_out_s - h_in) / self.eta)

        # Equation 2: Power consumption
        eq_power = W_shaft - mdot * (h_out - h_in)

        # Equation 3: Outlet pressure
        eq_pressure = ports['outlet'].P - self.P_out

        # Update outlet
        ports['outlet'].mdot = mdot
        ports['outlet'].h = h_out
        ports['outlet'].P = self.P_out

        return np.array([eq_efficiency, eq_power, eq_pressure])
