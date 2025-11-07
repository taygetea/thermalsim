"""
Constant-pressure heater/cooler component.

Adds or removes heat from a fluid stream while maintaining constant pressure.
Can represent boilers, condensers, or simple heat exchangers.
"""

import numpy as np
from thermal_sim.core.component import Component
from thermal_sim.core.variable import Variable
from thermal_sim.core.port import MassFlowPort
from thermal_sim.properties.coolprop_wrapper import FluidProperties


class ConstantPressureHeater(Component):
    """
    Heats or cools fluid at constant pressure.

    Governing equations:
        1. P_out = P_set (constant pressure)
        2. Q = mdot * (h_out - h_in) (energy balance)

    State variables:
        - h_out: outlet specific enthalpy [J/kg]
        - mdot: mass flow rate [kg/s]

    Parameters:
        P: Operating pressure [Pa]
        Q: Heat addition rate [W] (positive = heating, negative = cooling)
        fluid: Fluid name for CoolProp (default 'Water')
    """

    def __init__(self, name: str, P: float, Q: float, fluid: str = 'Water'):
        super().__init__(name)

        self.P = P
        self.Q = Q
        self.fluid_name = fluid
        self.fluid = FluidProperties(fluid)

        # Create ports with initial reasonable values
        self.inlet = MassFlowPort('inlet', direction='in')
        self.outlet = MassFlowPort('outlet', direction='out')

        # Initialize with reasonable defaults for water at operating pressure
        self.inlet.P = P
        self.inlet.h = 1e6  # ~239°C for water
        self.inlet.mdot = 100.0
        self.outlet.P = P
        self.outlet.h = 1e6
        self.outlet.mdot = 100.0

        self.ports = {
            'inlet': self.inlet,
            'outlet': self.outlet,
        }

    def get_variables(self):
        # Set initial guess based on operating pressure and heat addition
        # For water: low P → h~0.2 MJ/kg, high P liquid → h~1.4 MJ/kg, high P vapor → h~2.7 MJ/kg
        if self.P > 1e6:  # High pressure
            if self.Q > 0:  # Boiler (heating)
                h_initial = 2.7e6  # Saturated vapor
            else:  # High pressure cooler
                h_initial = 1.4e6  # Saturated liquid
        else:  # Low pressure
            h_initial = 2e5  # Saturated liquid at low P

        return [
            Variable('h_out', kind='algebraic', initial=h_initial, units='J/kg'),
            Variable('mdot', kind='algebraic', initial=100.0, units='kg/s'),
        ]

    def get_initial_state(self):
        # Use same logic as get_variables for consistency
        if self.P > 1e6:
            if self.Q > 0:
                h0 = 2.7e6
            else:
                h0 = 1.4e6
        else:
            h0 = 2e5
        mdot0 = 100.0
        return np.array([h0, mdot0])

    def residual(self, state, ports, t):
        h_out, mdot = state

        # Get inlet conditions from connected component
        h_in = ports['inlet'].h

        # Equation 1: Outlet pressure equals setpoint
        eq_pressure = ports['outlet'].P - self.P

        # Equation 2: Energy balance
        # Q = mdot * (h_out - h_in)
        # Rearranged as residual:
        eq_energy = self.Q - mdot * (h_out - h_in)

        # Update outlet port for next component
        ports['outlet'].mdot = mdot
        ports['outlet'].h = h_out
        ports['outlet'].P = self.P

        return np.array([eq_pressure, eq_energy])
