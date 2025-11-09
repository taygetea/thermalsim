"""
Simple pipe with pressure drop (no heat transfer for MVP).
"""

import numpy as np
from thermal_sim.core.component import Component
from thermal_sim.core.variable import Variable
from thermal_sim.core.port import MassFlowPort


class Pipe(Component):
    """
    Simple connecting pipe with friction pressure drop.

    Assumes:
        - No heat loss (adiabatic)
        - Pressure drop proportional to mdot²

    Governing equations:
        1. h_out = h_in (adiabatic)
        2. P_out = P_in - K * mdot² (friction)

    State variables:
        - P_out: outlet pressure [Pa]
        - mdot: mass flow rate [kg/s]

    Parameters:
        K: Pressure drop coefficient [Pa/(kg/s)²]
           Typical values: 1e-3 to 1e3 depending on length/diameter
    """

    def __init__(self, name: str, K: float = 1e2):
        super().__init__(name)

        self.K = K

        self.inlet = MassFlowPort('inlet', direction='in')
        self.outlet = MassFlowPort('outlet', direction='out')

        # Initialize with reasonable defaults
        self.inlet.P = 1e6
        self.inlet.h = 1e6
        self.inlet.mdot = 100.0
        self.outlet.P = 1e6
        self.outlet.h = 1e6
        self.outlet.mdot = 100.0

        self.ports = {
            'inlet': self.inlet,
            'outlet': self.outlet,
        }

    def get_variables(self):
        return [
            Variable('P_out', kind='algebraic', initial=1e6, units='Pa'),
            Variable('mdot', kind='algebraic', initial=100.0, units='kg/s'),
        ]

    def get_initial_state(self):
        return np.array([1e6, 100.0])

    def residual(self, state, ports, t, state_dot=None):
        P_out, mdot = state

        h_in = ports['inlet'].h
        P_in = ports['inlet'].P

        # Equation 1: Adiabatic (enthalpy conserved)
        eq_enthalpy = ports['outlet'].h - h_in

        # Equation 2: Pressure drop
        # ΔP = K * mdot²
        eq_pressure = P_out - (P_in - self.K * mdot**2)

        # Update outlet
        ports['outlet'].mdot = mdot
        ports['outlet'].h = h_in
        ports['outlet'].P = P_out

        return np.array([eq_enthalpy, eq_pressure])
