"""
Two-phase heater/cooler component.

Handles heating/cooling that may cross phase boundaries (subcooled → two-phase → superheated).
Can represent boilers, evaporators, or condensers with two-phase flow.
"""

import numpy as np
from thermal_sim.core.component import Component
from thermal_sim.core.variable import Variable
from thermal_sim.core.port import MassFlowPort
from thermal_sim.properties.coolprop_wrapper import FluidProperties


class TwoPhaseHeater(Component):
    """
    Heats or cools fluid at constant pressure, handling two-phase flow.

    Unlike ConstantPressureHeater (single-phase only), this component:
    - Detects and reports phase (liquid, two_phase, vapor)
    - Tracks quality when in two-phase region
    - Works correctly across phase boundaries

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

    Example:
        # Boiler that produces two-phase mixture
        boiler = TwoPhaseHeater('boiler', P=1e6, Q=50e6)

        # After solution, check phase:
        phase = boiler.get_outlet_phase()
        if phase == 'two_phase':
            quality = boiler.get_outlet_quality()
            print(f"Outlet quality: {quality:.1%}")
    """

    def __init__(self, name: str, P: float, Q: float, fluid: str = 'Water'):
        super().__init__(name)

        self.P = P
        self.Q = Q
        self.fluid_name = fluid
        self.fluid = FluidProperties(fluid)

        # Create ports
        self.inlet = MassFlowPort('inlet', direction='in')
        self.outlet = MassFlowPort('outlet', direction='out')

        # Initialize with reasonable defaults
        self.inlet.P = P
        self.inlet.h = 1e6
        self.inlet.mdot = 100.0
        self.outlet.P = P
        self.outlet.h = 1.5e6  # Likely in two-phase for water
        self.outlet.mdot = 100.0

        self.ports = {
            'inlet': self.inlet,
            'outlet': self.outlet,
        }

        # Cache for diagnostics (set during residual evaluation)
        self._last_outlet_h = None
        self._last_phase = None
        self._last_quality = None

    def get_variables(self):
        # Initial guess based on heat addition
        if self.P > 1e6:  # High pressure
            if self.Q > 0:  # Heating
                h_initial = 2.5e6  # Likely two-phase or vapor
            else:  # Cooling
                h_initial = 1.4e6  # Liquid
        else:  # Low pressure
            if self.Q > 0:
                h_initial = 2.0e6  # Two-phase
            else:
                h_initial = 2e5  # Liquid

        return [
            Variable('h_out', kind='algebraic', initial=h_initial, units='J/kg'),
            Variable('mdot', kind='algebraic', initial=100.0, units='kg/s'),
        ]

    def get_initial_state(self):
        # Use same logic as get_variables
        if self.P > 1e6:
            if self.Q > 0:
                h0 = 2.5e6
            else:
                h0 = 1.4e6
        else:
            if self.Q > 0:
                h0 = 2.0e6
            else:
                h0 = 2e5

        mdot0 = 100.0
        return np.array([h0, mdot0])

    def residual(self, state, ports, t, state_dot=None):
        h_out, mdot = state

        # Get inlet conditions
        h_in = ports['inlet'].h

        # Equation 1: Outlet pressure equals setpoint
        eq_pressure = ports['outlet'].P - self.P

        # Equation 2: Energy balance
        # Q = mdot * (h_out - h_in)
        eq_energy = self.Q - mdot * (h_out - h_in)

        # Update outlet port
        ports['outlet'].mdot = mdot
        ports['outlet'].h = h_out
        ports['outlet'].P = self.P

        # Cache outlet state for diagnostics
        self._last_outlet_h = h_out
        self._last_phase = self.fluid.phase_from_ph(self.P, h_out)
        self._last_quality = self.fluid.quality_from_enthalpy(self.P, h_out)

        return np.array([eq_pressure, eq_energy])

    # =========================================================================
    # Diagnostic Methods (Phase 3 additions)
    # =========================================================================

    def get_outlet_phase(self) -> str | None:
        """
        Get the current outlet phase.

        Returns:
            phase: One of 'liquid', 'two_phase', 'vapor', or None if not yet solved

        Example:
            >>> result = graph.solve_steady_state()
            >>> phase = boiler.get_outlet_phase()
            >>> print(f"Boiler outlet: {phase}")
        """
        return self._last_phase

    def get_outlet_quality(self) -> float | None:
        """
        Get the current outlet quality (vapor fraction).

        Returns:
            quality: Vapor quality in [0, 1] if two-phase, None otherwise

        Example:
            >>> result = graph.solve_steady_state()
            >>> if boiler.get_outlet_phase() == 'two_phase':
            ...     quality = boiler.get_outlet_quality()
            ...     print(f"Outlet quality: {quality:.1%}")
        """
        return self._last_quality

    def get_outlet_temperature(self) -> float | None:
        """
        Get the current outlet temperature.

        Returns:
            T: Temperature [K], or None if not yet solved

        Note:
            For two-phase mixtures, returns saturation temperature.
        """
        if self._last_outlet_h is None:
            return None

        if self._last_phase == 'two_phase':
            # Two-phase: return saturation temperature
            return self.fluid.saturation_temperature(self.P)
        else:
            # Single-phase: compute from (P, h)
            return self.fluid.temperature(self.P, self._last_outlet_h)

    def __repr__(self):
        """String representation with phase info if available"""
        base = f"TwoPhaseHeater('{self.name}', P={self.P/1e6:.1f} MPa, Q={self.Q/1e6:.1f} MW)"

        if self._last_phase is not None:
            if self._last_phase == 'two_phase' and self._last_quality is not None:
                return f"{base} [phase: {self._last_phase}, quality: {self._last_quality:.2f}]"
            else:
                return f"{base} [phase: {self._last_phase}]"

        return base
