"""Unit tests for TwoPhaseHeater component."""

import pytest
import numpy as np
from thermal_sim.components.two_phase_heater import TwoPhaseHeater
from thermal_sim.properties.coolprop_wrapper import FluidProperties


class TestTwoPhaseHeaterCreation:
    """Test TwoPhaseHeater initialization"""

    def test_creation(self):
        """TwoPhaseHeater should initialize with required parameters"""
        heater = TwoPhaseHeater('test_heater', P=1e6, Q=50e6)

        assert heater.name == 'test_heater'
        assert heater.P == 1e6
        assert heater.Q == 50e6
        assert heater.fluid_name == 'Water'

    def test_custom_fluid(self):
        """TwoPhaseHeater should accept custom fluid"""
        heater = TwoPhaseHeater('co2_heater', P=5e6, Q=10e6, fluid='CO2')

        assert heater.fluid_name == 'CO2'

    def test_has_ports(self):
        """TwoPhaseHeater should have inlet and outlet MassFlowPorts"""
        heater = TwoPhaseHeater('heater', P=1e6, Q=10e6)

        assert 'inlet' in heater.ports
        assert 'outlet' in heater.ports
        assert heater.ports['inlet'].direction == 'in'
        assert heater.ports['outlet'].direction == 'out'


class TestTwoPhaseHeaterInterface:
    """Test Component interface implementation"""

    @pytest.fixture
    def heater(self):
        """Standard heater for tests"""
        return TwoPhaseHeater('heater', P=1e6, Q=50e6)

    def test_get_variables(self, heater):
        """get_variables() should return 2 algebraic variables"""
        variables = heater.get_variables()

        assert len(variables) == 2
        assert variables[0].name == 'h_out'
        assert variables[1].name == 'mdot'
        assert all(v.kind == 'algebraic' for v in variables)

    def test_get_initial_state(self, heater):
        """get_initial_state() should return array of length 2"""
        state = heater.get_initial_state()

        assert isinstance(state, np.ndarray)
        assert len(state) == 2
        assert all(np.isfinite(state))

    def test_residual_shape(self, heater):
        """residual() should return array matching state length"""
        state = np.array([1.5e6, 100.0])  # h_out, mdot
        ports = heater.ports

        # Set inlet conditions
        ports['inlet'].h = 1.0e6
        ports['inlet'].P = 1e6

        residual = heater.residual(state, ports, t=0.0)

        assert isinstance(residual, np.ndarray)
        assert len(residual) == 2


class TestTwoPhaseHeaterPhysics:
    """Test physical correctness of TwoPhaseHeater"""

    def test_energy_balance_simple(self):
        """Energy balance: Q = mdot * (h_out - h_in)"""
        heater = TwoPhaseHeater('heater', P=1e6, Q=50e6)

        # Set up conditions
        h_in = 1.0e6  # J/kg
        mdot = 100.0  # kg/s
        h_out = h_in + heater.Q / mdot  # Should be 1.5e6 J/kg

        state = np.array([h_out, mdot])
        heater.ports['inlet'].h = h_in
        heater.ports['inlet'].P = 1e6

        residual = heater.residual(state, heater.ports, t=0.0)

        # Energy residual should be near zero
        eq_energy = residual[1]
        assert abs(eq_energy) < 1.0, f"Energy balance not satisfied: {eq_energy}"

    def test_negative_Q_cooling(self):
        """Negative Q should cool the fluid (h_out < h_in)"""
        condenser = TwoPhaseHeater('condenser', P=1e5, Q=-80e6)

        h_in = 2.5e6  # Vapor
        mdot = 200.0
        h_out = h_in + condenser.Q / mdot  # Should be 2.1e6 J/kg

        state = np.array([h_out, mdot])
        condenser.ports['inlet'].h = h_in

        residual = condenser.residual(state, condenser.ports, t=0.0)

        # Should have h_out < h_in
        assert h_out < h_in, "Cooling should reduce enthalpy"

        # Energy balance should be satisfied
        assert abs(residual[1]) < 1.0


class TestTwoPhaseDetection:
    """Test phase detection and quality tracking"""

    @pytest.fixture
    def water(self):
        return FluidProperties('Water')

    def test_subcooled_liquid_detection(self, water):
        """Heater with subcooled outlet should detect 'liquid' phase"""
        heater = TwoPhaseHeater('heater', P=1e5, Q=10e6)

        # Set up for subcooled liquid outlet
        h_f = water.enthalpy_saturated_liquid(1e5)
        h_in = h_f - 100e3  # Subcooled inlet
        mdot = 100.0
        h_out = h_f - 50e3  # Still subcooled

        state = np.array([h_out, mdot])
        heater.ports['inlet'].h = h_in
        heater.ports['inlet'].P = 1e5

        heater.residual(state, heater.ports, t=0.0)

        # Check phase detection
        phase = heater.get_outlet_phase()
        assert phase == 'liquid', f"Expected 'liquid', got '{phase}'"

        # Quality should be None for single-phase
        quality = heater.get_outlet_quality()
        assert quality is None, "Liquid should have quality = None"

    def test_two_phase_detection(self, water):
        """Heater with two-phase outlet should detect 'two_phase' and quality"""
        heater = TwoPhaseHeater('heater', P=1e5, Q=100e6)

        # Set up for two-phase outlet
        h_f = water.enthalpy_saturated_liquid(1e5)
        h_g = water.enthalpy_saturated_vapor(1e5)
        h_in = h_f  # Saturated liquid inlet
        mdot = 500.0  # kg/s
        h_out = 0.4 * h_f + 0.6 * h_g  # 60% quality

        state = np.array([h_out, mdot])
        heater.ports['inlet'].h = h_in
        heater.ports['inlet'].P = 1e5

        heater.residual(state, heater.ports, t=0.0)

        # Check phase detection
        phase = heater.get_outlet_phase()
        assert phase == 'two_phase', f"Expected 'two_phase', got '{phase}'"

        # Check quality
        quality = heater.get_outlet_quality()
        assert quality is not None, "Two-phase should have quality"
        assert 0 <= quality <= 1, f"Quality {quality} not in [0,1]"
        assert abs(quality - 0.6) < 0.01, f"Expected quality ~0.6, got {quality}"

    def test_superheated_vapor_detection(self, water):
        """Heater with superheated outlet should detect 'vapor' phase"""
        heater = TwoPhaseHeater('heater', P=1e5, Q=200e6)

        # Set up for superheated vapor outlet
        h_g = water.enthalpy_saturated_vapor(1e5)
        h_in = h_g  # Saturated vapor inlet
        mdot = 100.0
        h_out = h_g + 200e3  # Superheated

        state = np.array([h_out, mdot])
        heater.ports['inlet'].h = h_in
        heater.ports['inlet'].P = 1e5

        heater.residual(state, heater.ports, t=0.0)

        # Check phase detection
        phase = heater.get_outlet_phase()
        assert phase == 'vapor', f"Expected 'vapor', got '{phase}'"

        # Quality should be None for single-phase
        quality = heater.get_outlet_quality()
        assert quality is None, "Vapor should have quality = None"


class TestTwoPhaseHeaterDiagnostics:
    """Test diagnostic methods"""

    def test_get_outlet_temperature_liquid(self):
        """get_outlet_temperature() should work for liquid"""
        water = FluidProperties('Water')
        heater = TwoPhaseHeater('heater', P=1e6, Q=50e6)

        h_in = 1.0e6
        h_out = 1.2e6  # Subcooled liquid
        mdot = 100.0

        state = np.array([h_out, mdot])
        heater.ports['inlet'].h = h_in
        heater.ports['inlet'].P = 1e6

        heater.residual(state, heater.ports, t=0.0)

        T_out = heater.get_outlet_temperature()
        assert T_out is not None
        assert T_out > 0, "Temperature should be positive"

        # Verify it matches CoolProp
        T_expected = water.temperature(1e6, h_out)
        assert abs(T_out - T_expected) < 0.1, \
            f"Temperature mismatch: {T_out} vs {T_expected}"

    def test_get_outlet_temperature_two_phase(self):
        """get_outlet_temperature() should return T_sat for two-phase"""
        water = FluidProperties('Water')
        heater = TwoPhaseHeater('heater', P=1e5, Q=100e6)

        h_f = water.enthalpy_saturated_liquid(1e5)
        h_g = water.enthalpy_saturated_vapor(1e5)
        h_out = 0.5 * (h_f + h_g)  # Two-phase
        mdot = 500.0

        state = np.array([h_out, mdot])
        heater.ports['inlet'].h = h_f
        heater.ports['inlet'].P = 1e5

        heater.residual(state, heater.ports, t=0.0)

        T_out = heater.get_outlet_temperature()
        T_sat = water.saturation_temperature(1e5)

        assert T_out is not None
        assert abs(T_out - T_sat) < 0.1, \
            f"Two-phase temp should be T_sat: {T_out} vs {T_sat}"

    def test_repr_with_phase_info(self):
        """__repr__ should include phase info after residual evaluation"""
        water = FluidProperties('Water')
        heater = TwoPhaseHeater('boiler', P=10e6, Q=100e6)

        # Before evaluation
        repr_before = repr(heater)
        assert 'TwoPhaseHeater' in repr_before
        assert 'phase:' not in repr_before  # No phase info yet

        # After evaluation with two-phase result
        h_f = water.enthalpy_saturated_liquid(10e6)
        h_g = water.enthalpy_saturated_vapor(10e6)
        h_out = 0.7 * h_f + 0.3 * h_g
        state = np.array([h_out, 100.0])
        heater.ports['inlet'].h = h_f

        heater.residual(state, heater.ports, t=0.0)

        repr_after = repr(heater)
        assert 'phase: two_phase' in repr_after
        assert 'quality:' in repr_after


class TestTwoPhaseHeaterEdgeCases:
    """Test edge cases and error handling"""

    def test_zero_mass_flow(self):
        """Zero mass flow should not cause division by zero"""
        heater = TwoPhaseHeater('heater', P=1e5, Q=50e6)

        state = np.array([1.5e6, 0.0])  # mdot = 0
        heater.ports['inlet'].h = 1.0e6

        # Should not raise error
        residual = heater.residual(state, heater.ports, t=0.0)

        # Residual should indicate non-physical solution
        assert np.all(np.isfinite(residual)), "Residual should be finite"

    def test_phase_transition_crossing(self):
        """Heater that crosses phase boundary should work correctly"""
        water = FluidProperties('Water')
        heater = TwoPhaseHeater('boiler', P=1e5, Q=300e6)

        # Inlet: subcooled liquid
        # Outlet: calculate from energy balance (should be superheated vapor)
        h_f = water.enthalpy_saturated_liquid(1e5)
        h_g = water.enthalpy_saturated_vapor(1e5)

        h_in = h_f - 50e3  # Subcooled (~367 kJ/kg)
        mdot = 100.0  # kg/s
        h_out = h_in + heater.Q / mdot  # +3000 kJ/kg → ~3367 kJ/kg

        # Verify outlet is indeed superheated (crosses through two-phase)
        # h_g ≈ 2675 kJ/kg, so h_out should be > h_g
        assert h_out > h_g, f"Outlet should be superheated: {h_out/1e3:.1f} kJ/kg > {h_g/1e3:.1f} kJ/kg"

        state = np.array([h_out, mdot])
        heater.ports['inlet'].h = h_in

        # Should handle phase transition without issues
        residual = heater.residual(state, heater.ports, t=0.0)

        # Energy residual should be near zero
        assert abs(residual[1]) < 1.0, f"Energy balance not satisfied: {residual[1]}"

        # Should detect vapor phase
        phase = heater.get_outlet_phase()
        assert phase == 'vapor', f"Expected superheated vapor, got {phase}"
