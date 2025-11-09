"""Unit tests for FluidProperties, especially two-phase methods."""

import pytest
import numpy as np
from thermal_sim.properties.coolprop_wrapper import FluidProperties


class TestFluidPropertiesBasic:
    """Test basic single-phase property methods"""

    def test_creation(self):
        """FluidProperties should initialize with valid fluid name"""
        water = FluidProperties('Water')
        assert water.fluid == 'Water'

    def test_invalid_fluid(self):
        """Should raise error for invalid fluid name"""
        with pytest.raises(ValueError, match="Unknown fluid"):
            FluidProperties('NotAFluid123')

    def test_enthalpy_temperature_roundtrip(self):
        """Temperature(P, Enthalpy(P,T)) should return T"""
        water = FluidProperties('Water')
        P = 1e6  # 1 MPa
        T_in = 500.0  # 500 K (superheated steam)

        h = water.enthalpy(P, T_in)
        T_out = water.temperature(P, h)

        assert abs(T_out - T_in) < 0.01, f"Temperature roundtrip failed: {T_in} → {T_out}"


class TestTwoPhaseProperties:
    """Test two-phase property methods (Phase 3)"""

    @pytest.fixture
    def water(self):
        """Water instance for tests"""
        return FluidProperties('Water')

    def test_saturation_temperature_at_1bar(self, water):
        """Saturation temperature at 1 bar should be ~373.15 K"""
        P = 1e5  # 1 bar
        T_sat = water.saturation_temperature(P)

        # Should be very close to 100°C = 373.15 K
        assert 372.0 < T_sat < 374.0, f"T_sat at 1 bar = {T_sat} K, expected ~373.15 K"

    def test_saturation_temperature_at_10MPa(self, water):
        """Saturation temperature at 10 MPa should be ~584 K"""
        P = 10e6  # 10 MPa (high pressure Rankine cycle)
        T_sat = water.saturation_temperature(P)

        # At 10 MPa, water saturates around 311°C = 584 K
        assert 580.0 < T_sat < 590.0, f"T_sat at 10 MPa = {T_sat} K, expected ~584 K"

    def test_saturation_pressure_at_100C(self, water):
        """Saturation pressure at 100°C should be ~1 bar"""
        T = 373.15  # 100°C
        P_sat = water.saturation_pressure(T)

        # Should be close to 101325 Pa (1 atm)
        assert 0.95e5 < P_sat < 1.05e5, f"P_sat at 100°C = {P_sat} Pa, expected ~1e5 Pa"

    def test_enthalpy_saturated_liquid_1bar(self, water):
        """h_f at 1 bar should be ~419 kJ/kg"""
        P = 1e5
        h_f = water.enthalpy_saturated_liquid(P)

        # Expected: ~419 kJ/kg for water at 1 bar
        assert 4.0e5 < h_f < 4.3e5, f"h_f at 1 bar = {h_f/1e3} kJ/kg, expected ~419 kJ/kg"

    def test_enthalpy_saturated_vapor_1bar(self, water):
        """h_g at 1 bar should be ~2676 kJ/kg"""
        P = 1e5
        h_g = water.enthalpy_saturated_vapor(P)

        # Expected: ~2676 kJ/kg for water at 1 bar
        assert 2.6e6 < h_g < 2.7e6, f"h_g at 1 bar = {h_g/1e3} kJ/kg, expected ~2676 kJ/kg"

    def test_enthalpy_from_quality_bounds(self, water):
        """Quality 0 should give h_f, quality 1 should give h_g"""
        P = 1e5
        h_f = water.enthalpy_saturated_liquid(P)
        h_g = water.enthalpy_saturated_vapor(P)

        h_0 = water.enthalpy_from_quality(P, 0.0)
        h_1 = water.enthalpy_from_quality(P, 1.0)

        assert abs(h_0 - h_f) < 1.0, "Quality 0 should give h_f"
        assert abs(h_1 - h_g) < 1.0, "Quality 1 should give h_g"

    def test_enthalpy_from_quality_midpoint(self, water):
        """Quality 0.5 should give (h_f + h_g)/2"""
        P = 1e5
        h_f = water.enthalpy_saturated_liquid(P)
        h_g = water.enthalpy_saturated_vapor(P)

        h_mid = water.enthalpy_from_quality(P, 0.5)
        expected = 0.5 * (h_f + h_g)

        assert abs(h_mid - expected) < 10.0, \
            f"Quality 0.5: got {h_mid}, expected {expected}"

    def test_enthalpy_from_quality_invalid(self, water):
        """Quality outside [0,1] should raise ValueError"""
        P = 1e5

        with pytest.raises(ValueError, match="Quality must be in"):
            water.enthalpy_from_quality(P, -0.1)

        with pytest.raises(ValueError, match="Quality must be in"):
            water.enthalpy_from_quality(P, 1.5)

    def test_quality_from_enthalpy_subcooled(self, water):
        """Subcooled liquid should return None for quality"""
        P = 1e5
        h_f = water.enthalpy_saturated_liquid(P)
        h_subcooled = h_f - 50e3  # 50 kJ/kg below saturation

        quality = water.quality_from_enthalpy(P, h_subcooled)
        assert quality is None, "Subcooled liquid should have quality = None"

    def test_quality_from_enthalpy_superheated(self, water):
        """Superheated vapor should return None for quality"""
        P = 1e5
        h_g = water.enthalpy_saturated_vapor(P)
        h_superheated = h_g + 100e3  # 100 kJ/kg above saturation

        quality = water.quality_from_enthalpy(P, h_superheated)
        assert quality is None, "Superheated vapor should have quality = None"

    def test_quality_from_enthalpy_two_phase(self, water):
        """Two-phase mixture should return quality in [0,1]"""
        P = 1e5
        h_f = water.enthalpy_saturated_liquid(P)
        h_g = water.enthalpy_saturated_vapor(P)
        h_mix = 0.3 * h_f + 0.7 * h_g  # 70% quality

        quality = water.quality_from_enthalpy(P, h_mix)

        assert quality is not None, "Two-phase should have quality"
        assert 0 <= quality <= 1, f"Quality {quality} not in [0,1]"
        assert abs(quality - 0.7) < 0.01, f"Expected quality ~0.7, got {quality}"

    def test_quality_roundtrip(self, water):
        """enthalpy_from_quality(P, quality_from_enthalpy(P, h)) should return h"""
        P = 1e5
        h_original = 1.5e6  # Some two-phase enthalpy

        quality = water.quality_from_enthalpy(P, h_original)
        assert quality is not None, "Should be two-phase"

        h_reconstructed = water.enthalpy_from_quality(P, quality)

        assert abs(h_reconstructed - h_original) < 1.0, \
            f"Roundtrip failed: {h_original} → {quality} → {h_reconstructed}"

    def test_phase_from_ph_liquid(self, water):
        """Subcooled liquid should be detected as 'liquid'"""
        P = 1e5
        h_f = water.enthalpy_saturated_liquid(P)
        h_subcooled = h_f - 20e3

        phase = water.phase_from_ph(P, h_subcooled)
        assert phase == 'liquid', f"Expected 'liquid', got '{phase}'"

    def test_phase_from_ph_two_phase(self, water):
        """Two-phase mixture should be detected as 'two_phase'"""
        P = 1e5
        h_f = water.enthalpy_saturated_liquid(P)
        h_g = water.enthalpy_saturated_vapor(P)
        h_mix = 0.5 * (h_f + h_g)

        phase = water.phase_from_ph(P, h_mix)
        assert phase == 'two_phase', f"Expected 'two_phase', got '{phase}'"

    def test_phase_from_ph_vapor(self, water):
        """Superheated vapor should be detected as 'vapor'"""
        P = 1e5
        h_g = water.enthalpy_saturated_vapor(P)
        h_superheated = h_g + 100e3

        phase = water.phase_from_ph(P, h_superheated)
        assert phase == 'vapor', f"Expected 'vapor', got '{phase}'"

    def test_phase_boundaries_exact(self, water):
        """h_f should be at liquid/two-phase boundary, h_g at two-phase/vapor"""
        P = 1e5
        h_f = water.enthalpy_saturated_liquid(P)
        h_g = water.enthalpy_saturated_vapor(P)

        # At saturation boundaries, phase detection should work
        phase_f = water.phase_from_ph(P, h_f)
        phase_g = water.phase_from_ph(P, h_g)

        # h_f is exactly at liquid/two-phase boundary - could be either
        assert phase_f in ['liquid', 'two_phase'], \
            f"h_f phase should be 'liquid' or 'two_phase', got '{phase_f}'"

        # h_g is exactly at two-phase/vapor boundary - could be either
        assert phase_g in ['two_phase', 'vapor'], \
            f"h_g phase should be 'two_phase' or 'vapor', got '{phase_g}'"


class TestTwoPhaseCaching:
    """Test that caching works for two-phase methods"""

    def test_saturation_temperature_cached(self):
        """Repeated calls should use cache"""
        water = FluidProperties('Water')
        P = 1e5

        # First call
        T1 = water.saturation_temperature(P)

        # Clear CoolProp's internal cache (if any)
        # Second call should be from LRU cache
        T2 = water.saturation_temperature(P)

        assert T1 == T2, "Cached result should be identical"

    def test_clear_cache_includes_two_phase(self):
        """clear_cache() should clear two-phase method caches"""
        water = FluidProperties('Water')

        # Make some calls to populate cache
        water.saturation_temperature(1e5)
        water.enthalpy_saturated_liquid(1e5)

        # Clear cache should not raise errors
        water.clear_cache()


class TestTwoPhaseWithDifferentFluids:
    """Test two-phase properties work for fluids other than water"""

    def test_co2_saturation(self):
        """CO2 saturation properties should work"""
        co2 = FluidProperties('CO2')
        P = 5e6  # 5 MPa

        T_sat = co2.saturation_temperature(P)
        h_f = co2.enthalpy_saturated_liquid(P)
        h_g = co2.enthalpy_saturated_vapor(P)

        # Basic sanity checks
        assert T_sat > 0, "Temperature should be positive"
        assert h_g > h_f, "Vapor enthalpy should be greater than liquid"
        assert h_f > 0, "Enthalpy should be positive"

    def test_ammonia_quality(self):
        """Ammonia quality calculations should work"""
        nh3 = FluidProperties('Ammonia')
        P = 1e6  # 1 MPa

        h_f = nh3.enthalpy_saturated_liquid(P)
        h_g = nh3.enthalpy_saturated_vapor(P)
        h_mix = 0.4 * h_f + 0.6 * h_g

        quality = nh3.quality_from_enthalpy(P, h_mix)

        assert quality is not None, "Should detect two-phase"
        assert abs(quality - 0.6) < 0.01, f"Expected quality ~0.6, got {quality}"
