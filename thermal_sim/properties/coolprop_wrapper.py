"""
Wrapper around CoolProp with caching for performance.

CoolProp is the industry-standard thermodynamic property library.
This wrapper adds LRU caching to avoid redundant expensive calls.
"""

from functools import lru_cache
from CoolProp.CoolProp import PropsSI


class FluidProperties:
    """
    Interface to fluid thermodynamic properties via CoolProp.

    All methods use SI units:
        Pressure: Pa
        Temperature: K
        Enthalpy: J/kg
        Entropy: J/(kg·K)
        Density: kg/m³

    Example:
        water = FluidProperties('Water')
        h = water.enthalpy(P=1e6, T=500.0)  # Enthalpy at 1 MPa, 500 K
        T = water.temperature(P=1e6, h=h)   # Should return 500.0
    """

    def __init__(self, fluid_name: str):
        """
        Initialize for a specific fluid.

        Args:
            fluid_name: CoolProp fluid name (e.g., 'Water', 'Air', 'CO2', 'Helium')
        """
        self.fluid = fluid_name

        # Validate that CoolProp recognizes this fluid
        try:
            PropsSI('T', 'P', 1e5, 'Q', 0, fluid_name)
        except ValueError as e:
            raise ValueError(f"Unknown fluid '{fluid_name}' for CoolProp") from e

    @lru_cache(maxsize=10000)
    def enthalpy(self, P: float, T: float) -> float:
        """Get specific enthalpy from pressure and temperature"""
        return PropsSI('H', 'P', P, 'T', T, self.fluid)

    @lru_cache(maxsize=10000)
    def temperature(self, P: float, h: float) -> float:
        """Get temperature from pressure and enthalpy"""
        return PropsSI('T', 'P', P, 'H', h, self.fluid)

    @lru_cache(maxsize=10000)
    def entropy(self, P: float, h: float) -> float:
        """Get specific entropy from pressure and enthalpy"""
        return PropsSI('S', 'P', P, 'H', h, self.fluid)

    @lru_cache(maxsize=10000)
    def entropy_from_pt(self, P: float, T: float) -> float:
        """Get specific entropy from pressure and temperature"""
        return PropsSI('S', 'P', P, 'T', T, self.fluid)

    @lru_cache(maxsize=10000)
    def enthalpy_from_ps(self, P: float, s: float) -> float:
        """Get specific enthalpy from pressure and entropy (isentropic relations)"""
        return PropsSI('H', 'P', P, 'S', s, self.fluid)

    @lru_cache(maxsize=10000)
    def density(self, P: float, T: float) -> float:
        """Get density from pressure and temperature"""
        return PropsSI('D', 'P', P, 'T', T, self.fluid)

    # =========================================================================
    # Two-Phase Properties (Phase 3)
    # =========================================================================

    @lru_cache(maxsize=1000)
    def saturation_temperature(self, P: float) -> float:
        """
        Get saturation temperature at pressure P.

        Args:
            P: Pressure [Pa]

        Returns:
            T_sat: Saturation temperature [K]

        Example:
            >>> water = FluidProperties('Water')
            >>> T_sat = water.saturation_temperature(1e5)  # ~373 K at 1 bar
        """
        return PropsSI('T', 'P', P, 'Q', 0.0, self.fluid)

    @lru_cache(maxsize=1000)
    def saturation_pressure(self, T: float) -> float:
        """
        Get saturation pressure at temperature T.

        Args:
            T: Temperature [K]

        Returns:
            P_sat: Saturation pressure [Pa]

        Example:
            >>> water = FluidProperties('Water')
            >>> P_sat = water.saturation_pressure(373.15)  # ~1e5 Pa at 100°C
        """
        return PropsSI('P', 'T', T, 'Q', 0.0, self.fluid)

    @lru_cache(maxsize=1000)
    def enthalpy_saturated_liquid(self, P: float) -> float:
        """
        Get saturated liquid enthalpy (h_f) at pressure P.

        Args:
            P: Pressure [Pa]

        Returns:
            h_f: Saturated liquid specific enthalpy [J/kg]

        Example:
            >>> water = FluidProperties('Water')
            >>> h_f = water.enthalpy_saturated_liquid(1e5)  # ~419 kJ/kg
        """
        return PropsSI('H', 'P', P, 'Q', 0.0, self.fluid)

    @lru_cache(maxsize=1000)
    def enthalpy_saturated_vapor(self, P: float) -> float:
        """
        Get saturated vapor enthalpy (h_g) at pressure P.

        Args:
            P: Pressure [Pa]

        Returns:
            h_g: Saturated vapor specific enthalpy [J/kg]

        Example:
            >>> water = FluidProperties('Water')
            >>> h_g = water.enthalpy_saturated_vapor(1e5)  # ~2676 kJ/kg
        """
        return PropsSI('H', 'P', P, 'Q', 1.0, self.fluid)

    @lru_cache(maxsize=10000)
    def enthalpy_from_quality(self, P: float, quality: float) -> float:
        """
        Get enthalpy from pressure and quality.

        Args:
            P: Pressure [Pa]
            quality: Vapor quality (0 = saturated liquid, 1 = saturated vapor)

        Returns:
            h: Specific enthalpy [J/kg]

        Raises:
            ValueError: If quality is not in [0, 1]

        Example:
            >>> water = FluidProperties('Water')
            >>> h = water.enthalpy_from_quality(1e5, 0.5)  # 50% quality
        """
        if not 0 <= quality <= 1:
            raise ValueError(f"Quality must be in [0, 1], got {quality}")
        return PropsSI('H', 'P', P, 'Q', quality, self.fluid)

    def quality_from_enthalpy(self, P: float, h: float) -> float | None:
        """
        Compute quality from pressure and enthalpy.

        Returns None if fluid is single-phase (subcooled or superheated).

        Args:
            P: Pressure [Pa]
            h: Specific enthalpy [J/kg]

        Returns:
            quality: Vapor quality in [0, 1], or None if single-phase

        Example:
            >>> water = FluidProperties('Water')
            >>> x = water.quality_from_enthalpy(1e5, 1.5e6)  # Two-phase
            >>> x_none = water.quality_from_enthalpy(1e5, 5e5)  # Subcooled → None
        """
        h_f = self.enthalpy_saturated_liquid(P)
        h_g = self.enthalpy_saturated_vapor(P)

        if h < h_f:
            return None  # Subcooled liquid
        elif h > h_g:
            return None  # Superheated vapor
        else:
            # Two-phase: x = (h - h_f) / h_fg
            return (h - h_f) / (h_g - h_f)

    def phase_from_ph(self, P: float, h: float) -> str:
        """
        Determine phase from pressure and enthalpy.

        Args:
            P: Pressure [Pa]
            h: Specific enthalpy [J/kg]

        Returns:
            phase: One of 'liquid', 'two_phase', 'vapor'

        Example:
            >>> water = FluidProperties('Water')
            >>> water.phase_from_ph(1e5, 5e5)      # 'liquid'
            >>> water.phase_from_ph(1e5, 1.5e6)    # 'two_phase'
            >>> water.phase_from_ph(1e5, 3e6)      # 'vapor'
        """
        h_f = self.enthalpy_saturated_liquid(P)
        h_g = self.enthalpy_saturated_vapor(P)

        if h < h_f:
            return 'liquid'
        elif h > h_g:
            return 'vapor'
        else:
            return 'two_phase'

    def clear_cache(self):
        """Clear LRU caches (useful for memory management in long runs)"""
        self.enthalpy.cache_clear()
        self.temperature.cache_clear()
        self.entropy.cache_clear()
        self.entropy_from_pt.cache_clear()
        self.enthalpy_from_ps.cache_clear()
        self.density.cache_clear()
        # Two-phase caches
        self.saturation_temperature.cache_clear()
        self.saturation_pressure.cache_clear()
        self.enthalpy_saturated_liquid.cache_clear()
        self.enthalpy_saturated_vapor.cache_clear()
        self.enthalpy_from_quality.cache_clear()
