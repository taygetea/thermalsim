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

    def clear_cache(self):
        """Clear LRU caches (useful for memory management in long runs)"""
        self.enthalpy.cache_clear()
        self.temperature.cache_clear()
        self.entropy.cache_clear()
        self.entropy_from_pt.cache_clear()
        self.enthalpy_from_ps.cache_clear()
        self.density.cache_clear()
