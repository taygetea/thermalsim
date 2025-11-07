"""
Variable metadata for state vector construction and debugging.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Variable:
    """
    Describes a state variable in the system.

    Attributes:
        name: Human-readable identifier (e.g., 'h_out', 'P_inlet')
        kind: Whether this appears as dx/dt (differential) or algebraic constraint
        initial: Initial guess for solver (must be physically reasonable)
        units: String description for documentation (not enforced)
    """
    name: str
    kind: Literal['differential', 'algebraic']
    initial: float
    units: str = ""

    def __repr__(self) -> str:
        return f"Variable({self.name}: {self.kind}, x0={self.initial} {self.units})"
