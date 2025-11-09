"""
Port definitions for typed connections between components.

Ports are the interface through which components exchange mass, energy, and information.
Type checking at connection time prevents physically invalid configurations.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from abc import ABC, abstractmethod


class Port(ABC):
    """Base class for all port types"""

    @abstractmethod
    def compatible_with(self, other: 'Port') -> bool:
        """Check if this port can connect to another port"""
        pass


@dataclass
class MassFlowPort(Port):
    """
    Represents a stream of flowing fluid (liquid or gas).

    Convention:
        - mdot > 0: flow OUT of the component
        - mdot < 0: flow INTO the component

    Attributes:
        name: Unique identifier for this port
        direction: Whether this port is an inlet or outlet
        mdot: Mass flow rate [kg/s]
        h: Specific enthalpy [J/kg]
        P: Static pressure [Pa]
        component: Reference to parent component (set by component)
    """
    name: str
    direction: Literal['in', 'out']

    # State variables (updated during solution)
    mdot: float = 0.0
    h: float = 0.0
    P: float = 0.0

    # Metadata
    component: Optional[object] = field(default=None, repr=False)

    def compatible_with(self, other: Port) -> bool:
        """Mass ports can only connect to other mass ports with opposite direction"""
        if not isinstance(other, MassFlowPort):
            return False
        return self.direction != other.direction

    def __repr__(self) -> str:
        comp_name = self.component.name if self.component else "unattached"
        return f"MassFlowPort({comp_name}.{self.name}, {self.direction})"


@dataclass
class HeatPort(Port):
    """
    Represents pure heat transfer (no mass flow).

    Convention:
        - Q > 0: heat flow INTO the component
        - Q < 0: heat flow OUT of the component

    Attributes:
        name: Unique identifier for this port
        direction: Whether this port receives or provides heat
        Q: Heat flow rate [W]
        T: Temperature [K]
        component: Reference to parent component
    """
    name: str
    direction: Literal['in', 'out']

    Q: float = 0.0
    T: float = 0.0

    component: Optional[object] = field(default=None, repr=False)

    def compatible_with(self, other: Port) -> bool:
        """Heat ports connect to heat ports with opposite direction"""
        if not isinstance(other, HeatPort):
            return False
        return self.direction != other.direction

    def __repr__(self) -> str:
        comp_name = self.component.name if self.component else "unattached"
        return f"HeatPort({comp_name}.{self.name}, {self.direction})"


@dataclass
class ScalarPort(Port):
    """
    Represents a scalar control signal or measurement.

    Used for control systems (PID controllers, sensors, actuators).

    Attributes:
        name: Unique identifier
        direction: 'in' for actuator/command, 'out' for sensor/measurement
        value: Scalar signal value (units depend on context)
    """
    name: str
    direction: Literal['in', 'out']
    value: float = 0.0

    component: Optional[object] = field(default=None, repr=False)

    def compatible_with(self, other: Port) -> bool:
        """Scalar ports connect to scalar ports with opposite direction"""
        if not isinstance(other, ScalarPort):
            return False
        return self.direction != other.direction

    def __repr__(self) -> str:
        comp_name = self.component.name if self.component else "unattached"
        return f"ScalarPort({comp_name}.{self.name}, {self.direction})"
