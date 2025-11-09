"""
Base class for all thermal system components.

Components are the building blocks of thermal systems. Each component:
  - Declares its state variables
  - Defines residual equations that must equal zero at solution
  - Exposes ports for connection to other components
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
from thermal_sim.core.variable import Variable
from thermal_sim.core.port import Port


class Component(ABC):
    """
    Abstract base class for all thermal components.

    Subclasses must implement:
        - get_variables(): declare state variables
        - get_initial_state(): provide initial guess
        - residual(): define governing equations

    Attributes:
        name: Unique identifier for this component in the system
        ports: Dictionary of ports (populated by subclass __init__)
    """

    def __init__(self, name: str):
        """
        Initialize component.

        Args:
            name: Unique identifier (will be checked by ThermalGraph)
        """
        self.name = name
        self.ports: Dict[str, Port] = {}

    @abstractmethod
    def get_variables(self) -> List[Variable]:
        """
        Declare state variables for this component.

        Returns:
            List of Variable objects describing the state vector for this component.
            Order matters: must match the order in residual() and get_initial_state().

        Example:
            return [
                Variable('h_out', kind='differential', initial=1e6, units='J/kg'),
                Variable('mdot', kind='algebraic', initial=100.0, units='kg/s'),
            ]
        """
        pass

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """
        Provide initial guess for state variables.

        Returns:
            1D numpy array of length len(get_variables())

        Note:
            Should return the same values as Variable.initial in get_variables(),
            but as a numpy array for solver consumption.
        """
        pass

    @abstractmethod
    def residual(self,
                 state: np.ndarray,
                 ports: Dict[str, Port],
                 t: float,
                 state_dot: np.ndarray | None = None) -> np.ndarray:
        """
        Compute residual vector for this component's equations.

        The solver will drive this residual to zero.

        Args:
            state: This component's state variables (local slice of global state)
            ports: Dictionary mapping port names to Port objects (with current values)
            t: Current simulation time [s]
            state_dot: Time derivatives (dx/dt) for differential variables.
                      None for steady-state/algebraic systems (default).
                      Provided by DAE solver for transient simulations.

        Returns:
            1D numpy array of residuals (same length as state vector)

        Convention - Algebraic systems (Phase 0-2):
            Residual form: f(x) = 0

            For example, if the physics is:
                Q = mdot * (h_out - h_in)

            Write as residual:
                residual[0] = Q - mdot * (h_out - h_in)

            Solver will find state values that make residual[0] = 0.

        Convention - DAE systems (Phase 3+):
            For differential variables: residual = f(x,t) - dx/dt
            For algebraic variables: residual = f(x,t)

            For example, Tank with level dynamics:
                d(level)/dt = (mdot_in - mdot_out) / (rho * A)

            Write as residual:
                residual[0] = (mdot_in - mdot_out)/(rho*A) - state_dot[0]

            Algebraic constraint (outlet flow):
                residual[1] = mdot_out - valve_cv * sqrt(dP)

        Side effects:
            Should update outlet port values based on computed state.
            Example: ports['outlet'].h = h_out

        Backward compatibility:
            Existing components work unchanged because state_dot=None by default
            and algebraic variables don't reference state_dot.
        """
        pass

    def get_state_size(self) -> int:
        """Convenience: return number of state variables"""
        return len(self.get_variables())

    def get_state_names(self) -> List[str]:
        """Convenience: return list of variable names for debugging"""
        return [v.name for v in self.get_variables()]

    def __repr__(self) -> str:
        port_names = ', '.join(self.ports.keys())
        return f"{self.__class__.__name__}('{self.name}', ports=[{port_names}])"
