# Thermal System Simulator - MVP Specification

**Version:** 0.1.0  
**Date:** 2025-11-07  
**Target Timeline:** 4 weeks  
**Reference Document:** See `ARCHITECTURE.md` for complete design rationale  
**Authors:** Collaborative specification by GPT-4 (systems architect) and Claude (numerical physicist)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Core Interfaces](#core-interfaces)
5. [Component Library](#component-library)
6. [Property Interface](#property-interface)
7. [Target Usage Example](#target-usage-example)
8. [Implementation Details](#implementation-details)
9. [Testing Requirements](#testing-requirements)
10. [Success Criteria](#success-criteria)
11. [Explicit Non-Goals](#explicit-non-goals)
12. [Dependencies](#dependencies)
13. [Deliverables](#deliverables)

---

## Quick Start

**What we're building:** A minimal but extensible thermal system simulator that models single-phase fluid loops (Rankine cycles, simple heat engines) using a graph-based component architecture.

**Core capability:** User connects pre-built components (heaters, turbines, pumps) via typed ports. The system automatically assembles governing equations and solves using scipy's BDF solver.

**Success metric:** Simulate a closed Rankine cycle achieving 30-35% thermal efficiency, demonstrating correct thermodynamic behavior.

**Why it matters:** Proves the architecture (typed ports, graph assembly, component extensibility) before adding complexity (two-phase flow, controls, multi-rate integration).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   USER SCRIPT                           │
│  • Define components (boiler, turbine, etc.)           │
│  • Connect via typed ports                              │
│  • Call graph.solve()                                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              GRAPH KERNEL (we build)                    │
│  • ThermalGraph: topology management                    │
│  • Port: type-safe connections (mass vs heat)          │
│  • Assembly: converts graph → global residual function  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│          COMPONENT LIBRARY (we build)                   │
│  • Component base class with standard interface         │
│  • Implementations: Heater, Turbine, Pump, Pipe        │
│  • Each defines: state variables + residual equations   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│           SOLVER (scipy.integrate)                      │
│  • solve_ivp with method='BDF' (for stiff problems)    │
│  • Adaptive timestepping, error control                 │
│  • Future: swap to SUNDIALS IDA for full DAE support   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│        PROPERTIES (CoolProp wrapper)                    │
│  • Fluid property evaluation: h(P,T), T(P,h), etc.    │
│  • LRU caching for performance                          │
└─────────────────────────────────────────────────────────┘
```

**Key principle:** Each layer has a clean interface. User never sees scipy details. Components never see graph topology. Graph never implements physics.

---

## Directory Structure

```
thermal_sim/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── port.py              # Port type definitions
│   ├── variable.py          # Variable metadata
│   ├── component.py         # Component base class
│   └── graph.py             # ThermalGraph + assembly
├── components/
│   ├── __init__.py
│   ├── heater.py           # ConstantPressureHeater
│   ├── turbine.py          # Turbine
│   ├── pump.py             # Pump
│   └── pipe.py             # Simple pipe with pressure drop
├── properties/
│   ├── __init__.py
│   └── coolprop_wrapper.py # CoolProp interface with caching
└── utils/
    ├── __init__.py
    └── validators.py       # Conservation checks

tests/
├── unit/
│   ├── test_ports.py
│   ├── test_component.py
│   └── test_graph.py
├── integration/
│   └── test_rankine.py
└── fixtures/
    └── test_data.py

examples/
├── rankine_cycle.py        # Main demo
└── simple_loop.py          # Minimal 2-component test

docs/
└── README.md               # Installation and basic usage

requirements.txt
setup.py
pytest.ini
```

---

## Core Interfaces

### 1. Ports (`core/port.py`)

Ports enforce type safety and represent physical connections between components.

```python
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
class SignalPort(Port):
    """
    Represents a control signal or measurement (future use).
    
    Attributes:
        name: Unique identifier
        direction: 'in' for actuator, 'out' for sensor
        value: Signal value (units depend on context)
    """
    name: str
    direction: Literal['in', 'out']
    value: float = 0.0
    
    component: Optional[object] = field(default=None, repr=False)
    
    def compatible_with(self, other: Port) -> bool:
        """Signal ports connect to signal ports"""
        if not isinstance(other, SignalPort):
            return False
        return self.direction != other.direction
```

**Key design decisions:**
- Ports are **dataclasses** for simplicity and automatic `__eq__`, `__hash__`
- Direction checking prevents nonsensical connections (two outlets)
- Component reference allows tracing connections back to topology
- Sign conventions are explicit in docstrings

---

### 2. Variables (`core/variable.py`)

Metadata for state variables to enable introspection and debugging.

```python
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
```

---

### 3. Component Base Class (`core/component.py`)

All components inherit from this ABC and implement three key methods.

```python
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
                 t: float) -> np.ndarray:
        """
        Compute residual vector for this component's equations.
        
        The solver will drive this residual to zero.
        
        Args:
            state: This component's state variables (local slice of global state)
            ports: Dictionary mapping port names to Port objects (with current values)
            t: Current simulation time [s]
            
        Returns:
            1D numpy array of residuals (same length as state vector)
            
        Convention:
            Residual form: f(x) = 0
            
            For example, if the physics is:
                Q = mdot * (h_out - h_in)
            
            Write as residual:
                residual[0] = Q - mdot * (h_out - h_in)
                
            Solver will find state values that make residual[0] = 0.
        
        Side effects:
            Should update outlet port values based on computed state.
            Example: ports['outlet'].h = h_out
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
```

**Critical notes for implementers:**
1. **Order matters**: Variable order in `get_variables()` must match array indices in `residual()`
2. **Residual form**: Always write equations as `f(x) = 0`, not `x = ...`
3. **Update ports**: `residual()` should set outlet port values as side effect
4. **Units**: Everything in SI (Pa, kg/s, J/kg, K, W). No conversion needed.

---

### 4. Graph (`core/graph.py`)

The graph manages topology, validates connections, and assembles the global system.

```python
"""
ThermalGraph: topology manager and equation assembler.

The graph is responsible for:
  - Storing components and their connections
  - Validating type compatibility at connection time
  - Assembling component equations into global residual function
  - Invoking the solver
"""

from typing import List, Tuple, Callable, Dict, Optional
import numpy as np
from scipy.integrate import solve_ivp
import warnings

from thermal_sim.core.component import Component
from thermal_sim.core.port import Port, MassFlowPort


class ThermalGraph:
    """
    Manages system topology and orchestrates solution.
    
    Usage:
        graph = ThermalGraph()
        graph.add_component(boiler)
        graph.add_component(turbine)
        graph.connect(boiler.outlet, turbine.inlet)
        result = graph.solve(t_span=(0, 1000))
    """
    
    def __init__(self):
        self.components: List[Component] = []
        self.connections: List[Tuple[Port, Port]] = []
        self._component_offsets: Dict[str, int] = {}
        
    def add_component(self, component: Component) -> None:
        """
        Add a component to the system.
        
        Args:
            component: Component instance to add
            
        Raises:
            ValueError: If component with this name already exists
        """
        if component.name in [c.name for c in self.components]:
            raise ValueError(f"Component '{component.name}' already exists")
        
        # Set component reference in all ports
        for port in component.ports.values():
            port.component = component
            
        self.components.append(component)
        
    def connect(self, port_a: Port, port_b: Port) -> None:
        """
        Connect two ports.
        
        Enforces type safety and direction compatibility.
        Uses reference sharing: both components will reference the same Port object.
        
        Args:
            port_a: First port (typically an outlet)
            port_b: Second port (typically an inlet)
            
        Raises:
            TypeError: If ports are incompatible types or directions
            
        Implementation note:
            We use reference sharing rather than explicit synchronization.
            After connection, port_b is replaced by reference to port_a.
            This ensures conservation automatically: what leaves A enters B.
        """
        if not port_a.compatible_with(port_b):
            raise TypeError(
                f"Cannot connect {port_a} to {port_b}: "
                f"incompatible types or directions"
            )
        
        # Store connection for topology tracking
        self.connections.append((port_a, port_b))
        
        # Implement reference sharing:
        # Replace the inlet port with reference to outlet port
        if port_a.direction == 'out' and port_b.direction == 'in':
            # port_b's component should now reference port_a
            port_b.component.ports[port_b.name] = port_a
        elif port_a.direction == 'in' and port_b.direction == 'out':
            port_a.component.ports[port_a.name] = port_b
        else:
            warnings.warn("Unusual connection direction pairing")
            
    def _build_state_mapping(self) -> None:
        """
        Build mapping from component names to global state vector offsets.
        
        Called internally by assemble().
        """
        self._component_offsets = {}
        offset = 0
        
        for comp in self.components:
            self._component_offsets[comp.name] = offset
            offset += comp.get_state_size()
            
        self._total_state_size = offset
        
    def assemble(self) -> Tuple[Callable, np.ndarray]:
        """
        Assemble global residual function and initial state.
        
        Returns:
            (residual_func, y0) where:
                residual_func(t, y) -> np.ndarray: global residual vector
                y0: initial state vector for all components
                
        Note:
            This is called automatically by solve(), but can be called
            explicitly for debugging or custom solver setup.
        """
        if not self.components:
            raise ValueError("Cannot assemble empty graph")
            
        # Build state mapping
        self._build_state_mapping()
        
        # Construct initial state vector
        y0_parts = []
        for comp in self.components:
            y0_parts.append(comp.get_initial_state())
        y0 = np.concatenate(y0_parts)
        
        # Build residual function
        def residual_func(t: float, y: np.ndarray) -> np.ndarray:
            """
            Global residual function: F(t, y) = 0
            
            Args:
                t: Current time [s]
                y: Global state vector (concatenated component states)
                
            Returns:
                Global residual vector (solver drives this to zero)
            """
            residuals = []
            
            for comp in self.components:
                # Extract this component's state slice
                offset = self._component_offsets[comp.name]
                size = comp.get_state_size()
                local_state = y[offset:offset+size]
                
                # Compute component residual
                # (ports are already shared via reference, so no synchronization needed)
                res = comp.residual(local_state, comp.ports, t)
                
                if res.shape != (size,):
                    raise ValueError(
                        f"Component {comp.name} returned residual of shape {res.shape}, "
                        f"expected ({size},)"
                    )
                
                residuals.append(res)
            
            return np.concatenate(residuals)
        
        return residual_func, y0
    
    def solve(self, 
              t_span: Tuple[float, float],
              method: str = 'BDF',
              **solver_options) -> object:
        """
        Solve the system over a time interval.
        
        Args:
            t_span: (t_start, t_end) in seconds
            method: ODE solver method (default 'BDF' for stiff problems)
            **solver_options: Additional arguments passed to scipy.integrate.solve_ivp
                Common options:
                    rtol: Relative tolerance (default 1e-3)
                    atol: Absolute tolerance (default 1e-6)
                    max_step: Maximum timestep (default inf)
                    
        Returns:
            Result object from scipy.integrate.solve_ivp with attributes:
                .t: Time points
                .y: State values (shape: [n_states, n_timepoints])
                .success: Whether solver succeeded
                .message: Solver status message
                
        Solver notes:
            - Uses scipy's BDF (Backward Differentiation Formula) by default
            - BDF is appropriate for stiff problems (typical in thermal systems)
            - Future versions will support SUNDIALS IDA for full DAE capability
            - For now, system must be in ODE form (algebraic variables eliminated)
        """
        residual_func, y0 = self.assemble()
        
        # Wrap residual as ODE: dy/dt = F(t, y)
        # (For true DAE support, we'll switch to IDA later)
        def ode_func(t, y):
            return residual_func(t, y)
        
        result = solve_ivp(
            ode_func,
            t_span=t_span,
            y0=y0,
            method=method,
            dense_output=True,
            **solver_options
        )
        
        if not result.success:
            warnings.warn(f"Solver failed: {result.message}")
            
        # Attach metadata for post-processing
        result.component_names = [c.name for c in self.components]
        result.component_offsets = self._component_offsets.copy()
        result.state_names = {}
        for comp in self.components:
            offset = self._component_offsets[comp.name]
            result.state_names[comp.name] = comp.get_state_names()
        
        return result
    
    def get_component_state(self, result: object, component_name: str) -> np.ndarray:
        """
        Extract one component's state trajectory from solution.
        
        Args:
            result: Solution object from solve()
            component_name: Name of component to extract
            
        Returns:
            State array of shape [n_states, n_timepoints] for this component
        """
        offset = self._component_offsets[component_name]
        comp = next(c for c in self.components if c.name == component_name)
        size = comp.get_state_size()
        
        return result.y[offset:offset+size, :]
    
    def validate_topology(self) -> List[str]:
        """
        Check for common topology errors.
        
        Returns:
            List of warning messages (empty if no issues)
            
        Checks:
            - Disconnected components
            - Unconnected ports
            - Mass flow conservation
        """
        warnings = []
        
        # Check for unconnected ports
        for comp in self.components:
            for port_name, port in comp.ports.items():
                is_connected = any(
                    port is conn_port 
                    for conn in self.connections 
                    for conn_port in conn
                )
                if not is_connected:
                    warnings.append(
                        f"Component '{comp.name}' has unconnected port '{port_name}'"
                    )
        
        return warnings
```

**Implementation notes:**

1. **Port sharing mechanism**: When connecting `A.outlet` to `B.inlet`, we make `B.inlet` reference the same object as `A.outlet`. This ensures automatic conservation - whatever `A` writes to the outlet, `B` immediately sees at its inlet.

2. **State vector layout**: Global state is `[comp1_states, comp2_states, ...]`. The `_component_offsets` dict tracks where each component's slice begins.

3. **Residual assembly**: Loop through components, extract local state slice, call component's `residual()`, concatenate results.

4. **ODE vs DAE**: Currently wrapped for scipy's ODE solver. When we upgrade to SUNDIALS IDA, we'll modify this to pass residuals directly without the `dy/dt =` wrapper.

---

## Component Library

Implement these four components to complete the MVP.

### Component 1: `ConstantPressureHeater` (`components/heater.py`)

```python
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
        
        # Create ports
        self.inlet = MassFlowPort(f"{name}_inlet", direction='in')
        self.outlet = MassFlowPort(f"{name}_outlet", direction='out')
        
        self.ports = {
            'inlet': self.inlet,
            'outlet': self.outlet,
        }
        
    def get_variables(self):
        return [
            Variable('h_out', kind='algebraic', initial=1e6, units='J/kg'),
            Variable('mdot', kind='algebraic', initial=100.0, units='kg/s'),
        ]
    
    def get_initial_state(self):
        # Reasonable defaults for water
        h0 = 1e6  # ~239°C saturated liquid at 10 MPa
        mdot0 = 100.0  # kg/s
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
```

---

### Component 2: `Turbine` (`components/turbine.py`)

```python
"""
Turbine: converts fluid enthalpy to shaft work via expansion.
"""

import numpy as np
from thermal_sim.core.component import Component
from thermal_sim.core.variable import Variable
from thermal_sim.core.port import MassFlowPort
from thermal_sim.properties.coolprop_wrapper import FluidProperties


class Turbine(Component):
    """
    Expands high-pressure fluid to produce shaft work.
    
    Governing equations:
        1. h_out = h_in - η * (h_in - h_out_isentropic)
        2. W_shaft = mdot * (h_in - h_out)
        3. P_out = P_set
    
    State variables:
        - h_out: outlet enthalpy [J/kg]
        - W_shaft: shaft power output [W]
        - mdot: mass flow rate [kg/s]
    
    Parameters:
        efficiency: Isentropic efficiency (0 < η < 1)
        P_out: Outlet pressure [Pa]
        fluid: Fluid name
    """
    
    def __init__(self, name: str, efficiency: float, P_out: float, fluid: str = 'Water'):
        super().__init__(name)
        
        if not 0 < efficiency <= 1:
            raise ValueError(f"Efficiency must be in (0,1], got {efficiency}")
        
        self.eta = efficiency
        self.P_out = P_out
        self.fluid_name = fluid
        self.fluid = FluidProperties(fluid)
        
        self.inlet = MassFlowPort(f"{name}_inlet", direction='in')
        self.outlet = MassFlowPort(f"{name}_outlet", direction='out')
        
        self.ports = {
            'inlet': self.inlet,
            'outlet': self.outlet,
        }
        
    def get_variables(self):
        return [
            Variable('h_out', kind='algebraic', initial=2e6, units='J/kg'),
            Variable('W_shaft', kind='algebraic', initial=1e6, units='W'),
            Variable('mdot', kind='algebraic', initial=100.0, units='kg/s'),
        ]
    
    def get_initial_state(self):
        return np.array([2e6, 1e6, 100.0])
    
    def residual(self, state, ports, t):
        h_out, W_shaft, mdot = state
        
        # Inlet conditions
        h_in = ports['inlet'].h
        P_in = ports['inlet'].P
        
        # Compute isentropic outlet enthalpy
        s_in = self.fluid.entropy(P_in, h_in)
        h_out_s = self.fluid.enthalpy_from_ps(self.P_out, s_in)
        
        # Equation 1: Isentropic efficiency relation
        # h_out = h_in - η * (h_in - h_out_s)
        eq_efficiency = h_out - (h_in - self.eta * (h_in - h_out_s))
        
        # Equation 2: Power extraction
        # W = mdot * (h_in - h_out)
        eq_power = W_shaft - mdot * (h_in - h_out)
        
        # Equation 3: Outlet pressure
        eq_pressure = ports['outlet'].P - self.P_out
        
        # Update outlet
        ports['outlet'].mdot = mdot
        ports['outlet'].h = h_out
        ports['outlet'].P = self.P_out
        
        return np.array([eq_efficiency, eq_power, eq_pressure])
```

---

### Component 3: `Pump` (`components/pump.py`)

```python
"""
Pump: increases fluid pressure using shaft work.
"""

import numpy as np
from thermal_sim.core.component import Component
from thermal_sim.core.variable import Variable
from thermal_sim.core.port import MassFlowPort
from thermal_sim.properties.coolprop_wrapper import FluidProperties


class Pump(Component):
    """
    Increases pressure of liquid using shaft power.
    
    Governing equations:
        1. h_out = h_in + (h_out_isentropic - h_in) / η
        2. W_shaft = mdot * (h_out - h_in)
        3. P_out = P_set
    
    State variables:
        - h_out: outlet enthalpy [J/kg]
        - W_shaft: shaft power input [W]
        - mdot: mass flow rate [kg/s]
    
    Parameters:
        efficiency: Isentropic efficiency
        P_out: Outlet pressure [Pa]
        fluid: Fluid name
    """
    
    def __init__(self, name: str, efficiency: float, P_out: float, fluid: str = 'Water'):
        super().__init__(name)
        
        if not 0 < efficiency <= 1:
            raise ValueError(f"Efficiency must be in (0,1], got {efficiency}")
        
        self.eta = efficiency
        self.P_out = P_out
        self.fluid_name = fluid
        self.fluid = FluidProperties(fluid)
        
        self.inlet = MassFlowPort(f"{name}_inlet", direction='in')
        self.outlet = MassFlowPort(f"{name}_outlet", direction='out')
        
        self.ports = {
            'inlet': self.inlet,
            'outlet': self.outlet,
        }
        
    def get_variables(self):
        return [
            Variable('h_out', kind='algebraic', initial=1e5, units='J/kg'),
            Variable('W_shaft', kind='algebraic', initial=1e5, units='W'),
            Variable('mdot', kind='algebraic', initial=100.0, units='kg/s'),
        ]
    
    def get_initial_state(self):
        return np.array([1e5, 1e5, 100.0])
    
    def residual(self, state, ports, t):
        h_out, W_shaft, mdot = state
        
        h_in = ports['inlet'].h
        P_in = ports['inlet'].P
        
        # Isentropic outlet enthalpy
        s_in = self.fluid.entropy(P_in, h_in)
        h_out_s = self.fluid.enthalpy_from_ps(self.P_out, s_in)
        
        # Equation 1: Efficiency relation (pump uses more work than isentropic)
        # h_out = h_in + (h_out_s - h_in) / η
        eq_efficiency = h_out - (h_in + (h_out_s - h_in) / self.eta)
        
        # Equation 2: Power consumption
        eq_power = W_shaft - mdot * (h_out - h_in)
        
        # Equation 3: Outlet pressure
        eq_pressure = ports['outlet'].P - self.P_out
        
        # Update outlet
        ports['outlet'].mdot = mdot
        ports['outlet'].h = h_out
        ports['outlet'].P = self.P_out
        
        return np.array([eq_efficiency, eq_power, eq_pressure])
```

---

### Component 4: `Pipe` (`components/pipe.py`)

```python
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
        
        self.inlet = MassFlowPort(f"{name}_inlet", direction='in')
        self.outlet = MassFlowPort(f"{name}_outlet", direction='out')
        
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
    
    def residual(self, state, ports, t):
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
```

---

## Property Interface

### CoolProp Wrapper (`properties/coolprop_wrapper.py`)

```python
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
```

---

## Target Usage Example

This is what users should be able to write with the MVP.

### Example: `examples/rankine_cycle.py`

```python
"""
Simple Rankine cycle demonstration.

This example creates a basic steam power cycle:
    Boiler → Turbine → Condenser → Pump → (back to Boiler)

Expected result: ~30-35% thermal efficiency
"""

from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.components.pump import Pump
import numpy as np


def main():
    # Create system graph
    graph = ThermalGraph()
    
    # Define operating conditions
    P_high = 10e6   # 10 MPa (high pressure)
    P_low = 10e3    # 10 kPa (condenser pressure)
    
    # Create components
    boiler = ConstantPressureHeater(
        name='boiler',
        P=P_high,
        Q=100e6,     # 100 MW heat input
        fluid='Water'
    )
    
    turbine = Turbine(
        name='turbine',
        efficiency=0.85,
        P_out=P_low,
        fluid='Water'
    )
    
    condenser = ConstantPressureHeater(
        name='condenser',
        P=P_low,
        Q=-80e6,     # 80 MW heat rejection (negative = cooling)
        fluid='Water'
    )
    
    pump = Pump(
        name='pump',
        efficiency=0.80,
        P_out=P_high,
        fluid='Water'
    )
    
    # Add components to graph
    for comp in [boiler, turbine, condenser, pump]:
        graph.add_component(comp)
    
    # Connect in cycle
    graph.connect(boiler.outlet, turbine.inlet)
    graph.connect(turbine.outlet, condenser.inlet)
    graph.connect(condenser.outlet, pump.inlet)
    graph.connect(pump.outlet, boiler.inlet)
    
    # Check topology
    warnings = graph.validate_topology()
    if warnings:
        print("Topology warnings:")
        for w in warnings:
            print(f"  {w}")
    
    # Solve
    print("Solving Rankine cycle...")
    result = graph.solve(
        t_span=(0, 1000),  # Simulate 1000 seconds
        rtol=1e-6,
        atol=1e-8
    )
    
    if result.success:
        print(f"✓ Solution converged in {len(result.t)} time steps")
        
        # Extract results
        turbine_state = graph.get_component_state(result, 'turbine')
        pump_state = graph.get_component_state(result, 'pump')
        
        # Turbine state: [h_out, W_shaft, mdot]
        W_turbine = turbine_state[1, -1]  # Final turbine power
        W_pump = pump_state[1, -1]        # Final pump power
        
        Q_in = boiler.Q
        W_net = W_turbine - W_pump
        
        efficiency = W_net / Q_in
        
        print(f"\nResults:")
        print(f"  Turbine power: {W_turbine/1e6:.2f} MW")
        print(f"  Pump power: {W_pump/1e6:.2f} MW")
        print(f"  Net power: {W_net/1e6:.2f} MW")
        print(f"  Thermal efficiency: {efficiency*100:.1f}%")
        
        # Validate
        if 0.30 < efficiency < 0.40:
            print("✓ Efficiency in expected range (30-40%)")
        else:
            print(f"⚠ Efficiency {efficiency*100:.1f}% outside expected range")
    else:
        print(f"✗ Solver failed: {result.message}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
```

---

## Implementation Details

### Port Connection Algorithm

The key insight is **reference sharing** rather than explicit synchronization:

```python
# In ThermalGraph.connect():
def connect(self, port_a: Port, port_b: Port):
    # Validate compatibility
    if not port_a.compatible_with(port_b):
        raise TypeError(...)
    
    # Store connection for topology tracking
    self.connections.append((port_a, port_b))
    
    # Implement reference sharing:
    # Make inlet reference the same object as outlet
    if port_a.direction == 'out' and port_b.direction == 'in':
        # port_b's component now references port_a
        port_b.component.ports[port_b.name] = port_a
    elif port_a.direction == 'in' and port_b.direction == 'out':
        # Reverse case
        port_a.component.ports[port_a.name] = port_b
```

**Why this works:**
- When component A updates `outlet.h = 1e6`, it writes to the Port object
- Component B's `inlet` references the same Port object
- When B reads `inlet.h`, it gets the value A wrote
- Conservation is automatic: no separate synchronization step needed

**Tradeoff:**
- ✅ Simple, no synchronization loop
- ✅ Guarantees conservation (physically impossible to have different values)
- ❌ Makes debugging slightly harder (references can be confusing)
- ❌ Doesn't support multiple components reading from one port (star topology)

For MVP, the benefits outweigh the limitations. Star topologies can be added in Phase 2 via explicit junction components.

---

### Assembly Algorithm Pseudocode

```python
def assemble(self):
    # Step 1: Build mapping from component names to state vector indices
    offsets = {}
    offset = 0
    for comp in components:
        offsets[comp.name] = offset
        offset += comp.get_state_size()
    
    # Step 2: Concatenate initial states
    y0 = concatenate([comp.get_initial_state() for comp in components])
    
    # Step 3: Build global residual function
    def residual_func(t, y):
        residuals = []
        for comp in components:
            # Extract this component's slice
            start = offsets[comp.name]
            end = start + comp.get_state_size()
            local_state = y[start:end]
            
            # Ports already synchronized via reference sharing
            # Just call component residual
            res = comp.residual(local_state, comp.ports, t)
            residuals.append(res)
        
        return concatenate(residuals)
    
    return residual_func, y0
```

**Key points:**
- State vector layout is deterministic: order in `components` list
- No explicit port synchronization needed (done via references)
- Each component sees only its own state slice
- Global residual is concatenation of local residuals

---

### Solver Integration Notes

**Current approach (MVP):**
```python
from scipy.integrate import solve_ivp

result = solve_ivp(
    residual_func,  # Our assembled function
    t_span=(0, 1000),
    y0=initial_state,
    method='BDF',   # Backward Differentiation Formula
    rtol=1e-6,      # Relative tolerance
    atol=1e-8       # Absolute tolerance
)
```

**BDF (Backward Differentiation Formula):**
- Implicit method good for stiff problems
- Variable order (up to 5)
- Adaptive timestepping
- Suitable for most thermal systems

**Future upgrade path (Phase 2):**
```python
from scikits.odes import dae

# For true DAE support:
solver = dae('ida', residual_func, ...)
result = solver.solve(t_values, y0, yp0)
```

**IDA (SUNDIALS):**
- True DAE solver (handles algebraic constraints natively)
- More robust for systems with conservation constraints
- Requires algebraic variables to be marked explicitly
- Supports sparse Jacobians

**Code should be structured to make this swap easy:**
- Keep solver invocation in one place (`ThermalGraph.solve()`)
- Don't let solver details leak into component code
- Use abstract interface (could add `SolverBackend` ABC later)

---

## Testing Requirements

### Unit Tests

Use `pytest` for all testing. Aim for >90% coverage of core modules.

#### `tests/unit/test_ports.py`

```python
"""Unit tests for port types and compatibility checking."""

import pytest
from thermal_sim.core.port import MassFlowPort, HeatPort


def test_mass_port_creation():
    """Test basic MassFlowPort instantiation"""
    port = MassFlowPort('test', 'in')
    assert port.name == 'test'
    assert port.direction == 'in'
    assert port.mdot == 0.0
    assert port.h == 0.0
    assert port.P == 0.0


def test_mass_port_compatibility():
    """Inlet and outlet should be compatible"""
    inlet = MassFlowPort('in', 'in')
    outlet = MassFlowPort('out', 'out')
    
    assert inlet.compatible_with(outlet)
    assert outlet.compatible_with(inlet)


def test_mass_port_incompatibility_same_direction():
    """Two inlets or two outlets should not connect"""
    inlet1 = MassFlowPort('in1', 'in')
    inlet2 = MassFlowPort('in2', 'in')
    
    assert not inlet1.compatible_with(inlet2)


def test_mass_heat_port_incompatibility():
    """Mass and heat ports should not connect"""
    mass_port = MassFlowPort('m', 'in')
    heat_port = HeatPort('h', 'in')
    
    assert not mass_port.compatible_with(heat_port)


def test_heat_port_compatibility():
    """Heat ports should connect with opposite directions"""
    heat_in = HeatPort('in', 'in')
    heat_out = HeatPort('out', 'out')
    
    assert heat_in.compatible_with(heat_out)
```

#### `tests/unit/test_component.py`

```python
"""Unit tests for component base class and implementations."""

import pytest
import numpy as np
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.core.port import MassFlowPort


def test_heater_creation():
    """Test heater component instantiation"""
    heater = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    
    assert heater.name == 'h1'
    assert heater.P == 1e6
    assert heater.Q == 1e6
    assert 'inlet' in heater.ports
    assert 'outlet' in heater.ports


def test_heater_state_size():
    """Heater should have 2 state variables"""
    heater = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    assert heater.get_state_size() == 2


def test_heater_initial_state():
    """Initial state should be reasonable array"""
    heater = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    y0 = heater.get_initial_state()
    
    assert isinstance(y0, np.ndarray)
    assert y0.shape == (2,)
    assert np.all(np.isfinite(y0))


def test_heater_residual_shape():
    """Residual should return correct shape"""
    heater = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    
    state = np.array([1e6, 100.0])
    ports = {
        'inlet': MassFlowPort('in', 'in'),
        'outlet': MassFlowPort('out', 'out')
    }
    ports['inlet'].h = 5e5
    ports['inlet'].P = 1e6
    
    res = heater.residual(state, ports, t=0)
    
    assert isinstance(res, np.ndarray)
    assert res.shape == (2,)


def test_turbine_efficiency_validation():
    """Turbine should reject invalid efficiencies"""
    with pytest.raises(ValueError):
        Turbine('t1', efficiency=1.5, P_out=1e5)
    
    with pytest.raises(ValueError):
        Turbine('t1', efficiency=0.0, P_out=1e5)
```

#### `tests/unit/test_graph.py`

```python
"""Unit tests for graph topology and assembly."""

import pytest
from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.pump import Pump


def test_graph_creation():
    """Test empty graph instantiation"""
    graph = ThermalGraph()
    assert len(graph.components) == 0
    assert len(graph.connections) == 0


def test_add_component():
    """Test adding components"""
    graph = ThermalGraph()
    heater = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    
    graph.add_component(heater)
    assert len(graph.components) == 1
    assert graph.components[0] is heater


def test_duplicate_component_name():
    """Should not allow duplicate names"""
    graph = ThermalGraph()
    h1 = ConstantPressureHeater('heater', P=1e6, Q=1e6)
    h2 = ConstantPressureHeater('heater', P=2e6, Q=2e6)
    
    graph.add_component(h1)
    with pytest.raises(ValueError, match="already exists"):
        graph.add_component(h2)


def test_connect_compatible_ports():
    """Should allow connecting compatible ports"""
    graph = ThermalGraph()
    h1 = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    p1 = Pump('p1', efficiency=0.8, P_out=2e6)
    
    graph.add_component(h1)
    graph.add_component(p1)
    
    # Should not raise
    graph.connect(h1.outlet, p1.inlet)
    assert len(graph.connections) == 1


def test_connect_incompatible_ports():
    """Should reject incompatible port connections"""
    graph = ThermalGraph()
    h1 = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    h2 = ConstantPressureHeater('h2', P=1e6, Q=1e6)
    
    graph.add_component(h1)
    graph.add_component(h2)
    
    # Two outlets should not connect
    with pytest.raises(TypeError):
        graph.connect(h1.outlet, h2.outlet)


def test_simple_loop_assembly():
    """Test that a simple 2-component loop assembles"""
    graph = ThermalGraph()
    h1 = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    p1 = Pump('p1', efficiency=0.8, P_out=2e6)
    
    graph.add_component(h1)
    graph.add_component(p1)
    graph.connect(h1.outlet, p1.inlet)
    graph.connect(p1.outlet, h1.inlet)
    
    residual_func, y0 = graph.assemble()
    
    assert callable(residual_func)
    assert len(y0) == 5  # heater(2) + pump(3)
    
    # Test that residual is callable
    res = residual_func(0, y0)
    assert len(res) == 5
```

---

### Integration Tests

#### `tests/integration/test_rankine.py`

```python
"""Integration test for full Rankine cycle."""

import pytest
import numpy as np
from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.components.pump import Pump


def build_rankine_cycle():
    """Helper to construct standard Rankine cycle"""
    graph = ThermalGraph()
    
    P_high = 10e6
    P_low = 10e3
    
    boiler = ConstantPressureHeater('boiler', P=P_high, Q=100e6)
    turbine = Turbine('turbine', efficiency=0.85, P_out=P_low)
    condenser = ConstantPressureHeater('condenser', P=P_low, Q=-80e6)
    pump = Pump('pump', efficiency=0.80, P_out=P_high)
    
    for comp in [boiler, turbine, condenser, pump]:
        graph.add_component(comp)
    
    graph.connect(boiler.outlet, turbine.inlet)
    graph.connect(turbine.outlet, condenser.inlet)
    graph.connect(condenser.outlet, pump.inlet)
    graph.connect(pump.outlet, boiler.inlet)
    
    return graph


def test_rankine_cycle_converges():
    """Rankine cycle should reach steady state"""
    graph = build_rankine_cycle()
    
    result = graph.solve(t_span=(0, 1000), rtol=1e-6, atol=1e-8)
    
    assert result.success, f"Solver failed: {result.message}"
    assert len(result.t) > 10, "Should take multiple time steps"


def test_rankine_cycle_efficiency():
    """Thermal efficiency should be in expected range"""
    graph = build_rankine_cycle()
    
    result = graph.solve(t_span=(0, 1000))
    assert result.success
    
    # Extract final power values
    turbine_state = graph.get_component_state(result, 'turbine')
    pump_state = graph.get_component_state(result, 'pump')
    
    W_turbine = turbine_state[1, -1]  # turbine power
    W_pump = pump_state[1, -1]        # pump power
    
    Q_in = 100e6  # from boiler
    W_net = W_turbine - W_pump
    efficiency = W_net / Q_in
    
    # Rankine cycle at these conditions should be 30-40% efficient
    assert 0.25 < efficiency < 0.42, \
        f"Efficiency {efficiency:.1%} outside expected range (25-42%)"


def test_rankine_energy_balance():
    """Energy should be conserved: Q_in = W_net + Q_out"""
    graph = build_rankine_cycle()
    
    result = graph.solve(t_span=(0, 1000))
    assert result.success
    
    turbine_state = graph.get_component_state(result, 'turbine')
    pump_state = graph.get_component_state(result, 'pump')
    
    W_turbine = turbine_state[1, -1]
    W_pump = pump_state[1, -1]
    
    Q_in = 100e6
    Q_out = 80e6
    W_net = W_turbine - W_pump
    
    energy_balance = Q_in - W_net - Q_out
    
    # Should be close to zero (within 1%)
    assert abs(energy_balance) / Q_in < 0.01, \
        f"Energy imbalance: {energy_balance/1e6:.2f} MW"


def test_rankine_mass_conservation():
    """Mass flow should be conserved around the loop"""
    graph = build_rankine_cycle()
    
    result = graph.solve(t_span=(0, 1000))
    assert result.success
    
    # Extract mass flows from each component
    boiler_state = graph.get_component_state(result, 'boiler')
    turbine_state = graph.get_component_state(result, 'turbine')
    
    mdot_boiler = boiler_state[1, -1]
    mdot_turbine = turbine_state[2, -1]
    
    # Should all be equal
    np.testing.assert_allclose(mdot_boiler, mdot_turbine, rtol=1e-4)
```

---

## Success Criteria

The MVP is considered complete when all of the following are satisfied:

### 1. Code Completeness
- [ ] All core modules implemented (`port`, `component`, `graph`)
- [ ] All four components implemented (`heater`, `turbine`, `pump`, `pipe`)
- [ ] Property wrapper functional with CoolProp
- [ ] Total codebase < 1500 lines (excluding tests)

### 2. Testing
- [ ] All unit tests pass (pytest)
- [ ] All integration tests pass
- [ ] Test coverage > 85% for core modules
- [ ] No critical warnings from pytest

### 3. Functional Validation
- [ ] Rankine cycle example runs to completion
- [ ] Solver converges without manual intervention
- [ ] Thermal efficiency in range 30-35%
- [ ] Energy balance error < 1%
- [ ] Mass conservation error < 0.1%

### 4. Extensibility
- [ ] Can add a new component by implementing 3 methods
- [ ] New component integrates without modifying core
- [ ] Example of adding custom component documented

### 5. Documentation
- [ ] README.md with installation instructions
- [ ] API docstrings for all public classes/methods
- [ ] Working example (rankine_cycle.py) with comments
- [ ] Brief technical report (2-3 pages) covering:
  - Architecture decisions
  - Performance characteristics
  - Known limitations
  - Next steps

---

## Explicit Non-Goals

The following features are **explicitly deferred** to future phases. Do not implement them in the MVP:

❌ **GUI or visualization**
- Just return numpy arrays and print to console
- Plotting can be done externally with matplotlib

❌ **Control systems**
- No PID controllers, no feedback loops
- Components have fixed parameters

❌ **Two-phase flow**
- Single-phase liquid or vapor only
- No boiling curves, no void fraction

❌ **Multi-rate integration**
- Single solver for entire system
- No sub-stepping or hierarchical time scales

❌ **Spatial discretization**
- All components are 0D (lumped parameter)
- No finite element/volume within components

❌ **Unit checking with pint**
- Use raw floats in SI units
- Document expected units in docstrings

❌ **Custom Jacobians**
- Let scipy use finite differences
- Analytic Jacobians are Phase 2

❌ **Advanced solvers**
- scipy's BDF is sufficient
- SUNDIALS IDA is Phase 2

❌ **Optimization or parameter estimation**
- No inverse problems
- Just forward simulation

❌ **Parallel execution**
- Single-threaded is fine
- GPU acceleration is future work

❌ **External interfaces**
- No FMI, no co-simulation
- No WebSocket, no live dashboards

---

## Dependencies

### `requirements.txt`

```
# Core numerical libraries
numpy>=1.24.0
scipy>=1.10.0

# Thermodynamic properties
CoolProp>=6.4.0

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Optional: for better error messages
colorama>=0.4.0
```

### `setup.py` (if packaging)

```python
from setuptools import setup, find_packages

setup(
    name='thermal_sim',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'CoolProp>=6.4.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ]
    },
    python_requires='>=3.9',
)
```

---

## Deliverables

### 1. Code (`thermal_sim/` package)
- Organized per directory structure above
- All modules have docstrings
- Code follows PEP 8 style

### 2. Tests (`tests/`)
- Unit tests for all core modules
- Integration test for Rankine cycle
- All tests pass with `pytest`

### 3. Working Example (`examples/rankine_cycle.py`)
- Demonstrates complete workflow
- Includes comments explaining each step
- Runs successfully and prints results

### 4. README (`README.md`)

Should include:
- **Installation**: How to install dependencies
- **Quick start**: Minimal working example
- **Running tests**: `pytest` command
- **Project structure**: Brief overview of modules
- **License**: MIT or similar

### 5. Technical Report

Brief document (2-3 pages) covering:

**Architecture Section:**
- Three-layer design (graph, components, solver)
- Typed-port system for safety
- Reference-sharing for port synchronization

**Performance Section:**
- Benchmark results (wall time for 1000s simulation)
- Solver statistics (steps, rejections)
- Bottleneck analysis (where time is spent)

**Validation Section:**
- Rankine cycle efficiency compared to textbook values
- Energy/mass balance closure
- Convergence behavior

**Limitations and Future Work:**
- What MVP cannot do (two-phase, controls, etc.)
- Path to Phase 2 (SUNDIALS, multi-rate, two-phase)
- Known bugs or edge cases

---

## Questions and Clarifications

If anything in this specification is ambiguous or unclear, please:

1. **Check ARCHITECTURE.md** - Full design rationale is there
2. **Ask specific questions** - Don't guess on implementation details
3. **Provide alternatives** - If you see a better way, explain the tradeoff

**Common questions anticipated:**

**Q: Should I implement all components at once or incrementally?**
A: Incremental. Start with `ConstantPressureHeater` and `Pipe`, test simple 2-component loop. Then add `Turbine` and `Pump`.

**Q: What if CoolProp doesn't have a property I need?**
A: For MVP, use only (P,T) and (P,h) lookups. These always work for single-phase fluids.

**Q: How do I handle units internally?**
A: Everything is SI. No conversion. Document expected units in docstrings and examples.

**Q: Should I optimize for speed?**
A: Not yet. Correctness first. Profile later to find bottlenecks.

**Q: What if the solver doesn't converge?**
A: First, check initial conditions. Then try tighter tolerances or smaller `max_step`. Document the issue if it persists.

---

## Final Notes

**Philosophy**: This MVP is about proving the architecture, not building a production tool. Prioritize:
1. ✅ Clean interfaces
2. ✅ Correct physics
3. ✅ Extensibility
4. ⏸️ Performance (measure, don't optimize)

**Timeline**: 4 weeks is aggressive but achievable:
- Week 1: Core abstractions + simple 2-component test
- Week 2: All four components
- Week 3: Rankine cycle working
- Week 4: Tests, docs, report

**Success looks like**: Someone can clone the repo, run `examples/rankine_cycle.py`, see thermal efficiency printed, and understand how to add a new component by reading `components/heater.py`.

Good luck! 🚀

---

**Document Version**: 0.1.0  
**Last Updated**: 2025-11-07  
**Next Review**: After MVP completion
