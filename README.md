# Thermal System Simulator

An extensible, physics-based framework for simulating coupled thermal-hydraulic systems with transient dynamics and control.

## Overview

The Thermal System Simulator is a Python-based tool for modeling fluid loops, power cycles (Rankine, Brayton), and thermal management systems with two-phase flow, dynamics, and feedback control. It uses a graph-based architecture where components are connected via typed ports, and the framework automatically assembles and solves the governing differential-algebraic equations (DAEs).

**Current Status**: Phase 3 Complete - Full DAE solver with two-phase flow, dynamics, and control systems

### Solver Capabilities

- ✅ **`solve_steady_state()`**: Multi-backend steady-state solver
  - **Backends**: scipy (default), diffeqpy (Julia NonlinearSolve.jl), sequential
  - **Features**: Automatic sequential initialization, graceful fallback chain
  - **Status**: Production ready for complex systems

- ✅ **`solve_transient()`**: Transient DAE solver (Phase 3)
  - **Method**: BDF (Backward Differentiation Formula) via scipy
  - **Capabilities**: Mixed differential-algebraic systems
  - **Features**: Automatic algebraic constraint solving at each time step
  - **Performance**: >3000× real-time for typical systems
  - **Status**: Production ready for dynamics and control

- ✅ **`solve_sequential()`**: Sequential propagation solver
  - **Purpose**: Initialization helper and fallback for simultaneous solvers
  - **Use case**: Generates thermodynamically consistent initial guesses

**Key Features**:
- **Two-Phase Flow**: Quality tracking, phase detection, boiling/condensation
- **Dynamics & Control**: Differential equations, PID controllers, control valves
- **Transient Simulation**: Full DAE solver for time-varying systems
- **Typed Port System**: Type-safe connections (MassFlowPort, HeatPort, ScalarPort)
- **Automatic Conservation**: Reference-sharing mechanism enforces conservation laws
- **High Performance**: >3000× real-time simulation speed
- **Extensible Components**: Add new components by implementing 3 simple methods
- **Industry-Standard Properties**: CoolProp integration with LRU caching

## Installation

### Requirements
- Python 3.9 or higher
- numpy, scipy, CoolProp

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd thermalsim

# Install in development mode
pip install -e .

# Or install with test dependencies
pip install -e ".[dev]"
```

### Verify installation

```bash
python -c "from thermal_sim.core import ThermalGraph; print('Success!')"
```

## Quick Start

Here's a minimal example simulating a Rankine cycle:

```python
from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.components.pump import Pump

# Create system graph
graph = ThermalGraph()

# Define components
boiler = ConstantPressureHeater(name='boiler', P=10e6, Q=100e6)
turbine = Turbine(name='turbine', efficiency=0.85, P_out=10e3)
condenser = ConstantPressureHeater(name='condenser', P=10e3, Q=-80e6)
pump = Pump(name='pump', efficiency=0.80, P_out=10e6)

# Add to graph
for comp in [boiler, turbine, condenser, pump]:
    graph.add_component(comp)

# Connect in cycle
graph.connect(boiler.outlet, turbine.inlet)
graph.connect(turbine.outlet, condenser.inlet)
graph.connect(condenser.outlet, pump.inlet)
graph.connect(pump.outlet, boiler.inlet)

# Solve for steady state (multi-backend with automatic initialization)
result = graph.solve_steady_state(backend='scipy')

# Extract results
turbine_state = graph.get_component_state(result, 'turbine')
W_turbine = turbine_state[1, 0]  # Turbine power output
print(f"Turbine power: {W_turbine/1e6:.2f} MW")
print(f"Solver used: {result.solver_used}")
```

## Running Examples

The `examples/` directory contains working demonstrations:

### Steady-State Examples
```bash
# Rankine cycle steady state
python examples/rankine_cycle.py
```

### Transient Dynamics Examples (Phase 3)
```bash
# Tank filling with level dynamics
python examples/tank_filling.py

# Controlled tank level system
python examples/controlled_tank_simple.py

# Full Rankine cycle with level control (7 components)
python examples/rankine_with_level_control.py
```

Expected output from `tank_filling.py`:
```
TANK FILLING SIMULATION
Tank fills from 1.00 m → 1.40 m over 200 seconds
✓ Level change rate matches theory (within 5%)
✓ System reaches steady state
Performance: 4785× real-time
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=thermal_sim --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

## Project Structure

```
thermal_sim/
├── core/               # Core abstractions
│   ├── port.py        # Typed ports (MassFlowPort, HeatPort, ScalarPort)
│   ├── variable.py    # State variable metadata (algebraic/differential)
│   ├── component.py   # Component base class with DAE support
│   └── graph.py       # ThermalGraph - topology manager & DAE solver
├── components/         # Component library
│   ├── heater.py      # ConstantPressureHeater (single-phase)
│   ├── two_phase_heater.py  # TwoPhaseHeater (boiling/condensation)
│   ├── turbine.py     # Turbine (isentropic expansion)
│   ├── pump.py        # Pump (isentropic compression)
│   ├── pipe.py        # Simple pipe with pressure drop
│   ├── tank.py        # SimpleTank (level dynamics)
│   ├── pid_controller.py    # PID controller with anti-windup
│   └── control_valve.py     # Control valve with lag dynamics
└── properties/         # Thermophysical properties
    └── coolprop_wrapper.py  # CoolProp interface with two-phase support

tests/
├── unit/              # 112 unit tests
└── integration/       # 32 integration tests
                       # Total: 144 tests, all passing

examples/
├── rankine_cycle.py           # Steady-state Rankine cycle
├── tank_filling.py            # Transient tank dynamics
├── controlled_tank_simple.py  # Closed-loop level control
└── rankine_with_level_control.py  # Full integrated system
```

## Architecture

The simulator uses a layered architecture:

1. **User Script**: Define components and connections
2. **Graph Kernel**: Manages topology, validates connections, assembles equations
3. **Component Library**: Physics models (each defines state variables + residual equations)
4. **Solver Backend**: Multi-backend strategy
   - **Primary**: scipy.optimize.root (fast simultaneous solver)
   - **Advanced**: diffeqpy/NonlinearSolve.jl (robust for stiff systems)
   - **Initialization**: Sequential propagation for good initial guesses
   - **Fallback**: Automatic degradation chain ensures solution
5. **Properties**: CoolProp for fluid properties

Key design principles:
- **Reference Sharing**: Connected ports share the same object → automatic conservation
- **Residual Form**: All equations written as f(x) = 0 for solver
- **SI Units Only**: No conversion overhead (Pa, K, J/kg, kg/s, W)
- **Robust Initialization**: Sequential solver provides basin-of-attraction for simultaneous methods

## Adding a New Component

To add a custom component, inherit from `Component` and implement 3 methods:

```python
from thermal_sim.core.component import Component
from thermal_sim.core.variable import Variable
import numpy as np

class MyComponent(Component):
    def get_variables(self):
        """Declare state variables"""
        return [
            Variable('my_var', kind='algebraic', initial=1.0, units='kg/s')
        ]

    def get_initial_state(self):
        """Provide initial guess"""
        return np.array([1.0])

    def residual(self, state, ports, t, state_dot=None):
        """Define governing equations as f(x) = 0 (or f(x,t) - dx/dt for DAE)"""
        my_var = state[0]

        # For algebraic equations
        eq1 = ...  # Some residual equation

        # For differential equations (if state_dot is provided)
        if state_dot is not None:
            # DAE form: d(my_var)/dt = some_rate
            # Residual: some_rate - state_dot[0]
            rate = ...  # Compute rate of change
            eq1 = rate - state_dot[0]

        # Update outlet ports
        ports['outlet'].mdot = my_var

        return np.array([eq1])
```

See `thermal_sim/components/heater.py` for a complete example.

## Current Capabilities (Phase 3 Complete)

**Implemented Features:**
- ✅ Two-phase flow with quality tracking
- ✅ Transient dynamics (differential equations)
- ✅ Control systems (PID controllers, control valves)
- ✅ DAE solver (mixed differential-algebraic systems)
- ✅ Multi-backend steady-state solver
- ✅ Automatic initialization strategies
- ✅ High performance (>3000× real-time)

**Future Enhancements:**
- ⏳ Spatial discretization (1D distributed models)
- ⏳ Custom Jacobians for faster convergence
- ⏳ Multi-rate integration
- ⏳ Automatic ScalarPort connection in graph
- ⏳ GUI/visualization tools
- ⏳ More control components (valves with hysteresis, etc.)

## Documentation

- `ARCHITECTURE.md`: High-level design philosophy
- `MVP_SPEC.md`: Detailed implementation guide
- `CLAUDE.md`: Development guidelines for AI assistants

## Contributing

This is currently a research/educational project. Contributions, issues, and feature requests are welcome.

## License

MIT License (or specify your chosen license)

## Citation

If you use this simulator in academic work, please cite:

```
Thermal System Simulator v0.1.0
https://github.com/yourusername/thermalsim
```

## Contact

For questions or support, please open an issue on GitHub.

---

**Version**: 0.3.0 (Phase 3 Complete)
**Last Updated**: 2025-11-09
**Python**: 3.11+
**Status**: Production Ready - Full DAE solver with dynamics and control
**Tests**: 144/144 passing
**Performance**: >3000× real-time
