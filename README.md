# Thermal System Simulator

An extensible, physics-based framework for simulating coupled thermal-hydraulic systems.

## Overview

The Thermal System Simulator is a Python-based tool for modeling single-phase fluid loops, power cycles (Rankine, Brayton), and thermal management systems. It uses a graph-based architecture where components are connected via typed ports, and the framework automatically assembles and solves the governing equations.

**Current Status**: Phase 2 Complete - Multi-backend steady-state solver with automatic initialization

### Solver Capabilities

- ✅ **`solve_steady_state()`**: Multi-backend steady-state solver (Phase 2 complete)
  - **Backends**: scipy (default), diffeqpy (Julia NonlinearSolve.jl), sequential
  - **Features**: Automatic sequential initialization, graceful fallback chain
  - **Architecture**: Combines accuracy of simultaneous methods with reliability of sequential solving
  - **Status**: Production ready for Rankine cycles
- ✅ **`solve_sequential()`**: Sequential propagation solver
  - **Purpose**: Initialization helper and fallback for simultaneous solvers
  - **Architecture**: Permanent component of solver strategy (not temporary)
  - **Use case**: Generates thermodynamically consistent initial guesses
- ✅ **`solve()`**: Transient BDF solver (working for ODE systems)

**Key Features**:
- **Typed Port System**: Type-safe connections prevent physically invalid configurations
- **Automatic Conservation**: Reference-sharing mechanism enforces mass/energy conservation
- **Extensible Components**: Add new components by implementing 3 simple methods
- **DAE-Ready Architecture**: Designed for eventual upgrade to full DAE solvers
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

```bash
# Run the Rankine cycle example
python examples/rankine_cycle.py
```

Expected output:
```
Solving Rankine cycle steady state...
✓ Solution converged using: scipy (sequential init)
  Residual norm: 2.98e-08

Results:
  Turbine power: 80.29 MW
  Pump power: 1.29 MW
  Net power: 79.00 MW
  Thermal efficiency: 31.3%
✓ Efficiency in expected range (30-40%)
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
│   ├── port.py        # Typed port system (MassFlowPort, HeatPort)
│   ├── variable.py    # State variable metadata
│   ├── component.py   # Component base class
│   └── graph.py       # ThermalGraph - topology manager
├── components/         # Component library
│   ├── heater.py      # ConstantPressureHeater (boiler/condenser)
│   ├── turbine.py     # Turbine (isentropic expansion)
│   ├── pump.py        # Pump (isentropic compression)
│   └── pipe.py        # Simple pipe with pressure drop
└── properties/         # Thermophysical properties
    └── coolprop_wrapper.py  # CoolProp interface with caching

tests/
├── unit/              # Unit tests for each module
└── integration/       # Full system tests

examples/
└── rankine_cycle.py   # Main demonstration
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

    def residual(self, state, ports, t):
        """Define governing equations as f(x) = 0"""
        my_var = state[0]

        # Your physics equations here
        eq1 = ...  # Some residual equation

        # Update outlet ports
        ports['outlet'].mdot = my_var

        return np.array([eq1])
```

See `thermal_sim/components/heater.py` for a complete example.

## Known Limitations (MVP Phase)

The following features are **not yet implemented** and are planned for future releases:

- ❌ Two-phase flow (only single-phase liquid/vapor)
- ❌ Control systems (no PID controllers)
- ❌ Spatial discretization (0D lumped only)
- ❌ Custom Jacobians (uses finite differences)
- ❌ Multi-rate integration
- ❌ GUI/visualization tools

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

**Version**: 0.2.0 (Phase 2 Complete)
**Last Updated**: 2025-11-09
**Python**: 3.9+
**Status**: Phase 2 - Multi-backend solver with automatic initialization
