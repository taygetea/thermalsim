# CLAUDE.md

This file provides guidance to Claude Code when working on the Thermal System Simulator.

## Project Overview

**Thermal System Simulator** - An extensible, physics-based framework for simulating coupled thermal-hydraulic systems. Currently in MVP phase implementing single-phase Rankine cycle simulation.

- **Language**: Python 3.11+
- **Core Stack**: numpy, scipy, CoolProp
- **Architecture**: Graph-based component assembly with typed ports
- **Current Phase**: MVP (single-phase, 0D lumped components)

## Quick Start Guide for Claude Code Sessions

### Project Status (MVP Phase)

**Current Goal**: Implement minimal viable product with:
- Core graph/port/component infrastructure
- Four basic components: Heater, Turbine, Pump, Pipe
- Working Rankine cycle example achieving 30-35% efficiency
- >85% test coverage

**What's Implemented**: [Track as development progresses]
- [ ] Core abstractions (port.py, component.py, variable.py, graph.py)
- [ ] Property wrapper (coolprop_wrapper.py)
- [ ] Components (heater, turbine, pump, pipe)
- [ ] Rankine cycle example
- [ ] Test suite

### Complete File Location Map

```
thermal_sim/
│
├── core/                           # Core abstractions
│   ├── port.py                     # ⭐ Typed port system (MassFlowPort, HeatPort)
│   ├── variable.py                 # State variable metadata
│   ├── component.py                # ⭐ Component base class (ABC with residual interface)
│   └── graph.py                    # ⭐ ThermalGraph - topology manager & solver interface
│
├── components/                     # Component library
│   ├── heater.py                   # ConstantPressureHeater (boiler/condenser)
│   ├── turbine.py                  # Turbine (isentropic expansion)
│   ├── pump.py                     # Pump (isentropic compression)
│   └── pipe.py                     # Simple pipe with pressure drop
│
├── properties/                     # Thermophysical properties
│   └── coolprop_wrapper.py         # ⭐ CoolProp interface with LRU caching
│
└── utils/                          # Utilities
    └── validators.py               # Conservation checks (future)

tests/
├── unit/                           # Unit tests for each module
│   ├── test_ports.py
│   ├── test_component.py
│   └── test_graph.py
├── integration/                    # Full system tests
│   └── test_rankine.py             # ⭐ Main validation test
└── fixtures/
    └── test_data.py

examples/
├── rankine_cycle.py                # ⭐ Main demonstration
└── simple_loop.py                  # Minimal 2-component test

docs/
├── ARCHITECTURE.md                 # ⭐ Full design rationale
└── MVP_SPEC.md                     # ⭐ Implementation guide
```

**Key: ⭐ = Core file, frequently referenced**

## Key Design Principles

1. **Layered Architecture**: Physics, topology, and numerics are distinct modules
2. **Typed Ports**: Type safety at connection time (mass vs heat ports)
3. **Reference Sharing**: Ports connected via shared object reference (automatic conservation)
4. **DAE-Native Design**: Components define residuals f(x, dx/dt, t) = 0
5. **Component Extensibility**: Add new components by implementing 3 methods
6. **SI Units Only**: No conversion overhead, document units in docstrings

## Common Development Tasks

### Add a New Component

1. Create file in `thermal_sim/components/`
2. Inherit from `Component` base class
3. Implement three methods:
   - `get_variables()` → declare state variables
   - `get_initial_state()` → provide initial guess
   - `residual(state, ports, t)` → define governing equations
4. Add unit tests in `tests/unit/`
5. Test integration in existing examples

**Example skeleton**:
```python
class MyComponent(Component):
    def get_variables(self):
        return [Variable('x', kind='algebraic', initial=1.0)]

    def get_initial_state(self):
        return np.array([1.0])

    def residual(self, state, ports, t):
        # Define equations as f(x) = 0
        # Update outlet ports
        return np.array([...])
```

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=thermal_sim --cov-report=html

# Specific test file
pytest tests/unit/test_ports.py

# Integration test only
pytest tests/integration/test_rankine.py -v
```

### Run Examples

```bash
# Main Rankine cycle demo
python examples/rankine_cycle.py

# Simple 2-component test
python examples/simple_loop.py
```

### Debug Solver Issues

1. Check initial conditions: `component.get_initial_state()`
2. Verify port connections: `graph.validate_topology()`
3. Check residual shape: Should match state vector length
4. Try tighter tolerances: `graph.solve(..., rtol=1e-8, atol=1e-10)`
5. Reduce max_step: `graph.solve(..., max_step=10.0)`

### View Component State During Solution

```python
result = graph.solve(...)
comp_state = graph.get_component_state(result, 'turbine')
# comp_state shape: [n_vars, n_timepoints]
```

## Architecture Quick Reference

### Port Connection Mechanism

```python
# When you call:
graph.connect(comp_a.outlet, comp_b.inlet)

# Behind the scenes:
# - Type compatibility checked
# - Direction compatibility checked
# - Reference sharing implemented:
#   comp_b.ports['inlet'] = comp_a.ports['outlet']
# - Now both reference same Port object
# - Conservation automatic: what A writes, B reads
```

### Component Residual Pattern

```python
def residual(self, state, ports, t):
    # Extract state variables
    var1, var2 = state

    # Read from inlet ports
    h_in = ports['inlet'].h

    # Define equations (residual form: f(x) = 0)
    eq1 = ... - ...  # First equation
    eq2 = ... - ...  # Second equation

    # Update outlet ports (side effect)
    ports['outlet'].h = computed_enthalpy
    ports['outlet'].P = computed_pressure

    return np.array([eq1, eq2])
```

### State Vector Layout

Global state = `[comp1_states..., comp2_states..., comp3_states...]`

Graph tracks offsets for slicing during residual evaluation.

## Testing Strategy

### Unit Tests
- Test each module in isolation
- Mock dependencies (use dummy Port objects)
- Check shapes, types, edge cases
- Target: >90% coverage

### Integration Tests
- Full Rankine cycle from start to finish
- Validate physical correctness:
  - Thermal efficiency in expected range (30-35%)
  - Energy balance closure (<1% error)
  - Mass conservation (<0.1% error)
- Check solver convergence

### Validation Approach
- Compare against textbook Rankine cycle performance
- Check property evaluations against NIST tables
- Verify isentropic relations in turbine/pump

## Dependencies & Setup

```bash
# Install dependencies
pip install numpy scipy CoolProp pytest pytest-cov

# Or if using setup.py:
pip install -e .
pip install -e ".[dev]"

# Verify installation
python -c "import CoolProp; print(CoolProp.__version__)"
```

**Python Version**: Requires 3.11+ (for typing features)

## Development Workflow

### Issue-Driven Development

When starting a session:
1. Check open issues: `gh issue list`
2. Pick an issue or create new one: `gh issue create`
3. Implement and test
4. Commit with reference: `git commit -m "Fix turbine efficiency (#issue)"`
5. Close issue: `gh issue close <number>`

### Git Workflow
- Commit directly to main (small project, single developer)
- Reference spec documents in commit messages
- Keep commits focused and atomic

## Known Limitations (MVP Phase)

**Explicitly NOT Implemented** (see MVP_SPEC.md section 11):
- ❌ Two-phase flow (only single-phase liquid/vapor)
- ❌ Control systems (no PID, no feedback)
- ❌ Spatial discretization (0D lumped only)
- ❌ Multi-rate integration
- ❌ Custom Jacobians (finite diff only)
- ❌ GUI/visualization (console output only)
- ❌ Unit checking with pint (raw SI floats)

**Future Phases** (see ARCHITECTURE.md section 16):
- Phase 1: Two-phase flow + controls
- Phase 2: Multi-rate solver + MPC
- Phase 3: Exotic domains (fusion, cryogenics)
- Phase 4: GPU/distributed computing

## Success Criteria

MVP is complete when:
- [ ] All unit tests pass (>85% coverage)
- [ ] Rankine cycle example runs successfully
- [ ] Thermal efficiency: 30-35%
- [ ] Energy balance error: <1%
- [ ] Mass conservation error: <0.1%
- [ ] Can add new component without modifying core
- [ ] Documentation complete (docstrings, README, report)

## Quick Reference Commands

```bash
# Testing
pytest                              # Run all tests
pytest -k rankine                   # Run Rankine tests only
pytest --cov=thermal_sim            # With coverage

# Examples
python examples/rankine_cycle.py    # Main demo
python -m pytest tests/integration  # Integration tests

# Code quality
black thermal_sim/                  # Format code (if using black)
pylint thermal_sim/                 # Lint (if using pylint)

# CoolProp verification
python -c "from CoolProp.CoolProp import PropsSI; print(PropsSI('H','P',1e6,'T',500,'Water'))"
```

## Physical Constants & Typical Values

**Rankine Cycle Operating Conditions**:
- High pressure (boiler): 10 MPa (10e6 Pa)
- Low pressure (condenser): 10 kPa (10e3 Pa)
- Typical efficiency: 30-40%
- Steam enthalpy range: ~200 kJ/kg (liquid) to ~3000 kJ/kg (superheated)

**Component Efficiencies**:
- Turbine isentropic efficiency: 0.80-0.90
- Pump isentropic efficiency: 0.75-0.85

**SI Units Reference**:
- Pressure: Pa (Pascal)
- Temperature: K (Kelvin)
- Enthalpy: J/kg
- Entropy: J/(kg·K)
- Mass flow: kg/s
- Power: W (Watt)

## Important Notes for Development

### Residual Equation Sign Conventions
Always write as `f(x) = 0` form:
- ✅ Good: `residual = Q_in - mdot * (h_out - h_in)`
- ❌ Bad: `h_out = h_in + Q_in/mdot` (not residual form)

### Port Update Side Effects
Component `residual()` methods MUST update outlet ports:
```python
ports['outlet'].h = computed_value  # Required!
ports['outlet'].P = self.P_out       # Required!
```

### Variable Order Consistency
Order in `get_variables()` must match order in:
- `get_initial_state()` array
- `residual()` state unpacking
- `residual()` return array

### CoolProp Property Calls
Always use (P, h) or (P, T) pairs - these work for all single-phase states.

Avoid quality-based lookups (Q) in MVP.

## Pending Tasks

*This section will be updated as work progresses*

### Phase 0: Project Setup
- [x] Create repository structure
- [x] Write specification documents
- [ ] Set up Python package structure
- [ ] Configure pytest
- [ ] Create requirements.txt

### Phase 1: Core Infrastructure
- [ ] Implement port types (port.py)
- [ ] Implement variable metadata (variable.py)
- [ ] Implement component base class (component.py)
- [ ] Implement graph kernel (graph.py)
- [ ] Implement property wrapper (coolprop_wrapper.py)

### Phase 2: Component Library
- [ ] Implement ConstantPressureHeater
- [ ] Implement Turbine
- [ ] Implement Pump
- [ ] Implement Pipe

### Phase 3: Integration
- [ ] Create simple 2-component example
- [ ] Create full Rankine cycle example
- [ ] Write integration tests
- [ ] Validate against textbook values

### Phase 4: Documentation
- [ ] Write README.md
- [ ] Complete API docstrings
- [ ] Write technical report
- [ ] Add usage examples

---

**Document Version**: 0.1.0
**Last Updated**: 2025-11-07
**See Also**: ARCHITECTURE.md (design rationale), MVP_SPEC.md (implementation details)
