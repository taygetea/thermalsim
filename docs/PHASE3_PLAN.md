# Phase 3 Implementation Plan: Two-Phase Flow + Controls + Dynamics

**Status:** Planning
**Date:** 2025-11-09
**Prerequisites:** Phase 2 complete (multi-backend solver)

---

## Executive Summary

**Goal:** Extend the simulator from algebraic steady-state systems to full differential-algebraic equations (DAEs) with two-phase flow and feedback control.

**Key Additions:**
- Two-phase thermodynamics (boiling, condensation, quality tracking)
- Dynamic components with differential equations (tanks, thermal masses)
- Control systems (PID controllers, valves, feedback loops)
- Transient DAE solver for time-varying simulations

**Architecture Impact:**
- `solve_steady_state()` → **NO CHANGES** (still used for initialization)
- `solve_transient(tspan)` → **NEW** (DAE solver for dynamics)
- Component interface → **EXTENDED** (supports `Variable.kind='differential'`)

---

## Background: From Algebraic to DAE

### Current State (Phase 0-2)

**All variables are algebraic:**
```python
# Example: Turbine
Variable('h_out', kind='algebraic')
Variable('W_shaft', kind='algebraic')
Variable('mdot', kind='algebraic')

# Equations: f(x) = 0
eq1 = h_out - (h_in - eta * (h_in - h_out_isentropic))
eq2 = W_shaft - mdot * (h_in - h_out)
```

**Solvers:**
- `solve_steady_state()` → scipy.optimize.root or diffeqpy/NonlinearSolve
- All equations solved simultaneously at steady-state

### Phase 3 Target

**Mix of differential and algebraic variables:**
```python
# Example: Tank with level control
Variable('level', kind='differential')      # d(level)/dt
Variable('temperature', kind='differential') # d(T)/dt
Variable('mdot_out', kind='algebraic')      # Constraint

# Equations: M(x)·dx/dt = f(x,t)
eq1 = d(level)/dt - (mdot_in - mdot_out)/(rho * A)
eq2 = d(T)/dt - Q_in/(m * cp)
eq3 = mdot_out - valve_position * sqrt(rho * (P_tank - P_downstream))
```

**Solvers:**
- `solve_steady_state()` → **STILL USED** for initialization (finds f(x,0) = 0)
- `solve_transient(tspan)` → **NEW** DAE solver (integrates M(x)·dx/dt = f(x,t))

---

## Phase 3 Features

### 1. Two-Phase Flow Support

**Thermodynamic Capabilities:**
- Quality (vapor fraction) tracking: `0 ≤ x ≤ 1`
- Saturation properties (bubble point, dew point)
- Enthalpy-quality relationships: `h = h_f + x·h_fg`
- Void fraction models (homogeneous, slip)
- Phase transition detection

**New Components:**
- `TwoPhaseHeater` - handles boiling within component
- `TwoPhaseCondenser` - handles condensation
- `Separator` - liquid/vapor separation (drum, moisture separator)
- `Mixer` - two-phase stream mixing

**CoolProp Integration:**
```python
# Enable quality-based lookups
h = fluid.enthalpy_from_pq(P, quality)  # Previously avoided
T_sat = fluid.saturation_temp(P)
h_f = fluid.enthalpy_from_pq(P, 0.0)   # Saturated liquid
h_g = fluid.enthalpy_from_pq(P, 1.0)   # Saturated vapor

# Detect phase
if h < h_f:
    phase = 'subcooled_liquid'
elif h > h_g:
    phase = 'superheated_vapor'
else:
    phase = 'two_phase'
    quality = (h - h_f) / (h_g - h_f)
```

### 2. Dynamic Components

**Tank with Level Dynamics:**
```python
class Tank(Component):
    """Storage tank with level and temperature dynamics"""

    def get_variables(self):
        return [
            Variable('level', kind='differential', initial=2.0, units='m'),
            Variable('T', kind='differential', initial=300.0, units='K'),
            Variable('mdot_out', kind='algebraic', initial=100.0, units='kg/s'),
        ]

    def residual(self, state, ports, t):
        level, T, mdot_out = state

        # Geometry
        A = self.cross_section_area  # m^2
        rho = self.fluid.density(self.P, T)

        # Mass balance: d(level)/dt
        eq1 = ports['inlet'].mdot - mdot_out - rho * A * state_dot[0]

        # Energy balance: d(T)/dt (assuming constant pressure)
        m = rho * A * level
        cp = self.fluid.heat_capacity(self.P, T)
        Q_loss = self.U * self.A_wall * (T - T_ambient)
        eq2 = (ports['inlet'].mdot * ports['inlet'].h
               - mdot_out * self.fluid.enthalpy(self.P, T)
               - Q_loss
               - m * cp * state_dot[1])

        # Outlet mass flow (valve equation)
        eq3 = mdot_out - self.valve_cv * sqrt(rho * (self.P - ports['outlet'].P))

        # Update outlet port
        ports['outlet'].mdot = mdot_out
        ports['outlet'].h = self.fluid.enthalpy(self.P, T)

        return np.array([eq1, eq2, eq3])
```

**Thermal Mass:**
```python
class ThermalMass(Component):
    """Thermal capacitance (metal mass, heat exchanger wall, etc.)"""

    def get_variables(self):
        return [
            Variable('T_wall', kind='differential', initial=400.0, units='K'),
        ]

    def residual(self, state, ports, t):
        T_wall = state[0]

        # Heat transfer from fluid to wall
        Q_conv = self.h * self.A * (ports['fluid'].T - T_wall)

        # Thermal capacitance: Q = m·cp·dT/dt
        eq1 = Q_conv - self.mass * self.cp * state_dot[0]

        return np.array([eq1])
```

### 3. Control Systems

**PID Controller:**
```python
class PIDController(Component):
    """PID controller with anti-windup"""

    def __init__(self, name, Kp=1.0, Ki=0.1, Kd=0.01, setpoint=0.0):
        super().__init__(name)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint

        # Ports
        self.measurement = ScalarPort('measurement', direction='in')
        self.output = ScalarPort('output', direction='out')

        self.ports = {
            'measurement': self.measurement,
            'output': self.output,
        }

    def get_variables(self):
        return [
            Variable('integral', kind='differential', initial=0.0),
            Variable('derivative', kind='algebraic', initial=0.0),
            Variable('output_signal', kind='algebraic', initial=0.0),
        ]

    def residual(self, state, ports, t):
        integral, derivative, output_signal = state

        # Error
        error = self.setpoint - ports['measurement'].value

        # PID equation
        eq1 = self.Ki * error - state_dot[0]  # d(integral)/dt = Ki * error
        eq2 = derivative - self.Kd * error    # Derivative term (simplified)
        eq3 = output_signal - (self.Kp * error + integral + derivative)

        # Update output port
        ports['output'].value = np.clip(output_signal, 0.0, 1.0)  # Anti-windup

        return np.array([eq1, eq2, eq3])
```

**Control Valve:**
```python
class ControlValve(Component):
    """Valve with position control (first-order lag)"""

    def get_variables(self):
        return [
            Variable('position', kind='differential', initial=0.5),  # 0 = closed, 1 = open
            Variable('mdot', kind='algebraic', initial=100.0),
        ]

    def residual(self, state, ports, t):
        position, mdot = state

        # Valve dynamics: first-order lag
        tau = 5.0  # Time constant (seconds)
        eq1 = (ports['command'].value - position) / tau - state_dot[0]

        # Flow equation (with position modulation)
        Cv = self.Cv_max * position
        rho = self.fluid.density(ports['inlet'].P, ports['inlet'].h)
        dP = ports['inlet'].P - ports['outlet'].P
        eq2 = mdot - Cv * sqrt(rho * max(dP, 0))

        # Update ports
        ports['outlet'].mdot = mdot
        ports['outlet'].h = ports['inlet'].h  # Assume isenthalpic

        return np.array([eq1, eq2])
```

---

## Implementation Steps

### Step 1: Extend Component Interface for Derivatives

**File:** `thermal_sim/core/component.py`

**Changes:**
```python
class Component(ABC):

    @abstractmethod
    def residual(self, state, ports, t, state_dot=None):
        """
        Compute residual: M(x)·ẋ - f(x,t) = 0

        Args:
            state: Current state variables
            ports: Port dictionary
            t: Current time
            state_dot: Time derivatives (dx/dt). For algebraic systems, all zeros.
                      For DAE systems, solver provides this.

        Returns:
            Residual array (same length as state)

        Note:
            - For algebraic variables: residual should not reference state_dot
            - For differential variables: residual = f(...) - state_dot[i]
        """
        pass
```

**Backward Compatibility:**
All existing components work unchanged because `state_dot=None` is default and algebraic variables don't use it.

### Step 2: Update Graph Assembly for DAE

**File:** `thermal_sim/core/graph.py`

**Add method:**
```python
def assemble_dae(self) -> Tuple[Callable, np.ndarray, np.ndarray, List[bool]]:
    """
    Assemble DAE system for transient solver.

    Returns:
        residual_func: Function(t, y, ydot) -> residuals
        y0: Initial state vector
        ydot0: Initial derivatives
        algebraic_vars: Boolean mask (True = algebraic, False = differential)
    """
    self._build_state_mapping()

    # Collect initial conditions
    y0 = []
    ydot0 = []
    algebraic_vars = []

    for comp in self.components:
        y0.append(comp.get_initial_state())

        # Get variable types
        for var in comp.get_variables():
            if var.kind == 'algebraic':
                ydot0.append(0.0)  # Algebraic vars have zero derivative
                algebraic_vars.append(True)
            else:  # differential
                ydot0.append(0.0)  # Will be computed by solver
                algebraic_vars.append(False)

    y0 = np.concatenate(y0)
    ydot0 = np.array(ydot0)

    # Build residual function
    def residual_func(t, y, ydot):
        residuals = []
        for comp in self.components:
            offset = self._component_offsets[comp.name]
            size = comp.get_state_size()
            state = y[offset:offset+size]
            state_dot = ydot[offset:offset+size]

            res = comp.residual(state, comp.ports, t, state_dot)
            residuals.append(res)

        return np.concatenate(residuals)

    return residual_func, y0, ydot0, algebraic_vars
```

### Step 3: Implement Transient Solver

**File:** `thermal_sim/core/graph.py`

**Add method:**
```python
def solve_transient(self, tspan, backend='scipy', **solver_options):
    """
    Solve transient DAE system: M(x)·dx/dt = f(x,t)

    Uses steady-state solution as initial condition (or user-provided IC).

    Args:
        tspan: (t_start, t_end) or array of time points
        backend: 'scipy' or 'diffeqpy'
        **solver_options: Passed to backend solver
            y0: Optional custom initial condition (otherwise uses solve_steady_state)
            method: Integration method (BDF, Radau, etc.)
            rtol, atol: Tolerances

    Returns:
        Result object with:
            .t: Time points
            .y: State trajectory [n_vars, n_timepoints]
            .success: Convergence flag
            .message: Status message
    """
    # Get or compute initial condition
    if 'y0' in solver_options:
        y0 = solver_options.pop('y0')
    else:
        # Use steady-state as IC
        print("Computing steady-state initial condition...")
        ss_result = self.solve_steady_state()
        if not ss_result.success:
            raise RuntimeError("Failed to find steady-state IC")
        y0 = ss_result.x

    # Assemble DAE
    residual_func, _, ydot0, algebraic_vars = self.assemble_dae()

    if backend == 'scipy':
        return self._solve_transient_scipy(
            residual_func, y0, ydot0, tspan, algebraic_vars, **solver_options
        )
    elif backend == 'diffeqpy':
        return self._solve_transient_diffeqpy(
            residual_func, y0, ydot0, tspan, algebraic_vars, **solver_options
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

def _solve_transient_scipy(self, residual_func, y0, ydot0, tspan, algebraic_vars, **opts):
    """Scipy backend for transient DAE solving"""
    from scipy.integrate import solve_ivp

    # For DAE, we need to use a method that handles implicit equations
    # scipy's solve_ivp with 'Radau' or 'BDF' can handle stiff ODEs
    # For true DAE, we may need to use a wrapper or manual implementation

    # Option 1: Convert DAE to ODE by solving algebraic vars at each step
    # Option 2: Use Radau (implicit, handles algebraic constraints)

    method = opts.pop('method', 'BDF')
    rtol = opts.pop('rtol', 1e-6)
    atol = opts.pop('atol', 1e-8)

    # Define ODE function (will need to handle algebraic constraints)
    def ode_func(t, y):
        # For pure algebraic system, this won't work
        # Need proper DAE solver or algebraic elimination
        ydot = np.zeros_like(y)
        residuals = residual_func(t, y, ydot)

        # This is simplified - proper DAE handling needed
        # May need to solve algebraic equations at each time step
        return -residuals  # Placeholder

    result = solve_ivp(
        ode_func,
        (tspan[0], tspan[1]),
        y0,
        method=method,
        rtol=rtol,
        atol=atol,
        **opts
    )

    # Package result
    result.component_names = [c.name for c in self.components]
    result.component_offsets = self._component_offsets.copy()
    result.state_names = {}
    for comp in self.components:
        result.state_names[comp.name] = comp.get_state_names()

    return result
```

**Note:** Proper DAE solving with scipy is complex. For Phase 3, we should use:
- **Option A:** diffeqpy with Julia's DAE solvers (IDA, Rodas5, etc.)
- **Option B:** Separate algebraic and differential equations, solve algebraic at each step
- **Option C:** Use SUNDIALS IDA via scikits.odes (if we can get it to compile)

### Step 4: Two-Phase Property Extensions

**File:** `thermal_sim/properties/coolprop_wrapper.py`

**Add methods:**
```python
class FluidProperties:

    def saturation_temperature(self, P: float) -> float:
        """Get saturation temperature at pressure P"""
        return PropsSI('T', 'P', P, 'Q', 0.0, self.fluid_name)

    def saturation_pressure(self, T: float) -> float:
        """Get saturation pressure at temperature T"""
        return PropsSI('P', 'T', T, 'Q', 0.0, self.fluid_name)

    def enthalpy_saturated_liquid(self, P: float) -> float:
        """Saturated liquid enthalpy at pressure P"""
        return PropsSI('H', 'P', P, 'Q', 0.0, self.fluid_name)

    def enthalpy_saturated_vapor(self, P: float) -> float:
        """Saturated vapor enthalpy at pressure P"""
        return PropsSI('H', 'P', P, 'Q', 1.0, self.fluid_name)

    def enthalpy_from_quality(self, P: float, quality: float) -> float:
        """Enthalpy from pressure and quality (0 ≤ x ≤ 1)"""
        if not 0 <= quality <= 1:
            raise ValueError(f"Quality must be in [0,1], got {quality}")
        return PropsSI('H', 'P', P, 'Q', quality, self.fluid_name)

    def quality_from_enthalpy(self, P: float, h: float) -> float:
        """
        Compute quality from pressure and enthalpy.
        Returns None if single-phase.
        """
        h_f = self.enthalpy_saturated_liquid(P)
        h_g = self.enthalpy_saturated_vapor(P)

        if h < h_f:
            return None  # Subcooled liquid
        elif h > h_g:
            return None  # Superheated vapor
        else:
            return (h - h_f) / (h_g - h_f)  # Two-phase

    def phase_from_ph(self, P: float, h: float) -> str:
        """Determine phase from P and h"""
        h_f = self.enthalpy_saturated_liquid(P)
        h_g = self.enthalpy_saturated_vapor(P)

        if h < h_f:
            return 'liquid'
        elif h > h_g:
            return 'vapor'
        else:
            return 'two_phase'
```

### Step 5: Create Two-Phase Components

**File:** `thermal_sim/components/two_phase_heater.py`

```python
class TwoPhaseHeater(Component):
    """
    Heater that can handle two-phase flow (boiler).

    Tracks outlet quality if in two-phase region.
    """

    def __init__(self, name: str, P: float, Q: float, fluid: str = 'Water'):
        super().__init__(name)
        self.P = P
        self.Q = Q
        self.fluid = FluidProperties(fluid)

        self.inlet = MassFlowPort('inlet', direction='in')
        self.outlet = MassFlowPort('outlet', direction='out')

        # Initialize
        self.inlet.P = P
        self.inlet.h = 1e6
        self.outlet.P = P
        self.outlet.h = 2.5e6  # Likely two-phase

        self.ports = {'inlet': self.inlet, 'outlet': self.outlet}

    def get_variables(self):
        return [
            Variable('h_out', kind='algebraic', initial=2.5e6, units='J/kg'),
            Variable('mdot', kind='algebraic', initial=100.0, units='kg/s'),
        ]

    def get_state_size(self):
        return 2

    def get_initial_state(self):
        return np.array([2.5e6, 100.0])

    def residual(self, state, ports, t, state_dot=None):
        h_out, mdot = state

        # Energy balance (same as single-phase heater)
        eq1 = self.Q - mdot * (h_out - ports['inlet'].h)

        # Mass flow continuity
        eq2 = mdot - ports['inlet'].mdot

        # Update outlet port
        ports['outlet'].h = h_out
        ports['outlet'].P = self.P
        ports['outlet'].mdot = mdot

        # Can add diagnostics
        phase = self.fluid.phase_from_ph(self.P, h_out)
        if phase == 'two_phase':
            quality = self.fluid.quality_from_enthalpy(self.P, h_out)
            # Could log or store: self.outlet_quality = quality

        return np.array([eq1, eq2])
```

### Step 6: Create Example Demonstrating Phase 3 Features

**File:** `examples/rankine_with_level_control.py`

```python
"""
Rankine cycle with hotwell level control.

Demonstrates:
- Two-phase flow (condenser outlet is saturated liquid)
- Dynamic component (hotwell tank with level)
- Control system (PID controller for level)
- Transient simulation (startup, disturbance response)
"""

from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.two_phase_heater import TwoPhaseHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.components.pump import Pump
from thermal_sim.components.tank import Tank
from thermal_sim.components.control_valve import ControlValve
from thermal_sim.components.pid_controller import PIDController

# Create graph
graph = ThermalGraph()

# High and low pressure
P_high = 10e6  # 10 MPa
P_low = 10e3   # 10 kPa

# Components
boiler = TwoPhaseHeater('boiler', P=P_high, Q=252e6)
turbine = Turbine('turbine', efficiency=0.85, P_out=P_low)
condenser = TwoPhaseHeater('condenser', P=P_low, Q=-173e6)  # Cooling
hotwell = Tank('hotwell', volume=10.0, P=P_low)  # Storage tank
pump = Pump('pump', efficiency=0.80, P_out=P_high)
valve = ControlValve('valve', Cv_max=100.0)

# Control system
level_controller = PIDController(
    'level_controller',
    Kp=10.0,
    Ki=1.0,
    Kd=0.5,
    setpoint=5.0  # Maintain 5m level
)

# Add components
for comp in [boiler, turbine, condenser, hotwell, valve, pump, level_controller]:
    graph.add_component(comp)

# Connect fluid loop
graph.connect(boiler.outlet, turbine.inlet)
graph.connect(turbine.outlet, condenser.inlet)
graph.connect(condenser.outlet, hotwell.inlet)
graph.connect(hotwell.outlet, valve.inlet)
graph.connect(valve.outlet, pump.inlet)
graph.connect(pump.outlet, boiler.inlet)

# Connect control loop
graph.connect(hotwell.level_sensor, level_controller.measurement)
graph.connect(level_controller.output, valve.command)

# Solve transient
print("Simulating startup transient...")
result = graph.solve_transient(
    tspan=(0, 1000),  # 0 to 1000 seconds
    backend='scipy',
    method='BDF'
)

if result.success:
    # Extract and plot results
    import matplotlib.pyplot as plt

    t = result.t
    hotwell_state = graph.get_component_state(result, 'hotwell')
    valve_state = graph.get_component_state(result, 'valve')

    level = hotwell_state[0, :]  # First variable is level
    valve_position = valve_state[0, :]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t, level)
    ax1.axhline(5.0, color='r', linestyle='--', label='Setpoint')
    ax1.set_ylabel('Hotwell Level (m)')
    ax1.legend()
    ax1.grid()

    ax2.plot(t, valve_position * 100)
    ax2.set_ylabel('Valve Position (%)')
    ax2.set_xlabel('Time (s)')
    ax2.grid()

    plt.tight_layout()
    plt.savefig('rankine_level_control.png')
    print("Results saved to rankine_level_control.png")
```

---

## Testing Strategy

### Unit Tests

**Test two-phase properties:**
```python
def test_two_phase_quality_calculation():
    fluid = FluidProperties('Water')
    P = 1e6  # 1 MPa

    h_f = fluid.enthalpy_saturated_liquid(P)
    h_g = fluid.enthalpy_saturated_vapor(P)
    h_mid = 0.5 * (h_f + h_g)

    quality = fluid.quality_from_enthalpy(P, h_mid)
    assert 0.49 < quality < 0.51  # Should be ~0.5

def test_phase_detection():
    fluid = FluidProperties('Water')
    P = 1e6

    h_subcooled = fluid.enthalpy_saturated_liquid(P) - 1000
    h_two_phase = fluid.enthalpy_saturated_liquid(P) + 1000
    h_superheated = fluid.enthalpy_saturated_vapor(P) + 1000

    assert fluid.phase_from_ph(P, h_subcooled) == 'liquid'
    assert fluid.phase_from_ph(P, h_two_phase) == 'two_phase'
    assert fluid.phase_from_ph(P, h_superheated) == 'vapor'
```

**Test dynamic components:**
```python
def test_tank_level_dynamics():
    tank = Tank('test_tank', volume=1.0, P=1e5)

    # Setup ports
    tank.ports['inlet'].mdot = 10.0  # kg/s in
    tank.ports['outlet'].P = 0.9e5   # Slight pressure drop

    # Get residual
    state = tank.get_initial_state()
    state_dot = np.zeros_like(state)

    residuals = tank.residual(state, tank.ports, 0.0, state_dot)

    # With inflow > outflow, level should increase (positive derivative)
    # residual = mdot_in - mdot_out - rho*A*dlevel/dt
    # If residual = 0, then dlevel/dt = (mdot_in - mdot_out)/(rho*A)
    assert residuals is not None
```

### Integration Tests

```python
def test_transient_solver_simple_tank():
    """Test transient solver with simple tank filling"""
    graph = ThermalGraph()
    tank = Tank('tank', volume=1.0, P=1e5)
    # ... setup

    result = graph.solve_transient(tspan=(0, 100), backend='scipy')

    assert result.success
    assert result.t[-1] == 100.0
    # Level should increase over time with constant inflow
    tank_state = graph.get_component_state(result, 'tank')
    level = tank_state[0, :]
    assert level[-1] > level[0]

def test_pid_controller_convergence():
    """Test PID controller brings system to setpoint"""
    # ... build controlled system
    result = graph.solve_transient(tspan=(0, 500))

    # Check controller drives error to zero
    measured_state = graph.get_component_state(result, 'measured_component')
    final_value = measured_state[-1, -1]
    assert abs(final_value - setpoint) < 0.01  # Within 1% of setpoint
```

---

## File Structure Changes

### New Files to Create

```
thermal_sim/components/
├── tank.py                    # NEW: Tank with level/temp dynamics
├── two_phase_heater.py       # NEW: Boiler with two-phase support
├── control_valve.py          # NEW: Valve with position control
├── pid_controller.py         # NEW: PID controller
├── separator.py              # NEW: Liquid/vapor separator
└── mixer.py                  # NEW: Two-phase mixer

thermal_sim/core/
├── control_port.py           # NEW: ScalarPort for control signals

examples/
├── rankine_with_level_control.py  # NEW: Full Phase 3 demo
└── tank_filling.py                # NEW: Simple dynamic example

tests/unit/
├── test_two_phase_properties.py   # NEW
├── test_tank.py                    # NEW
├── test_pid_controller.py          # NEW

tests/integration/
├── test_transient_solver.py        # NEW
└── test_level_control.py           # NEW
```

### Files to Modify

```
thermal_sim/core/component.py
- Update residual() signature to accept state_dot parameter

thermal_sim/core/graph.py
- Add assemble_dae() method
- Add solve_transient() method
- Add _solve_transient_scipy() method
- Add _solve_transient_diffeqpy() method (optional)

thermal_sim/core/port.py
- Add ScalarPort class for control signals

thermal_sim/properties/coolprop_wrapper.py
- Add two-phase property methods
- Add phase detection methods

requirements.txt
- No changes (diffeqpy already added in Phase 2)
```

---

## Milestones

### Milestone 1: Two-Phase Properties (Week 1)
- [ ] Extend FluidProperties with two-phase methods
- [ ] Add phase detection logic
- [ ] Unit tests for quality calculations
- [ ] Create TwoPhaseHeater component
- [ ] Validate against CoolProp directly

### Milestone 2: DAE Infrastructure (Week 2)
- [ ] Update Component.residual() signature
- [ ] Implement ThermalGraph.assemble_dae()
- [ ] Create simple Tank component
- [ ] Test tank with manual time-stepping
- [ ] Verify backward compatibility (all existing tests pass)

### Milestone 3: Transient Solver (Week 3)
- [ ] Implement solve_transient() with scipy backend
- [ ] Create simple tank filling example
- [ ] Debug DAE solver integration
- [ ] Performance testing
- [ ] Compare scipy vs diffeqpy (if available)

### Milestone 4: Control Systems (Week 4)
- [ ] Create PIDController component
- [ ] Create ControlValve component
- [ ] Create ScalarPort for control signals
- [ ] Simple control loop test (tank + valve + PID)

### Milestone 5: Full Integration (Week 5)
- [ ] Build rankine_with_level_control.py example
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Documentation updates

### Milestone 6: Validation & Polish (Week 6)
- [ ] Validate against published Rankine cycle dynamics
- [ ] Test edge cases (phase transitions, controller saturation)
- [ ] Code review and refactoring
- [ ] Update README, ARCHITECTURE, CLAUDE.md
- [ ] Create Phase 3 completion report

---

## Success Criteria

**Phase 3 Complete When:**
- [ ] Tank level dynamics work correctly (mass balance verified)
- [ ] Two-phase heater handles boiling (quality tracking)
- [ ] PID controller achieves setpoint (< 2% steady-state error)
- [ ] Rankine cycle with level control simulates startup
- [ ] All unit tests pass (> 90% coverage)
- [ ] Integration tests pass
- [ ] Performance: > 5× real-time for 1000s simulation
- [ ] Documentation complete

---

## Known Challenges

### 1. DAE Solver Selection

**Challenge:** scipy's solve_ivp doesn't handle DAE well (only ODE).

**Options:**
- **A:** Use diffeqpy with Julia's DAE solvers (Rodas5, IDA)
  - Pro: Robust, well-tested
  - Con: Julia dependency, type conversion issues
- **B:** Solve algebraic constraints at each time step (index reduction)
  - Pro: Use scipy BDF
  - Con: Complex implementation, may be slow
- **C:** Get scikits.odes/SUNDIALS working
  - Pro: Industry standard
  - Con: Compilation issues (tried in Phase 2, failed)

**Recommendation:** Start with Option B (manual algebraic solve), then try Option A if needed.

### 2. Two-Phase Discontinuities

**Challenge:** Phase transitions create discontinuities (h_f → two-phase → h_g)

**Mitigation:**
- Use small smoothing regions near saturation
- Careful event detection in solver
- Provide good initial guesses

### 3. Control Loop Stability

**Challenge:** PID gains need tuning, can oscillate or become unstable

**Mitigation:**
- Provide default gains based on system time constants
- Add anti-windup to integrator
- Document tuning guidelines
- Consider auto-tuning (future)

### 4. Performance

**Challenge:** DAE solving is computationally expensive

**Mitigation:**
- Use sparse Jacobians (future)
- Provide analytical Jacobians for core components
- Profile and optimize hot paths
- Consider JIT compilation (numba)

---

## Dependencies

**Required:**
- numpy >= 1.24.0
- scipy >= 1.10.0
- CoolProp >= 6.4.0
- diffeqpy >= 2.5.0 (already installed from Phase 2)

**Optional:**
- matplotlib >= 3.5.0 (for plotting transients)
- numba >= 0.56.0 (for performance)
- pytest >= 7.0.0 (testing)

---

## Resources

**Reference Materials:**
- SUNDIALS IDA User Guide: https://sundials.readthedocs.io/en/latest/ida/
- DifferentialEquations.jl DAE Docs: https://diffeq.sciml.ai/stable/tutorials/dae_example/
- CoolProp Two-Phase Docs: http://www.coolprop.org/fluid_properties/TwoPhase.html
- PID Control Tuning: Ziegler-Nichols, Cohen-Coon methods

**Example Systems:**
- Textbook Rankine cycle dynamics (El-Wakil, "Nuclear Heat Transport")
- Boiler drum level control (three-element control)
- Pressurizer level control (nuclear power plants)

---

**Document Version**: 1.0.0
**Created**: 2025-11-09
**Status**: Planning - Ready for Implementation
**Next Steps**: Begin Milestone 1 (Two-Phase Properties)
