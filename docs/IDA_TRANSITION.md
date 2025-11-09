# SUNDIALS IDA Transition Guide (Phase 2)

**STATUS**: ✅ COMPLETED WITH ALTERNATIVE APPROACH (diffeqpy/NonlinearSolve)
**Completion Date**: 2025-11-09
**See**: docs/PHASE2_IMPLEMENTATION_PLAN.md for actual implementation

---

## What Actually Happened

**Original Plan**: Use SUNDIALS IDA via scikits.odes

**What We Did Instead**: Multi-backend solver with scipy, diffeqpy/NonlinearSolve.jl, and sequential

**Why the Change**:
1. scikits.odes had compilation issues (SUNDIALS 6.4.1 API mismatch)
2. diffeqpy IDA had Python/Julia type conversion bugs
3. **Key insight**: For steady-state, NonlinearSolve.jl is more appropriate than IDA
4. IDA is for transient DAE problems M(x)·dx/dt = f(x,t), not steady-state f(x) = 0

**Implementation Highlights**:
- Multi-backend solve_steady_state(): scipy (default), diffeqpy, sequential
- Automatic sequential initialization when direct solvers fail
- Sequential solver is PERMANENT (provides initialization, not temporary)
- Graceful fallback chain: scipy → sequential init → sequential fallback
- Result: "scipy (sequential init)" - best of both worlds

**Test Results**: 22/26 tests passing, Rankine cycle: 31.3% efficiency

---

## Original Plan (For Reference)

**Goal**: Replace temporary sequential solver with SUNDIALS IDA for proper DAE solving.

**Critical Steps**:
1. Install: `pip install scikits.odes`
2. Edit `thermal_sim/core/graph.py`:
   - Replace existing `solve_steady_state()` method with IDA version (see Step 2)
   - **CRITICAL**: Include the `residual_func_ida` adapter (signature mismatch!)
3. Update `examples/rankine_cycle.py`: `solve_sequential()` → `solve_steady_state()`
4. Delete `solve_sequential()` method entirely
5. Run tests: Should get same ~31% efficiency
6. Update docs to remove "temporary solver" warnings

**Most Common Pitfall**: Forgetting the residual function adapter. IDA needs `(t, y, yp, result)` signature, we have `(t, y) -> array`.

**If You Get Stuck**: See troubleshooting section, especially "TypeError about residual function signature"

---

## Overview

This document provides a detailed roadmap for transitioning from the temporary sequential solver to the proper SUNDIALS IDA DAE solver. This transition is **Phase 2** of the project roadmap.

**Current Status (Phase 1 - MVP)**: Using `solve_sequential()` - a temporary workaround that propagates states around the cycle sequentially and uses root finding to close the loop.

**Phase 2 Goal**: Replace with `solve_steady_state()` using SUNDIALS IDA - a proper DAE solver that solves all equations simultaneously.

### Post-Transition Architecture

After Phase 2 completion, the simulator will have **two permanent solver entry points**:

- **`solve_steady_state()`** — Steady-state DAE solution via SUNDIALS IDA
  - Used for finding equilibrium states
  - Handles pure algebraic systems (all derivatives = 0)
  - Generalizes to arbitrary component topologies

- **`solve()`** — Transient integration via BDF (scipy's solve_ivp)
  - Used for time-dependent simulations
  - Handles ODE systems and simple dynamics

The temporary `solve_sequential()` method will be **completely removed**. There will be no "collapsing" of solvers - these two methods serve distinct purposes and will both remain permanent parts of the architecture.

## Files Requiring Modification

For quick orientation, here are ALL files that need changes in Phase 2:

### Code Changes:
1. **`thermal_sim/core/graph.py`** (CRITICAL)
   - Replace `solve_steady_state()` method (lines ~229-315)
   - Delete `solve_sequential()` method (lines ~317-545)
   - Remove all `# TODO_IDA:` comments throughout file

2. **`examples/rankine_cycle.py`**
   - Line ~77-80: Change `solve_sequential()` → `solve_steady_state()`
   - Remove TODO_IDA comments

3. **`requirements.txt`**
   - Uncomment `scikits.odes>=2.7.0`

### Documentation Changes:
4. **`README.md`**
   - Remove "Solver Status" warning section (lines 11-27)
   - Update Quick Start example to use `solve_steady_state()`

5. **`ARCHITECTURE.md`**
   - Remove Section 7.1 MVP implementation note about sequential solver
   - Update solver table

6. **`CLAUDE.md`**
   - Remove sequential solver from development workflow
   - Update solver section

7. **`docs/IDA_TRANSITION.md`** (this file)
   - Mark as COMPLETED at top

### Test Changes:
8. **`tests/integration/test_rankine.py`**
   - Keep sequential solver tests for validation comparison
   - Add new IDA-based tests

### No Changes Needed:
- ✅ All component files (`thermal_sim/components/*.py`) - residuals are permanent
- ✅ `thermal_sim/core/port.py`, `component.py`, `variable.py` - no changes
- ✅ `thermal_sim/properties/coolprop_wrapper.py` - no changes

## Why This Transition is Necessary

### Current Limitations (Sequential Solver)
1. **Topology Restrictions**: Only handles 4-component cycles with known ordering
2. **Manual Propagation**: Requires hardcoded sequential state updates
3. **Limited Convergence**: Root finding on a single variable (mdot) may fail for complex cycles
4. **Not Scalable**: Doesn't generalize to arbitrary component graphs
5. **Temporary Hacks**: Bounds checking and error handling to guide solver away from invalid regions

### Benefits of SUNDIALS IDA
1. **Simultaneous Solution**: Solves all component equations at once
2. **General Topology**: Works with any DAG (directed acyclic graph) or cycles
3. **Robust Convergence**: Newton-Krylov methods with adaptive step sizing
4. **DAE-Native**: Designed for differential-algebraic equations M(x)·ẋ = f(x,t)
5. **Industry Standard**: Used in real engineering simulation tools

## Phase 2 Implementation Steps

### Step 1: Install SUNDIALS IDA

**System Dependencies** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y libsundials-dev swig
```

**Python Package**:
```bash
pip install scikits.odes>=2.7.0
```

**Verify Installation**:
```python
from scikits.odes import dae
print("SUNDIALS IDA available!")
```

### Step 2: Replace `ThermalGraph.solve_steady_state()` Method

**File**: `thermal_sim/core/graph.py`

**⚠️ Important Notes**:
- There's an **existing** `solve_steady_state()` method (lines ~229-315) that uses scipy's root finder
- This existing method **does not work** (hence why we created `solve_sequential()`)
- **REPLACE** the existing `solve_steady_state()` entirely with the IDA implementation below
- Do NOT add a new method - overwrite the old one

**Existing Method to Replace** (lines ~229-315):
```python
def solve_steady_state(self, method: str = 'hybr', **solver_options) -> object:
    """
    Find steady-state solution using root finding.
    ...
    """
    # Uses scipy.optimize.root - this doesn't work well for our system
```

**Also in File** - Temporary solver to DELETE later (lines ~317-545):
```python
def solve_sequential(self, max_iter: int = 100, tol: float = 1e-6):
    """⚠️ TEMPORARY SOLVER - REPLACE WITH SUNDIALS IDA (Phase 2) ⚠️"""
    # Sequential propagation logic...
    # Uses brentq to find mdot that closes loop
```

**Phase 2 Implementation**:
```python
def solve_steady_state(self, t_span=(0, 1), rtol=1e-6, atol=1e-8):
    """
    Solve for steady-state using SUNDIALS IDA DAE solver.

    This solver handles general DAE systems of the form:
        M(x) · dx/dt = f(x, t)

    For steady-state problems where all derivatives are zero:
        0 = f(x, t)

    Parameters
    ----------
    t_span : tuple
        Time span (start, end). For steady-state, just use (0, 1).
    rtol : float
        Relative tolerance for solver convergence.
    atol : float
        Absolute tolerance for solver convergence.

    Returns
    -------
    result : DAESolution
        Solution object with attributes:
        - success: bool
        - message: str
        - values: Solution values at final time
        - errors: Residual errors
    """
    from scikits.odes import dae

    # Assemble system - returns residual_func(t, y) -> array
    residual_func, y0 = self.assemble()

    # CRITICAL: IDA needs signature (t, y, yp, result) not (t, y) -> array
    # Create adapter to convert between signatures
    def residual_func_ida(t, y, yp, result):
        """
        Adapter for IDA's residual function signature.

        IDA expects: f(t, y, yp, result) that populates result array
        We have: f(t, y) that returns array

        For steady-state: yp (derivatives) are all zero and can be ignored
        """
        result[:] = residual_func(t, y)

    # For steady-state, all variables are algebraic (no differential equations)
    # IDA expects indices of algebraic variables (or boolean mask)
    algvar_flags = np.arange(len(y0))  # All variables are algebraic

    # SUNDIALS IDA requires derivatives y' as well as y
    # For steady-state, y' = 0 initially
    yp0 = np.zeros(len(y0))

    # Create DAE solver with adapted residual function
    solver = dae(
        'ida',
        residual_func_ida,  # Use adapter, not residual_func directly!
        algebraic_vars_idx=algvar_flags,
        rtol=rtol,
        atol=atol,
        old_api=False  # Use new API
    )

    # Solve
    solution = solver.solve(t_span, y0, yp0)

    # Extract steady-state values (last time point)
    if solution.flag >= 0:  # Success
        y_final = solution.values.y[-1]

        # Evaluate final residual using original function
        residual_final = residual_func(t_span[1], y_final)

        return type('DAESolution', (), {
            'success': True,
            'message': 'Steady-state solution converged',
            'x': y_final,  # For compatibility with current API
            'fun': residual_final,
            'values': solution.values,
            'errors': solution.errors
        })()
    else:
        return type('DAESolution', (), {
            'success': False,
            'message': solution.message,
            'x': y0,
            'fun': np.full(len(y0), np.inf)
        })()
```

**Key Changes**:
1. Import `from scikits.odes import dae`
2. **CRITICAL**: Create `residual_func_ida` adapter to convert signature
   - Current `assemble()` returns `(t, y) -> array`
   - IDA requires `(t, y, yp, result)` that populates result
3. Set all variables as algebraic: `algvar_flags = np.arange(len(y0))`
   - Alternatively use boolean mask: `[True] * len(y0)`
4. Initialize derivatives to zero: `yp0 = np.zeros(len(y0))`
5. Create IDA solver with adapted residual function
6. Return standardized solution object

**Why the Adapter is Needed**:
The current architecture keeps components clean by having `residual()` methods return arrays. IDA uses an in-place modification pattern for performance. The adapter bridges these two designs without requiring changes to component code.

**Note on Future Dynamic Variables**:

In MVP (Phase 1-2), all state variables are algebraic - we're solving steady-state problems where all derivatives are zero. This is why we set:
- `algvar_flags = np.arange(len(y0))` - All variables are algebraic
- `yp0 = np.zeros(len(y0))` - All derivatives are zero

In **future phases** (Phase 3+), when dynamic components are introduced (e.g., thermal capacitance, fluid inertia), the system will include both differential and algebraic variables:

```python
# Future: Mixed DAE system example
algvar_flags = []  # List of algebraic variable indices only
# If variables 0, 1, 3 are algebraic and 2, 4 are differential:
algvar_flags = [0, 1, 3]

# Or using boolean mask (True = algebraic, False = differential):
algvar_mask = [True, True, False, True, False]  # Same example
```

At that point, components will declare variable types in `get_variables()` with `kind='differential'` vs `kind='algebraic'`, and the graph will build the appropriate index arrays.

### Step 3: Update Component Initial Conditions

**Why**: IDA benefits from better initial guesses than the sequential solver.

**Files**: All components in `thermal_sim/components/`

**Recommended Approach**:
1. Keep current `get_initial_state()` methods
2. Optionally add a `compute_thermodynamic_initial_state()` helper that uses CoolProp
3. For complex systems, consider using sequential solver output as initial guess for IDA

**Example** (optional enhancement):
```python
def get_initial_state_from_pressures(self, P_in, P_out=None):
    """Compute thermodynamically consistent initial state."""
    if P_out is None:
        P_out = self.P_out

    # Use CoolProp to get realistic enthalpy values
    # This is optional - current fixed values work fine
    h_in = self.fluid.enthalpy_from_pt(P_in, 500.0)  # Reasonable temp
    # ... component-specific logic
```

### Step 4: Update Examples

**File**: `examples/rankine_cycle.py`

**Current Code** (lines 77-80):
```python
# TODO_IDA Phase 2: Replace solve_sequential() with solve_steady_state() once SUNDIALS IDA is integrated
# The sequential solver is a temporary workaround for the coupled algebraic system
print("Solving Rankine cycle (using temporary sequential solver)...")
result = graph.solve_sequential()  # TEMPORARY - will be solve_steady_state() with IDA
```

**Phase 2 Update**:
```python
# Solve for steady state using SUNDIALS IDA
print("Solving Rankine cycle...")
result = graph.solve_steady_state()
```

**That's it!** All TODO_IDA comments can be removed.

### Step 5: Remove Sequential Solver

**File**: `thermal_sim/core/graph.py`

**Action**: Delete entire `solve_sequential()` method (lines ~305-513)

**Why**: No longer needed - IDA handles everything the sequential solver did, plus more.

### Step 6: Update Tests

**File**: `tests/integration/test_rankine.py`

**Current** (if exists):
```python
result = graph.solve_sequential()
```

**Phase 2 Update**:
```python
result = graph.solve_steady_state()
```

**New Test Cases to Add**:
1. Test with different solver tolerances
2. Test convergence from poor initial guesses
3. Test with modified cycle topologies (e.g., reheat Rankine)
4. Verify residual norms are below tolerance

**Template for New IDA Tests**:
```python
def test_rankine_ida_converges():
    """IDA solver should converge for Rankine cycle"""
    graph = build_rankine_cycle()

    result = graph.solve_steady_state()

    assert result.success, f"IDA solver failed: {result.message}"
    assert np.all(np.isfinite(result.x)), "Solution contains NaN or Inf"
    assert np.linalg.norm(result.fun) < 1e-6, "Residual norm too large"


def test_rankine_ida_efficiency():
    """IDA solver should produce correct thermal efficiency"""
    graph = build_rankine_cycle()

    result = graph.solve_steady_state()
    assert result.success

    turbine_state = graph.get_component_state(result, 'turbine')
    pump_state = graph.get_component_state(result, 'pump')

    W_turbine = turbine_state[1, -1]
    W_pump = pump_state[1, -1]
    Q_in = 252e6  # Must match component definition

    efficiency = (W_turbine - W_pump) / Q_in

    # Should match sequential solver within tolerance
    assert 0.30 < efficiency < 0.35, \
        f"Efficiency {efficiency:.1%} outside expected range"
```

**Strategy**: Keep the sequential solver tests temporarily to validate that IDA produces equivalent results, then remove them once validated.

### Step 7: Update requirements.txt

**File**: `requirements.txt`

**Current**: Contains placeholder comment for scikits.odes
```
# scikits.odes>=2.7.0  # For Phase 2 - SUNDIALS IDA integration
```

**Update to**:
```
scikits.odes>=2.7.0  # SUNDIALS IDA for DAE solving
```

### Step 8: Update Documentation

**Files to Update**:
- `README.md`: Change solver description
- `ARCHITECTURE.md`: Update Section 7.1 (Solver Infrastructure)
- `CLAUDE.md`: Remove sequential solver references
- `docs/IDA_TRANSITION.md`: Mark as "COMPLETED" at top

**README.md Changes**:

1. Remove "Solver Status" section (lines 11-27) - no longer temporary!
2. Update Quick Start example (line 73-74):
```python
# Old
result = graph.solve_sequential()

# New
result = graph.solve_steady_state()
```

**README.md Solver Section to Add**:
```markdown
## Solver

The simulator uses **SUNDIALS IDA** for solving the coupled differential-algebraic
equations that arise from component interconnections. IDA is an industry-standard
DAE solver that handles:
- Simultaneous solution of all component equations
- Arbitrary component topologies (including cycles)
- Robust convergence for stiff systems
```

## Migration Checklist

Use this checklist when performing the Phase 2 transition:

- [ ] **Environment Setup**
  - [ ] Install `libsundials-dev` system package
  - [ ] Install `scikits.odes` Python package
  - [ ] Verify import: `from scikits.odes import dae`

- [ ] **Code Changes**
  - [ ] Implement new `solve_steady_state()` in `graph.py`
  - [ ] Update `examples/rankine_cycle.py`
  - [ ] Update any other examples in `examples/`
  - [ ] Delete `solve_sequential()` method
  - [ ] Remove all `# TODO_IDA:` comments

- [ ] **Testing**
  - [ ] Run `pytest tests/integration/test_rankine.py`
  - [ ] Verify efficiency still in 30-40% range
  - [ ] Check residual norms < 1e-6
  - [ ] Test with different rtol/atol values
  - [ ] Validate against compute_rankine_states.py output

- [ ] **Documentation**
  - [ ] Update README.md solver section
  - [ ] Update ARCHITECTURE.md Section 8
  - [ ] Update CLAUDE.md to remove sequential solver
  - [ ] Mark this file as COMPLETED

- [ ] **Validation**
  - [ ] Compare IDA results to sequential solver results
  - [ ] Efficiency should match (±0.5%)
  - [ ] Component powers should match (±1%)
  - [ ] Mass flow rates should match (±0.1%)

### Completion Protocol

When all checklist items above are complete:

1. **Run full test suite** one final time:
   ```bash
   pytest tests/ -v --cov=thermal_sim
   ```

2. **Verify example runs successfully**:
   ```bash
   python examples/rankine_cycle.py
   # Should show efficiency ~31% using IDA
   ```

3. **Commit changes** with descriptive message:
   ```bash
   git add .
   git commit -m "Phase 2: SUNDIALS IDA integration complete

   - Replace solve_steady_state() with IDA implementation
   - Remove temporary solve_sequential() method
   - Update all examples and tests
   - Remove TODO_IDA comments
   - Update documentation

   All tests passing. Rankine cycle achieves 31.3% efficiency with IDA."
   git push
   ```

4. **Tag release** as v0.2.0:
   ```bash
   git tag -a v0.2.0 -m "Phase 2: SUNDIALS IDA integration"
   git push origin v0.2.0
   ```

5. **Update this file's status** at the bottom to:
   ```
   **Status**: COMPLETED — IDA integrated
   **Completion Date**: [date]
   **Effective Version**: v0.2.0
   ```

6. **Archive this file** (optional) by moving to `docs/archive/IDA_TRANSITION.md`

## Expected Behavior Changes

### Performance
- **Startup Time**: Slightly slower (IDA initialization overhead)
- **Convergence Time**: May be faster for complex systems
- **Memory Usage**: Slightly higher (IDA internal state)

### Convergence
- **Robustness**: Better - handles larger deviations from initial guess
- **Failure Modes**: Different error messages (IDA-specific)
  - Error codes available via `solution.flag` attribute
  - Negative flag values indicate failures (see IDA documentation)
  - Consider logging `solution.flag` for debugging convergence issues
- **Tolerance**: More consistent across different cycle configurations

### Results
- **Accuracy**: Should be identical within solver tolerances
- **Efficiency**: Expect ±0.1% variation due to different numerical methods
- **Residuals**: IDA may achieve lower residual norms

## Troubleshooting Phase 2 Issues

### Issue: TypeError about residual function signature

**Symptoms**:
- `TypeError: residual_func() takes 2 positional arguments but 4 were given`
- Error during solver setup or first iteration

**Cause**: Forgot to use the adapter - passed `residual_func` directly to IDA instead of `residual_func_ida`

**Solution**:
Make sure you have the adapter function defined and use it:
```python
def residual_func_ida(t, y, yp, result):
    result[:] = residual_func(t, y)

solver = dae('ida', residual_func_ida, ...)  # Use adapter!
```

### Issue: IDA fails to converge

**Symptoms**: `solution.flag < 0` or "IDA_CONV_FAIL" error

**Solutions**:
1. Check initial conditions: `print(y0)` - look for NaN, Inf, or zeros
2. Tighten tolerances: `rtol=1e-8, atol=1e-10`
3. Verify port initialization in component constructors
4. Try using sequential solver output as initial guess:
   ```python
   seq_result = graph.solve_sequential()
   y0 = seq_result.x
   # Then pass y0 to solve_steady_state()
   ```

### Issue: "Algebraic variable has non-zero derivative"

**Symptoms**: Error during IDA setup

**Solution**: Ensure `yp0 = np.zeros(len(y0))` for all steady-state problems.

### Issue: CoolProp errors during IDA solve

**Symptoms**: "Input out of range" from CoolProp

**Solution**:
1. Add bounds checking in component `residual()` methods
2. Initialize ports with physically valid values in constructors
3. Use IDA's constraint options to keep variables in valid ranges

### Issue: Results differ from sequential solver

**Symptoms**: Efficiency changes by >1%

**Investigation**:
1. Check residual norms: `np.linalg.norm(result.fun)`
2. Print component states: `graph.get_component_state(result, 'turbine')`
3. Verify port values after solution
4. Check for coding errors in residual equations

## References

### SUNDIALS IDA Documentation
- Official: https://computing.llnl.gov/projects/sundials/ida
- scikits.odes API: https://scikits-odes.readthedocs.io/

### Relevant Code Locations
- Current solver: `thermal_sim/core/graph.py:305-513` (`solve_sequential()`)
- Target solver: `thermal_sim/core/graph.py` (new `solve_steady_state()`)
- Component residuals: `thermal_sim/components/*.py` (no changes needed)

### Example DAE Systems
For reference on using SUNDIALS IDA, see:
- scikits.odes examples: https://github.com/bmcage/odes/tree/master/ipython_examples
- SUNDIALS C examples: https://github.com/LLNL/sundials/tree/main/examples/ida

## Appendix: Residual Function Interface

The residual function interface does NOT need to change for IDA. However, for reference:

**Current Interface** (used by both sequential solver and will be used by IDA):
```python
def residual_func(t, y):
    """Evaluate system residuals."""
    residuals = np.zeros(len(y))
    # ... populate residuals from component equations
    return residuals
```

**IDA Interface** (scikits.odes):
```python
def residual_func_ida(t, y, yp, result):
    """
    Evaluate DAE residuals: f(t, y, y') = 0

    Parameters
    ----------
    t : float
        Current time
    y : ndarray
        State variables
    yp : ndarray
        Derivatives (dy/dt)
    result : ndarray (output)
        Residual values to populate
    """
    # For steady-state algebraic equations: f(y) = 0
    # yp will be zero, so ignore it
    residuals = np.zeros(len(y))
    # ... component residual evaluation
    result[:] = residuals
```

**Adapter** (if needed):
```python
# If existing residual_func has signature residual_func(t, y)
def residual_func_ida(t, y, yp, result):
    result[:] = residual_func(t, y)
```

However, this adapter is usually not needed - just define `residual_func` with the IDA signature from the start.

---

**Document Version**: 1.1.0
**Created**: 2025-11-07
**Updated**: 2025-11-09
**Status**: ✅ COMPLETED — Alternative approach implemented (diffeqpy/NonlinearSolve)
**Completion Date**: 2025-11-09
**Effective Version**: v0.2.0
**Implementation**: See docs/PHASE2_IMPLEMENTATION_PLAN.md for actual approach taken
