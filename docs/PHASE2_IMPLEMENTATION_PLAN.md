# Phase 2 Implementation Plan: diffeqpy Integration

**Status:** Ready to implement
**Date:** 2025-11-08
**Context:** Phase 1 complete, transitioning from solve_sequential() to proper solver

---

## Background: The Journey to diffeqpy

**Original Plan (IDA_TRANSITION.md):** Use SUNDIALS IDA via scikits.odes

**What Happened:**
1. Attempted `pip install scikits.odes` → compilation failure (SUNDIALS 6.4.1 API changed, headers missing)
2. Tried building from source → system-dependent, fragile, version mismatches
3. Explored diffeqpy as alternative → installed successfully
4. Tested diffeqpy IDA → Python/Julia type conversion bugs (PyArray issues)
5. **Insight:** For steady-state, we don't need IDA at all - use NonlinearSolve.jl!

**What We Set Up:**
- ✓ Julia 1.12.1 installed (via jill.py, auto-managed)
- ✓ diffeqpy 2.5.4 installed (via uv)
- ✓ DifferentialEquations.jl precompiled (~6 min first time)
- ✓ NonlinearSolve.jl available (no type conversion issues)

## Executive Summary

**Problem:** Originally planned to use SUNDIALS IDA via scikits.odes for Phase 2, but:
- scikits.odes has compilation issues (SUNDIALS 6.4.1 API mismatch)
- diffeqpy IDA has Python/Julia type conversion bugs

**Solution:** Use diffeqpy's **NonlinearSolve.jl** instead of IDA for steady-state problems.

**Why this works:**
- Steady-state is always solving f(x) = 0 (pure nonlinear, not DAE)
- NonlinearSolve is the RIGHT tool for this job
- IDA is for transient dynamics M(x)·dx/dt = f(x,t) (future Phase 1+)
- Creates ZERO tech debt - both solvers will coexist

---

## Architecture Decision

### Current Situation (Phase 1 Complete)

**Components:**
- All 4 implemented: ConstantPressureHeater, Turbine, Pump, Pipe
- All variables are `Variable.kind='algebraic'` (no differential equations yet)
- Residual interface: `residual(state, ports, t) -> np.ndarray`

**Existing Solvers:**
1. `solve()` - scipy.integrate.solve_ivp (transient ODE)
2. `solve_steady_state()` - scipy.optimize.root (steady-state)
3. `solve_sequential()` - **TEMPORARY HACK** (Rankine only, 4 components)

**Current Usage:**
- examples/rankine_cycle.py line 80: uses `solve_sequential()`
- tests/integration/test_rankine.py: tests both `solve()` and `solve_sequential()`
- TODO_IDA comment at line 77: "Replace with solve_steady_state() once SUNDIALS IDA integrated"

### Long-Term Vision (from ARCHITECTURE.md)

**Mathematical Foundation:**
```
M(x) · ẋ = f(x, t)
```

**Phases:**
- Phase 0-1 (MVP): Single-phase, all algebraic → **COMPLETED**
- Phase 1: Two-phase + controls → **WILL ADD DIFFERENTIAL EQUATIONS**
- Phase 2: Multirate + MPC → **FULL DAE CAPABILITY**

### The Key Insight: Two Different Problems

#### Problem 1: Steady-State (dx/dt = 0)
```
M(x) · ẋ = f(x, t)
→  M(x) · 0 = f(x, t)
→  f(x, t) = 0         # Pure nonlinear system
```
**Solver:** NonlinearSolve.jl (or scipy.optimize.root)
**When:** Initialization, equilibrium finding, current MVP

#### Problem 2: Transient Dynamics (dx/dt ≠ 0)
```
M(x) · ẋ = f(x, t)     # Full DAE
```
**Solver:** IDA, OrdinaryDiffEq.jl, or scipy BDF
**When:** Phase 1+ (after adding differential equations)

### Why NonlinearSolve Creates Zero Tech Debt

**Even with differential equations in Phase 1+:**

1. **Steady-state initialization** will ALWAYS use NonlinearSolve
   - Before transient: find f(x,0) = 0
   - After transient: find equilibrium f(x,∞) = 0

2. **Different solvers for different jobs:**
   - `solve_steady_state()` → NonlinearSolve (now and forever)
   - `solve_transient()` → DAE solver (Phase 1+, to be added)

3. **Component interface already supports both:**
   - `Variable.kind = 'differential'` → future dynamics
   - `Variable.kind = 'algebraic'` → constraints
   - Interface is already DAE-ready

---

## Implementation Plan

### Multi-Solver Architecture

```python
class ThermalGraph:

    def solve_steady_state(self, method='diffeqpy', **opts):
        """
        Solve for steady-state: f(x) = 0

        Always uses nonlinear solver (even when system has diff eqs).
        This is THE steady-state solver for all phases.

        Args:
            method: 'diffeqpy' or 'scipy' (fallback)
            **opts: Solver-specific options

        Returns:
            Result object with .x, .success, .y (compatible with existing code)
        """
        if method == 'diffeqpy':
            return self._solve_steady_diffeqpy(**opts)
        elif method == 'scipy':
            return self._solve_steady_scipy(**opts)  # Current implementation
        else:
            raise ValueError(f"Unknown method: {method}")

    def _solve_steady_diffeqpy(self, **opts):
        """Implementation using Julia's NonlinearSolve.jl"""
        from diffeqpy import de

        # Use existing assemble() method
        residual_func, y0 = self.assemble()

        # Wrapper for NonlinearSolve signature: f(u, p) -> residual
        def nonlinear_func(u, p):
            return residual_func(0.0, u)  # t=0 for steady-state

        # Create and solve problem
        prob = de.NonlinearProblem(nonlinear_func, y0)
        sol = de.solve(prob, **opts)

        # Package result (compatible with existing interface)
        result = type('Result', (), {})()
        result.x = np.array(sol.u)
        result.success = (sol.retcode.name == 'Success')
        result.message = str(sol.retcode)
        result.fun = nonlinear_func(result.x, None)

        # For get_component_state compatibility
        result.y = result.x.reshape(-1, 1)
        result.t = np.array([0.0])

        # Attach metadata (from existing code)
        result.component_names = [c.name for c in self.components]
        result.component_offsets = self._component_offsets.copy()
        result.state_names = {}
        for comp in self.components:
            result.state_names[comp.name] = comp.get_state_names()

        # Trigger final residual to update ports
        _ = residual_func(0.0, result.x)

        return result

    def _solve_steady_scipy(self, **opts):
        """Current implementation - keep as fallback"""
        # Existing solve_steady_state() code (lines 241-315 in graph.py)
        ...

    def solve_transient(self, tspan, method='scipy', **opts):
        """
        FUTURE: Solve dynamics M(x)·dx/dt = f(x,t)

        Phase 1+ implementation when differential equations are added.
        For now, raises NotImplementedError.
        """
        raise NotImplementedError(
            "Transient solver not yet implemented. "
            "Use solve_steady_state() for equilibrium problems."
        )
```

### Step-by-Step Implementation

#### Step 1: Test NonlinearSolve ✓
- Created: `test_nonlinearsolve.py`
- Tests simple 2-variable and 10-variable systems
- Confirms no Python/Julia type issues

#### Step 2: Implement _solve_steady_diffeqpy()
File: `thermal_sim/core/graph.py`
- Add import: `from diffeqpy import de` (lazy import)
- Add method around line 316 (after current solve_steady_state)
- Use existing `assemble()` method
- Return result compatible with existing interface

#### Step 3: Update solve_steady_state()
File: `thermal_sim/core/graph.py`
- Change signature to accept `method` parameter
- Default to 'diffeqpy' (or 'scipy' for safety during transition)
- Rename current implementation to `_solve_steady_scipy()`
- Add method dispatch

#### Step 4: Delete solve_sequential()
File: `thermal_sim/core/graph.py`
- Remove entire method (lines 317-551)
- This is the TEMPORARY hack for 4-component Rankine only

#### Step 5: Update Example
File: `examples/rankine_cycle.py`
- Line 80: Change from `solve_sequential()` to `solve_steady_state()`
- Remove TODO_IDA comment (line 77-79)

#### Step 6: Update Tests
File: `tests/integration/test_rankine.py`
- Lines 139-240: Remove all `test_rankine_sequential_*` tests
- Line 225-239: Remove `test_sequential_solver_only_supports_4_components`
- Keep tests for `solve()` (transient ODE solver)

#### Step 7: Remove TODO_IDA Comments
Search entire codebase for "TODO_IDA" and remove/update comments.

#### Step 8: Update requirements.txt
Already has: `scikits.odes>=2.7.0`
Change to: `diffeqpy>=2.5.0`

---

## Testing Strategy

### Unit Tests (Existing)
- `tests/unit/test_graph.py` - graph topology tests
- `tests/unit/test_component.py` - component interface tests
- Should all pass unchanged

### Integration Tests
File: `tests/integration/test_rankine.py`

**Keep (transient solver):**
- `test_rankine_cycle_converges()` - uses solve()
- `test_rankine_cycle_efficiency()` - uses solve()
- `test_rankine_energy_balance()` - uses solve()
- `test_rankine_mass_conservation()` - uses solve()

**Remove (sequential solver):**
- `test_rankine_sequential_converges()`
- `test_rankine_sequential_efficiency()`
- `test_rankine_sequential_energy_balance()`
- `test_rankine_sequential_mass_conservation()`
- `test_sequential_solver_only_supports_4_components()`

**Add (new steady-state solver):**
```python
def test_rankine_steady_state_diffeqpy():
    """Test solve_steady_state with diffeqpy backend"""
    graph = build_rankine_cycle_sequential()  # Use same setup

    result = graph.solve_steady_state(method='diffeqpy')

    assert result.success
    # Check efficiency, energy balance, etc.
    ...

def test_rankine_steady_state_scipy():
    """Test solve_steady_state with scipy backend (fallback)"""
    graph = build_rankine_cycle_sequential()

    result = graph.solve_steady_state(method='scipy')

    assert result.success
    ...
```

### Verification Checklist
- [ ] `pytest tests/unit/` passes
- [ ] `pytest tests/integration/` passes
- [ ] `python examples/rankine_cycle.py` runs successfully
- [ ] Efficiency in range 28-35%
- [ ] Energy balance < 1% error
- [ ] Mass conservation < 0.1% error

---

## Dependencies

### Currently Installed
- Julia 1.12.1 (installed in ~/.local/bin/julia)
- diffeqpy 2.5.4 (via uv pip install)
- Julia packages precompiled (~6 minutes first time)

### Required Additions
None - everything already installed!

### Known Issues
- diffeqpy IDA has Python/Julia type conversion bugs → Don't use IDA
- scikits.odes won't compile against SUNDIALS 6.4.1 → Use diffeqpy instead
- Both issues avoided by using NonlinearSolve.jl

---

## Migration Path (Zero Tech Debt)

### Phase 0-1 (NOW - This Implementation)
**Components:** All `Variable.kind='algebraic'`
**Solvers:**
- `solve_steady_state(method='diffeqpy')` → NonlinearSolve.jl
- `solve_steady_state(method='scipy')` → scipy.optimize.root (fallback)
- `solve()` → scipy.integrate.solve_ivp BDF (transient ODE)

**Delete:** `solve_sequential()` (temporary hack)

### Phase 1 (Two-Phase + Controls - Future)
**Components:** Add `Variable.kind='differential'` for:
- Tank levels (dh/dt)
- Valve positions
- Controller states

**Solvers:**
- `solve_steady_state()` → **UNCHANGED** (still NonlinearSolve for initialization)
- `solve_transient(tspan)` → **NEW** (DAE solver for dynamics)
  - Option A: diffeqpy DAE solvers (if type bridge improves)
  - Option B: scipy.integrate.solve_ivp BDF (already works)
  - Option C: Custom wrapper to Julia

### Phase 2 (Multirate + MPC - Future)
**Solvers:**
- `solve_steady_state()` → **UNCHANGED**
- `solve_transient()` → Enhanced with multirate scheduling

**No changes to steady-state solver across all phases!**

---

## File Modifications Summary

### Files to Modify
1. `thermal_sim/core/graph.py`:
   - Add `_solve_steady_diffeqpy()` method
   - Update `solve_steady_state()` with method parameter
   - Rename current impl to `_solve_steady_scipy()`
   - Delete `solve_sequential()` entirely

2. `examples/rankine_cycle.py`:
   - Line 80: `solve_sequential()` → `solve_steady_state()`
   - Remove TODO_IDA comment (lines 77-79)

3. `tests/integration/test_rankine.py`:
   - Remove sequential solver tests (lines 139-240)
   - Add diffeqpy and scipy steady-state tests

4. `requirements.txt`:
   - Update: `scikits.odes>=2.7.0` → `diffeqpy>=2.5.0`

### Files to Create
1. `test_nonlinearsolve.py` - Verification that NonlinearSolve works ✓

### Files to Update (Documentation)
1. `docs/ARCHITECTURE.md` - Update solver table
2. `docs/IDA_TRANSITION.md` - Note we used NonlinearSolve instead
3. `CLAUDE.md` - Update quick reference
4. `README.md` - Update installation/usage

---

## Expected Results

### After Implementation
```bash
$ python examples/rankine_cycle.py
Solving Rankine cycle (using diffeqpy NonlinearSolve)...
✓ Solution converged (residual norm: 1.23e-10)

Results:
  Turbine power: 79.15 MW
  Pump power: 1.02 MW
  Net power: 78.13 MW
  Thermal efficiency: 31.0%
✓ Efficiency in expected range (30-40%)
```

### Performance Expectations
- NonlinearSolve.jl should be 2-5x faster than scipy.optimize.root
- First run: ~2-3 seconds (Julia JIT compilation)
- Subsequent runs: <0.5 seconds

---

## Rollback Plan

If diffeqpy NonlinearSolve doesn't work:

1. Set default to `method='scipy'` in `solve_steady_state()`
2. Keep `_solve_steady_diffeqpy()` but mark as experimental
3. scipy.optimize.root is proven to work (Phase 1 tested)

---

## Next Session Checklist

1. [ ] Run `python test_nonlinearsolve.py` to verify NonlinearSolve works
2. [ ] Implement `_solve_steady_diffeqpy()` in graph.py
3. [ ] Update `solve_steady_state()` with method parameter
4. [ ] Delete `solve_sequential()`
5. [ ] Update examples/rankine_cycle.py
6. [ ] Update tests/integration/test_rankine.py
7. [ ] Run full test suite: `pytest`
8. [ ] Run example: `python examples/rankine_cycle.py`
9. [ ] Update documentation
10. [ ] Commit changes with message: "Phase 2: Replace solve_sequential with diffeqpy NonlinearSolve"

---

## References

- ARCHITECTURE.md Section 3: Mathematical foundation (M(x)·ẋ = f(x,t))
- ARCHITECTURE.md Section 16: Development roadmap
- IDA_TRANSITION.md: Original IDA transition plan (superseded by this doc)
- Current graph.py: lines 241-315 (solve_steady_state), 317-551 (solve_sequential)

---

**Document Status:** Ready for implementation
**Confidence Level:** High - NonlinearSolve is the RIGHT tool for steady-state
**Risk Level:** Low - scipy fallback available, zero tech debt
