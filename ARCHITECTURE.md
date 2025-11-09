THERMAL SYSTEM SIMULATOR — ARCHITECTURE SPECIFICATION

Version: 1.0
Date: 2025-11-07
Maintainers: Olivia (human overseer), GPT (Systems Architecture), Claude (Numerical Physics)
Purpose: Defines the canonical design of the Thermal System Simulator — an extensible, high-fidelity, numerically robust framework for thermo-hydraulic and energy-system simulation.

This document describes what the system is and why it is structured this way.
Implementation details (the how) live in MVP_SPEC.md.

1 · Overview
1.1 Goals

Model coupled thermal–hydraulic systems from small Rankine cycles to full power-plant loops.

Support both academic research and industrial prototyping.

Unify physical transparency, numerical stability, and software extensibility.

1.2 Philosophy
Principle	Meaning
Layered design	Physics, topology, and numerics are distinct modules.
Declarative topology	Systems are described, not hand-wired.
Conservation first	Mass / energy balance enforced automatically.
DAE-native	Differential–algebraic formulation as first-class citizen.
Hybrid Jacobians	Analytic for built-ins, AD for user code.
Deterministic core + UQ-ready	Core solves deterministically; data structures natively support ensembles.
Smooth dynamics	All discontinuities regularized for solver stability.
2 · Layered Architecture
User / DSL
   ↓
Graph Kernel  ─── topology, ports, conservation
   ↓
Component Library ─── physics models
   ↓
Solver Backend ─── DAE / ODE / multirate
   ↓
Numerics & Utilities ─── properties, caching, units


Each layer is replaceable and communicates only through defined interfaces.

3 · Mathematical Foundation
3.1 Global Form

    M(x) · ẋ = f(x, t)

where M(x) may be singular (algebraic equations).
M, f are assembled automatically from components.

3.2 Component Form

    f_i(x_i, ẋ_i, p_i, t) = 0

Each component contributes local residuals; the graph enforces inter-component constraints.


3.3 Port Conservation (Implementation Note)
# When graph.connect(A.outlet, B.inlet) is called,
# both components share the same Port object.
# Updates by A are immediately visible to B.
# This enforces conservation by reference sharing.

4 · Typed Port System
@dataclass
class MassPort:
    mdot: float      # kg/s
    h: float         # J/kg
    P: float         # Pa
    direction: Literal['in','out']

@dataclass
class HeatPort:
    Q: float         # W
    T: float         # K
    direction: Literal['in','out']


Only matching port types may connect.

Opposing directions required.

Junctions share the same object to guarantee mass/energy consistency.

5 · Component Interface
class Component(ABC):
    name: str
    ports: dict[str,Port]
    provides_jacobian: bool = False

    @abstractmethod
    def get_variables(self) -> list[Variable]: ...
    @abstractmethod
    def get_initial_state(self) -> np.ndarray: ...
    @abstractmethod
    def residual(self, x, dxdt, ports, t) -> np.ndarray: ...
    def jacobian(self, x, dxdt, ports, t) -> np.ndarray | None: ...

Variable Descriptor
@dataclass
class Variable:
    name: str
    kind: Literal['differential','algebraic']
    initial: float


Components may be differential, algebraic, or mixed.

6 · Graph Kernel
Responsibilities

Manage components / connections.

Validate port compatibility.

Assemble global state vector and residuals.

Provide unified solver interface.

Assembly Outline
def assemble(self):
    offsets, y0 = build_state_index()
    def residual_func(t, y, ydot):
        self._update_ports(y)
        res=[]
        for comp in self.components:
            x,dx = slice_state(y, ydot, comp)
            res.append(comp.residual(x,dx,comp.ports,t))
        return np.concatenate(res)
    return residual_func, y0

7 · Solver Layer
7.1 Primary Backends (Phase 2 Complete)
Type	Library	Notes
Steady-state (primary)	scipy.optimize.root	Fast simultaneous solver with sequential init
Steady-state (robust)	diffeqpy/NonlinearSolve.jl	Julia backend for stiff systems
Initialization	Sequential propagation	Provides basin-of-attraction for simultaneous methods
ODE (transient)	SciPy solve_ivp(method="BDF")	For transient dynamics
Multirate	Custom scheduler	Future phase

✅ Phase 2 Implementation (Current):

The simulator uses a multi-backend steady-state solver with automatic initialization:

**solve_steady_state(backend='scipy')**:
1. Try scipy.optimize.root direct → often fails from default initial guess
2. On failure: use solve_sequential() to generate thermodynamically consistent initial guess
3. Retry scipy with sequential initial conditions → typically succeeds
4. On failure: return sequential result as fallback

**Result**: "scipy (sequential init)" - combines accuracy of simultaneous solving
with robustness of sequential initialization.

**Architecture Rationale**:
- Sequential solver is PERMANENT (not temporary) - provides initialization service
- Simultaneous solvers are more accurate when they converge
- Sequential propagation ensures thermodynamically feasible starting point
- Graceful degradation: scipy → sequential init → diffeqpy → sequential fallback

See docs/PHASE2_IMPLEMENTATION_PLAN.md for implementation details.

7.2 Failure Recovery

Reduce timestep

Switch to implicit Euler

Dump checkpoint

Abort with diagnostics

8 · Numerical Governance
8.1 Regularization
def smooth_step(x,x0,width):
    return 0.5*(1+np.tanh((x-x0)/width))
# width ≈ 0.01 × characteristic scale

8.2 Conservation Audits

Each step checks:

mass_err  = Σṁ_in - Σṁ_out
energy_err = ΣQ̇ + ΣẆ


Drift > tolerance ⇒ warning / correction.

8.3 Diagnostics

Log residual norms, Jacobian condition numbers, adaptive step history.

9 · Thermophysical Properties
class FluidProperties(ABC):
    def enthalpy(self,P,T): ...
    def temperature(self,P,h): ...
    def density(self,P,T): ...
    def entropy(self,P,h): ...


Default: CoolProp wrapper with LRU caching.
Future: analytic fits, GPU vectorization, alternate EOS tables.

10 · Jacobian Policy
Component Type	Strategy	Implementation
Core library	Analytic	provides_jacobian = True
User extension	AD (JAX / CasADi)	provides_jacobian = False
Fallback	Finite difference	solver-side
11 · Units & Dimensional Analysis
Mode	Behaviour
Development	pint validation active
Production	pure SI floats, zero overhead
CI Test	run both modes, assert bit-level consistency
12 · Uncertainty & Sensitivity

Design for Uncertainty Analysis
The core remains deterministic but supports efficient UQ:

Result arrays carry an explicit sample axis for ensembles.

Property and residual evaluations are vectorized.

Future: automatic adjoint sensitivities via JAX.

External wrappers implement sampling (Monte Carlo, LHS, Sobol).

13 · Extensibility Model

Components register dynamically (entry points or plugin loader).

New fluids, solvers, or schedulers can be added independently.

Declarative DSL or YAML schema maps to identical runtime objects.

Example:

components:
  - type: Boiler
    id: boiler
    params: {P: 10e6, Q: 100e6}
connections:
  - from: boiler.outlet
    to: turbine.inlet

14 · Validation & Testing

Unit tests: component residuals, property wrappers.

Integration tests: canonical Rankine cycle → η ≈ 30–35 %.

Regression: long-transient energy closure < 1 %.

Benchmark: compare against RELAP5 / TRACE data.

15 · Data & Visualization

Output: HDF5 hierarchical structure
/component/state_vars, /global/efficiency

Realtime: optional ZeroMQ / WebSocket telemetry hooks.

16 · Development Roadmap
Phase	Features	Milestone
0	Single-phase Rankine	validate architecture
1	Two-phase + controls	full plant demo
2	Multirate + MPC	scalable transient solver
3	Fusion / cryo	exotic domains
4	GPU / distributed	HPC integration
17 · Repository Layout
thermal_sim/
├─ core/
│  ├─ port.py
│  ├─ variable.py
│  ├─ component.py
│  ├─ graph.py
│  ├─ solver_interface.py
│  └─ scheduler.py
├─ components/
│  ├─ boiler.py  turbine.py  condenser.py  pump.py  pipe.py
├─ properties/
│  └─ coolprop_wrapper.py
├─ tests/
│  ├─ unit/  integration/  regression/
└─ examples/
   ├─ rankine_cycle.py  pwr_loop.py  cryo_cooler.py

18 · Development Guidelines

Language: Python 3.11+

Dependencies: numpy, scipy, CoolProp, jax (optional), pytest

Style: PEP-8, NumPy-style docstrings

Coverage: ≥ 90 % core modules

Performance target: ≥ 10× real-time for Rankine simulation

Docs: Sphinx-based API + Conceptual Guide

19 · Deliverables

Core framework implementing this architecture.

Verified component library.

Validation notebooks & benchmarks.

Automated test suite.

Full API / design documentation.

20 · Summary

This specification defines the enduring architecture of the Thermal System Simulator:

Graph-assembled, DAE-centric, physically conservative.

Architecturally clean and future-ready.

Numerically disciplined and verifiable.

Extensible from educational examples to industrial-scale analysis.

Treat this document as the constitution of the codebase;
implementation guides (like MVP_SPEC.md) interpret it for specific builds.
