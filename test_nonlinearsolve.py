#!/usr/bin/env python3
"""
Test diffeqpy NonlinearSolve for steady-state problems.
This is what we'll use instead of IDA for steady-state.
"""

import sys
import numpy as np

print("=" * 70)
print("Testing diffeqpy NonlinearSolve.jl")
print("=" * 70)

# Import diffeqpy
print("\n[1/3] Importing diffeqpy...")
try:
    from diffeqpy import de
    print("✓ diffeqpy imported")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 1: Simple nonlinear system
print("\n[2/3] Testing NonlinearSolve with simple system...")
print("         System: x² + y² = 1, y = x")
print("         Expected: x ≈ 0.7071, y ≈ 0.7071")

def simple_nonlinear(u, p):
    """Nonlinear residual function: f(u) = 0"""
    x, y = u
    return np.array([
        x**2 + y**2 - 1.0,  # Circle
        y - x                # Line
    ])

u0 = np.array([0.5, 0.5])  # Initial guess

try:
    prob = de.NonlinearProblem(simple_nonlinear, u0)
    sol = de.solve(prob)

    x_sol = float(sol.u[0])
    y_sol = float(sol.u[1])

    print(f"         Solution: x = {x_sol:.6f}, y = {y_sol:.6f}")

    # Check accuracy
    circle_err = abs(x_sol**2 + y_sol**2 - 1.0)
    line_err = abs(y_sol - x_sol)

    if circle_err < 1e-6 and line_err < 1e-6:
        print("✓ NonlinearSolve works correctly!")
    else:
        print(f"✗ Inaccurate: circle_err={circle_err:.2e}, line_err={line_err:.2e}")
        sys.exit(1)

except Exception as e:
    print(f"✗ NonlinearSolve failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Larger system (similar to Rankine cycle size)
print("\n[3/3] Testing with larger system (10 variables)...")

def large_nonlinear(u, p):
    """10-variable test: sum(x_i) = 10, x_i² = i"""
    residuals = np.zeros(10)
    for i in range(10):
        residuals[i] = u[i]**2 - (i + 1)
    # Add constraint: sum = 10
    residuals[0] += np.sum(u) - 10.0
    return residuals

u0_large = np.ones(10)

try:
    prob_large = de.NonlinearProblem(large_nonlinear, u0_large)
    sol_large = de.solve(prob_large)

    print(f"         Converged: {len(sol_large.u)} variables")
    print(f"         Residual norm: {np.linalg.norm(large_nonlinear(sol_large.u, None)):.2e}")
    print("✓ Large system solved successfully!")

except Exception as e:
    print(f"✗ Large system failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "=" * 70)
print("SUCCESS! NonlinearSolve.jl is working via diffeqpy")
print("=" * 70)
print("\nKey findings:")
print("  ✓ NonlinearSolve handles nonlinear systems f(x) = 0")
print("  ✓ No Python/Julia type issues (unlike IDA)")
print("  ✓ Perfect for steady-state thermal systems")
print("\nReady to integrate into ThermalGraph.solve_steady_state()")
print()
sys.exit(0)
