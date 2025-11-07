"""Check the solution quality."""

from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.components.pump import Pump
from scipy.optimize import root
import numpy as np

# Create system
graph = ThermalGraph()
P_high, P_low = 10e6, 10e3

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

residual_func, y0 = graph.assemble()

# Solve
result = root(lambda y: residual_func(0, y), y0, method='hybr')

print("Solution converged:", result.success)
print("\nFinal state:")
print(result.x)
print("\nFinal residuals:")
print(residual_func(0, result.x))
print(f"\nMax abs residual: {np.max(np.abs(residual_func(0, result.x)))}")

# Check port values after solution
print("\n=== Port Values After Solution ===")
# Trigger residual evaluation to update ports
_ = residual_func(0, result.x)

print(f"\nBoiler:")
print(f"  Inlet: P={boiler.ports['inlet'].P/1e6:.2f} MPa, h={boiler.ports['inlet'].h/1e6:.3f} MJ/kg")
print(f"  Outlet: P={boiler.ports['outlet'].P/1e6:.2f} MPa, h={boiler.ports['outlet'].h/1e6:.3f} MJ/kg")

print(f"\nTurbine:")
print(f"  Inlet: P={turbine.ports['inlet'].P/1e6:.2f} MPa, h={turbine.ports['inlet'].h/1e6:.3f} MJ/kg")
print(f"  Outlet: P={turbine.ports['outlet'].P/1e3:.2f} kPa, h={turbine.ports['outlet'].h/1e6:.3f} MJ/kg")

print(f"\nCondenser:")
print(f"  Inlet: P={condenser.ports['inlet'].P/1e3:.2f} kPa, h={condenser.ports['inlet'].h/1e6:.3f} MJ/kg")
print(f"  Outlet: P={condenser.ports['outlet'].P/1e3:.2f} kPa, h={condenser.ports['outlet'].h/1e6:.3f} MJ/kg")

print(f"\nPump:")
print(f"  Inlet: P={pump.ports['inlet'].P/1e3:.2f} kPa, h={pump.ports['inlet'].h/1e6:.3f} MJ/kg")
print(f"  Outlet: P={pump.ports['outlet'].P/1e6:.2f} MPa, h={pump.ports['outlet'].h/1e6:.3f} MJ/kg")
