"""Debug script to check initial residuals."""

from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.components.pump import Pump
import numpy as np

# Create system graph
graph = ThermalGraph()

# Define operating conditions
P_high = 10e6
P_low = 10e3

# Create components
boiler = ConstantPressureHeater('boiler', P=P_high, Q=100e6)
turbine = Turbine('turbine', efficiency=0.85, P_out=P_low)
condenser = ConstantPressureHeater('condenser', P=P_low, Q=-80e6)
pump = Pump('pump', efficiency=0.80, P_out=P_high)

# Add components to graph
for comp in [boiler, turbine, condenser, pump]:
    graph.add_component(comp)

# Connect in cycle
graph.connect(boiler.outlet, turbine.inlet)
graph.connect(turbine.outlet, condenser.inlet)
graph.connect(condenser.outlet, pump.inlet)
graph.connect(pump.outlet, boiler.inlet)

# Assemble and check initial residuals
residual_func, y0 = graph.assemble()

print("Initial state vector:")
print(y0)
print("\nInitial residuals:")
try:
    res = residual_func(0, y0)
    print(res)
    print(f"\nMax abs residual: {np.max(np.abs(res))}")
    print(f"Contains NaN: {np.any(np.isnan(res))}")
    print(f"Contains Inf: {np.any(np.isinf(res))}")
except Exception as e:
    print(f"Error computing residuals: {e}")
    import traceback
    traceback.print_exc()
