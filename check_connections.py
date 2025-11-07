"""Check if port connections are properly shared."""

from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.components.pump import Pump

# Create system
graph = ThermalGraph()
P_high, P_low = 10e6, 10e3

boiler = ConstantPressureHeater('boiler', P=P_high, Q=100e6)
turbine = Turbine('turbine', efficiency=0.85, P_out=P_low)
condenser = ConstantPressureHeater('condenser', P=P_low, Q=-80e6)
pump = Pump('pump', efficiency=0.80, P_out=P_high)

for comp in [boiler, turbine, condenser, pump]:
    graph.add_component(comp)

print("BEFORE connections:")
print(f"Boiler outlet: {id(boiler.ports['outlet'])}")
print(f"Turbine inlet: {id(turbine.ports['inlet'])}")
print(f"Are they the same object? {boiler.ports['outlet'] is turbine.ports['inlet']}")

# Connect
graph.connect(boiler.outlet, turbine.inlet)

print("\nAFTER connection:")
print(f"Boiler outlet: {id(boiler.ports['outlet'])}")
print(f"Turbine inlet: {id(turbine.ports['inlet'])}")
print(f"Are they the same object? {boiler.ports['outlet'] is turbine.ports['inlet']}")

# Update boiler outlet and check if turbine sees it
boiler.ports['outlet'].h = 9.999e6
print(f"\nAfter setting boiler.ports['outlet'].h = 9.999e6:")
print(f"  boiler.ports['outlet'].h = {boiler.ports['outlet'].h}")
print(f"  turbine.ports['inlet'].h = {turbine.ports['inlet'].h}")
