"""Unit tests for graph topology and assembly."""

import pytest
from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.pump import Pump


def test_graph_creation():
    """Test empty graph instantiation"""
    graph = ThermalGraph()
    assert len(graph.components) == 0
    assert len(graph.connections) == 0


def test_add_component():
    """Test adding components"""
    graph = ThermalGraph()
    heater = ConstantPressureHeater('h1', P=1e6, Q=1e6)

    graph.add_component(heater)
    assert len(graph.components) == 1
    assert graph.components[0] is heater


def test_duplicate_component_name():
    """Should not allow duplicate names"""
    graph = ThermalGraph()
    h1 = ConstantPressureHeater('heater', P=1e6, Q=1e6)
    h2 = ConstantPressureHeater('heater', P=2e6, Q=2e6)

    graph.add_component(h1)
    with pytest.raises(ValueError, match="already exists"):
        graph.add_component(h2)


def test_connect_compatible_ports():
    """Should allow connecting compatible ports"""
    graph = ThermalGraph()
    h1 = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    p1 = Pump('p1', efficiency=0.8, P_out=2e6)

    graph.add_component(h1)
    graph.add_component(p1)

    # Should not raise
    graph.connect(h1.outlet, p1.inlet)
    assert len(graph.connections) == 1


def test_connect_incompatible_ports():
    """Should reject incompatible port connections"""
    graph = ThermalGraph()
    h1 = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    h2 = ConstantPressureHeater('h2', P=1e6, Q=1e6)

    graph.add_component(h1)
    graph.add_component(h2)

    # Two outlets should not connect
    with pytest.raises(TypeError):
        graph.connect(h1.outlet, h2.outlet)


def test_simple_loop_assembly():
    """Test that a simple 2-component loop assembles"""
    graph = ThermalGraph()
    h1 = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    p1 = Pump('p1', efficiency=0.8, P_out=2e6)

    graph.add_component(h1)
    graph.add_component(p1)
    graph.connect(h1.outlet, p1.inlet)
    graph.connect(p1.outlet, h1.inlet)

    residual_func, y0 = graph.assemble()

    assert callable(residual_func)
    assert len(y0) == 5  # heater(2) + pump(3)

    # Test that residual is callable
    res = residual_func(0, y0)
    assert len(res) == 5
