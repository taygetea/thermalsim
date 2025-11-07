"""Unit tests for port types and compatibility checking."""

import pytest
from thermal_sim.core.port import MassFlowPort, HeatPort


def test_mass_port_creation():
    """Test basic MassFlowPort instantiation"""
    port = MassFlowPort('test', 'in')
    assert port.name == 'test'
    assert port.direction == 'in'
    assert port.mdot == 0.0
    assert port.h == 0.0
    assert port.P == 0.0


def test_mass_port_compatibility():
    """Inlet and outlet should be compatible"""
    inlet = MassFlowPort('in', 'in')
    outlet = MassFlowPort('out', 'out')

    assert inlet.compatible_with(outlet)
    assert outlet.compatible_with(inlet)


def test_mass_port_incompatibility_same_direction():
    """Two inlets or two outlets should not connect"""
    inlet1 = MassFlowPort('in1', 'in')
    inlet2 = MassFlowPort('in2', 'in')

    assert not inlet1.compatible_with(inlet2)


def test_mass_heat_port_incompatibility():
    """Mass and heat ports should not connect"""
    mass_port = MassFlowPort('m', 'in')
    heat_port = HeatPort('h', 'in')

    assert not mass_port.compatible_with(heat_port)


def test_heat_port_compatibility():
    """Heat ports should connect with opposite directions"""
    heat_in = HeatPort('in', 'in')
    heat_out = HeatPort('out', 'out')

    assert heat_in.compatible_with(heat_out)
