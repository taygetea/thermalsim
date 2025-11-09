"""Integration tests for transient dynamics (Phase 3 Milestone 3)."""

import pytest
import numpy as np
from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.tank import SimpleTank
from thermal_sim.properties.coolprop_wrapper import FluidProperties


class TestTransientSolverBasic:
    """Basic transient solver functionality"""

    def test_solve_transient_exists(self):
        """solve_transient method should exist on ThermalGraph"""
        graph = ThermalGraph()
        assert hasattr(graph, 'solve_transient')

    def test_solve_transient_rejects_pure_algebraic(self):
        """solve_transient should reject systems with no differential variables"""
        from thermal_sim.components.heater import ConstantPressureHeater

        graph = ThermalGraph()
        heater = ConstantPressureHeater('heater', P=1e6, Q=10e6)
        graph.add_component(heater)

        with pytest.raises(ValueError, match="no differential variables"):
            graph.solve_transient(tspan=(0, 10))


class TestTankFilling:
    """Test transient tank filling dynamics"""

    @pytest.fixture
    def tank_system(self):
        """Create a simple tank system"""
        graph = ThermalGraph()

        tank = SimpleTank(
            'tank',
            A=10.0,
            P=2e5,
            P_downstream=1e5,
            C_v=0.01,
            level_initial=1.0
        )

        graph.add_component(tank)

        # Set inlet flow
        water = FluidProperties('Water')
        rho = water.density(2e5, 300.0)
        dP = 2e5 - 1e5
        mdot_out_ss = tank.C_v * np.sqrt(rho * dP)

        # Inlet flow 20% higher than steady-state outlet
        tank.ports['inlet'].mdot = mdot_out_ss * 1.2

        return graph, tank, mdot_out_ss

    def test_tank_filling_converges(self, tank_system):
        """Transient solver should converge for tank filling"""
        graph, tank, _ = tank_system

        # Provide initial condition to avoid steady-state solve
        y0 = tank.get_initial_state()

        result = graph.solve_transient(
            tspan=(0, 100),
            y0=y0,
            rtol=1e-6,
            atol=1e-8
        )

        assert result.success, f"Solver failed: {result.message}"
        assert len(result.t) > 10, "Should take multiple time steps"

    def test_tank_level_increases(self, tank_system):
        """Tank level should increase when inlet > outlet"""
        graph, tank, mdot_out_ss = tank_system

        # Initial condition
        y0 = tank.get_initial_state()

        result = graph.solve_transient(
            tspan=(0, 100),
            y0=y0
        )

        # Extract level history
        tank_state = graph.get_component_state(result, 'tank')
        level_initial = tank_state[0, 0]
        level_final = tank_state[0, -1]

        assert level_final > level_initial, \
            f"Level should increase: {level_initial} → {level_final}"

    def test_tank_level_change_rate(self, tank_system):
        """Level change rate should match mass balance"""
        graph, tank, mdot_out_ss = tank_system

        water = FluidProperties('Water')
        rho = water.density(2e5, 300.0)

        # Initial condition
        y0 = tank.get_initial_state()

        result = graph.solve_transient(
            tspan=(0, 50),
            y0=y0
        )

        # Extract states
        tank_state = graph.get_component_state(result, 'tank')
        level_history = tank_state[0, :]
        mdot_out_history = tank_state[1, :]

        # Compute average rate of level change
        dlevel_dt_actual = (level_history[-1] - level_history[0]) / (result.t[-1] - result.t[0])

        # Expected: dlevel/dt = (mdot_in - mdot_out) / (rho * A)
        mdot_in = tank.ports['inlet'].mdot
        mdot_out_avg = np.mean(mdot_out_history)
        dlevel_dt_expected = (mdot_in - mdot_out_avg) / (rho * tank.A)

        # Should match within 5%
        rel_error = abs(dlevel_dt_actual - dlevel_dt_expected) / abs(dlevel_dt_expected)
        assert rel_error < 0.05, \
            f"Level change rate mismatch: {dlevel_dt_actual:.4f} vs {dlevel_dt_expected:.4f}"

    def test_tank_draining(self):
        """Tank should drain when inlet < outlet"""
        graph = ThermalGraph()

        tank = SimpleTank(
            'tank',
            A=10.0,
            P=2e5,
            P_downstream=1e5,
            C_v=0.01,
            level_initial=2.0  # Start higher
        )

        graph.add_component(tank)

        # Compute steady-state outlet
        water = FluidProperties('Water')
        rho = water.density(2e5, 300.0)
        dP = 2e5 - 1e5
        mdot_out_ss = tank.C_v * np.sqrt(rho * dP)

        # Set inlet flow lower than outlet
        tank.ports['inlet'].mdot = mdot_out_ss * 0.8  # 20% less

        y0 = tank.get_initial_state()

        result = graph.solve_transient(
            tspan=(0, 100),
            y0=y0
        )

        tank_state = graph.get_component_state(result, 'tank')
        level_initial = tank_state[0, 0]
        level_final = tank_state[0, -1]

        assert level_final < level_initial, \
            f"Level should decrease: {level_initial} → {level_final}"


class TestTransientMetadata:
    """Test that transient results have correct metadata"""

    def test_result_has_time_array(self):
        """Result should have time array"""
        graph = ThermalGraph()
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)
        graph.add_component(tank)

        tank.ports['inlet'].mdot = 100.0
        y0 = tank.get_initial_state()

        result = graph.solve_transient(tspan=(0, 10), y0=y0)

        assert hasattr(result, 't')
        assert len(result.t) > 0
        assert result.t[0] == 0.0

    def test_result_has_state_trajectory(self):
        """Result should have state trajectory y"""
        graph = ThermalGraph()
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)
        graph.add_component(tank)

        tank.ports['inlet'].mdot = 100.0
        y0 = tank.get_initial_state()

        result = graph.solve_transient(tspan=(0, 10), y0=y0)

        assert hasattr(result, 'y')
        assert result.y.shape[0] == 2  # 2 state variables (level, mdot_out)
        assert result.y.shape[1] == len(result.t)

    def test_result_has_component_metadata(self):
        """Result should have component metadata for extraction"""
        graph = ThermalGraph()
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)
        graph.add_component(tank)

        tank.ports['inlet'].mdot = 100.0
        y0 = tank.get_initial_state()

        result = graph.solve_transient(tspan=(0, 10), y0=y0)

        assert hasattr(result, 'component_names')
        assert hasattr(result, 'component_offsets')
        assert hasattr(result, 'state_names')
        assert 'tank' in result.component_names


class TestTransientSolverOptions:
    """Test solver options and configuration"""

    def test_custom_initial_condition(self):
        """Should accept custom initial condition"""
        graph = ThermalGraph()
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01, level_initial=1.0)
        graph.add_component(tank)

        tank.ports['inlet'].mdot = 100.0

        # Custom IC with different level
        y0_custom = np.array([2.5, 100.0])  # level=2.5 m

        result = graph.solve_transient(tspan=(0, 10), y0=y0_custom)

        tank_state = graph.get_component_state(result, 'tank')
        level_initial = tank_state[0, 0]

        assert abs(level_initial - 2.5) < 1e-6, "Should use custom initial condition"

    def test_time_point_specification(self):
        """Should support array of specific time points"""
        graph = ThermalGraph()
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)
        graph.add_component(tank)

        tank.ports['inlet'].mdot = 100.0
        y0 = tank.get_initial_state()

        t_points = np.array([0, 1, 2, 5, 10, 20])

        result = graph.solve_transient(tspan=t_points, y0=y0)

        # Should evaluate at specified points
        assert len(result.t) == len(t_points)
        np.testing.assert_allclose(result.t, t_points, rtol=1e-10)

    def test_tolerance_options(self):
        """Should accept rtol and atol options"""
        graph = ThermalGraph()
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)
        graph.add_component(tank)

        tank.ports['inlet'].mdot = 100.0
        y0 = tank.get_initial_state()

        # Should not raise error
        result = graph.solve_transient(
            tspan=(0, 10),
            y0=y0,
            rtol=1e-8,
            atol=1e-10
        )

        assert result.success
