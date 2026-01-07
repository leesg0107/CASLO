//! Main simulation runner
//!
//! Orchestrates the simulation of the complete multi-quadrotor
//! cable-suspended load system.

use nalgebra::{Vector3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::dynamics::{
    SystemState, SystemDynamics, SystemParams, SystemInput,
    LoadState, LoadParams, CableState, CableInput, MultiCableState,
    QuadrotorInput, CableParams, QuadrotorParams,
};

use super::{SimConfig, SystemSensors};

/// Simulation output for one timestep
#[derive(Debug, Clone)]
pub struct SimStep {
    /// Simulation time [s]
    pub time: f64,
    /// System state
    pub state: SystemState,
    /// Control inputs applied
    pub inputs: SystemInput,
}

/// Simulation history
#[derive(Debug, Clone, Default)]
pub struct SimHistory {
    /// Time stamps [s]
    pub times: Vec<f64>,
    /// Load positions [m]
    pub load_positions: Vec<Vector3<f64>>,
    /// Load velocities [m/s]
    pub load_velocities: Vec<Vector3<f64>>,
    /// Load orientations
    pub load_orientations: Vec<UnitQuaternion<f64>>,
    /// Quadrotor positions [m] (per quadrotor)
    pub quad_positions: Vec<Vec<Vector3<f64>>>,
    /// Cable tensions [N]
    pub cable_tensions: Vec<Vec<f64>>,
}

impl SimHistory {
    pub fn new(num_quads: usize) -> Self {
        Self {
            times: Vec::new(),
            load_positions: Vec::new(),
            load_velocities: Vec::new(),
            load_orientations: Vec::new(),
            quad_positions: vec![Vec::new(); num_quads],
            cable_tensions: Vec::new(),
        }
    }

    /// Record a simulation step
    pub fn record(&mut self, step: &SimStep) {
        self.times.push(step.time);
        self.load_positions.push(step.state.load.position);
        self.load_velocities.push(step.state.load.velocity);
        self.load_orientations.push(step.state.load.orientation);

        for (i, quad) in step.state.quadrotors.iter().enumerate() {
            self.quad_positions[i].push(quad.position);
        }

        self.cable_tensions.push(step.state.cables.tensions());
    }

    /// Get simulation duration
    pub fn duration(&self) -> f64 {
        if self.times.is_empty() {
            0.0
        } else {
            *self.times.last().unwrap() - self.times[0]
        }
    }

    /// Get number of recorded steps
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

/// Main simulator
pub struct Simulator {
    /// Configuration
    pub config: SimConfig,
    /// System dynamics model
    dynamics: SystemDynamics,
    /// Current system state
    state: SystemState,
    /// Current simulation time
    time: f64,
    /// Sensor models
    sensors: SystemSensors,
    /// History recorder
    history: SimHistory,
}

impl Simulator {
    /// Create a new simulator from configuration
    pub fn new(config: SimConfig) -> Self {
        let num_quads = config.physics.quadrotors.len();

        // Build system parameters from config
        let load_params = config.physics.load.to_params();
        let cable_params: Vec<CableParams> = config.physics.cables
            .iter()
            .map(|c| c.to_params())
            .collect();
        let quad_params: Vec<QuadrotorParams> = config.physics.quadrotors
            .iter()
            .map(|q| q.to_params())
            .collect();

        let system_params = SystemParams {
            load: load_params,
            cables: cable_params.clone(),
            quadrotors: quad_params,
        };

        let dynamics = SystemDynamics::new(system_params);

        // Initialize state from config
        let load_state = LoadState {
            position: config.initial_state.load_position,
            velocity: config.initial_state.load_velocity,
            orientation: UnitQuaternion::from_euler_angles(0.0, 0.0, config.initial_state.load_yaw),
            angular_velocity: Vector3::zeros(),
        };

        let cable_states: Vec<CableState> = config.initial_state.cable_tensions
            .iter()
            .map(|&t| CableState::pointing_down(t))
            .collect();

        let cables = MultiCableState::new(cable_states);

        let cable_dynamics: Vec<_> = cable_params.into_iter()
            .map(|p| crate::dynamics::CableDynamics::new(p))
            .collect();

        let state = SystemState::new(load_state, cables, &dynamics.load, &cable_dynamics);

        let sensors = SystemSensors::new(num_quads, &config.sensors);
        let history = SimHistory::new(num_quads);

        Self {
            config,
            dynamics,
            state,
            time: 0.0,
            sensors,
            history,
        }
    }

    /// Reset simulation to initial state
    pub fn reset(&mut self) {
        let load_state = LoadState {
            position: self.config.initial_state.load_position,
            velocity: self.config.initial_state.load_velocity,
            orientation: UnitQuaternion::from_euler_angles(0.0, 0.0, self.config.initial_state.load_yaw),
            angular_velocity: Vector3::zeros(),
        };

        let cable_states: Vec<CableState> = self.config.initial_state.cable_tensions
            .iter()
            .map(|&t| CableState::pointing_down(t))
            .collect();

        let cables = MultiCableState::new(cable_states);

        let cable_dynamics: Vec<_> = self.dynamics.cables.clone();

        self.state = SystemState::new(load_state, cables, &self.dynamics.load, &cable_dynamics);
        self.time = 0.0;
        self.history = SimHistory::new(self.dynamics.num_agents());
    }

    /// Step simulation forward by dt
    pub fn step(&mut self, input: SystemInput) -> SimStep {
        // Record current state
        let step = SimStep {
            time: self.time,
            state: self.state.clone(),
            inputs: input.clone(),
        };

        self.history.record(&step);

        // Integrate dynamics
        self.state = self.dynamics.integrate(&self.state, &input, self.config.dt);
        self.time += self.config.dt;

        step
    }

    /// Run simulation for the configured duration with a controller
    pub fn run<C>(&mut self, mut controller: C) -> &SimHistory
    where
        C: FnMut(f64, &SystemState) -> SystemInput,
    {
        while self.time < self.config.duration {
            let input = controller(self.time, &self.state);
            self.step(input);
        }

        &self.history
    }

    /// Run simulation with zero input (free fall / dynamics test)
    pub fn run_free(&mut self) -> &SimHistory {
        let n = self.dynamics.num_agents();
        self.run(|_t, _state| SystemInput::zeros(n))
    }

    /// Get current simulation time
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get current state
    pub fn state(&self) -> &SystemState {
        &self.state
    }

    /// Get simulation history
    pub fn history(&self) -> &SimHistory {
        &self.history
    }

    /// Get system dynamics
    pub fn dynamics(&self) -> &SystemDynamics {
        &self.dynamics
    }

    /// Get number of quadrotors
    pub fn num_quadrotors(&self) -> usize {
        self.dynamics.num_agents()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simulator_creation() {
        let config = SimConfig::default();
        let sim = Simulator::new(config);

        assert_eq!(sim.time(), 0.0);
        assert_eq!(sim.num_quadrotors(), 3);
    }

    #[test]
    fn test_simulator_step() {
        let mut config = SimConfig::default();
        config.dt = 0.01;

        let mut sim = Simulator::new(config);
        let input = SystemInput::zeros(3);

        let step = sim.step(input);

        assert_eq!(step.time, 0.0);
        assert_relative_eq!(sim.time(), 0.01, epsilon = 1e-10);
    }

    #[test]
    fn test_simulator_free_fall() {
        let mut config = SimConfig::default();
        config.dt = 0.01;
        config.duration = 0.1;

        let mut sim = Simulator::new(config);
        let history = sim.run_free();

        // Load should be falling (positive z velocity in NED)
        let final_vel = history.load_velocities.last().unwrap();
        assert!(final_vel.z > 0.0, "Load should be accelerating downward");
    }

    #[test]
    fn test_simulator_reset() {
        let mut config = SimConfig::default();
        config.dt = 0.01;
        config.duration = 0.1;

        let mut sim = Simulator::new(config);
        sim.run_free();

        // State should have changed
        assert!(sim.time() > 0.0);

        // Reset
        sim.reset();

        assert_eq!(sim.time(), 0.0);
        assert_relative_eq!(
            sim.state().load.position,
            Vector3::zeros(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_history_recording() {
        let mut config = SimConfig::default();
        config.dt = 0.01;
        config.duration = 0.1;

        let mut sim = Simulator::new(config);
        sim.run_free();

        let history = sim.history();

        assert!(!history.is_empty());
        assert_eq!(history.load_positions.len(), history.times.len());
        assert_eq!(history.quad_positions[0].len(), history.times.len());
    }

    #[test]
    fn test_simulator_with_controller() {
        let mut config = SimConfig::default();
        config.dt = 0.01;
        config.duration = 0.05;

        let mut sim = Simulator::new(config);

        // Simple constant input controller
        let controller = |_t: f64, _state: &SystemState| {
            let mut input = SystemInput::zeros(3);
            for quad_input in &mut input.quadrotors {
                quad_input.thrust = 10.0; // Some thrust
            }
            input
        };

        let history = sim.run(controller);

        assert!(!history.is_empty());
    }
}
