//! Simulation configuration
//!
//! Defines configuration structures for setting up simulations.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use crate::dynamics::{LoadParams, CableParams, QuadrotorParams};

/// Simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    /// Simulation time step [s]
    pub dt: f64,
    /// Total simulation duration [s]
    pub duration: f64,
    /// Real-time factor (1.0 = real time, 0 = as fast as possible)
    pub real_time_factor: f64,
    /// Physical parameters
    pub physics: PhysicsConfig,
    /// Initial state configuration
    pub initial_state: InitialStateConfig,
    /// Sensor configuration
    pub sensors: SensorConfig,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            dt: 0.001, // 1 kHz simulation
            duration: 10.0,
            real_time_factor: 0.0, // As fast as possible
            physics: PhysicsConfig::default(),
            initial_state: InitialStateConfig::default(),
            sensors: SensorConfig::default(),
        }
    }
}

/// Physical parameters configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    /// Gravity magnitude [m/s²]
    pub gravity: f64,
    /// Air density [kg/m³]
    pub air_density: f64,
    /// Enable aerodynamic drag
    pub enable_drag: bool,
    /// Load parameters
    pub load: LoadPhysicsConfig,
    /// Cable parameters (per cable)
    pub cables: Vec<CablePhysicsConfig>,
    /// Quadrotor parameters (per quadrotor)
    pub quadrotors: Vec<QuadrotorPhysicsConfig>,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            air_density: 1.225,
            enable_drag: false,
            load: LoadPhysicsConfig::default(),
            cables: vec![CablePhysicsConfig::default(); 3],
            quadrotors: vec![QuadrotorPhysicsConfig::default(); 3],
        }
    }
}

/// Load physical parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadPhysicsConfig {
    /// Mass [kg]
    pub mass: f64,
    /// Inertia diagonal [kg·m²]
    pub inertia: Vector3<f64>,
    /// Attachment point positions in body frame [m]
    pub attachment_points: Vec<Vector3<f64>>,
}

impl Default for LoadPhysicsConfig {
    fn default() -> Self {
        // Triangular attachment pattern
        Self {
            mass: 1.0,
            inertia: Vector3::new(0.01, 0.01, 0.01),
            attachment_points: vec![
                Vector3::new(0.1, 0.0, 0.0),
                Vector3::new(-0.05, 0.087, 0.0),
                Vector3::new(-0.05, -0.087, 0.0),
            ],
        }
    }
}

impl LoadPhysicsConfig {
    pub fn to_params(&self) -> LoadParams {
        LoadParams::new(self.mass, self.inertia, self.attachment_points.clone())
    }
}

/// Cable physical parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CablePhysicsConfig {
    /// Cable length [m]
    pub length: f64,
    /// Minimum tension [N]
    pub min_tension: f64,
    /// Maximum tension [N]
    pub max_tension: f64,
}

impl Default for CablePhysicsConfig {
    fn default() -> Self {
        Self {
            length: 1.0,
            min_tension: 0.0,
            max_tension: 100.0,
        }
    }
}

impl CablePhysicsConfig {
    pub fn to_params(&self) -> CableParams {
        let mut params = CableParams::new(self.length);
        params.min_tension = self.min_tension;
        params.max_tension = self.max_tension;
        params
    }
}

/// Quadrotor physical parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadrotorPhysicsConfig {
    /// Mass [kg]
    pub mass: f64,
    /// Inertia diagonal [kg·m²]
    pub inertia: Vector3<f64>,
    /// Arm length [m]
    pub arm_length: f64,
    /// Maximum thrust [N]
    pub max_thrust: f64,
    /// Maximum torque [N·m]
    pub max_torque: Vector3<f64>,
}

impl Default for QuadrotorPhysicsConfig {
    fn default() -> Self {
        Self {
            mass: 1.0,
            inertia: Vector3::new(0.01, 0.01, 0.02),
            arm_length: 0.2,
            max_thrust: 30.0,
            max_torque: Vector3::new(1.0, 1.0, 0.5),
        }
    }
}

impl QuadrotorPhysicsConfig {
    pub fn to_params(&self) -> QuadrotorParams {
        let mut params = QuadrotorParams::new(self.mass, self.inertia, self.arm_length);
        params.max_thrust = self.max_thrust;
        params.max_torque = self.max_torque;
        params
    }
}

/// Initial state configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialStateConfig {
    /// Initial load position [m]
    pub load_position: Vector3<f64>,
    /// Initial load velocity [m/s]
    pub load_velocity: Vector3<f64>,
    /// Initial load yaw angle [rad]
    pub load_yaw: f64,
    /// Initial cable tensions [N]
    pub cable_tensions: Vec<f64>,
}

impl Default for InitialStateConfig {
    fn default() -> Self {
        Self {
            load_position: Vector3::new(0.0, 0.0, 0.0),
            load_velocity: Vector3::zeros(),
            load_yaw: 0.0,
            cable_tensions: vec![10.0, 10.0, 10.0],
        }
    }
}

/// Sensor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorConfig {
    /// Position measurement noise std dev [m]
    pub position_noise_std: f64,
    /// Velocity measurement noise std dev [m/s]
    pub velocity_noise_std: f64,
    /// Orientation measurement noise std dev [rad]
    pub orientation_noise_std: f64,
    /// IMU accelerometer noise std dev [m/s²]
    pub accel_noise_std: f64,
    /// IMU gyroscope noise std dev [rad/s]
    pub gyro_noise_std: f64,
    /// Sensor update rate [Hz]
    pub update_rate: f64,
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            position_noise_std: 0.01,
            velocity_noise_std: 0.05,
            orientation_noise_std: 0.01,
            accel_noise_std: 0.1,
            gyro_noise_std: 0.01,
            update_rate: 100.0, // 100 Hz
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SimConfig::default();
        assert_eq!(config.dt, 0.001);
        assert_eq!(config.physics.cables.len(), 3);
        assert_eq!(config.physics.quadrotors.len(), 3);
    }

    #[test]
    fn test_physics_conversion() {
        let load_config = LoadPhysicsConfig::default();
        let load_params = load_config.to_params();

        assert_eq!(load_params.mass, load_config.mass);
        assert_eq!(load_params.attachment_points.len(), 3);
    }

    #[test]
    fn test_cable_conversion() {
        let cable_config = CablePhysicsConfig {
            length: 2.0,
            min_tension: 1.0,
            max_tension: 50.0,
        };

        let cable_params = cable_config.to_params();
        assert_eq!(cable_params.length, 2.0);
        assert_eq!(cable_params.min_tension, 1.0);
        assert_eq!(cable_params.max_tension, 50.0);
    }
}
