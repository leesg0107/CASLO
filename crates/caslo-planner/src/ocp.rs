

//! Optimal Control Problem (OCP) Definition
//!
//! Implements the finite-time OCP formulation from Eq. 6 of the paper:
//!
//! ```text
//! minimize    J = Σₖ (‖xₖ - xₖ,ref‖²_Q + ‖uₖ - uₖ,ref‖²_R) + ‖xₙ - xₙ,ref‖²_P
//! subject to  x₀ = x_init
//!             xₖ₊₁ = f(xₖ, uₖ)     (load-cable dynamics)
//!             h(xₖ, uₖ) ≤ 0        (path constraints)
//! ```

use nalgebra::{Vector3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::config::{PlannerConfig, Obstacle};

/// State vector for the OCP
///
/// The state includes load pose/velocity and cable states for all quadrotors.
/// For n quadrotors:
/// - Load position: 3
/// - Load velocity: 3
/// - Load orientation (quaternion): 4
/// - Load angular velocity: 3
/// - Per cable (×n):
///   - Cable direction: 3 (unit vector, 2 DOF but represented as 3)
///   - Cable angular velocity: 3
///   - Cable tension: 1
///
/// Total: 13 + 7n states
#[derive(Debug, Clone)]
pub struct OcpState {
    /// Load position in world frame [m]
    pub load_position: Vector3<f64>,
    /// Load velocity in world frame [m/s]
    pub load_velocity: Vector3<f64>,
    /// Load orientation (world to body)
    pub load_orientation: UnitQuaternion<f64>,
    /// Load angular velocity in body frame [rad/s]
    pub load_angular_velocity: Vector3<f64>,
    /// Cable states for each quadrotor
    pub cables: Vec<CableState>,
}

/// Cable state in the OCP
#[derive(Debug, Clone)]
pub struct CableState {
    /// Unit vector from quadrotor to attachment point
    pub direction: Vector3<f64>,
    /// Angular velocity of cable direction [rad/s]
    pub angular_velocity: Vector3<f64>,
    /// Cable tension [N]
    pub tension: f64,
}

impl OcpState {
    /// Create a new OCP state with default values
    /// Note: Cable direction convention is FROM quad TOWARD load (downward)
    /// In NED coordinate frame (Z-down), cables pointing DOWN have positive Z component
    pub fn new(num_cables: usize) -> Self {
        Self {
            load_position: Vector3::zeros(),
            load_velocity: Vector3::zeros(),
            load_orientation: UnitQuaternion::identity(),
            load_angular_velocity: Vector3::zeros(),
            cables: vec![
                CableState {
                    // Cable direction points FROM quad TOWARD load (downward in NED = +Z)
                    direction: Vector3::new(0.0, 0.0, 1.0),
                    angular_velocity: Vector3::zeros(),
                    tension: 5.0,
                };
                num_cables
            ],
        }
    }

    /// Total state dimension
    pub fn dimension(num_cables: usize) -> usize {
        13 + 7 * num_cables
    }

    /// Pack state into a flat vector (ACADOS layout)
    /// Layout: [p_L(3), v_L(3), q_L(4), omega_L(3), s_all(3*n), r_all(3*n), t_all(n)]
    /// This matches the state vector in caslo_ocp.py
    pub fn to_vector(&self) -> Vec<f64> {
        let n = self.cables.len();
        let mut v = Vec::with_capacity(Self::dimension(n));

        // Load states (13 elements)
        v.extend(self.load_position.iter());
        v.extend(self.load_velocity.iter());
        v.push(self.load_orientation.w);
        v.push(self.load_orientation.i);
        v.push(self.load_orientation.j);
        v.push(self.load_orientation.k);
        v.extend(self.load_angular_velocity.iter());

        // ALL cable directions (3*n elements)
        for cable in &self.cables {
            v.extend(cable.direction.iter());
        }

        // ALL cable angular velocities (3*n elements)
        for cable in &self.cables {
            v.extend(cable.angular_velocity.iter());
        }

        // ALL cable tensions (n elements)
        for cable in &self.cables {
            v.push(cable.tension);
        }

        v
    }

    /// Unpack state from a flat vector (ACADOS layout)
    pub fn from_vector(v: &[f64], num_cables: usize) -> Option<Self> {
        if v.len() != Self::dimension(num_cables) {
            return None;
        }

        let load_position = Vector3::new(v[0], v[1], v[2]);
        let load_velocity = Vector3::new(v[3], v[4], v[5]);
        let load_orientation = UnitQuaternion::from_quaternion(
            nalgebra::Quaternion::new(v[6], v[7], v[8], v[9])
        );
        let load_angular_velocity = Vector3::new(v[10], v[11], v[12]);

        // Parse cable states from ACADOS layout
        let n = num_cables;
        let s_start = 13;
        let r_start = 13 + 3 * n;
        let t_start = 13 + 6 * n;

        let mut cables = Vec::with_capacity(n);
        for i in 0..n {
            cables.push(CableState {
                direction: Vector3::new(
                    v[s_start + 3*i], v[s_start + 3*i + 1], v[s_start + 3*i + 2]
                ),
                angular_velocity: Vector3::new(
                    v[r_start + 3*i], v[r_start + 3*i + 1], v[r_start + 3*i + 2]
                ),
                tension: v[t_start + i],
            });
        }

        Some(Self {
            load_position,
            load_velocity,
            load_orientation,
            load_angular_velocity,
            cables,
        })
    }
}

/// Control input vector for the OCP
///
/// The control inputs are the cable angular jerk (γ) and tension acceleration (λ)
/// from Eq. 3:
/// - r̈ᵢ = γᵢ  (cable angular jerk)
/// - ẗᵢ = λᵢ  (tension acceleration)
///
/// For n quadrotors: 4n control inputs
#[derive(Debug, Clone)]
pub struct OcpControl {
    /// Control inputs for each cable
    pub cables: Vec<CableControl>,
}

/// Control input for a single cable
#[derive(Debug, Clone, Copy)]
pub struct CableControl {
    /// Cable angular jerk [rad/s³]
    pub angular_jerk: Vector3<f64>,
    /// Tension acceleration [N/s²]
    pub tension_acceleration: f64,
}

impl OcpControl {
    /// Create zero control input
    pub fn new(num_cables: usize) -> Self {
        Self {
            cables: vec![
                CableControl {
                    angular_jerk: Vector3::zeros(),
                    tension_acceleration: 0.0,
                };
                num_cables
            ],
        }
    }

    /// Total control dimension
    pub fn dimension(num_cables: usize) -> usize {
        4 * num_cables
    }

    /// Pack control into a flat vector (ACADOS layout)
    /// Layout: [gamma_all(3*n), lambda_all(n)]
    /// This matches the control vector in caslo_ocp.py
    pub fn to_vector(&self) -> Vec<f64> {
        let n = self.cables.len();
        let mut v = Vec::with_capacity(Self::dimension(n));

        // ALL angular jerks (3*n elements)
        for cable in &self.cables {
            v.extend(cable.angular_jerk.iter());
        }

        // ALL tension accelerations (n elements)
        for cable in &self.cables {
            v.push(cable.tension_acceleration);
        }

        v
    }

    /// Unpack control from a flat vector (ACADOS layout)
    pub fn from_vector(v: &[f64], num_cables: usize) -> Option<Self> {
        if v.len() != Self::dimension(num_cables) {
            return None;
        }

        let n = num_cables;
        let gamma_start = 0;
        let lambda_start = 3 * n;

        let mut cables = Vec::with_capacity(n);
        for i in 0..n {
            cables.push(CableControl {
                angular_jerk: Vector3::new(
                    v[gamma_start + 3*i], v[gamma_start + 3*i + 1], v[gamma_start + 3*i + 2]
                ),
                tension_acceleration: v[lambda_start + i],
            });
        }

        Some(Self { cables })
    }
}

/// Reference trajectory point for tracking
#[derive(Debug, Clone)]
pub struct ReferencePoint {
    /// Desired load position
    pub position: Vector3<f64>,
    /// Desired load velocity
    pub velocity: Vector3<f64>,
    /// Desired load orientation
    pub orientation: UnitQuaternion<f64>,
    /// Desired load angular velocity
    pub angular_velocity: Vector3<f64>,
}

impl Default for ReferencePoint {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        }
    }
}

/// OCP problem definition
///
/// Contains all information needed to set up and solve the OCP.
#[derive(Debug, Clone)]
pub struct OcpDefinition {
    /// Number of quadrotors/cables
    pub num_quadrotors: usize,
    /// Planner configuration
    pub config: PlannerConfig,
    /// System parameters
    pub system: SystemParameters,
    /// Obstacles for avoidance
    pub obstacles: Vec<Obstacle>,
}

/// System parameters required for OCP dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemParameters {
    /// Load mass [kg]
    pub load_mass: f64,
    /// Load inertia tensor [kg·m²]
    pub load_inertia: [f64; 9], // Row-major 3x3
    /// Quadrotor masses [kg]
    pub quadrotor_masses: Vec<f64>,
    /// Cable lengths [m]
    pub cable_lengths: Vec<f64>,
    /// Attachment points in load body frame [m]
    pub attachment_points: Vec<[f64; 3]>,
    /// Gravity acceleration [m/s²]
    pub gravity: f64,
}

impl SystemParameters {
    /// Create system parameters for n quadrotors
    pub fn new(num_quadrotors: usize) -> Self {
        Self {
            load_mass: 0.3,
            load_inertia: [
                0.01, 0.0, 0.0,
                0.0, 0.01, 0.0,
                0.0, 0.0, 0.01,
            ],
            quadrotor_masses: vec![0.6; num_quadrotors],
            cable_lengths: vec![1.0; num_quadrotors],
            attachment_points: default_attachment_points(num_quadrotors),
            gravity: 9.81,
        }
    }
}

/// Generate default attachment points for n quadrotors
///
/// Places attachment points in a regular polygon pattern on the load
fn default_attachment_points(n: usize) -> Vec<[f64; 3]> {
    use std::f64::consts::PI;

    let radius = 0.1; // 10cm from center
    (0..n)
        .map(|i| {
            let angle = 2.0 * PI * i as f64 / n as f64;
            [radius * angle.cos(), radius * angle.sin(), 0.0]
        })
        .collect()
}

impl OcpDefinition {
    /// Create a new OCP definition
    pub fn new(num_quadrotors: usize) -> Self {
        Self {
            num_quadrotors,
            config: PlannerConfig::default(),
            system: SystemParameters::new(num_quadrotors),
            obstacles: Vec::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(num_quadrotors: usize, config: PlannerConfig) -> Self {
        Self {
            num_quadrotors,
            config,
            system: SystemParameters::new(num_quadrotors),
            obstacles: Vec::new(),
        }
    }

    /// State dimension
    pub fn nx(&self) -> usize {
        OcpState::dimension(self.num_quadrotors)
    }

    /// Control dimension
    pub fn nu(&self) -> usize {
        OcpControl::dimension(self.num_quadrotors)
    }

    /// Number of shooting nodes
    pub fn n(&self) -> usize {
        self.config.horizon.num_nodes
    }

    /// Add an obstacle
    pub fn add_obstacle(&mut self, obstacle: Obstacle) {
        self.obstacles.push(obstacle);
    }

    /// Set system parameters
    pub fn set_system(&mut self, system: SystemParameters) {
        self.system = system;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_dimension() {
        assert_eq!(OcpState::dimension(3), 13 + 7 * 3); // 34
        assert_eq!(OcpState::dimension(4), 13 + 7 * 4); // 41
    }

    #[test]
    fn test_control_dimension() {
        assert_eq!(OcpControl::dimension(3), 4 * 3); // 12
        assert_eq!(OcpControl::dimension(4), 4 * 4); // 16
    }

    #[test]
    fn test_state_vector_roundtrip() {
        let mut state = OcpState::new(3);
        state.load_position = Vector3::new(1.0, 2.0, 3.0);
        state.load_velocity = Vector3::new(0.1, 0.2, 0.3);
        state.cables[0].tension = 10.0;
        state.cables[1].direction = Vector3::new(0.0, -1.0, 0.0);

        let v = state.to_vector();
        let recovered = OcpState::from_vector(&v, 3).unwrap();

        assert!((recovered.load_position - state.load_position).norm() < 1e-10);
        assert!((recovered.load_velocity - state.load_velocity).norm() < 1e-10);
        assert!((recovered.cables[0].tension - state.cables[0].tension).abs() < 1e-10);
    }

    #[test]
    fn test_control_vector_roundtrip() {
        let mut control = OcpControl::new(3);
        control.cables[0].angular_jerk = Vector3::new(1.0, 2.0, 3.0);
        control.cables[1].tension_acceleration = 5.0;

        let v = control.to_vector();
        let recovered = OcpControl::from_vector(&v, 3).unwrap();

        assert!((recovered.cables[0].angular_jerk - control.cables[0].angular_jerk).norm() < 1e-10);
        assert!((recovered.cables[1].tension_acceleration - control.cables[1].tension_acceleration).abs() < 1e-10);
    }

    #[test]
    fn test_ocp_definition() {
        let ocp = OcpDefinition::new(3);
        assert_eq!(ocp.nx(), 34);
        assert_eq!(ocp.nu(), 12);
        assert_eq!(ocp.n(), 20);
    }
}
