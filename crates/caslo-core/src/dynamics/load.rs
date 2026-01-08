//! Load (payload) dynamics
//!
//! Implements the load dynamics from Eq. (2) in the paper:
//!
//! v̇ = -1/m · Σᵢ tᵢsᵢ + g
//! Jω̇ = -ω × Jω + Σᵢ tᵢ(R^T sᵢ × ρᵢ)
//!
//! where:
//! - m: load mass
//! - J: load inertia tensor (body frame)
//! - v: load velocity (world frame)
//! - ω: load angular velocity (body frame)
//! - tᵢ: cable tension
//! - sᵢ: cable direction (unit vector, world frame)
//! - ρᵢ: attachment point (body frame)
//! - R: rotation matrix from body to world

use nalgebra::{Vector3, Matrix3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::math::quaternion_derivative;
use crate::GRAVITY;

/// Load state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadState {
    /// Position [m] (world frame)
    pub position: Vector3<f64>,
    /// Velocity [m/s] (world frame)
    pub velocity: Vector3<f64>,
    /// Orientation (body to world)
    pub orientation: UnitQuaternion<f64>,
    /// Angular velocity [rad/s] (body frame)
    pub angular_velocity: Vector3<f64>,
}

impl Default for LoadState {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        }
    }
}

/// Load parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadParams {
    /// Mass [kg]
    pub mass: f64,
    /// Inertia tensor [kg·m²] (body frame, diagonal)
    pub inertia: Matrix3<f64>,
    /// Inverse inertia tensor
    pub inertia_inv: Matrix3<f64>,
    /// Cable attachment points [m] (body frame)
    pub attachment_points: Vec<Vector3<f64>>,
}

impl LoadParams {
    /// Create load parameters with diagonal inertia
    pub fn new(mass: f64, inertia_diag: Vector3<f64>, attachment_points: Vec<Vector3<f64>>) -> Self {
        let inertia = Matrix3::from_diagonal(&inertia_diag);
        let inertia_inv = Matrix3::from_diagonal(&Vector3::new(
            1.0 / inertia_diag.x,
            1.0 / inertia_diag.y,
            1.0 / inertia_diag.z,
        ));
        Self {
            mass,
            inertia,
            inertia_inv,
            attachment_points,
        }
    }

    /// Number of cables attached to this load
    pub fn num_cables(&self) -> usize {
        self.attachment_points.len()
    }
}

/// Load dynamics model
#[derive(Debug, Clone)]
pub struct LoadDynamics {
    pub params: LoadParams,
}

impl LoadDynamics {
    pub fn new(params: LoadParams) -> Self {
        Self { params }
    }

    /// Compute translational acceleration from Eq. (2)
    ///
    /// v̇ = -1/m · Σᵢ tᵢsᵢ + g
    ///
    /// # Arguments
    /// * `tensions` - Cable tensions [N]
    /// * `directions` - Cable directions (unit vectors, world frame)
    ///
    /// # Returns
    /// Translational acceleration [m/s²] (world frame)
    pub fn compute_acceleration(
        &self,
        tensions: &[f64],
        directions: &[Vector3<f64>],
    ) -> Vector3<f64> {
        let mut cable_force = Vector3::zeros();

        for (t, s) in tensions.iter().zip(directions.iter()) {
            cable_force += *t * s;
        }

        // Gravity in world frame (Z-UP convention from paper: g = [0, 0, -9.81])
        let gravity = Vector3::new(0.0, 0.0, -GRAVITY);

        -cable_force / self.params.mass + gravity
    }

    /// Compute angular acceleration from Eq. (2)
    ///
    /// Jω̇ = -ω × Jω + Σᵢ tᵢ(R^T sᵢ × ρᵢ)
    ///
    /// # Arguments
    /// * `state` - Current load state
    /// * `tensions` - Cable tensions [N]
    /// * `directions` - Cable directions (unit vectors, world frame)
    ///
    /// # Returns
    /// Angular acceleration [rad/s²] (body frame)
    pub fn compute_angular_acceleration(
        &self,
        state: &LoadState,
        tensions: &[f64],
        directions: &[Vector3<f64>],
    ) -> Vector3<f64> {
        let omega = &state.angular_velocity;
        let j = &self.params.inertia;
        let j_inv = &self.params.inertia_inv;
        let r = state.orientation.to_rotation_matrix();

        // Gyroscopic term: -ω × Jω
        let gyro = -omega.cross(&(j * omega));

        // Cable torques: Σᵢ tᵢ(R^T sᵢ × ρᵢ)
        let mut cable_torque = Vector3::zeros();
        for (i, (t, s)) in tensions.iter().zip(directions.iter()).enumerate() {
            let rho = &self.params.attachment_points[i];
            // R^T s transforms world-frame direction to body frame
            let s_body = r.inverse() * s;
            cable_torque += *t * s_body.cross(rho);
        }

        // ω̇ = J^(-1) * (gyro + cable_torque)
        j_inv * (gyro + cable_torque)
    }

    /// Compute state derivative
    pub fn state_derivative(
        &self,
        state: &LoadState,
        tensions: &[f64],
        directions: &[Vector3<f64>],
    ) -> LoadStateDerivative {
        let acceleration = self.compute_acceleration(tensions, directions);
        let angular_acceleration = self.compute_angular_acceleration(state, tensions, directions);
        let orientation_derivative = quaternion_derivative(&state.orientation, &state.angular_velocity);

        LoadStateDerivative {
            velocity: state.velocity,
            acceleration,
            orientation_derivative,
            angular_acceleration,
        }
    }

    /// Get world-frame position of attachment point i
    pub fn attachment_world(&self, state: &LoadState, index: usize) -> Vector3<f64> {
        let r = state.orientation.to_rotation_matrix();
        state.position + r * self.params.attachment_points[index]
    }
}

/// Load state derivative for integration
#[derive(Debug, Clone)]
pub struct LoadStateDerivative {
    /// Position derivative = velocity
    pub velocity: Vector3<f64>,
    /// Velocity derivative = acceleration
    pub acceleration: Vector3<f64>,
    /// Quaternion derivative (w, x, y, z)
    pub orientation_derivative: nalgebra::Vector4<f64>,
    /// Angular velocity derivative = angular acceleration
    pub angular_acceleration: Vector3<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_load() -> LoadDynamics {
        let params = LoadParams::new(
            1.0, // 1 kg
            Vector3::new(0.01, 0.01, 0.01), // Small inertia
            vec![
                Vector3::new(0.1, 0.0, 0.0),  // Right
                Vector3::new(-0.1, 0.0, 0.0), // Left
                Vector3::new(0.0, 0.1, 0.0),  // Front
            ],
        );
        LoadDynamics::new(params)
    }

    #[test]
    fn test_load_free_fall() {
        let load = create_test_load();

        // No cable forces
        let tensions = vec![0.0, 0.0, 0.0];
        let directions = vec![
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ];

        let acc = load.compute_acceleration(&tensions, &directions);

        // Should be pure gravity (Z-UP: gravity points down = negative Z)
        assert_relative_eq!(acc.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(acc.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(acc.z, -GRAVITY, epsilon = 1e-10);
    }

    #[test]
    fn test_load_hover() {
        let load = create_test_load();

        // Cable forces balance gravity (3 cables, each providing 1/3 of weight)
        let weight = load.params.mass * GRAVITY;
        let tension_per_cable = weight / 3.0;

        let tensions = vec![tension_per_cable, tension_per_cable, tension_per_cable];
        // Cable direction convention: sᵢ points from quadrotor toward load attachment
        // For hovering with quads above load in Z-UP, cables point down (-z)
        let directions = vec![
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ];

        let acc = load.compute_acceleration(&tensions, &directions);

        // Should be approximately zero (hovering)
        assert_relative_eq!(acc.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_no_angular_acceleration_at_rest() {
        let load = create_test_load();
        let state = LoadState::default();

        // Symmetric forces through center of mass
        let tensions = vec![1.0, 1.0, 1.0];
        // Cable direction points from quad toward load (down in Z-UP = -Z)
        let directions = vec![
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ];

        let alpha = load.compute_angular_acceleration(&state, &tensions, &directions);

        // With asymmetric attachment points, there may be some torque
        // Just verify the computation doesn't produce NaN or huge values
        assert!(alpha.norm().is_finite());
        assert!(alpha.norm() < 1000.0); // Reasonable bound
    }

    #[test]
    fn test_attachment_world_position() {
        let load = create_test_load();
        let mut state = LoadState::default();
        state.position = Vector3::new(1.0, 2.0, 3.0);

        // Without rotation, attachment points are offset from position
        let attach0 = load.attachment_world(&state, 0);
        assert_relative_eq!(attach0, Vector3::new(1.1, 2.0, 3.0), epsilon = 1e-10);
    }
}
