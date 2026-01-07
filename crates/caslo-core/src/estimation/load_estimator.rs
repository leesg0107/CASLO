//! Load-Cable State Estimator
//!
//! Implements the EKF-based load-cable state estimator from the paper.
//! The state vector includes load position, velocity, orientation, angular velocity,
//! and cable directions and angular velocities.

use nalgebra::{Vector3, UnitQuaternion, DVector, DMatrix};
use serde::{Deserialize, Serialize};

use super::ExtendedKalmanFilter;
use crate::dynamics::{LoadState, CableState, MultiCableState};

/// Load-cable estimator state indices
#[derive(Debug, Clone)]
pub struct LoadCableStateIndices {
    /// Load position start index (3 elements)
    pub load_pos: usize,
    /// Load velocity start index (3 elements)
    pub load_vel: usize,
    /// Load orientation start index (4 elements - quaternion)
    pub load_quat: usize,
    /// Load angular velocity start index (3 elements)
    pub load_omega: usize,
    /// Cable states start index (per cable: 3 direction + 3 angular velocity)
    pub cables_start: usize,
    /// Total state dimension
    pub total_dim: usize,
}

impl LoadCableStateIndices {
    pub fn new(num_cables: usize) -> Self {
        // State layout:
        // [pos(3), vel(3), quat(4), omega(3), cable1_dir(3), cable1_omega(3), ...]
        let load_pos = 0;
        let load_vel = 3;
        let load_quat = 6;
        let load_omega = 10;
        let cables_start = 13;
        let total_dim = cables_start + num_cables * 6;

        Self {
            load_pos,
            load_vel,
            load_quat,
            load_omega,
            cables_start,
            total_dim,
        }
    }

    /// Get cable direction start index for cable i
    pub fn cable_dir(&self, i: usize) -> usize {
        self.cables_start + i * 6
    }

    /// Get cable angular velocity start index for cable i
    pub fn cable_omega(&self, i: usize) -> usize {
        self.cables_start + i * 6 + 3
    }
}

/// Load-cable state estimator
#[derive(Debug, Clone)]
pub struct LoadCableEstimator {
    /// Underlying EKF
    ekf: ExtendedKalmanFilter,
    /// State indices
    indices: LoadCableStateIndices,
    /// Number of cables
    num_cables: usize,
}

impl LoadCableEstimator {
    /// Create a new estimator for n cables
    pub fn new(num_cables: usize) -> Self {
        let indices = LoadCableStateIndices::new(num_cables);
        let meas_dim = num_cables * 3; // Quadrotor positions as measurements

        let mut ekf = ExtendedKalmanFilter::new(indices.total_dim, meas_dim);

        // Set default process noise
        let mut q_diag = vec![0.01; indices.total_dim];
        // Higher noise for velocities
        for i in indices.load_vel..indices.load_vel + 3 {
            q_diag[i] = 0.1;
        }
        for i in indices.load_omega..indices.load_omega + 3 {
            q_diag[i] = 0.1;
        }
        ekf.set_process_noise(&q_diag);

        // Set measurement noise
        let r_diag = vec![0.001; meas_dim]; // Position measurement noise
        ekf.set_measurement_noise(&r_diag);

        Self {
            ekf,
            indices,
            num_cables,
        }
    }

    /// Initialize estimator with known state
    pub fn initialize(&mut self, load: &LoadState, cables: &MultiCableState) {
        let mut x = DVector::zeros(self.indices.total_dim);

        // Load position
        x[self.indices.load_pos] = load.position.x;
        x[self.indices.load_pos + 1] = load.position.y;
        x[self.indices.load_pos + 2] = load.position.z;

        // Load velocity
        x[self.indices.load_vel] = load.velocity.x;
        x[self.indices.load_vel + 1] = load.velocity.y;
        x[self.indices.load_vel + 2] = load.velocity.z;

        // Load orientation (quaternion: w, x, y, z)
        x[self.indices.load_quat] = load.orientation.w;
        x[self.indices.load_quat + 1] = load.orientation.i;
        x[self.indices.load_quat + 2] = load.orientation.j;
        x[self.indices.load_quat + 3] = load.orientation.k;

        // Load angular velocity
        x[self.indices.load_omega] = load.angular_velocity.x;
        x[self.indices.load_omega + 1] = load.angular_velocity.y;
        x[self.indices.load_omega + 2] = load.angular_velocity.z;

        // Cable states
        for (i, cable) in cables.cables.iter().enumerate() {
            let dir_idx = self.indices.cable_dir(i);
            x[dir_idx] = cable.direction.x;
            x[dir_idx + 1] = cable.direction.y;
            x[dir_idx + 2] = cable.direction.z;

            let omega_idx = self.indices.cable_omega(i);
            x[omega_idx] = cable.angular_velocity.x;
            x[omega_idx + 1] = cable.angular_velocity.y;
            x[omega_idx + 2] = cable.angular_velocity.z;
        }

        self.ekf.state.x = x;
        self.ekf.state.p = DMatrix::identity(self.indices.total_dim, self.indices.total_dim) * 0.1;
    }

    /// Prediction step
    pub fn predict(&mut self, dt: f64) {
        let indices = self.indices.clone();
        let num_cables = self.num_cables;

        // Simple constant velocity model for prediction
        let f = move |x: &DVector<f64>| {
            let mut x_new = x.clone();

            // Position += velocity * dt
            for i in 0..3 {
                x_new[indices.load_pos + i] += x[indices.load_vel + i] * dt;
            }

            // Quaternion integration (simplified)
            let omega = Vector3::new(
                x[indices.load_omega],
                x[indices.load_omega + 1],
                x[indices.load_omega + 2],
            );
            let omega_norm = omega.norm();
            if omega_norm > 1e-10 {
                // Small angle approximation for quaternion update
                let dq_vec = omega * dt * 0.5;
                x_new[indices.load_quat + 1] += dq_vec.x * x[indices.load_quat];
                x_new[indices.load_quat + 2] += dq_vec.y * x[indices.load_quat];
                x_new[indices.load_quat + 3] += dq_vec.z * x[indices.load_quat];
            }

            // Normalize quaternion
            let quat_norm = (
                x_new[indices.load_quat].powi(2) +
                x_new[indices.load_quat + 1].powi(2) +
                x_new[indices.load_quat + 2].powi(2) +
                x_new[indices.load_quat + 3].powi(2)
            ).sqrt();
            for i in 0..4 {
                x_new[indices.load_quat + i] /= quat_norm;
            }

            // Cable direction updates: ṡ = r × s
            for i in 0..num_cables {
                let dir_idx = indices.cables_start + i * 6;
                let omega_idx = dir_idx + 3;

                let s = Vector3::new(x[dir_idx], x[dir_idx + 1], x[dir_idx + 2]);
                let r = Vector3::new(x[omega_idx], x[omega_idx + 1], x[omega_idx + 2]);

                let s_dot = r.cross(&s);
                let s_new = s + s_dot * dt;
                let s_norm = s_new.norm();

                x_new[dir_idx] = s_new.x / s_norm;
                x_new[dir_idx + 1] = s_new.y / s_norm;
                x_new[dir_idx + 2] = s_new.z / s_norm;
            }

            x_new
        };

        // Simplified Jacobian (identity + linear terms)
        let f_jac = move |_x: &DVector<f64>| {
            let mut jac = DMatrix::identity(indices.total_dim, indices.total_dim);

            // Position-velocity coupling
            for i in 0..3 {
                jac[(indices.load_pos + i, indices.load_vel + i)] = dt;
            }

            jac
        };

        self.ekf.predict(f, f_jac);
    }

    /// Update with quadrotor position measurements
    pub fn update(&mut self, quad_positions: &[Vector3<f64>], cable_lengths: &[f64], attachment_offsets: &[Vector3<f64>]) {
        let indices = self.indices.clone();
        let indices_jac = self.indices.clone();
        let num_cables = self.num_cables;
        let lengths = cable_lengths.to_vec();
        let lengths_jac = cable_lengths.to_vec();
        let offsets = attachment_offsets.to_vec();

        // Measurement function: quadrotor positions from load state and cable directions
        // pᵢ = p + R(q)ρᵢ - lᵢsᵢ
        let h = move |x: &DVector<f64>| {
            let mut z = DVector::zeros(num_cables * 3);

            let load_pos = Vector3::new(
                x[indices.load_pos],
                x[indices.load_pos + 1],
                x[indices.load_pos + 2],
            );

            let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                x[indices.load_quat],
                x[indices.load_quat + 1],
                x[indices.load_quat + 2],
                x[indices.load_quat + 3],
            ));

            for i in 0..num_cables {
                let dir_idx = indices.cables_start + i * 6;
                let cable_dir = Vector3::new(
                    x[dir_idx],
                    x[dir_idx + 1],
                    x[dir_idx + 2],
                );

                // pᵢ = p + R(q)ρᵢ - lᵢsᵢ
                let quad_pos = load_pos + q * offsets[i] - lengths[i] * cable_dir;

                z[i * 3] = quad_pos.x;
                z[i * 3 + 1] = quad_pos.y;
                z[i * 3 + 2] = quad_pos.z;
            }

            z
        };

        // Simplified measurement Jacobian
        let h_jac = move |_x: &DVector<f64>| {
            let mut jac = DMatrix::zeros(num_cables * 3, indices_jac.total_dim);

            // Position directly affects quad positions
            for i in 0..num_cables {
                for j in 0..3 {
                    jac[(i * 3 + j, indices_jac.load_pos + j)] = 1.0;
                }

                // Cable direction affects quad position
                let dir_idx = indices_jac.cables_start + i * 6;
                for j in 0..3 {
                    jac[(i * 3 + j, dir_idx + j)] = -lengths_jac[i];
                }
            }

            jac
        };

        // Build measurement vector
        let mut z = DVector::zeros(self.num_cables * 3);
        for (i, pos) in quad_positions.iter().enumerate() {
            z[i * 3] = pos.x;
            z[i * 3 + 1] = pos.y;
            z[i * 3 + 2] = pos.z;
        }

        self.ekf.update(&z, h, h_jac);
    }

    /// Extract load state from EKF estimate
    pub fn get_load_state(&self) -> LoadState {
        let x = &self.ekf.state.x;

        LoadState {
            position: Vector3::new(
                x[self.indices.load_pos],
                x[self.indices.load_pos + 1],
                x[self.indices.load_pos + 2],
            ),
            velocity: Vector3::new(
                x[self.indices.load_vel],
                x[self.indices.load_vel + 1],
                x[self.indices.load_vel + 2],
            ),
            orientation: UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                x[self.indices.load_quat],
                x[self.indices.load_quat + 1],
                x[self.indices.load_quat + 2],
                x[self.indices.load_quat + 3],
            )),
            angular_velocity: Vector3::new(
                x[self.indices.load_omega],
                x[self.indices.load_omega + 1],
                x[self.indices.load_omega + 2],
            ),
        }
    }

    /// Extract cable states from EKF estimate
    pub fn get_cable_states(&self) -> MultiCableState {
        let x = &self.ekf.state.x;
        let mut cables = Vec::with_capacity(self.num_cables);

        for i in 0..self.num_cables {
            let dir_idx = self.indices.cable_dir(i);
            let omega_idx = self.indices.cable_omega(i);

            cables.push(CableState {
                direction: Vector3::new(x[dir_idx], x[dir_idx + 1], x[dir_idx + 2]),
                angular_velocity: Vector3::new(x[omega_idx], x[omega_idx + 1], x[omega_idx + 2]),
                angular_acceleration: Vector3::zeros(),
                tension: 0.0, // Not estimated
                tension_rate: 0.0,
            });
        }

        MultiCableState::new(cables)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_estimator_creation() {
        let estimator = LoadCableEstimator::new(3);
        assert_eq!(estimator.num_cables, 3);
        assert_eq!(estimator.indices.total_dim, 13 + 3 * 6);
    }

    #[test]
    fn test_estimator_initialization() {
        let mut estimator = LoadCableEstimator::new(3);

        let load = LoadState {
            position: Vector3::new(1.0, 2.0, 3.0),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        };

        let cables = MultiCableState::new(vec![
            CableState::pointing_down(10.0),
            CableState::pointing_down(10.0),
            CableState::pointing_down(10.0),
        ]);

        estimator.initialize(&load, &cables);

        let estimated_load = estimator.get_load_state();
        assert_relative_eq!(estimated_load.position, load.position, epsilon = 1e-10);
    }

    #[test]
    fn test_estimator_prediction() {
        let mut estimator = LoadCableEstimator::new(3);

        let load = LoadState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1.0, 0.0, 0.0),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        };

        let cables = MultiCableState::new(vec![
            CableState::pointing_down(10.0),
            CableState::pointing_down(10.0),
            CableState::pointing_down(10.0),
        ]);

        estimator.initialize(&load, &cables);
        estimator.predict(0.1);

        let estimated_load = estimator.get_load_state();

        // Position should have moved with velocity
        assert_relative_eq!(estimated_load.position.x, 0.1, epsilon = 0.01);
    }
}
