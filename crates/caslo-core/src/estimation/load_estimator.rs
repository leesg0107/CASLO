//! Load-Cable State Estimator
//!
//! Implements the EKF-based load-cable state estimator from the paper.
//! The state vector includes load position, velocity, orientation, angular velocity,
//! cable directions, angular velocities, angular accelerations, tensions, and tension rates.
//!
//! State layout matches Eq. 1:
//! x = [p, v, q, ω, s₁, r₁, ṙ₁, t₁, ṫ₁, ..., sₙ, rₙ, ṙₙ, tₙ, ṫₙ]

use nalgebra::{Vector3, UnitQuaternion, DVector, DMatrix};
use serde::{Deserialize, Serialize};

use super::ExtendedKalmanFilter;
use crate::dynamics::{LoadState, CableState, MultiCableState};

/// Load-cable estimator state indices (Eq. 1)
///
/// State layout:
/// [p(3), v(3), q(4), ω(3), s₁(3), r₁(3), ṙ₁(3), t₁(1), ṫ₁(1), ...]
///
/// Per cable: s(3) + r(3) + ṙ(3) + t(1) + ṫ(1) = 11
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
    /// Cable states start index
    pub cables_start: usize,
    /// Elements per cable: s(3) + r(3) + ṙ(3) + t(1) + ṫ(1) = 11
    pub per_cable_dim: usize,
    /// Total state dimension
    pub total_dim: usize,
}

impl LoadCableStateIndices {
    pub fn new(num_cables: usize) -> Self {
        // State layout matching Eq. 1:
        // [p(3), v(3), q(4), ω(3), s₁(3), r₁(3), ṙ₁(3), t₁(1), ṫ₁(1), ...]
        let load_pos = 0;
        let load_vel = 3;
        let load_quat = 6;
        let load_omega = 10;
        let cables_start = 13;
        let per_cable_dim = 11; // s(3) + r(3) + ṙ(3) + t(1) + ṫ(1)
        let total_dim = cables_start + num_cables * per_cable_dim;

        Self {
            load_pos,
            load_vel,
            load_quat,
            load_omega,
            cables_start,
            per_cable_dim,
            total_dim,
        }
    }

    /// Get cable direction start index for cable i (sᵢ)
    pub fn cable_dir(&self, i: usize) -> usize {
        self.cables_start + i * self.per_cable_dim
    }

    /// Get cable angular velocity start index for cable i (rᵢ)
    pub fn cable_omega(&self, i: usize) -> usize {
        self.cables_start + i * self.per_cable_dim + 3
    }

    /// Get cable angular acceleration start index for cable i (ṙᵢ)
    pub fn cable_alpha(&self, i: usize) -> usize {
        self.cables_start + i * self.per_cable_dim + 6
    }

    /// Get cable tension index for cable i (tᵢ)
    pub fn cable_tension(&self, i: usize) -> usize {
        self.cables_start + i * self.per_cable_dim + 9
    }

    /// Get cable tension rate index for cable i (ṫᵢ)
    pub fn cable_tension_rate(&self, i: usize) -> usize {
        self.cables_start + i * self.per_cable_dim + 10
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
    /// Create a new estimator for n cables (Eq. 1 state layout)
    pub fn new(num_cables: usize) -> Self {
        let indices = LoadCableStateIndices::new(num_cables);
        // Measurements: quadrotor positions (3*n) + cable tensions from accelerometer (n)
        let meas_dim = num_cables * 4;

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
        // Higher noise for cable angular accelerations and tension rates (driven by control)
        for cable_idx in 0..num_cables {
            // ṙᵢ has high process noise (driven by control input γ)
            let alpha_idx = indices.cable_alpha(cable_idx);
            for j in 0..3 {
                q_diag[alpha_idx + j] = 1.0;
            }
            // ṫᵢ has high process noise (driven by control input λ)
            q_diag[indices.cable_tension_rate(cable_idx)] = 10.0;
        }
        ekf.set_process_noise(&q_diag);

        // Set measurement noise
        let mut r_diag = vec![0.001; meas_dim];
        // Tension measurements from accelerometer have higher noise
        for i in 0..num_cables {
            r_diag[num_cables * 3 + i] = 1.0;
        }
        ekf.set_measurement_noise(&r_diag);

        Self {
            ekf,
            indices,
            num_cables,
        }
    }

    /// Initialize estimator with known state (Eq. 1 layout)
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

        // Cable states (Eq. 1: sᵢ, rᵢ, ṙᵢ, tᵢ, ṫᵢ)
        for (i, cable) in cables.cables.iter().enumerate() {
            // Direction sᵢ
            let dir_idx = self.indices.cable_dir(i);
            x[dir_idx] = cable.direction.x;
            x[dir_idx + 1] = cable.direction.y;
            x[dir_idx + 2] = cable.direction.z;

            // Angular velocity rᵢ
            let omega_idx = self.indices.cable_omega(i);
            x[omega_idx] = cable.angular_velocity.x;
            x[omega_idx + 1] = cable.angular_velocity.y;
            x[omega_idx + 2] = cable.angular_velocity.z;

            // Angular acceleration ṙᵢ
            let alpha_idx = self.indices.cable_alpha(i);
            x[alpha_idx] = cable.angular_acceleration.x;
            x[alpha_idx + 1] = cable.angular_acceleration.y;
            x[alpha_idx + 2] = cable.angular_acceleration.z;

            // Tension tᵢ
            x[self.indices.cable_tension(i)] = cable.tension;

            // Tension rate ṫᵢ
            x[self.indices.cable_tension_rate(i)] = cable.tension_rate;
        }

        self.ekf.state.x = x;
        self.ekf.state.p = DMatrix::identity(self.indices.total_dim, self.indices.total_dim) * 0.1;
    }

    /// Prediction step (Eq. 3 dynamics)
    ///
    /// Integrates the cable dynamics:
    /// - ṡᵢ = rᵢ × sᵢ (direction kinematics)
    /// - ṙᵢ = αᵢ (angular velocity from acceleration)
    /// - ṫᵢ = tension_rate
    pub fn predict(&mut self, dt: f64) {
        let indices = self.indices.clone();
        let num_cables = self.num_cables;
        let per_cable = self.indices.per_cable_dim;

        // Prediction model based on Eq. 3
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

            // Cable dynamics (Eq. 3)
            for i in 0..num_cables {
                let dir_idx = indices.cables_start + i * per_cable;
                let omega_idx = dir_idx + 3;
                let alpha_idx = dir_idx + 6;
                let tension_idx = dir_idx + 9;
                let tension_rate_idx = dir_idx + 10;

                // ṡ = r × s (direction kinematics)
                let s = Vector3::new(x[dir_idx], x[dir_idx + 1], x[dir_idx + 2]);
                let r = Vector3::new(x[omega_idx], x[omega_idx + 1], x[omega_idx + 2]);
                let s_dot = r.cross(&s);
                let s_new = s + s_dot * dt;
                let s_norm = s_new.norm();
                x_new[dir_idx] = s_new.x / s_norm;
                x_new[dir_idx + 1] = s_new.y / s_norm;
                x_new[dir_idx + 2] = s_new.z / s_norm;

                // ṙ = α (angular velocity from acceleration)
                let alpha = Vector3::new(x[alpha_idx], x[alpha_idx + 1], x[alpha_idx + 2]);
                let r_new = r + alpha * dt;
                x_new[omega_idx] = r_new.x;
                x_new[omega_idx + 1] = r_new.y;
                x_new[omega_idx + 2] = r_new.z;

                // α stays constant (driven by control input γ in OCP)

                // ṫ = tension_rate
                let t = x[tension_idx];
                let t_dot = x[tension_rate_idx];
                x_new[tension_idx] = (t + t_dot * dt).max(0.0); // Tension can't be negative

                // tension_rate stays constant (driven by control input λ in OCP)
            }

            x_new
        };

        // Jacobian with proper coupling terms
        let f_jac = move |_x: &DVector<f64>| {
            let mut jac = DMatrix::identity(indices.total_dim, indices.total_dim);

            // Position-velocity coupling
            for i in 0..3 {
                jac[(indices.load_pos + i, indices.load_vel + i)] = dt;
            }

            // Cable state couplings
            for i in 0..num_cables {
                let omega_idx = indices.cables_start + i * per_cable + 3;
                let alpha_idx = indices.cables_start + i * per_cable + 6;
                let tension_idx = indices.cables_start + i * per_cable + 9;
                let tension_rate_idx = indices.cables_start + i * per_cable + 10;

                // Angular velocity from acceleration
                for j in 0..3 {
                    jac[(omega_idx + j, alpha_idx + j)] = dt;
                }

                // Tension from tension rate
                jac[(tension_idx, tension_rate_idx)] = dt;
            }

            jac
        };

        self.ekf.predict(f, f_jac);
    }

    /// Update with quadrotor position and tension measurements
    ///
    /// Measurements:
    /// - Quadrotor positions (from GPS/MOCAP)
    /// - Cable tensions (from accelerometer via Eq. 14)
    pub fn update(
        &mut self,
        quad_positions: &[Vector3<f64>],
        cable_tensions: &[f64],
        cable_lengths: &[f64],
        attachment_offsets: &[Vector3<f64>],
    ) {
        let indices = self.indices.clone();
        let indices_jac = self.indices.clone();
        let num_cables = self.num_cables;
        let per_cable = self.indices.per_cable_dim;
        let per_cable_jac = self.indices.per_cable_dim;
        let lengths = cable_lengths.to_vec();
        let lengths_jac = cable_lengths.to_vec();
        let offsets = attachment_offsets.to_vec();

        // Measurement function:
        // - Quadrotor positions: pᵢ = p + R(q)ρᵢ - lᵢsᵢ (Eq. 5)
        // - Cable tensions: tᵢ (direct measurement from accelerometer)
        let h = move |x: &DVector<f64>| {
            let mut z = DVector::zeros(num_cables * 4); // 3 for position + 1 for tension

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
                let dir_idx = indices.cables_start + i * per_cable;
                let tension_idx = indices.cables_start + i * per_cable + 9;

                let cable_dir = Vector3::new(
                    x[dir_idx],
                    x[dir_idx + 1],
                    x[dir_idx + 2],
                );

                // Quadrotor position: pᵢ = p + R(q)ρᵢ - lᵢsᵢ
                let quad_pos = load_pos + q * offsets[i] - lengths[i] * cable_dir;

                z[i * 3] = quad_pos.x;
                z[i * 3 + 1] = quad_pos.y;
                z[i * 3 + 2] = quad_pos.z;

                // Tension measurement
                z[num_cables * 3 + i] = x[tension_idx];
            }

            z
        };

        // Measurement Jacobian
        let h_jac = move |_x: &DVector<f64>| {
            let mut jac = DMatrix::zeros(num_cables * 4, indices_jac.total_dim);

            for i in 0..num_cables {
                // Position affects quad positions
                for j in 0..3 {
                    jac[(i * 3 + j, indices_jac.load_pos + j)] = 1.0;
                }

                // Cable direction affects quad position
                let dir_idx = indices_jac.cables_start + i * per_cable_jac;
                for j in 0..3 {
                    jac[(i * 3 + j, dir_idx + j)] = -lengths_jac[i];
                }

                // Tension measurement directly observes tension state
                let tension_idx = indices_jac.cables_start + i * per_cable_jac + 9;
                jac[(num_cables * 3 + i, tension_idx)] = 1.0;
            }

            jac
        };

        // Build measurement vector
        let mut z = DVector::zeros(self.num_cables * 4);
        for (i, pos) in quad_positions.iter().enumerate() {
            z[i * 3] = pos.x;
            z[i * 3 + 1] = pos.y;
            z[i * 3 + 2] = pos.z;
        }
        for (i, &tension) in cable_tensions.iter().enumerate() {
            z[self.num_cables * 3 + i] = tension;
        }

        self.ekf.update(&z, h, h_jac);
    }

    /// Update with only quadrotor positions (backward compatible)
    pub fn update_positions_only(
        &mut self,
        quad_positions: &[Vector3<f64>],
        cable_lengths: &[f64],
        attachment_offsets: &[Vector3<f64>],
    ) {
        // Use zero tension measurements with high uncertainty
        let cable_tensions = vec![0.0; self.num_cables];
        self.update(quad_positions, &cable_tensions, cable_lengths, attachment_offsets);
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

    /// Extract cable states from EKF estimate (Eq. 1)
    pub fn get_cable_states(&self) -> MultiCableState {
        let x = &self.ekf.state.x;
        let mut cables = Vec::with_capacity(self.num_cables);

        for i in 0..self.num_cables {
            let dir_idx = self.indices.cable_dir(i);
            let omega_idx = self.indices.cable_omega(i);
            let alpha_idx = self.indices.cable_alpha(i);
            let tension_idx = self.indices.cable_tension(i);
            let tension_rate_idx = self.indices.cable_tension_rate(i);

            cables.push(CableState {
                direction: Vector3::new(x[dir_idx], x[dir_idx + 1], x[dir_idx + 2]),
                angular_velocity: Vector3::new(x[omega_idx], x[omega_idx + 1], x[omega_idx + 2]),
                angular_acceleration: Vector3::new(x[alpha_idx], x[alpha_idx + 1], x[alpha_idx + 2]),
                tension: x[tension_idx],
                tension_rate: x[tension_rate_idx],
            });
        }

        MultiCableState::new(cables)
    }

    /// Get estimated tensions for all cables
    pub fn get_tensions(&self) -> Vec<f64> {
        let x = &self.ekf.state.x;
        (0..self.num_cables)
            .map(|i| x[self.indices.cable_tension(i)])
            .collect()
    }

    /// Get estimated tension rates for all cables
    pub fn get_tension_rates(&self) -> Vec<f64> {
        let x = &self.ekf.state.x;
        (0..self.num_cables)
            .map(|i| x[self.indices.cable_tension_rate(i)])
            .collect()
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
        // State dim: 13 (load) + 11*3 (cables) = 46
        assert_eq!(estimator.indices.total_dim, 13 + 3 * 11);
    }

    #[test]
    fn test_state_indices() {
        let indices = LoadCableStateIndices::new(3);
        // Check cable indices
        assert_eq!(indices.cable_dir(0), 13);
        assert_eq!(indices.cable_omega(0), 16);
        assert_eq!(indices.cable_alpha(0), 19);
        assert_eq!(indices.cable_tension(0), 22);
        assert_eq!(indices.cable_tension_rate(0), 23);
        // Second cable
        assert_eq!(indices.cable_dir(1), 24);
        assert_eq!(indices.cable_tension(1), 33);
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

        // Check tension is initialized
        let tensions = estimator.get_tensions();
        assert_eq!(tensions.len(), 3);
        assert_relative_eq!(tensions[0], 10.0, epsilon = 1e-10);
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

    #[test]
    fn test_tension_estimation() {
        let mut estimator = LoadCableEstimator::new(3);

        let load = LoadState {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        };

        let cables = MultiCableState::new(vec![
            CableState::pointing_down(5.0),
            CableState::pointing_down(10.0),
            CableState::pointing_down(15.0),
        ]);

        estimator.initialize(&load, &cables);

        let tensions = estimator.get_tensions();
        assert_relative_eq!(tensions[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(tensions[1], 10.0, epsilon = 1e-10);
        assert_relative_eq!(tensions[2], 15.0, epsilon = 1e-10);
    }
}
