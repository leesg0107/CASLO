//! Extended Kalman Filter (EKF) implementation
//!
//! A generic EKF implementation that can be specialized for
//! load-cable state estimation.

use nalgebra::{DMatrix, DVector, RealField};
use serde::{Deserialize, Serialize};

/// EKF state container
#[derive(Debug, Clone)]
pub struct EkfState {
    /// State estimate
    pub x: DVector<f64>,
    /// Covariance matrix
    pub p: DMatrix<f64>,
}

impl EkfState {
    pub fn new(state_dim: usize) -> Self {
        Self {
            x: DVector::zeros(state_dim),
            p: DMatrix::identity(state_dim, state_dim),
        }
    }

    pub fn with_initial(x: DVector<f64>, p: DMatrix<f64>) -> Self {
        assert_eq!(x.len(), p.nrows());
        assert_eq!(p.nrows(), p.ncols());
        Self { x, p }
    }
}

/// EKF noise parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EkfNoiseParams {
    /// Process noise covariance diagonal
    pub q_diag: Vec<f64>,
    /// Measurement noise covariance diagonal
    pub r_diag: Vec<f64>,
}

/// Generic Extended Kalman Filter
#[derive(Debug, Clone)]
pub struct ExtendedKalmanFilter {
    /// Current state estimate and covariance
    pub state: EkfState,
    /// Process noise covariance (Q)
    pub q: DMatrix<f64>,
    /// Measurement noise covariance (R)
    pub r: DMatrix<f64>,
}

impl ExtendedKalmanFilter {
    /// Create a new EKF
    pub fn new(state_dim: usize, meas_dim: usize) -> Self {
        Self {
            state: EkfState::new(state_dim),
            q: DMatrix::identity(state_dim, state_dim) * 0.01,
            r: DMatrix::identity(meas_dim, meas_dim) * 0.1,
        }
    }

    /// Set process noise covariance
    pub fn set_process_noise(&mut self, q_diag: &[f64]) {
        for (i, &val) in q_diag.iter().enumerate() {
            if i < self.q.nrows() {
                self.q[(i, i)] = val;
            }
        }
    }

    /// Set measurement noise covariance
    pub fn set_measurement_noise(&mut self, r_diag: &[f64]) {
        for (i, &val) in r_diag.iter().enumerate() {
            if i < self.r.nrows() {
                self.r[(i, i)] = val;
            }
        }
    }

    /// Prediction step
    ///
    /// # Arguments
    /// * `f` - State transition function: x_next = f(x)
    /// * `f_jacobian` - Jacobian of f at current state
    pub fn predict<F, J>(&mut self, f: F, f_jacobian: J)
    where
        F: Fn(&DVector<f64>) -> DVector<f64>,
        J: Fn(&DVector<f64>) -> DMatrix<f64>,
    {
        // State prediction
        self.state.x = f(&self.state.x);

        // Covariance prediction: P = F * P * F' + Q
        let f_mat = f_jacobian(&self.state.x);
        self.state.p = &f_mat * &self.state.p * f_mat.transpose() + &self.q;
    }

    /// Update step with measurement
    ///
    /// # Arguments
    /// * `z` - Measurement vector
    /// * `h` - Measurement function: z = h(x)
    /// * `h_jacobian` - Jacobian of h at current state
    pub fn update<H, J>(&mut self, z: &DVector<f64>, h: H, h_jacobian: J)
    where
        H: Fn(&DVector<f64>) -> DVector<f64>,
        J: Fn(&DVector<f64>) -> DMatrix<f64>,
    {
        // Measurement prediction
        let z_pred = h(&self.state.x);

        // Innovation
        let y = z - z_pred;

        // Jacobian
        let h_mat = h_jacobian(&self.state.x);

        // Innovation covariance: S = H * P * H' + R
        let s = &h_mat * &self.state.p * h_mat.transpose() + &self.r;

        // Kalman gain: K = P * H' * S^(-1)
        let s_inv = s.clone().try_inverse().unwrap_or_else(|| {
            // If not invertible, use pseudoinverse or regularized inverse
            DMatrix::identity(s.nrows(), s.ncols())
        });
        let k = &self.state.p * h_mat.transpose() * s_inv;

        // State update
        self.state.x = &self.state.x + &k * y;

        // Covariance update (Joseph form for numerical stability)
        let i = DMatrix::identity(self.state.p.nrows(), self.state.p.ncols());
        let i_kh = &i - &k * &h_mat;
        self.state.p = &i_kh * &self.state.p * i_kh.transpose() + &k * &self.r * k.transpose();
    }

    /// Get current state estimate
    pub fn state_estimate(&self) -> &DVector<f64> {
        &self.state.x
    }

    /// Get current covariance
    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.state.p
    }

    /// Get state dimension
    pub fn state_dim(&self) -> usize {
        self.state.x.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ekf_creation() {
        let ekf = ExtendedKalmanFilter::new(4, 2);
        assert_eq!(ekf.state_dim(), 4);
        assert_eq!(ekf.q.nrows(), 4);
        assert_eq!(ekf.r.nrows(), 2);
    }

    #[test]
    fn test_ekf_linear_prediction() {
        let mut ekf = ExtendedKalmanFilter::new(2, 1);
        ekf.state.x = DVector::from_vec(vec![1.0, 0.5]);
        ekf.state.p = DMatrix::identity(2, 2);

        let dt = 0.1;

        // Simple constant velocity model: x' = x + v*dt
        let f = |x: &DVector<f64>| {
            DVector::from_vec(vec![x[0] + x[1] * dt, x[1]])
        };

        let f_jac = |_x: &DVector<f64>| {
            DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0])
        };

        ekf.predict(f, f_jac);

        // x should have moved
        assert_relative_eq!(ekf.state.x[0], 1.0 + 0.5 * dt, epsilon = 1e-10);
        assert_relative_eq!(ekf.state.x[1], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_ekf_update() {
        let mut ekf = ExtendedKalmanFilter::new(1, 1);
        ekf.state.x = DVector::from_vec(vec![0.0]);
        ekf.state.p = DMatrix::from_vec(1, 1, vec![1.0]);
        ekf.r = DMatrix::from_vec(1, 1, vec![1.0]);

        // Measurement of actual state
        let z = DVector::from_vec(vec![1.0]);

        let h = |x: &DVector<f64>| x.clone();
        let h_jac = |_x: &DVector<f64>| DMatrix::identity(1, 1);

        ekf.update(&z, h, h_jac);

        // State should move toward measurement
        // With equal P and R, should be halfway
        assert_relative_eq!(ekf.state.x[0], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_ekf_convergence() {
        let mut ekf = ExtendedKalmanFilter::new(1, 1);
        ekf.state.x = DVector::from_vec(vec![0.0]);
        ekf.state.p = DMatrix::from_vec(1, 1, vec![10.0]); // High initial uncertainty
        ekf.r = DMatrix::from_vec(1, 1, vec![0.1]); // Low measurement noise
        ekf.q = DMatrix::from_vec(1, 1, vec![0.01]); // Low process noise

        let true_state = 5.0;
        let h = |x: &DVector<f64>| x.clone();
        let h_jac = |_x: &DVector<f64>| DMatrix::identity(1, 1);
        let f = |x: &DVector<f64>| x.clone();
        let f_jac = |_x: &DVector<f64>| DMatrix::identity(1, 1);

        // Run multiple updates with noisy measurements
        for _ in 0..50 {
            ekf.predict(f, f_jac);
            let z = DVector::from_vec(vec![true_state + 0.1]); // Slightly noisy
            ekf.update(&z, h, h_jac);
        }

        // Should converge to true state
        assert_relative_eq!(ekf.state.x[0], true_state, epsilon = 0.5);
    }
}
