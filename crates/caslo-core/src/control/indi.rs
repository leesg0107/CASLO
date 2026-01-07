//! INDI (Incremental Nonlinear Dynamic Inversion) Controller
//!
//! Implements the INDI trajectory tracking controller from the paper.
//! From Supplementary Materials Eq. (S9):
//!
//! τ_des = τ_f + J(α_des - ω̇_f)
//!
//! Where:
//! - τ_f: Current (filtered) torque
//! - α_des: Desired angular acceleration
//! - ω̇_f: Current (filtered) angular acceleration
//! - J: Inertia tensor

use nalgebra::{Vector3, Matrix3};
use serde::{Deserialize, Serialize};

/// INDI controller parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndiParams {
    /// Filter cutoff frequency for angular acceleration [rad/s]
    pub omega_filter_freq: f64,
    /// Filter cutoff frequency for torque [rad/s]
    pub torque_filter_freq: f64,
}

impl Default for IndiParams {
    fn default() -> Self {
        Self {
            omega_filter_freq: 50.0,  // 50 rad/s ≈ 8 Hz
            torque_filter_freq: 50.0,
        }
    }
}

/// First-order low-pass filter
#[derive(Debug, Clone)]
pub struct LowPassFilter {
    /// Filter time constant
    tau: f64,
    /// Current filtered value
    value: Vector3<f64>,
}

impl LowPassFilter {
    pub fn new(cutoff_freq: f64) -> Self {
        Self {
            tau: 1.0 / cutoff_freq,
            value: Vector3::zeros(),
        }
    }

    /// Update filter with new measurement
    pub fn update(&mut self, measurement: &Vector3<f64>, dt: f64) -> Vector3<f64> {
        let alpha = dt / (self.tau + dt);
        self.value = self.value * (1.0 - alpha) + measurement * alpha;
        self.value
    }

    /// Get current filtered value
    pub fn value(&self) -> Vector3<f64> {
        self.value
    }

    /// Reset filter state
    pub fn reset(&mut self, value: Vector3<f64>) {
        self.value = value;
    }
}

/// INDI attitude controller
#[derive(Debug, Clone)]
pub struct IndiController {
    /// Controller parameters
    pub params: IndiParams,
    /// Angular acceleration filter
    omega_dot_filter: LowPassFilter,
    /// Applied torque filter
    torque_filter: LowPassFilter,
    /// Previous angular velocity (for differentiation)
    prev_omega: Vector3<f64>,
    /// Previous control torque
    prev_torque: Vector3<f64>,
}

impl IndiController {
    pub fn new(params: IndiParams) -> Self {
        Self {
            omega_dot_filter: LowPassFilter::new(params.omega_filter_freq),
            torque_filter: LowPassFilter::new(params.torque_filter_freq),
            prev_omega: Vector3::zeros(),
            prev_torque: Vector3::zeros(),
            params,
        }
    }

    /// Compute INDI torque command
    ///
    /// Implements Eq. (S9): τ_des = τ_f + J(α_des - ω̇_f)
    ///
    /// # Arguments
    /// * `omega` - Current angular velocity measurement (body frame)
    /// * `alpha_des` - Desired angular acceleration (body frame)
    /// * `inertia` - Inertia tensor
    /// * `dt` - Time step
    ///
    /// # Returns
    /// Desired torque command (body frame)
    pub fn compute(
        &mut self,
        omega: &Vector3<f64>,
        alpha_des: &Vector3<f64>,
        inertia: &Matrix3<f64>,
        dt: f64,
    ) -> Vector3<f64> {
        // Estimate angular acceleration from measurements
        let omega_dot_raw = (omega - self.prev_omega) / dt;
        self.prev_omega = *omega;

        // Filter the angular acceleration
        let omega_dot_f = self.omega_dot_filter.update(&omega_dot_raw, dt);

        // Filter the previous torque
        let torque_f = self.torque_filter.value();

        // INDI control law: τ_des = τ_f + J(α_des - ω̇_f)
        let torque_des = torque_f + inertia * (alpha_des - omega_dot_f);

        // Update torque filter with new command
        self.torque_filter.update(&torque_des, dt);
        self.prev_torque = torque_des;

        torque_des
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.omega_dot_filter.reset(Vector3::zeros());
        self.torque_filter.reset(Vector3::zeros());
        self.prev_omega = Vector3::zeros();
        self.prev_torque = Vector3::zeros();
    }

    /// Initialize with current state
    pub fn initialize(&mut self, omega: &Vector3<f64>, torque: &Vector3<f64>) {
        self.prev_omega = *omega;
        self.prev_torque = *torque;
        self.torque_filter.reset(*torque);
    }
}

impl Default for IndiController {
    fn default() -> Self {
        Self::new(IndiParams::default())
    }
}

/// Full INDI trajectory tracking controller
///
/// Combines position control, attitude control, and INDI for torque
#[derive(Debug, Clone)]
pub struct IndiTrajectoryController {
    /// INDI inner loop controller
    pub indi: IndiController,
    /// Position gains
    pub position_gains: super::PositionGains,
    /// Attitude gains
    pub attitude_gains: super::AttitudeGains,
}

impl IndiTrajectoryController {
    pub fn new(
        indi_params: IndiParams,
        position_gains: super::PositionGains,
        attitude_gains: super::AttitudeGains,
    ) -> Self {
        Self {
            indi: IndiController::new(indi_params),
            position_gains,
            attitude_gains,
        }
    }
}

impl Default for IndiTrajectoryController {
    fn default() -> Self {
        Self::new(
            IndiParams::default(),
            super::PositionGains::default(),
            super::AttitudeGains::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_low_pass_filter() {
        let mut filter = LowPassFilter::new(10.0);

        // Step response
        let step = Vector3::new(1.0, 0.0, 0.0);
        let dt = 0.01;

        let mut filtered = Vector3::zeros();
        for _ in 0..100 {
            filtered = filter.update(&step, dt);
        }

        // After settling, should approach the step value
        assert_relative_eq!(filtered.x, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_indi_zero_error() {
        let mut indi = IndiController::default();
        let inertia = Matrix3::from_diagonal(&Vector3::new(0.01, 0.01, 0.02));
        let dt = 0.003; // 300 Hz

        // Already at desired state
        let omega = Vector3::zeros();
        let alpha_des = Vector3::zeros();

        // Initialize to get stable filter state
        indi.initialize(&omega, &Vector3::zeros());

        // Run a few iterations
        let mut torque = Vector3::zeros();
        for _ in 0..10 {
            torque = indi.compute(&omega, &alpha_des, &inertia, dt);
        }

        // Torque should be near zero when no angular acceleration needed
        assert!(torque.norm() < 0.1);
    }

    #[test]
    fn test_indi_step_response() {
        let mut indi = IndiController::default();
        let inertia = Matrix3::from_diagonal(&Vector3::new(0.01, 0.01, 0.02));
        let dt = 0.003;

        let omega = Vector3::zeros();
        let alpha_des = Vector3::new(10.0, 0.0, 0.0); // Step in desired accel

        indi.initialize(&omega, &Vector3::zeros());

        let torque = indi.compute(&omega, &alpha_des, &inertia, dt);

        // Initial torque should be approximately J * alpha_des
        let expected = inertia * alpha_des;
        assert_relative_eq!(torque.x, expected.x, epsilon = 0.01);
    }

    #[test]
    fn test_filter_reset() {
        let mut filter = LowPassFilter::new(10.0);

        // Run some updates
        filter.update(&Vector3::new(1.0, 0.0, 0.0), 0.01);
        filter.update(&Vector3::new(1.0, 0.0, 0.0), 0.01);

        // Reset
        filter.reset(Vector3::new(5.0, 5.0, 5.0));

        assert_relative_eq!(filter.value(), Vector3::new(5.0, 5.0, 5.0), epsilon = 1e-10);
    }
}
