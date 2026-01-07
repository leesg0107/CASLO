//! Position controller
//!
//! Implements position tracking control for both load and quadrotors.
//! Computes desired acceleration from position and velocity errors.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Position controller gains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionGains {
    /// Proportional gain
    pub kp: Vector3<f64>,
    /// Derivative gain
    pub kd: Vector3<f64>,
    /// Integral gain
    pub ki: Vector3<f64>,
}

impl Default for PositionGains {
    fn default() -> Self {
        Self {
            kp: Vector3::new(6.0, 6.0, 8.0),
            kd: Vector3::new(4.0, 4.0, 5.0),
            ki: Vector3::new(0.1, 0.1, 0.1),
        }
    }
}

/// Position controller with PID
#[derive(Debug, Clone)]
pub struct PositionController {
    /// Controller gains
    pub gains: PositionGains,
    /// Integral error accumulator
    integral_error: Vector3<f64>,
    /// Maximum integral error magnitude
    max_integral: f64,
}

impl PositionController {
    pub fn new(gains: PositionGains) -> Self {
        Self {
            gains,
            integral_error: Vector3::zeros(),
            max_integral: 5.0,
        }
    }

    /// Compute desired acceleration from position error
    ///
    /// a_des = kp * (p_des - p) + kd * (v_des - v) + ki * ∫error + a_ff
    ///
    /// # Arguments
    /// * `pos` - Current position
    /// * `vel` - Current velocity
    /// * `pos_des` - Desired position
    /// * `vel_des` - Desired velocity
    /// * `acc_ff` - Feedforward acceleration
    /// * `dt` - Time step for integral update
    pub fn compute(
        &mut self,
        pos: &Vector3<f64>,
        vel: &Vector3<f64>,
        pos_des: &Vector3<f64>,
        vel_des: &Vector3<f64>,
        acc_ff: &Vector3<f64>,
        dt: f64,
    ) -> Vector3<f64> {
        let pos_error = pos_des - pos;
        let vel_error = vel_des - vel;

        // Update integral with anti-windup
        self.integral_error += pos_error * dt;
        let int_mag = self.integral_error.norm();
        if int_mag > self.max_integral {
            self.integral_error *= self.max_integral / int_mag;
        }

        // PID control
        let p_term = self.gains.kp.component_mul(&pos_error);
        let d_term = self.gains.kd.component_mul(&vel_error);
        let i_term = self.gains.ki.component_mul(&self.integral_error);

        p_term + d_term + i_term + acc_ff
    }

    /// Reset integral accumulator
    pub fn reset(&mut self) {
        self.integral_error = Vector3::zeros();
    }
}

impl Default for PositionController {
    fn default() -> Self {
        Self::new(PositionGains::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_position_controller_at_setpoint() {
        let mut controller = PositionController::default();

        let pos = Vector3::new(1.0, 2.0, 3.0);
        let vel = Vector3::new(0.0, 0.0, 0.0);
        let pos_des = pos;
        let vel_des = vel;
        let acc_ff = Vector3::zeros();

        let acc = controller.compute(&pos, &vel, &pos_des, &vel_des, &acc_ff, 0.01);

        // At setpoint with zero velocity, output should be zero (before integral kicks in)
        assert_relative_eq!(acc.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_position_controller_proportional() {
        let mut controller = PositionController::default();
        controller.gains.ki = Vector3::zeros(); // Disable integral

        let pos = Vector3::new(0.0, 0.0, 0.0);
        let vel = Vector3::zeros();
        let pos_des = Vector3::new(1.0, 0.0, 0.0);
        let vel_des = Vector3::zeros();
        let acc_ff = Vector3::zeros();

        let acc = controller.compute(&pos, &vel, &pos_des, &vel_des, &acc_ff, 0.01);

        // Should produce acceleration toward setpoint
        assert!(acc.x > 0.0);
        assert_relative_eq!(acc.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(acc.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_position_controller_derivative() {
        let mut controller = PositionController::default();
        controller.gains.kp = Vector3::zeros();
        controller.gains.ki = Vector3::zeros();

        let pos = Vector3::zeros();
        let vel = Vector3::new(1.0, 0.0, 0.0); // Moving right
        let pos_des = Vector3::zeros();
        let vel_des = Vector3::zeros(); // Should stop
        let acc_ff = Vector3::zeros();

        let acc = controller.compute(&pos, &vel, &pos_des, &vel_des, &acc_ff, 0.01);

        // Should produce deceleration
        assert!(acc.x < 0.0);
    }

    #[test]
    fn test_integral_windup() {
        let mut controller = PositionController::new(PositionGains {
            kp: Vector3::zeros(),
            kd: Vector3::zeros(),
            ki: Vector3::new(1.0, 1.0, 1.0),
        });
        controller.max_integral = 1.0;

        let pos = Vector3::zeros();
        let vel = Vector3::zeros();
        let pos_des = Vector3::new(100.0, 0.0, 0.0); // Large error
        let vel_des = Vector3::zeros();
        let acc_ff = Vector3::zeros();

        // Run many iterations
        for _ in 0..1000 {
            controller.compute(&pos, &vel, &pos_des, &vel_des, &acc_ff, 0.1);
        }

        // Integral should be clamped
        assert!(controller.integral_error.norm() <= controller.max_integral + 1e-10);
    }
}
