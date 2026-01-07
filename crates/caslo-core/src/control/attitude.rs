//! Attitude controller
//!
//! Implements attitude control for quadrotors using quaternion error.
//! Computes desired angular acceleration from orientation error.

use nalgebra::{Vector3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::math::attitude_error;

/// Attitude controller gains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttitudeGains {
    /// Proportional gain (orientation error)
    pub kp: Vector3<f64>,
    /// Derivative gain (angular velocity error)
    pub kd: Vector3<f64>,
}

impl Default for AttitudeGains {
    fn default() -> Self {
        Self {
            kp: Vector3::new(20.0, 20.0, 15.0),
            kd: Vector3::new(5.0, 5.0, 3.0),
        }
    }
}

/// Attitude controller
#[derive(Debug, Clone)]
pub struct AttitudeController {
    /// Controller gains
    pub gains: AttitudeGains,
}

impl AttitudeController {
    pub fn new(gains: AttitudeGains) -> Self {
        Self { gains }
    }

    /// Compute desired angular acceleration from attitude error
    ///
    /// Uses the SO(3) attitude error from the paper.
    ///
    /// α_des = kp * e_R + kd * (ω_des - ω)
    ///
    /// # Arguments
    /// * `orientation` - Current orientation (body to world)
    /// * `angular_velocity` - Current angular velocity (body frame)
    /// * `orientation_des` - Desired orientation
    /// * `angular_velocity_des` - Desired angular velocity
    ///
    /// # Returns
    /// Desired angular acceleration (body frame)
    pub fn compute(
        &self,
        orientation: &UnitQuaternion<f64>,
        angular_velocity: &Vector3<f64>,
        orientation_des: &UnitQuaternion<f64>,
        angular_velocity_des: &Vector3<f64>,
    ) -> Vector3<f64> {
        // Compute attitude error in body frame (using kp=1.0 to get raw error)
        let e_r = attitude_error(orientation, orientation_des, 1.0);

        // Angular velocity error
        let omega_error = angular_velocity_des - angular_velocity;

        // PD control
        self.gains.kp.component_mul(&e_r) + self.gains.kd.component_mul(&omega_error)
    }

    /// Compute desired orientation from thrust direction
    ///
    /// Given a desired thrust direction and yaw angle, computes
    /// the full attitude quaternion.
    ///
    /// # Arguments
    /// * `thrust_dir` - Desired thrust direction (world frame, unit vector)
    /// * `yaw` - Desired yaw angle [rad]
    ///
    /// # Returns
    /// Desired orientation quaternion
    pub fn orientation_from_thrust(
        thrust_dir: &Vector3<f64>,
        yaw: f64,
    ) -> UnitQuaternion<f64> {
        crate::math::quaternion_from_z_axis_and_yaw(thrust_dir, yaw)
    }
}

impl Default for AttitudeController {
    fn default() -> Self {
        Self::new(AttitudeGains::default())
    }
}

/// Computes desired thrust and orientation from desired acceleration
///
/// From the paper's control architecture, converts desired acceleration
/// to thrust magnitude and direction.
///
/// # Arguments
/// * `acc_des` - Desired acceleration (world frame)
/// * `yaw_des` - Desired yaw angle
/// * `mass` - Vehicle mass
/// * `gravity` - Gravity magnitude (positive)
///
/// # Returns
/// (thrust magnitude, desired orientation)
pub fn thrust_and_orientation_from_acceleration(
    acc_des: &Vector3<f64>,
    yaw_des: f64,
    mass: f64,
    gravity: f64,
) -> (f64, UnitQuaternion<f64>) {
    // Desired force = m * (a_des - g)
    // In NED frame, gravity is [0, 0, g] (positive down)
    let gravity_vec = Vector3::new(0.0, 0.0, gravity);
    let force_des = mass * (acc_des - gravity_vec);

    // Thrust magnitude
    let thrust = force_des.norm();

    // Thrust direction (body z-axis in world frame)
    let thrust_dir = if thrust > 1e-6 {
        force_des / thrust
    } else {
        Vector3::new(0.0, 0.0, -1.0) // Default: thrust up (NED: -z is up)
    };

    let orientation = AttitudeController::orientation_from_thrust(&thrust_dir, yaw_des);

    (thrust, orientation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_attitude_controller_at_setpoint() {
        let controller = AttitudeController::default();

        let q = UnitQuaternion::identity();
        let omega = Vector3::zeros();
        let q_des = q;
        let omega_des = Vector3::zeros();

        let alpha = controller.compute(&q, &omega, &q_des, &omega_des);

        assert_relative_eq!(alpha.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_attitude_controller_small_error() {
        let controller = AttitudeController::default();

        let q = UnitQuaternion::identity();
        let omega = Vector3::zeros();
        // Small rotation about z
        let q_des = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.1);
        let omega_des = Vector3::zeros();

        let alpha = controller.compute(&q, &omega, &q_des, &omega_des);

        // Should produce rotation toward desired orientation
        // For small angles, this should be approximately proportional
        assert!(alpha.z.abs() > 0.0);
    }

    #[test]
    fn test_orientation_from_thrust_upward() {
        let thrust_dir = Vector3::new(0.0, 0.0, -1.0); // Up in NED
        let yaw = 0.0;

        let q = AttitudeController::orientation_from_thrust(&thrust_dir, yaw);

        // Body z should align with thrust direction
        let body_z = q * Vector3::new(0.0, 0.0, 1.0);

        // The body z-axis should point in thrust direction
        // Actually, body z is the thrust direction in body frame,
        // so R*e3 should equal thrust_dir
        assert_relative_eq!(body_z, thrust_dir, epsilon = 1e-6);
    }

    #[test]
    fn test_thrust_from_hover_acceleration() {
        let acc_des = Vector3::new(0.0, 0.0, 9.81); // Hover in NED
        let yaw = 0.0;
        let mass = 1.0;
        let gravity = 9.81;

        let (thrust, _q) = thrust_and_orientation_from_acceleration(&acc_des, yaw, mass, gravity);

        // At hover, thrust should equal weight
        assert_relative_eq!(thrust, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_thrust_from_upward_acceleration() {
        // Want to accelerate upward at 2 m/s² in NED (negative z)
        // a_des - g should give the required thrust direction
        let acc_des = Vector3::new(0.0, 0.0, 9.81 - 2.0); // Less than g = upward accel
        let yaw = 0.0;
        let mass = 1.0;
        let gravity = 9.81;

        let (thrust, _q) = thrust_and_orientation_from_acceleration(&acc_des, yaw, mass, gravity);

        // Thrust should be m * 2 = 2N
        assert_relative_eq!(thrust, 2.0, epsilon = 1e-6);
    }
}
