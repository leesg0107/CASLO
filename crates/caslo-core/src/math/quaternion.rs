//! Quaternion operations for attitude representation
//!
//! Implements quaternion mathematics as used in the paper:
//! - Quaternion derivative: q̇ = 1/2 Λ(q)[0;ω] (Eq. 2)
//! - Quaternion to rotation matrix conversion
//! - Quaternion multiplication

use nalgebra::{Matrix4, UnitQuaternion, Vector3, Vector4, Quaternion};

/// Compute the quaternion derivative given angular velocity
///
/// From Eq. 2: q̇ = 1/2 Λ(q)[0;ω]
///
/// # Arguments
/// * `q` - Current orientation as unit quaternion
/// * `omega` - Angular velocity in body frame [rad/s]
///
/// # Returns
/// Quaternion derivative as Vector4 (w, x, y, z)
pub fn quaternion_derivative(q: &UnitQuaternion<f64>, omega: &Vector3<f64>) -> Vector4<f64> {
    // Λ(q) is the quaternion multiplication matrix
    // For q = [w, x, y, z]^T and p = [0, ωx, ωy, ωz]^T
    // q ⊗ p = Λ(q) * p
    //
    // Λ(q) = [w  -x  -y  -z]
    //        [x   w  -z   y]
    //        [y   z   w  -x]
    //        [z  -y   x   w]

    let w = q.w;
    let x = q.i;
    let y = q.j;
    let z = q.k;

    // [0, ωx, ωy, ωz]^T
    let omega_quat = Vector4::new(0.0, omega.x, omega.y, omega.z);

    // Quaternion multiplication matrix Λ(q)
    let lambda = Matrix4::new(
        w, -x, -y, -z,
        x,  w, -z,  y,
        y,  z,  w, -x,
        z, -y,  x,  w,
    );

    0.5 * lambda * omega_quat
}

/// Integrate quaternion using the derivative
///
/// q_new = normalize(q + q̇ * dt)
///
/// # Arguments
/// * `q` - Current quaternion
/// * `q_dot` - Quaternion derivative
/// * `dt` - Time step [s]
///
/// # Returns
/// New unit quaternion
pub fn integrate_quaternion(
    q: &UnitQuaternion<f64>,
    q_dot: &Vector4<f64>,
    dt: f64,
) -> UnitQuaternion<f64> {
    let q_vec = Vector4::new(q.w, q.i, q.j, q.k);
    let q_new = q_vec + q_dot * dt;

    // Normalize to maintain unit quaternion constraint
    let quat = Quaternion::new(q_new[0], q_new[1], q_new[2], q_new[3]);
    UnitQuaternion::from_quaternion(quat)
}

/// Compute quaternion from axis-angle representation
///
/// # Arguments
/// * `axis` - Rotation axis (will be normalized)
/// * `angle` - Rotation angle [rad]
pub fn quaternion_from_axis_angle(axis: &Vector3<f64>, angle: f64) -> UnitQuaternion<f64> {
    UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(*axis), angle)
}

/// Compute the angular velocity from two quaternions and time step
///
/// Useful for numerical differentiation
///
/// # Arguments
/// * `q1` - Previous quaternion
/// * `q2` - Current quaternion
/// * `dt` - Time step [s]
///
/// # Returns
/// Angular velocity in body frame [rad/s]
pub fn angular_velocity_from_quaternions(
    q1: &UnitQuaternion<f64>,
    q2: &UnitQuaternion<f64>,
    dt: f64,
) -> Vector3<f64> {
    // q2 = q1 ⊗ Δq, so Δq = q1^(-1) ⊗ q2
    let delta_q = q1.inverse() * q2;

    // For small angles, axis-angle approximation
    let angle = 2.0 * delta_q.w.acos();

    if angle.abs() < 1e-10 {
        return Vector3::zeros();
    }

    let axis = Vector3::new(delta_q.i, delta_q.j, delta_q.k);
    let axis_norm = axis.norm();

    if axis_norm < 1e-10 {
        return Vector3::zeros();
    }

    (angle / dt) * (axis / axis_norm)
}

/// Quaternion error between desired and current orientation
///
/// Returns the rotation needed to go from `q_current` to `q_desired`
/// expressed in body frame
pub fn quaternion_error(
    q_desired: &UnitQuaternion<f64>,
    q_current: &UnitQuaternion<f64>,
) -> Vector3<f64> {
    let q_error = q_current.inverse() * q_desired;

    // Extract axis-angle error
    let angle = 2.0 * q_error.w.acos();

    if angle.abs() < 1e-10 {
        return Vector3::zeros();
    }

    let axis = Vector3::new(q_error.i, q_error.j, q_error.k);
    let axis_norm = axis.norm();

    if axis_norm < 1e-10 {
        return Vector3::zeros();
    }

    angle * (axis / axis_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_quaternion_derivative_zero_angular_velocity() {
        let q = UnitQuaternion::identity();
        let omega = Vector3::zeros();

        let q_dot = quaternion_derivative(&q, &omega);

        assert_relative_eq!(q_dot.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_derivative_pure_rotation() {
        let q = UnitQuaternion::identity();
        let omega = Vector3::new(0.0, 0.0, 1.0); // Rotate around z-axis at 1 rad/s

        let q_dot = quaternion_derivative(&q, &omega);

        // For identity quaternion and z-rotation:
        // q̇ = 0.5 * [0, 0, 0, 1]^T
        assert_relative_eq!(q_dot[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(q_dot[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(q_dot[2], 0.0, epsilon = 1e-10);
        assert_relative_eq!(q_dot[3], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_integration() {
        let q = UnitQuaternion::identity();
        let omega = Vector3::new(0.0, 0.0, PI); // 180 deg/s around z
        let dt = 0.01;

        // Integrate for one step
        let q_dot = quaternion_derivative(&q, &omega);
        let q_new = integrate_quaternion(&q, &q_dot, dt);

        // Should still be a unit quaternion
        let quat_norm = (q_new.w.powi(2) + q_new.i.powi(2) + q_new.j.powi(2) + q_new.k.powi(2)).sqrt();
        assert_relative_eq!(quat_norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_from_axis_angle() {
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let angle = PI / 2.0; // 90 degrees

        let q = quaternion_from_axis_angle(&axis, angle);

        // For 90 deg rotation around z: q = [cos(45°), 0, 0, sin(45°)]
        assert_relative_eq!(q.w, (PI / 4.0).cos(), epsilon = 1e-10);
        assert_relative_eq!(q.i, 0.0, epsilon = 1e-10);
        assert_relative_eq!(q.j, 0.0, epsilon = 1e-10);
        assert_relative_eq!(q.k, (PI / 4.0).sin(), epsilon = 1e-10);
    }

    #[test]
    fn test_angular_velocity_recovery() {
        let q1 = UnitQuaternion::identity();
        let omega_expected = Vector3::new(0.1, 0.2, 0.3);
        let dt = 0.001;

        // Integrate to get q2
        let q_dot = quaternion_derivative(&q1, &omega_expected);
        let q2 = integrate_quaternion(&q1, &q_dot, dt);

        // Recover angular velocity
        let omega_recovered = angular_velocity_from_quaternions(&q1, &q2, dt);

        assert_relative_eq!(omega_recovered, omega_expected, epsilon = 1e-3);
    }

    #[test]
    fn test_quaternion_error_identity() {
        let q = UnitQuaternion::identity();
        let error = quaternion_error(&q, &q);

        assert_relative_eq!(error.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_error_90deg() {
        let q_current = UnitQuaternion::identity();
        let q_desired = quaternion_from_axis_angle(&Vector3::new(0.0, 0.0, 1.0), PI / 2.0);

        let error = quaternion_error(&q_desired, &q_current);

        // Error should be approximately [0, 0, π/2]
        assert_relative_eq!(error.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error.z, PI / 2.0, epsilon = 1e-10);
    }
}
