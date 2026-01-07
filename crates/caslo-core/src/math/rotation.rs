//! SO(3) rotation utilities
//!
//! Provides rotation matrix operations and conversions used in the dynamics equations.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

/// Skew-symmetric matrix from vector (hat operator)
///
/// For v = [x, y, z]^T:
/// ```text
/// [v]× = [ 0  -z   y]
///        [ z   0  -x]
///        [-y   x   0]
/// ```
///
/// Used in: ω × Jω and cross products
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0,
    )
}

/// Cross product using skew-symmetric matrix
///
/// a × b = [a]× * b
pub fn cross_matrix(a: &Vector3<f64>, b: &Vector3<f64>) -> Vector3<f64> {
    skew(a) * b
}

/// Rotation matrix from quaternion
///
/// Extracts the 3x3 rotation matrix R(q) ∈ SO(3) from unit quaternion
pub fn rotation_matrix_from_quaternion(q: &UnitQuaternion<f64>) -> Matrix3<f64> {
    *q.to_rotation_matrix().matrix()
}

/// Rotate a vector by quaternion
///
/// v' = R(q) * v
pub fn rotate_vector(q: &UnitQuaternion<f64>, v: &Vector3<f64>) -> Vector3<f64> {
    q.transform_vector(v)
}

/// Rotate a vector by the inverse of quaternion
///
/// v' = R(q)^T * v
pub fn rotate_vector_inverse(q: &UnitQuaternion<f64>, v: &Vector3<f64>) -> Vector3<f64> {
    q.inverse_transform_vector(v)
}

/// Body z-axis in world frame (thrust direction for quadrotor)
///
/// z_body = R(q) * [0, 0, 1]^T
pub fn body_z_axis(q: &UnitQuaternion<f64>) -> Vector3<f64> {
    rotate_vector(q, &Vector3::new(0.0, 0.0, 1.0))
}

/// Compute rotation matrix that aligns z-axis with given direction
///
/// Useful for computing desired attitude from thrust direction
///
/// # Arguments
/// * `z_desired` - Desired z-axis direction (will be normalized)
/// * `yaw` - Desired yaw angle [rad]
///
/// # Returns
/// Unit quaternion with z-axis aligned to `z_desired`
pub fn quaternion_from_z_axis_and_yaw(z_desired: &Vector3<f64>, yaw: f64) -> UnitQuaternion<f64> {
    let z = z_desired.normalize();

    // Choose x-axis perpendicular to z and in the horizontal plane
    // x_c = [cos(yaw), sin(yaw), 0]^T
    let x_c = Vector3::new(yaw.cos(), yaw.sin(), 0.0);

    // y = z × x_c (normalized)
    let y = z.cross(&x_c);
    let y_norm = y.norm();

    let y = if y_norm > 1e-6 {
        y / y_norm
    } else {
        // z is vertical, use yaw directly
        Vector3::new(-yaw.sin(), yaw.cos(), 0.0)
    };

    // x = y × z
    let x = y.cross(&z);

    // Construct rotation matrix
    let rot = Matrix3::from_columns(&[x, y, z]);

    UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(rot))
}

/// Compute the angular velocity that rotates from current to desired orientation
///
/// This is the "attitude error" in axis-angle form, scaled by gain
///
/// # Arguments
/// * `q_current` - Current orientation
/// * `q_desired` - Desired orientation
/// * `kp` - Proportional gain
pub fn attitude_error(
    q_current: &UnitQuaternion<f64>,
    q_desired: &UnitQuaternion<f64>,
    kp: f64,
) -> Vector3<f64> {
    let q_error = q_current.inverse() * q_desired;

    // Convert to axis-angle
    let angle = 2.0 * q_error.w.clamp(-1.0, 1.0).acos();

    if angle.abs() < 1e-10 {
        return Vector3::zeros();
    }

    let axis = Vector3::new(q_error.i, q_error.j, q_error.k);
    let axis_norm = axis.norm();

    if axis_norm < 1e-10 {
        return Vector3::zeros();
    }

    kp * angle * (axis / axis_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_skew_symmetric() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let s = skew(&v);

        // Skew symmetric: S^T = -S
        assert_relative_eq!(s, -s.transpose(), epsilon = 1e-10);

        // Diagonal should be zero
        assert_relative_eq!(s[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(s[(1, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(s[(2, 2)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cross_product() {
        let a = Vector3::new(1.0, 0.0, 0.0);
        let b = Vector3::new(0.0, 1.0, 0.0);

        let c = cross_matrix(&a, &b);

        // x × y = z
        assert_relative_eq!(c, Vector3::new(0.0, 0.0, 1.0), epsilon = 1e-10);
    }

    #[test]
    fn test_body_z_axis_identity() {
        let q = UnitQuaternion::identity();
        let z = body_z_axis(&q);

        assert_relative_eq!(z, Vector3::new(0.0, 0.0, 1.0), epsilon = 1e-10);
    }

    #[test]
    fn test_body_z_axis_rotated() {
        // Rotate 90 degrees around y-axis
        let q = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)),
            PI / 2.0,
        );
        let z = body_z_axis(&q);

        // z-axis should now point along x
        assert_relative_eq!(z, Vector3::new(1.0, 0.0, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_from_z_axis() {
        let z_desired = Vector3::new(0.0, 0.0, 1.0);
        let yaw = 0.0;

        let q = quaternion_from_z_axis_and_yaw(&z_desired, yaw);
        let z_actual = body_z_axis(&q);

        assert_relative_eq!(z_actual, z_desired.normalize(), epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_from_tilted_z_axis() {
        // Tilted 45 degrees forward (around y-axis)
        let z_desired = Vector3::new(1.0, 0.0, 1.0).normalize();
        let yaw = 0.0;

        let q = quaternion_from_z_axis_and_yaw(&z_desired, yaw);
        let z_actual = body_z_axis(&q);

        assert_relative_eq!(z_actual, z_desired, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_matrix_orthogonal() {
        let q = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(1.0, 1.0, 1.0)),
            1.0,
        );
        let r = rotation_matrix_from_quaternion(&q);

        // R * R^T = I
        let identity = r * r.transpose();
        assert_relative_eq!(identity, Matrix3::identity(), epsilon = 1e-10);

        // det(R) = 1
        assert_relative_eq!(r.determinant(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_attitude_error_zero() {
        let q = UnitQuaternion::identity();
        let error = attitude_error(&q, &q, 1.0);

        assert_relative_eq!(error.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_attitude_error_90deg() {
        let q_current = UnitQuaternion::identity();
        let q_desired = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            PI / 2.0,
        );

        let error = attitude_error(&q_current, &q_desired, 1.0);

        // Error should be ~π/2 around z-axis
        assert_relative_eq!(error.z, PI / 2.0, epsilon = 1e-10);
        assert_relative_eq!(error.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error.y, 0.0, epsilon = 1e-10);
    }
}
