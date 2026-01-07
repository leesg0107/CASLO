//! Unit sphere S² operations
//!
//! Operations for unit vectors (directions) used for cable directions in the paper.
//! Cable direction sᵢ ∈ S² satisfies ‖sᵢ‖ = 1.

use nalgebra::Vector3;

/// Normalize a vector to unit length
///
/// Returns None if the vector is too small to normalize safely.
pub fn normalize(v: &Vector3<f64>) -> Option<Vector3<f64>> {
    let norm = v.norm();
    if norm < 1e-10 {
        None
    } else {
        Some(v / norm)
    }
}

/// Project a vector onto the unit sphere
///
/// If the vector is zero, returns the default direction [0, 0, -1] (downward)
pub fn project_to_sphere(v: &Vector3<f64>) -> Vector3<f64> {
    normalize(v).unwrap_or(Vector3::new(0.0, 0.0, -1.0))
}

/// Compute the tangent space basis at a point on S²
///
/// Returns two orthonormal vectors (e1, e2) perpendicular to s
pub fn tangent_basis(s: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    // Find a vector not parallel to s
    let not_parallel = if s.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };

    // e1 = s × not_parallel (normalized)
    let e1 = s.cross(&not_parallel).normalize();

    // e2 = s × e1
    let e2 = s.cross(&e1);

    (e1, e2)
}

/// Cable direction derivative from angular velocity
///
/// From Eq. 3: ṡᵢ = rᵢ × sᵢ
///
/// # Arguments
/// * `s` - Cable direction (unit vector)
/// * `r` - Cable angular velocity
///
/// # Returns
/// Time derivative of cable direction
pub fn direction_derivative(s: &Vector3<f64>, r: &Vector3<f64>) -> Vector3<f64> {
    r.cross(s)
}

/// Integrate cable direction while maintaining unit constraint
///
/// s_new = normalize(s + ṡ * dt)
///
/// # Arguments
/// * `s` - Current direction (unit vector)
/// * `s_dot` - Direction derivative
/// * `dt` - Time step [s]
///
/// # Returns
/// New direction (unit vector)
pub fn integrate_direction(s: &Vector3<f64>, s_dot: &Vector3<f64>, dt: f64) -> Vector3<f64> {
    let s_new = s + s_dot * dt;
    project_to_sphere(&s_new)
}

/// Spherical linear interpolation (slerp) between two directions
///
/// # Arguments
/// * `s1` - Start direction (unit vector)
/// * `s2` - End direction (unit vector)
/// * `t` - Interpolation parameter [0, 1]
///
/// # Returns
/// Interpolated direction (unit vector)
pub fn slerp(s1: &Vector3<f64>, s2: &Vector3<f64>, t: f64) -> Vector3<f64> {
    let dot = s1.dot(s2).clamp(-1.0, 1.0);
    let theta = dot.acos();

    if theta.abs() < 1e-10 {
        return *s1;
    }

    let sin_theta = theta.sin();
    let w1 = ((1.0 - t) * theta).sin() / sin_theta;
    let w2 = (t * theta).sin() / sin_theta;

    project_to_sphere(&(w1 * s1 + w2 * s2))
}

/// Angle between two directions
///
/// Returns the angle in radians [0, π]
pub fn angle_between(s1: &Vector3<f64>, s2: &Vector3<f64>) -> f64 {
    let dot = s1.dot(s2).clamp(-1.0, 1.0);
    dot.acos()
}

/// Check if a vector is a valid unit vector
pub fn is_unit_vector(v: &Vector3<f64>, tolerance: f64) -> bool {
    (v.norm() - 1.0).abs() < tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_normalize() {
        let v = Vector3::new(3.0, 4.0, 0.0);
        let n = normalize(&v).unwrap();

        assert_relative_eq!(n.norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(n, Vector3::new(0.6, 0.8, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = Vector3::zeros();
        assert!(normalize(&v).is_none());
    }

    #[test]
    fn test_project_to_sphere() {
        let v = Vector3::new(2.0, 0.0, 0.0);
        let s = project_to_sphere(&v);

        assert_relative_eq!(s, Vector3::new(1.0, 0.0, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_tangent_basis_orthonormal() {
        let s = Vector3::new(0.0, 0.0, 1.0);
        let (e1, e2) = tangent_basis(&s);

        // e1 and e2 should be orthonormal
        assert_relative_eq!(e1.norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(e2.norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(e1.dot(&e2), 0.0, epsilon = 1e-10);

        // Both should be perpendicular to s
        assert_relative_eq!(e1.dot(&s), 0.0, epsilon = 1e-10);
        assert_relative_eq!(e2.dot(&s), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_direction_derivative() {
        let s = Vector3::new(0.0, 0.0, -1.0); // Pointing down
        let r = Vector3::new(1.0, 0.0, 0.0);  // Rotating around x

        let s_dot = direction_derivative(&s, &r);

        // r × s = [1,0,0] × [0,0,-1] = [0,1,0]
        assert_relative_eq!(s_dot, Vector3::new(0.0, 1.0, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_direction_maintains_unit() {
        let s = Vector3::new(0.0, 0.0, -1.0);
        let s_dot = Vector3::new(0.1, 0.0, 0.0);
        let dt = 0.01;

        let s_new = integrate_direction(&s, &s_dot, dt);

        assert_relative_eq!(s_new.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_slerp_endpoints() {
        let s1 = Vector3::new(1.0, 0.0, 0.0);
        let s2 = Vector3::new(0.0, 1.0, 0.0);

        let at_0 = slerp(&s1, &s2, 0.0);
        let at_1 = slerp(&s1, &s2, 1.0);

        assert_relative_eq!(at_0, s1, epsilon = 1e-10);
        assert_relative_eq!(at_1, s2, epsilon = 1e-10);
    }

    #[test]
    fn test_slerp_midpoint() {
        let s1 = Vector3::new(1.0, 0.0, 0.0);
        let s2 = Vector3::new(0.0, 1.0, 0.0);

        let mid = slerp(&s1, &s2, 0.5);

        // Midpoint should be at 45 degrees
        let expected = Vector3::new(1.0, 1.0, 0.0).normalize();
        assert_relative_eq!(mid, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_angle_between() {
        let s1 = Vector3::new(1.0, 0.0, 0.0);
        let s2 = Vector3::new(0.0, 1.0, 0.0);

        let angle = angle_between(&s1, &s2);

        assert_relative_eq!(angle, PI / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_angle_between_same() {
        let s = Vector3::new(1.0, 0.0, 0.0);

        let angle = angle_between(&s, &s);

        assert_relative_eq!(angle, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_angle_between_opposite() {
        let s1 = Vector3::new(1.0, 0.0, 0.0);
        let s2 = Vector3::new(-1.0, 0.0, 0.0);

        let angle = angle_between(&s1, &s2);

        assert_relative_eq!(angle, PI, epsilon = 1e-10);
    }
}
