//! Integration with caslo-core dynamics
//!
//! Provides conversion utilities between the planner state representation
//! and the caslo-core simulation types.

use nalgebra::{Vector3, UnitQuaternion};
use caslo_core::dynamics::{LoadState, CableState as CoreCableState, MultiCableState};
use caslo_core::estimation::InitializationResult;

use crate::ocp::{OcpState, CableState, ReferencePoint};
use crate::trajectory::TrackingReference;

/// Convert caslo-core LoadState and MultiCableState to planner OcpState
///
/// Note: caslo-core uses 2nd-order cable model, but planner uses 3rd-order.
/// The angular_jerk field is set to zero during conversion.
pub fn core_to_ocp(load: &LoadState, cables: &MultiCableState) -> OcpState {
    OcpState {
        load_position: load.position,
        load_velocity: load.velocity,
        load_orientation: load.orientation,
        load_angular_velocity: load.angular_velocity,
        cables: cables
            .cables
            .iter()
            .map(|cable| CableState {
                direction: cable.direction,
                angular_velocity: cable.angular_velocity,
                angular_acceleration: cable.angular_acceleration,
                angular_jerk: Vector3::zeros(),  // caslo-core uses 2nd-order model
                tension: cable.tension,
                tension_rate: cable.tension_rate,
            })
            .collect(),
    }
}

/// Convert planner OcpState to caslo-core LoadState and MultiCableState
///
/// Note: Quadrotor states must be computed separately from kinematics.
pub fn ocp_to_core(ocp: &OcpState) -> (LoadState, MultiCableState) {
    let load = LoadState {
        position: ocp.load_position,
        velocity: ocp.load_velocity,
        orientation: ocp.load_orientation,
        angular_velocity: ocp.load_angular_velocity,
    };

    let cables: Vec<CoreCableState> = ocp
        .cables
        .iter()
        .map(|cable| CoreCableState {
            direction: cable.direction,
            angular_velocity: cable.angular_velocity,
            angular_acceleration: cable.angular_acceleration,
            tension: cable.tension,
            tension_rate: cable.tension_rate,
        })
        .collect();

    (load, MultiCableState::new(cables))
}

/// Convert initialization result to OcpState
///
/// Creates an initial OcpState from pose estimation result.
/// Velocities are set to zero (stationary initial condition).
/// Note: caslo-core uses 2nd-order model, angular_jerk is set to zero.
pub fn init_result_to_ocp(
    init: &InitializationResult,
    cable_states: &[CoreCableState],
) -> OcpState {
    OcpState {
        load_position: init.position,
        load_velocity: Vector3::zeros(),
        load_orientation: init.orientation,
        load_angular_velocity: Vector3::zeros(),
        cables: cable_states
            .iter()
            .map(|cable| CableState {
                direction: cable.direction,
                angular_velocity: cable.angular_velocity,
                angular_acceleration: cable.angular_acceleration,
                angular_jerk: Vector3::zeros(),  // caslo-core uses 2nd-order model
                tension: cable.tension,
                tension_rate: cable.tension_rate,
            })
            .collect(),
    }
}

/// Create a reference point from desired load state
pub fn load_state_to_reference(load: &LoadState) -> ReferencePoint {
    ReferencePoint {
        position: load.position,
        velocity: load.velocity,
        orientation: load.orientation,
        angular_velocity: load.angular_velocity,
    }
}

/// Generate a hover reference trajectory
pub fn generate_hover_reference(
    position: Vector3<f64>,
    orientation: UnitQuaternion<f64>,
    num_points: usize,
) -> Vec<ReferencePoint> {
    vec![
        ReferencePoint {
            position,
            velocity: Vector3::zeros(),
            orientation,
            angular_velocity: Vector3::zeros(),
        };
        num_points
    ]
}

/// Generate a linear position trajectory
///
/// Creates a trajectory that moves from start to end position
/// with smooth velocity profile (trapezoidal).
pub fn generate_linear_trajectory(
    start_pos: Vector3<f64>,
    end_pos: Vector3<f64>,
    orientation: UnitQuaternion<f64>,
    duration: f64,
    num_points: usize,
) -> Vec<ReferencePoint> {
    let mut reference = Vec::with_capacity(num_points);
    let direction = end_pos - start_pos;
    let distance = direction.norm();

    if distance < 1e-6 {
        return generate_hover_reference(start_pos, orientation, num_points);
    }

    let max_vel = 2.0 * distance / duration; // Trapezoidal profile peak velocity

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1).max(1) as f64;
        let (pos, vel) = trapezoidal_profile(start_pos, end_pos, t, max_vel);

        reference.push(ReferencePoint {
            position: pos,
            velocity: vel,
            orientation,
            angular_velocity: Vector3::zeros(),
        });
    }

    reference
}

/// Trapezoidal velocity profile for smooth motion
fn trapezoidal_profile(
    start: Vector3<f64>,
    end: Vector3<f64>,
    t_normalized: f64, // 0 to 1
    max_vel: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let direction = end - start;
    let distance = direction.norm();

    if distance < 1e-6 {
        return (start, Vector3::zeros());
    }

    let unit_dir = direction / distance;

    // Acceleration phase: 0 to 0.25
    // Constant velocity: 0.25 to 0.75
    // Deceleration phase: 0.75 to 1.0
    let t = t_normalized.clamp(0.0, 1.0);

    let (s, v) = if t < 0.25 {
        // Acceleration phase
        let tau = t / 0.25;
        let s = 0.125 * tau * tau;
        let v = 0.5 * tau;
        (s, v)
    } else if t < 0.75 {
        // Constant velocity phase
        let tau = (t - 0.25) / 0.5;
        let s = 0.125 + 0.5 * tau;
        (s, 0.5)
    } else {
        // Deceleration phase
        let tau = (t - 0.75) / 0.25;
        let s = 0.625 + 0.5 * tau - 0.125 * tau * tau;
        let v = 0.5 * (1.0 - tau);
        (s.min(1.0), v)
    };

    let position = start + s * distance * unit_dir;
    let velocity = v * max_vel * unit_dir;

    (position, velocity)
}

/// Convert tracking reference to control inputs for each quadrotor
///
/// This is used by the onboard controller to extract per-quadrotor references.
pub fn reference_for_quadrotor(
    reference: &TrackingReference,
    quad_idx: usize,
    attachment_point: Vector3<f64>,
    cable_length: f64,
) -> Option<QuadrotorTrackingRef> {
    if quad_idx >= reference.cable_directions.len() {
        return None;
    }

    // Compute desired quadrotor position from kinematics
    // p_i = p_L + R_L * rho_i - l_i * s_i
    let rotation = reference.load_orientation.to_rotation_matrix();
    let quad_position = reference.load_position
        + rotation * attachment_point
        - cable_length * reference.cable_directions[quad_idx];

    // Compute desired quadrotor velocity (time derivative of position)
    // v_i = v_L + omega_L × (R_L * rho_i) - l_i * (r_i × s_i)
    let rho_world = rotation * attachment_point;
    let omega_cross_rho = reference.load_angular_velocity.cross(&rho_world);
    let s_dot = reference.cable_angular_velocities[quad_idx]
        .cross(&reference.cable_directions[quad_idx]);
    let quad_velocity = reference.load_velocity + omega_cross_rho - cable_length * s_dot;

    Some(QuadrotorTrackingRef {
        position: quad_position,
        velocity: quad_velocity,
        cable_direction: reference.cable_directions[quad_idx],
        cable_angular_velocity: reference.cable_angular_velocities[quad_idx],
        cable_tension: reference.cable_tensions[quad_idx],
    })
}

/// Per-quadrotor tracking reference
#[derive(Debug, Clone)]
pub struct QuadrotorTrackingRef {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub cable_direction: Vector3<f64>,
    pub cable_angular_velocity: Vector3<f64>,
    pub cable_tension: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hover_reference() {
        let pos = Vector3::new(1.0, 2.0, 3.0);
        let ori = UnitQuaternion::identity();
        let refs = generate_hover_reference(pos, ori, 20);

        assert_eq!(refs.len(), 20);
        for r in &refs {
            assert_relative_eq!(r.position, pos, epsilon = 1e-10);
            assert_relative_eq!(r.velocity.norm(), 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_linear_trajectory() {
        let start = Vector3::new(0.0, 0.0, 1.0);
        let end = Vector3::new(2.0, 0.0, 1.0);
        let ori = UnitQuaternion::identity();
        let refs = generate_linear_trajectory(start, end, ori, 2.0, 21);

        assert_eq!(refs.len(), 21);

        // First point at start
        assert_relative_eq!(refs[0].position, start, epsilon = 0.01);

        // Last point at end
        assert_relative_eq!(refs[20].position, end, epsilon = 0.01);

        // Middle should have maximum velocity
        let mid_vel = refs[10].velocity.norm();
        assert!(mid_vel > 0.0);
    }

    #[test]
    fn test_trapezoidal_profile_bounds() {
        let start = Vector3::zeros();
        let end = Vector3::new(1.0, 0.0, 0.0);

        let (pos_0, _) = trapezoidal_profile(start, end, 0.0, 1.0);
        let (pos_1, _) = trapezoidal_profile(start, end, 1.0, 1.0);

        assert_relative_eq!(pos_0, start, epsilon = 1e-10);
        assert_relative_eq!(pos_1, end, epsilon = 1e-10);
    }

    #[test]
    fn test_state_conversion_roundtrip() {
        let load = LoadState {
            position: Vector3::new(1.0, 2.0, 3.0),
            velocity: Vector3::new(0.1, 0.2, 0.3),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        };

        let cables = MultiCableState::new(vec![
            CoreCableState::pointing_down(10.0),
            CoreCableState::pointing_down(5.0),
            CoreCableState::pointing_down(7.0),
        ]);

        let ocp = core_to_ocp(&load, &cables);
        let (load_back, cables_back) = ocp_to_core(&ocp);

        assert_relative_eq!(load_back.position, load.position, epsilon = 1e-10);
        assert_relative_eq!(load_back.velocity, load.velocity, epsilon = 1e-10);
        assert_relative_eq!(cables_back.cables[0].tension, 10.0, epsilon = 1e-10);
    }
}
