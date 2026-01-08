//! Planned trajectory representation and interpolation
//!
//! Handles the output trajectory from the OCP solver, providing
//! interpolation for the high-frequency tracking controller.

use nalgebra::{Vector3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::ocp::{OcpState, OcpControl, CableState, CableControl};

/// A complete planned trajectory from the OCP solver
///
/// Contains discrete state and control sequences along with timing
/// for interpolation.
#[derive(Debug, Clone)]
pub struct PlannedTrajectory {
    /// Time stamps for each node [s]
    pub times: Vec<f64>,
    /// State sequence
    pub states: Vec<OcpState>,
    /// Control sequence (one less than states for N+1 formulation)
    pub controls: Vec<OcpControl>,
    /// Generation timestamp (for staleness detection)
    pub generated_at: f64,
    /// Whether the trajectory is valid/feasible
    pub is_valid: bool,
}

impl PlannedTrajectory {
    /// Create an empty trajectory
    pub fn empty(num_cables: usize) -> Self {
        Self {
            times: Vec::new(),
            states: Vec::new(),
            controls: Vec::new(),
            generated_at: 0.0,
            is_valid: false,
        }
    }

    /// Create a hover trajectory at a fixed position
    pub fn hover(
        position: Vector3<f64>,
        orientation: UnitQuaternion<f64>,
        num_cables: usize,
        horizon_time: f64,
        num_nodes: usize,
    ) -> Self {
        let dt = horizon_time / num_nodes as f64;
        let times: Vec<f64> = (0..=num_nodes).map(|i| i as f64 * dt).collect();

        let mut state = OcpState::new(num_cables);
        state.load_position = position;
        state.load_orientation = orientation;

        let states = vec![state; num_nodes + 1];
        let controls = vec![OcpControl::new(num_cables); num_nodes];

        Self {
            times,
            states,
            controls,
            generated_at: 0.0,
            is_valid: true,
        }
    }

    /// Interpolate state at a given time
    ///
    /// Uses linear interpolation for positions/velocities and
    /// spherical interpolation for quaternions.
    pub fn interpolate_state(&self, t: f64) -> Option<InterpolatedState> {
        if self.times.is_empty() || !self.is_valid {
            return None;
        }

        // Clamp to trajectory bounds
        let t_start = self.times[0];
        let t_end = *self.times.last().unwrap();

        if t <= t_start {
            return Some(self.state_to_interpolated(0));
        }
        if t >= t_end {
            return Some(self.state_to_interpolated(self.states.len() - 1));
        }

        // Find bracketing indices
        let idx = self.times.iter()
            .position(|&time| time > t)
            .unwrap_or(self.times.len() - 1)
            .saturating_sub(1);

        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let alpha = (t - t0) / (t1 - t0);

        Some(self.interpolate_between(idx, idx + 1, alpha))
    }

    /// Interpolate control at a given time
    pub fn interpolate_control(&self, t: f64) -> Option<OcpControl> {
        if self.controls.is_empty() || !self.is_valid {
            return None;
        }

        let t_start = self.times[0];
        let t_end = self.times[self.controls.len()]; // Control ends one step before

        if t <= t_start {
            return Some(self.controls[0].clone());
        }
        if t >= t_end {
            return Some(self.controls.last().unwrap().clone());
        }

        // Find bracketing index
        let idx = self.times.iter()
            .position(|&time| time > t)
            .unwrap_or(self.controls.len())
            .saturating_sub(1)
            .min(self.controls.len() - 1);

        Some(self.controls[idx].clone())
    }

    /// Get the reference for the tracking controller at time t
    pub fn get_tracking_reference(&self, t: f64) -> Option<TrackingReference> {
        let state = self.interpolate_state(t)?;
        let control = self.interpolate_control(t)?;

        Some(TrackingReference {
            load_position: state.load_position,
            load_velocity: state.load_velocity,
            load_acceleration: state.load_acceleration,
            load_orientation: state.load_orientation,
            load_angular_velocity: state.load_angular_velocity,
            cable_directions: state.cable_directions,
            cable_angular_velocities: state.cable_angular_velocities,
            cable_tensions: state.cable_tensions,
            cable_angular_jerks: control.cables.iter()
                .map(|c| c.angular_jerk)
                .collect(),
            cable_tension_accelerations: control.cables.iter()
                .map(|c| c.tension_acceleration)
                .collect(),
        })
    }

    fn state_to_interpolated(&self, idx: usize) -> InterpolatedState {
        let state = &self.states[idx];

        // Compute acceleration from control if available
        let acceleration = if idx < self.controls.len() {
            // Would need dynamics model here; for now use zero
            Vector3::zeros()
        } else {
            Vector3::zeros()
        };

        InterpolatedState {
            load_position: state.load_position,
            load_velocity: state.load_velocity,
            load_acceleration: acceleration,
            load_orientation: state.load_orientation,
            load_angular_velocity: state.load_angular_velocity,
            cable_directions: state.cables.iter().map(|c| c.direction).collect(),
            cable_angular_velocities: state.cables.iter().map(|c| c.angular_velocity).collect(),
            cable_tensions: state.cables.iter().map(|c| c.tension).collect(),
        }
    }

    fn interpolate_between(&self, idx0: usize, idx1: usize, alpha: f64) -> InterpolatedState {
        let s0 = &self.states[idx0];
        let s1 = &self.states[idx1];

        // Linear interpolation for vectors
        let load_position = s0.load_position.lerp(&s1.load_position, alpha);
        let load_velocity = s0.load_velocity.lerp(&s1.load_velocity, alpha);
        let load_angular_velocity = s0.load_angular_velocity.lerp(&s1.load_angular_velocity, alpha);

        // SLERP for quaternion
        let load_orientation = s0.load_orientation.slerp(&s1.load_orientation, alpha);

        // Interpolate cable states
        let num_cables = s0.cables.len();
        let mut cable_directions = Vec::with_capacity(num_cables);
        let mut cable_angular_velocities = Vec::with_capacity(num_cables);
        let mut cable_tensions = Vec::with_capacity(num_cables);

        for i in 0..num_cables {
            let dir = s0.cables[i].direction.lerp(&s1.cables[i].direction, alpha);
            // Renormalize direction
            let dir_normalized = if dir.norm() > 1e-10 {
                dir.normalize()
            } else {
                s0.cables[i].direction
            };
            cable_directions.push(dir_normalized);

            cable_angular_velocities.push(
                s0.cables[i].angular_velocity.lerp(&s1.cables[i].angular_velocity, alpha)
            );
            cable_tensions.push(
                s0.cables[i].tension * (1.0 - alpha) + s1.cables[i].tension * alpha
            );
        }

        // Approximate acceleration from velocity difference
        let dt = self.times[idx1] - self.times[idx0];
        let load_acceleration = if dt > 1e-10 {
            (s1.load_velocity - s0.load_velocity) / dt
        } else {
            Vector3::zeros()
        };

        InterpolatedState {
            load_position,
            load_velocity,
            load_acceleration,
            load_orientation,
            load_angular_velocity,
            cable_directions,
            cable_angular_velocities,
            cable_tensions,
        }
    }

    /// Check if trajectory is stale (too old to use)
    pub fn is_stale(&self, current_time: f64, max_age: f64) -> bool {
        current_time - self.generated_at > max_age
    }

    /// Get remaining trajectory duration from current time
    pub fn remaining_duration(&self, current_time: f64) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        let t_end = *self.times.last().unwrap();
        (t_end - current_time).max(0.0)
    }
}

/// Interpolated state at a continuous time
#[derive(Debug, Clone)]
pub struct InterpolatedState {
    pub load_position: Vector3<f64>,
    pub load_velocity: Vector3<f64>,
    pub load_acceleration: Vector3<f64>,
    pub load_orientation: UnitQuaternion<f64>,
    pub load_angular_velocity: Vector3<f64>,
    pub cable_directions: Vec<Vector3<f64>>,
    pub cable_angular_velocities: Vec<Vector3<f64>>,
    pub cable_tensions: Vec<f64>,
}

/// Complete reference for the tracking controller
///
/// Provides all quantities needed by the onboard INDI controller.
#[derive(Debug, Clone)]
pub struct TrackingReference {
    // Load references
    pub load_position: Vector3<f64>,
    pub load_velocity: Vector3<f64>,
    pub load_acceleration: Vector3<f64>,
    pub load_orientation: UnitQuaternion<f64>,
    pub load_angular_velocity: Vector3<f64>,

    // Cable references
    pub cable_directions: Vec<Vector3<f64>>,
    pub cable_angular_velocities: Vec<Vector3<f64>>,
    pub cable_tensions: Vec<f64>,

    // Feedforward terms for INDI
    pub cable_angular_jerks: Vec<Vector3<f64>>,
    pub cable_tension_accelerations: Vec<f64>,
}

impl TrackingReference {
    /// Get the reference for a specific quadrotor
    pub fn for_quadrotor(&self, idx: usize) -> Option<QuadrotorReference> {
        if idx >= self.cable_directions.len() {
            return None;
        }

        Some(QuadrotorReference {
            cable_direction: self.cable_directions[idx],
            cable_angular_velocity: self.cable_angular_velocities[idx],
            cable_tension: self.cable_tensions[idx],
            cable_angular_jerk: self.cable_angular_jerks[idx],
            cable_tension_acceleration: self.cable_tension_accelerations[idx],
        })
    }
}

/// Reference for a single quadrotor's controller
#[derive(Debug, Clone, Copy)]
pub struct QuadrotorReference {
    pub cable_direction: Vector3<f64>,
    pub cable_angular_velocity: Vector3<f64>,
    pub cable_tension: f64,
    pub cable_angular_jerk: Vector3<f64>,
    pub cable_tension_acceleration: f64,
}

/// Trajectory segment for efficient real-time lookup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectorySegment {
    /// Start time of segment
    pub t_start: f64,
    /// End time of segment
    pub t_end: f64,
    /// Polynomial coefficients for position (degree 5)
    pub position_coeffs: [[f64; 6]; 3],
    /// Polynomial coefficients for orientation (as axis-angle)
    pub orientation_coeffs: [[f64; 4]; 3],
}

impl TrajectorySegment {
    /// Evaluate position at time t
    pub fn position(&self, t: f64) -> Vector3<f64> {
        let tau = (t - self.t_start) / (self.t_end - self.t_start);
        Vector3::new(
            Self::eval_poly(&self.position_coeffs[0], tau),
            Self::eval_poly(&self.position_coeffs[1], tau),
            Self::eval_poly(&self.position_coeffs[2], tau),
        )
    }

    /// Evaluate velocity at time t
    pub fn velocity(&self, t: f64) -> Vector3<f64> {
        let tau = (t - self.t_start) / (self.t_end - self.t_start);
        let dt = self.t_end - self.t_start;
        Vector3::new(
            Self::eval_poly_deriv(&self.position_coeffs[0], tau) / dt,
            Self::eval_poly_deriv(&self.position_coeffs[1], tau) / dt,
            Self::eval_poly_deriv(&self.position_coeffs[2], tau) / dt,
        )
    }

    /// Evaluate acceleration at time t
    pub fn acceleration(&self, t: f64) -> Vector3<f64> {
        let tau = (t - self.t_start) / (self.t_end - self.t_start);
        let dt = self.t_end - self.t_start;
        let dt2 = dt * dt;
        Vector3::new(
            Self::eval_poly_deriv2(&self.position_coeffs[0], tau) / dt2,
            Self::eval_poly_deriv2(&self.position_coeffs[1], tau) / dt2,
            Self::eval_poly_deriv2(&self.position_coeffs[2], tau) / dt2,
        )
    }

    fn eval_poly(coeffs: &[f64; 6], tau: f64) -> f64 {
        coeffs[0]
            + coeffs[1] * tau
            + coeffs[2] * tau.powi(2)
            + coeffs[3] * tau.powi(3)
            + coeffs[4] * tau.powi(4)
            + coeffs[5] * tau.powi(5)
    }

    fn eval_poly_deriv(coeffs: &[f64; 6], tau: f64) -> f64 {
        coeffs[1]
            + 2.0 * coeffs[2] * tau
            + 3.0 * coeffs[3] * tau.powi(2)
            + 4.0 * coeffs[4] * tau.powi(3)
            + 5.0 * coeffs[5] * tau.powi(4)
    }

    fn eval_poly_deriv2(coeffs: &[f64; 6], tau: f64) -> f64 {
        2.0 * coeffs[2]
            + 6.0 * coeffs[3] * tau
            + 12.0 * coeffs[4] * tau.powi(2)
            + 20.0 * coeffs[5] * tau.powi(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hover_trajectory() {
        let pos = Vector3::new(1.0, 2.0, 3.0);
        let ori = UnitQuaternion::identity();
        let traj = PlannedTrajectory::hover(pos, ori, 3, 2.0, 20);

        assert_eq!(traj.times.len(), 21);
        assert_eq!(traj.states.len(), 21);
        assert_eq!(traj.controls.len(), 20);
        assert!(traj.is_valid);

        // All states should be at hover position
        for state in &traj.states {
            assert!((state.load_position - pos).norm() < 1e-10);
        }
    }

    #[test]
    fn test_interpolation() {
        let pos = Vector3::new(0.0, 0.0, 1.0);
        let ori = UnitQuaternion::identity();
        let mut traj = PlannedTrajectory::hover(pos, ori, 3, 2.0, 20);

        // Modify first and last state for interpolation test
        traj.states[0].load_position = Vector3::new(0.0, 0.0, 0.0);
        traj.states[20].load_position = Vector3::new(0.0, 0.0, 2.0);

        // Interpolate at midpoint
        let state = traj.interpolate_state(1.0).unwrap();
        // Should be somewhere between 0 and 2
        assert!(state.load_position.z > 0.0);
        assert!(state.load_position.z < 2.0);
    }

    #[test]
    fn test_tracking_reference() {
        let pos = Vector3::new(1.0, 2.0, 3.0);
        let ori = UnitQuaternion::identity();
        let traj = PlannedTrajectory::hover(pos, ori, 3, 2.0, 20);

        let reference = traj.get_tracking_reference(0.5).unwrap();
        assert!((reference.load_position - pos).norm() < 1e-10);
        assert_eq!(reference.cable_directions.len(), 3);

        let quad_ref = reference.for_quadrotor(0).unwrap();
        assert!(quad_ref.cable_tension > 0.0);
    }

    #[test]
    fn test_trajectory_staleness() {
        let traj = PlannedTrajectory::hover(
            Vector3::zeros(),
            UnitQuaternion::identity(),
            3,
            2.0,
            20,
        );

        assert!(!traj.is_stale(0.5, 1.0));
        assert!(traj.is_stale(1.5, 1.0));
    }
}
