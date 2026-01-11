//! Planned trajectory representation and interpolation
//!
//! Handles the output trajectory from the OCP solver, providing
//! interpolation for the high-frequency tracking controller.
//!
//! Key capability: Converts load-cable trajectories to per-quadrotor
//! trajectories using kinematic constraint derivatives (Eq. 5, S1).

use nalgebra::{Vector3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use caslo_core::kinematics::{
    KinematicConstraint, LoadKinematicState, CableKinematicState, QuadrotorTrajectoryPoint,
};
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
    ///
    /// Returns a TrackingReference containing load-cable state and derivatives,
    /// which can be used with KinematicConstraint to compute per-quadrotor trajectories.
    ///
    /// This method now properly extracts all state variables from Eq. 1,
    /// including cable angular accelerations ṙᵢ and tension rates ṫᵢ.
    pub fn get_tracking_reference(&self, t: f64) -> Option<TrackingReference> {
        let state = self.interpolate_state(t)?;
        let control = self.interpolate_control(t)?;

        // Cable states and control inputs from OCP (Eq. 1 and Eq. 3 with 3rd-order model)
        Some(TrackingReference {
            load_position: state.load_position,
            load_velocity: state.load_velocity,
            load_acceleration: state.load_acceleration,
            load_jerk: Vector3::zeros(), // Would need higher-order derivatives
            load_orientation: state.load_orientation,
            load_angular_velocity: state.load_angular_velocity,
            load_angular_acceleration: Vector3::zeros(), // Would need dynamics
            load_angular_jerk: Vector3::zeros(),
            cable_directions: state.cable_directions,
            cable_angular_velocities: state.cable_angular_velocities,
            cable_angular_accelerations: state.cable_angular_accelerations, // ṙᵢ from state
            cable_angular_jerks: state.cable_angular_jerks, // r̈ᵢ from state (3rd-order model)
            cable_tensions: state.cable_tensions,
            cable_tension_rates: state.cable_tension_rates, // ṫᵢ from state
            cable_angular_snaps: control.cables.iter()
                .map(|c| c.angular_snap)  // γᵢ = r⃛ᵢ (angular snap - control input)
                .collect(),
            cable_tension_accelerations: control.cables.iter()
                .map(|c| c.tension_acceleration)
                .collect(),
            quadrotor_trajectories: None,
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
            cable_angular_accelerations: state.cables.iter().map(|c| c.angular_acceleration).collect(),
            cable_angular_jerks: state.cables.iter().map(|c| c.angular_jerk).collect(),
            cable_tensions: state.cables.iter().map(|c| c.tension).collect(),
            cable_tension_rates: state.cables.iter().map(|c| c.tension_rate).collect(),
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

        // Interpolate cable states (including higher-order derivatives from Eq. 1 with 3rd-order model)
        let num_cables = s0.cables.len();
        let mut cable_directions = Vec::with_capacity(num_cables);
        let mut cable_angular_velocities = Vec::with_capacity(num_cables);
        let mut cable_angular_accelerations = Vec::with_capacity(num_cables);
        let mut cable_angular_jerks = Vec::with_capacity(num_cables);
        let mut cable_tensions = Vec::with_capacity(num_cables);
        let mut cable_tension_rates = Vec::with_capacity(num_cables);

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
            cable_angular_accelerations.push(
                s0.cables[i].angular_acceleration.lerp(&s1.cables[i].angular_acceleration, alpha)
            );
            cable_angular_jerks.push(
                s0.cables[i].angular_jerk.lerp(&s1.cables[i].angular_jerk, alpha)
            );
            cable_tensions.push(
                s0.cables[i].tension * (1.0 - alpha) + s1.cables[i].tension * alpha
            );
            cable_tension_rates.push(
                s0.cables[i].tension_rate * (1.0 - alpha) + s1.cables[i].tension_rate * alpha
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
            cable_angular_accelerations,
            cable_angular_jerks,
            cable_tensions,
            cable_tension_rates,
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

    /// Resample the trajectory to get xinit for OCP at given time
    ///
    /// This implements the trajectory resampling from Fig. 8 of the paper.
    /// The resampled state includes all derivatives (ṙᵢ, ṫᵢ) needed for
    /// smooth trajectory continuation.
    ///
    /// # Arguments
    /// * `t` - Current time at which to sample the trajectory
    ///
    /// # Returns
    /// OcpState suitable for use as xinit in the next OCP solve
    pub fn resample_for_xinit(&self, t: f64) -> Option<OcpState> {
        if self.times.is_empty() || !self.is_valid {
            return None;
        }

        let state = self.interpolate_state(t)?;
        let num_cables = state.cable_directions.len();

        Some(OcpState {
            load_position: state.load_position,
            load_velocity: state.load_velocity,
            load_orientation: state.load_orientation,
            load_angular_velocity: state.load_angular_velocity,
            cables: (0..num_cables)
                .map(|i| CableState {
                    direction: state.cable_directions[i],
                    angular_velocity: state.cable_angular_velocities[i],
                    angular_acceleration: state.cable_angular_accelerations[i],
                    angular_jerk: state.cable_angular_jerks[i],  // r̈ᵢ from 3rd-order model
                    tension: state.cable_tensions[i],
                    tension_rate: state.cable_tension_rates[i],
                })
                .collect(),
        })
    }

    /// Resample trajectory for warm-starting the next OCP solve
    ///
    /// Returns shifted state and control sequences for warm-start initialization.
    /// The trajectories are shifted by the given time offset.
    ///
    /// # Arguments
    /// * `t_current` - Current time
    /// * `horizon_time` - New horizon duration
    /// * `num_nodes` - Number of nodes in the new horizon
    ///
    /// # Returns
    /// Tuple of (state_sequence, control_sequence) for warm-starting
    pub fn resample_for_warmstart(
        &self,
        t_current: f64,
        horizon_time: f64,
        num_nodes: usize,
    ) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        if self.times.is_empty() || !self.is_valid {
            return None;
        }

        let dt = horizon_time / num_nodes as f64;
        let num_cables = self.states[0].cables.len();

        let mut states = Vec::with_capacity(num_nodes + 1);
        let mut controls = Vec::with_capacity(num_nodes);

        for k in 0..=num_nodes {
            let t = t_current + k as f64 * dt;

            // Get state at this time
            if let Some(xinit) = self.resample_for_xinit(t) {
                states.push(xinit.to_vector());
            } else {
                // Beyond trajectory end - use last state
                if let Some(last) = self.states.last() {
                    states.push(last.to_vector());
                } else {
                    return None;
                }
            }

            // Get control at this time (except for terminal node)
            if k < num_nodes {
                if let Some(control) = self.interpolate_control(t) {
                    controls.push(control.to_vector());
                } else {
                    controls.push(OcpControl::new(num_cables).to_vector());
                }
            }
        }

        Some((states, controls))
    }
}

/// Interpolated state at a continuous time (Eq. 1)
///
/// Contains all state variables including higher-order cable derivatives.
#[derive(Debug, Clone)]
pub struct InterpolatedState {
    pub load_position: Vector3<f64>,
    pub load_velocity: Vector3<f64>,
    pub load_acceleration: Vector3<f64>,
    pub load_orientation: UnitQuaternion<f64>,
    pub load_angular_velocity: Vector3<f64>,
    /// Cable directions sᵢ
    pub cable_directions: Vec<Vector3<f64>>,
    /// Cable angular velocities rᵢ
    pub cable_angular_velocities: Vec<Vector3<f64>>,
    /// Cable angular accelerations ṙᵢ (from Eq. 1)
    pub cable_angular_accelerations: Vec<Vector3<f64>>,
    /// Cable angular jerks r̈ᵢ (from Eq. 1 with 3rd-order model)
    pub cable_angular_jerks: Vec<Vector3<f64>>,
    /// Cable tensions tᵢ
    pub cable_tensions: Vec<f64>,
    /// Cable tension rates ṫᵢ (from Eq. 1)
    pub cable_tension_rates: Vec<f64>,
}

/// Complete reference for the tracking controller
///
/// Provides all quantities needed by the onboard INDI controller.
/// Includes per-quadrotor trajectories computed from kinematic constraints.
#[derive(Debug, Clone)]
pub struct TrackingReference {
    // Load references
    pub load_position: Vector3<f64>,
    pub load_velocity: Vector3<f64>,
    pub load_acceleration: Vector3<f64>,
    pub load_jerk: Vector3<f64>,
    pub load_orientation: UnitQuaternion<f64>,
    pub load_angular_velocity: Vector3<f64>,
    pub load_angular_acceleration: Vector3<f64>,
    pub load_angular_jerk: Vector3<f64>,

    // Cable state references (per cable) from Eq. 1 with 3rd-order model
    pub cable_directions: Vec<Vector3<f64>>,
    pub cable_angular_velocities: Vec<Vector3<f64>>,
    pub cable_angular_accelerations: Vec<Vector3<f64>>,  // ṙᵢ (in state)
    pub cable_angular_jerks: Vec<Vector3<f64>>,          // r̈ᵢ (in state for 3rd-order model)
    pub cable_tensions: Vec<f64>,
    pub cable_tension_rates: Vec<f64>,

    // Feedforward control inputs from planner (per cable)
    // From paper Eq. 3 with 3rd-order model: γᵢ = r⃛ᵢ (angular snap), λᵢ = ẗᵢ (tension acceleration)
    pub cable_angular_snaps: Vec<Vector3<f64>>,
    pub cable_tension_accelerations: Vec<f64>,

    // Pre-computed quadrotor trajectories (optional, computed on demand)
    #[doc(hidden)]
    pub quadrotor_trajectories: Option<Vec<QuadrotorTrajectoryPoint>>,
}

impl TrackingReference {
    /// Create a new tracking reference (3rd-order cable model)
    pub fn new(
        load_position: Vector3<f64>,
        load_velocity: Vector3<f64>,
        load_acceleration: Vector3<f64>,
        load_orientation: UnitQuaternion<f64>,
        load_angular_velocity: Vector3<f64>,
        cable_directions: Vec<Vector3<f64>>,
        cable_angular_velocities: Vec<Vector3<f64>>,
        cable_tensions: Vec<f64>,
        cable_angular_snaps: Vec<Vector3<f64>>,
        cable_tension_accelerations: Vec<f64>,
    ) -> Self {
        let n = cable_directions.len();
        Self {
            load_position,
            load_velocity,
            load_acceleration,
            load_jerk: Vector3::zeros(),
            load_orientation,
            load_angular_velocity,
            load_angular_acceleration: Vector3::zeros(),
            load_angular_jerk: Vector3::zeros(),
            cable_directions,
            cable_angular_velocities,
            cable_angular_accelerations: vec![Vector3::zeros(); n],
            cable_angular_jerks: vec![Vector3::zeros(); n],  // r̈ᵢ from state
            cable_tensions,
            cable_tension_rates: vec![0.0; n],
            cable_angular_snaps,  // γᵢ = r⃛ᵢ (control)
            cable_tension_accelerations,
            quadrotor_trajectories: None,
        }
    }

    /// Get the reference for a specific quadrotor using kinematic constraint
    ///
    /// This computes the quadrotor trajectory from the load-cable state
    /// using Eq. (5) and Eq. (S1) from the paper.
    pub fn for_quadrotor(
        &self,
        idx: usize,
        constraint: &KinematicConstraint,
    ) -> Option<QuadrotorReference> {
        if idx >= self.cable_directions.len() {
            return None;
        }

        // Build load kinematic state
        let load_state = LoadKinematicState {
            position: self.load_position,
            velocity: self.load_velocity,
            acceleration: self.load_acceleration,
            jerk: self.load_jerk,
            orientation: self.load_orientation,
            angular_velocity: self.load_angular_velocity,
            angular_acceleration: self.load_angular_acceleration,
            angular_jerk: self.load_angular_jerk,
        };

        // Build cable kinematic state
        let cable_state = CableKinematicState {
            direction: self.cable_directions[idx],
            angular_velocity: self.cable_angular_velocities[idx],
            angular_acceleration: self.cable_angular_accelerations[idx],
            angular_jerk: self.cable_angular_jerks[idx],
        };

        // Compute quadrotor trajectory from kinematic constraint (Eq. S1)
        let quad_traj = constraint.quadrotor_trajectory_point(&load_state, &cable_state, idx);

        Some(QuadrotorReference {
            position: quad_traj.position,
            velocity: quad_traj.velocity,
            acceleration: quad_traj.acceleration,
            jerk: quad_traj.jerk,
            cable_direction: self.cable_directions[idx],
            cable_angular_velocity: self.cable_angular_velocities[idx],
            cable_angular_acceleration: self.cable_angular_accelerations[idx],
            cable_angular_jerk: self.cable_angular_jerks[idx],  // r̈ᵢ from state
            cable_tension: self.cable_tensions[idx],
            cable_tension_rate: self.cable_tension_rates[idx],
            cable_angular_snap: self.cable_angular_snaps[idx],  // γᵢ = r⃛ᵢ (control)
            cable_tension_acceleration: self.cable_tension_accelerations[idx],
        })
    }

    /// Get all quadrotor references at once
    pub fn all_quadrotor_references(
        &self,
        constraint: &KinematicConstraint,
    ) -> Vec<QuadrotorReference> {
        (0..self.cable_directions.len())
            .filter_map(|i| self.for_quadrotor(i, constraint))
            .collect()
    }

    /// Get number of quadrotors/cables
    pub fn num_quadrotors(&self) -> usize {
        self.cable_directions.len()
    }
}

/// Reference for a single quadrotor's controller
///
/// Contains the full trajectory reference computed from kinematic constraint
/// derivatives (Eq. 5, S1), enabling trajectory-based control.
#[derive(Debug, Clone)]
pub struct QuadrotorReference {
    // Quadrotor trajectory from kinematic constraint (Eq. S1)
    /// Quadrotor position [m] (world frame)
    pub position: Vector3<f64>,
    /// Quadrotor velocity [m/s] (world frame)
    pub velocity: Vector3<f64>,
    /// Quadrotor acceleration [m/s²] (world frame)
    pub acceleration: Vector3<f64>,
    /// Quadrotor jerk [m/s³] (world frame) - for feedforward
    pub jerk: Vector3<f64>,

    // Cable state references (3rd-order model)
    /// Cable direction (unit vector, world frame)
    pub cable_direction: Vector3<f64>,
    /// Cable angular velocity [rad/s] rᵢ
    pub cable_angular_velocity: Vector3<f64>,
    /// Cable angular acceleration [rad/s²] ṙᵢ (in state)
    pub cable_angular_acceleration: Vector3<f64>,
    /// Cable angular jerk [rad/s³] r̈ᵢ (in state for 3rd-order model)
    pub cable_angular_jerk: Vector3<f64>,
    /// Cable tension [N]
    pub cable_tension: f64,
    /// Cable tension rate [N/s]
    pub cable_tension_rate: f64,

    // Feedforward control inputs from planner (Eq. 3 with 3rd-order model)
    /// Cable angular snap [rad/s⁴] γᵢ = r⃛ᵢ (control input)
    pub cable_angular_snap: Vector3<f64>,
    /// Tension acceleration [N/s²] λᵢ = ẗᵢ
    pub cable_tension_acceleration: f64,
}

impl Default for QuadrotorReference {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            acceleration: Vector3::zeros(),
            jerk: Vector3::zeros(),
            cable_direction: Vector3::new(0.0, 0.0, -1.0),
            cable_angular_velocity: Vector3::zeros(),
            cable_angular_acceleration: Vector3::zeros(),
            cable_angular_jerk: Vector3::zeros(),
            cable_tension: 0.0,
            cable_tension_rate: 0.0,
            cable_angular_snap: Vector3::zeros(),
            cable_tension_acceleration: 0.0,
        }
    }
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

        // Create kinematic constraint for testing
        let constraint = KinematicConstraint::new(
            vec![
                Vector3::new(0.1, 0.0, 0.0),
                Vector3::new(-0.05, 0.087, 0.0),
                Vector3::new(-0.05, -0.087, 0.0),
            ],
            vec![1.0, 1.0, 1.0],
        );

        let quad_ref = reference.for_quadrotor(0, &constraint).unwrap();
        assert!(quad_ref.cable_tension > 0.0);
        // Quadrotor position should differ from load position by approximately cable length
        // (NED convention: cable s = [0,0,1] points from quad toward load)
        let cable_length = 1.0;
        let expected_offset = cable_length; // |p_quad - p_load| ≈ cable_length
        let actual_offset = (quad_ref.position - pos).norm();
        assert!((actual_offset - expected_offset).abs() < 0.2,
            "Quadrotor should be ~1m from load, got {}", actual_offset);
    }

    #[test]
    fn test_different_quadrotor_positions() {
        let pos = Vector3::new(0.0, 0.0, 0.0);
        let ori = UnitQuaternion::identity();
        let traj = PlannedTrajectory::hover(pos, ori, 3, 2.0, 20);

        let reference = traj.get_tracking_reference(0.5).unwrap();

        // Create kinematic constraint
        let constraint = KinematicConstraint::new(
            vec![
                Vector3::new(0.1, 0.0, 0.0),   // Cable 0: offset in +x
                Vector3::new(-0.05, 0.087, 0.0), // Cable 1: offset in -x, +y
                Vector3::new(-0.05, -0.087, 0.0), // Cable 2: offset in -x, -y
            ],
            vec![1.0, 1.0, 1.0],
        );

        let quad_refs = reference.all_quadrotor_references(&constraint);
        assert_eq!(quad_refs.len(), 3);

        // Each quadrotor should have a different position due to different attachment points
        let pos0 = quad_refs[0].position;
        let pos1 = quad_refs[1].position;
        let pos2 = quad_refs[2].position;

        assert!((pos0 - pos1).norm() > 0.01, "Quad 0 and 1 should have different positions");
        assert!((pos1 - pos2).norm() > 0.01, "Quad 1 and 2 should have different positions");
        assert!((pos0 - pos2).norm() > 0.01, "Quad 0 and 2 should have different positions");
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
