//! Quadrotor Trajectory Tracker
//!
//! Implements the onboard INDI trajectory tracking controller from Eq. 15
//! of the paper:
//!
//! Tᵢ,des·zᵢ,des/mᵢ = Kp(pᵢ,ref - pᵢ) + Kv(vᵢ,ref - vᵢ) + aᵢ,ref + fext/mᵢ
//!
//! Where:
//! - Tᵢ,des: Desired thrust magnitude
//! - zᵢ,des: Desired thrust direction (body z-axis)
//! - mᵢ: Quadrotor mass
//! - pᵢ,ref, vᵢ,ref, aᵢ,ref: Reference position, velocity, acceleration
//! - fext: External force (cable tension in our case)
//! - Kp, Kv: Position and velocity gains
//!
//! The controller runs at high frequency (e.g., 300Hz) on each quadrotor,
//! tracking the trajectory computed from the centralized planner.

use nalgebra::{Vector3, UnitQuaternion, Matrix3};
use serde::{Deserialize, Serialize};

use super::indi::{IndiController, IndiParams};

/// Quadrotor trajectory tracker gains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackerGains {
    /// Position proportional gain [1/s²]
    pub kp: Vector3<f64>,
    /// Velocity derivative gain [1/s]
    pub kv: Vector3<f64>,
}

impl Default for TrackerGains {
    fn default() -> Self {
        Self {
            // Default gains from paper (typical aggressive tracking)
            kp: Vector3::new(10.0, 10.0, 15.0),
            kv: Vector3::new(5.0, 5.0, 8.0),
        }
    }
}

/// External force estimate (cable tension)
#[derive(Debug, Clone, Default)]
pub struct ExternalForce {
    /// Estimated external force in world frame [N]
    pub force: Vector3<f64>,
}

impl ExternalForce {
    /// Create from cable tension and direction
    ///
    /// The cable exerts force on the quadrotor in the direction of the cable
    /// (from quadrotor toward load attachment point)
    pub fn from_cable(tension: f64, cable_direction: &Vector3<f64>) -> Self {
        // Cable direction points from quad toward load
        // Force on quad is in the same direction (being pulled by load)
        Self {
            force: tension * cable_direction,
        }
    }
}

/// Reference trajectory point for the quadrotor
#[derive(Debug, Clone, Default)]
pub struct QuadrotorTrajectoryRef {
    /// Reference position [m] (world frame)
    pub position: Vector3<f64>,
    /// Reference velocity [m/s] (world frame)
    pub velocity: Vector3<f64>,
    /// Reference acceleration [m/s²] (world frame) - feedforward
    pub acceleration: Vector3<f64>,
    /// Reference jerk [m/s³] (world frame) - optional feedforward
    pub jerk: Vector3<f64>,
}

/// Current quadrotor state for the tracker
#[derive(Debug, Clone, Default)]
pub struct QuadrotorTrackerState {
    /// Current position [m] (world frame)
    pub position: Vector3<f64>,
    /// Current velocity [m/s] (world frame)
    pub velocity: Vector3<f64>,
    /// Current orientation (body to world)
    pub orientation: UnitQuaternion<f64>,
    /// Current angular velocity [rad/s] (body frame)
    pub angular_velocity: Vector3<f64>,
}

/// Output of the trajectory tracker
#[derive(Debug, Clone)]
pub struct TrackerOutput {
    /// Desired thrust magnitude [N]
    pub thrust: f64,
    /// Desired thrust direction (world frame, unit vector)
    pub thrust_direction: Vector3<f64>,
    /// Desired body z-axis (world frame, unit vector)
    pub body_z_des: Vector3<f64>,
    /// Desired orientation (body to world)
    pub orientation_des: UnitQuaternion<f64>,
    /// Desired angular velocity [rad/s] (body frame)
    pub angular_velocity_des: Vector3<f64>,
    /// Desired torque [N·m] (body frame)
    pub torque: Vector3<f64>,
}

impl Default for TrackerOutput {
    fn default() -> Self {
        Self {
            thrust: 0.0,
            thrust_direction: Vector3::new(0.0, 0.0, 1.0),
            body_z_des: Vector3::new(0.0, 0.0, 1.0),
            orientation_des: UnitQuaternion::identity(),
            angular_velocity_des: Vector3::zeros(),
            torque: Vector3::zeros(),
        }
    }
}

/// Quadrotor trajectory tracker implementing Eq. 15
///
/// This is the onboard controller that runs at high frequency (300Hz)
/// on each individual quadrotor, tracking the trajectory provided by
/// the centralized planner.
#[derive(Debug, Clone)]
pub struct QuadrotorTracker {
    /// Quadrotor mass [kg]
    pub mass: f64,
    /// Quadrotor inertia [kg·m²]
    pub inertia: Matrix3<f64>,
    /// Trajectory tracking gains
    pub gains: TrackerGains,
    /// INDI controller for attitude inner loop
    pub indi: IndiController,
    /// Attitude gains for angular velocity computation
    pub attitude_kp: f64,
    /// Gravity constant [m/s²]
    pub gravity: f64,
    /// Maximum thrust [N]
    pub max_thrust: f64,
    /// Minimum thrust [N]
    pub min_thrust: f64,
}

impl QuadrotorTracker {
    /// Create a new trajectory tracker
    pub fn new(mass: f64, inertia: Matrix3<f64>, gains: TrackerGains) -> Self {
        Self {
            mass,
            inertia,
            gains,
            indi: IndiController::new(IndiParams::default()),
            attitude_kp: 8.0,
            gravity: 9.81,
            max_thrust: mass * 4.0 * 9.81, // 4g max
            min_thrust: 0.1,
        }
    }

    /// Compute control output from Eq. 15
    ///
    /// Tᵢ,des·zᵢ,des/mᵢ = Kp(pᵢ,ref - pᵢ) + Kv(vᵢ,ref - vᵢ) + aᵢ,ref + fext/mᵢ
    ///
    /// # Arguments
    /// * `state` - Current quadrotor state
    /// * `reference` - Reference trajectory point
    /// * `external_force` - External force (cable tension)
    /// * `dt` - Time step [s]
    ///
    /// # Returns
    /// Control output (thrust, orientation, torque)
    pub fn compute(
        &mut self,
        state: &QuadrotorTrackerState,
        reference: &QuadrotorTrajectoryRef,
        external_force: &ExternalForce,
        dt: f64,
    ) -> TrackerOutput {
        // === Step 1: Position control (Eq. 15) ===
        // Compute desired acceleration in world frame
        let pos_error = reference.position - state.position;
        let vel_error = reference.velocity - state.velocity;

        // From dynamics: m*a = T*z + m*g_vec + F_cable
        // => T*z/m = a - g_vec/m - F_cable/m
        //          = a + g_up - F_cable/m
        // where g_up = [0,0,+g] compensates gravity pointing down
        //
        // For desired thrust to compensate cable force, we SUBTRACT it
        // (cable pulling down requires MORE upward thrust)
        let gravity_compensation = Vector3::new(0.0, 0.0, self.gravity);
        let cable_compensation = -external_force.force / self.mass; // Negate to counteract

        let a_des = self.gains.kp.component_mul(&pos_error)
            + self.gains.kv.component_mul(&vel_error)
            + reference.acceleration
            + cable_compensation
            + gravity_compensation;

        // === Step 2: Thrust and direction extraction ===
        // T_des * z_des / m = a_des
        // => T_des = m * ||a_des||
        // => z_des = a_des / ||a_des||

        let a_des_norm = a_des.norm();

        let (thrust, body_z_des) = if a_des_norm > 1e-6 {
            let t = self.mass * a_des_norm;
            let z = a_des / a_des_norm;
            (t.clamp(self.min_thrust, self.max_thrust), z)
        } else {
            // Fallback: hover
            (self.mass * self.gravity, Vector3::new(0.0, 0.0, 1.0))
        };

        // === Step 3: Compute desired orientation ===
        // We need to find R_des such that R_des * [0,0,1] = z_des
        // Also need to specify yaw (heading)
        let orientation_des = self.compute_orientation_from_thrust_direction(
            &body_z_des,
            0.0, // Could add yaw reference here
        );

        // === Step 4: Attitude control ===
        // Compute angular velocity command from attitude error
        let (angular_velocity_des, angular_acceleration_des) = self.compute_attitude_control(
            &state.orientation,
            &orientation_des,
            &state.angular_velocity,
        );

        // === Step 5: INDI inner loop for torque ===
        let torque = self.indi.compute(
            &state.angular_velocity,
            &angular_acceleration_des,
            &self.inertia,
            dt,
        );

        TrackerOutput {
            thrust,
            thrust_direction: body_z_des,
            body_z_des,
            orientation_des,
            angular_velocity_des,
            torque,
        }
    }

    /// Compute desired orientation from thrust direction and yaw
    fn compute_orientation_from_thrust_direction(
        &self,
        body_z_des: &Vector3<f64>,
        yaw_des: f64,
    ) -> UnitQuaternion<f64> {
        // body_z_des = R * [0,0,1]^T is the desired thrust direction
        // We also want to achieve a desired yaw angle

        // Project desired heading onto plane perpendicular to z_des
        let yaw_dir = Vector3::new(yaw_des.cos(), yaw_des.sin(), 0.0);

        // Compute body x and y axes
        let body_y_temp = body_z_des.cross(&yaw_dir);
        let body_y_des = if body_y_temp.norm() > 1e-6 {
            body_y_temp.normalize()
        } else {
            // body_z_des is parallel to yaw_dir, use different reference
            let alt_ref = if body_z_des.z.abs() < 0.9 {
                Vector3::new(0.0, 0.0, 1.0)
            } else {
                Vector3::new(1.0, 0.0, 0.0)
            };
            body_z_des.cross(&alt_ref).normalize()
        };

        let body_x_des = body_y_des.cross(body_z_des);

        // Build rotation matrix
        let rot_matrix = nalgebra::Matrix3::from_columns(&[body_x_des, body_y_des, *body_z_des]);
        UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(rot_matrix))
    }

    /// Compute attitude control (angular velocity and acceleration commands)
    fn compute_attitude_control(
        &self,
        orientation: &UnitQuaternion<f64>,
        orientation_des: &UnitQuaternion<f64>,
        angular_velocity: &Vector3<f64>,
    ) -> (Vector3<f64>, Vector3<f64>) {
        // Compute orientation error
        let q_error = orientation.inverse() * orientation_des;

        // Extract axis-angle error
        let axis_angle = q_error.scaled_axis();

        // Simple proportional control for angular velocity
        let omega_des = self.attitude_kp * axis_angle;

        // Angular velocity error
        let omega_error = omega_des - angular_velocity;

        // Desired angular acceleration (simple proportional)
        let alpha_des = 10.0 * omega_error;

        (omega_des, alpha_des)
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.indi.reset();
    }

    /// Initialize controller with current state
    pub fn initialize(&mut self, state: &QuadrotorTrackerState) {
        self.indi.initialize(&state.angular_velocity, &Vector3::zeros());
    }
}

impl Default for QuadrotorTracker {
    fn default() -> Self {
        let mass = 0.6; // kg
        let inertia = Matrix3::from_diagonal(&Vector3::new(0.004, 0.004, 0.006));
        Self::new(mass, inertia, TrackerGains::default())
    }
}

/// Collection of trackers for multi-quadrotor system
#[derive(Debug, Clone)]
pub struct MultiQuadrotorTracker {
    /// Individual trackers
    pub trackers: Vec<QuadrotorTracker>,
}

impl MultiQuadrotorTracker {
    /// Create trackers for n quadrotors with same parameters
    pub fn uniform(n: usize, mass: f64, inertia: Matrix3<f64>, gains: TrackerGains) -> Self {
        let trackers = (0..n)
            .map(|_| QuadrotorTracker::new(mass, inertia.clone(), gains.clone()))
            .collect();
        Self { trackers }
    }

    /// Number of quadrotors
    pub fn num_quadrotors(&self) -> usize {
        self.trackers.len()
    }

    /// Compute control for all quadrotors
    pub fn compute_all(
        &mut self,
        states: &[QuadrotorTrackerState],
        references: &[QuadrotorTrajectoryRef],
        external_forces: &[ExternalForce],
        dt: f64,
    ) -> Vec<TrackerOutput> {
        self.trackers
            .iter_mut()
            .zip(states.iter())
            .zip(references.iter())
            .zip(external_forces.iter())
            .map(|(((tracker, state), reference), ext_force)| {
                tracker.compute(state, reference, ext_force, dt)
            })
            .collect()
    }

    /// Reset all trackers
    pub fn reset_all(&mut self) {
        for tracker in &mut self.trackers {
            tracker.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_tracker() -> QuadrotorTracker {
        QuadrotorTracker::default()
    }

    #[test]
    fn test_hover_control() {
        let mut tracker = create_test_tracker();

        let state = QuadrotorTrackerState {
            position: Vector3::new(0.0, 0.0, 1.0),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        };

        let reference = QuadrotorTrajectoryRef {
            position: Vector3::new(0.0, 0.0, 1.0), // Same position
            velocity: Vector3::zeros(),
            acceleration: Vector3::zeros(),
            jerk: Vector3::zeros(),
        };

        let external_force = ExternalForce::default(); // No cable tension

        let output = tracker.compute(&state, &reference, &external_force, 0.003);

        // At hover, thrust should be approximately m*g
        let expected_thrust = tracker.mass * tracker.gravity;
        assert_relative_eq!(output.thrust, expected_thrust, epsilon = 0.5);

        // Thrust direction should be up
        assert_relative_eq!(output.body_z_des.z, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_position_tracking() {
        let mut tracker = create_test_tracker();

        // Current state: at origin
        let state = QuadrotorTrackerState {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        };

        // Reference: 1m above
        let reference = QuadrotorTrajectoryRef {
            position: Vector3::new(0.0, 0.0, 1.0),
            velocity: Vector3::zeros(),
            acceleration: Vector3::zeros(),
            jerk: Vector3::zeros(),
        };

        let external_force = ExternalForce::default();
        let output = tracker.compute(&state, &reference, &external_force, 0.003);

        // With position error below, should have higher thrust
        let hover_thrust = tracker.mass * tracker.gravity;
        assert!(output.thrust > hover_thrust, "Should thrust more to go up");
    }

    #[test]
    fn test_with_cable_tension() {
        let mut tracker = create_test_tracker();

        let state = QuadrotorTrackerState {
            position: Vector3::new(0.0, 0.0, 1.0),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        };

        let reference = QuadrotorTrajectoryRef {
            position: Vector3::new(0.0, 0.0, 1.0),
            velocity: Vector3::zeros(),
            acceleration: Vector3::zeros(),
            jerk: Vector3::zeros(),
        };

        // Cable pulling down with 5N
        let cable_direction = Vector3::new(0.0, 0.0, -1.0);
        let external_force = ExternalForce::from_cable(5.0, &cable_direction);

        let output = tracker.compute(&state, &reference, &external_force, 0.003);

        // With cable pulling down, thrust should be higher than hover
        let hover_thrust = tracker.mass * tracker.gravity;
        assert!(output.thrust > hover_thrust, "Should compensate for cable tension");
    }

    #[test]
    fn test_multi_quadrotor_different_outputs() {
        let mass = 0.6;
        let inertia = Matrix3::from_diagonal(&Vector3::new(0.004, 0.004, 0.006));
        let gains = TrackerGains::default();
        let mut multi = MultiQuadrotorTracker::uniform(3, mass, inertia, gains);

        // Different states for each quadrotor
        let states = vec![
            QuadrotorTrackerState {
                position: Vector3::new(1.0, 0.0, 1.0),
                ..Default::default()
            },
            QuadrotorTrackerState {
                position: Vector3::new(-0.5, 0.866, 1.0),
                ..Default::default()
            },
            QuadrotorTrackerState {
                position: Vector3::new(-0.5, -0.866, 1.0),
                ..Default::default()
            },
        ];

        // Same reference for all (load center)
        let reference = QuadrotorTrajectoryRef {
            position: Vector3::new(0.0, 0.0, 2.0), // Different from state
            ..Default::default()
        };
        let references = vec![reference.clone(), reference.clone(), reference.clone()];

        // Different cable tensions
        let external_forces = vec![
            ExternalForce::from_cable(5.0, &Vector3::new(0.0, 0.0, -1.0)),
            ExternalForce::from_cable(3.0, &Vector3::new(0.0, 0.0, -1.0)),
            ExternalForce::from_cable(7.0, &Vector3::new(0.0, 0.0, -1.0)),
        ];

        let outputs = multi.compute_all(&states, &references, &external_forces, 0.003);

        assert_eq!(outputs.len(), 3);

        // Outputs should be different due to different states and cable tensions
        let thrusts: Vec<f64> = outputs.iter().map(|o| o.thrust).collect();
        assert!((thrusts[0] - thrusts[1]).abs() > 0.01 || (thrusts[1] - thrusts[2]).abs() > 0.01,
            "Different states/cables should produce different thrusts");
    }
}
