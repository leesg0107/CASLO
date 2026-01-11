//! Kinematic Constraint Derivatives
//!
//! Implements Eq. (5) and Eq. (S1) from the paper for computing
//! quadrotor trajectories from load-cable state.
//!
//! The kinematic constraint relates quadrotor position to load-cable state:
//!
//! Eq. (5):  pᵢ = p + R(q)ρᵢ - lᵢsᵢ
//!
//! Taking derivatives (Eq. S1 in supplementary material):
//!
//! Velocity:  ṗᵢ = ṗ + ω × (Rρᵢ) - lᵢ(rᵢ × sᵢ)
//!
//! Acceleration:  p̈ᵢ = p̈ + α × (Rρᵢ) + ω × (ω × (Rρᵢ)) - lᵢ(ṙᵢ × sᵢ + rᵢ × ṡᵢ)
//!
//! Jerk:  p⃛ᵢ = p⃛ + α̇ × (Rρᵢ) + 2α × (ω × (Rρᵢ)) + ω × (ω × (ω × (Rρᵢ)))
//!              - lᵢ(r̈ᵢ × sᵢ + 2ṙᵢ × ṡᵢ + rᵢ × s̈ᵢ)
//!
//! where:
//! - p: load position (world frame)
//! - R(q): rotation matrix from load body frame to world frame
//! - ρᵢ: attachment point in load body frame
//! - lᵢ: cable length
//! - sᵢ: cable direction (unit vector, world frame)
//! - rᵢ: cable angular velocity (such that ṡᵢ = rᵢ × sᵢ)
//! - ω: load angular velocity (body frame)
//! - α: load angular acceleration (body frame)

use nalgebra::{Vector3, UnitQuaternion};

/// Load state required for kinematic constraint computation
#[derive(Debug, Clone)]
pub struct LoadKinematicState {
    /// Position [m] (world frame)
    pub position: Vector3<f64>,
    /// Velocity [m/s] (world frame)
    pub velocity: Vector3<f64>,
    /// Acceleration [m/s²] (world frame)
    pub acceleration: Vector3<f64>,
    /// Jerk [m/s³] (world frame)
    pub jerk: Vector3<f64>,
    /// Orientation (body to world)
    pub orientation: UnitQuaternion<f64>,
    /// Angular velocity [rad/s] (body frame)
    pub angular_velocity: Vector3<f64>,
    /// Angular acceleration [rad/s²] (body frame)
    pub angular_acceleration: Vector3<f64>,
    /// Angular jerk [rad/s³] (body frame)
    pub angular_jerk: Vector3<f64>,
}

impl Default for LoadKinematicState {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            acceleration: Vector3::zeros(),
            jerk: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
            angular_acceleration: Vector3::zeros(),
            angular_jerk: Vector3::zeros(),
        }
    }
}

/// Cable state required for kinematic constraint computation
#[derive(Debug, Clone)]
pub struct CableKinematicState {
    /// Cable direction (unit vector, world frame)
    pub direction: Vector3<f64>,
    /// Cable angular velocity [rad/s]
    /// (such that ṡ = r × s)
    pub angular_velocity: Vector3<f64>,
    /// Cable angular acceleration [rad/s²]
    pub angular_acceleration: Vector3<f64>,
    /// Cable angular jerk [rad/s³]
    pub angular_jerk: Vector3<f64>,
}

impl Default for CableKinematicState {
    fn default() -> Self {
        Self {
            direction: Vector3::new(0.0, 0.0, -1.0), // Pointing down
            angular_velocity: Vector3::zeros(),
            angular_acceleration: Vector3::zeros(),
            angular_jerk: Vector3::zeros(),
        }
    }
}

/// Quadrotor trajectory point derived from kinematic constraint
#[derive(Debug, Clone)]
pub struct QuadrotorTrajectoryPoint {
    /// Position [m] (world frame)
    pub position: Vector3<f64>,
    /// Velocity [m/s] (world frame)
    pub velocity: Vector3<f64>,
    /// Acceleration [m/s²] (world frame)
    pub acceleration: Vector3<f64>,
    /// Jerk [m/s³] (world frame)
    pub jerk: Vector3<f64>,
}

impl Default for QuadrotorTrajectoryPoint {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            acceleration: Vector3::zeros(),
            jerk: Vector3::zeros(),
        }
    }
}

/// Kinematic constraint calculator
///
/// Computes quadrotor trajectories from load-cable state using
/// the kinematic constraint and its derivatives.
#[derive(Debug, Clone)]
pub struct KinematicConstraint {
    /// Attachment points in load body frame [m]
    pub attachment_points: Vec<Vector3<f64>>,
    /// Cable lengths [m]
    pub cable_lengths: Vec<f64>,
}

impl KinematicConstraint {
    /// Create a new kinematic constraint calculator
    pub fn new(attachment_points: Vec<Vector3<f64>>, cable_lengths: Vec<f64>) -> Self {
        assert_eq!(attachment_points.len(), cable_lengths.len());
        Self {
            attachment_points,
            cable_lengths,
        }
    }

    /// Number of cables/quadrotors
    pub fn num_cables(&self) -> usize {
        self.cable_lengths.len()
    }

    /// Compute quadrotor position from Eq. (5)
    ///
    /// pᵢ = p + R(q)ρᵢ - lᵢsᵢ
    pub fn quadrotor_position(
        &self,
        load: &LoadKinematicState,
        cable: &CableKinematicState,
        cable_index: usize,
    ) -> Vector3<f64> {
        let r = load.orientation.to_rotation_matrix();
        let rho = &self.attachment_points[cable_index];
        let l = self.cable_lengths[cable_index];
        let s = &cable.direction;

        load.position + r * rho - l * s
    }

    /// Compute quadrotor velocity from Eq. (S1)
    ///
    /// ṗᵢ = ṗ + ω_world × (Rρᵢ) - lᵢ(rᵢ × sᵢ)
    ///
    /// Note: ω is in body frame, so we convert: ω_world = R * ω
    pub fn quadrotor_velocity(
        &self,
        load: &LoadKinematicState,
        cable: &CableKinematicState,
        cable_index: usize,
    ) -> Vector3<f64> {
        let r = load.orientation.to_rotation_matrix();
        let rho = &self.attachment_points[cable_index];
        let l = self.cable_lengths[cable_index];
        let s = &cable.direction;
        let r_cable = &cable.angular_velocity;

        // ω in world frame
        let omega_world = r * load.angular_velocity;

        // Rρᵢ (attachment point in world frame, relative to load COM)
        let rho_world = r * rho;

        // ṡ = r × s
        let s_dot = r_cable.cross(s);

        load.velocity + omega_world.cross(&rho_world) - l * s_dot
    }

    /// Compute quadrotor acceleration from Eq. (S1)
    ///
    /// p̈ᵢ = p̈ + α_world × (Rρᵢ) + ω_world × (ω_world × (Rρᵢ)) - lᵢ(ṙᵢ × sᵢ + rᵢ × ṡᵢ)
    pub fn quadrotor_acceleration(
        &self,
        load: &LoadKinematicState,
        cable: &CableKinematicState,
        cable_index: usize,
    ) -> Vector3<f64> {
        let r = load.orientation.to_rotation_matrix();
        let rho = &self.attachment_points[cable_index];
        let l = self.cable_lengths[cable_index];
        let s = &cable.direction;
        let r_cable = &cable.angular_velocity;
        let r_dot = &cable.angular_acceleration;

        // Convert body frame angular quantities to world frame
        let omega_world = r * load.angular_velocity;
        let alpha_world = r * load.angular_acceleration;

        // Rρᵢ
        let rho_world = r * rho;

        // Cable direction derivatives
        let s_dot = r_cable.cross(s);
        let s_ddot = r_dot.cross(s) + r_cable.cross(&s_dot);

        // Centripetal term: ω × (ω × (Rρᵢ))
        let centripetal = omega_world.cross(&omega_world.cross(&rho_world));

        // Angular acceleration term: α × (Rρᵢ)
        let angular_accel_term = alpha_world.cross(&rho_world);

        // Cable contribution: -l(ṙ × s + r × ṡ)
        let cable_term = r_dot.cross(s) + r_cable.cross(&s_dot);

        load.acceleration + angular_accel_term + centripetal - l * cable_term
    }

    /// Compute quadrotor jerk from Eq. (S1)
    ///
    /// p⃛ᵢ = p⃛ + α̇_world × (Rρᵢ) + 2α_world × (ω_world × (Rρᵢ))
    ///       + ω_world × (ω_world × (ω_world × (Rρᵢ)))
    ///       + α_world × (ω_world × (Rρᵢ))
    ///       - lᵢ(r̈ᵢ × sᵢ + 2ṙᵢ × ṡᵢ + rᵢ × s̈ᵢ)
    pub fn quadrotor_jerk(
        &self,
        load: &LoadKinematicState,
        cable: &CableKinematicState,
        cable_index: usize,
    ) -> Vector3<f64> {
        let r = load.orientation.to_rotation_matrix();
        let rho = &self.attachment_points[cable_index];
        let l = self.cable_lengths[cable_index];
        let s = &cable.direction;
        let r_cable = &cable.angular_velocity;
        let r_dot = &cable.angular_acceleration;
        let r_ddot = &cable.angular_jerk;

        // Convert body frame angular quantities to world frame
        let omega_world = r * load.angular_velocity;
        let alpha_world = r * load.angular_acceleration;
        let jerk_angular_world = r * load.angular_jerk;

        // Rρᵢ
        let rho_world = r * rho;

        // Cable direction derivatives
        let s_dot = r_cable.cross(s);
        let s_ddot = r_dot.cross(s) + r_cable.cross(&s_dot);
        let s_tdot = r_ddot.cross(s) + 2.0 * r_dot.cross(&s_dot) + r_cable.cross(&s_ddot);

        // Term 1: α̇ × (Rρᵢ)
        let term1 = jerk_angular_world.cross(&rho_world);

        // Term 2: α × (ω × (Rρᵢ)) + ω × (α × (Rρᵢ))
        let omega_cross_rho = omega_world.cross(&rho_world);
        let alpha_cross_rho = alpha_world.cross(&rho_world);
        let term2 = alpha_world.cross(&omega_cross_rho) + omega_world.cross(&alpha_cross_rho);

        // Term 3: ω × (ω × (ω × (Rρᵢ)))
        let term3 = omega_world.cross(&omega_world.cross(&omega_cross_rho));

        // Cable contribution: -l(r̈ × s + 2ṙ × ṡ + r × s̈)
        let cable_term = r_ddot.cross(s) + 2.0 * r_dot.cross(&s_dot) + r_cable.cross(&s_ddot);

        load.jerk + term1 + term2 + term3 - l * cable_term
    }

    /// Compute full quadrotor trajectory point (position, velocity, acceleration, jerk)
    pub fn quadrotor_trajectory_point(
        &self,
        load: &LoadKinematicState,
        cable: &CableKinematicState,
        cable_index: usize,
    ) -> QuadrotorTrajectoryPoint {
        QuadrotorTrajectoryPoint {
            position: self.quadrotor_position(load, cable, cable_index),
            velocity: self.quadrotor_velocity(load, cable, cable_index),
            acceleration: self.quadrotor_acceleration(load, cable, cable_index),
            jerk: self.quadrotor_jerk(load, cable, cable_index),
        }
    }

    /// Compute trajectory points for all quadrotors
    pub fn all_quadrotor_trajectories(
        &self,
        load: &LoadKinematicState,
        cables: &[CableKinematicState],
    ) -> Vec<QuadrotorTrajectoryPoint> {
        cables
            .iter()
            .enumerate()
            .map(|(i, cable)| self.quadrotor_trajectory_point(load, cable, i))
            .collect()
    }
}

/// Convert OCP load state and its derivatives to LoadKinematicState
///
/// This is used to bridge the planner output to the kinematic constraint calculator.
#[derive(Debug, Clone)]
pub struct LoadStateWithDerivatives {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub orientation: UnitQuaternion<f64>,
    pub angular_velocity: Vector3<f64>,
}

impl LoadStateWithDerivatives {
    /// Compute acceleration from load dynamics
    ///
    /// From Eq. 2: v̇ = -1/m · Σᵢ tᵢsᵢ + g
    pub fn compute_acceleration(
        &self,
        tensions: &[f64],
        directions: &[Vector3<f64>],
        mass: f64,
        gravity: f64,
    ) -> Vector3<f64> {
        let mut cable_force = Vector3::zeros();
        for (t, s) in tensions.iter().zip(directions.iter()) {
            cable_force += *t * s;
        }
        let g = Vector3::new(0.0, 0.0, -gravity);
        -cable_force / mass + g
    }

    /// Compute angular acceleration from load dynamics
    ///
    /// From Eq. 2: Jω̇ = -ω × Jω + Σᵢ tᵢ(R^T sᵢ × ρᵢ)
    pub fn compute_angular_acceleration(
        &self,
        tensions: &[f64],
        directions: &[Vector3<f64>],
        attachment_points: &[Vector3<f64>],
        inertia: &nalgebra::Matrix3<f64>,
        inertia_inv: &nalgebra::Matrix3<f64>,
    ) -> Vector3<f64> {
        let r = self.orientation.to_rotation_matrix();
        let omega = &self.angular_velocity;

        // Gyroscopic term
        let gyro = -omega.cross(&(inertia * omega));

        // Cable torques
        let mut cable_torque = Vector3::zeros();
        for (i, (t, s)) in tensions.iter().zip(directions.iter()).enumerate() {
            let rho = &attachment_points[i];
            let s_body = r.inverse() * s;
            cable_torque += *t * s_body.cross(rho);
        }

        inertia_inv * (gyro + cable_torque)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_constraint() -> KinematicConstraint {
        KinematicConstraint::new(
            vec![
                Vector3::new(0.1, 0.0, 0.0),
                Vector3::new(-0.05, 0.087, 0.0),
                Vector3::new(-0.05, -0.087, 0.0),
            ],
            vec![1.0, 1.0, 1.0],
        )
    }

    #[test]
    fn test_position_at_rest() {
        let constraint = create_test_constraint();
        let load = LoadKinematicState::default();
        let cable = CableKinematicState::default(); // Pointing down

        let pos = constraint.quadrotor_position(&load, &cable, 0);

        // Quadrotor should be above the attachment point (cable pointing down)
        // p_i = 0 + R*[0.1, 0, 0] - 1.0*[0, 0, -1] = [0.1, 0, 1]
        assert_relative_eq!(pos.x, 0.1, epsilon = 1e-10);
        assert_relative_eq!(pos.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(pos.z, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_velocity_at_rest() {
        let constraint = create_test_constraint();
        let load = LoadKinematicState::default();
        let cable = CableKinematicState::default();

        let vel = constraint.quadrotor_velocity(&load, &cable, 0);

        // All zeros when at rest
        assert_relative_eq!(vel.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_velocity_with_load_translation() {
        let constraint = create_test_constraint();
        let mut load = LoadKinematicState::default();
        load.velocity = Vector3::new(1.0, 0.0, 0.0); // Moving in x direction
        let cable = CableKinematicState::default();

        let vel = constraint.quadrotor_velocity(&load, &cable, 0);

        // Quadrotor should have same velocity as load when no rotation
        assert_relative_eq!(vel.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(vel.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(vel.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_velocity_with_cable_rotation() {
        let constraint = create_test_constraint();
        let load = LoadKinematicState::default();
        let mut cable = CableKinematicState::default();
        cable.angular_velocity = Vector3::new(1.0, 0.0, 0.0); // Rotating around x-axis

        let vel = constraint.quadrotor_velocity(&load, &cable, 0);

        // ṡ = r × s = [1,0,0] × [0,0,-1] = [0,1,0]
        // ṗᵢ = 0 - 1.0 * [0,1,0] = [0,-1,0]
        assert_relative_eq!(vel.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(vel.y, -1.0, epsilon = 1e-10);
        assert_relative_eq!(vel.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_acceleration_at_rest() {
        let constraint = create_test_constraint();
        let load = LoadKinematicState::default();
        let cable = CableKinematicState::default();

        let acc = constraint.quadrotor_acceleration(&load, &cable, 0);

        // All zeros when at rest
        assert_relative_eq!(acc.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_different_cables_different_positions() {
        let constraint = create_test_constraint();
        let load = LoadKinematicState::default();

        let cable1 = CableKinematicState::default();
        let mut cable2 = CableKinematicState::default();
        cable2.direction = Vector3::new(0.0, 0.1, -0.995).normalize(); // Slightly tilted

        let pos1 = constraint.quadrotor_position(&load, &cable1, 0);
        let pos2 = constraint.quadrotor_position(&load, &cable2, 1);

        // Different cables should give different positions
        assert!((pos1 - pos2).norm() > 0.01);
    }

    #[test]
    fn test_all_quadrotor_trajectories() {
        let constraint = create_test_constraint();
        let load = LoadKinematicState::default();
        let cables = vec![
            CableKinematicState::default(),
            CableKinematicState::default(),
            CableKinematicState::default(),
        ];

        let trajectories = constraint.all_quadrotor_trajectories(&load, &cables);

        assert_eq!(trajectories.len(), 3);
        // Each quadrotor should have different position due to different attachment points
        assert!((trajectories[0].position - trajectories[1].position).norm() > 0.01);
        assert!((trajectories[1].position - trajectories[2].position).norm() > 0.01);
    }
}
