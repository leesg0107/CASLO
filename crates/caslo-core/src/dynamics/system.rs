//! Full system dynamics
//!
//! Combines load, cable, and quadrotor dynamics into a complete
//! multi-quadrotor cable-suspended load system.
//!
//! The kinematic constraint from Eq. (5):
//! pᵢ = p + R(q)ρᵢ - lᵢsᵢ
//!
//! Ensures quadrotor positions are determined by:
//! - Load position p and orientation q
//! - Attachment point ρᵢ in load body frame
//! - Cable length lᵢ and direction sᵢ

use nalgebra::{Vector3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use super::{
    LoadState, LoadParams, LoadDynamics,
    CableState, CableParams, CableDynamics, CableInput, MultiCableState,
    QuadrotorState, QuadrotorParams, QuadrotorDynamics, QuadrotorInput,
};

/// Full system state
#[derive(Debug, Clone)]
pub struct SystemState {
    /// Load state
    pub load: LoadState,
    /// Cable states (one per quadrotor)
    pub cables: MultiCableState,
    /// Quadrotor states (derived from kinematic constraint)
    pub quadrotors: Vec<QuadrotorState>,
}

impl SystemState {
    /// Create system state with given load and cable states
    ///
    /// Quadrotor states are computed from the kinematic constraint
    pub fn new(
        load: LoadState,
        cables: MultiCableState,
        load_dynamics: &LoadDynamics,
        cable_dynamics: &[CableDynamics],
    ) -> Self {
        let mut quadrotors = Vec::with_capacity(cables.cables.len());

        for (i, (cable, cable_dyn)) in cables.cables.iter().zip(cable_dynamics.iter()).enumerate() {
            let attach_world = load_dynamics.attachment_world(&load, i);
            let quad_pos = cable_dyn.quadrotor_position(&attach_world, &cable.direction);

            // For now, assume quadrotor orientation aligns with cable direction
            // (thrust pointing toward load)
            let quad_orientation = orientation_from_cable_direction(&cable.direction);

            quadrotors.push(QuadrotorState {
                position: quad_pos,
                velocity: Vector3::zeros(), // Will be computed from kinematics
                orientation: quad_orientation,
                angular_velocity: Vector3::zeros(),
            });
        }

        Self {
            load,
            cables,
            quadrotors,
        }
    }

    /// Number of quadrotors in the system
    pub fn num_quadrotors(&self) -> usize {
        self.cables.cables.len()
    }
}

/// Compute quadrotor orientation from cable direction
///
/// The quadrotor's body z-axis should point opposite to the cable direction
/// (thrust toward the load)
fn orientation_from_cable_direction(cable_dir: &Vector3<f64>) -> UnitQuaternion<f64> {
    // Desired body z-axis (thrust direction) is opposite to cable direction
    let z_desired = -cable_dir;

    // Choose an arbitrary x-axis perpendicular to z
    let x_temp = if z_desired.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };

    let y_desired = z_desired.cross(&x_temp).normalize();
    let x_desired = y_desired.cross(&z_desired);

    // Build rotation matrix and convert to quaternion
    let rot = nalgebra::Matrix3::from_columns(&[x_desired, y_desired, z_desired]);
    UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(rot))
}

/// System parameters
#[derive(Debug, Clone)]
pub struct SystemParams {
    /// Load parameters
    pub load: LoadParams,
    /// Cable parameters (one per cable)
    pub cables: Vec<CableParams>,
    /// Quadrotor parameters (one per quadrotor)
    pub quadrotors: Vec<QuadrotorParams>,
}

impl SystemParams {
    /// Create system with identical cables and quadrotors
    pub fn uniform(
        load: LoadParams,
        cable: CableParams,
        quadrotor: QuadrotorParams,
    ) -> Self {
        let n = load.num_cables();
        Self {
            load,
            cables: vec![cable; n],
            quadrotors: vec![quadrotor; n],
        }
    }
}

/// System control input
#[derive(Debug, Clone)]
pub struct SystemInput {
    /// Cable inputs (angular jerk and tension acceleration)
    pub cables: Vec<CableInput>,
    /// Quadrotor inputs (thrust and torque)
    pub quadrotors: Vec<QuadrotorInput>,
}

impl SystemInput {
    /// Create zero input for given number of agents
    pub fn zeros(n: usize) -> Self {
        Self {
            cables: vec![CableInput::default(); n],
            quadrotors: vec![QuadrotorInput::default(); n],
        }
    }
}

/// Full system dynamics
#[derive(Debug, Clone)]
pub struct SystemDynamics {
    /// Load dynamics
    pub load: LoadDynamics,
    /// Cable dynamics (one per cable)
    pub cables: Vec<CableDynamics>,
    /// Quadrotor dynamics (one per quadrotor)
    pub quadrotors: Vec<QuadrotorDynamics>,
}

impl SystemDynamics {
    pub fn new(params: SystemParams) -> Self {
        let load = LoadDynamics::new(params.load);
        let cables = params.cables.into_iter().map(CableDynamics::new).collect();
        let quadrotors = params.quadrotors.into_iter().map(QuadrotorDynamics::new).collect();

        Self { load, cables, quadrotors }
    }

    /// Number of agents (quadrotors) in the system
    pub fn num_agents(&self) -> usize {
        self.cables.len()
    }

    /// Integrate full system state forward in time
    pub fn integrate(&self, state: &SystemState, input: &SystemInput, dt: f64) -> SystemState {
        // Get current cable tensions and directions for load dynamics
        let tensions = state.cables.tensions();
        let directions = state.cables.directions();

        // Integrate load dynamics
        let load_deriv = self.load.state_derivative(&state.load, &tensions, &directions);
        let new_load = integrate_load_state(&state.load, &load_deriv, dt);

        // Integrate cable dynamics
        let new_cables: Vec<CableState> = state.cables.cables.iter()
            .zip(self.cables.iter())
            .zip(input.cables.iter())
            .map(|((cable_state, cable_dyn), cable_input)| {
                cable_dyn.integrate(cable_state, cable_input, dt)
            })
            .collect();

        // Compute quadrotor states from kinematic constraint
        let mut new_quadrotors = Vec::with_capacity(self.num_agents());
        for (i, (cable, cable_dyn)) in new_cables.iter().zip(self.cables.iter()).enumerate() {
            let attach_world = self.load.attachment_world(&new_load, i);
            let quad_pos = cable_dyn.quadrotor_position(&attach_world, &cable.direction);
            let quad_orientation = orientation_from_cable_direction(&cable.direction);

            // Compute quadrotor velocity from load kinematics (simplified)
            // Full derivation would involve differentiating the kinematic constraint
            new_quadrotors.push(QuadrotorState {
                position: quad_pos,
                velocity: Vector3::zeros(), // TODO: Compute from constraint derivative
                orientation: quad_orientation,
                angular_velocity: Vector3::zeros(),
            });
        }

        SystemState {
            load: new_load,
            cables: MultiCableState::new(new_cables),
            quadrotors: new_quadrotors,
        }
    }

    /// Check if all cables are taut (positive tension)
    pub fn all_cables_taut(&self, state: &SystemState) -> bool {
        state.cables.cables.iter().all(|c| c.tension > 0.0)
    }

    /// Compute total cable force on load
    pub fn total_cable_force(&self, state: &SystemState) -> Vector3<f64> {
        let mut force = Vector3::zeros();
        for cable in &state.cables.cables {
            force += cable.tension * cable.direction;
        }
        force
    }
}

/// Helper to integrate load state
fn integrate_load_state(
    state: &LoadState,
    deriv: &super::load::LoadStateDerivative,
    dt: f64,
) -> LoadState {
    // Convert Vector4 derivative to Quaternion for integration
    let q_dot = nalgebra::Quaternion::new(
        deriv.orientation_derivative[0],
        deriv.orientation_derivative[1],
        deriv.orientation_derivative[2],
        deriv.orientation_derivative[3],
    );

    let new_orientation = UnitQuaternion::from_quaternion(
        state.orientation.quaternion() + q_dot * dt
    );

    LoadState {
        position: state.position + deriv.velocity * dt,
        velocity: state.velocity + deriv.acceleration * dt,
        orientation: new_orientation,
        angular_velocity: state.angular_velocity + deriv.angular_acceleration * dt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_system() -> (SystemDynamics, SystemState) {
        // Create a simple 3-quadrotor system
        let load_params = LoadParams::new(
            1.0,
            Vector3::new(0.01, 0.01, 0.01),
            vec![
                Vector3::new(0.1, 0.0, 0.0),
                Vector3::new(-0.05, 0.087, 0.0), // 120° apart
                Vector3::new(-0.05, -0.087, 0.0),
            ],
        );

        let cable_params = CableParams::new(1.0);
        let quad_params = QuadrotorParams::default();

        let params = SystemParams::uniform(load_params, cable_params, quad_params);
        let dynamics = SystemDynamics::new(params);

        // Initial state: load at origin, cables pointing down
        let load_state = LoadState::default();
        let cable_states = vec![
            CableState::pointing_down(10.0),
            CableState::pointing_down(10.0),
            CableState::pointing_down(10.0),
        ];

        let state = SystemState::new(
            load_state,
            MultiCableState::new(cable_states),
            &dynamics.load,
            &dynamics.cables,
        );

        (dynamics, state)
    }

    #[test]
    fn test_system_creation() {
        let (dynamics, state) = create_test_system();

        assert_eq!(dynamics.num_agents(), 3);
        assert_eq!(state.num_quadrotors(), 3);
    }

    #[test]
    fn test_quadrotor_positions() {
        let (dynamics, state) = create_test_system();

        // Quadrotors should be above the load (since cables point down)
        for quad in &state.quadrotors {
            assert!(quad.position.z > state.load.position.z);
        }
    }

    #[test]
    fn test_cables_taut() {
        let (dynamics, state) = create_test_system();
        assert!(dynamics.all_cables_taut(&state));
    }

    #[test]
    fn test_total_cable_force() {
        let (dynamics, state) = create_test_system();
        let force = dynamics.total_cable_force(&state);

        // With 3 cables pointing down at 10N each, total force is 30N down
        assert_relative_eq!(force.z, -30.0, epsilon = 1e-10);
        assert_relative_eq!(force.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(force.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_system_integration() {
        let (dynamics, state) = create_test_system();
        let input = SystemInput::zeros(3);
        let dt = 0.01;

        let new_state = dynamics.integrate(&state, &input, dt);

        // State should have changed (load falling under gravity)
        assert!(new_state.load.velocity.norm() > 0.0);
    }

    #[test]
    fn test_orientation_from_cable_direction() {
        let cable_dir = Vector3::new(0.0, 0.0, -1.0); // Down
        let orientation = orientation_from_cable_direction(&cable_dir);

        // Thrust should point up (opposite to cable direction)
        let thrust_dir = orientation * Vector3::new(0.0, 0.0, 1.0);
        assert_relative_eq!(thrust_dir, Vector3::new(0.0, 0.0, 1.0), epsilon = 1e-10);
    }
}
