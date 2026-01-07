//! Cable dynamics
//!
//! Implements cable kinematics from Eq. (3) in the paper:
//!
//! ṡᵢ = rᵢ × sᵢ
//! r̈ᵢ = γᵢ  (jerk input)
//! ẗᵢ = λᵢ  (tension rate input)
//!
//! where:
//! - sᵢ ∈ S² is the cable direction (unit vector)
//! - rᵢ is the cable angular velocity
//! - tᵢ is the cable tension
//! - γᵢ is the cable angular jerk (control input)
//! - λᵢ is the tension acceleration (control input)

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use crate::math::sphere::{direction_derivative, integrate_direction, project_to_sphere};

/// Cable state for a single cable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CableState {
    /// Cable direction (unit vector, world frame) sᵢ ∈ S²
    pub direction: Vector3<f64>,
    /// Cable angular velocity [rad/s] rᵢ
    pub angular_velocity: Vector3<f64>,
    /// Cable angular acceleration [rad/s²] ṙᵢ
    pub angular_acceleration: Vector3<f64>,
    /// Cable tension [N] tᵢ
    pub tension: f64,
    /// Cable tension rate [N/s] ṫᵢ
    pub tension_rate: f64,
}

impl CableState {
    /// Create a new cable state
    pub fn new(direction: Vector3<f64>, tension: f64) -> Self {
        Self {
            direction: project_to_sphere(&direction),
            angular_velocity: Vector3::zeros(),
            angular_acceleration: Vector3::zeros(),
            tension,
            tension_rate: 0.0,
        }
    }

    /// Create cable pointing straight down (common initial state)
    pub fn pointing_down(tension: f64) -> Self {
        Self::new(Vector3::new(0.0, 0.0, -1.0), tension)
    }

    /// Compute direction derivative: ṡ = r × s
    pub fn direction_derivative(&self) -> Vector3<f64> {
        direction_derivative(&self.direction, &self.angular_velocity)
    }
}

impl Default for CableState {
    fn default() -> Self {
        Self::pointing_down(0.0)
    }
}

/// Cable parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CableParams {
    /// Cable length [m]
    pub length: f64,
    /// Minimum tension [N]
    pub min_tension: f64,
    /// Maximum tension [N]
    pub max_tension: f64,
    /// Maximum angular velocity magnitude [rad/s]
    pub max_angular_velocity: f64,
}

impl Default for CableParams {
    fn default() -> Self {
        Self {
            length: 1.0,
            min_tension: 0.0,
            max_tension: 100.0,
            max_angular_velocity: 10.0,
        }
    }
}

impl CableParams {
    pub fn new(length: f64) -> Self {
        Self {
            length,
            ..Default::default()
        }
    }
}

/// Cable control inputs
#[derive(Debug, Clone, Default)]
pub struct CableInput {
    /// Angular jerk [rad/s³] γᵢ
    pub angular_jerk: Vector3<f64>,
    /// Tension acceleration [N/s²] λᵢ
    pub tension_acceleration: f64,
}

/// Cable dynamics model
#[derive(Debug, Clone)]
pub struct CableDynamics {
    pub params: CableParams,
}

impl CableDynamics {
    pub fn new(params: CableParams) -> Self {
        Self { params }
    }

    /// Integrate cable state forward in time
    ///
    /// Implements Eq. (3):
    /// - ṡᵢ = rᵢ × sᵢ
    /// - r̈ᵢ = γᵢ
    /// - ẗᵢ = λᵢ
    pub fn integrate(&self, state: &CableState, input: &CableInput, dt: f64) -> CableState {
        // Direction derivative: ṡ = r × s
        let s_dot = state.direction_derivative();

        // Integrate direction on S² manifold
        let new_direction = integrate_direction(&state.direction, &s_dot, dt);

        // Integrate angular velocity: ṙ = angular_acceleration
        let new_angular_velocity = state.angular_velocity + state.angular_acceleration * dt;

        // Integrate angular acceleration: r̈ = γ
        let new_angular_acceleration = state.angular_acceleration + input.angular_jerk * dt;

        // Integrate tension: ṫ = tension_rate
        let new_tension = (state.tension + state.tension_rate * dt)
            .clamp(self.params.min_tension, self.params.max_tension);

        // Integrate tension rate: ẗ = λ
        let new_tension_rate = state.tension_rate + input.tension_acceleration * dt;

        CableState {
            direction: new_direction,
            angular_velocity: new_angular_velocity,
            angular_acceleration: new_angular_acceleration,
            tension: new_tension,
            tension_rate: new_tension_rate,
        }
    }

    /// Compute quadrotor position from load attachment point
    ///
    /// From Eq. (5): pᵢ = p + R(q)ρᵢ - lᵢsᵢ
    ///
    /// # Arguments
    /// * `attachment_world` - World position of attachment point on load
    /// * `direction` - Cable direction (unit vector)
    ///
    /// # Returns
    /// Quadrotor position in world frame
    pub fn quadrotor_position(
        &self,
        attachment_world: &Vector3<f64>,
        direction: &Vector3<f64>,
    ) -> Vector3<f64> {
        attachment_world - self.params.length * direction
    }

    /// Compute required thrust direction for the quadrotor
    ///
    /// The thrust must be aligned with the cable direction (pointing toward load)
    pub fn thrust_direction(&self, direction: &Vector3<f64>) -> Vector3<f64> {
        // Thrust points opposite to cable direction (toward load)
        -direction
    }
}

/// Multi-cable state (for entire system)
#[derive(Debug, Clone, Default)]
pub struct MultiCableState {
    pub cables: Vec<CableState>,
}

impl MultiCableState {
    pub fn new(cables: Vec<CableState>) -> Self {
        Self { cables }
    }

    /// Get all cable directions
    pub fn directions(&self) -> Vec<Vector3<f64>> {
        self.cables.iter().map(|c| c.direction).collect()
    }

    /// Get all cable tensions
    pub fn tensions(&self) -> Vec<f64> {
        self.cables.iter().map(|c| c.tension).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_cable_state_default() {
        let cable = CableState::default();
        assert_relative_eq!(cable.direction, Vector3::new(0.0, 0.0, -1.0), epsilon = 1e-10);
        assert_relative_eq!(cable.tension, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cable_direction_unit() {
        let cable = CableState::new(Vector3::new(1.0, 1.0, 1.0), 10.0);
        assert_relative_eq!(cable.direction.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_direction_derivative() {
        let mut cable = CableState::pointing_down(10.0);
        cable.angular_velocity = Vector3::new(1.0, 0.0, 0.0);

        // ṡ = r × s = [1,0,0] × [0,0,-1] = [0,1,0]
        let s_dot = cable.direction_derivative();
        assert_relative_eq!(s_dot, Vector3::new(0.0, 1.0, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_cable_integration_no_input() {
        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let mut state = CableState::pointing_down(10.0);
        state.angular_velocity = Vector3::new(1.0, 0.0, 0.0);

        let input = CableInput::default();
        let dt = 0.01;

        let new_state = dynamics.integrate(&state, &input, dt);

        // Direction should have rotated slightly
        assert_relative_eq!(new_state.direction.norm(), 1.0, epsilon = 1e-10);
        // Angular velocity should be unchanged
        assert_relative_eq!(new_state.angular_velocity, state.angular_velocity, epsilon = 1e-10);
        // Tension should be unchanged
        assert_relative_eq!(new_state.tension, state.tension, epsilon = 1e-10);
    }

    #[test]
    fn test_cable_integration_with_jerk() {
        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let state = CableState::pointing_down(10.0);
        let input = CableInput {
            angular_jerk: Vector3::new(1.0, 0.0, 0.0),
            tension_acceleration: 0.0,
        };
        let dt = 0.1;

        let new_state = dynamics.integrate(&state, &input, dt);

        // Angular acceleration should have increased
        assert_relative_eq!(new_state.angular_acceleration.x, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_quadrotor_position() {
        let params = CableParams::new(2.0);
        let dynamics = CableDynamics::new(params);

        let attachment = Vector3::new(0.0, 0.0, 0.0);
        let direction = Vector3::new(0.0, 0.0, -1.0); // Cable pointing down

        let quad_pos = dynamics.quadrotor_position(&attachment, &direction);

        // Quadrotor should be above the attachment point
        assert_relative_eq!(quad_pos, Vector3::new(0.0, 0.0, 2.0), epsilon = 1e-10);
    }

    #[test]
    fn test_thrust_direction() {
        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let cable_direction = Vector3::new(0.0, 0.0, -1.0);
        let thrust_dir = dynamics.thrust_direction(&cable_direction);

        // Thrust should point up (opposite to cable direction)
        assert_relative_eq!(thrust_dir, Vector3::new(0.0, 0.0, 1.0), epsilon = 1e-10);
    }

    #[test]
    fn test_tension_clamping() {
        let mut params = CableParams::new(1.0);
        params.min_tension = 1.0;
        params.max_tension = 50.0;
        let dynamics = CableDynamics::new(params);

        let mut state = CableState::pointing_down(10.0);
        state.tension_rate = -100.0; // Rapidly decreasing tension

        let input = CableInput::default();
        let dt = 0.1;

        let new_state = dynamics.integrate(&state, &input, dt);

        // Tension should be clamped to minimum
        assert_relative_eq!(new_state.tension, 1.0, epsilon = 1e-10);
    }
}
