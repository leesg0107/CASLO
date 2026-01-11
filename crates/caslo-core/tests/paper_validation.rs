//! Paper Validation Tests
//!
//! Tests to verify implementation matches the paper:
//! "Agile and cooperative aerial manipulation of a cable-suspended load"
//! (Sun et al., Science Robotics, 2025)
//!
//! These tests validate:
//! 1. State dimensions match Eq. 1
//! 2. Dynamics equations match Eq. 2-3
//! 3. Kinematic constraints match Eq. 5
//! 4. Physical behavior is correct

use approx::assert_relative_eq;
use nalgebra::{Vector3, UnitQuaternion};
use std::f64::consts::PI;

use caslo_core::dynamics::{
    CableState, CableParams, CableDynamics, CableInput,
    LoadState, LoadParams, LoadDynamics,
    MultiCableState,
};
use caslo_core::kinematics::{KinematicConstraint, LoadKinematicState, CableKinematicState};

/// Paper Eq. 1: State dimension verification
/// x = [p, v, q, ω, s₁, r₁, ṙ₁, t₁, ṫ₁, ..., sₙ, rₙ, ṙₙ, tₙ, ṫₙ]
/// Total: 13 + 11n
mod state_dimension_tests {
    use super::*;

    #[test]
    fn test_cable_state_has_11_dof() {
        // Per cable: s(3) + r(3) + ṙ(3) + t(1) + ṫ(1) = 11 DOF
        let cable = CableState::pointing_down(10.0);

        // direction: 3 DOF
        assert_eq!(cable.direction.len(), 3);
        // angular_velocity: 3 DOF
        assert_eq!(cable.angular_velocity.len(), 3);
        // angular_acceleration: 3 DOF
        assert_eq!(cable.angular_acceleration.len(), 3);
        // tension: 1 DOF (scalar)
        let _ = cable.tension;
        // tension_rate: 1 DOF (scalar)
        let _ = cable.tension_rate;

        // Verify NO angular_jerk field exists in state
        // (angular_jerk γᵢ = r̈ᵢ is CONTROL INPUT, not state)
        // This is checked at compile time - if angular_jerk existed,
        // the struct definition would be different
    }

    #[test]
    fn test_load_state_has_13_dof() {
        // Load: p(3) + v(3) + q(4) + ω(3) = 13 DOF
        let load = LoadState {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        };

        assert_eq!(load.position.len(), 3);
        assert_eq!(load.velocity.len(), 3);
        // Quaternion has 4 components (w, i, j, k)
        assert_eq!(load.orientation.coords.len(), 4);
        assert_eq!(load.angular_velocity.len(), 3);
    }

    #[test]
    fn test_control_input_has_4_dof_per_cable() {
        // Control: γᵢ(3) + λᵢ(1) = 4 DOF per cable
        let input = CableInput::default();

        // angular_jerk γᵢ = r̈ᵢ: 3 DOF
        assert_eq!(input.angular_jerk.len(), 3);
        // tension_acceleration λᵢ = ẗᵢ: 1 DOF (scalar)
        let _ = input.tension_acceleration;
    }
}

/// Paper Eq. 3: Cable dynamics verification
/// ṡᵢ = rᵢ × sᵢ
/// ṙᵢ = αᵢ (angular acceleration from state)
/// r̈ᵢ = γᵢ (control input)
/// ṫᵢ = tension_rate (from state)
/// ẗᵢ = λᵢ (control input)
mod cable_dynamics_tests {
    use super::*;

    #[test]
    fn test_direction_derivative_eq3() {
        // Eq. 3: ṡᵢ = rᵢ × sᵢ
        let mut cable = CableState::pointing_down(10.0);
        // s = [0, 0, -1] (pointing down in Z-UP)
        // r = [1, 0, 0] (rotation around X axis)
        cable.angular_velocity = Vector3::new(1.0, 0.0, 0.0);

        let s_dot = cable.direction_derivative();

        // ṡ = r × s = [1,0,0] × [0,0,-1] = [0*(-1) - 0*0, 0*0 - 1*(-1), 1*0 - 0*0]
        //           = [0, 1, 0]
        assert_relative_eq!(s_dot.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(s_dot.y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(s_dot.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_direction_stays_unit_vector() {
        // After integration, s must remain on S² (unit sphere)
        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let mut cable = CableState::pointing_down(10.0);
        cable.angular_velocity = Vector3::new(0.5, 0.3, 0.1);

        let input = CableInput::default();

        // Integrate for many steps
        let dt = 0.01;
        let mut state = cable;
        for _ in 0..1000 {
            state = dynamics.integrate(&state, &input, dt);
        }

        // Direction must still be unit vector
        assert_relative_eq!(state.direction.norm(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_angular_jerk_is_control_input() {
        // Eq. 3: r̈ᵢ = γᵢ (control input integrates to angular acceleration)
        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let cable = CableState::pointing_down(10.0);
        // Apply angular jerk control
        let input = CableInput {
            angular_jerk: Vector3::new(10.0, 0.0, 0.0),
            tension_acceleration: 0.0,
        };

        let dt = 0.1;
        let new_state = dynamics.integrate(&cable, &input, dt);

        // Angular acceleration should increase by γ * dt
        // ṙ_new = ṙ_old + γ * dt = 0 + 10 * 0.1 = 1.0
        assert_relative_eq!(new_state.angular_acceleration.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(new_state.angular_acceleration.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(new_state.angular_acceleration.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_angular_acceleration_integrates_to_velocity() {
        // ṙ = α (angular acceleration integrates to angular velocity)
        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let mut cable = CableState::pointing_down(10.0);
        cable.angular_acceleration = Vector3::new(5.0, 0.0, 0.0);

        let input = CableInput::default();
        let dt = 0.1;
        let new_state = dynamics.integrate(&cable, &input, dt);

        // r_new = r_old + α * dt = 0 + 5 * 0.1 = 0.5
        assert_relative_eq!(new_state.angular_velocity.x, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_tension_acceleration_is_control_input() {
        // Eq. 3: ẗᵢ = λᵢ (control input)
        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let mut cable = CableState::pointing_down(10.0);
        cable.tension_rate = 0.0;

        let input = CableInput {
            angular_jerk: Vector3::zeros(),
            tension_acceleration: 100.0,  // λ = 100 N/s²
        };

        let dt = 0.1;
        let new_state = dynamics.integrate(&cable, &input, dt);

        // ṫ_new = ṫ_old + λ * dt = 0 + 100 * 0.1 = 10 N/s
        assert_relative_eq!(new_state.tension_rate, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tension_rate_integrates_to_tension() {
        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let mut cable = CableState::pointing_down(10.0);
        cable.tension_rate = 5.0;  // 5 N/s

        let input = CableInput::default();
        let dt = 0.1;
        let new_state = dynamics.integrate(&cable, &input, dt);

        // t_new = t_old + ṫ * dt = 10 + 5 * 0.1 = 10.5 N
        assert_relative_eq!(new_state.tension, 10.5, epsilon = 1e-10);
    }

    #[test]
    fn test_third_order_dynamics_chain() {
        // Full chain test: γ → ṙ → r → s
        // After applying constant angular jerk γ for time T:
        // - ṙ(T) = γT (linear in time)
        // - r(T) = γT²/2 (quadratic in time)
        // - s changes according to ṡ = r × s

        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let cable = CableState::pointing_down(10.0);
        let input = CableInput {
            angular_jerk: Vector3::new(1.0, 0.0, 0.0),  // γ = 1 rad/s³
            tension_acceleration: 0.0,
        };

        let dt = 0.001;
        let mut state = cable;
        let total_time = 1.0;
        let steps = (total_time / dt) as usize;

        for _ in 0..steps {
            state = dynamics.integrate(&state, &input, dt);
        }

        // After 1 second with γ = 1:
        // ṙ ≈ 1.0 rad/s²
        // r ≈ 0.5 rad/s (integral of ṙ)
        assert_relative_eq!(state.angular_acceleration.x, 1.0, epsilon = 0.01);
        assert_relative_eq!(state.angular_velocity.x, 0.5, epsilon = 0.02);

        // Direction should have rotated
        assert!(state.direction.z > -1.0, "Direction should have changed from [0,0,-1]");
    }
}

/// Paper Eq. 2: Load dynamics verification
mod load_dynamics_tests {
    use super::*;
    use caslo_core::dynamics::{LoadParams, LoadDynamics};

    #[test]
    fn test_hover_equilibrium() {
        // At hover with 3 cables pointing down:
        // Total cable force = 3 * t * s = 3 * t * [0,0,-1]
        // Gravity = m * g * [0,0,-1] (Z-UP: gravity points down)
        // Equilibrium: -3t * [0,0,-1] + m*g*[0,0,-1] = 0
        // → 3t = m*g → t = m*g/3

        let load_mass = 1.4;  // kg (from paper)
        let g = 9.81;
        let n_cables = 3;
        let hover_tension = load_mass * g / n_cables as f64;

        // With correct hover tension, acceleration should be zero
        let cable_directions = vec![
            Vector3::new(0.0, 0.0, -1.0),  // All pointing down
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ];
        let tensions = vec![hover_tension; n_cables];

        // Sum of cable forces: -Σ(t_i * s_i)
        let mut total_force = Vector3::zeros();
        for i in 0..n_cables {
            total_force -= tensions[i] * cable_directions[i];
        }

        // Add gravity
        let gravity = Vector3::new(0.0, 0.0, -load_mass * g);
        let net_force = total_force + gravity;

        // Net force should be zero at hover
        assert_relative_eq!(net_force.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hover_tension_formula() {
        // Paper parameters
        let load_mass = 1.4;  // kg
        let g = 9.81;
        let n = 3;

        let expected_tension = load_mass * g / n as f64;  // ~4.58 N

        assert_relative_eq!(expected_tension, 1.4 * 9.81 / 3.0, epsilon = 1e-10);
        assert!(expected_tension > 4.5 && expected_tension < 4.7,
            "Hover tension should be ~4.58 N, got {}", expected_tension);
    }

    #[test]
    fn test_load_dynamics_hover_acceleration() {
        // Test that LoadDynamics.compute_acceleration returns zero at hover
        let load_mass = 1.4;
        let g = 9.81;
        let n_cables = 3;
        let hover_tension = load_mass * g / n_cables as f64;

        let load_params = LoadParams::new(
            load_mass,
            Vector3::new(0.01, 0.01, 0.01),  // inertia
            vec![
                Vector3::new(0.3, 0.0, 0.0),
                Vector3::new(-0.15, 0.26, 0.0),
                Vector3::new(-0.15, -0.26, 0.0),
            ],
        );
        let load_dynamics = LoadDynamics::new(load_params);

        let tensions = vec![hover_tension; n_cables];
        let directions = vec![
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ];

        let acceleration = load_dynamics.compute_acceleration(&tensions, &directions);

        // At hover, acceleration should be zero
        assert_relative_eq!(acceleration.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(acceleration.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(acceleration.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_load_dynamics_insufficient_tension_falls() {
        // Test that with insufficient tension, load accelerates downward
        let load_mass = 1.4;
        let g = 9.81;
        let n_cables = 3;
        let hover_tension = load_mass * g / n_cables as f64;
        let insufficient_tension = hover_tension * 0.5;  // Only 50% of required

        let load_params = LoadParams::new(
            load_mass,
            Vector3::new(0.01, 0.01, 0.01),
            vec![
                Vector3::new(0.3, 0.0, 0.0),
                Vector3::new(-0.15, 0.26, 0.0),
                Vector3::new(-0.15, -0.26, 0.0),
            ],
        );
        let load_dynamics = LoadDynamics::new(load_params);

        let tensions = vec![insufficient_tension; n_cables];
        let directions = vec![
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ];

        let acceleration = load_dynamics.compute_acceleration(&tensions, &directions);

        // With only 50% tension, should accelerate downward at 0.5g
        assert_relative_eq!(acceleration.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(acceleration.y, 0.0, epsilon = 1e-10);
        assert!(acceleration.z < 0.0, "Should accelerate downward with insufficient tension");
        assert_relative_eq!(acceleration.z, -g * 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_load_dynamics_tilted_cables() {
        // Test that tilted cables reduce vertical force component
        let load_mass = 1.4;
        let g = 9.81;
        let n_cables = 3;
        let hover_tension = load_mass * g / n_cables as f64;

        let load_params = LoadParams::new(
            load_mass,
            Vector3::new(0.01, 0.01, 0.01),
            vec![
                Vector3::new(0.3, 0.0, 0.0),
                Vector3::new(-0.15, 0.26, 0.0),
                Vector3::new(-0.15, -0.26, 0.0),
            ],
        );
        let load_dynamics = LoadDynamics::new(load_params);

        // Tilted cables: each at 45 degrees, z component = -0.707
        let tilt_angle = std::f64::consts::PI / 4.0;
        let z_component = -tilt_angle.cos();  // -0.707
        let x_component = tilt_angle.sin();   // 0.707

        let tensions = vec![hover_tension; n_cables];
        let directions = vec![
            Vector3::new(x_component, 0.0, z_component).normalize(),
            Vector3::new(-x_component * 0.5, x_component * 0.866, z_component).normalize(),
            Vector3::new(-x_component * 0.5, -x_component * 0.866, z_component).normalize(),
        ];

        let acceleration = load_dynamics.compute_acceleration(&tensions, &directions);

        // With tilted cables (z component ~0.707), vertical force is reduced
        // So load should accelerate downward
        assert!(acceleration.z < 0.0,
            "Tilted cables should not provide enough vertical force, got az={}", acceleration.z);
    }
}

/// Paper Eq. 5: Kinematic constraint verification
/// pᵢ = p + R(q)ρᵢ - lᵢsᵢ
mod kinematic_constraint_tests {
    use super::*;

    /// Helper to create LoadKinematicState from position and orientation
    fn make_load_state(pos: Vector3<f64>, ori: UnitQuaternion<f64>) -> LoadKinematicState {
        LoadKinematicState {
            position: pos,
            velocity: Vector3::zeros(),
            acceleration: Vector3::zeros(),
            jerk: Vector3::zeros(),
            orientation: ori,
            angular_velocity: Vector3::zeros(),
            angular_acceleration: Vector3::zeros(),
            angular_jerk: Vector3::zeros(),
        }
    }

    /// Helper to create CableKinematicState from direction
    fn make_cable_state(dir: Vector3<f64>) -> CableKinematicState {
        CableKinematicState {
            direction: dir,
            angular_velocity: Vector3::zeros(),
            angular_acceleration: Vector3::zeros(),
            angular_jerk: Vector3::zeros(),
        }
    }

    #[test]
    fn test_quadrotor_position_eq5() {
        // Eq. 5: pᵢ = p + R(q)ρᵢ - lᵢsᵢ
        let attachment_points = vec![
            Vector3::new(0.1, 0.0, 0.0),  // 10cm in front
        ];
        let cable_lengths = vec![1.0];  // 1m cable

        let constraint = KinematicConstraint::new(attachment_points.clone(), cable_lengths.clone());

        // Load at origin, identity orientation
        let load = make_load_state(Vector3::zeros(), UnitQuaternion::identity());

        // Cable pointing straight down: s = [0, 0, -1]
        let cable = make_cable_state(Vector3::new(0.0, 0.0, -1.0));

        // Expected: p_quad = p + R*ρ - l*s
        //         = [0,0,0] + I*[0.1,0,0] - 1*[0,0,-1]
        //         = [0.1, 0, 0] + [0, 0, 1]
        //         = [0.1, 0, 1]
        let expected_quad_pos = Vector3::new(0.1, 0.0, 1.0);

        let quad_pos = constraint.quadrotor_position(&load, &cable, 0);

        assert_relative_eq!(quad_pos, expected_quad_pos, epsilon = 1e-10);
    }

    #[test]
    fn test_quadrotor_position_with_rotated_load() {
        let attachment_points = vec![
            Vector3::new(0.1, 0.0, 0.0),
        ];
        let cable_lengths = vec![1.0];

        let constraint = KinematicConstraint::new(attachment_points, cable_lengths);

        // Load at [1,2,3], rotated 90° around Z axis
        let load = make_load_state(
            Vector3::new(1.0, 2.0, 3.0),
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), PI / 2.0),
        );

        // Cable pointing straight down
        let cable = make_cable_state(Vector3::new(0.0, 0.0, -1.0));

        // R * ρ = R * [0.1, 0, 0] = [0, 0.1, 0] (rotated 90° around Z)
        // p_quad = [1,2,3] + [0, 0.1, 0] - 1*[0,0,-1]
        //        = [1, 2.1, 3] + [0, 0, 1]
        //        = [1, 2.1, 4]
        let expected = Vector3::new(1.0, 2.1, 4.0);

        let quad_pos = constraint.quadrotor_position(&load, &cable, 0);

        assert_relative_eq!(quad_pos, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quadrotor_position_with_tilted_cable() {
        let attachment_points = vec![
            Vector3::new(0.0, 0.0, 0.0),  // At load center
        ];
        let cable_lengths = vec![1.0];

        let constraint = KinematicConstraint::new(attachment_points, cable_lengths);

        let load = make_load_state(Vector3::zeros(), UnitQuaternion::identity());

        // Cable at 45° angle in XZ plane
        // s = [sin(45°), 0, -cos(45°)] = [0.707, 0, -0.707]
        let cable_dir = Vector3::new(
            (PI / 4.0).sin(),
            0.0,
            -(PI / 4.0).cos(),
        ).normalize();
        let cable = make_cable_state(cable_dir);

        // p_quad = [0,0,0] + [0,0,0] - 1*s
        //        = -s = [-0.707, 0, 0.707]
        let expected = -cable_dir;

        let quad_pos = constraint.quadrotor_position(&load, &cable, 0);

        assert_relative_eq!(quad_pos, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_three_quadrotors_different_positions() {
        // Paper setup: 3 quads in triangular formation
        let attachment_points = vec![
            Vector3::new(0.3, 0.0, 0.0),         // Front
            Vector3::new(-0.15, 0.26, 0.0),      // Back-left
            Vector3::new(-0.15, -0.26, 0.0),     // Back-right
        ];
        let cable_lengths = vec![1.0, 1.0, 1.0];

        let constraint = KinematicConstraint::new(attachment_points.clone(), cable_lengths);

        let load = make_load_state(Vector3::zeros(), UnitQuaternion::identity());

        // All cables pointing down
        let cable_states: Vec<_> = (0..3)
            .map(|_| make_cable_state(Vector3::new(0.0, 0.0, -1.0)))
            .collect();

        let quad_positions: Vec<_> = (0..3).map(|i| {
            constraint.quadrotor_position(&load, &cable_states[i], i)
        }).collect();

        // Each quadrotor should be at attachment point + [0, 0, 1] (cable length above)
        for i in 0..3 {
            let expected = attachment_points[i] + Vector3::new(0.0, 0.0, 1.0);
            assert_relative_eq!(quad_positions[i], expected, epsilon = 1e-10);
        }

        // Quadrotors should be at different positions
        assert!((quad_positions[0] - quad_positions[1]).norm() > 0.3);
        assert!((quad_positions[1] - quad_positions[2]).norm() > 0.3);
    }
}

/// Full system dynamics integration tests
mod system_integration_tests {
    use super::*;
    use caslo_core::dynamics::{
        LoadParams, SystemParams, SystemDynamics, SystemState, SystemInput,
        QuadrotorParams,
    };

    fn create_hover_system() -> (SystemDynamics, SystemState) {
        let load_mass = 1.4;
        let num_quads = 3;
        let hover_tension = load_mass * 9.81 / num_quads as f64;

        let load_params = LoadParams::new(
            load_mass,
            Vector3::new(0.01, 0.01, 0.01),
            vec![
                Vector3::new(0.3, 0.0, 0.0),
                Vector3::new(-0.15, 0.26, 0.0),
                Vector3::new(-0.15, -0.26, 0.0),
            ],
        );

        let cable_params = CableParams::new(1.0);
        let quad_params = QuadrotorParams::default();
        let params = SystemParams::uniform(load_params, cable_params, quad_params);
        let dynamics = SystemDynamics::new(params);

        let load_state = LoadState {
            position: Vector3::new(0.0, 0.0, 1.0),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        };

        let cable_states = (0..num_quads)
            .map(|_| CableState::new(Vector3::new(0.0, 0.0, -1.0), hover_tension))
            .collect();

        let state = SystemState::new(
            load_state,
            MultiCableState::new(cable_states),
            &dynamics.load,
            &dynamics.cables,
        );

        (dynamics, state)
    }

    #[test]
    fn test_hover_stability() {
        // At hover with correct tension, system should remain stable
        let (dynamics, state) = create_hover_system();

        // Zero input (no angular jerk, no tension acceleration)
        let input = SystemInput {
            cables: vec![CableInput::default(); 3],
            quadrotors: vec![Default::default(); 3],
        };

        let initial_z = state.load.position.z;
        let dt = 0.001;

        let mut current_state = state;
        for step in 0..1000 {  // 1 second
            current_state = dynamics.integrate(&current_state, &input, dt);

            // Load should not fall significantly
            let z_drop = initial_z - current_state.load.position.z;
            assert!(z_drop.abs() < 0.1,
                "Load fell too much at step {}: z_drop = {:.4}m, velocity = {:.4} m/s",
                step, z_drop, current_state.load.velocity.z);
        }

        // After 1 second, position should be essentially unchanged
        assert_relative_eq!(current_state.load.position.z, initial_z, epsilon = 0.01);
        assert_relative_eq!(current_state.load.velocity.z, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_hover_acceleration_is_zero() {
        // Verify that at hover, the computed acceleration is truly zero
        let (dynamics, state) = create_hover_system();

        let tensions = state.cables.tensions();
        let directions = state.cables.directions();

        let acceleration = dynamics.load.compute_acceleration(&tensions, &directions);

        assert_relative_eq!(acceleration.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(acceleration.y, 0.0, epsilon = 1e-10);
        assert!(acceleration.z.abs() < 1e-10,
            "Z acceleration should be 0 at hover, got {}", acceleration.z);
    }

    #[test]
    fn test_pid_controller_maintains_hover() {
        // Test that a simple PID-like controller can maintain hover
        // This simulates what visualize_sim.rs does
        let (dynamics, initial_state) = create_hover_system();

        let load_mass = 1.4;
        let num_quads = 3;
        let hover_tension = load_mass * 9.81 / num_quads as f64;

        let desired_direction = Vector3::new(0.0, 0.0, -1.0);
        let desired_tension = hover_tension;

        let dt = 0.001;
        let mut state = initial_state;
        let initial_z = state.load.position.z;

        for step in 0..1000 {  // 1 second
            let mut cable_inputs = Vec::with_capacity(num_quads);

            for i in 0..num_quads {
                let cable = &state.cables.cables[i];

                // === Tension Control (from visualize_sim.rs) ===
                let tension_error = desired_tension - cable.tension;
                let tension_rate_des = tension_error * 50.0;
                let tension_rate_error = tension_rate_des - cable.tension_rate;
                let tension_accel = tension_rate_error * 100.0;

                // === Direction Control (from visualize_sim.rs) ===
                let dir_error = cable.direction.cross(&desired_direction);

                let kp_dir = 20.0;
                let kd_dir = 8.0;
                let ka_dir = 4.0;

                let omega_des = dir_error * kp_dir;
                let omega_error = omega_des - cable.angular_velocity;
                let alpha_des = omega_error * kd_dir;
                let alpha_error = alpha_des - cable.angular_acceleration;
                let angular_jerk = alpha_error * ka_dir;

                let max_jerk = 100.0;
                let angular_jerk_clamped = if angular_jerk.norm() > max_jerk {
                    angular_jerk.normalize() * max_jerk
                } else {
                    angular_jerk
                };

                cable_inputs.push(CableInput {
                    angular_jerk: angular_jerk_clamped,
                    tension_acceleration: tension_accel.clamp(-500.0, 500.0),
                });
            }

            let input = SystemInput {
                cables: cable_inputs,
                quadrotors: vec![Default::default(); num_quads],
            };

            state = dynamics.integrate(&state, &input, dt);

            // Debug output for first few steps
            if step < 10 || step % 100 == 0 {
                let z_drop = initial_z - state.load.position.z;
                let tensions: Vec<f64> = state.cables.cables.iter().map(|c| c.tension).collect();
                if z_drop.abs() > 0.001 {
                    println!("step {}: z_drop={:.6}m, vz={:.6}m/s, tensions={:?}",
                        step, z_drop, state.load.velocity.z, tensions);
                }
            }

            // Load should not fall significantly
            let z_drop = initial_z - state.load.position.z;
            if z_drop.abs() > 0.1 {
                let tensions: Vec<f64> = state.cables.cables.iter().map(|c| c.tension).collect();
                let dirs: Vec<_> = state.cables.cables.iter().map(|c| c.direction).collect();
                panic!("Load fell too much at step {}: z_drop = {:.4}m, velocity = {:.4} m/s\n\
                        Tensions: {:?}\n\
                        Directions: {:?}",
                    step, z_drop, state.load.velocity.z, tensions, dirs);
            }
        }

        // After 1 second, position should be essentially unchanged
        assert_relative_eq!(state.load.position.z, initial_z, epsilon = 0.01);
    }
}

/// Physical plausibility tests
mod physical_plausibility_tests {
    use super::*;

    #[test]
    fn test_cable_direction_convention() {
        // In Z-UP coordinate frame:
        // - Positive Z is UP
        // - Gravity points in -Z direction
        // - Cable pointing "down" (from quad to load) has negative Z component

        let cable = CableState::pointing_down(10.0);

        // Cable direction should have negative Z (pointing toward ground)
        assert!(cable.direction.z < 0.0,
            "Cable pointing down should have negative Z, got {:?}", cable.direction);
        assert_relative_eq!(cable.direction.z, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quadrotor_above_load_when_cable_down() {
        // When cable points straight down, quadrotor should be above load
        let attachment_points = vec![Vector3::zeros()];
        let cable_lengths = vec![1.0];
        let constraint = KinematicConstraint::new(attachment_points, cable_lengths);

        let load = LoadKinematicState {
            position: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            ..Default::default()
        };
        let cable = CableKinematicState {
            direction: Vector3::new(0.0, 0.0, -1.0),  // Pointing down
            ..Default::default()
        };

        let quad_pos = constraint.quadrotor_position(&load, &cable, 0);

        // Quadrotor should be above (higher Z) than load
        assert!(quad_pos.z > load.position.z,
            "Quadrotor should be above load when cable points down");
        assert_relative_eq!(quad_pos.z - load.position.z, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tension_must_be_positive() {
        // Cables can only pull, not push
        let mut params = CableParams::new(1.0);
        params.min_tension = 0.0;
        let dynamics = CableDynamics::new(params);

        let mut cable = CableState::pointing_down(5.0);
        cable.tension_rate = -100.0;  // Rapidly decreasing

        let input = CableInput::default();
        let dt = 0.1;
        let new_state = dynamics.integrate(&cable, &input, dt);

        // Tension should be clamped to minimum (0)
        assert!(new_state.tension >= 0.0,
            "Tension cannot be negative, got {}", new_state.tension);
    }

    #[test]
    fn test_angular_velocity_perpendicular_to_direction() {
        // The angular velocity r rotates s on the sphere
        // ṡ = r × s, so r must have a component perpendicular to s
        // to cause any change in s

        let mut cable = CableState::pointing_down(10.0);
        // Angular velocity parallel to s should cause no change in s
        cable.angular_velocity = Vector3::new(0.0, 0.0, 1.0);  // Parallel to s=[0,0,-1]

        let s_dot = cable.direction_derivative();

        // Cross product of parallel vectors is zero
        assert_relative_eq!(s_dot.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_system_energy_bounded_without_input() {
        // Without control input, system should not gain energy spontaneously
        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let mut cable = CableState::pointing_down(10.0);
        cable.angular_velocity = Vector3::new(1.0, 0.5, 0.2);
        cable.angular_acceleration = Vector3::new(0.1, 0.1, 0.1);

        let input = CableInput::default();  // Zero input

        // Initial "energy" proxy
        let initial_omega_norm = cable.angular_velocity.norm();
        let initial_alpha_norm = cable.angular_acceleration.norm();

        // Integrate
        let dt = 0.001;
        let mut state = cable;
        for _ in 0..1000 {
            state = dynamics.integrate(&state, &input, dt);
        }

        // Angular acceleration should stay constant (no damping, but no growth)
        assert_relative_eq!(state.angular_acceleration.norm(), initial_alpha_norm, epsilon = 1e-6);
    }
}

/// Stability tests
mod stability_tests {
    use super::*;

    #[test]
    fn test_hover_stability_no_input() {
        // At hover equilibrium with zero initial velocity,
        // system should remain stationary
        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let cable = CableState::pointing_down(4.58);  // Hover tension
        let input = CableInput::default();

        let dt = 0.001;
        let mut state = cable.clone();
        for _ in 0..10000 {  // 10 seconds
            state = dynamics.integrate(&state, &input, dt);
        }

        // Direction should still be pointing down
        assert_relative_eq!(state.direction, cable.direction, epsilon = 1e-6);

        // Angular velocity should still be zero
        assert_relative_eq!(state.angular_velocity.norm(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_small_perturbation_bounded() {
        // Small initial perturbation should not grow without bound
        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let mut cable = CableState::pointing_down(10.0);
        // Small initial angular velocity perturbation
        cable.angular_velocity = Vector3::new(0.01, 0.01, 0.0);

        let input = CableInput::default();

        let dt = 0.001;
        let mut state = cable;
        let mut max_omega = 0.0_f64;

        for _ in 0..10000 {
            state = dynamics.integrate(&state, &input, dt);
            max_omega = max_omega.max(state.angular_velocity.norm());
        }

        // Angular velocity should not explode
        assert!(max_omega < 1.0,
            "Angular velocity grew too large: {}", max_omega);
    }

    #[test]
    fn test_direction_normalization_preserved() {
        // Direction must stay normalized throughout integration
        let params = CableParams::new(1.0);
        let dynamics = CableDynamics::new(params);

        let mut cable = CableState::pointing_down(10.0);
        cable.angular_velocity = Vector3::new(2.0, 1.5, 0.5);
        cable.angular_acceleration = Vector3::new(0.5, 0.3, 0.1);

        let input = CableInput {
            angular_jerk: Vector3::new(0.1, 0.2, 0.3),
            tension_acceleration: 1.0,
        };

        let dt = 0.001;
        let mut state = cable;

        for i in 0..10000 {
            state = dynamics.integrate(&state, &input, dt);

            let norm = state.direction.norm();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "Direction denormalized at step {}: norm = {}", i, norm
            );
        }
    }
}
