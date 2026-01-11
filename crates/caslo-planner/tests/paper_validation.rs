//! Paper Validation Tests for Planner
//!
//! Tests to verify OCP implementation matches the paper:
//! "Agile and cooperative aerial manipulation of a cable-suspended load"
//! (Sun et al., Science Robotics, 2025)

use approx::assert_relative_eq;
use nalgebra::{Vector3, UnitQuaternion};

use caslo_planner::ocp::{OcpState, OcpControl, CableState, CableControl};

/// Paper Eq. 1: OCP state dimension verification (3rd-order cable model)
mod ocp_state_dimension_tests {
    use super::*;

    #[test]
    fn test_ocp_state_dimension_eq1() {
        // Paper Eq. 1 with 3rd-order cable model:
        // x = [p, v, q, ω, s₁..sₙ, r₁..rₙ, ṙ₁..ṙₙ, r̈₁..r̈ₙ, t₁..tₙ, ṫ₁..ṫₙ]
        // Dimension = 13 + 14n

        // 3 quadrotors
        assert_eq!(OcpState::dimension(3), 13 + 14 * 3);  // 55
        // 4 quadrotors
        assert_eq!(OcpState::dimension(4), 13 + 14 * 4);  // 69
        // 6 quadrotors
        assert_eq!(OcpState::dimension(6), 13 + 14 * 6);  // 97
    }

    #[test]
    fn test_ocp_control_dimension_eq3() {
        // Paper Eq. 3 with 3rd-order cable model: u = [γ₁, λ₁, ..., γₙ, λₙ]
        // γᵢ = r⃛ᵢ (angular SNAP - 4th derivative, 3 DOF)
        // λᵢ = ẗᵢ (tension acceleration, 1 DOF)
        // Dimension = 4n

        assert_eq!(OcpControl::dimension(3), 4 * 3);  // 12
        assert_eq!(OcpControl::dimension(4), 4 * 4);  // 16
    }

    #[test]
    fn test_cable_state_has_angular_jerk() {
        // 3rd-order cable model: angular_jerk (r̈ᵢ) is now in STATE
        // State has: s, r, ṙ, r̈, t, ṫ (14 DOF per cable)

        let cable = CableState {
            direction: Vector3::new(0.0, 0.0, -1.0),
            angular_velocity: Vector3::zeros(),
            angular_acceleration: Vector3::zeros(),
            angular_jerk: Vector3::zeros(),  // r̈ᵢ now in state!
            tension: 10.0,
            tension_rate: 0.0,
        };

        // Verify CableState has these 6 fields (14 DOF total)
        let _ = cable.direction;           // s: 3 DOF
        let _ = cable.angular_velocity;    // r: 3 DOF
        let _ = cable.angular_acceleration; // ṙ: 3 DOF
        let _ = cable.angular_jerk;         // r̈: 3 DOF (NEW for 3rd-order)
        let _ = cable.tension;             // t: 1 DOF
        let _ = cable.tension_rate;        // ṫ: 1 DOF
    }

    #[test]
    fn test_cable_control_has_angular_snap() {
        // 3rd-order cable model: control is now γᵢ = r⃛ᵢ (angular SNAP)
        let control = CableControl {
            angular_snap: Vector3::new(1.0, 2.0, 3.0),  // γ = r⃛: 3 DOF (4th derivative)
            tension_acceleration: 5.0,                   // λ: 1 DOF
        };

        assert_eq!(control.angular_snap.len(), 3);
        let _ = control.tension_acceleration;
    }
}

/// State vector layout tests
mod state_vector_layout_tests {
    use super::*;

    #[test]
    fn test_state_vector_layout_matches_eq1() {
        // Paper Eq. 1 layout:
        // [p(3), v(3), q(4), ω(3), s₁(3), r₁(3), ṙ₁(3), t₁(1), ṫ₁(1), ...]

        let mut state = OcpState::new(3);
        state.load_position = Vector3::new(1.0, 2.0, 3.0);
        state.load_velocity = Vector3::new(0.1, 0.2, 0.3);
        state.load_orientation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        state.load_angular_velocity = Vector3::new(0.01, 0.02, 0.03);

        state.cables[0].direction = Vector3::new(0.0, 0.0, -1.0);
        state.cables[0].angular_velocity = Vector3::new(0.5, 0.0, 0.0);
        state.cables[0].angular_acceleration = Vector3::new(0.1, 0.0, 0.0);
        state.cables[0].tension = 10.0;
        state.cables[0].tension_rate = 1.0;

        let v = state.to_vector();
        assert_eq!(v.len(), 55);  // 13 + 14*3 (3rd-order cable model)

        // Check positions (indices 0-2)
        assert_relative_eq!(v[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(v[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(v[2], 3.0, epsilon = 1e-10);

        // Check velocities (indices 3-5)
        assert_relative_eq!(v[3], 0.1, epsilon = 1e-10);
        assert_relative_eq!(v[4], 0.2, epsilon = 1e-10);
        assert_relative_eq!(v[5], 0.3, epsilon = 1e-10);

        // Quaternion (indices 6-9): w, x, y, z
        // Index 6 is w (scalar part)
        let q = state.load_orientation;
        assert_relative_eq!(v[6], q.w, epsilon = 1e-10);
        assert_relative_eq!(v[7], q.i, epsilon = 1e-10);
        assert_relative_eq!(v[8], q.j, epsilon = 1e-10);
        assert_relative_eq!(v[9], q.k, epsilon = 1e-10);

        // Angular velocity (indices 10-12)
        assert_relative_eq!(v[10], 0.01, epsilon = 1e-10);
        assert_relative_eq!(v[11], 0.02, epsilon = 1e-10);
        assert_relative_eq!(v[12], 0.03, epsilon = 1e-10);

        // Cable 0 direction s₀ (indices 13-15)
        assert_relative_eq!(v[13], 0.0, epsilon = 1e-10);
        assert_relative_eq!(v[14], 0.0, epsilon = 1e-10);
        assert_relative_eq!(v[15], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_acados_layout_all_s_then_all_r() {
        // ACADOS layout groups by type (3rd-order cable model):
        // [p, v, q, ω, s_all(3*n), r_all(3*n), ṙ_all(3*n), r̈_all(3*n), t_all(n), ṫ_all(n)]
        // Total: 13 + 14*n

        let mut state = OcpState::new(3);

        // Set distinct values for each cable to verify layout
        for i in 0..3 {
            state.cables[i].direction = Vector3::new(i as f64 + 0.1, i as f64 + 0.2, -(i as f64 + 0.3));
            state.cables[i].angular_velocity = Vector3::new(i as f64 * 0.1, 0.0, 0.0);
            state.cables[i].angular_acceleration = Vector3::new(0.0, i as f64 * 0.1, 0.0);
            state.cables[i].angular_jerk = Vector3::new(0.0, 0.0, i as f64 * 0.01);
            state.cables[i].tension = 10.0 + i as f64;
            state.cables[i].tension_rate = 1.0 + i as f64 * 0.1;
        }

        let v = state.to_vector();
        assert_eq!(v.len(), 55);  // 13 + 14*3

        // s_all starts at index 13
        // s₀ = [0.1, 0.2, -0.3] at indices 13, 14, 15
        assert_relative_eq!(v[13], 0.1, epsilon = 1e-10);
        // s₁ = [1.1, 1.2, -1.3] at indices 16, 17, 18
        assert_relative_eq!(v[16], 1.1, epsilon = 1e-10);
        // s₂ = [2.1, 2.2, -2.3] at indices 19, 20, 21
        assert_relative_eq!(v[19], 2.1, epsilon = 1e-10);

        // r_all starts at index 13 + 3*3 = 22
        // r₀ = [0.0, 0.0, 0.0] at indices 22, 23, 24
        assert_relative_eq!(v[22], 0.0, epsilon = 1e-10);
        // r₁ = [0.1, 0.0, 0.0] at indices 25, 26, 27
        assert_relative_eq!(v[25], 0.1, epsilon = 1e-10);
        // r₂ = [0.2, 0.0, 0.0] at indices 28, 29, 30
        assert_relative_eq!(v[28], 0.2, epsilon = 1e-10);

        // ṙ_all starts at index 13 + 6*3 = 31
        // ṙ₀ = [0.0, 0.0, 0.0] at indices 31, 32, 33
        assert_relative_eq!(v[32], 0.0, epsilon = 1e-10);  // y component
        // ṙ₁ = [0.0, 0.1, 0.0] at indices 34, 35, 36
        assert_relative_eq!(v[35], 0.1, epsilon = 1e-10);  // y component

        // r̈_all (angular_jerk) starts at index 13 + 9*3 = 40
        // r̈₀ = [0.0, 0.0, 0.0] at indices 40, 41, 42
        assert_relative_eq!(v[42], 0.0, epsilon = 1e-10);  // z component
        // r̈₁ = [0.0, 0.0, 0.01] at indices 43, 44, 45
        assert_relative_eq!(v[45], 0.01, epsilon = 1e-10);  // z component
        // r̈₂ = [0.0, 0.0, 0.02] at indices 46, 47, 48
        assert_relative_eq!(v[48], 0.02, epsilon = 1e-10);  // z component

        // t_all starts at index 13 + 12*3 = 49
        assert_relative_eq!(v[49], 10.0, epsilon = 1e-10);  // t₀
        assert_relative_eq!(v[50], 11.0, epsilon = 1e-10);  // t₁
        assert_relative_eq!(v[51], 12.0, epsilon = 1e-10);  // t₂

        // ṫ_all starts at index 13 + 12*3 + 3 = 52
        assert_relative_eq!(v[52], 1.0, epsilon = 1e-10);   // ṫ₀
        assert_relative_eq!(v[53], 1.1, epsilon = 1e-10);   // ṫ₁
        assert_relative_eq!(v[54], 1.2, epsilon = 1e-10);   // ṫ₂
    }

    #[test]
    fn test_state_vector_roundtrip() {
        let mut state = OcpState::new(3);
        state.load_position = Vector3::new(1.5, 2.5, 3.5);
        state.load_velocity = Vector3::new(0.1, 0.2, 0.3);
        state.load_orientation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        state.load_angular_velocity = Vector3::new(0.01, 0.02, 0.03);

        for i in 0..3 {
            state.cables[i].direction = Vector3::new(0.1 * (i + 1) as f64, 0.0, -0.9);
            state.cables[i].angular_velocity = Vector3::new(0.5, 0.3, 0.1);
            state.cables[i].angular_acceleration = Vector3::new(0.05, 0.03, 0.01);
            state.cables[i].tension = 5.0 + i as f64;
            state.cables[i].tension_rate = 0.5 + i as f64 * 0.1;
        }

        let v = state.to_vector();
        let recovered = OcpState::from_vector(&v, 3).unwrap();

        assert_relative_eq!(recovered.load_position, state.load_position, epsilon = 1e-10);
        assert_relative_eq!(recovered.load_velocity, state.load_velocity, epsilon = 1e-10);

        for i in 0..3 {
            assert_relative_eq!(recovered.cables[i].direction, state.cables[i].direction, epsilon = 1e-10);
            assert_relative_eq!(recovered.cables[i].angular_velocity, state.cables[i].angular_velocity, epsilon = 1e-10);
            assert_relative_eq!(recovered.cables[i].angular_acceleration, state.cables[i].angular_acceleration, epsilon = 1e-10);
            assert_relative_eq!(recovered.cables[i].tension, state.cables[i].tension, epsilon = 1e-10);
            assert_relative_eq!(recovered.cables[i].tension_rate, state.cables[i].tension_rate, epsilon = 1e-10);
        }
    }
}

/// Control vector layout tests
mod control_vector_layout_tests {
    use super::*;

    #[test]
    fn test_control_vector_layout() {
        // Control layout: [γ_all(3*n), λ_all(n)]
        let mut control = OcpControl::new(3);

        control.cables[0].angular_snap = Vector3::new(1.0, 2.0, 3.0);
        control.cables[1].angular_snap = Vector3::new(4.0, 5.0, 6.0);
        control.cables[2].angular_snap = Vector3::new(7.0, 8.0, 9.0);

        control.cables[0].tension_acceleration = 10.0;
        control.cables[1].tension_acceleration = 20.0;
        control.cables[2].tension_acceleration = 30.0;

        let v = control.to_vector();
        assert_eq!(v.len(), 12);  // 4 * 3

        // γ₀ at indices 0, 1, 2
        assert_relative_eq!(v[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(v[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(v[2], 3.0, epsilon = 1e-10);

        // γ₁ at indices 3, 4, 5
        assert_relative_eq!(v[3], 4.0, epsilon = 1e-10);

        // γ₂ at indices 6, 7, 8
        assert_relative_eq!(v[6], 7.0, epsilon = 1e-10);

        // λ_all at indices 9, 10, 11
        assert_relative_eq!(v[9], 10.0, epsilon = 1e-10);
        assert_relative_eq!(v[10], 20.0, epsilon = 1e-10);
        assert_relative_eq!(v[11], 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_control_vector_roundtrip() {
        let mut control = OcpControl::new(3);

        for i in 0..3 {
            control.cables[i].angular_snap = Vector3::new(
                (i + 1) as f64,
                (i + 2) as f64,
                (i + 3) as f64,
            );
            control.cables[i].tension_acceleration = (i + 10) as f64;
        }

        let v = control.to_vector();
        let recovered = OcpControl::from_vector(&v, 3).unwrap();

        for i in 0..3 {
            assert_relative_eq!(
                recovered.cables[i].angular_snap,
                control.cables[i].angular_snap,
                epsilon = 1e-10
            );
            assert_relative_eq!(
                recovered.cables[i].tension_acceleration,
                control.cables[i].tension_acceleration,
                epsilon = 1e-10
            );
        }
    }
}

/// Default initialization tests
mod initialization_tests {
    use super::*;

    #[test]
    fn test_default_cable_direction_z_up() {
        // Z-UP coordinate frame: cables point DOWN (-Z)
        let state = OcpState::new(3);

        for cable in &state.cables {
            // Direction should be [0, 0, -1] (pointing down)
            assert_relative_eq!(cable.direction.x, 0.0, epsilon = 1e-10);
            assert_relative_eq!(cable.direction.y, 0.0, epsilon = 1e-10);
            assert_relative_eq!(cable.direction.z, -1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_default_velocities_zero() {
        let state = OcpState::new(3);

        // Load velocities should be zero
        assert_relative_eq!(state.load_velocity.norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(state.load_angular_velocity.norm(), 0.0, epsilon = 1e-10);

        // Cable velocities should be zero
        for cable in &state.cables {
            assert_relative_eq!(cable.angular_velocity.norm(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(cable.angular_acceleration.norm(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(cable.tension_rate, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_default_tension_positive() {
        let state = OcpState::new(3);

        for cable in &state.cables {
            assert!(cable.tension > 0.0, "Default tension should be positive");
        }
    }

    #[test]
    fn test_default_orientation_identity() {
        let state = OcpState::new(3);

        // Identity quaternion: w=1, x=y=z=0
        let q = state.load_orientation;
        assert_relative_eq!(q.w, 1.0, epsilon = 1e-10);
        assert_relative_eq!(q.i, 0.0, epsilon = 1e-10);
        assert_relative_eq!(q.j, 0.0, epsilon = 1e-10);
        assert_relative_eq!(q.k, 0.0, epsilon = 1e-10);
    }
}

/// Consistency tests between caslo-core and caslo-planner
mod cross_crate_consistency_tests {
    use super::*;
    use caslo_planner::integration::{core_to_ocp, ocp_to_core};
    use caslo_core::dynamics::{LoadState, CableState as CoreCableState, MultiCableState};

    #[test]
    fn test_core_to_ocp_preserves_dimensions() {
        let load = LoadState {
            position: Vector3::new(1.0, 2.0, 3.0),
            velocity: Vector3::new(0.1, 0.2, 0.3),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::new(0.01, 0.02, 0.03),
        };

        let cables = MultiCableState::new(vec![
            CoreCableState::pointing_down(10.0),
            CoreCableState::pointing_down(10.0),
            CoreCableState::pointing_down(10.0),
        ]);

        let ocp_state = core_to_ocp(&load, &cables);

        // Should have 3 cables
        assert_eq!(ocp_state.cables.len(), 3);

        // Vector dimension should be 13 + 14*3 = 55 (3rd-order cable model)
        let v = ocp_state.to_vector();
        assert_eq!(v.len(), 55);
    }

    #[test]
    fn test_roundtrip_preserves_state() {
        let load = LoadState {
            position: Vector3::new(1.0, 2.0, 3.0),
            velocity: Vector3::new(0.1, 0.2, 0.3),
            orientation: UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3),
            angular_velocity: Vector3::new(0.01, 0.02, 0.03),
        };

        let mut cable0 = CoreCableState::pointing_down(10.0);
        cable0.angular_velocity = Vector3::new(0.5, 0.3, 0.1);
        cable0.angular_acceleration = Vector3::new(0.05, 0.03, 0.01);
        cable0.tension_rate = 0.5;

        let cables = MultiCableState::new(vec![
            cable0.clone(),
            CoreCableState::pointing_down(8.0),
            CoreCableState::pointing_down(12.0),
        ]);

        // Core -> OCP -> Core
        let ocp_state = core_to_ocp(&load, &cables);
        let (load_back, cables_back) = ocp_to_core(&ocp_state);

        // Load state should be preserved
        assert_relative_eq!(load_back.position, load.position, epsilon = 1e-10);
        assert_relative_eq!(load_back.velocity, load.velocity, epsilon = 1e-10);

        // Cable states should be preserved
        assert_relative_eq!(cables_back.cables[0].tension, 10.0, epsilon = 1e-10);
        assert_relative_eq!(cables_back.cables[0].angular_velocity, cable0.angular_velocity, epsilon = 1e-10);
        assert_relative_eq!(cables_back.cables[0].angular_acceleration, cable0.angular_acceleration, epsilon = 1e-10);
        assert_relative_eq!(cables_back.cables[0].tension_rate, cable0.tension_rate, epsilon = 1e-10);
    }
}
