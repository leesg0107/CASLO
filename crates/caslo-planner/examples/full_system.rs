//! Full System Demonstration
//!
//! This example demonstrates the complete CASLO system:
//! - Multi-quadrotor cable-suspended load dynamics
//! - Online kinodynamic motion planner
//! - Tracking reference generation
//!
//! Simulates a 3-quadrotor system moving the load from one position to another.

use nalgebra::{Vector3, UnitQuaternion};

use caslo_core::dynamics::{
    LoadState, LoadParams,
    CableState, CableParams, MultiCableState,
    QuadrotorParams,
    SystemState, SystemParams, SystemDynamics, SystemInput,
};

use caslo_planner::config::PlannerConfig;
use caslo_planner::ocp::SystemParameters;
use caslo_planner::controller::MotionPlanner;
use caslo_planner::integration::{core_to_ocp, reference_for_quadrotor};

fn main() {
    println!("=== CASLO Full System Demonstration ===\n");

    // === System Configuration ===
    let num_quadrotors = 3;

    // Load parameters: 0.3 kg payload with small inertia
    let load_params = LoadParams::new(
        0.3,  // mass [kg]
        Vector3::new(0.001, 0.001, 0.001),  // inertia [kg·m²]
        generate_attachment_points(num_quadrotors, 0.1),  // 10cm radius
    );

    // Cable parameters: 1m cables
    let cable_params = CableParams::new(1.0);

    // Quadrotor parameters: default Crazyflie-like
    let quad_params = QuadrotorParams::default();

    // Create system dynamics
    let system_params = SystemParams::uniform(load_params, cable_params.clone(), quad_params);
    let dynamics = SystemDynamics::new(system_params);

    println!("System created with {} quadrotors", num_quadrotors);
    println!("Cable length: {} m", cable_params.length);

    // === Initial State ===
    // Load at 2m height, cables pointing down
    let initial_load = LoadState {
        position: Vector3::new(0.0, 0.0, 2.0),
        velocity: Vector3::zeros(),
        orientation: UnitQuaternion::identity(),
        angular_velocity: Vector3::zeros(),
    };

    let initial_cables = MultiCableState::new(vec![
        CableState::pointing_down(10.0),  // ~10N tension each (supports load)
        CableState::pointing_down(10.0),
        CableState::pointing_down(10.0),
    ]);

    let mut state = SystemState::new(
        initial_load,
        initial_cables.clone(),
        &dynamics.load,
        &dynamics.cables,
    );

    println!("\nInitial load position: {:?}", state.load.position);
    for (i, quad) in state.quadrotors.iter().enumerate() {
        println!("Quadrotor {} position: {:?}", i, quad.position);
    }

    // === Motion Planner Setup ===
    let planner_config = PlannerConfig::default();
    let system_ocp_params = SystemParameters::new(num_quadrotors);

    // Create motion planner (Note: actual ACADOS solver requires the feature flag)
    let planner_result = MotionPlanner::new(
        num_quadrotors,
        planner_config.clone(),
        system_ocp_params,
    );

    let mut planner = match planner_result {
        Ok(p) => p,
        Err(e) => {
            println!("\nNote: Planner creation failed (expected without ACADOS): {:?}", e);
            println!("\nDemonstrating state conversion and reference generation instead...\n");
            demonstrate_conversions(&state, &initial_cables, &dynamics);
            return;
        }
    };

    // === Set Goal: Move to new position ===
    let goal_position = Vector3::new(1.0, 1.0, 2.5);
    planner.go_to(goal_position, UnitQuaternion::identity());

    println!("\nGoal position: {:?}", goal_position);
    println!("Planner state: {:?}", planner.state());

    // === Simulation Loop ===
    let dt = 0.01;  // 100 Hz simulation
    let total_time = 5.0;  // 5 second simulation
    let mut time = 0.0;
    let mut step = 0;

    println!("\n=== Starting Simulation ===\n");

    while time < total_time {
        // Convert system state to OCP state
        let ocp_state = core_to_ocp(&state.load, &state.cables);

        // Update planner (runs at 10 Hz internally)
        let reference = planner.update(&ocp_state, dt);

        match reference {
            Ok(Some(ref tracking_ref)) => {
                // Extract per-quadrotor references
                for i in 0..num_quadrotors {
                    let quad_ref = reference_for_quadrotor(
                        tracking_ref,
                        i,
                        dynamics.load.params.attachment_points[i],
                        dynamics.cables[i].params.length,
                    );

                    if let Some(ref q) = quad_ref {
                        if step % 100 == 0 {  // Print every 1 second
                            println!("t={:.1}s Quad {} ref pos: [{:.3}, {:.3}, {:.3}]",
                                time, i, q.position.x, q.position.y, q.position.z);
                        }
                    }
                }
            }
            Ok(None) => {
                if step % 100 == 0 {
                    println!("t={:.1}s No reference available", time);
                }
            }
            Err(e) => {
                println!("Planner error: {:?}", e);
                break;
            }
        }

        // Simulate system forward (with zero input for demonstration)
        let input = SystemInput::zeros(num_quadrotors);
        state = dynamics.integrate(&state, &input, dt);

        // Check goal reached
        if planner.is_goal_reached(&ocp_state, 0.1) {
            println!("\n*** Goal reached at t={:.2}s ***", time);
            break;
        }

        time += dt;
        step += 1;
    }

    println!("\n=== Simulation Complete ===");
    println!("Final load position: {:?}", state.load.position);
    println!("Planner state: {:?}", planner.state());
}

/// Generate attachment points evenly spaced around a circle
fn generate_attachment_points(n: usize, radius: f64) -> Vec<Vector3<f64>> {
    use std::f64::consts::PI;

    (0..n)
        .map(|i| {
            let angle = 2.0 * PI * i as f64 / n as f64;
            Vector3::new(
                radius * angle.cos(),
                radius * angle.sin(),
                0.0,
            )
        })
        .collect()
}

/// Demonstrate state conversions when full planner isn't available
fn demonstrate_conversions(state: &SystemState, cables: &MultiCableState, dynamics: &SystemDynamics) {
    use caslo_planner::integration::{
        generate_hover_reference, generate_linear_trajectory,
    };
    use caslo_planner::trajectory::TrackingReference;

    println!("=== State Conversion Demo ===\n");

    // Convert to OCP state
    let ocp_state = core_to_ocp(&state.load, cables);
    println!("OCP State:");
    println!("  Load position: {:?}", ocp_state.load_position);
    println!("  Load velocity: {:?}", ocp_state.load_velocity);
    println!("  Num cables: {}", ocp_state.cables.len());
    for (i, cable) in ocp_state.cables.iter().enumerate() {
        println!("  Cable {}: dir={:?}, tension={:.2}N",
            i, cable.direction, cable.tension);
    }

    // Generate hover reference
    println!("\n=== Hover Reference ===");
    let hover_pos = Vector3::new(0.0, 0.0, 2.0);
    let hover_refs = generate_hover_reference(
        hover_pos,
        UnitQuaternion::identity(),
        5,
    );
    println!("Generated {} hover reference points at {:?}",
        hover_refs.len(), hover_pos);

    // Generate linear trajectory
    println!("\n=== Linear Trajectory ===");
    let start = Vector3::new(0.0, 0.0, 2.0);
    let end = Vector3::new(1.0, 1.0, 2.5);
    let trajectory = generate_linear_trajectory(
        start,
        end,
        UnitQuaternion::identity(),
        2.0,  // 2 second duration
        21,   // 21 points
    );

    println!("Trajectory from {:?} to {:?}", start, end);
    println!("Sample points:");
    for (i, point) in trajectory.iter().enumerate().step_by(5) {
        println!("  t={:.1}s: pos=[{:.3}, {:.3}, {:.3}], vel=[{:.3}, {:.3}, {:.3}]",
            i as f64 * 0.1,
            point.position.x, point.position.y, point.position.z,
            point.velocity.x, point.velocity.y, point.velocity.z);
    }

    // Create tracking reference manually
    println!("\n=== Tracking Reference ===");
    let num_cables = 3;
    let tracking_ref = TrackingReference::new(
        start,
        Vector3::zeros(),
        Vector3::zeros(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        vec![
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ],
        vec![
            Vector3::zeros(),
            Vector3::zeros(),
            Vector3::zeros(),
        ],
        vec![10.0, 10.0, 10.0],
        vec![
            Vector3::zeros(),
            Vector3::zeros(),
            Vector3::zeros(),
        ],
        vec![0.0, 0.0, 0.0],
    );

    // Extract per-quadrotor references
    for i in 0..num_cables {
        if let Some(quad_ref) = reference_for_quadrotor(
            &tracking_ref,
            i,
            dynamics.load.params.attachment_points[i],
            dynamics.cables[i].params.length,
        ) {
            println!("Quadrotor {} reference:", i);
            println!("  Position: [{:.3}, {:.3}, {:.3}]",
                quad_ref.position.x, quad_ref.position.y, quad_ref.position.z);
            println!("  Tension: {:.2}N", quad_ref.cable_tension);
        }
    }

    println!("\n=== Demo Complete ===");
    println!("\nTo run with full ACADOS solver:");
    println!("  1. Install ACADOS and set ACADOS_SOURCE_DIR");
    println!("  2. Run: python codegen/caslo_ocp.py");
    println!("  3. Build: cargo build --features acados");
}
