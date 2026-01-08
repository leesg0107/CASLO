//! Closed-Loop CASLO System Visualization with MPC Planner
//!
//! This example demonstrates the full closed-loop system:
//! - caslo-core SystemDynamics
//! - ACADOS MPC Planner for trajectory optimization
//! - Cable tension and direction control
//! - Real-time 3D visualization
//!
//! Controls:
//! - Mouse drag: Rotate view
//! - Scroll: Zoom
//! - Space: Pause/Resume
//! - R: Reset
//! - G: Go to goal
//! - H: Return to hover
//! - M: Toggle MPC (when enabled)

use kiss3d::light::Light;
use kiss3d::window::Window;
use kiss3d::scene::SceneNode;
use kiss3d::nalgebra as na;
use na::{Point3, Translation3};

/// Convert physics coordinates (NED: Z-down) to visualization coordinates (Y-up)
/// Physics NED: X=north, Y=east, Z=down (positive Z = underground)
/// Kiss3d:      X=right, Y=up,   Z=forward
/// Transform: viz_x = physics_x, viz_y = -physics_z, viz_z = physics_y
fn to_viz(v: &nalgebra::Vector3<f64>) -> Point3<f32> {
    Point3::new(v.x as f32, -v.z as f32, v.y as f32)
}

fn to_viz_translation(v: &nalgebra::Vector3<f64>) -> Translation3<f32> {
    Translation3::new(v.x as f32, -v.z as f32, v.y as f32)
}

// Import caslo-core
use caslo_core::dynamics::{
    LoadState, LoadParams,
    CableState, CableParams, CableInput, MultiCableState,
    QuadrotorParams,
    SystemState, SystemParams, SystemDynamics, SystemInput,
};
use caslo_core::control::{PositionController, PositionGains};

// Import caslo-planner for MPC
use caslo_planner::ocp::{OcpState, OcpControl, ReferencePoint};
use caslo_planner::solver::{AcadosSolver, SolverBuilder, SolveOptions};

use nalgebra::{Vector3, UnitQuaternion};

const GRAVITY: f64 = 9.81;

/// Convert SystemState to OcpState for MPC
fn system_to_ocp_state(state: &SystemState) -> OcpState {
    OcpState {
        load_position: state.load.position,
        load_velocity: state.load.velocity,
        load_orientation: state.load.orientation,
        load_angular_velocity: state.load.angular_velocity,
        cables: state.cables.cables.iter().map(|c| {
            caslo_planner::ocp::CableState {
                direction: c.direction,
                angular_velocity: c.angular_velocity,
                tension: c.tension,
            }
        }).collect(),
    }
}

/// Generate reference trajectory toward goal
fn generate_reference(
    current_pos: Vector3<f64>,
    goal_pos: Vector3<f64>,
    n_steps: usize,
    horizon: f64,
) -> Vec<ReferencePoint> {
    let direction = goal_pos - current_pos;
    let distance = direction.norm();

    // Generate smooth trajectory toward goal
    (0..n_steps).map(|k| {
        let t = k as f64 / (n_steps - 1) as f64;
        // Smooth interpolation
        let alpha = 3.0 * t * t - 2.0 * t * t * t; // Smoothstep

        // Limit speed to avoid aggressive motions
        let max_travel = 0.5 * horizon; // Max 0.5 m/s effective speed
        let travel = (alpha * distance).min(max_travel);
        let pos = if distance > 0.01 {
            current_pos + direction.normalize() * travel
        } else {
            goal_pos
        };

        // Velocity should be smooth
        let vel = if distance > 0.01 && t < 0.9 {
            direction.normalize() * (travel / horizon).min(0.5)
        } else {
            Vector3::zeros()
        };

        ReferencePoint {
            position: pos,
            velocity: vel,
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        }
    }).collect()
}

fn main() {
    println!("=== CASLO Closed-Loop Simulation ===\n");

    // ==========================================
    // 1. Setup CASLO System
    // ==========================================
    let num_quads = 3;
    let cable_length = 0.8;
    let load_mass = 0.3;

    let load_params = LoadParams::new(
        load_mass,
        Vector3::new(0.001, 0.001, 0.001),
        generate_attachment_points(num_quads, 0.1),
    );

    let cable_params = CableParams::new(cable_length);
    let quad_params = QuadrotorParams::default();

    let system_params = SystemParams::uniform(load_params, cable_params, quad_params);
    let dynamics = SystemDynamics::new(system_params);

    println!("System: {} quadrotors", num_quads);
    println!("Load mass: {} kg", load_mass);
    println!("Cable length: {} m\n", cable_length);

    // ==========================================
    // 2. Controller Setup
    // ==========================================
    // Gentle position gains for stability with cable dynamics (fallback)
    let mut position_controller = PositionController::new(PositionGains {
        kp: Vector3::new(1.0, 1.0, 2.0),
        kd: Vector3::new(1.5, 1.5, 2.0),
        ki: Vector3::new(0.1, 0.1, 0.1),
    });

    // Try to create MPC solver (requires acados feature)
    let mut mpc_solver: Option<AcadosSolver> = match SolverBuilder::new(num_quads).build() {
        Ok(solver) => {
            if solver.is_initialized() {
                println!("MPC Solver: ACADOS initialized successfully!");
                Some(solver)
            } else {
                println!("MPC Solver: Using fallback (ACADOS not available)");
                None
            }
        }
        Err(e) => {
            println!("MPC Solver: Failed to initialize ({:?}), using fallback", e);
            None
        }
    };
    let use_mpc = mpc_solver.is_some();

    // ==========================================
    // 3. Initial State (NED: Z-down, negative Z = above ground)
    // ==========================================
    let hover_position = Vector3::new(0.0, 0.0, -2.0);  // 2m above ground
    let goal_position = Vector3::new(3.0, 2.0, -2.5);   // Farther goal for visibility
    let mut target_position = hover_position; // Start at hover

    // Initial cable tension to hover (total thrust = weight)
    let hover_thrust_per_quad = load_mass * GRAVITY / num_quads as f64;
    let initial_tension = hover_thrust_per_quad * 1.05;

    let initial_load = LoadState {
        position: hover_position,
        velocity: Vector3::zeros(),
        orientation: UnitQuaternion::identity(),
        angular_velocity: Vector3::zeros(),
    };

    // Cable direction convention: sᵢ points from quadrotor toward load attachment
    // In NED (z-down), for quads above load, cables point in +Z direction
    let initial_cables = MultiCableState::new(vec![
        CableState::new(Vector3::new(0.0, 0.0, 1.0), initial_tension),
        CableState::new(Vector3::new(0.0, 0.0, 1.0), initial_tension),
        CableState::new(Vector3::new(0.0, 0.0, 1.0), initial_tension),
    ]);

    let mut state = SystemState::new(
        initial_load,
        initial_cables,
        &dynamics.load,
        &dynamics.cables,
    );

    println!("Initial position: {:?}", state.load.position);
    println!("Hover thrust per quad: {:.2} N", hover_thrust_per_quad);
    println!("Initial tension: {:.2} N\n", initial_tension);

    // ==========================================
    // 4. Visualization Setup
    // ==========================================
    let mut window = Window::new("CASLO Closed-Loop Simulation");
    window.set_light(Light::StickToCamera);
    window.set_background_color(0.05, 0.05, 0.1);

    // Load visualization (width, height, depth in Y-up system)
    let mut load_viz = window.add_cube(0.15, 0.08, 0.15);
    load_viz.set_color(0.2, 0.4, 0.9);

    // Quadrotor visualizations
    let quad_colors = [(0.9, 0.2, 0.2), (0.2, 0.9, 0.2), (0.9, 0.9, 0.2)];
    let mut quad_vizs: Vec<SceneNode> = Vec::new();
    for i in 0..num_quads {
        let mut quad = window.add_sphere(0.06);
        quad.set_color(quad_colors[i].0, quad_colors[i].1, quad_colors[i].2);
        quad_vizs.push(quad);
    }

    // Target marker (current target - cyan)
    let mut target_viz = window.add_sphere(0.08);
    target_viz.set_color(0.0, 1.0, 1.0);

    // Goal marker (orange)
    let mut goal_viz = window.add_sphere(0.1);
    goal_viz.set_color(1.0, 0.5, 0.0);
    goal_viz.set_local_translation(to_viz_translation(&goal_position));

    // Camera (Y-up coordinate system, viewing NED scene)
    let eye = Point3::new(5.0, 4.0, 3.0);   // Looking from front-right-above
    let at = Point3::new(0.5, 2.0, 0.5);    // Looking at hover/goal area

    // Simulation state
    let sim_dt = 0.002; // 500 Hz physics
    let control_dt = 0.01; // 100 Hz control
    let mut sim_time = 0.0_f64;
    let mut control_time = 0.0_f64;
    let mut paused = true; // Start paused
    let mut trajectory_history: Vec<Vector3<f64>> = Vec::new();

    // Control output
    let mut desired_tensions = vec![initial_tension; num_quads];
    let mut desired_directions = vec![Vector3::new(0.0, 0.0, 1.0); num_quads];

    println!("Controls:");
    println!("  Space: Pause/Resume");
    println!("  R: Reset");
    println!("  G: Go to goal");
    println!("  H: Hover at start");
    println!("\nPress Space to start simulation...\n");

    // ==========================================
    // 5. Main Loop
    // ==========================================
    while window.render_with_camera(&mut kiss3d::camera::ArcBall::new(eye, at)) {
        // Handle input
        for event in window.events().iter() {
            match event.value {
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::Space, kiss3d::event::Action::Press, _) => {
                    paused = !paused;
                    println!("{}", if paused { "Paused" } else { "Running" });
                }
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::R, kiss3d::event::Action::Press, _) => {
                    // Reset
                    sim_time = 0.0;
                    control_time = 0.0;
                    trajectory_history.clear();
                    position_controller.reset();
                    target_position = hover_position;

                    let initial_load = LoadState {
                        position: hover_position,
                        velocity: Vector3::zeros(),
                        orientation: UnitQuaternion::identity(),
                        angular_velocity: Vector3::zeros(),
                    };
                    // Cable direction: +Z in NED (pointing from quad toward load)
                    let initial_cables = MultiCableState::new(vec![
                        CableState::new(Vector3::new(0.0, 0.0, 1.0), initial_tension),
                        CableState::new(Vector3::new(0.0, 0.0, 1.0), initial_tension),
                        CableState::new(Vector3::new(0.0, 0.0, 1.0), initial_tension),
                    ]);
                    state = SystemState::new(
                        initial_load,
                        initial_cables,
                        &dynamics.load,
                        &dynamics.cables,
                    );
                    desired_tensions = vec![initial_tension; num_quads];
                    desired_directions = vec![Vector3::new(0.0, 0.0, 1.0); num_quads];
                    println!("Reset to initial state");
                }
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::G, kiss3d::event::Action::Press, _) => {
                    target_position = goal_position;
                    println!("Target: Goal {:?}", goal_position);
                }
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::H, kiss3d::event::Action::Press, _) => {
                    target_position = hover_position;
                    println!("Target: Hover {:?}", hover_position);
                }
                _ => {}
            }
        }

        if !paused {
            // Run physics steps
            for _ in 0..8 {
                // === Control Loop (at control_dt rate) ===
                if sim_time >= control_time {
                    control_time += control_dt;

                    let pos = state.load.position;

                    // DEBUG: Print every second
                    if (sim_time * 10.0).fract() < 0.01 && sim_time > 0.1 {
                        println!("DEBUG t={:.2}s pos=[{:.3},{:.3},{:.3}]", sim_time, pos.x, pos.y, pos.z);
                    }

                    // Try MPC first, fall back to reactive controller
                    let mut mpc_control: Option<OcpControl> = None;

                    if let Some(ref mut solver) = mpc_solver {
                        // Convert state for MPC
                        let ocp_state = system_to_ocp_state(&state);

                        // Generate reference trajectory
                        let reference = generate_reference(
                            pos,
                            target_position,
                            21, // N+1 points
                            2.0, // 2 second horizon
                        );

                        // Solve MPC
                        match solver.solve(&ocp_state, &reference, &SolveOptions::default()) {
                            Ok(trajectory) => {
                                // Get the first control
                                if let Some(control) = trajectory.controls.first() {
                                    mpc_control = Some(control.clone());

                                    // Extract desired tensions and directions from MPC solution
                                    // The MPC solves for each cable individually!
                                    if let Some(next_state) = trajectory.states.get(1) {
                                        for i in 0..num_quads {
                                            if i < next_state.cables.len() {
                                                desired_tensions[i] = next_state.cables[i].tension;
                                                desired_directions[i] = next_state.cables[i].direction;
                                            }
                                        }
                                    }

                                    // DEBUG: Print MPC success occasionally
                                    if trajectory_history.len() % 500 == 0 {
                                        println!("MPC OK: T=[{:.2},{:.2},{:.2}]",
                                            desired_tensions[0], desired_tensions[1], desired_tensions[2]);
                                    }
                                }
                            }
                            Err(e) => {
                                // Print all MPC failures
                                if trajectory_history.len() % 100 == 0 || trajectory_history.len() < 10 {
                                    println!("MPC failed: {:?}", e);
                                }
                            }
                        }
                    }

                    // Fallback: use reactive controller if MPC not available or failed
                    if mpc_control.is_none() {
                        let vel = state.load.velocity;
                        let acc_des = position_controller.compute(
                            &pos, &vel, &target_position, &Vector3::zeros(), &Vector3::zeros(), control_dt
                        );

                        // Simple force allocation
                        let desired_total_force = Vector3::new(
                            -load_mass * acc_des.x,
                            -load_mass * acc_des.y,
                            load_mass * (GRAVITY - acc_des.z),
                        );

                        let total_tension = desired_total_force.norm();
                        let base_tension = (total_tension / num_quads as f64).clamp(0.3, 15.0);

                        let force_dir = if total_tension > 0.01 {
                            desired_total_force / total_tension
                        } else {
                            Vector3::new(0.0, 0.0, 1.0)
                        };

                        // Limit tilt angle
                        let max_tilt = 0.35;
                        let horizontal = Vector3::new(force_dir.x, force_dir.y, 0.0);
                        let horiz_mag = horizontal.norm();

                        let clamped_dir = if horiz_mag > max_tilt {
                            let scale = max_tilt / horiz_mag;
                            let clamped_horiz = horizontal * scale;
                            let z_component = (1.0 - clamped_horiz.norm_squared()).sqrt();
                            Vector3::new(clamped_horiz.x, clamped_horiz.y, z_component)
                        } else if force_dir.norm() > 0.01 {
                            force_dir.normalize()
                        } else {
                            Vector3::new(0.0, 0.0, 1.0)
                        };

                        for i in 0..num_quads {
                            desired_tensions[i] = base_tension;
                            desired_directions[i] = clamped_dir;
                        }
                    }

                    // Debug output
                    if trajectory_history.len() % 100 == 0 && trajectory_history.len() > 0 {
                        let error = (target_position - pos).norm();
                        let mode = if mpc_control.is_some() { "MPC" } else { "PID" };
                        println!("t={:.1}s [{:}] pos=[{:.2},{:.2},{:.2}] err={:.3}m T=[{:.1},{:.1},{:.1}]N",
                            sim_time, mode, pos.x, pos.y, pos.z, error,
                            desired_tensions[0], desired_tensions[1], desired_tensions[2]);
                    }
                }

                // === Create control input ===
                let mut cable_inputs = Vec::with_capacity(num_quads);
                for i in 0..num_quads {
                    let cable = &state.cables.cables[i];

                    // Tension control
                    let tension_error = desired_tensions[i] - cable.tension;
                    let tension_accel = tension_error * 100.0; // P-gain for tension

                    // Direction control via angular velocity (gentle gains for stability)
                    // To rotate from current direction s to desired direction s_des,
                    // angular velocity r should be perpendicular to both: r ∝ s × s_des
                    let current_dir = cable.direction;
                    let desired_dir = desired_directions[i];

                    // Direction error (cross product gives rotation axis, magnitude gives sin of angle)
                    let dir_error = current_dir.cross(&desired_dir);

                    // Very gentle direction control to avoid instability
                    let kp_dir = 1.0;   // Reduced from 5.0
                    let kd_dir = 0.5;   // Reduced from 2.0
                    let desired_omega = dir_error * kp_dir;

                    // Angular velocity error with damping
                    let omega_error = desired_omega - cable.angular_velocity;

                    // Desired angular acceleration
                    let desired_alpha = omega_error * kd_dir - cable.angular_velocity * 0.3;  // Add damping

                    // Angular jerk to achieve desired acceleration (reduced gain)
                    let alpha_error = desired_alpha - cable.angular_acceleration;
                    let angular_jerk = alpha_error * 10.0;  // Reduced from 50.0

                    cable_inputs.push(CableInput {
                        angular_jerk,
                        tension_acceleration: tension_accel,
                    });
                }

                let input = SystemInput {
                    cables: cable_inputs,
                    quadrotors: vec![Default::default(); num_quads],
                };

                // === Integrate dynamics ===
                state = dynamics.integrate(&state, &input, sim_dt);

                sim_time += sim_dt;
            }

            // Record trajectory
            if trajectory_history.len() < 10000 {
                trajectory_history.push(state.load.position);
            }
        }

        // ==========================================
        // 6. Update Visualization (convert Z-up to Y-up)
        // ==========================================

        // Update load
        let load_pos = state.load.position;
        load_viz.set_local_translation(to_viz_translation(&load_pos));

        // Rotation also needs coordinate transform (swap pitch/yaw for Y-up)
        let (roll, pitch, yaw) = state.load.orientation.euler_angles();
        let load_rot = na::UnitQuaternion::from_euler_angles(roll as f32, yaw as f32, pitch as f32);
        load_viz.set_local_rotation(load_rot);

        // Update quadrotors and draw cables
        for (i, quad_viz) in quad_vizs.iter_mut().enumerate() {
            let quad_pos = state.quadrotors[i].position;
            quad_viz.set_local_translation(to_viz_translation(&quad_pos));

            // Draw cable
            let attach = dynamics.load.attachment_world(&state.load, i);
            window.draw_line(
                &to_viz(&attach),
                &to_viz(&quad_pos),
                &Point3::new(0.9, 0.9, 0.9),
            );
        }

        // Update target marker
        target_viz.set_local_translation(to_viz_translation(&target_position));

        // Draw trajectory history (last 500 points)
        let start_idx = trajectory_history.len().saturating_sub(500);
        for i in (start_idx + 1)..trajectory_history.len() {
            let p1 = &trajectory_history[i - 1];
            let p2 = &trajectory_history[i];
            let alpha = (i - start_idx) as f32 / 500.0;
            window.draw_line(
                &to_viz(p1),
                &to_viz(p2),
                &Point3::new(0.2, 0.4 * alpha, 0.8 * alpha),
            );
        }

        // Draw ground grid (Y=0 plane in visualization = Z=0 in physics)
        draw_ground(&mut window, 0.0);

        // Draw coordinate axes (physics frame shown in visualization)
        let origin = Point3::origin();
        // X-axis (red) - same in both frames
        window.draw_line(&origin, &Point3::new(0.5, 0.0, 0.0), &Point3::new(1.0, 0.0, 0.0));
        // Z-axis (blue) -> Y in viz (up)
        window.draw_line(&origin, &Point3::new(0.0, 0.5, 0.0), &Point3::new(0.0, 0.0, 1.0));
        // Y-axis (green) -> Z in viz
        window.draw_line(&origin, &Point3::new(0.0, 0.0, 0.5), &Point3::new(0.0, 1.0, 0.0));
    }
}

fn generate_attachment_points(n: usize, radius: f64) -> Vec<Vector3<f64>> {
    use std::f64::consts::PI;
    (0..n)
        .map(|i| {
            let angle = 2.0 * PI * i as f64 / n as f64;
            Vector3::new(radius * angle.cos(), radius * angle.sin(), 0.0)
        })
        .collect()
}

fn draw_ground(window: &mut Window, y: f32) {
    // Ground is Y=0 plane in visualization (Z=0 in physics)
    let size = 4.0_f32;
    let step = 0.5_f32;
    let color = Point3::new(0.2, 0.2, 0.25);

    let mut x = -size;
    while x <= size {
        // Lines parallel to Z-axis (physics Y)
        window.draw_line(&Point3::new(x, y, -size), &Point3::new(x, y, size), &color);
        // Lines parallel to X-axis
        window.draw_line(&Point3::new(-size, y, x), &Point3::new(size, y, x), &color);
        x += step;
    }
}
