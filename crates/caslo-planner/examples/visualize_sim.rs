//! Closed-Loop CASLO System Visualization with MPC Planner
//!
//! This example demonstrates the full closed-loop system with paper scenarios:
//! - Figure-8 trajectories (Slow, Medium, Medium Plus, Fast)
//! - Obstacle avoidance scenarios
//! - Paper system parameters (1.4kg load, 1m cables, 0.6kg quads)
//!
//! Controls:
//! - Mouse drag: Rotate view
//! - Scroll: Zoom
//! - Space: Pause/Resume
//! - R: Reset
//! - 1-4: Select Figure-8 scenario (1=Slow, 2=Medium, 3=Medium+, 4=Fast)
//! - 5: Narrow passage scenario
//! - 6: Horizontal gap scenario
//! - H: Return to hover
//! - M: Toggle MPC (when enabled)

use kiss3d::light::Light;
use kiss3d::window::Window;
use kiss3d::scene::SceneNode;
use kiss3d::nalgebra as na;
use na::{Point3, Translation3};

/// Convert physics coordinates (Z-UP from paper) to visualization coordinates (Y-up)
/// Physics Z-UP: X=forward, Y=left, Z=up (positive Z = above ground)
/// Kiss3d:       X=right,   Y=up,  Z=forward
/// Transform: viz_x = physics_x, viz_y = physics_z (Z maps to Y), viz_z = physics_y
fn to_viz(v: &nalgebra::Vector3<f64>) -> Point3<f32> {
    Point3::new(v.x as f32, v.z as f32, v.y as f32)
}

fn to_viz_translation(v: &nalgebra::Vector3<f64>) -> Translation3<f32> {
    Translation3::new(v.x as f32, v.z as f32, v.y as f32)
}

// Import caslo-core
use caslo_core::dynamics::{
    LoadState, LoadParams,
    CableState, CableParams, CableInput, MultiCableState,
    QuadrotorParams,
    SystemState, SystemParams, SystemDynamics, SystemInput,
};
use caslo_core::control::{PositionController, PositionGains};
// Paper-based control components (Eq. 15, Eq. 5)
use caslo_core::control::{
    QuadrotorTracker, TrackerGains, MultiQuadrotorTracker,
    QuadrotorTrackerState, QuadrotorTrajectoryRef, ExternalForce,
};
use caslo_core::kinematics::{
    KinematicConstraint, LoadKinematicState, CableKinematicState,
};

// Import caslo-planner for MPC and scenarios
use caslo_planner::ocp::{OcpState, OcpControl, ReferencePoint};
use caslo_planner::solver::{AcadosSolver, SolverBuilder, SolveOptions};
use caslo_planner::scenarios::{
    PaperSystemParams, Figure8Params, PaperOcpParams,
    NarrowPassageScenario, HorizontalGapScenario,
    generate_figure8_trajectory, TrajectoryPoint,
};

use nalgebra::{Vector3, UnitQuaternion};

const GRAVITY: f64 = 9.81;

/// Scenario selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scenario {
    Hover,
    Figure8Slow,
    Figure8Medium,
    Figure8MediumPlus,
    Figure8Fast,
    NarrowPassage,
    HorizontalGap,
}

impl Scenario {
    fn name(&self) -> &'static str {
        match self {
            Scenario::Hover => "Hover",
            Scenario::Figure8Slow => "Figure-8 Slow (1 m/s)",
            Scenario::Figure8Medium => "Figure-8 Medium (2 m/s)",
            Scenario::Figure8MediumPlus => "Figure-8 Medium+ (2 m/s, 4 m/s²)",
            Scenario::Figure8Fast => "Figure-8 Fast (5 m/s, 8 m/s²)",
            Scenario::NarrowPassage => "Narrow Passage",
            Scenario::HorizontalGap => "Horizontal Gap",
        }
    }
}

/// Convert SystemState to OcpState for MPC
///
/// Uses equilibrium-based cable derivatives (angular velocities, accelerations,
/// tension rates all set to zero) which provides dynamically consistent initial
/// conditions for the MPC solver. The actual cable derivative estimation is
/// handled by the MPC's trajectory resampling mechanism.
///
/// This approach follows the paper's methodology where cable derivatives are
/// not directly measured but estimated from the previous MPC solution.
fn system_to_ocp_state(state: &SystemState) -> OcpState {
    let cable_directions: Vec<Vector3<f64>> = state.cables.cables.iter()
        .map(|c| {
            if c.direction.norm() > 1e-6 {
                c.direction.normalize()
            } else {
                Vector3::new(0.0, 0.0, -1.0)
            }
        })
        .collect();

    let cable_tensions: Vec<f64> = state.cables.cables.iter()
        .map(|c| c.tension.clamp(0.5, 50.0))
        .collect();

    // Use equilibrium-based initialization:
    // - Load pose/velocity from simulation (measured)
    // - Cable directions from simulation (measured)
    // - Cable tensions from simulation (measured)
    // - Cable angular velocities = 0 (equilibrium assumption)
    // - Cable angular accelerations = 0 (equilibrium assumption)
    // - Tension rates = 0 (equilibrium assumption)
    //
    // The MPC will then use trajectory resampling to provide better estimates
    // of these derivatives from its previous solution.
    OcpState::from_measured_with_equilibrium_derivatives(
        state.load.position,
        state.load.velocity,
        state.load.orientation,
        state.load.angular_velocity,
        &cable_directions,
        &cable_tensions,
    )
}

/// Generate reference trajectory from Figure-8 parameters
fn generate_figure8_reference(
    params: &Figure8Params,
    current_time: f64,
    n_steps: usize,
    horizon: f64,
) -> Vec<ReferencePoint> {
    let dt = horizon / (n_steps - 1) as f64;

    (0..n_steps).map(|k| {
        let t = current_time + k as f64 * dt;
        ReferencePoint {
            position: params.position(t),
            velocity: params.velocity(t),
            orientation: params.orientation(t),
            angular_velocity: params.angular_velocity(t),
        }
    }).collect()
}

/// Generate reference for point-to-point motion
fn generate_point_reference(
    current_pos: Vector3<f64>,
    goal_pos: Vector3<f64>,
    n_steps: usize,
    horizon: f64,
) -> Vec<ReferencePoint> {
    let direction = goal_pos - current_pos;
    let distance = direction.norm();

    (0..n_steps).map(|k| {
        let t = k as f64 / (n_steps - 1) as f64;
        let alpha = 3.0 * t * t - 2.0 * t * t * t;

        let max_travel = 0.5 * horizon;
        let travel = (alpha * distance).min(max_travel);
        let pos = if distance > 0.01 {
            current_pos + direction.normalize() * travel
        } else {
            goal_pos
        };

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
    println!("=== CASLO Paper Scenarios Simulation ===\n");

    // ==========================================
    // 1. Setup CASLO System with Paper Parameters
    // ==========================================
    let paper_params = PaperSystemParams::paper_3quad();
    let ocp_params = PaperOcpParams::default();

    let num_quads = paper_params.num_quads;
    let cable_length = paper_params.cable_lengths[0];
    let load_mass = paper_params.load_mass;

    let load_params = LoadParams::new(
        load_mass,
        paper_params.load_inertia,
        paper_params.attachment_points.clone(),
    );

    let cable_params = CableParams::new(cable_length);

    let quad_params = QuadrotorParams {
        mass: paper_params.quadrotor_mass,
        inertia: nalgebra::Matrix3::from_diagonal(&paper_params.quadrotor_inertia),
        max_thrust: paper_params.max_thrust,
        ..Default::default()
    };

    let system_params = SystemParams::uniform(load_params, cable_params, quad_params);
    let dynamics = SystemDynamics::new(system_params);

    println!("System Configuration (from paper):");
    println!("  Load mass: {} kg", load_mass);
    println!("  Load dimensions: {:?} m", paper_params.load_dimensions);
    println!("  Cable length: {} m", cable_length);
    println!("  Quadrotors: {} × {} kg", num_quads, paper_params.quadrotor_mass);
    println!("  Max thrust: {} N per quad", paper_params.max_thrust);
    println!("  Hover tension: {:.2} N per cable\n", paper_params.hover_tension());

    // ==========================================
    // 2. Controller Setup (Paper Architecture)
    // ==========================================
    // Position controller (fallback)
    let mut position_controller = PositionController::new(PositionGains {
        kp: Vector3::new(2.0, 2.0, 4.0),
        kd: Vector3::new(3.0, 3.0, 4.0),
        ki: Vector3::new(0.1, 0.1, 0.1),
    });

    // Kinematic Constraint (Eq. 5) - converts cable states to quadrotor trajectories
    let kinematic_constraint = KinematicConstraint::new(
        paper_params.attachment_points.clone(),
        paper_params.cable_lengths.clone(),
    );

    // Quadrotor Trajectory Trackers (Eq. 15) - onboard controllers at 300Hz
    let quad_inertia = nalgebra::Matrix3::from_diagonal(&paper_params.quadrotor_inertia);
    let tracker_gains = TrackerGains {
        kp: Vector3::new(10.0, 10.0, 15.0),  // Paper gains
        kv: Vector3::new(5.0, 5.0, 8.0),
    };
    let mut quadrotor_trackers = MultiQuadrotorTracker::uniform(
        num_quads,
        paper_params.quadrotor_mass,
        quad_inertia,
        tracker_gains,
    );

    println!("Control Architecture (Paper-based):");
    println!("  KinematicConstraint: Eq. 5 + derivatives (S1)");
    println!("  QuadrotorTracker: Eq. 15 trajectory tracking");
    println!("  INDI: Eq. S9 (integrated in tracker)\n");

    // Create system parameters for MPC solver matching the paper
    // load_inertia is a Vector3 containing diagonal elements [Ixx, Iyy, Izz]
    let mpc_system_params = caslo_planner::ocp::SystemParameters {
        load_mass: paper_params.load_mass,
        load_inertia: [
            paper_params.load_inertia.x, 0.0, 0.0,  // Row 1: [Ixx, 0, 0]
            0.0, paper_params.load_inertia.y, 0.0,  // Row 2: [0, Iyy, 0]
            0.0, 0.0, paper_params.load_inertia.z,  // Row 3: [0, 0, Izz]
        ],
        quadrotor_masses: vec![paper_params.quadrotor_mass; num_quads],
        cable_lengths: paper_params.cable_lengths.clone(),
        attachment_points: paper_params.attachment_points.iter()
            .map(|p| [p.x, p.y, p.z])
            .collect(),
        gravity: 9.81,
    };

    let mut mpc_solver: Option<AcadosSolver> = match SolverBuilder::new(num_quads).build() {
        Ok(mut solver) => {
            if solver.is_initialized() {
                // Set system parameters BEFORE solving! This is critical.
                if let Err(e) = solver.set_parameters(&mpc_system_params) {
                    println!("MPC Solver: Failed to set parameters ({:?}), using fallback", e);
                    None
                } else {
                    println!("MPC Solver: ACADOS initialized with paper parameters!");
                    println!("  Load mass: {} kg", mpc_system_params.load_mass);
                    println!("  Cable lengths: {:?}", mpc_system_params.cable_lengths);
                    Some(solver)
                }
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

    // ==========================================
    // 3. Initial State
    // ==========================================
    // Start at x=2.5 (right side of Figure-8) for smooth transition
    let hover_position = Vector3::new(2.5, 0.0, 1.0);
    let initial_tension = paper_params.hover_tension();

    let initial_load = LoadState {
        position: hover_position,
        velocity: Vector3::zeros(),
        orientation: UnitQuaternion::identity(),
        angular_velocity: Vector3::zeros(),
    };

    let initial_cables = MultiCableState::new(
        (0..num_quads)
            .map(|_| CableState::new(Vector3::new(0.0, 0.0, -1.0), initial_tension))
            .collect()
    );

    let mut state = SystemState::new(
        initial_load,
        initial_cables,
        &dynamics.load,
        &dynamics.cables,
    );

    // ==========================================
    // 4. Scenario State
    // ==========================================
    let mut current_scenario = Scenario::Hover;
    let mut figure8_params: Option<Figure8Params> = None;
    let mut scenario_start_time = 0.0;
    let mut target_position = hover_position;
    let mut transition_start_pos = hover_position;  // Position when scenario changed
    const TRANSITION_DURATION: f64 = 2.0;  // Smooth transition over 2 seconds

    // Obstacle scenarios
    let narrow_passage = NarrowPassageScenario::default();
    let horizontal_gap = HorizontalGapScenario::default();

    // ==========================================
    // 5. Visualization Setup
    // ==========================================
    let mut window = Window::new("CASLO Paper Scenarios");
    window.set_light(Light::StickToCamera);
    window.set_background_color(0.05, 0.05, 0.1);

    // Load visualization
    let mut load_viz = window.add_cube(
        paper_params.load_dimensions.x as f32,
        paper_params.load_dimensions.z as f32,
        paper_params.load_dimensions.y as f32,
    );
    load_viz.set_color(0.2, 0.4, 0.9);

    // Quadrotor visualizations
    let quad_colors = [(0.9, 0.2, 0.2), (0.2, 0.9, 0.2), (0.9, 0.9, 0.2), (0.2, 0.9, 0.9)];
    let mut quad_vizs: Vec<SceneNode> = Vec::new();
    for i in 0..num_quads {
        let mut quad = window.add_sphere(0.08);
        quad.set_color(quad_colors[i % 4].0, quad_colors[i % 4].1, quad_colors[i % 4].2);
        quad_vizs.push(quad);
    }

    // Target marker
    let mut target_viz = window.add_sphere(0.1);
    target_viz.set_color(0.0, 1.0, 1.0);

    // Camera - ArcBall allows mouse drag to rotate view
    // Left mouse drag: rotate around target
    // Right mouse drag: pan view
    // Scroll: zoom in/out
    let eye = Point3::new(6.0, 4.0, 4.0);
    let at = Point3::new(0.0, 0.0, 1.0);
    let mut camera = kiss3d::camera::ArcBall::new(eye, at);

    // Simulation state
    let sim_dt = 0.002;
    let control_dt = 0.01;
    let mut sim_time = 0.0_f64;
    let mut control_time = 0.0_f64;
    let mut paused = true;
    let mut trajectory_history: Vec<Vector3<f64>> = Vec::new();

    // Control output
    let mut desired_tensions = vec![initial_tension; num_quads];
    let mut desired_directions = vec![Vector3::new(0.0, 0.0, -1.0); num_quads];
    let mut mpc_controls: Option<OcpControl> = None;  // MPC optimized controls [γᵢ, λᵢ]
    let mut mpc_next_state: Option<OcpState> = None;  // MPC predicted next state (for angular_jerk)

    // Performance metrics
    let mut position_errors: Vec<f64> = Vec::new();
    let mut max_velocity: f64 = 0.0;
    let mut max_acceleration: f64 = 0.0;

    println!("Controls:");
    println!("  Space: Pause/Resume");
    println!("  R: Reset");
    println!("  H: Hover");
    println!("  1: Figure-8 Slow (1 m/s)");
    println!("  2: Figure-8 Medium (2 m/s)");
    println!("  3: Figure-8 Medium+ (2 m/s, 4 m/s²) - Baseline methods crash!");
    println!("  4: Figure-8 Fast (5 m/s, 8 m/s²) - Only CASLO succeeds!");
    println!("  5: Narrow Passage");
    println!("  6: Horizontal Gap");
    println!("\nCamera:");
    println!("  Left mouse drag: Rotate view");
    println!("  Right mouse drag: Pan view");
    println!("  Scroll wheel: Zoom in/out");
    println!("\nPress Space to start simulation...\n");

    // ==========================================
    // 6. Main Loop
    // ==========================================
    while window.render_with_camera(&mut camera) {
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
                    position_errors.clear();
                    max_velocity = 0.0;
                    max_acceleration = 0.0;
                    position_controller.reset();
                    quadrotor_trackers.reset_all();  // Reset paper-based trackers
                    // Clear MPC stored trajectory for fresh start
                    if let Some(ref mut solver) = mpc_solver {
                        solver.clear_stored_trajectory();
                        solver.reset_warm_start();
                    }
                    current_scenario = Scenario::Hover;
                    figure8_params = None;
                    target_position = hover_position;

                    let initial_load = LoadState {
                        position: hover_position,
                        velocity: Vector3::zeros(),
                        orientation: UnitQuaternion::identity(),
                        angular_velocity: Vector3::zeros(),
                    };
                    let initial_cables = MultiCableState::new(
                        (0..num_quads)
                            .map(|_| CableState::new(Vector3::new(0.0, 0.0, -1.0), initial_tension))
                            .collect()
                    );
                    state = SystemState::new(
                        initial_load,
                        initial_cables,
                        &dynamics.load,
                        &dynamics.cables,
                    );
                    desired_tensions = vec![initial_tension; num_quads];
                    desired_directions = vec![Vector3::new(0.0, 0.0, -1.0); num_quads];
                    println!("Reset to initial state");
                }
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::H, kiss3d::event::Action::Press, _) => {
                    current_scenario = Scenario::Hover;
                    figure8_params = None;
                    target_position = hover_position;
                    // Clear MPC trajectory on scenario change for fresh start
                    if let Some(ref mut solver) = mpc_solver {
                        solver.clear_stored_trajectory();
                    }
                    println!("Scenario: Hover");
                }
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::Key1, kiss3d::event::Action::Press, _) => {
                    current_scenario = Scenario::Figure8Slow;
                    figure8_params = Some(Figure8Params::slow());
                    scenario_start_time = sim_time;
                    transition_start_pos = state.load.position;  // Record position for smooth transition
                    position_errors.clear();
                    if let Some(ref mut solver) = mpc_solver {
                        solver.clear_stored_trajectory();
                    }
                    println!("Scenario: {} (Period: {:.1}s) - Transitioning from {:?}",
                        current_scenario.name(), figure8_params.as_ref().unwrap().period(), transition_start_pos);
                }
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::Key2, kiss3d::event::Action::Press, _) => {
                    current_scenario = Scenario::Figure8Medium;
                    figure8_params = Some(Figure8Params::medium());
                    scenario_start_time = sim_time;
                    transition_start_pos = state.load.position;
                    position_errors.clear();
                    if let Some(ref mut solver) = mpc_solver {
                        solver.clear_stored_trajectory();
                    }
                    println!("Scenario: {} (Period: {:.1}s) - Transitioning from {:?}",
                        current_scenario.name(), figure8_params.as_ref().unwrap().period(), transition_start_pos);
                }
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::Key3, kiss3d::event::Action::Press, _) => {
                    current_scenario = Scenario::Figure8MediumPlus;
                    figure8_params = Some(Figure8Params::medium_plus());
                    scenario_start_time = sim_time;
                    transition_start_pos = state.load.position;
                    position_errors.clear();
                    if let Some(ref mut solver) = mpc_solver {
                        solver.clear_stored_trajectory();
                    }
                    println!("Scenario: {} (Period: {:.1}s)", current_scenario.name(), figure8_params.as_ref().unwrap().period());
                    println!("  WARNING: Baseline methods crash on this scenario!");
                }
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::Key4, kiss3d::event::Action::Press, _) => {
                    current_scenario = Scenario::Figure8Fast;
                    figure8_params = Some(Figure8Params::fast());
                    scenario_start_time = sim_time;
                    transition_start_pos = state.load.position;
                    position_errors.clear();
                    if let Some(ref mut solver) = mpc_solver {
                        solver.clear_stored_trajectory();
                    }
                    println!("Scenario: {} (Period: {:.1}s)", current_scenario.name(), figure8_params.as_ref().unwrap().period());
                    println!("  WARNING: Only CASLO succeeds on this scenario!");
                }
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::Key5, kiss3d::event::Action::Press, _) => {
                    current_scenario = Scenario::NarrowPassage;
                    figure8_params = None;
                    target_position = narrow_passage.goal;
                    scenario_start_time = sim_time;
                    position_errors.clear();
                    if let Some(ref mut solver) = mpc_solver {
                        solver.clear_stored_trajectory();
                    }
                    println!("Scenario: {}", current_scenario.name());
                    println!("  Gap width: {} m, System width: ~1.4 m", narrow_passage.gap_width);
                    println!("  Required: ~70° load tilt");
                }
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::Key6, kiss3d::event::Action::Press, _) => {
                    current_scenario = Scenario::HorizontalGap;
                    figure8_params = None;
                    target_position = horizontal_gap.goal;
                    scenario_start_time = sim_time;
                    position_errors.clear();
                    if let Some(ref mut solver) = mpc_solver {
                        solver.clear_stored_trajectory();
                    }
                    println!("Scenario: {}", current_scenario.name());
                    println!("  Gap height: {} m, System height: ~1.2 m", horizontal_gap.gap_height);
                    println!("  Required: Rapid cable inclination change");
                }
                _ => {}
            }
        }

        if !paused {
            for _ in 0..8 {
                // === Control Loop ===
                if sim_time >= control_time {
                    control_time += control_dt;

                    let pos = state.load.position;
                    let vel = state.load.velocity;

                    // Update performance metrics
                    let vel_mag = vel.norm();
                    if vel_mag > max_velocity {
                        max_velocity = vel_mag;
                    }

                    // Compute reference based on scenario
                    let reference = match current_scenario {
                        Scenario::Hover => {
                            target_position = hover_position;
                            generate_point_reference(pos, hover_position, 21, 2.0)
                        }
                        Scenario::Figure8Slow | Scenario::Figure8Medium |
                        Scenario::Figure8MediumPlus | Scenario::Figure8Fast => {
                            if let Some(ref params) = figure8_params {
                                let t = sim_time - scenario_start_time;

                                // Smooth transition: blend from starting position to Figure-8
                                // Using smooth quintic blending over TRANSITION_DURATION seconds
                                let blend_factor = if t < TRANSITION_DURATION {
                                    // Quintic blend: smooth acceleration/deceleration
                                    let tau = t / TRANSITION_DURATION;
                                    10.0 * tau.powi(3) - 15.0 * tau.powi(4) + 6.0 * tau.powi(5)
                                } else {
                                    1.0
                                };

                                let raw_ref_pos = params.position(t);
                                // During transition, blend between start position and Figure-8
                                let ref_pos = transition_start_pos.lerp(&raw_ref_pos, blend_factor);
                                target_position = ref_pos;

                                // Track position error (only after transition)
                                if t >= TRANSITION_DURATION {
                                    let error = (pos - ref_pos).norm();
                                    position_errors.push(error);
                                }

                                // Generate blended reference trajectory
                                if blend_factor < 1.0 {
                                    // During transition, generate smooth point-to-point to current blended target
                                    generate_point_reference(pos, ref_pos, 21, 2.0)
                                } else {
                                    generate_figure8_reference(params, t, 21, 2.0)
                                }
                            } else {
                                generate_point_reference(pos, hover_position, 21, 2.0)
                            }
                        }
                        Scenario::NarrowPassage => {
                            target_position = narrow_passage.goal;
                            generate_point_reference(pos, narrow_passage.goal, 21, 2.0)
                        }
                        Scenario::HorizontalGap => {
                            target_position = horizontal_gap.goal;
                            generate_point_reference(pos, horizontal_gap.goal, 21, 2.0)
                        }
                    };

                    // Try MPC - Get optimized control inputs [γᵢ, λᵢ] directly
                    // Using paper's trajectory resampling approach for OCP initialization
                    let mut mpc_success = false;
                    mpc_controls = None;  // Reset for this control cycle
                    mpc_next_state = None;

                    if let Some(ref mut solver) = mpc_solver {
                        let ocp_state = system_to_ocp_state(&state);

                        // Use solve_at_time for trajectory resampling (paper approach)
                        // This passes current simulation time so MPC can resample its
                        // previous solution for initialization instead of using raw state
                        let solve_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            solver.solve_at_time(&ocp_state, &reference, &SolveOptions::default(), sim_time)
                        }));

                        match solve_result {
                            Ok(Ok(trajectory)) => {
                                // Only use MPC controls if trajectory is valid
                                // (fallback solver marks trajectory as invalid)
                                if trajectory.is_valid {
                                    // Get MPC's optimized control inputs from controls[0]
                                    // These are the angular snap (r⃛ᵢ) and tension acceleration (λᵢ)
                                    // that MPC computed to optimally track the trajectory
                                    if let Some(control) = trajectory.controls.get(0) {
                                        mpc_controls = Some(control.clone());
                                        mpc_success = true;
                                    }
                                    // Also store states[1] which contains angular_jerk for the simulator
                                    // (MPC uses 3rd-order model with snap as control, simulator uses 2nd-order with jerk)
                                    if let Some(next_state) = trajectory.states.get(1) {
                                        mpc_next_state = Some(next_state.clone());
                                    }
                                }

                                // Update desired states for visualization (even from fallback)
                                if let Some(next_state) = trajectory.states.get(1) {
                                    for i in 0..num_quads {
                                        if i < next_state.cables.len() {
                                            desired_tensions[i] = next_state.cables[i].tension;
                                            desired_directions[i] = next_state.cables[i].direction;
                                        }
                                    }
                                }
                            }
                            Ok(Err(_e)) => {}
                            Err(_) => {
                                mpc_solver = None;
                            }
                        }
                    }

                    // Fallback control - Feedforward + Feedback controller
                    if !mpc_success {
                        let pos_error = target_position - pos;
                        let distance = pos_error.norm();

                        // Get reference velocity and acceleration from trajectory
                        let (ref_vel, ref_acc) = if let Some(ref params) = figure8_params {
                            let t = sim_time - scenario_start_time;
                            (params.velocity(t), params.acceleration(t))
                        } else {
                            (Vector3::zeros(), Vector3::zeros())
                        };

                        // Feedforward + Feedback acceleration
                        // a_des = a_ref + kp * (p_ref - p) + kd * (v_ref - v)
                        let kp = 4.0;  // Position gain
                        let kd = 4.0;  // Velocity gain (critical damping: kd = 2*sqrt(kp))

                        let vel_error = ref_vel - vel;
                        let acc_des = ref_acc + pos_error * kp + vel_error * kd;

                        // Clamp acceleration based on scenario
                        let max_acc = match current_scenario {
                            Scenario::Figure8Fast => 12.0,       // Allow more than 8 m/s² for tracking
                            Scenario::Figure8MediumPlus => 6.0,  // Allow more than 4 m/s²
                            Scenario::Figure8Medium => 4.0,
                            Scenario::Figure8Slow => 2.0,
                            _ => 3.0,
                        };

                        let acc_clamped = if acc_des.norm() > max_acc {
                            acc_des.normalize() * max_acc
                        } else {
                            acc_des
                        };

                        // From load dynamics Eq. 2: m*a = -Σ(t_i * s_i) + m*g
                        // Where s_i points FROM quad TOWARD load (i.e., down when quad is above load)
                        // Rearranging: Σ(t_i * s_i) = m*g - m*a = m*(g - a)
                        //
                        // For Z-UP: g = [0, 0, -9.81], so:
                        // cable_force_dir = (g - a_des) / |g - a_des|
                        //                 = [-ax, -ay, -9.81 - az] / |...|
                        //
                        // This gives the AVERAGE cable direction (pointing down toward load)
                        let desired_cable_force = Vector3::new(
                            -load_mass * acc_clamped.x,
                            -load_mass * acc_clamped.y,
                            -load_mass * (acc_clamped.z + GRAVITY),
                        );

                        let force_mag = desired_cable_force.norm();
                        // Cable direction s_i points FROM quad TOWARD load
                        // At hover: cables point DOWN, so s = [0, 0, -1]
                        let cable_dir_avg = if force_mag > 0.01 {
                            desired_cable_force.normalize()
                        } else {
                            Vector3::new(0.0, 0.0, -1.0)  // Default: cables point down
                        };

                        // Allow more tilt for aggressive maneuvers
                        let max_tilt = match current_scenario {
                            Scenario::Figure8Fast => 0.8,        // ~53 degrees
                            Scenario::Figure8MediumPlus => 0.6,  // ~37 degrees
                            _ => 0.5,                            // ~30 degrees
                        };

                        let horiz = Vector3::new(cable_dir_avg.x, cable_dir_avg.y, 0.0);
                        let horiz_mag = horiz.norm();

                        let cable_dir_clamped = if horiz_mag > max_tilt {
                            let scale = max_tilt / horiz_mag;
                            let h = horiz * scale;
                            let z_sign = if cable_dir_avg.z < 0.0 { -1.0 } else { 1.0 };
                            let z = z_sign * (1.0 - h.norm_squared()).sqrt();
                            Vector3::new(h.x, h.y, z).normalize()
                        } else {
                            cable_dir_avg
                        };

                        // Tension per cable (equal distribution for simplicity)
                        let tension_base = (force_mag / num_quads as f64)
                            .clamp(ocp_params.tension_min + 1.0, ocp_params.tension_max);

                        // Distribute directions with slight fanning for stability
                        for i in 0..num_quads {
                            let attach = &dynamics.load.params.attachment_points[i];

                            // Small fanning based on attachment position
                            let fan_factor = 0.1;
                            let fan = Vector3::new(-attach.x * fan_factor, -attach.y * fan_factor, 0.0);
                            let dir = (cable_dir_clamped + fan).normalize();

                            desired_directions[i] = dir;
                            desired_tensions[i] = tension_base;
                        }
                    }

                    // Status output
                    if trajectory_history.len() % 200 == 0 && trajectory_history.len() > 0 {
                        let error = (target_position - pos).norm();
                        let mode = if mpc_success { "MPC" } else { "PID" };

                        // Calculate RMSE if tracking figure-8
                        let rmse = if !position_errors.is_empty() {
                            let sum_sq: f64 = position_errors.iter().map(|e| e * e).sum();
                            (sum_sq / position_errors.len() as f64).sqrt()
                        } else {
                            0.0
                        };

                        println!("[{}] {} t={:.1}s pos=[{:.2},{:.2},{:.2}] err={:.3}m vel={:.2}m/s RMSE={:.3}m",
                            current_scenario.name(), mode, sim_time,
                            pos.x, pos.y, pos.z, error, vel_mag, rmse);

                        // Show cable tensions (actual vs desired)
                        print!("  Tensions: ");
                        for i in 0..num_quads {
                            print!("T{}={:.1}/{:.1}N ", i, state.cables.cables[i].tension, desired_tensions[i]);
                        }
                        println!();

                        // Show cable directions
                        print!("  Directions: ");
                        for i in 0..num_quads {
                            let d = &state.cables.cables[i].direction;
                            let dd = &desired_directions[i];
                            print!("s{}=[{:.2},{:.2},{:.2}]/[{:.2},{:.2},{:.2}] ",
                                i, d.x, d.y, d.z, dd.x, dd.y, dd.z);
                        }
                        println!();
                    }
                }

                // === Apply control using Paper Architecture ===
                // Step 1: Build LoadKinematicState from current state
                // Note: acceleration/jerk are computed from dynamics, not stored in state
                let load_kinematic = LoadKinematicState {
                    position: state.load.position,
                    velocity: state.load.velocity,
                    acceleration: Vector3::zeros(), // Will use feedforward from MPC
                    jerk: Vector3::zeros(),
                    orientation: state.load.orientation,
                    angular_velocity: state.load.angular_velocity,
                    angular_acceleration: Vector3::zeros(),
                    angular_jerk: Vector3::zeros(),
                };

                // Step 2: Build CableKinematicStates from current cable states
                let cable_kinematics: Vec<CableKinematicState> = state.cables.cables.iter()
                    .enumerate()
                    .map(|(i, cable)| {
                        CableKinematicState {
                            // Use desired direction from MPC/fallback controller
                            direction: desired_directions[i],
                            angular_velocity: cable.angular_velocity,
                            angular_acceleration: cable.angular_acceleration,
                            angular_jerk: Vector3::zeros(), // Control input
                        }
                    })
                    .collect();

                // Step 3: Compute quadrotor trajectories using Kinematic Constraint (Eq. 5 + S1)
                let quad_trajectory_points = kinematic_constraint.all_quadrotor_trajectories(
                    &load_kinematic,
                    &cable_kinematics,
                );

                // Step 4: Build QuadrotorTrackerStates and References
                let quad_states: Vec<QuadrotorTrackerState> = state.quadrotors.iter()
                    .map(|q| QuadrotorTrackerState {
                        position: q.position,
                        velocity: q.velocity,
                        orientation: q.orientation,
                        angular_velocity: q.angular_velocity,
                    })
                    .collect();

                let quad_refs: Vec<QuadrotorTrajectoryRef> = quad_trajectory_points.iter()
                    .map(|traj| QuadrotorTrajectoryRef {
                        position: traj.position,
                        velocity: traj.velocity,
                        acceleration: traj.acceleration,
                        jerk: traj.jerk,
                    })
                    .collect();

                // Step 5: Compute external forces (cable tension)
                let external_forces: Vec<ExternalForce> = state.cables.cables.iter()
                    .map(|cable| ExternalForce::from_cable(cable.tension, &cable.direction))
                    .collect();

                // Step 6: Run QuadrotorTracker (Eq. 15) for each quadrotor
                let tracker_outputs = quadrotor_trackers.compute_all(
                    &quad_states,
                    &quad_refs,
                    &external_forces,
                    sim_dt,
                );

                // Step 7: Apply control inputs to cable dynamics
                // Paper architecture: MPC uses 3rd-order model with snap as control
                // Simulator uses 2nd-order model with jerk as control
                // Bridge: Use MPC's predicted angular_jerk state as simulator input
                // When MPC fails, fall back to PD-based control from tracker outputs
                let cable_inputs: Vec<CableInput> = if let (Some(ref mpc_ctrl), Some(ref next_state)) = (&mpc_controls, &mpc_next_state) {
                    // === MPC Success: Use angular_jerk from MPC's predicted next state ===
                    // MPC's 3rd-order model: control = angular_snap (r⃛ᵢ), state includes angular_jerk (r̈ᵢ)
                    // Simulator's 2nd-order model: control = angular_jerk (r̈ᵢ)
                    // We use states[1].angular_jerk which is the result after applying control[0]
                    (0..mpc_ctrl.cables.len()).map(|i| {
                        CableInput {
                            angular_jerk: next_state.cables[i].angular_jerk,
                            tension_acceleration: mpc_ctrl.cables[i].tension_acceleration,
                        }
                    }).collect()
                } else {
                    // === MPC Fallback: PD-based control from tracker outputs ===
                    let mut inputs = Vec::with_capacity(num_quads);
                    for i in 0..num_quads {
                        let cable = &state.cables.cables[i];
                        let tracker_output = &tracker_outputs[i];

                        // === Direction Control ===
                        // Desired thrust direction from tracker should match desired cable direction
                        // (thrust is opposite to cable direction)
                        let desired_cable_dir = -tracker_output.body_z_des;

                        // Normalize directions before computing error
                        let current_dir = if cable.direction.norm() > 1e-6 {
                            cable.direction.normalize()
                        } else {
                            Vector3::new(0.0, 0.0, -1.0)
                        };
                        let target_dir = if desired_cable_dir.norm() > 1e-6 {
                            desired_cable_dir.normalize()
                        } else {
                            Vector3::new(0.0, 0.0, -1.0)
                        };

                        // Use cross product for direction error (small angle approximation)
                        let dir_error = current_dir.cross(&target_dir);

                        // 3rd order PD cascade for cable direction (Eq. 3)
                        // More conservative gains for stability
                        let kp_dir = 15.0;   // Direction gain (reduced from 25)
                        let kd_dir = 8.0;    // Angular velocity gain (reduced from 10)
                        let ka_dir = 4.0;    // Angular acceleration gain (reduced from 5)

                        let omega_des = dir_error * kp_dir;
                        let omega_error = omega_des - cable.angular_velocity;
                        let alpha_des = omega_error * kd_dir;
                        let alpha_error = alpha_des - cable.angular_acceleration;
                        // Add damping on angular_acceleration
                        let angular_jerk = alpha_error * ka_dir - cable.angular_acceleration * 1.0;

                        let max_jerk = 50.0;  // Reduced from 100
                        let angular_jerk_clamped = if angular_jerk.norm() > max_jerk {
                            angular_jerk.normalize() * max_jerk
                        } else {
                            angular_jerk
                        };

                        // === Tension Control ===
                        // Desired tension from tracker thrust magnitude
                        // T_quad * cos(tilt_angle) ≈ T_cable (simplified)
                        let desired_tension = (tracker_output.thrust / num_quads as f64)
                            .clamp(ocp_params.tension_min + 0.5, ocp_params.tension_max);

                        // PD control for tension with more conservative gains
                        // to prevent oscillation and divergence
                        let kp_tension = 15.0;   // Position gain (was 50)
                        let kd_tension = 8.0;    // Derivative gain for damping

                        let tension_error = desired_tension - cable.tension;
                        let tension_rate_des = tension_error * kp_tension;
                        let tension_rate_error = tension_rate_des - cable.tension_rate;
                        // Add damping on tension_rate directly
                        let tension_accel = (tension_rate_error * kd_tension - cable.tension_rate * 2.0)
                            .clamp(-200.0, 200.0);  // Reduced from 500

                        inputs.push(CableInput {
                            angular_jerk: angular_jerk_clamped,
                            tension_acceleration: tension_accel,
                        });
                    }
                    inputs
                };

                let input = SystemInput {
                    cables: cable_inputs,
                    quadrotors: vec![Default::default(); num_quads],
                };

                state = dynamics.integrate(&state, &input, sim_dt);
                sim_time += sim_dt;
            }

            if trajectory_history.len() < 10000 {
                trajectory_history.push(state.load.position);
            }
        }

        // ==========================================
        // 7. Update Visualization
        // ==========================================
        let load_pos = state.load.position;
        load_viz.set_local_translation(to_viz_translation(&load_pos));

        let (roll, pitch, yaw) = state.load.orientation.euler_angles();
        let load_rot = na::UnitQuaternion::from_euler_angles(roll as f32, yaw as f32, pitch as f32);
        load_viz.set_local_rotation(load_rot);

        // Color based on tracking error
        if let Some(ref params) = figure8_params {
            let t = sim_time - scenario_start_time;
            let ref_pos = params.position(t);
            let error = (load_pos - ref_pos).norm();
            if error < 0.2 {
                load_viz.set_color(0.2, 0.8, 0.2); // Green
            } else if error < 0.5 {
                load_viz.set_color(0.8, 0.8, 0.2); // Yellow
            } else {
                load_viz.set_color(0.8, 0.2, 0.2); // Red
            }
        } else {
            load_viz.set_color(0.2, 0.4, 0.9);
        }

        // Update quadrotors and draw cables
        let cable_colors = [
            Point3::new(1.0, 0.3, 0.3),
            Point3::new(0.3, 1.0, 0.3),
            Point3::new(0.3, 0.3, 1.0),
            Point3::new(1.0, 1.0, 0.3),
        ];
        for (i, quad_viz) in quad_vizs.iter_mut().enumerate() {
            let quad_pos = state.quadrotors[i].position;
            quad_viz.set_local_translation(to_viz_translation(&quad_pos));

            let attach = dynamics.load.attachment_world(&state.load, i);
            let cable = &state.cables.cables[i];
            let tension_normalized = (cable.tension / 10.0).clamp(0.2, 1.0) as f32;

            window.draw_line(
                &to_viz(&attach),
                &to_viz(&quad_pos),
                &Point3::new(
                    cable_colors[i % 4].x * tension_normalized,
                    cable_colors[i % 4].y * tension_normalized,
                    cable_colors[i % 4].z * tension_normalized,
                ),
            );

            // Draw force vector
            let force_scale = 0.08;
            let force_vec = cable.tension * cable.direction * force_scale;
            let force_end = attach + force_vec;
            window.draw_line(
                &to_viz(&attach),
                &to_viz(&force_end),
                &Point3::new(1.0, 1.0, 0.0),
            );
        }

        // Draw target
        target_viz.set_local_translation(to_viz_translation(&target_position));

        // Draw reference trajectory for figure-8
        if let Some(ref params) = figure8_params {
            let n_points = 100;
            let period = params.period();
            for i in 1..n_points {
                let t1 = (i - 1) as f64 * period / n_points as f64;
                let t2 = i as f64 * period / n_points as f64;
                let p1 = params.position(t1);
                let p2 = params.position(t2);
                window.draw_line(
                    &to_viz(&p1),
                    &to_viz(&p2),
                    &Point3::new(0.5, 0.5, 0.5),
                );
            }
        }

        // Draw trajectory history
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

        // Draw ground
        draw_ground(&mut window, 0.0);

        // Draw coordinate axes
        let origin = Point3::origin();
        window.draw_line(&origin, &Point3::new(0.5, 0.0, 0.0), &Point3::new(1.0, 0.0, 0.0));
        window.draw_line(&origin, &Point3::new(0.0, 0.5, 0.0), &Point3::new(0.0, 0.0, 1.0));
        window.draw_line(&origin, &Point3::new(0.0, 0.0, 0.5), &Point3::new(0.0, 1.0, 0.0));
    }
}

fn draw_ground(window: &mut Window, y: f32) {
    let size = 5.0_f32;
    let step = 0.5_f32;
    let color = Point3::new(0.2, 0.2, 0.25);

    let mut x = -size;
    while x <= size {
        window.draw_line(&Point3::new(x, y, -size), &Point3::new(x, y, size), &color);
        window.draw_line(&Point3::new(-size, y, x), &Point3::new(size, y, x), &color);
        x += step;
    }
}
