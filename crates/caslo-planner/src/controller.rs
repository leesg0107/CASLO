//! Receding-Horizon Motion Planner
//!
//! Main interface for the online kinodynamic motion planner.
//! Runs at 10 Hz and provides reference trajectories to the
//! onboard tracking controllers.

use std::collections::VecDeque;
use std::time::Instant;

use nalgebra::{Vector3, UnitQuaternion};
use thiserror::Error;

use crate::config::{PlannerConfig, Obstacle};
use crate::ocp::{OcpDefinition, OcpState, ReferencePoint, SystemParameters};
use crate::solver::{AcadosSolver, SolverBuilder, SolverError, SolveOptions, SolveStatistics};
use crate::trajectory::{PlannedTrajectory, TrackingReference};
use crate::integration::{generate_hover_reference, generate_linear_trajectory};

/// Motion planner errors
#[derive(Debug, Error)]
pub enum PlannerError {
    #[error("Solver error: {0}")]
    SolverError(#[from] SolverError),
    #[error("No valid trajectory available")]
    NoValidTrajectory,
    #[error("Target unreachable")]
    TargetUnreachable,
    #[error("Planning timeout")]
    Timeout,
    #[error("Invalid goal: {0}")]
    InvalidGoal(String),
}

/// Motion planner state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannerState {
    /// Waiting for initial state
    Idle,
    /// Planning in progress
    Planning,
    /// Following a valid trajectory
    Executing,
    /// Reached the goal
    GoalReached,
    /// Error state
    Error,
}

/// Goal specification for the planner
#[derive(Debug, Clone)]
pub enum PlannerGoal {
    /// Hover at a fixed position
    Hover {
        position: Vector3<f64>,
        orientation: UnitQuaternion<f64>,
    },
    /// Move to a waypoint
    Waypoint {
        position: Vector3<f64>,
        orientation: UnitQuaternion<f64>,
        velocity: Vector3<f64>,
    },
    /// Follow a pre-defined path
    Path {
        waypoints: Vec<ReferencePoint>,
    },
    /// Track a dynamic target with sampled positions
    /// (sampled at regular intervals over the horizon)
    DynamicSampled {
        /// Pre-sampled target positions over the horizon
        sampled_positions: Vec<Vector3<f64>>,
        /// Pre-sampled target velocities over the horizon
        sampled_velocities: Vec<Vector3<f64>>,
    },
}

/// Online receding-horizon motion planner
///
/// This is the main interface for trajectory planning. It:
/// - Runs the OCP solver at 10 Hz
/// - Maintains a valid trajectory for the tracking controller
/// - Handles goal updates and replanning
/// - Manages warm-starting for real-time performance
pub struct MotionPlanner {
    /// Solver instance
    solver: AcadosSolver,
    /// Planner configuration
    config: PlannerConfig,
    /// System parameters
    system: SystemParameters,
    /// Current planner state
    state: PlannerState,
    /// Current goal
    goal: Option<PlannerGoal>,
    /// Current trajectory
    trajectory: Option<PlannedTrajectory>,
    /// History of planned trajectories (for debugging)
    trajectory_history: VecDeque<PlannedTrajectory>,
    /// Maximum history size
    max_history: usize,
    /// Obstacles in the environment
    obstacles: Vec<Obstacle>,
    /// Current time
    current_time: f64,
    /// Last planning timestamp
    last_plan_time: Option<Instant>,
    /// Statistics from last solve
    last_stats: SolveStatistics,
    /// Number of quadrotors
    num_quadrotors: usize,
}

impl MotionPlanner {
    /// Create a new motion planner
    pub fn new(
        num_quadrotors: usize,
        config: PlannerConfig,
        system: SystemParameters,
    ) -> Result<Self, PlannerError> {
        let solver = SolverBuilder::new(num_quadrotors)
            .with_config(config.clone())
            .with_system(system.clone())
            .build()?;

        Ok(Self {
            solver,
            config,
            system,
            state: PlannerState::Idle,
            goal: None,
            trajectory: None,
            trajectory_history: VecDeque::new(),
            max_history: 10,
            obstacles: Vec::new(),
            current_time: 0.0,
            last_plan_time: None,
            last_stats: SolveStatistics::default(),
            num_quadrotors,
        })
    }

    /// Set the goal for the planner
    pub fn set_goal(&mut self, goal: PlannerGoal) {
        self.goal = Some(goal);
        self.state = PlannerState::Planning;
    }

    /// Set hover goal at current position
    pub fn hover_at(&mut self, position: Vector3<f64>, orientation: UnitQuaternion<f64>) {
        self.set_goal(PlannerGoal::Hover { position, orientation });
    }

    /// Set waypoint goal
    pub fn go_to(&mut self, position: Vector3<f64>, orientation: UnitQuaternion<f64>) {
        self.set_goal(PlannerGoal::Waypoint {
            position,
            orientation,
            velocity: Vector3::zeros(),
        });
    }

    /// Add an obstacle to avoid
    pub fn add_obstacle(&mut self, obstacle: Obstacle) {
        self.obstacles.push(obstacle);
    }

    /// Clear all obstacles
    pub fn clear_obstacles(&mut self) {
        self.obstacles.clear();
    }

    /// Update the planner with current state
    ///
    /// This is the main update function called at 10 Hz.
    /// Returns the reference for the tracking controller.
    pub fn update(
        &mut self,
        current_state: &OcpState,
        dt: f64,
    ) -> Result<Option<TrackingReference>, PlannerError> {
        self.current_time += dt;

        // Check if we need to replan
        let should_replan = self.should_replan();

        if should_replan && self.goal.is_some() {
            self.plan(current_state)?;
        }

        // Get reference from current trajectory
        if let Some(ref trajectory) = self.trajectory {
            if trajectory.is_valid {
                let reference = trajectory.get_tracking_reference(self.current_time);
                return Ok(reference);
            }
        }

        // No valid trajectory - return None or error
        if self.state == PlannerState::Error {
            return Err(PlannerError::NoValidTrajectory);
        }

        Ok(None)
    }

    /// Plan a new trajectory
    fn plan(&mut self, current_state: &OcpState) -> Result<(), PlannerError> {
        let start_time = Instant::now();

        // Generate reference based on goal
        let reference = self.generate_reference(current_state)?;

        // Solve OCP
        let options = SolveOptions {
            max_sqp_iter: self.config.solver.max_iterations,
            warm_start: self.config.solver.warm_start,
            timeout_ms: Some(self.config.solver.max_cpu_time_ms),
            ..Default::default()
        };

        let result = self.solver.solve(current_state, &reference, &options);

        match result {
            Ok(mut trajectory) => {
                trajectory.generated_at = self.current_time;

                // Store in history
                if self.trajectory_history.len() >= self.max_history {
                    self.trajectory_history.pop_front();
                }
                self.trajectory_history.push_back(trajectory.clone());

                self.trajectory = Some(trajectory);
                self.state = PlannerState::Executing;
                self.last_stats = self.solver.statistics().clone();
                self.last_plan_time = Some(start_time);

                Ok(())
            }
            Err(e) => {
                self.state = PlannerState::Error;
                Err(PlannerError::SolverError(e))
            }
        }
    }

    /// Generate reference trajectory based on current goal
    fn generate_reference(
        &self,
        current_state: &OcpState,
    ) -> Result<Vec<ReferencePoint>, PlannerError> {
        let num_points = self.config.horizon.num_nodes + 1;

        match &self.goal {
            Some(PlannerGoal::Hover { position, orientation }) => {
                Ok(generate_hover_reference(*position, *orientation, num_points))
            }
            Some(PlannerGoal::Waypoint { position, orientation, velocity: _ }) => {
                Ok(generate_linear_trajectory(
                    current_state.load_position,
                    *position,
                    *orientation,
                    self.config.horizon.horizon_time,
                    num_points,
                ))
            }
            Some(PlannerGoal::Path { waypoints }) => {
                if waypoints.len() >= num_points {
                    Ok(waypoints[..num_points].to_vec())
                } else {
                    // Extend with last point
                    let mut extended = waypoints.clone();
                    let last = waypoints.last().cloned().unwrap_or_default();
                    while extended.len() < num_points {
                        extended.push(last.clone());
                    }
                    Ok(extended)
                }
            }
            Some(PlannerGoal::DynamicSampled { sampled_positions, sampled_velocities }) => {
                let mut reference = Vec::with_capacity(num_points);

                for i in 0..num_points {
                    let pos = sampled_positions.get(i)
                        .copied()
                        .unwrap_or_else(|| sampled_positions.last().copied().unwrap_or(Vector3::zeros()));
                    let vel = sampled_velocities.get(i)
                        .copied()
                        .unwrap_or_else(|| sampled_velocities.last().copied().unwrap_or(Vector3::zeros()));

                    reference.push(ReferencePoint {
                        position: pos,
                        velocity: vel,
                        orientation: UnitQuaternion::identity(),
                        angular_velocity: Vector3::zeros(),
                    });
                }
                Ok(reference)
            }
            None => {
                // No goal - hover at current position
                Ok(generate_hover_reference(
                    current_state.load_position,
                    current_state.load_orientation,
                    num_points,
                ))
            }
        }
    }

    /// Check if replanning is needed
    fn should_replan(&self) -> bool {
        // Always replan if no trajectory
        if self.trajectory.is_none() {
            return true;
        }

        // Check timing
        let plan_period = 1.0 / self.config.solver.frequency;
        if let Some(last_time) = self.last_plan_time {
            if last_time.elapsed().as_secs_f64() < plan_period {
                return false;
            }
        }

        // Check if trajectory is stale
        if let Some(ref trajectory) = self.trajectory {
            if trajectory.is_stale(self.current_time, plan_period * 2.0) {
                return true;
            }
        }

        true
    }

    /// Get current planner state
    pub fn state(&self) -> PlannerState {
        self.state
    }

    /// Get current trajectory (if valid)
    pub fn trajectory(&self) -> Option<&PlannedTrajectory> {
        self.trajectory.as_ref()
    }

    /// Get last solve statistics
    pub fn statistics(&self) -> &SolveStatistics {
        &self.last_stats
    }

    /// Check if goal has been reached
    pub fn is_goal_reached(&self, current_state: &OcpState, tolerance: f64) -> bool {
        match &self.goal {
            Some(PlannerGoal::Hover { position, .. }) => {
                (current_state.load_position - position).norm() < tolerance
            }
            Some(PlannerGoal::Waypoint { position, .. }) => {
                (current_state.load_position - position).norm() < tolerance
            }
            _ => false,
        }
    }

    /// Emergency stop - generate trajectory to current position
    pub fn emergency_stop(&mut self, current_state: &OcpState) {
        self.hover_at(current_state.load_position, current_state.load_orientation);
        self.solver.reset_warm_start();
    }

    /// Get time remaining in current trajectory
    pub fn remaining_time(&self) -> f64 {
        self.trajectory
            .as_ref()
            .map(|t| t.remaining_duration(self.current_time))
            .unwrap_or(0.0)
    }

    /// Reset the planner
    pub fn reset(&mut self) {
        self.state = PlannerState::Idle;
        self.goal = None;
        self.trajectory = None;
        self.trajectory_history.clear();
        self.current_time = 0.0;
        self.last_plan_time = None;
        self.solver.reset_warm_start();
    }

    /// Update system parameters
    pub fn set_system_parameters(&mut self, params: SystemParameters) -> Result<(), PlannerError> {
        self.system = params.clone();
        self.solver.set_parameters(&params)?;
        Ok(())
    }
}

/// Simplified interface for common planning tasks
pub struct SimplePlanner {
    planner: MotionPlanner,
}

impl SimplePlanner {
    /// Create a simple planner with default configuration
    pub fn new(num_quadrotors: usize) -> Result<Self, PlannerError> {
        let config = PlannerConfig::default();
        let system = SystemParameters::new(num_quadrotors);
        let planner = MotionPlanner::new(num_quadrotors, config, system)?;
        Ok(Self { planner })
    }

    /// Plan a trajectory from current state to goal position
    pub fn plan_to(
        &mut self,
        current_state: &OcpState,
        goal_position: Vector3<f64>,
    ) -> Result<PlannedTrajectory, PlannerError> {
        self.planner.go_to(goal_position, current_state.load_orientation);
        self.planner.update(current_state, 0.0)?;

        self.planner
            .trajectory()
            .cloned()
            .ok_or(PlannerError::NoValidTrajectory)
    }

    /// Get hover trajectory at position
    pub fn hover_trajectory(
        &mut self,
        position: Vector3<f64>,
        orientation: UnitQuaternion<f64>,
    ) -> PlannedTrajectory {
        PlannedTrajectory::hover(
            position,
            orientation,
            self.planner.num_quadrotors,
            self.planner.config.horizon.horizon_time,
            self.planner.config.horizon.num_nodes,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_state(n: usize) -> OcpState {
        let mut state = OcpState::new(n);
        state.load_position = Vector3::new(0.0, 0.0, 2.0);
        state
    }

    #[test]
    fn test_planner_creation() {
        let config = PlannerConfig::default();
        let system = SystemParameters::new(3);
        let planner = MotionPlanner::new(3, config, system);
        assert!(planner.is_ok());
    }

    #[test]
    fn test_planner_state_transitions() {
        let config = PlannerConfig::default();
        let system = SystemParameters::new(3);
        let mut planner = MotionPlanner::new(3, config, system).unwrap();

        assert_eq!(planner.state(), PlannerState::Idle);

        planner.hover_at(Vector3::new(0.0, 0.0, 2.0), UnitQuaternion::identity());
        assert_eq!(planner.state(), PlannerState::Planning);
    }

    #[test]
    fn test_goal_reached() {
        let config = PlannerConfig::default();
        let system = SystemParameters::new(3);
        let mut planner = MotionPlanner::new(3, config, system).unwrap();

        let target = Vector3::new(1.0, 0.0, 2.0);
        planner.go_to(target, UnitQuaternion::identity());

        // State at target
        let mut state = create_test_state(3);
        state.load_position = target;

        assert!(planner.is_goal_reached(&state, 0.1));
    }

    #[test]
    fn test_simple_planner() {
        let simple = SimplePlanner::new(3);
        assert!(simple.is_ok());
    }

    #[test]
    fn test_hover_trajectory_generation() {
        let mut simple = SimplePlanner::new(3).unwrap();
        let pos = Vector3::new(1.0, 2.0, 3.0);
        let ori = UnitQuaternion::identity();

        let traj = simple.hover_trajectory(pos, ori);
        assert!(traj.is_valid);
        assert!((traj.states[0].load_position - pos).norm() < 1e-10);
    }

    #[test]
    fn test_emergency_stop() {
        let config = PlannerConfig::default();
        let system = SystemParameters::new(3);
        let mut planner = MotionPlanner::new(3, config, system).unwrap();

        let state = create_test_state(3);
        planner.go_to(Vector3::new(10.0, 0.0, 2.0), UnitQuaternion::identity());
        planner.emergency_stop(&state);

        // Goal should now be hover at current position
        assert!(planner.is_goal_reached(&state, 0.1));
    }

    #[test]
    fn test_replanning_timing() {
        let config = PlannerConfig {
            solver: crate::config::SolverConfig {
                frequency: 10.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let system = SystemParameters::new(3);
        let mut planner = MotionPlanner::new(3, config, system).unwrap();

        // First update should trigger planning
        assert!(planner.should_replan());
    }
}
