//! Path constraints for the OCP
//!
//! Implements the inequality constraints from Eq. 8-12 of the paper:
//! - Eq. 8: Quadrotor thrust limits
//! - Eq. 10: Cable tautness (tension bounds)
//! - Eq. 11: Inter-quadrotor collision avoidance
//! - Eq. 12: Obstacle avoidance

use nalgebra::{Vector3, Matrix3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::config::{ConstraintConfig, Obstacle};
use crate::ocp::{OcpState, SystemParameters};

/// Result of constraint evaluation
#[derive(Debug, Clone)]
pub struct ConstraintEvaluation {
    /// Constraint values (negative = satisfied, positive = violated)
    pub values: Vec<f64>,
    /// Names for debugging
    pub names: Vec<String>,
    /// Whether all constraints are satisfied
    pub all_satisfied: bool,
    /// Maximum violation (0 if all satisfied)
    pub max_violation: f64,
}

impl ConstraintEvaluation {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            names: Vec::new(),
            all_satisfied: true,
            max_violation: 0.0,
        }
    }

    pub fn add(&mut self, name: &str, value: f64) {
        self.names.push(name.to_string());
        self.values.push(value);
        if value > 0.0 {
            self.all_satisfied = false;
            self.max_violation = self.max_violation.max(value);
        }
    }
}

impl Default for ConstraintEvaluation {
    fn default() -> Self {
        Self::new()
    }
}

/// Constraint evaluator for the OCP
pub struct ConstraintEvaluator {
    config: ConstraintConfig,
    system: SystemParameters,
    obstacles: Vec<Obstacle>,
}

impl ConstraintEvaluator {
    pub fn new(
        config: ConstraintConfig,
        system: SystemParameters,
        obstacles: Vec<Obstacle>,
    ) -> Self {
        Self {
            config,
            system,
            obstacles,
        }
    }

    /// Evaluate all path constraints at a given state
    pub fn evaluate(&self, state: &OcpState) -> ConstraintEvaluation {
        let mut eval = ConstraintEvaluation::new();

        // Compute quadrotor positions from kinematic constraint
        let quad_positions = self.compute_quadrotor_positions(state);

        // Eq. 8: Thrust bounds (computed from cable states)
        self.evaluate_thrust_constraints(state, &quad_positions, &mut eval);

        // Eq. 10: Cable tension bounds
        self.evaluate_tension_constraints(state, &mut eval);

        // Eq. 11: Inter-quadrotor collision avoidance
        self.evaluate_collision_constraints(&quad_positions, &mut eval);

        // Eq. 12: Obstacle avoidance
        self.evaluate_obstacle_constraints(state, &quad_positions, &mut eval);

        eval
    }

    /// Compute quadrotor positions from load state using kinematic constraint (Eq. 5)
    ///
    /// pᵢ = p + R(q)ρᵢ - lᵢsᵢ
    fn compute_quadrotor_positions(&self, state: &OcpState) -> Vec<Vector3<f64>> {
        let n = state.cables.len();
        let mut positions = Vec::with_capacity(n);

        let rotation = state.load_orientation.to_rotation_matrix();

        for i in 0..n {
            let rho = Vector3::new(
                self.system.attachment_points[i][0],
                self.system.attachment_points[i][1],
                self.system.attachment_points[i][2],
            );
            let cable_length = self.system.cable_lengths[i];
            let cable_dir = state.cables[i].direction;

            // Eq. 5: pᵢ = p + R(q)ρᵢ - lᵢsᵢ
            let pos = state.load_position + rotation * rho - cable_length * cable_dir;
            positions.push(pos);
        }

        positions
    }

    /// Evaluate thrust constraints (Eq. 8)
    ///
    /// The required thrust depends on cable tension and direction.
    /// For near-vertical cables, thrust ≈ weight + tension
    fn evaluate_thrust_constraints(
        &self,
        state: &OcpState,
        _quad_positions: &[Vector3<f64>],
        eval: &mut ConstraintEvaluation,
    ) {
        for (i, cable) in state.cables.iter().enumerate() {
            let quad_mass = self.system.quadrotor_masses[i];
            let g = self.system.gravity;

            // Approximate thrust from cable tension and direction
            // The actual thrust depends on quadrotor dynamics, but we can
            // estimate a lower bound
            let cable_force_z = cable.tension * cable.direction.z;
            let required_thrust = quad_mass * g - cable_force_z;

            // T_min ≤ T ≤ T_max
            let thrust_min_violation = self.config.thrust_min - required_thrust;
            let thrust_max_violation = required_thrust - self.config.thrust_max;

            eval.add(&format!("thrust_min_{}", i), thrust_min_violation);
            eval.add(&format!("thrust_max_{}", i), thrust_max_violation);
        }
    }

    /// Evaluate cable tension constraints (Eq. 10)
    ///
    /// t_min ≤ tᵢ ≤ t_max for all cables
    fn evaluate_tension_constraints(
        &self,
        state: &OcpState,
        eval: &mut ConstraintEvaluation,
    ) {
        for (i, cable) in state.cables.iter().enumerate() {
            // t_min ≤ t
            let tension_min_violation = self.config.tension_min - cable.tension;
            // t ≤ t_max
            let tension_max_violation = cable.tension - self.config.tension_max;

            eval.add(&format!("tension_min_{}", i), tension_min_violation);
            eval.add(&format!("tension_max_{}", i), tension_max_violation);
        }
    }

    /// Evaluate inter-quadrotor collision avoidance (Eq. 11)
    ///
    /// ‖pᵢ - pⱼ‖ ≥ d_min for all pairs i ≠ j
    fn evaluate_collision_constraints(
        &self,
        quad_positions: &[Vector3<f64>],
        eval: &mut ConstraintEvaluation,
    ) {
        let n = quad_positions.len();
        let d_min = self.config.inter_quad_distance_min;

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = (quad_positions[i] - quad_positions[j]).norm();
                // Constraint: d_min² - ‖pᵢ - pⱼ‖² ≤ 0
                let violation = d_min - dist;
                eval.add(&format!("collision_{}_{}", i, j), violation);
            }
        }
    }

    /// Evaluate obstacle avoidance constraints (Eq. 12)
    ///
    /// For each obstacle: d²_min ≤ (p - p_o)ᵀ C (p - p_o)
    fn evaluate_obstacle_constraints(
        &self,
        state: &OcpState,
        quad_positions: &[Vector3<f64>],
        eval: &mut ConstraintEvaluation,
    ) {
        // Check load position
        for (obs_idx, obstacle) in self.obstacles.iter().enumerate() {
            let delta = state.load_position - obstacle.center;
            let shape = Matrix3::from_row_slice(&[
                obstacle.shape[(0, 0)], obstacle.shape[(0, 1)], obstacle.shape[(0, 2)],
                obstacle.shape[(1, 0)], obstacle.shape[(1, 1)], obstacle.shape[(1, 2)],
                obstacle.shape[(2, 0)], obstacle.shape[(2, 1)], obstacle.shape[(2, 2)],
            ]);
            let dist_sq = delta.dot(&(shape * delta));
            let safe_dist_sq = obstacle.safe_distance * obstacle.safe_distance;

            // Constraint: d²_min - dist² ≤ 0
            let violation = safe_dist_sq - dist_sq;
            eval.add(&format!("obstacle_load_{}", obs_idx), violation);
        }

        // Check each quadrotor position
        for (quad_idx, pos) in quad_positions.iter().enumerate() {
            for (obs_idx, obstacle) in self.obstacles.iter().enumerate() {
                let delta = pos - obstacle.center;
                let shape = Matrix3::from_row_slice(&[
                    obstacle.shape[(0, 0)], obstacle.shape[(0, 1)], obstacle.shape[(0, 2)],
                    obstacle.shape[(1, 0)], obstacle.shape[(1, 1)], obstacle.shape[(1, 2)],
                    obstacle.shape[(2, 0)], obstacle.shape[(2, 1)], obstacle.shape[(2, 2)],
                ]);
                let dist_sq = delta.dot(&(shape * delta));
                let safe_dist_sq = obstacle.safe_distance * obstacle.safe_distance;

                let violation = safe_dist_sq - dist_sq;
                eval.add(&format!("obstacle_quad_{}_{}", quad_idx, obs_idx), violation);
            }
        }
    }

    /// Get constraint dimensions for OCP setup
    pub fn num_constraints(&self) -> ConstraintDimensions {
        let n = self.system.quadrotor_masses.len();
        let num_obstacles = self.obstacles.len();

        ConstraintDimensions {
            // Thrust bounds: 2 per quadrotor (min and max)
            thrust: 2 * n,
            // Tension bounds: 2 per cable
            tension: 2 * n,
            // Collision: n*(n-1)/2 pairs
            collision: n * (n - 1) / 2,
            // Obstacles: (n+1) * num_obstacles (load + each quad)
            obstacle: (n + 1) * num_obstacles,
        }
    }
}

/// Constraint dimensions for the OCP
#[derive(Debug, Clone, Copy)]
pub struct ConstraintDimensions {
    pub thrust: usize,
    pub tension: usize,
    pub collision: usize,
    pub obstacle: usize,
}

impl ConstraintDimensions {
    pub fn total(&self) -> usize {
        self.thrust + self.tension + self.collision + self.obstacle
    }
}

/// Control input bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlBounds {
    /// Lower bounds for control inputs
    pub lower: Vec<f64>,
    /// Upper bounds for control inputs
    pub upper: Vec<f64>,
}

impl ControlBounds {
    /// Create control bounds for n cables
    pub fn new(n: usize, config: &ConstraintConfig) -> Self {
        let mut lower = Vec::with_capacity(4 * n);
        let mut upper = Vec::with_capacity(4 * n);

        for _ in 0..n {
            // Angular jerk bounds (3 components)
            lower.extend([-config.max_angular_jerk; 3]);
            upper.extend([config.max_angular_jerk; 3]);

            // Tension acceleration bounds
            lower.push(-config.max_tension_acceleration);
            upper.push(config.max_tension_acceleration);
        }

        Self { lower, upper }
    }
}

/// State bounds (soft constraints on state variables)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateBounds {
    /// Lower bounds for state
    pub lower: Vec<f64>,
    /// Upper bounds for state
    pub upper: Vec<f64>,
}

impl StateBounds {
    /// Create state bounds for n cables
    pub fn new(n: usize, config: &ConstraintConfig) -> Self {
        let state_dim = 13 + 7 * n;
        let mut lower = vec![f64::NEG_INFINITY; state_dim];
        let mut upper = vec![f64::INFINITY; state_dim];

        // Cable tension bounds are in the state
        for i in 0..n {
            let tension_idx = 13 + 7 * i + 6; // tension is last element of each cable state
            lower[tension_idx] = config.tension_min;
            upper[tension_idx] = config.tension_max;
        }

        Self { lower, upper }
    }
}

/// Workspace bounds (optional position limits)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceBounds {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub z_min: f64,
    pub z_max: f64,
}

impl Default for WorkspaceBounds {
    fn default() -> Self {
        Self {
            x_min: -100.0,
            x_max: 100.0,
            y_min: -100.0,
            y_max: 100.0,
            z_min: 0.0,    // Ground
            z_max: 50.0,   // Ceiling
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocp::{OcpState, CableState};

    fn create_test_state(n: usize) -> OcpState {
        let mut state = OcpState::new(n);
        state.load_position = Vector3::new(0.0, 0.0, 2.0);
        // Set reasonable tensions
        for cable in &mut state.cables {
            cable.tension = 5.0;
            cable.direction = Vector3::new(0.0, 0.0, -1.0);
        }
        state
    }

    fn create_test_system(n: usize) -> SystemParameters {
        SystemParameters::new(n)
    }

    #[test]
    fn test_tension_constraints_satisfied() {
        let config = ConstraintConfig::default();
        let system = create_test_system(3);
        let evaluator = ConstraintEvaluator::new(config, system, Vec::new());

        let state = create_test_state(3);
        let eval = evaluator.evaluate(&state);

        // Check tension constraints are satisfied
        for name in &eval.names {
            if name.starts_with("tension_") {
                let idx = eval.names.iter().position(|n| n == name).unwrap();
                assert!(
                    eval.values[idx] <= 0.0,
                    "Constraint {} violated: {}",
                    name,
                    eval.values[idx]
                );
            }
        }
    }

    #[test]
    fn test_tension_constraint_violation() {
        let config = ConstraintConfig {
            tension_min: 10.0, // Higher than our state's 5.0
            ..ConstraintConfig::default()
        };
        let system = create_test_system(3);
        let evaluator = ConstraintEvaluator::new(config, system, Vec::new());

        let state = create_test_state(3);
        let eval = evaluator.evaluate(&state);

        // Should detect violation
        assert!(!eval.all_satisfied);
        assert!(eval.max_violation > 0.0);
    }

    #[test]
    fn test_collision_avoidance() {
        let config = ConstraintConfig::default();
        let system = create_test_system(3);
        let evaluator = ConstraintEvaluator::new(config, system, Vec::new());

        let state = create_test_state(3);
        let eval = evaluator.evaluate(&state);

        // Count collision constraints
        let collision_count = eval.names.iter()
            .filter(|n| n.starts_with("collision_"))
            .count();

        // For 3 quadrotors: C(3,2) = 3 pairs
        assert_eq!(collision_count, 3);
    }

    #[test]
    fn test_obstacle_avoidance() {
        let config = ConstraintConfig::default();
        let system = create_test_system(3);
        let obstacle = Obstacle::sphere(Vector3::new(0.0, 0.0, 2.0), 0.5);
        let evaluator = ConstraintEvaluator::new(config, system, vec![obstacle]);

        let state = create_test_state(3);
        let eval = evaluator.evaluate(&state);

        // Should detect obstacle at load position
        let load_obstacle_violation = eval.names.iter()
            .enumerate()
            .find(|(_, n)| *n == "obstacle_load_0")
            .map(|(i, _)| eval.values[i])
            .unwrap();

        // Load is at obstacle center, so should be violated
        assert!(load_obstacle_violation > 0.0);
    }

    #[test]
    fn test_constraint_dimensions() {
        let config = ConstraintConfig::default();
        let system = create_test_system(4);
        let evaluator = ConstraintEvaluator::new(config, system, vec![
            Obstacle::sphere(Vector3::zeros(), 1.0),
            Obstacle::sphere(Vector3::new(5.0, 0.0, 0.0), 1.0),
        ]);

        let dims = evaluator.num_constraints();

        assert_eq!(dims.thrust, 8);      // 2 * 4 quads
        assert_eq!(dims.tension, 8);     // 2 * 4 cables
        assert_eq!(dims.collision, 6);   // C(4,2) = 6 pairs
        assert_eq!(dims.obstacle, 10);   // (4+1) * 2 obstacles
        assert_eq!(dims.total(), 32);
    }

    #[test]
    fn test_control_bounds() {
        let config = ConstraintConfig {
            max_angular_jerk: 100.0,
            max_tension_acceleration: 500.0,
            ..ConstraintConfig::default()
        };

        let bounds = ControlBounds::new(3, &config);

        assert_eq!(bounds.lower.len(), 12); // 4 * 3 cables
        assert_eq!(bounds.upper.len(), 12);

        // Check angular jerk bounds
        assert_eq!(bounds.lower[0], -100.0);
        assert_eq!(bounds.upper[0], 100.0);

        // Check tension acceleration bounds
        assert_eq!(bounds.lower[3], -500.0);
        assert_eq!(bounds.upper[3], 500.0);
    }
}
