//! Planner configuration
//!
//! Configuration parameters for the online kinodynamic motion planner.

use nalgebra::{Vector3, Matrix3};
use serde::{Deserialize, Serialize};

/// Main planner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerConfig {
    /// OCP horizon configuration
    pub horizon: HorizonConfig,
    /// Cost function weights
    pub weights: CostWeights,
    /// Constraint bounds
    pub constraints: ConstraintConfig,
    /// Solver configuration
    pub solver: SolverConfig,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            horizon: HorizonConfig::default(),
            weights: CostWeights::default(),
            constraints: ConstraintConfig::default(),
            solver: SolverConfig::default(),
        }
    }
}

/// Horizon configuration for OCP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonConfig {
    /// Number of shooting nodes (N in paper)
    pub num_nodes: usize,
    /// Total horizon time [s]
    pub horizon_time: f64,
    /// Whether to use non-uniform discretization
    /// (intervals increase along horizon for better near-term accuracy)
    pub non_uniform: bool,
}

impl Default for HorizonConfig {
    fn default() -> Self {
        Self {
            num_nodes: 20,        // Paper uses N=20
            horizon_time: 2.0,    // Paper uses 2s horizon
            non_uniform: true,    // Paper uses non-equidistant segments
        }
    }
}

impl HorizonConfig {
    /// Compute time intervals for each segment
    ///
    /// If non_uniform is true, intervals linearly increase along the horizon
    /// for higher fidelity in the near future.
    pub fn compute_intervals(&self) -> Vec<f64> {
        if !self.non_uniform {
            let dt = self.horizon_time / self.num_nodes as f64;
            return vec![dt; self.num_nodes];
        }

        // Linear increase: dt_k = dt_0 + k * delta
        // Sum constraint: Σ dt_k = horizon_time
        // dt_0 + (dt_0 + delta) + ... + (dt_0 + (N-1)*delta) = horizon_time
        // N*dt_0 + delta * (0 + 1 + ... + N-1) = horizon_time
        // N*dt_0 + delta * N*(N-1)/2 = horizon_time

        let n = self.num_nodes as f64;
        // Choose dt_0 such that last interval is ~2x first interval
        let ratio = 2.0;
        let dt_0 = self.horizon_time / (n + (ratio - 1.0) * n * (n - 1.0) / (2.0 * (n - 1.0)));
        let delta = (ratio - 1.0) * dt_0 / (n - 1.0);

        (0..self.num_nodes)
            .map(|k| dt_0 + k as f64 * delta)
            .collect()
    }
}

/// Cost function weights for OCP (Eq. 6 in paper)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostWeights {
    // Load tracking weights
    /// Position tracking weight
    pub w_position: Vector3<f64>,
    /// Velocity tracking weight
    pub w_velocity: Vector3<f64>,
    /// Orientation tracking weight (axis-angle error)
    pub w_orientation: Vector3<f64>,
    /// Angular velocity tracking weight
    pub w_angular_velocity: Vector3<f64>,

    // Control smoothness weights
    /// Cable angular jerk weight (γ)
    pub w_cable_angular_jerk: f64,
    /// Cable tension acceleration weight (λ)
    pub w_tension_acceleration: f64,

    // Terminal cost scaling
    pub terminal_scale: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            // Position tracking (higher weight on z for altitude)
            w_position: Vector3::new(10.0, 10.0, 20.0),
            // Velocity tracking
            w_velocity: Vector3::new(1.0, 1.0, 2.0),
            // Orientation tracking
            w_orientation: Vector3::new(5.0, 5.0, 5.0),
            // Angular velocity
            w_angular_velocity: Vector3::new(0.5, 0.5, 0.5),
            // Smoothness (small weights to not dominate tracking)
            w_cable_angular_jerk: 0.01,
            w_tension_acceleration: 0.001,
            // Terminal cost (typically 10x stage cost)
            terminal_scale: 10.0,
        }
    }
}

/// Constraint bounds configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintConfig {
    // Thrust constraints (Eq. 8)
    /// Minimum thrust per quadrotor [N]
    pub thrust_min: f64,
    /// Maximum thrust per quadrotor [N]
    pub thrust_max: f64,

    // Cable constraints (Eq. 10)
    /// Minimum cable tension [N] (must be > 0 for tautness)
    pub tension_min: f64,
    /// Maximum cable tension [N]
    pub tension_max: f64,

    // Collision avoidance (Eq. 11)
    /// Minimum distance between quadrotors [m]
    pub inter_quad_distance_min: f64,

    // Control input bounds
    /// Maximum cable angular jerk [rad/s³]
    pub max_angular_jerk: f64,
    /// Maximum tension acceleration [N/s²]
    pub max_tension_acceleration: f64,
}

impl Default for ConstraintConfig {
    fn default() -> Self {
        Self {
            // Paper: 0.6 kg quadrotor with max 20N thrust
            thrust_min: 0.0,
            thrust_max: 20.0,
            // Cable must be taut
            tension_min: 0.5,      // Small positive value
            tension_max: 50.0,     // Reasonable upper bound
            // Safe distance between quadrotors
            inter_quad_distance_min: 0.8,
            // Control bounds (tuned for smooth trajectories)
            max_angular_jerk: 100.0,
            max_tension_acceleration: 500.0,
        }
    }
}

/// Obstacle definition for avoidance (Eq. 12)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obstacle {
    /// Center position of no-fly zone
    pub center: Vector3<f64>,
    /// Shape matrix C (diagonal for ellipsoid)
    /// d²_min ≤ (p - p_o)^T C (p - p_o)
    pub shape: Matrix3<f64>,
    /// Minimum safe distance from center
    pub safe_distance: f64,
}

impl Obstacle {
    /// Create a spherical obstacle
    pub fn sphere(center: Vector3<f64>, radius: f64) -> Self {
        Self {
            center,
            shape: Matrix3::identity(),
            safe_distance: radius,
        }
    }

    /// Create a vertical cylinder (for walls)
    pub fn vertical_cylinder(center: Vector3<f64>, radius: f64) -> Self {
        Self {
            center,
            // Zero weight on z means infinite extent in z
            shape: Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 0.0)),
            safe_distance: radius,
        }
    }

    /// Create a horizontal cylinder (for horizontal gaps)
    pub fn horizontal_cylinder(center: Vector3<f64>, radius: f64, axis: Vector3<f64>) -> Self {
        // Project out the cylinder axis
        let axis = axis.normalize();
        let proj = Matrix3::identity() - axis * axis.transpose();
        Self {
            center,
            shape: proj,
            safe_distance: radius,
        }
    }
}

/// Solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Planner update frequency [Hz]
    pub frequency: f64,
    /// Maximum solver iterations per call
    pub max_iterations: usize,
    /// Solver tolerance
    pub tolerance: f64,
    /// Use warm-starting from previous solution
    pub warm_start: bool,
    /// Maximum CPU time per solve [ms]
    pub max_cpu_time_ms: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            frequency: 10.0,        // Paper: 10 Hz
            max_iterations: 50,     // RTI typically converges fast
            tolerance: 1e-6,
            warm_start: true,       // Essential for real-time
            max_cpu_time_ms: 50.0,  // Paper achieves ~15ms
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_intervals() {
        let config = HorizonConfig {
            num_nodes: 10,
            horizon_time: 2.0,
            non_uniform: false,
        };

        let intervals = config.compute_intervals();
        assert_eq!(intervals.len(), 10);

        let total: f64 = intervals.iter().sum();
        assert!((total - 2.0).abs() < 1e-10);

        // All intervals should be equal
        for dt in &intervals {
            assert!((dt - 0.2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_non_uniform_intervals() {
        let config = HorizonConfig {
            num_nodes: 20,
            horizon_time: 2.0,
            non_uniform: true,
        };

        let intervals = config.compute_intervals();
        assert_eq!(intervals.len(), 20);

        let total: f64 = intervals.iter().sum();
        assert!((total - 2.0).abs() < 1e-6);

        // First interval should be smaller than last
        assert!(intervals[0] < intervals[19]);
    }
}
