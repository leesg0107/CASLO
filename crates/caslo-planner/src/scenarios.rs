//! Experimental Scenarios from the Paper
//!
//! Implements the experimental scenarios from:
//! "Agile and cooperative aerial manipulation of a cable-suspended load"
//! (Science Robotics, Sun et al., 2025)
//!
//! Reference data available at: https://doi.org/10.5061/dryad.n2z34tn6w

use nalgebra::{Vector3, UnitQuaternion};
use std::f64::consts::PI;

/// System parameters from the paper's experimental setup
#[derive(Debug, Clone)]
pub struct PaperSystemParams {
    /// Load mass [kg]
    pub load_mass: f64,
    /// Load dimensions [m] (width, depth, height)
    pub load_dimensions: Vector3<f64>,
    /// Load inertia diagonal [kg·m²]
    pub load_inertia: Vector3<f64>,
    /// Attachment points on load [m] (3-point triangular configuration)
    pub attachment_points: Vec<Vector3<f64>>,
    /// Cable lengths [m]
    pub cable_lengths: Vec<f64>,
    /// Quadrotor mass [kg]
    pub quadrotor_mass: f64,
    /// Quadrotor inertia diagonal [kg·m²]
    pub quadrotor_inertia: Vector3<f64>,
    /// Maximum thrust per quadrotor [N]
    pub max_thrust: f64,
    /// Number of quadrotors
    pub num_quads: usize,
}

impl Default for PaperSystemParams {
    fn default() -> Self {
        Self::paper_3quad()
    }
}

impl PaperSystemParams {
    /// 3-quadrotor configuration from the paper
    pub fn paper_3quad() -> Self {
        Self {
            load_mass: 1.4,
            load_dimensions: Vector3::new(0.54, 0.45, 0.15),
            load_inertia: Vector3::new(0.03, 0.04, 0.05), // Approximated from dimensions
            // Triangular attachment configuration - symmetric around CoM
            // 120° apart, radius ~0.3m from center
            attachment_points: vec![
                Vector3::new(0.3, 0.0, 0.0),                        // Front
                Vector3::new(-0.15, 0.2598, 0.0),                   // Back-left (120°)
                Vector3::new(-0.15, -0.2598, 0.0),                  // Back-right (240°)
            ],
            cable_lengths: vec![1.0, 1.0, 1.0],
            quadrotor_mass: 0.6,
            quadrotor_inertia: Vector3::new(0.01, 0.01, 0.02),
            max_thrust: 20.0,
            num_quads: 3,
        }
    }

    /// 4-quadrotor configuration (scalability test)
    pub fn paper_4quad() -> Self {
        Self {
            load_mass: 1.4,
            load_dimensions: Vector3::new(0.54, 0.45, 0.15),
            load_inertia: Vector3::new(0.03, 0.04, 0.05),
            // Square attachment configuration
            attachment_points: vec![
                Vector3::new(0.25, 0.20, 0.06),
                Vector3::new(0.25, -0.20, 0.06),
                Vector3::new(-0.25, -0.20, 0.06),
                Vector3::new(-0.25, 0.20, 0.06),
            ],
            cable_lengths: vec![1.0, 1.0, 1.0, 1.0],
            quadrotor_mass: 0.6,
            quadrotor_inertia: Vector3::new(0.01, 0.01, 0.02),
            max_thrust: 20.0,
            num_quads: 4,
        }
    }

    /// Compute hover tension per cable [N]
    ///
    /// From Eq. 2: m_L * v̇ = -Σᵢ tᵢsᵢ + m_L * g
    /// At hover with vertical cables (s_i = [0,0,-1]):
    ///   0 = -n * t * (-1) + m_L * (-g)  →  n*t = m_L * g  →  t = m_L * g / n
    ///
    /// Note: This is the tension in the cable, which also supports the quadrotor.
    /// The quadrotor thrust must be T_i = t_i + m_q * g to hover.
    pub fn hover_tension(&self) -> f64 {
        self.load_mass * 9.81 / self.num_quads as f64
    }
}

/// Figure-8 trajectory parameters from the paper
#[derive(Debug, Clone, Copy)]
pub struct Figure8Params {
    /// Maximum velocity [m/s]
    pub max_velocity: f64,
    /// Maximum acceleration [m/s²]
    pub max_acceleration: f64,
    /// Maximum jerk [m/s³]
    pub max_jerk: f64,
    /// X amplitude [m]
    pub amplitude_x: f64,
    /// Y amplitude [m]
    pub amplitude_y: f64,
    /// Height [m]
    pub height: f64,
    /// Angular frequency for X [rad/s]
    pub omega_x: f64,
    /// Angular frequency for Y [rad/s] (2x omega_x for figure-8)
    pub omega_y: f64,
    /// Yaw rate [rad/s]
    pub yaw_rate: f64,
}

impl Figure8Params {
    /// Slow figure-8 trajectory (baseline, 1 m/s max)
    pub fn slow() -> Self {
        Self {
            max_velocity: 1.0,
            max_acceleration: 0.5,
            max_jerk: 0.25,
            amplitude_x: 2.5,
            amplitude_y: 2.0,
            height: 1.0,
            omega_x: 0.25,
            omega_y: 0.5,
            yaw_rate: 0.25,
        }
    }

    /// Medium figure-8 trajectory (baseline, 2 m/s max)
    pub fn medium() -> Self {
        Self {
            max_velocity: 2.0,
            max_acceleration: 2.0,
            max_jerk: 2.0,
            amplitude_x: 2.5,
            amplitude_y: 2.0,
            height: 1.0,
            omega_x: 0.5,
            omega_y: 1.0,
            yaw_rate: 0.25,
        }
    }

    /// Medium-plus figure-8 trajectory (challenging, 2 m/s, 4 m/s² acc)
    /// Note: Baseline methods crashed on this scenario
    pub fn medium_plus() -> Self {
        Self {
            max_velocity: 2.0,
            max_acceleration: 4.0,
            max_jerk: 8.0,
            amplitude_x: 1.0,
            amplitude_y: 1.0,
            height: 1.0,
            omega_x: 1.0,
            omega_y: 2.0,
            yaw_rate: 0.25,
        }
    }

    /// Fast figure-8 trajectory (most aggressive, 5 m/s, 8 m/s² acc)
    /// Note: Only CASLO succeeded on this scenario
    pub fn fast() -> Self {
        Self {
            max_velocity: 5.0,
            max_acceleration: 8.0,
            max_jerk: 16.0,
            amplitude_x: 2.5,
            amplitude_y: 2.0,
            height: 1.0,
            omega_x: 1.0,
            omega_y: 2.0,
            yaw_rate: 0.25,
        }
    }

    /// Get position at time t
    pub fn position(&self, t: f64) -> Vector3<f64> {
        Vector3::new(
            self.amplitude_x * (self.omega_x * t).cos(),
            self.amplitude_y * (self.omega_y * t).sin(),
            self.height,
        )
    }

    /// Get velocity at time t
    pub fn velocity(&self, t: f64) -> Vector3<f64> {
        Vector3::new(
            -self.amplitude_x * self.omega_x * (self.omega_x * t).sin(),
            self.amplitude_y * self.omega_y * (self.omega_y * t).cos(),
            0.0,
        )
    }

    /// Get acceleration at time t
    pub fn acceleration(&self, t: f64) -> Vector3<f64> {
        Vector3::new(
            -self.amplitude_x * self.omega_x.powi(2) * (self.omega_x * t).cos(),
            -self.amplitude_y * self.omega_y.powi(2) * (self.omega_y * t).sin(),
            0.0,
        )
    }

    /// Get jerk at time t
    pub fn jerk(&self, t: f64) -> Vector3<f64> {
        Vector3::new(
            self.amplitude_x * self.omega_x.powi(3) * (self.omega_x * t).sin(),
            -self.amplitude_y * self.omega_y.powi(3) * (self.omega_y * t).cos(),
            0.0,
        )
    }

    /// Get orientation at time t (rotating yaw)
    pub fn orientation(&self, t: f64) -> UnitQuaternion<f64> {
        let yaw = self.yaw_rate * t;
        UnitQuaternion::from_euler_angles(0.0, 0.0, yaw)
    }

    /// Get angular velocity at time t
    pub fn angular_velocity(&self, _t: f64) -> Vector3<f64> {
        Vector3::new(0.0, 0.0, self.yaw_rate)
    }

    /// Period of one complete figure-8 [s]
    pub fn period(&self) -> f64 {
        2.0 * PI / self.omega_x
    }
}

/// Obstacle avoidance scenario types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObstacleScenario {
    /// Narrow vertical passage (0.8m gap, system width ~1.4m)
    /// Requires ~70° load tilt to squeeze through
    NarrowPassage,
    /// Horizontal gap passage (0.6m gap, system height ~1.2m)
    /// Requires rapid cable inclination change from ~90° to ~180°
    HorizontalGap,
}

/// Obstacle configuration
#[derive(Debug, Clone)]
pub struct Obstacle {
    /// Center position [m]
    pub position: Vector3<f64>,
    /// Half-extents [m] (for box obstacles)
    pub half_extents: Vector3<f64>,
    /// Obstacle type
    pub obstacle_type: ObstacleType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObstacleType {
    /// Cylindrical obstacle
    Cylinder,
    /// Box obstacle
    Box,
}

impl Obstacle {
    /// Create a cylindrical obstacle
    pub fn cylinder(position: Vector3<f64>, radius: f64, height: f64) -> Self {
        Self {
            position,
            half_extents: Vector3::new(radius, radius, height / 2.0),
            obstacle_type: ObstacleType::Cylinder,
        }
    }

    /// Create a box obstacle
    pub fn box_obstacle(position: Vector3<f64>, half_extents: Vector3<f64>) -> Self {
        Self {
            position,
            half_extents,
            obstacle_type: ObstacleType::Box,
        }
    }
}

/// Narrow passage scenario from paper (Fig. 4A)
#[derive(Debug, Clone)]
pub struct NarrowPassageScenario {
    /// Gap width [m] (actual: 0.8m, clearance after no-fly zone: 0.2m)
    pub gap_width: f64,
    /// Obstacle cylinders
    pub obstacles: Vec<Obstacle>,
    /// Start position
    pub start: Vector3<f64>,
    /// Goal position
    pub goal: Vector3<f64>,
    /// Traversal speed [m/s]
    pub traversal_speed: f64,
}

impl Default for NarrowPassageScenario {
    fn default() -> Self {
        let gap_width = 0.8;
        let obstacle_radius = 0.3;
        let obstacle_height = 2.0;

        // Two cylinders forming a narrow gap
        let gap_center = Vector3::new(3.0, 0.0, 1.0);

        Self {
            gap_width,
            obstacles: vec![
                Obstacle::cylinder(
                    gap_center + Vector3::new(0.0, gap_width / 2.0 + obstacle_radius, 0.0),
                    obstacle_radius,
                    obstacle_height,
                ),
                Obstacle::cylinder(
                    gap_center - Vector3::new(0.0, gap_width / 2.0 + obstacle_radius, 0.0),
                    obstacle_radius,
                    obstacle_height,
                ),
            ],
            start: Vector3::new(0.0, 0.0, 1.0),
            goal: Vector3::new(6.0, 0.0, 1.0),
            traversal_speed: 4.0,
        }
    }
}

/// Horizontal gap scenario from paper (Fig. 4B)
#[derive(Debug, Clone)]
pub struct HorizontalGapScenario {
    /// Gap height [m] (actual: 0.6m, clearance: 0.2m)
    pub gap_height: f64,
    /// Obstacle cylinders
    pub obstacles: Vec<Obstacle>,
    /// Start position
    pub start: Vector3<f64>,
    /// Goal position
    pub goal: Vector3<f64>,
    /// Traversal speed [m/s]
    pub traversal_speed: f64,
}

impl Default for HorizontalGapScenario {
    fn default() -> Self {
        let gap_height = 0.6;
        let obstacle_radius = 0.3;
        let obstacle_length = 3.0;

        // Two horizontal cylinders forming a narrow gap
        let gap_center = Vector3::new(3.0, 0.0, 1.5);

        Self {
            gap_height,
            obstacles: vec![
                // Upper cylinder
                Obstacle::cylinder(
                    gap_center + Vector3::new(0.0, 0.0, gap_height / 2.0 + obstacle_radius),
                    obstacle_radius,
                    obstacle_length,
                ),
                // Lower cylinder
                Obstacle::cylinder(
                    gap_center - Vector3::new(0.0, 0.0, gap_height / 2.0 + obstacle_radius),
                    obstacle_radius,
                    obstacle_length,
                ),
            ],
            start: Vector3::new(0.0, 0.0, 1.5),
            goal: Vector3::new(6.0, 0.0, 1.5),
            traversal_speed: 4.0,
        }
    }
}

/// OCP parameters from the paper
#[derive(Debug, Clone)]
pub struct PaperOcpParams {
    /// Number of shooting nodes
    pub num_nodes: usize,
    /// Horizon time [s]
    pub horizon_time: f64,
    /// Position tracking weight
    pub weight_position: Vector3<f64>,
    /// Velocity tracking weight
    pub weight_velocity: Vector3<f64>,
    /// Orientation tracking weight
    pub weight_orientation: Vector3<f64>,
    /// Angular velocity tracking weight
    pub weight_angular_velocity: Vector3<f64>,
    /// Cable angular snap weight (γᵢ = r⃛ᵢ) - 4th derivative, control input
    pub weight_angular_snap: f64,
    /// Cable tension acceleration weight (λᵢ = ẗᵢ)
    pub weight_tension_acceleration: f64,
    /// Terminal cost scaling factor
    pub terminal_scaling: f64,
    /// Minimum cable tension [N]
    pub tension_min: f64,
    /// Maximum cable tension [N]
    pub tension_max: f64,
    /// Minimum inter-quadrotor distance [m]
    pub min_quad_distance: f64,
    /// Maximum cable angular snap [rad/s⁴] (γᵢ = r⃛ᵢ) - 4th derivative
    pub max_angular_snap: f64,
    /// Maximum tension acceleration [N/s²] (λᵢ = ẗᵢ)
    pub max_tension_acceleration: f64,
}

impl Default for PaperOcpParams {
    fn default() -> Self {
        Self {
            num_nodes: 20,
            horizon_time: 2.0,
            weight_position: Vector3::new(10.0, 10.0, 20.0),
            weight_velocity: Vector3::new(1.0, 1.0, 2.0),
            weight_orientation: Vector3::new(5.0, 5.0, 5.0),
            weight_angular_velocity: Vector3::new(0.5, 0.5, 0.5),
            // Control regularization weights
            // CRITICAL: Must be high enough to keep condition number < 10,000
            // κ(W) = max_weight / min_weight = 100 / 0.01 = 10,000 (acceptable)
            weight_angular_snap: 0.01,      // Increased 10x from 0.001
            weight_tension_acceleration: 0.01,  // Increased 100x from 0.0001
            terminal_scaling: 10.0,
            tension_min: 0.5,
            tension_max: 50.0,
            min_quad_distance: 0.2,  // Reduced for aggressive maneuvers (paper uses 0.8m)
            // Control bounds - reduced to prevent state-dependent infeasibility
            max_angular_snap: 200.0,    // Reduced from 1000
            max_tension_acceleration: 500.0,  // Reduced from 1000
        }
    }
}

/// Reference trajectory point for the controller
#[derive(Debug, Clone)]
pub struct TrajectoryPoint {
    pub time: f64,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub acceleration: Vector3<f64>,
    pub jerk: Vector3<f64>,
    pub orientation: UnitQuaternion<f64>,
    pub angular_velocity: Vector3<f64>,
}

/// Generate a figure-8 trajectory
pub fn generate_figure8_trajectory(
    params: &Figure8Params,
    duration: f64,
    dt: f64,
) -> Vec<TrajectoryPoint> {
    let n_points = (duration / dt).ceil() as usize + 1;

    (0..n_points)
        .map(|i| {
            let t = i as f64 * dt;
            TrajectoryPoint {
                time: t,
                position: params.position(t),
                velocity: params.velocity(t),
                acceleration: params.acceleration(t),
                jerk: params.jerk(t),
                orientation: params.orientation(t),
                angular_velocity: params.angular_velocity(t),
            }
        })
        .collect()
}

/// Minimum-snap trajectory segment for obstacle avoidance
#[derive(Debug, Clone)]
pub struct MinSnapSegment {
    /// Start position
    pub start: Vector3<f64>,
    /// End position
    pub end: Vector3<f64>,
    /// Duration [s]
    pub duration: f64,
    /// Polynomial coefficients for each axis (8 coefficients for min-snap)
    pub coeffs: [Vec<f64>; 3],
}

impl MinSnapSegment {
    /// Create a simple straight-line segment (not true min-snap, for demonstration)
    pub fn straight_line(start: Vector3<f64>, end: Vector3<f64>, duration: f64) -> Self {
        // Simple cubic polynomial: p(t) = a0 + a1*t + a2*t² + a3*t³
        // With zero velocity at start/end: p(0)=start, p(T)=end, p'(0)=0, p'(T)=0
        let coeffs: [Vec<f64>; 3] = [
            Self::cubic_coeffs(start.x, end.x, duration),
            Self::cubic_coeffs(start.y, end.y, duration),
            Self::cubic_coeffs(start.z, end.z, duration),
        ];

        Self {
            start,
            end,
            duration,
            coeffs,
        }
    }

    fn cubic_coeffs(p0: f64, pf: f64, t: f64) -> Vec<f64> {
        // Hermite spline with zero velocity endpoints
        let a0 = p0;
        let a1 = 0.0;
        let a2 = 3.0 * (pf - p0) / (t * t);
        let a3 = -2.0 * (pf - p0) / (t * t * t);
        vec![a0, a1, a2, a3]
    }

    /// Evaluate position at time t (0 <= t <= duration)
    pub fn position(&self, t: f64) -> Vector3<f64> {
        let t = t.clamp(0.0, self.duration);
        Vector3::new(
            self.eval_poly(&self.coeffs[0], t),
            self.eval_poly(&self.coeffs[1], t),
            self.eval_poly(&self.coeffs[2], t),
        )
    }

    /// Evaluate velocity at time t
    pub fn velocity(&self, t: f64) -> Vector3<f64> {
        let t = t.clamp(0.0, self.duration);
        Vector3::new(
            self.eval_poly_deriv(&self.coeffs[0], t, 1),
            self.eval_poly_deriv(&self.coeffs[1], t, 1),
            self.eval_poly_deriv(&self.coeffs[2], t, 1),
        )
    }

    /// Evaluate acceleration at time t
    pub fn acceleration(&self, t: f64) -> Vector3<f64> {
        let t = t.clamp(0.0, self.duration);
        Vector3::new(
            self.eval_poly_deriv(&self.coeffs[0], t, 2),
            self.eval_poly_deriv(&self.coeffs[1], t, 2),
            self.eval_poly_deriv(&self.coeffs[2], t, 2),
        )
    }

    fn eval_poly(&self, coeffs: &[f64], t: f64) -> f64 {
        coeffs.iter().enumerate().map(|(i, c)| c * t.powi(i as i32)).sum()
    }

    fn eval_poly_deriv(&self, coeffs: &[f64], t: f64, order: usize) -> f64 {
        if order >= coeffs.len() {
            return 0.0;
        }

        coeffs.iter()
            .enumerate()
            .skip(order)
            .map(|(i, c)| {
                let mut coeff = *c;
                for j in 0..order {
                    coeff *= (i - j) as f64;
                }
                coeff * t.powi((i - order) as i32)
            })
            .sum()
    }
}

/// Robustness test parameters
#[derive(Debug, Clone)]
pub struct RobustnessTestParams {
    /// Mass mismatch factor (e.g., 0.5 = 50% less, 1.5 = 50% more)
    pub mass_mismatch: f64,
    /// Inertia mismatch factor
    pub inertia_mismatch: f64,
    /// Center of gravity bias [m]
    pub cog_bias: Vector3<f64>,
    /// Wind velocity [m/s]
    pub wind_velocity: Vector3<f64>,
    /// Position noise std dev [m]
    pub position_noise: f64,
    /// Attitude noise std dev [rad]
    pub attitude_noise: f64,
    /// Velocity noise std dev [m/s]
    pub velocity_noise: f64,
}

impl Default for RobustnessTestParams {
    fn default() -> Self {
        Self {
            mass_mismatch: 1.0,
            inertia_mismatch: 1.0,
            cog_bias: Vector3::zeros(),
            wind_velocity: Vector3::zeros(),
            position_noise: 0.0,
            attitude_noise: 0.0,
            velocity_noise: 0.0,
        }
    }
}

impl RobustnessTestParams {
    /// 50% mass mismatch test
    pub fn mass_mismatch_50() -> Self {
        Self {
            mass_mismatch: 1.5,
            ..Default::default()
        }
    }

    /// Wind disturbance test (5 m/s)
    pub fn wind_5ms() -> Self {
        Self {
            wind_velocity: Vector3::new(5.0, 0.0, 0.0),
            ..Default::default()
        }
    }

    /// Noise level 4 from paper (threshold for 95% success)
    pub fn noise_level_4() -> Self {
        Self {
            position_noise: 0.07,
            attitude_noise: 7.0_f64.to_radians(),
            velocity_noise: 0.07,
            ..Default::default()
        }
    }

    /// Sloshing load test (basketball in basket)
    pub fn sloshing_load() -> Self {
        Self {
            mass_mismatch: 1.43, // 0.6kg basketball in 1.4kg basket
            cog_bias: Vector3::new(0.02, 0.02, 0.0), // Unknown CoG shift
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_figure8_trajectory() {
        let params = Figure8Params::slow();

        // At t=0
        let p0 = params.position(0.0);
        assert_relative_eq!(p0.x, params.amplitude_x, epsilon = 1e-10);
        assert_relative_eq!(p0.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(p0.z, params.height, epsilon = 1e-10);

        // Velocity at t=0 should be in Y direction
        let v0 = params.velocity(0.0);
        assert_relative_eq!(v0.x, 0.0, epsilon = 1e-10);
        assert!(v0.y.abs() > 0.0);
        assert_relative_eq!(v0.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_paper_system_params() {
        let params = PaperSystemParams::paper_3quad();

        assert_eq!(params.num_quads, 3);
        assert_relative_eq!(params.load_mass, 1.4, epsilon = 1e-10);
        assert_relative_eq!(params.quadrotor_mass, 0.6, epsilon = 1e-10);
        assert_eq!(params.attachment_points.len(), 3);

        // Check hover tension (load mass only, not including quadrotors)
        let hover_t = params.hover_tension();
        let expected = 1.4 * 9.81 / 3.0;  // ~4.58 N per cable
        assert_relative_eq!(hover_t, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_min_snap_segment() {
        let start = Vector3::new(0.0, 0.0, 1.0);
        let end = Vector3::new(3.0, 0.0, 1.0);
        let duration = 2.0;

        let segment = MinSnapSegment::straight_line(start, end, duration);

        // Start position
        let p0 = segment.position(0.0);
        assert_relative_eq!(p0.x, start.x, epsilon = 1e-10);
        assert_relative_eq!(p0.y, start.y, epsilon = 1e-10);

        // End position
        let pf = segment.position(duration);
        assert_relative_eq!(pf.x, end.x, epsilon = 1e-10);

        // Zero velocity at endpoints
        let v0 = segment.velocity(0.0);
        let vf = segment.velocity(duration);
        assert_relative_eq!(v0.norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(vf.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_figure8_scenarios() {
        // Test all four scenarios exist and have correct max values
        let slow = Figure8Params::slow();
        let medium = Figure8Params::medium();
        let medium_plus = Figure8Params::medium_plus();
        let fast = Figure8Params::fast();

        assert_relative_eq!(slow.max_velocity, 1.0, epsilon = 1e-10);
        assert_relative_eq!(medium.max_velocity, 2.0, epsilon = 1e-10);
        assert_relative_eq!(medium_plus.max_velocity, 2.0, epsilon = 1e-10);
        assert_relative_eq!(fast.max_velocity, 5.0, epsilon = 1e-10);

        assert_relative_eq!(slow.max_acceleration, 0.5, epsilon = 1e-10);
        assert_relative_eq!(fast.max_acceleration, 8.0, epsilon = 1e-10);
    }
}
