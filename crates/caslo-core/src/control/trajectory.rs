//! Trajectory tracking and generation
//!
//! Provides trajectory structures and tracking utilities for the
//! load and quadrotor system.

use nalgebra::{Vector3, UnitQuaternion};
use serde::{Deserialize, Serialize};

/// Single point on a trajectory with position and derivatives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    /// Position [m]
    pub position: Vector3<f64>,
    /// Velocity [m/s]
    pub velocity: Vector3<f64>,
    /// Acceleration [m/s²]
    pub acceleration: Vector3<f64>,
    /// Jerk [m/s³]
    pub jerk: Vector3<f64>,
    /// Yaw angle [rad]
    pub yaw: f64,
    /// Yaw rate [rad/s]
    pub yaw_rate: f64,
}

impl Default for TrajectoryPoint {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            acceleration: Vector3::zeros(),
            jerk: Vector3::zeros(),
            yaw: 0.0,
            yaw_rate: 0.0,
        }
    }
}

impl TrajectoryPoint {
    /// Create a hover point at given position
    pub fn hover(position: Vector3<f64>, yaw: f64) -> Self {
        Self {
            position,
            yaw,
            ..Default::default()
        }
    }
}

/// Load trajectory point (includes attitude)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTrajectoryPoint {
    /// Translational trajectory
    pub translation: TrajectoryPoint,
    /// Desired orientation
    pub orientation: UnitQuaternion<f64>,
    /// Desired angular velocity (body frame)
    pub angular_velocity: Vector3<f64>,
    /// Desired angular acceleration (body frame)
    pub angular_acceleration: Vector3<f64>,
}

impl Default for LoadTrajectoryPoint {
    fn default() -> Self {
        Self {
            translation: TrajectoryPoint::default(),
            orientation: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
            angular_acceleration: Vector3::zeros(),
        }
    }
}

/// A complete trajectory as a time-indexed sequence of points
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Time stamps for each point [s]
    pub times: Vec<f64>,
    /// Trajectory points
    pub points: Vec<TrajectoryPoint>,
}

impl Trajectory {
    /// Create a new trajectory
    pub fn new(times: Vec<f64>, points: Vec<TrajectoryPoint>) -> Self {
        assert_eq!(times.len(), points.len());
        Self { times, points }
    }

    /// Create a hover trajectory
    pub fn hover(position: Vector3<f64>, yaw: f64, duration: f64) -> Self {
        Self {
            times: vec![0.0, duration],
            points: vec![
                TrajectoryPoint::hover(position, yaw),
                TrajectoryPoint::hover(position, yaw),
            ],
        }
    }

    /// Sample trajectory at given time using linear interpolation
    pub fn sample(&self, t: f64) -> TrajectoryPoint {
        if self.times.is_empty() {
            return TrajectoryPoint::default();
        }

        // Clamp to trajectory bounds
        if t <= self.times[0] {
            return self.points[0].clone();
        }
        if t >= *self.times.last().unwrap() {
            return self.points.last().unwrap().clone();
        }

        // Find surrounding points
        let mut i = 0;
        while i < self.times.len() - 1 && self.times[i + 1] < t {
            i += 1;
        }

        // Linear interpolation factor
        let alpha = (t - self.times[i]) / (self.times[i + 1] - self.times[i]);

        self.interpolate(&self.points[i], &self.points[i + 1], alpha)
    }

    fn interpolate(&self, p1: &TrajectoryPoint, p2: &TrajectoryPoint, alpha: f64) -> TrajectoryPoint {
        TrajectoryPoint {
            position: p1.position * (1.0 - alpha) + p2.position * alpha,
            velocity: p1.velocity * (1.0 - alpha) + p2.velocity * alpha,
            acceleration: p1.acceleration * (1.0 - alpha) + p2.acceleration * alpha,
            jerk: p1.jerk * (1.0 - alpha) + p2.jerk * alpha,
            yaw: p1.yaw * (1.0 - alpha) + p2.yaw * alpha,
            yaw_rate: p1.yaw_rate * (1.0 - alpha) + p2.yaw_rate * alpha,
        }
    }

    /// Total trajectory duration
    pub fn duration(&self) -> f64 {
        if self.times.is_empty() {
            0.0
        } else {
            *self.times.last().unwrap() - self.times[0]
        }
    }

    /// Check if time is within trajectory bounds
    pub fn is_within(&self, t: f64) -> bool {
        !self.times.is_empty() && t >= self.times[0] && t <= *self.times.last().unwrap()
    }
}

/// Generate a simple line trajectory from start to end
pub fn generate_line_trajectory(
    start: Vector3<f64>,
    end: Vector3<f64>,
    speed: f64,
    yaw: f64,
) -> Trajectory {
    let distance = (end - start).norm();
    let duration = distance / speed;
    let direction = (end - start) / distance;
    let velocity = direction * speed;

    Trajectory::new(
        vec![0.0, duration],
        vec![
            TrajectoryPoint {
                position: start,
                velocity,
                yaw,
                ..Default::default()
            },
            TrajectoryPoint {
                position: end,
                velocity,
                yaw,
                ..Default::default()
            },
        ],
    )
}

/// Generate a circular trajectory in the XY plane
pub fn generate_circle_trajectory(
    center: Vector3<f64>,
    radius: f64,
    angular_speed: f64,
    duration: f64,
    dt: f64,
) -> Trajectory {
    let n_points = (duration / dt) as usize + 1;
    let mut times = Vec::with_capacity(n_points);
    let mut points = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let t = i as f64 * dt;
        let theta = angular_speed * t;

        let position = center + Vector3::new(
            radius * theta.cos(),
            radius * theta.sin(),
            0.0,
        );

        let velocity = Vector3::new(
            -radius * angular_speed * theta.sin(),
            radius * angular_speed * theta.cos(),
            0.0,
        );

        let acceleration = Vector3::new(
            -radius * angular_speed.powi(2) * theta.cos(),
            -radius * angular_speed.powi(2) * theta.sin(),
            0.0,
        );

        times.push(t);
        points.push(TrajectoryPoint {
            position,
            velocity,
            acceleration,
            jerk: Vector3::zeros(), // Could compute if needed
            yaw: theta, // Face tangent direction
            yaw_rate: angular_speed,
        });
    }

    Trajectory::new(times, points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_hover_trajectory() {
        let pos = Vector3::new(1.0, 2.0, 3.0);
        let yaw = 0.5;
        let traj = Trajectory::hover(pos, yaw, 10.0);

        // Should have constant position
        let p0 = traj.sample(0.0);
        let p5 = traj.sample(5.0);
        let p10 = traj.sample(10.0);

        assert_relative_eq!(p0.position, pos, epsilon = 1e-10);
        assert_relative_eq!(p5.position, pos, epsilon = 1e-10);
        assert_relative_eq!(p10.position, pos, epsilon = 1e-10);
    }

    #[test]
    fn test_trajectory_interpolation() {
        let traj = Trajectory::new(
            vec![0.0, 1.0],
            vec![
                TrajectoryPoint::hover(Vector3::zeros(), 0.0),
                TrajectoryPoint::hover(Vector3::new(1.0, 0.0, 0.0), 0.0),
            ],
        );

        let mid = traj.sample(0.5);
        assert_relative_eq!(mid.position, Vector3::new(0.5, 0.0, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_line_trajectory() {
        let start = Vector3::zeros();
        let end = Vector3::new(10.0, 0.0, 0.0);
        let speed = 2.0;

        let traj = generate_line_trajectory(start, end, speed, 0.0);

        assert_relative_eq!(traj.duration(), 5.0, epsilon = 1e-10);

        let mid = traj.sample(2.5);
        assert_relative_eq!(mid.position, Vector3::new(5.0, 0.0, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_circle_trajectory() {
        let center = Vector3::new(0.0, 0.0, 1.0);
        let radius = 2.0;
        let omega = 1.0; // rad/s
        let duration = 2.0 * PI; // One full circle

        let traj = generate_circle_trajectory(center, radius, omega, duration, 0.1);

        // At t=0, should be at (center.x + radius, center.y, center.z)
        let p0 = traj.sample(0.0);
        assert_relative_eq!(p0.position.x, center.x + radius, epsilon = 1e-10);
        assert_relative_eq!(p0.position.y, center.y, epsilon = 1e-10);

        // At t=PI/2/omega, should be at (center.x, center.y + radius, center.z)
        let p_quarter = traj.sample(PI / 2.0);
        assert_relative_eq!(p_quarter.position.x, center.x, epsilon = 0.1);
        assert_relative_eq!(p_quarter.position.y, center.y + radius, epsilon = 0.1);
    }

    #[test]
    fn test_trajectory_clamping() {
        let traj = Trajectory::hover(Vector3::new(1.0, 2.0, 3.0), 0.0, 5.0);

        // Before start
        let before = traj.sample(-1.0);
        assert_relative_eq!(before.position, Vector3::new(1.0, 2.0, 3.0), epsilon = 1e-10);

        // After end
        let after = traj.sample(100.0);
        assert_relative_eq!(after.position, Vector3::new(1.0, 2.0, 3.0), epsilon = 1e-10);
    }
}
