//! # CASLO Core
//!
//! Cable-Suspended Load Control System - Core library
//!
//! This library implements the dynamics, control, and estimation algorithms
//! from the paper "Agile and Cooperative Aerial Manipulation of a Cable-Suspended Load"
//! (Sun et al., Science Robotics 2025).
//!
//! ## Modules
//!
//! - [`math`]: Mathematical utilities (quaternions, SO(3), integrators)
//! - [`dynamics`]: Dynamic models (load, cable, quadrotor, full system)
//! - [`control`]: Controllers (INDI, trajectory tracking)
//! - [`estimation`]: State estimators (EKF, load-cable estimator)
//! - [`simulation`]: Simulation framework

pub mod math;
pub mod dynamics;
pub mod control;
pub mod estimation;
pub mod simulation;

// Common type aliases
use nalgebra::{Vector3, Matrix3, UnitQuaternion};

/// 3D vector type
pub type Vec3 = Vector3<f64>;

/// 3x3 matrix type
pub type Mat3 = Matrix3<f64>;

/// Unit quaternion type for rotations
pub type Quat = UnitQuaternion<f64>;

/// Gravity constant [m/s²]
pub const GRAVITY: f64 = 9.81;

/// Default gravity vector (pointing downward in NED or upward in ENU)
/// Using NED convention: z-down
pub fn gravity_ned() -> Vec3 {
    Vec3::new(0.0, 0.0, GRAVITY)
}

/// Gravity vector in ENU convention: z-up
pub fn gravity_enu() -> Vec3 {
    Vec3::new(0.0, 0.0, -GRAVITY)
}
