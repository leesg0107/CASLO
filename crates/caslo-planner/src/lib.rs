//! CASLO Motion Planner
//!
//! Online kinodynamic motion planner for cable-suspended load systems.
//!
//! This crate implements the trajectory-based framework from:
//! "Agile and Cooperative Aerial Manipulation of a Cable-Suspended Load"
//! (Sun et al., Science Robotics, 2025)
//!
//! # Architecture
//!
//! The planner solves a finite-time Optimal Control Problem (OCP) at 10 Hz:
//!
//! ```text
//! minimize    J = Σ (‖x - x_ref‖²_Q + ‖u - u_ref‖²_R) + ‖x_N - x_N,ref‖²_P
//! subject to  x₀ = x_init
//!             x_{k+1} = f(x_k, u_k)     (load-cable dynamics)
//!             h(x_k, u_k) ≤ 0           (path constraints)
//! ```
//!
//! # Components
//!
//! - [`ocp`]: OCP problem definition and configuration
//! - [`solver`]: ACADOS solver interface (FFI bindings)
//! - [`trajectory`]: Trajectory generation and interpolation
//! - [`constraints`]: Path constraint definitions
//! - [`integration`]: Integration with caslo-core types
//! - [`controller`]: Receding-horizon motion planner

pub mod config;
pub mod ocp;
pub mod trajectory;
pub mod constraints;
pub mod integration;
pub mod controller;
pub mod solver;
pub mod scenarios;

// ACADOS FFI bindings (only when feature is enabled)
#[cfg(feature = "acados")]
pub mod acados_ffi;

// Re-exports
pub use config::PlannerConfig;
pub use ocp::OcpDefinition;
pub use trajectory::PlannedTrajectory;
pub use controller::MotionPlanner;
