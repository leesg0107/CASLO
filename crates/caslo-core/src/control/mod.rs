//! Control algorithms for CASLO
//!
//! Implements the control framework from the paper:
//! - Position controller (for quadrotors)
//! - Attitude controller
//! - INDI (Incremental Nonlinear Dynamic Inversion) controller
//! - Trajectory tracking controller

pub mod position;
pub mod attitude;
pub mod indi;
pub mod trajectory;

pub use position::*;
pub use attitude::*;
pub use indi::*;
pub use trajectory::*;
