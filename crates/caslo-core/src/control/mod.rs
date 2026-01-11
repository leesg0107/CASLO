//! Control algorithms for CASLO
//!
//! Implements the control framework from the paper:
//! - Position controller (for quadrotors)
//! - Attitude controller
//! - INDI (Incremental Nonlinear Dynamic Inversion) controller
//! - Trajectory tracking controller
//! - Quadrotor trajectory tracker (Eq. 15 - onboard 300Hz controller)

pub mod position;
pub mod attitude;
pub mod indi;
pub mod trajectory;
pub mod quadrotor_tracker;

pub use position::*;
pub use attitude::*;
pub use indi::*;
pub use trajectory::*;
pub use quadrotor_tracker::*;
