//! Kinematic constraints module
//!
//! Implements the kinematic constraint and its derivatives from Eq. (5) and Eq. (S1)
//! of the paper for computing quadrotor trajectories from load-cable state.

mod constraint;

pub use constraint::*;
