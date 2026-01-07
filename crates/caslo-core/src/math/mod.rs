//! Mathematical utilities for CASLO
//!
//! Implements quaternion operations, SO(3) rotation utilities,
//! unit sphere (S²) operations, and numerical integrators.

pub mod quaternion;
pub mod rotation;
pub mod sphere;
pub mod integrator;

pub use quaternion::*;
pub use rotation::*;
pub use sphere::*;
pub use integrator::*;
