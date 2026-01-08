//! ACADOS Solver Interface
//!
//! Safe Rust wrapper around the generated ACADOS solver.
//! This module provides a high-level interface for solving the OCP.

use nalgebra::{Vector3, UnitQuaternion};
use thiserror::Error;

use crate::config::PlannerConfig;
use crate::ocp::{OcpState, OcpControl, OcpDefinition, SystemParameters, ReferencePoint};
use crate::trajectory::PlannedTrajectory;

#[cfg(feature = "acados")]
use crate::acados_ffi::{self, AcadosCapsule};

/// Solver errors
#[derive(Debug, Error)]
pub enum SolverError {
    #[error("Solver initialization failed")]
    InitializationFailed,
    #[error("Solver iteration failed with status {0}")]
    SolveFailed(i32),
    #[error("Invalid state dimension: expected {expected}, got {got}")]
    InvalidStateDimension { expected: usize, got: usize },
    #[error("Invalid control dimension: expected {expected}, got {got}")]
    InvalidControlDimension { expected: usize, got: usize },
    #[error("Reference trajectory too short")]
    ReferenceTrajectoryTooShort,
    #[error("Solver not available (ACADOS not compiled)")]
    SolverNotAvailable,
    #[error("Maximum iterations reached without convergence")]
    MaxIterationsReached,
    #[error("QP solver failed")]
    QpSolverFailed,
}

/// ACADOS solver status codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SolverStatus {
    Success = 0,
    MaxIterations = 1,
    QpFailure = 2,
    NaNDetected = 3,
    Unknown = -1,
}

impl From<i32> for SolverStatus {
    fn from(code: i32) -> Self {
        match code {
            0 => SolverStatus::Success,
            1 => SolverStatus::MaxIterations,
            2 => SolverStatus::QpFailure,
            3 => SolverStatus::NaNDetected,
            _ => SolverStatus::Unknown,
        }
    }
}

/// Solution statistics from solver
#[derive(Debug, Clone, Default)]
pub struct SolveStatistics {
    /// Number of SQP iterations
    pub sqp_iterations: usize,
    /// Number of QP iterations
    pub qp_iterations: usize,
    /// Total solve time [ms]
    pub solve_time_ms: f64,
    /// Residual (KKT) norm
    pub kkt_residual: f64,
    /// Objective value
    pub objective: f64,
}

/// Configuration for a single solve call
#[derive(Debug, Clone)]
pub struct SolveOptions {
    /// Maximum SQP iterations
    pub max_sqp_iter: usize,
    /// Maximum QP iterations per SQP step
    pub max_qp_iter: usize,
    /// Whether to warm-start from previous solution
    pub warm_start: bool,
    /// Timeout in milliseconds
    pub timeout_ms: Option<f64>,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            max_sqp_iter: 50,
            max_qp_iter: 50,
            warm_start: true,
            timeout_ms: Some(50.0),
        }
    }
}

/// High-level ACADOS solver wrapper
///
/// Manages the lifecycle of the ACADOS solver and provides
/// a safe interface for solving the cable-suspended load OCP.
pub struct AcadosSolver {
    /// OCP definition
    ocp: OcpDefinition,
    /// Current solver state (for warm-starting)
    warm_start_state: Option<Vec<Vec<f64>>>,
    warm_start_control: Option<Vec<Vec<f64>>>,
    /// Last solve statistics
    last_stats: SolveStatistics,
    /// Whether solver has been initialized
    initialized: bool,
    /// ACADOS solver capsule (only when feature is enabled)
    #[cfg(feature = "acados")]
    capsule: Option<AcadosCapsule>,
}

// Safety: The solver handle is only accessed from a single thread
// and all FFI calls are properly synchronized
unsafe impl Send for AcadosSolver {}

impl AcadosSolver {
    /// Create a new solver instance
    pub fn new(ocp: OcpDefinition) -> Result<Self, SolverError> {
        #[cfg(feature = "acados")]
        {
            // Initialize ACADOS solver
            let capsule = AcadosCapsule::new()
                .map_err(|e| SolverError::SolveFailed(e))?;

            Ok(Self {
                ocp,
                warm_start_state: None,
                warm_start_control: None,
                last_stats: SolveStatistics::default(),
                initialized: true,
                capsule: Some(capsule),
            })
        }

        #[cfg(not(feature = "acados"))]
        {
            // Return a stub solver for testing without ACADOS
            Ok(Self {
                ocp,
                warm_start_state: None,
                warm_start_control: None,
                last_stats: SolveStatistics::default(),
                initialized: false,
            })
        }
    }

    /// Solve the OCP given current state and reference trajectory
    ///
    /// # Arguments
    /// * `initial_state` - Current system state
    /// * `reference` - Reference trajectory to track
    /// * `options` - Solve options
    ///
    /// # Returns
    /// Planned trajectory or error
    pub fn solve(
        &mut self,
        initial_state: &OcpState,
        reference: &[ReferencePoint],
        options: &SolveOptions,
    ) -> Result<PlannedTrajectory, SolverError> {
        if reference.len() < self.ocp.n() {
            return Err(SolverError::ReferenceTrajectoryTooShort);
        }

        #[cfg(feature = "acados")]
        {
            self.solve_acados(initial_state, reference, options)
        }

        #[cfg(not(feature = "acados"))]
        {
            // Fallback: return a simple trajectory that moves toward the reference
            self.solve_fallback(initial_state, reference)
        }
    }

    /// Set system parameters
    pub fn set_parameters(&mut self, params: &SystemParameters) -> Result<(), SolverError> {
        self.ocp.set_system(params.clone());

        #[cfg(feature = "acados")]
        {
            // Update ACADOS parameters for each stage
            if let Some(ref mut capsule) = self.capsule {
                // Pack system parameters into the format expected by ACADOS
                let p = pack_system_parameters(params);
                for stage in 0..=self.ocp.n() {
                    capsule.set_parameters(stage, &p)
                        .map_err(|e| SolverError::SolveFailed(e))?;
                }
            }
        }

        Ok(())
    }

    /// Get the last solve statistics
    pub fn statistics(&self) -> &SolveStatistics {
        &self.last_stats
    }

    /// Check if solver is ready
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Reset warm-start data
    pub fn reset_warm_start(&mut self) {
        self.warm_start_state = None;
        self.warm_start_control = None;
    }

    #[cfg(feature = "acados")]
    fn solve_acados(
        &mut self,
        initial_state: &OcpState,
        reference: &[ReferencePoint],
        options: &SolveOptions,
    ) -> Result<PlannedTrajectory, SolverError> {
        let capsule = self.capsule.as_mut()
            .ok_or(SolverError::SolverNotAvailable)?;

        let n = self.ocp.n();
        eprintln!("ACADOS: n={}", n);

        // Set initial state constraint
        let x0 = initial_state.to_vector();
        eprintln!("ACADOS: x0 len={}", x0.len());
        eprintln!("ACADOS: x0 full = {:?}", &x0);
        capsule.set_initial_state(&x0)
            .map_err(|e| SolverError::SolveFailed(e))?;
        eprintln!("ACADOS: initial state OK");

        // Set reference trajectory for each stage
        for k in 0..n {
            let ref_k = &reference[k.min(reference.len() - 1)];
            let y_ref = reference_to_vector(ref_k, self.ocp.num_quadrotors);
            if k == 0 {
                eprintln!("ACADOS: ref[0] len={}", y_ref.len());
                eprintln!("ACADOS: ref[0] = {:?}", &y_ref);
            }
            capsule.set_reference(k, &y_ref)
                .map_err(|e| SolverError::SolveFailed(e))?;
        }
        eprintln!("ACADOS: stage refs set OK");

        // Set terminal reference
        let ref_e = &reference[(n - 1).min(reference.len() - 1)];
        let y_ref_e = reference_to_vector_terminal(ref_e);
        eprintln!("ACADOS: terminal ref len={}", y_ref_e.len());
        capsule.set_reference(n, &y_ref_e)
            .map_err(|e| SolverError::SolveFailed(e))?;
        eprintln!("ACADOS: terminal ref set OK, calling solve...");

        // Apply warm-start if available, otherwise initialize with current state
        if options.warm_start && self.warm_start_state.is_some() {
            if let Some(ref states) = self.warm_start_state {
                for (k, state) in states.iter().enumerate() {
                    if k <= n {
                        let _ = capsule.set_state_init(k, state);
                    }
                }
            }
            if let Some(ref controls) = self.warm_start_control {
                for (k, control) in controls.iter().enumerate() {
                    if k < n {
                        let _ = capsule.set_control_init(k, control);
                    }
                }
            }
        } else {
            // Initialize trajectory with current state (like C test does)
            let u0 = vec![0.0; acados_ffi::NU];
            capsule.initialize_trajectory(&x0, &u0)
                .map_err(|e| SolverError::SolveFailed(e))?;
            eprintln!("ACADOS: initialized trajectory");
        }

        // Solve
        eprintln!("ACADOS: calling solve...");
        let status = capsule.solve()
            .map_err(|e| SolverError::SolveFailed(e))?;
        eprintln!("ACADOS: solve returned status={}", status);

        // Extract statistics
        self.last_stats = SolveStatistics {
            sqp_iterations: capsule.get_sqp_iterations() as usize,
            qp_iterations: 0,
            solve_time_ms: capsule.get_solve_time() * 1000.0,
            kkt_residual: 0.0,
            objective: 0.0,
        };

        match SolverStatus::from(status) {
            SolverStatus::Success => {}
            SolverStatus::MaxIterations => return Err(SolverError::MaxIterationsReached),
            SolverStatus::QpFailure => return Err(SolverError::QpSolverFailed),
            _ => return Err(SolverError::SolveFailed(status)),
        }

        // Extract solution
        let mut states = Vec::with_capacity(n + 1);
        let mut controls = Vec::with_capacity(n);

        for k in 0..=n {
            let x = capsule.get_state(k);
            if let Some(state) = OcpState::from_vector(&x, self.ocp.num_quadrotors) {
                states.push(state);
            }
        }

        for k in 0..n {
            let u = capsule.get_control(k);
            if let Some(control) = OcpControl::from_vector(&u, self.ocp.num_quadrotors) {
                controls.push(control);
            }
        }

        // Store for warm-start
        self.warm_start_state = Some(states.iter().map(|s| s.to_vector()).collect());
        self.warm_start_control = Some(controls.iter().map(|c| c.to_vector()).collect());

        // Build trajectory
        let times: Vec<f64> = self.ocp.config.horizon.compute_intervals()
            .iter()
            .scan(0.0, |acc, &dt| {
                let t = *acc;
                *acc += dt;
                Some(t)
            })
            .chain(std::iter::once(self.ocp.config.horizon.horizon_time))
            .collect();

        Ok(PlannedTrajectory {
            times,
            states,
            controls,
            generated_at: 0.0, // Should be set by caller
            is_valid: true,
        })
    }

    #[cfg(not(feature = "acados"))]
    fn solve_fallback(
        &mut self,
        initial_state: &OcpState,
        reference: &[ReferencePoint],
    ) -> Result<PlannedTrajectory, SolverError> {
        // Simple fallback: interpolate toward reference
        // This is NOT optimal but allows testing without ACADOS

        let n = self.ocp.n();
        let horizon = self.ocp.config.horizon.horizon_time;

        let times: Vec<f64> = self.ocp.config.horizon.compute_intervals()
            .iter()
            .scan(0.0, |acc, &dt| {
                let t = *acc;
                *acc += dt;
                Some(t)
            })
            .chain(std::iter::once(horizon))
            .collect();

        let mut states = Vec::with_capacity(n + 1);
        let mut controls = Vec::with_capacity(n);

        // Start from initial state
        let mut current = initial_state.clone();
        states.push(current.clone());

        for k in 0..n {
            let ref_k = &reference[k.min(reference.len() - 1)];

            // Simple proportional interpolation toward reference
            let alpha = (k + 1) as f64 / n as f64;

            current.load_position = initial_state.load_position.lerp(&ref_k.position, alpha);
            current.load_velocity = initial_state.load_velocity.lerp(&ref_k.velocity, alpha * 0.5);
            current.load_orientation = initial_state.load_orientation.slerp(&ref_k.orientation, alpha);
            current.load_angular_velocity = initial_state.load_angular_velocity.lerp(&ref_k.angular_velocity, alpha * 0.5);

            states.push(current.clone());
            controls.push(OcpControl::new(self.ocp.num_quadrotors));
        }

        self.last_stats = SolveStatistics {
            sqp_iterations: 0,
            qp_iterations: 0,
            solve_time_ms: 0.0,
            kkt_residual: 0.0,
            objective: 0.0,
        };

        Ok(PlannedTrajectory {
            times,
            states,
            controls,
            generated_at: 0.0,
            is_valid: true,
        })
    }
}

impl Drop for AcadosSolver {
    fn drop(&mut self) {
        // AcadosCapsule has its own Drop implementation that handles cleanup
    }
}

// Helper functions for reference vector packing

#[cfg(feature = "acados")]
fn pack_system_parameters(params: &SystemParameters) -> Vec<f64> {
    // Pack system parameters into the format expected by ACADOS
    // NP = 22: m_L(1) + J_L(9) + l(3) + rho(9) = 22
    // From caslo_ocp.py:
    //   p = [m_L, J_L[0..9], l[0..n], rho[0..3*n]]
    let mut p = Vec::with_capacity(acados_ffi::NP);

    // Load mass (1)
    p.push(params.load_mass);

    // Load inertia (9 elements, row-major)
    p.extend(params.load_inertia.iter());

    // Cable lengths (3 for 3 quads)
    for i in 0..3 {
        p.push(params.cable_lengths.get(i).copied().unwrap_or(1.0));
    }

    // Attachment points rho (3*3 = 9 elements)
    // Default: regular polygon pattern, radius 0.1m
    let num_quads = 3;
    let radius = 0.1;
    for i in 0..num_quads {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (num_quads as f64);
        p.push(radius * angle.cos()); // x
        p.push(radius * angle.sin()); // y
        p.push(0.0);                   // z
    }

    // Ensure exactly NP elements
    p.truncate(acados_ffi::NP);
    while p.len() < acados_ffi::NP {
        p.push(0.0);
    }

    p
}

#[cfg(feature = "acados")]
fn reference_to_vector(ref_point: &ReferencePoint, num_quads: usize) -> Vec<f64> {
    // Pack reference into the format expected by ACADOS cost function
    let mut y = Vec::with_capacity(12 + 4 * num_quads);
    y.extend(ref_point.position.iter());
    y.extend(ref_point.velocity.iter());
    // Quaternion vector part for simplified tracking
    y.push(ref_point.orientation.i);
    y.push(ref_point.orientation.j);
    y.push(ref_point.orientation.k);
    y.extend(ref_point.angular_velocity.iter());
    // Control reference (zeros for regularization)
    y.extend(vec![0.0; 4 * num_quads]);
    y
}

#[cfg(feature = "acados")]
fn reference_to_vector_terminal(ref_point: &ReferencePoint) -> Vec<f64> {
    let mut y = Vec::with_capacity(12);
    y.extend(ref_point.position.iter());
    y.extend(ref_point.velocity.iter());
    y.push(ref_point.orientation.i);
    y.push(ref_point.orientation.j);
    y.push(ref_point.orientation.k);
    y.extend(ref_point.angular_velocity.iter());
    y
}

/// Builder for creating solvers with different configurations
pub struct SolverBuilder {
    num_quadrotors: usize,
    config: PlannerConfig,
    system: Option<SystemParameters>,
}

impl SolverBuilder {
    pub fn new(num_quadrotors: usize) -> Self {
        Self {
            num_quadrotors,
            config: PlannerConfig::default(),
            system: None,
        }
    }

    pub fn with_config(mut self, config: PlannerConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_system(mut self, system: SystemParameters) -> Self {
        self.system = Some(system);
        self
    }

    pub fn build(self) -> Result<AcadosSolver, SolverError> {
        let mut ocp = OcpDefinition::with_config(self.num_quadrotors, self.config);

        if let Some(system) = self.system {
            ocp.set_system(system);
        }

        AcadosSolver::new(ocp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_builder() {
        let solver = SolverBuilder::new(3)
            .with_config(PlannerConfig::default())
            .build();

        // Without ACADOS feature, this creates a stub solver
        assert!(solver.is_ok());
    }

    #[test]
    fn test_fallback_solver() {
        let mut solver = SolverBuilder::new(3).build().unwrap();

        let initial_state = OcpState::new(3);
        let reference: Vec<ReferencePoint> = (0..20)
            .map(|_| ReferencePoint {
                position: Vector3::new(1.0, 0.0, 2.0),
                ..Default::default()
            })
            .collect();

        let result = solver.solve(&initial_state, &reference, &SolveOptions::default());

        // Fallback should work without ACADOS
        #[cfg(not(feature = "acados"))]
        {
            assert!(result.is_ok());
            let traj = result.unwrap();
            assert!(traj.is_valid);
            assert_eq!(traj.states.len(), 21); // N+1 states
        }
    }

    #[test]
    fn test_solve_options_default() {
        let options = SolveOptions::default();
        assert_eq!(options.max_sqp_iter, 50);
        assert!(options.warm_start);
    }
}
