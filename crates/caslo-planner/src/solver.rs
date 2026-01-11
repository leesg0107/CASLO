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

/// ACADOS solver status codes (from acados/utils/types.h)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SolverStatus {
    Unknown = -1,          // ACADOS_UNKNOWN
    Success = 0,           // ACADOS_SUCCESS
    NaNDetected = 1,       // ACADOS_NAN_DETECTED
    MaxIterations = 2,     // ACADOS_MAXITER
    MinStep = 3,           // ACADOS_MINSTEP
    QpFailure = 4,         // ACADOS_QP_FAILURE
    Ready = 5,             // ACADOS_READY
    Unbounded = 6,         // ACADOS_UNBOUNDED
    Timeout = 7,           // ACADOS_TIMEOUT
    QpScalingFailed = 8,   // ACADOS_QPSCALING_BOUNDS_NOT_SATISFIED
    Infeasible = 9,        // ACADOS_INFEASIBLE
}

impl From<i32> for SolverStatus {
    fn from(code: i32) -> Self {
        match code {
            0 => SolverStatus::Success,
            1 => SolverStatus::NaNDetected,
            2 => SolverStatus::MaxIterations,
            3 => SolverStatus::MinStep,
            4 => SolverStatus::QpFailure,
            5 => SolverStatus::Ready,
            6 => SolverStatus::Unbounded,
            7 => SolverStatus::Timeout,
            8 => SolverStatus::QpScalingFailed,
            9 => SolverStatus::Infeasible,
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

/// Stored trajectory for resampling (paper approach)
///
/// The paper resamples the previous MPC solution for OCP initialization,
/// rather than using the raw simulation state which may have diverged.
#[derive(Debug, Clone)]
pub struct StoredTrajectory {
    /// Time at which this trajectory was generated
    pub generated_at: f64,
    /// Time points for each state in the trajectory
    pub times: Vec<f64>,
    /// State trajectory (N+1 states)
    pub states: Vec<OcpState>,
    /// Control trajectory (N controls)
    pub controls: Vec<OcpControl>,
}

impl StoredTrajectory {
    /// Resample the trajectory at a new time, shifting and extrapolating as needed
    ///
    /// From paper (page 12): "The other states (cable rate, cable tensions, and their
    /// higher-order derivatives) were directly estimated by resampling on the
    /// previously generated trajectory."
    ///
    /// # Arguments
    /// * `current_time` - Current simulation time
    /// * `dt` - Time step between samples
    /// * `n_samples` - Number of samples needed (N+1)
    /// * `measured_load_state` - Measured load state (position, velocity, orientation, angular_velocity)
    ///                           from simulation (these are observable, unlike cable derivatives)
    ///
    /// # Returns
    /// Resampled states for OCP initialization
    pub fn resample(
        &self,
        current_time: f64,
        dt: f64,
        n_samples: usize,
        measured_load_state: Option<&OcpState>,
    ) -> Vec<OcpState> {
        let mut resampled = Vec::with_capacity(n_samples);

        // Time offset from when trajectory was generated
        let time_offset = current_time - self.generated_at;

        for k in 0..n_samples {
            let target_time = time_offset + k as f64 * dt;

            // Find the interval in stored trajectory that contains target_time
            let state = self.interpolate_state(target_time);
            resampled.push(state);
        }

        // Override the first state's load pose/velocity with measured values
        // (load position/velocity are directly observable, cable derivatives are not)
        if let Some(measured) = measured_load_state {
            if let Some(first) = resampled.first_mut() {
                first.load_position = measured.load_position;
                first.load_velocity = measured.load_velocity;
                first.load_orientation = measured.load_orientation;
                first.load_angular_velocity = measured.load_angular_velocity;

                // Also update cable directions from measurement (observable)
                // but keep angular velocities, accelerations, tensions from resampled trajectory
                for (i, cable) in first.cables.iter_mut().enumerate() {
                    if i < measured.cables.len() {
                        cable.direction = measured.cables[i].direction;
                    }
                }
            }
        }

        resampled
    }

    /// Interpolate state at a given time offset within the stored trajectory
    fn interpolate_state(&self, target_time: f64) -> OcpState {
        // Handle edge cases
        if target_time <= 0.0 || self.times.is_empty() {
            return self.states.first().cloned().unwrap_or_else(|| OcpState::new(3));
        }

        let trajectory_duration = self.times.last().copied().unwrap_or(0.0);

        if target_time >= trajectory_duration {
            // Extrapolate beyond trajectory end using last state
            // For steady state, this is reasonable
            return self.states.last().cloned().unwrap_or_else(|| OcpState::new(3));
        }

        // Find interval [t_k, t_{k+1}] containing target_time
        let mut k = 0;
        for (i, &t) in self.times.iter().enumerate() {
            if t > target_time {
                k = i.saturating_sub(1);
                break;
            }
            k = i;
        }

        // Linear interpolation between states[k] and states[k+1]
        if k + 1 >= self.states.len() {
            return self.states.last().cloned().unwrap_or_else(|| OcpState::new(3));
        }

        let t_k = self.times[k];
        let t_k1 = self.times[k + 1];
        let alpha = if (t_k1 - t_k).abs() > 1e-10 {
            (target_time - t_k) / (t_k1 - t_k)
        } else {
            0.0
        };

        self.interpolate_states(&self.states[k], &self.states[k + 1], alpha)
    }

    /// Linear interpolation between two states
    fn interpolate_states(&self, s1: &OcpState, s2: &OcpState, alpha: f64) -> OcpState {
        let num_cables = s1.cables.len();

        OcpState {
            load_position: s1.load_position.lerp(&s2.load_position, alpha),
            load_velocity: s1.load_velocity.lerp(&s2.load_velocity, alpha),
            load_orientation: s1.load_orientation.slerp(&s2.load_orientation, alpha),
            load_angular_velocity: s1.load_angular_velocity.lerp(&s2.load_angular_velocity, alpha),
            cables: (0..num_cables).map(|i| {
                let c1 = &s1.cables[i];
                let c2 = &s2.cables[i];

                // Interpolate cable direction on S² using slerp-like approach
                // For small angles, linear interpolation + normalization works
                let dir_interp = c1.direction.lerp(&c2.direction, alpha);
                let direction = if dir_interp.norm() > 1e-6 {
                    dir_interp.normalize()
                } else {
                    Vector3::new(0.0, 0.0, -1.0)
                };

                crate::ocp::CableState {
                    direction,
                    angular_velocity: c1.angular_velocity.lerp(&c2.angular_velocity, alpha),
                    angular_acceleration: c1.angular_acceleration.lerp(&c2.angular_acceleration, alpha),
                    angular_jerk: c1.angular_jerk.lerp(&c2.angular_jerk, alpha),
                    tension: c1.tension + alpha * (c2.tension - c1.tension),
                    tension_rate: c1.tension_rate + alpha * (c2.tension_rate - c1.tension_rate),
                }
            }).collect(),
        }
    }

    /// Resample controls for warm-starting
    pub fn resample_controls(
        &self,
        current_time: f64,
        dt: f64,
        n_samples: usize,
    ) -> Vec<OcpControl> {
        let mut resampled = Vec::with_capacity(n_samples);
        let time_offset = current_time - self.generated_at;

        for k in 0..n_samples {
            let target_time = time_offset + k as f64 * dt;
            let control = self.interpolate_control(target_time);
            resampled.push(control);
        }

        resampled
    }

    fn interpolate_control(&self, target_time: f64) -> OcpControl {
        if target_time <= 0.0 || self.times.is_empty() || self.controls.is_empty() {
            return self.controls.first().cloned().unwrap_or_else(|| OcpControl::new(3));
        }

        let trajectory_duration = self.times.last().copied().unwrap_or(0.0);

        if target_time >= trajectory_duration {
            return self.controls.last().cloned().unwrap_or_else(|| OcpControl::new(3));
        }

        // Find interval
        let mut k = 0;
        for (i, &t) in self.times.iter().enumerate() {
            if t > target_time {
                k = i.saturating_sub(1);
                break;
            }
            k = i;
        }

        if k >= self.controls.len() {
            return self.controls.last().cloned().unwrap_or_else(|| OcpControl::new(3));
        }

        // Return control at interval start (zero-order hold)
        self.controls[k].clone()
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
    /// Stored trajectory for resampling (paper approach)
    stored_trajectory: Option<StoredTrajectory>,
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
                stored_trajectory: None,
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
                stored_trajectory: None,
            })
        }
    }

    /// Solve the OCP given current state and reference trajectory
    ///
    /// # Arguments
    /// * `initial_state` - Current system state (measured)
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
        // Use default time (0.0) for backwards compatibility
        self.solve_at_time(initial_state, reference, options, 0.0)
    }

    /// Solve the OCP with trajectory resampling (paper approach)
    ///
    /// From paper (page 12): "The other states (cable rate, cable tensions, and their
    /// higher-order derivatives) were directly estimated by resampling on the
    /// previously generated trajectory."
    ///
    /// # Arguments
    /// * `initial_state` - Current system state (measured load pose + cable directions)
    /// * `reference` - Reference trajectory to track
    /// * `options` - Solve options
    /// * `current_time` - Current simulation time for trajectory resampling
    ///
    /// # Returns
    /// Planned trajectory or error
    pub fn solve_at_time(
        &mut self,
        initial_state: &OcpState,
        reference: &[ReferencePoint],
        options: &SolveOptions,
        current_time: f64,
    ) -> Result<PlannedTrajectory, SolverError> {
        if reference.len() < self.ocp.n() {
            return Err(SolverError::ReferenceTrajectoryTooShort);
        }

        #[cfg(feature = "acados")]
        {
            self.solve_acados_with_resampling(initial_state, reference, options, current_time)
        }

        #[cfg(not(feature = "acados"))]
        {
            // Fallback: return a simple trajectory that moves toward the reference
            let _ = current_time; // Unused in fallback
            self.solve_fallback(initial_state, reference)
        }
    }

    /// Get the stored trajectory for external use (e.g., state estimation)
    pub fn stored_trajectory(&self) -> Option<&StoredTrajectory> {
        self.stored_trajectory.as_ref()
    }

    /// Clear stored trajectory (e.g., when resetting scenario)
    pub fn clear_stored_trajectory(&mut self) {
        self.stored_trajectory = None;
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
    fn solve_acados_with_resampling(
        &mut self,
        initial_state: &OcpState,
        reference: &[ReferencePoint],
        options: &SolveOptions,
        current_time: f64,
    ) -> Result<PlannedTrajectory, SolverError> {
        let capsule = self.capsule.as_mut()
            .ok_or(SolverError::SolverNotAvailable)?;

        let n = self.ocp.n();
        let dt = self.ocp.config.horizon.horizon_time / n as f64;

        // === Paper Approach: Trajectory Resampling ===
        // From paper (page 12): "The other states (cable rate, cable tensions, and their
        // higher-order derivatives) were directly estimated by resampling on the
        // previously generated trajectory."
        //
        // This means:
        // 1. Load pose/velocity: Use measured values (directly observable)
        // 2. Cable directions: Use measured values (observable from vision/encoders)
        // 3. Cable angular velocity, acceleration, tension, tension_rate: Resample from
        //    previous MPC trajectory (not directly observable, would diverge)

        let init_state_for_ocp = if options.warm_start {
            if let Some(ref stored) = self.stored_trajectory {
                // Resample from stored trajectory
                let resampled = stored.resample(current_time, dt, n + 1, Some(initial_state));

                // Use the first resampled state (which has measured load state but
                // resampled cable derivatives)
                if let Some(first) = resampled.first() {
                    eprintln!("ACADOS: Using resampled trajectory for initialization");
                    first.clone()
                } else {
                    initial_state.clone()
                }
            } else {
                // No stored trajectory yet - use equilibrium-based initialization
                // This provides dynamically consistent cable derivatives (all zero)
                // which is correct for starting from rest or near-hover conditions.
                eprintln!("ACADOS: No stored trajectory, using equilibrium-based init");

                let cable_directions: Vec<Vector3<f64>> = initial_state.cables.iter()
                    .map(|c| c.direction)
                    .collect();
                let cable_tensions: Vec<f64> = initial_state.cables.iter()
                    .map(|c| c.tension)
                    .collect();

                OcpState::from_measured_with_equilibrium_derivatives(
                    initial_state.load_position,
                    initial_state.load_velocity,
                    initial_state.load_orientation,
                    initial_state.load_angular_velocity,
                    &cable_directions,
                    &cable_tensions,
                )
            }
        } else {
            initial_state.clone()
        };

        // Set initial state constraint
        let x0 = init_state_for_ocp.to_vector();
        capsule.set_initial_state(&x0)
            .map_err(|e| SolverError::SolveFailed(e))?;

        // Set reference trajectory for each stage
        for k in 0..n {
            let ref_k = &reference[k.min(reference.len() - 1)];
            let y_ref = reference_to_vector(ref_k, self.ocp.num_quadrotors);
            capsule.set_reference(k, &y_ref)
                .map_err(|e| SolverError::SolveFailed(e))?;
        }

        // Set terminal reference
        let ref_e = &reference[(n - 1).min(reference.len() - 1)];
        let y_ref_e = reference_to_vector_terminal(ref_e);
        capsule.set_reference(n, &y_ref_e)
            .map_err(|e| SolverError::SolveFailed(e))?;

        // Initialize trajectory guess using resampled trajectory if available
        if options.warm_start {
            if let Some(ref stored) = self.stored_trajectory {
                // Resample full trajectory for warm-start
                let resampled_states = stored.resample(current_time, dt, n + 1, Some(initial_state));
                let resampled_controls = stored.resample_controls(current_time, dt, n);

                for (k, state) in resampled_states.iter().enumerate() {
                    if k <= n {
                        let x = state.to_vector();
                        let _ = capsule.set_state_init(k, &x);
                    }
                }
                for (k, control) in resampled_controls.iter().enumerate() {
                    if k < n {
                        let u = control.to_vector();
                        let _ = capsule.set_control_init(k, &u);
                    }
                }
                eprintln!("ACADOS: Warm-started with resampled trajectory");
            } else {
                // First solve - initialize with constant state
                let u0 = vec![0.0; acados_ffi::NU];
                capsule.initialize_trajectory(&x0, &u0)
                    .map_err(|e| SolverError::SolveFailed(e))?;
            }
        } else {
            // Cold start
            let u0 = vec![0.0; acados_ffi::NU];
            capsule.initialize_trajectory(&x0, &u0)
                .map_err(|e| SolverError::SolveFailed(e))?;
        }

        // Solve
        let status = capsule.solve()
            .map_err(|e| SolverError::SolveFailed(e))?;

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
            SolverStatus::MinStep => return Err(SolverError::SolveFailed(status)),
            SolverStatus::QpFailure => return Err(SolverError::QpSolverFailed),
            SolverStatus::NaNDetected => return Err(SolverError::SolveFailed(status)),
            SolverStatus::Infeasible => return Err(SolverError::SolveFailed(status)),
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

        // Build trajectory times
        let times: Vec<f64> = self.ocp.config.horizon.compute_intervals()
            .iter()
            .scan(0.0, |acc, &dt| {
                let t = *acc;
                *acc += dt;
                Some(t)
            })
            .chain(std::iter::once(self.ocp.config.horizon.horizon_time))
            .collect();

        // Store trajectory for future resampling (CRITICAL for paper approach)
        self.stored_trajectory = Some(StoredTrajectory {
            generated_at: current_time,
            times: times.clone(),
            states: states.clone(),
            controls: controls.clone(),
        });

        // Also store for legacy warm-start
        self.warm_start_state = Some(states.iter().map(|s| s.to_vector()).collect());
        self.warm_start_control = Some(controls.iter().map(|c| c.to_vector()).collect());

        eprintln!("ACADOS: Solve succeeded, status={}, time={:.2}ms",
            status, self.last_stats.solve_time_ms);

        Ok(PlannedTrajectory {
            times,
            states,
            controls,
            generated_at: current_time,
            is_valid: true,
        })
    }

    #[cfg(not(feature = "acados"))]
    fn solve_fallback(
        &mut self,
        initial_state: &OcpState,
        reference: &[ReferencePoint],
    ) -> Result<PlannedTrajectory, SolverError> {
        // Fallback: Return trajectory with is_valid = false
        // This signals to the caller that MPC did NOT succeed,
        // so the caller should use its own fallback controller (e.g., PD cascade)
        //
        // The fallback solver cannot compute proper optimal controls without
        // a real optimization solver, so we mark the trajectory as invalid.

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
            is_valid: false,  // Mark as INVALID so caller uses fallback controller
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

    // Cable lengths (n elements, where n=3)
    let num_quads = params.cable_lengths.len().min(3);
    for i in 0..3 {
        p.push(params.cable_lengths.get(i).copied().unwrap_or(1.0));
    }

    // Attachment points rho (3*n = 9 elements)
    // Use actual attachment points from params if available
    if params.attachment_points.len() >= num_quads {
        for i in 0..3 {
            if i < params.attachment_points.len() {
                let pt = &params.attachment_points[i];
                p.push(pt[0]); // x
                p.push(pt[1]); // y
                p.push(pt[2]); // z
            } else {
                // Fallback: regular polygon pattern
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / 3.0;
                p.push(0.1 * angle.cos());
                p.push(0.1 * angle.sin());
                p.push(0.0);
            }
        }
    } else {
        // Default: regular polygon pattern, radius 0.1m
        for i in 0..3 {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / 3.0;
            p.push(0.1 * angle.cos()); // x
            p.push(0.1 * angle.sin()); // y
            p.push(0.0);               // z
        }
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

        // Fallback solver returns a trajectory but marks it as invalid
        // because it's not a true MPC solution
        #[cfg(not(feature = "acados"))]
        {
            assert!(result.is_ok());
            let traj = result.unwrap();
            // Fallback trajectory is NOT valid (no MPC optimization)
            assert!(!traj.is_valid, "Fallback solver should return is_valid=false");
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
