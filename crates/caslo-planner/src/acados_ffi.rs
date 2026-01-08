//! ACADOS FFI Bindings for CASLO 3-Quadrotor Solver
//!
//! This module provides raw C bindings to the generated ACADOS solver.
//! These are wrapped by the `AcadosSolver` struct in solver.rs.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::os::raw::{c_double, c_int, c_void};

/// ACADOS dimension constants from generated code
pub const NX: usize = 34;  // State dimension: 13 + 7*3
pub const NZ: usize = 0;   // Algebraic dimension
pub const NU: usize = 12;  // Control dimension: 4*3
pub const NP: usize = 22;  // Parameter dimension
pub const N: usize = 20;   // Horizon length
pub const NY: usize = 24;  // Cost output dimension (state + control selection)
pub const NYN: usize = 12; // Terminal cost output dimension
pub const NBX0: usize = 34; // Initial state constraints

// Opaque types from ACADOS
#[repr(C)]
pub struct ocp_nlp_in { _private: [u8; 0] }

#[repr(C)]
pub struct ocp_nlp_out { _private: [u8; 0] }

#[repr(C)]
pub struct ocp_nlp_solver { _private: [u8; 0] }

#[repr(C)]
pub struct ocp_nlp_config { _private: [u8; 0] }

#[repr(C)]
pub struct ocp_nlp_dims { _private: [u8; 0] }

#[repr(C)]
pub struct ocp_nlp_plan_t { _private: [u8; 0] }

/// Solver capsule - opaque handle to the ACADOS solver
#[repr(C)]
pub struct caslo_3quad_solver_capsule {
    _private: [u8; 0],
}

extern "C" {
    // Capsule lifecycle
    pub fn caslo_3quad_acados_create_capsule() -> *mut caslo_3quad_solver_capsule;
    pub fn caslo_3quad_acados_free_capsule(capsule: *mut caslo_3quad_solver_capsule) -> c_int;

    // Solver creation and reset
    pub fn caslo_3quad_acados_create(capsule: *mut caslo_3quad_solver_capsule) -> c_int;
    pub fn caslo_3quad_acados_create_with_discretization(
        capsule: *mut caslo_3quad_solver_capsule,
        n_time_steps: c_int,
        new_time_steps: *const c_double,
    ) -> c_int;
    pub fn caslo_3quad_acados_reset(
        capsule: *mut caslo_3quad_solver_capsule,
        reset_qp_solver_mem: c_int,
    ) -> c_int;
    pub fn caslo_3quad_acados_free(capsule: *mut caslo_3quad_solver_capsule) -> c_int;

    // Solving
    pub fn caslo_3quad_acados_solve(capsule: *mut caslo_3quad_solver_capsule) -> c_int;

    // Parameter updates
    pub fn caslo_3quad_acados_update_params(
        capsule: *mut caslo_3quad_solver_capsule,
        stage: c_int,
        value: *const c_double,
        np: c_int,
    ) -> c_int;

    // Accessors for internal structures
    pub fn caslo_3quad_acados_get_nlp_in(
        capsule: *mut caslo_3quad_solver_capsule,
    ) -> *mut ocp_nlp_in;
    pub fn caslo_3quad_acados_get_nlp_out(
        capsule: *mut caslo_3quad_solver_capsule,
    ) -> *mut ocp_nlp_out;
    pub fn caslo_3quad_acados_get_nlp_solver(
        capsule: *mut caslo_3quad_solver_capsule,
    ) -> *mut ocp_nlp_solver;
    pub fn caslo_3quad_acados_get_nlp_config(
        capsule: *mut caslo_3quad_solver_capsule,
    ) -> *mut ocp_nlp_config;
    pub fn caslo_3quad_acados_get_nlp_dims(
        capsule: *mut caslo_3quad_solver_capsule,
    ) -> *mut ocp_nlp_dims;
    pub fn caslo_3quad_acados_get_nlp_opts(
        capsule: *mut caslo_3quad_solver_capsule,
    ) -> *mut c_void;

    // Print statistics
    pub fn caslo_3quad_acados_print_stats(capsule: *mut caslo_3quad_solver_capsule);

    // ACADOS common functions for setting/getting data
    pub fn ocp_nlp_constraints_model_set(
        config: *mut ocp_nlp_config,
        dims: *mut ocp_nlp_dims,
        in_: *mut ocp_nlp_in,
        out: *mut ocp_nlp_out,
        stage: c_int,
        field: *const i8,
        value: *mut c_void,
    ) -> c_int;

    pub fn ocp_nlp_cost_model_set(
        config: *mut ocp_nlp_config,
        dims: *mut ocp_nlp_dims,
        in_: *mut ocp_nlp_in,
        stage: c_int,
        field: *const i8,
        value: *mut c_void,
    ) -> c_int;

    pub fn ocp_nlp_out_set(
        config: *mut ocp_nlp_config,
        dims: *mut ocp_nlp_dims,
        out: *mut ocp_nlp_out,
        in_: *mut ocp_nlp_in,
        stage: c_int,
        field: *const i8,
        value: *mut c_void,
    );

    pub fn ocp_nlp_out_get(
        config: *mut ocp_nlp_config,
        dims: *mut ocp_nlp_dims,
        out: *mut ocp_nlp_out,
        stage: c_int,
        field: *const i8,
        value: *mut c_void,
    );

    pub fn ocp_nlp_get(
        solver: *mut ocp_nlp_solver,
        field: *const i8,
        value: *mut c_void,
    );
}

/// Safe wrapper around ACADOS solver capsule
pub struct AcadosCapsule {
    capsule: *mut caslo_3quad_solver_capsule,
    config: *mut ocp_nlp_config,
    dims: *mut ocp_nlp_dims,
    nlp_in: *mut ocp_nlp_in,
    nlp_out: *mut ocp_nlp_out,
    solver: *mut ocp_nlp_solver,
}

impl AcadosCapsule {
    /// Create and initialize a new ACADOS solver
    pub fn new() -> Result<Self, i32> {
        unsafe {
            let capsule = caslo_3quad_acados_create_capsule();
            if capsule.is_null() {
                return Err(-1);
            }

            // Use create_with_discretization like the C test does
            // N=20 is the default horizon, null time_steps uses default discretization
            let status = caslo_3quad_acados_create_with_discretization(
                capsule,
                N as c_int,
                std::ptr::null(),
            );
            if status != 0 {
                caslo_3quad_acados_free_capsule(capsule);
                return Err(status);
            }

            let config = caslo_3quad_acados_get_nlp_config(capsule);
            let dims = caslo_3quad_acados_get_nlp_dims(capsule);
            let nlp_in = caslo_3quad_acados_get_nlp_in(capsule);
            let nlp_out = caslo_3quad_acados_get_nlp_out(capsule);
            let solver = caslo_3quad_acados_get_nlp_solver(capsule);

            Ok(Self {
                capsule,
                config,
                dims,
                nlp_in,
                nlp_out,
                solver,
            })
        }
    }

    /// Set the initial state constraint (x0)
    pub fn set_initial_state(&mut self, x0: &[f64]) -> Result<(), i32> {
        if x0.len() != NX {
            return Err(-1);
        }

        let field = b"lbx\0".as_ptr() as *const i8;
        unsafe {
            let status = ocp_nlp_constraints_model_set(
                self.config,
                self.dims,
                self.nlp_in,
                self.nlp_out,
                0, // stage 0
                field,
                x0.as_ptr() as *mut c_void,
            );
            if status != 0 {
                return Err(status);
            }

            let field = b"ubx\0".as_ptr() as *const i8;
            let status = ocp_nlp_constraints_model_set(
                self.config,
                self.dims,
                self.nlp_in,
                self.nlp_out,
                0,
                field,
                x0.as_ptr() as *mut c_void,
            );
            if status != 0 {
                return Err(status);
            }
        }
        Ok(())
    }

    /// Set reference trajectory at a stage
    /// For stages 0..N-1: y_ref should have NY (24) elements
    /// For terminal stage N: y_ref should have NYN (12) elements
    pub fn set_reference(&mut self, stage: usize, y_ref: &[f64]) -> Result<(), i32> {
        if stage > N {
            return Err(-1);
        }

        // Validate dimensions
        let expected_len = if stage == N { NYN } else { NY };
        if y_ref.len() != expected_len {
            return Err(-2);
        }

        // Use "yref" for all stages (NONLINEAR_LS cost type)
        let field = b"yref\0".as_ptr() as *const i8;

        unsafe {
            let status = ocp_nlp_cost_model_set(
                self.config,
                self.dims,
                self.nlp_in,
                stage as c_int,
                field,
                y_ref.as_ptr() as *mut c_void,
            );
            if status != 0 {
                return Err(status);
            }
        }
        Ok(())
    }

    /// Set state initialization for warm-starting
    pub fn set_state_init(&mut self, stage: usize, x: &[f64]) -> Result<(), i32> {
        if x.len() != NX || stage > N {
            return Err(-1);
        }

        let field = b"x\0".as_ptr() as *const i8;
        unsafe {
            ocp_nlp_out_set(
                self.config,
                self.dims,
                self.nlp_out,
                self.nlp_in,
                stage as c_int,
                field,
                x.as_ptr() as *mut c_void,
            );
        }
        Ok(())
    }

    /// Set control initialization for warm-starting
    pub fn set_control_init(&mut self, stage: usize, u: &[f64]) -> Result<(), i32> {
        if u.len() != NU || stage >= N {
            return Err(-1);
        }

        let field = b"u\0".as_ptr() as *const i8;
        unsafe {
            ocp_nlp_out_set(
                self.config,
                self.dims,
                self.nlp_out,
                self.nlp_in,
                stage as c_int,
                field,
                u.as_ptr() as *mut c_void,
            );
        }
        Ok(())
    }

    /// Set parameters at a stage
    pub fn set_parameters(&mut self, stage: usize, params: &[f64]) -> Result<(), i32> {
        if params.len() != NP || stage > N {
            return Err(-1);
        }

        unsafe {
            let status = caslo_3quad_acados_update_params(
                self.capsule,
                stage as c_int,
                params.as_ptr(),
                NP as c_int,
            );
            if status != 0 {
                return Err(status);
            }
        }
        Ok(())
    }

    /// Initialize trajectory with given state and control
    /// This should be called before solve() for the first solve or when warm start is not available
    pub fn initialize_trajectory(&mut self, x_init: &[f64], u_init: &[f64]) -> Result<(), i32> {
        if x_init.len() != NX || u_init.len() != NU {
            return Err(-1);
        }

        let x_field = b"x\0".as_ptr() as *const i8;
        let u_field = b"u\0".as_ptr() as *const i8;

        unsafe {
            // Initialize all shooting nodes
            for i in 0..N {
                ocp_nlp_out_set(
                    self.config,
                    self.dims,
                    self.nlp_out,
                    self.nlp_in,
                    i as c_int,
                    x_field,
                    x_init.as_ptr() as *mut c_void,
                );
                ocp_nlp_out_set(
                    self.config,
                    self.dims,
                    self.nlp_out,
                    self.nlp_in,
                    i as c_int,
                    u_field,
                    u_init.as_ptr() as *mut c_void,
                );
            }
            // Terminal state
            ocp_nlp_out_set(
                self.config,
                self.dims,
                self.nlp_out,
                self.nlp_in,
                N as c_int,
                x_field,
                x_init.as_ptr() as *mut c_void,
            );
        }
        Ok(())
    }

    /// Solve the OCP
    pub fn solve(&mut self) -> Result<i32, i32> {
        unsafe {
            let status = caslo_3quad_acados_solve(self.capsule);
            Ok(status)
        }
    }

    /// Get state solution at a stage
    pub fn get_state(&self, stage: usize) -> Vec<f64> {
        let mut x = vec![0.0; NX];
        let field = b"x\0".as_ptr() as *const i8;
        unsafe {
            ocp_nlp_out_get(
                self.config,
                self.dims,
                self.nlp_out,
                stage as c_int,
                field,
                x.as_mut_ptr() as *mut c_void,
            );
        }
        x
    }

    /// Get control solution at a stage
    pub fn get_control(&self, stage: usize) -> Vec<f64> {
        let mut u = vec![0.0; NU];
        let field = b"u\0".as_ptr() as *const i8;
        unsafe {
            ocp_nlp_out_get(
                self.config,
                self.dims,
                self.nlp_out,
                stage as c_int,
                field,
                u.as_mut_ptr() as *mut c_void,
            );
        }
        u
    }

    /// Get solve time in seconds
    pub fn get_solve_time(&self) -> f64 {
        let mut time = 0.0f64;
        let field = b"time_tot\0".as_ptr() as *const i8;
        unsafe {
            ocp_nlp_get(
                self.solver,
                field,
                &mut time as *mut f64 as *mut c_void,
            );
        }
        time
    }

    /// Get number of SQP iterations
    pub fn get_sqp_iterations(&self) -> i32 {
        let mut iters = 0i32;
        let field = b"sqp_iter\0".as_ptr() as *const i8;
        unsafe {
            ocp_nlp_get(
                self.solver,
                field,
                &mut iters as *mut i32 as *mut c_void,
            );
        }
        iters
    }

    /// Print solver statistics
    pub fn print_stats(&self) {
        unsafe {
            caslo_3quad_acados_print_stats(self.capsule);
        }
    }

    /// Reset solver
    pub fn reset(&mut self) -> Result<(), i32> {
        unsafe {
            let status = caslo_3quad_acados_reset(self.capsule, 1);
            if status != 0 {
                return Err(status);
            }
        }
        Ok(())
    }
}

impl Drop for AcadosCapsule {
    fn drop(&mut self) {
        unsafe {
            caslo_3quad_acados_free(self.capsule);
            caslo_3quad_acados_free_capsule(self.capsule);
        }
    }
}

// Safety: The capsule is only accessed from a single thread
unsafe impl Send for AcadosCapsule {}
