//! Minimal test that replicates exactly what the C test does
//! This test should work if the FFI bindings are correct

#[cfg(feature = "acados")]
mod tests {
    use caslo_planner::acados_ffi::{AcadosCapsule, NX, NU, N};

    #[test]
    fn test_acados_like_c() {
        println!("Creating capsule...");
        let mut capsule = AcadosCapsule::new().expect("Failed to create capsule");
        println!("Capsule created");

        // Exactly matching the C test values
        let mut x0 = vec![0.0; NX];
        // quaternion w=1 at index 6
        x0[6] = 1.0;
        // cable directions pointing up (-Z in NED) at indices 15, 18, 21
        x0[15] = -1.0;
        x0[18] = -1.0;
        x0[21] = -1.0;
        // tensions at indices 31, 32, 33
        x0[31] = 5.0;
        x0[32] = 5.0;
        x0[33] = 5.0;

        println!("x0[0..10] = {:?}", &x0[0..10]);
        println!("x0[13..22] = {:?}", &x0[13..22]);
        println!("x0[31..34] = {:?}", &x0[31..34]);

        // Zero control
        let u0 = vec![0.0; NU];

        // Set initial state constraint
        println!("Setting initial state...");
        capsule.set_initial_state(&x0).expect("Failed to set initial state");
        println!("Initial state set");

        // Initialize trajectory (like C test does)
        println!("Initializing trajectory...");
        capsule.initialize_trajectory(&x0, &u0).expect("Failed to initialize trajectory");
        println!("Trajectory initialized");

        // Solve
        println!("Calling solve...");
        let status = capsule.solve().expect("Solve failed");
        println!("Solve returned status: {}", status);

        // Get results
        let solve_time = capsule.get_solve_time();
        let sqp_iters = capsule.get_sqp_iterations();
        println!("Solve time: {:.4} ms, SQP iterations: {}", solve_time * 1000.0, sqp_iters);

        // Check solution
        let x1 = capsule.get_state(1);
        println!("x[1] = {:?}", &x1[0..10]);

        assert_eq!(status, 0, "Solver should return success status");
    }
}
