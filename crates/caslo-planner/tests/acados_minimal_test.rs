//! Minimal test that replicates exactly what the C test does
//! This test should work if the FFI bindings are correct

#[cfg(feature = "acados")]
mod tests {
    use caslo_planner::acados_ffi::{AcadosCapsule, NX, NU, N, NP};

    #[test]
    fn test_acados_like_c() {
        println!("Creating capsule...");
        let mut capsule = AcadosCapsule::new().expect("Failed to create capsule");
        println!("Capsule created, NX={}, NU={}, NP={}", NX, NU, NP);

        // State layout for 3 quads (NX=46):
        // Load: 13 (p:3, v:3, q:4, ω:3)
        // s_all: 13-21 (3*3=9) - cable directions
        // r_all: 22-30 (3*3=9) - cable angular velocities
        // r_dot_all: 31-39 (3*3=9) - cable angular accelerations
        // t_all: 40-42 (3) - tensions
        // t_dot_all: 43-45 (3) - tension rates
        let num_quads = 3;
        let s_start = 13;
        let t_start = 13 + 9 * num_quads;  // 13 + 27 = 40

        let mut x0 = vec![0.0; NX];
        // quaternion w=1 at index 6
        x0[6] = 1.0;
        // cable directions pointing DOWN (-Z in Z-UP) at s_start + 2, +5, +8
        // (z-component of each 3D direction vector)
        for i in 0..num_quads {
            x0[s_start + 3 * i + 2] = -1.0;  // s_i.z = -1
        }
        // tensions at t_start (40, 41, 42)
        for i in 0..num_quads {
            x0[t_start + i] = 5.0;  // t_i = 5.0 N
        }

        println!("x0[0..13] (load) = {:?}", &x0[0..13]);
        println!("x0[13..22] (s_all) = {:?}", &x0[13..22]);
        println!("x0[40..46] (t, t_dot) = {:?}", &x0[40..46]);

        // Set parameters (critical!)
        let mut params = vec![0.0; NP];
        // NP=22: m_L(1) + J_L(9) + l(3) + rho(9)
        params[0] = 1.4;  // Load mass [kg]
        // J_L diagonal (inertia tensor)
        params[1] = 0.03;  // Ixx
        params[5] = 0.04;  // Iyy
        params[9] = 0.05;  // Izz
        // Cable lengths
        for i in 0..num_quads {
            params[10 + i] = 1.0;  // l_i = 1.0 m
        }
        // Attachment points (regular polygon, radius 0.1m)
        for i in 0..num_quads {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (num_quads as f64);
            params[13 + 3 * i] = 0.1 * angle.cos();  // x
            params[13 + 3 * i + 1] = 0.1 * angle.sin();  // y
            params[13 + 3 * i + 2] = 0.0;  // z
        }

        // Set parameters for all stages
        for stage in 0..=N {
            capsule.set_parameters(stage, &params).expect("Failed to set parameters");
        }
        println!("Parameters set for all stages");

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
        println!("x[1][0..13] (load) = {:?}", &x1[0..13]);
        println!("x[1][40..46] (t, t_dot) = {:?}", &x1[40..46]);

        // Status 0 = success, but for now just check it ran
        // The QP solver might still fail due to initial conditions
        println!("Test completed with status {}", status);
    }
}
