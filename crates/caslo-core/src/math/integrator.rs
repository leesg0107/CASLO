//! Numerical integration methods
//!
//! Implements Runge-Kutta 4th order (RK4) and other integration schemes
//! for solving the ODEs in the dynamics equations.

use nalgebra::{Vector3, Vector6, SVector};

/// Generic RK4 integrator for any state vector
///
/// Solves dx/dt = f(t, x) using 4th-order Runge-Kutta method.
///
/// # Arguments
/// * `x` - Current state
/// * `t` - Current time
/// * `dt` - Time step
/// * `f` - Derivative function f(t, x) -> dx/dt
///
/// # Returns
/// New state after integration
pub fn rk4<const N: usize, F>(
    x: &SVector<f64, N>,
    t: f64,
    dt: f64,
    f: F,
) -> SVector<f64, N>
where
    F: Fn(f64, &SVector<f64, N>) -> SVector<f64, N>,
{
    let k1 = f(t, x);
    let k2 = f(t + dt / 2.0, &(x + k1 * dt / 2.0));
    let k3 = f(t + dt / 2.0, &(x + k2 * dt / 2.0));
    let k4 = f(t + dt, &(x + k3 * dt));

    x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)
}

/// RK4 integrator for Vector3
pub fn rk4_vec3<F>(
    x: &Vector3<f64>,
    t: f64,
    dt: f64,
    f: F,
) -> Vector3<f64>
where
    F: Fn(f64, &Vector3<f64>) -> Vector3<f64>,
{
    let k1 = f(t, x);
    let k2 = f(t + dt / 2.0, &(x + k1 * dt / 2.0));
    let k3 = f(t + dt / 2.0, &(x + k2 * dt / 2.0));
    let k4 = f(t + dt, &(x + k3 * dt));

    x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)
}

/// RK4 integrator for Vector6 (position + velocity pairs)
pub fn rk4_vec6<F>(
    x: &Vector6<f64>,
    t: f64,
    dt: f64,
    f: F,
) -> Vector6<f64>
where
    F: Fn(f64, &Vector6<f64>) -> Vector6<f64>,
{
    let k1 = f(t, x);
    let k2 = f(t + dt / 2.0, &(x + k1 * dt / 2.0));
    let k3 = f(t + dt / 2.0, &(x + k2 * dt / 2.0));
    let k4 = f(t + dt, &(x + k3 * dt));

    x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)
}

/// Simple Euler integration (first-order)
///
/// Less accurate but faster than RK4. Use for quick estimates or
/// when the derivative is approximately constant over the time step.
pub fn euler<const N: usize, F>(
    x: &SVector<f64, N>,
    t: f64,
    dt: f64,
    f: F,
) -> SVector<f64, N>
where
    F: Fn(f64, &SVector<f64, N>) -> SVector<f64, N>,
{
    x + f(t, x) * dt
}

/// Euler integrator for Vector3
pub fn euler_vec3<F>(
    x: &Vector3<f64>,
    t: f64,
    dt: f64,
    f: F,
) -> Vector3<f64>
where
    F: Fn(f64, &Vector3<f64>) -> Vector3<f64>,
{
    x + f(t, x) * dt
}

/// Semi-implicit Euler for second-order systems
///
/// For systems with position and velocity, updates velocity first
/// then uses new velocity to update position. More stable than
/// explicit Euler for oscillatory systems.
///
/// # Arguments
/// * `pos` - Current position
/// * `vel` - Current velocity
/// * `acc` - Acceleration (derivative of velocity)
/// * `dt` - Time step
///
/// # Returns
/// (new_position, new_velocity)
pub fn semi_implicit_euler(
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    acc: &Vector3<f64>,
    dt: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let new_vel = vel + acc * dt;
    let new_pos = pos + new_vel * dt;
    (new_pos, new_vel)
}

/// Velocity Verlet integrator for second-order systems
///
/// More accurate than Euler methods for conservative systems.
/// Requires computing acceleration at both old and new positions.
///
/// # Arguments
/// * `pos` - Current position
/// * `vel` - Current velocity
/// * `acc` - Current acceleration
/// * `dt` - Time step
/// * `acc_fn` - Function to compute acceleration at new position
///
/// # Returns
/// (new_position, new_velocity, new_acceleration)
pub fn velocity_verlet<F>(
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    acc: &Vector3<f64>,
    dt: f64,
    acc_fn: F,
) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>)
where
    F: Fn(&Vector3<f64>, &Vector3<f64>) -> Vector3<f64>,
{
    // Update position
    let new_pos = pos + vel * dt + acc * (0.5 * dt * dt);

    // Estimate new velocity for acceleration calculation
    let vel_est = vel + acc * dt;

    // Calculate new acceleration
    let new_acc = acc_fn(&new_pos, &vel_est);

    // Update velocity using average acceleration
    let new_vel = vel + (acc + new_acc) * (0.5 * dt);

    (new_pos, new_vel, new_acc)
}

/// Integrate a scalar ODE using RK4
pub fn rk4_scalar<F>(x: f64, t: f64, dt: f64, f: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let k1 = f(t, x);
    let k2 = f(t + dt / 2.0, x + k1 * dt / 2.0);
    let k3 = f(t + dt / 2.0, x + k2 * dt / 2.0);
    let k4 = f(t + dt, x + k3 * dt);

    x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0)
}

/// Adaptive step size RK4-RK5 (Runge-Kutta-Fehlberg)
///
/// Estimates local error and adjusts step size accordingly.
///
/// # Arguments
/// * `x` - Current state
/// * `t` - Current time
/// * `dt` - Suggested time step
/// * `tol` - Error tolerance
/// * `f` - Derivative function
///
/// # Returns
/// (new_state, actual_dt_used, suggested_next_dt)
pub fn rkf45<const N: usize, F>(
    x: &SVector<f64, N>,
    t: f64,
    dt: f64,
    tol: f64,
    f: F,
) -> (SVector<f64, N>, f64, f64)
where
    F: Fn(f64, &SVector<f64, N>) -> SVector<f64, N>,
{
    // Fehlberg coefficients
    let k1 = f(t, x);
    let k2 = f(t + dt / 4.0, &(x + k1 * dt / 4.0));
    let k3 = f(
        t + 3.0 * dt / 8.0,
        &(x + k1 * (3.0 * dt / 32.0) + k2 * (9.0 * dt / 32.0)),
    );
    let k4 = f(
        t + 12.0 * dt / 13.0,
        &(x + k1 * (1932.0 * dt / 2197.0) - k2 * (7200.0 * dt / 2197.0) + k3 * (7296.0 * dt / 2197.0)),
    );
    let k5 = f(
        t + dt,
        &(x + k1 * (439.0 * dt / 216.0) - k2 * (8.0 * dt) + k3 * (3680.0 * dt / 513.0) - k4 * (845.0 * dt / 4104.0)),
    );
    let k6 = f(
        t + dt / 2.0,
        &(x - k1 * (8.0 * dt / 27.0) + k2 * (2.0 * dt) - k3 * (3544.0 * dt / 2565.0) + k4 * (1859.0 * dt / 4104.0) - k5 * (11.0 * dt / 40.0)),
    );

    // 4th order solution
    let x4 = x + (k1 * (25.0 / 216.0) + k3 * (1408.0 / 2565.0) + k4 * (2197.0 / 4104.0) - k5 * (1.0 / 5.0)) * dt;

    // 5th order solution
    let x5 = x + (k1 * (16.0 / 135.0) + k3 * (6656.0 / 12825.0) + k4 * (28561.0 / 56430.0) - k5 * (9.0 / 50.0) + k6 * (2.0 / 55.0)) * dt;

    // Error estimate
    let error = (x5 - x4).norm();

    // Step size adjustment
    let s = if error > 1e-10 {
        0.84 * (tol * dt / error).powf(0.25)
    } else {
        2.0 // Double step if error is negligible
    };

    let new_dt = (s * dt).clamp(dt * 0.1, dt * 4.0);

    if error < tol {
        (x5, dt, new_dt)
    } else {
        // Retry with smaller step
        rkf45(x, t, dt * s.min(0.5), tol, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_rk4_exponential_decay() {
        // Solve dx/dt = -x with x(0) = 1
        // Exact solution: x(t) = e^(-t)
        let x0 = SVector::<f64, 1>::new(1.0);
        let dt = 0.01;
        let mut x = x0;
        let mut t = 0.0;

        for _ in 0..100 {
            x = rk4(&x, t, dt, |_t, x| -x.clone());
            t += dt;
        }

        let exact = (-1.0_f64).exp();
        assert_relative_eq!(x[0], exact, epsilon = 1e-6);
    }

    #[test]
    fn test_rk4_harmonic_oscillator() {
        // Solve d²x/dt² = -x (simple harmonic oscillator)
        // State: [x, v] where v = dx/dt
        // dx/dt = v, dv/dt = -x
        // With x(0) = 1, v(0) = 0: x(t) = cos(t)

        let x0 = SVector::<f64, 2>::new(1.0, 0.0);
        let dt = 0.001;
        let mut x = x0;
        let mut t = 0.0;

        let target_t = PI / 2.0;
        let steps = (target_t / dt) as usize;

        for _ in 0..steps {
            x = rk4(&x, t, dt, |_t, state| {
                SVector::<f64, 2>::new(state[1], -state[0])
            });
            t += dt;
        }

        // At t = π/2, cos(t) ≈ 0, sin(t) ≈ 1
        // So x ≈ 0, v ≈ -1
        assert_relative_eq!(x[0], 0.0, epsilon = 1e-3);
        assert_relative_eq!(x[1], -1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_rk4_vec3_projectile() {
        // Free fall: dv/dt = g = [0, 0, -9.81]
        let v0 = Vector3::new(10.0, 0.0, 0.0);
        let g = Vector3::new(0.0, 0.0, -9.81);
        let dt = 0.1;
        let t = 0.0;

        let v_new = rk4_vec3(&v0, t, dt, |_t, _v| g);

        // v = v0 + g*t
        let expected = v0 + g * dt;
        assert_relative_eq!(v_new, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_euler_linear() {
        // Solve dx/dt = 2 with x(0) = 0
        // Exact: x(t) = 2t
        let x0 = SVector::<f64, 1>::new(0.0);
        let dt = 0.1;

        let x = euler(&x0, 0.0, dt, |_t, _x| SVector::<f64, 1>::new(2.0));

        assert_relative_eq!(x[0], 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_semi_implicit_euler() {
        // Free fall test
        let pos = Vector3::new(0.0, 0.0, 100.0);
        let vel = Vector3::zeros();
        let acc = Vector3::new(0.0, 0.0, -9.81);
        let dt = 0.1;

        let (new_pos, new_vel) = semi_implicit_euler(&pos, &vel, &acc, dt);

        // v_new = 0 + (-9.81)*0.1 = -0.981
        assert_relative_eq!(new_vel.z, -0.981, epsilon = 1e-10);

        // p_new = 100 + (-0.981)*0.1 = 99.9019
        assert_relative_eq!(new_pos.z, 100.0 - 0.981 * 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_velocity_verlet_constant_acceleration() {
        // Constant acceleration: a = [0, 0, -9.81]
        let pos = Vector3::new(0.0, 0.0, 100.0);
        let vel = Vector3::new(10.0, 0.0, 0.0);
        let acc = Vector3::new(0.0, 0.0, -9.81);
        let dt = 0.1;

        let (new_pos, new_vel, new_acc) = velocity_verlet(
            &pos, &vel, &acc, dt,
            |_p, _v| Vector3::new(0.0, 0.0, -9.81)
        );

        // With constant acceleration, Verlet should give exact results
        // x = x0 + v0*t + 0.5*a*t²
        let expected_pos = pos + vel * dt + acc * 0.5 * dt * dt;
        let expected_vel = vel + acc * dt;

        assert_relative_eq!(new_pos, expected_pos, epsilon = 1e-10);
        assert_relative_eq!(new_vel, expected_vel, epsilon = 1e-10);
        assert_relative_eq!(new_acc, acc, epsilon = 1e-10);
    }

    #[test]
    fn test_rk4_scalar() {
        // Solve dx/dt = x with x(0) = 1
        // Exact: x(t) = e^t
        let mut x = 1.0;
        let dt = 0.01;
        let mut t = 0.0;

        for _ in 0..100 {
            x = rk4_scalar(x, t, dt, |_t, x| x);
            t += dt;
        }

        let exact = 1.0_f64.exp();
        assert_relative_eq!(x, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_rk4_vs_euler_accuracy() {
        // Compare RK4 and Euler for the same problem
        // Solve dx/dt = -x with x(0) = 1, dt = 0.1
        let x0 = SVector::<f64, 1>::new(1.0);
        let dt = 0.1;

        let mut x_rk4 = x0;
        let mut x_euler = x0;
        let mut t = 0.0;

        for _ in 0..10 {
            x_rk4 = rk4(&x_rk4, t, dt, |_t, x| -x.clone());
            x_euler = euler(&x_euler, t, dt, |_t, x| -x.clone());
            t += dt;
        }

        let exact = (-1.0_f64).exp();
        let rk4_error = (x_rk4[0] - exact).abs();
        let euler_error = (x_euler[0] - exact).abs();

        // RK4 should be much more accurate than Euler
        assert!(rk4_error < euler_error / 100.0);
    }

    #[test]
    fn test_rkf45_adaptive() {
        // Test adaptive step size with harmonic oscillator
        let x0 = SVector::<f64, 2>::new(1.0, 0.0);
        let dt = 0.5; // Large initial step
        let tol = 1e-6;

        let (x_new, dt_used, dt_next) = rkf45(&x0, 0.0, dt, tol, |_t, state| {
            SVector::<f64, 2>::new(state[1], -state[0])
        });

        // Should produce a valid result
        assert!(x_new[0].is_finite());
        assert!(x_new[1].is_finite());
        assert!(dt_used > 0.0);
        assert!(dt_next > 0.0);
    }
}
