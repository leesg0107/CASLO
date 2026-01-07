//! Sensor models for simulation
//!
//! Provides noisy sensor measurements for realistic simulation.

use nalgebra::{Vector3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::dynamics::{QuadrotorState, LoadState};

/// Simple random number generator (xorshift)
/// Note: For production, use a proper RNG crate
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    /// Generate next random u64
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generate uniform random f64 in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Generate Gaussian random number using Box-Muller transform
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-10); // Avoid log(0)
        let u2 = self.next_f64();

        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Generate Gaussian random Vector3
    pub fn next_gaussian_vec3(&mut self, std_dev: f64) -> Vector3<f64> {
        Vector3::new(
            self.next_gaussian() * std_dev,
            self.next_gaussian() * std_dev,
            self.next_gaussian() * std_dev,
        )
    }
}

impl Default for SimpleRng {
    fn default() -> Self {
        Self::new(12345)
    }
}

/// Position sensor model (e.g., motion capture, GPS)
#[derive(Debug, Clone)]
pub struct PositionSensor {
    /// Noise standard deviation [m]
    noise_std: f64,
    /// Random number generator
    rng: SimpleRng,
}

impl PositionSensor {
    pub fn new(noise_std: f64, seed: u64) -> Self {
        Self {
            noise_std,
            rng: SimpleRng::new(seed),
        }
    }

    /// Get noisy position measurement
    pub fn measure(&mut self, true_position: &Vector3<f64>) -> Vector3<f64> {
        true_position + self.rng.next_gaussian_vec3(self.noise_std)
    }
}

/// IMU (Inertial Measurement Unit) sensor model
#[derive(Debug, Clone)]
pub struct ImuSensor {
    /// Accelerometer noise std dev [m/s²]
    accel_noise_std: f64,
    /// Gyroscope noise std dev [rad/s]
    gyro_noise_std: f64,
    /// Accelerometer bias
    accel_bias: Vector3<f64>,
    /// Gyroscope bias
    gyro_bias: Vector3<f64>,
    /// Random number generator
    rng: SimpleRng,
}

impl ImuSensor {
    pub fn new(accel_noise_std: f64, gyro_noise_std: f64, seed: u64) -> Self {
        Self {
            accel_noise_std,
            gyro_noise_std,
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
            rng: SimpleRng::new(seed),
        }
    }

    /// Set accelerometer bias
    pub fn set_accel_bias(&mut self, bias: Vector3<f64>) {
        self.accel_bias = bias;
    }

    /// Set gyroscope bias
    pub fn set_gyro_bias(&mut self, bias: Vector3<f64>) {
        self.gyro_bias = bias;
    }

    /// Measure acceleration (body frame)
    ///
    /// Returns specific force = acceleration - gravity (in body frame)
    pub fn measure_acceleration(
        &mut self,
        true_acceleration: &Vector3<f64>,
        orientation: &UnitQuaternion<f64>,
        gravity: f64,
    ) -> Vector3<f64> {
        // Gravity in world frame (NED: +z is down)
        let gravity_world = Vector3::new(0.0, 0.0, gravity);

        // Specific force in world frame
        let specific_force_world = true_acceleration - gravity_world;

        // Transform to body frame
        let specific_force_body = orientation.inverse() * specific_force_world;

        // Add noise and bias
        specific_force_body + self.accel_bias + self.rng.next_gaussian_vec3(self.accel_noise_std)
    }

    /// Measure angular velocity (body frame)
    pub fn measure_angular_velocity(&mut self, true_omega: &Vector3<f64>) -> Vector3<f64> {
        true_omega + self.gyro_bias + self.rng.next_gaussian_vec3(self.gyro_noise_std)
    }
}

/// IMU measurement
#[derive(Debug, Clone)]
pub struct ImuMeasurement {
    /// Acceleration (specific force, body frame) [m/s²]
    pub acceleration: Vector3<f64>,
    /// Angular velocity (body frame) [rad/s]
    pub angular_velocity: Vector3<f64>,
}

/// Complete sensor suite for a quadrotor
#[derive(Debug, Clone)]
pub struct QuadrotorSensors {
    /// Position sensor (e.g., motion capture)
    pub position: PositionSensor,
    /// IMU sensor
    pub imu: ImuSensor,
}

impl QuadrotorSensors {
    pub fn new(pos_noise: f64, accel_noise: f64, gyro_noise: f64, seed: u64) -> Self {
        Self {
            position: PositionSensor::new(pos_noise, seed),
            imu: ImuSensor::new(accel_noise, gyro_noise, seed + 1),
        }
    }

    /// Get all measurements for current state
    pub fn measure(
        &mut self,
        state: &QuadrotorState,
        acceleration: &Vector3<f64>,
        gravity: f64,
    ) -> QuadrotorMeasurement {
        QuadrotorMeasurement {
            position: self.position.measure(&state.position),
            imu: ImuMeasurement {
                acceleration: self.imu.measure_acceleration(
                    acceleration,
                    &state.orientation,
                    gravity,
                ),
                angular_velocity: self.imu.measure_angular_velocity(&state.angular_velocity),
            },
        }
    }
}

/// Complete quadrotor measurement
#[derive(Debug, Clone)]
pub struct QuadrotorMeasurement {
    /// Position measurement [m]
    pub position: Vector3<f64>,
    /// IMU measurement
    pub imu: ImuMeasurement,
}

/// Sensor suite for the complete system
#[derive(Debug, Clone)]
pub struct SystemSensors {
    /// Quadrotor sensors (one per quadrotor)
    pub quadrotors: Vec<QuadrotorSensors>,
}

impl SystemSensors {
    pub fn new(num_quadrotors: usize, config: &super::SensorConfig) -> Self {
        let quadrotors = (0..num_quadrotors)
            .map(|i| {
                QuadrotorSensors::new(
                    config.position_noise_std,
                    config.accel_noise_std,
                    config.gyro_noise_std,
                    (i * 1000) as u64,
                )
            })
            .collect();

        Self { quadrotors }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_uniform_range() {
        let mut rng = SimpleRng::new(12345);

        for _ in 0..100 {
            let x = rng.next_f64();
            assert!(x >= 0.0 && x < 1.0);
        }
    }

    #[test]
    fn test_gaussian_distribution() {
        let mut rng = SimpleRng::new(54321);

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let n = 10000;

        for _ in 0..n {
            let x = rng.next_gaussian();
            sum += x;
            sum_sq += x * x;
        }

        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;

        // Mean should be close to 0, variance close to 1
        assert!(mean.abs() < 0.1);
        assert!((variance - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_position_sensor_noise() {
        let mut sensor = PositionSensor::new(0.01, 12345);
        let true_pos = Vector3::new(1.0, 2.0, 3.0);

        let mut measurements = Vec::new();
        for _ in 0..100 {
            measurements.push(sensor.measure(&true_pos));
        }

        // Mean should be close to true position
        let mean: Vector3<f64> = measurements.iter().sum::<Vector3<f64>>() / 100.0;
        assert_relative_eq!(mean, true_pos, epsilon = 0.1);

        // Measurements should vary
        let first = measurements[0];
        let second = measurements[1];
        assert!((first - second).norm() > 0.0);
    }

    #[test]
    fn test_imu_sensor() {
        let mut imu = ImuSensor::new(0.1, 0.01, 12345);

        let true_accel = Vector3::new(0.0, 0.0, 9.81);
        let orientation = UnitQuaternion::identity();

        let measured = imu.measure_acceleration(&true_accel, &orientation, 9.81);

        // Specific force should be near zero when hovering
        assert!(measured.norm() < 1.0);
    }

    #[test]
    fn test_imu_bias() {
        let mut imu = ImuSensor::new(0.0, 0.0, 12345); // No noise
        imu.set_gyro_bias(Vector3::new(0.1, 0.0, 0.0));

        let true_omega = Vector3::zeros();
        let measured = imu.measure_angular_velocity(&true_omega);

        assert_relative_eq!(measured.x, 0.1, epsilon = 1e-10);
    }
}
