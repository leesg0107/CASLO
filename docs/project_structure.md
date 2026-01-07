# CASLO Project Structure

## Workspace 구조 (Cargo Workspace)

```
caslo/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── caslo-core/              # 핵심 물리/제어 라이브러리
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── math/            # 수학 유틸리티
│   │       ├── dynamics/        # 동역학 모델
│   │       ├── control/         # 제어기
│   │       ├── estimation/      # 상태 추정
│   │       └── simulation/      # 시뮬레이션 코어
│   │
│   └── caslo-viz/               # Bevy 시각화
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs
│           ├── app.rs           # Bevy App 설정
│           ├── rendering/       # 3D 렌더링
│           ├── ui/              # egui UI
│           └── camera.rs        # 카메라 컨트롤
│
├── examples/                     # 예제
│   ├── hover.rs                 # 호버링 테스트
│   ├── figure_eight.rs          # Figure-8 궤적 추적
│   └── obstacle_avoidance.rs    # 장애물 회피
│
└── docs/
    ├── paper_analysis.md
    └── project_structure.md
```

## caslo-core 상세 구조

```rust
// crates/caslo-core/src/lib.rs
pub mod math;
pub mod dynamics;
pub mod control;
pub mod estimation;
pub mod simulation;

// Re-exports
pub use dynamics::{LoadState, QuadrotorState, CableState, SystemState};
pub use simulation::{Simulator, SimConfig};
```

### math/ 모듈

```rust
// math/mod.rs
pub mod vector3;      // Vec3 wrapper (nalgebra)
pub mod quaternion;   // UnitQuaternion operations
pub mod rotation;     // SO(3) operations
pub mod integrator;   // RK4, etc.

// math/quaternion.rs
use nalgebra::{UnitQuaternion, Vector3, Vector4};

/// 쿼터니언 미분 (Eq. 2의 q̇ = 1/2 Λ(q)[0;ω])
pub fn quaternion_derivative(q: &UnitQuaternion<f64>, omega: &Vector3<f64>) -> Vector4<f64>;

/// 쿼터니언 곱셈 연산자 Λ(q)
pub fn quaternion_product_matrix(q: &UnitQuaternion<f64>) -> nalgebra::Matrix4<f64>;
```

### dynamics/ 모듈

```rust
// dynamics/mod.rs
pub mod load;
pub mod quadrotor;
pub mod cable;
pub mod system;
pub mod aerodynamics;

// dynamics/load.rs
use nalgebra::{Vector3, UnitQuaternion, Matrix3};

#[derive(Clone, Debug)]
pub struct LoadParams {
    pub mass: f64,                    // m [kg]
    pub inertia: Matrix3<f64>,        // J [kg·m²]
    pub attachment_points: Vec<Vector3<f64>>,  // ρᵢ in load frame
}

#[derive(Clone, Debug)]
pub struct LoadState {
    pub position: Vector3<f64>,       // p [m]
    pub velocity: Vector3<f64>,       // v [m/s]
    pub orientation: UnitQuaternion<f64>,  // q
    pub angular_velocity: Vector3<f64>,    // ω [rad/s] (load frame)
}

impl LoadState {
    /// Load dynamics (Eq. 2)
    /// Returns (v̇, ω̇)
    pub fn dynamics(
        &self,
        params: &LoadParams,
        cable_forces: &[(f64, Vector3<f64>)],  // (tᵢ, sᵢ)
        gravity: Vector3<f64>,
    ) -> (Vector3<f64>, Vector3<f64>);
}

// dynamics/cable.rs
#[derive(Clone, Debug)]
pub struct CableParams {
    pub length: f64,                  // lᵢ [m]
    pub attachment_offset: Vector3<f64>,  // 쿼드로터 CoG 기준 부착점
}

#[derive(Clone, Debug)]
pub struct CableState {
    pub direction: Vector3<f64>,      // sᵢ ∈ S² (unit vector, quad→load)
    pub angular_velocity: Vector3<f64>,    // rᵢ
    pub angular_acceleration: Vector3<f64>, // ṙᵢ
    pub angular_jerk: Vector3<f64>,        // r̈ᵢ
    pub tension: f64,                 // tᵢ [N]
    pub tension_rate: f64,            // ṫᵢ [N/s]
}

impl CableState {
    /// Cable kinematics (Eq. 3)
    /// Input: γᵢ (angular snap), λᵢ (tension acceleration)
    pub fn kinematics(&self, gamma: Vector3<f64>, lambda: f64) -> CableStateDot;
}

// dynamics/quadrotor.rs
#[derive(Clone, Debug)]
pub struct QuadrotorParams {
    pub mass: f64,                    // mᵢ [kg]
    pub inertia: Matrix3<f64>,        // Jᵢ [kg·m²]
    pub arm_length: f64,              // [m]
    pub thrust_coefficient: f64,      // cₜ
    pub torque_coefficient: f64,      // cₘ
    pub max_thrust: f64,              // Tmax [N]
    pub min_thrust: f64,              // Tmin [N]
}

#[derive(Clone, Debug)]
pub struct QuadrotorState {
    pub position: Vector3<f64>,       // pᵢ [m]
    pub velocity: Vector3<f64>,       // vᵢ [m/s]
    pub orientation: UnitQuaternion<f64>,  // qᵢ
    pub angular_velocity: Vector3<f64>,    // ωᵢ [rad/s] (body frame)
}

impl QuadrotorState {
    /// Quadrotor dynamics (Eq. 4)
    pub fn dynamics(
        &self,
        params: &QuadrotorParams,
        thrust: f64,                  // Tᵢ
        torque: Vector3<f64>,         // τᵢ
        cable_force: Vector3<f64>,    // tᵢsᵢ
        drag_force: Vector3<f64>,     // fₐ,ᵢ
        drag_torque: Vector3<f64>,    // τₐ,ᵢ
        gravity: Vector3<f64>,
    ) -> QuadrotorStateDot;

    /// Body z-axis (thrust direction)
    pub fn thrust_direction(&self) -> Vector3<f64>;
}

// dynamics/system.rs
#[derive(Clone, Debug)]
pub struct SystemState {
    pub load: LoadState,
    pub cables: Vec<CableState>,
    pub quadrotors: Vec<QuadrotorState>,
    pub time: f64,
}

#[derive(Clone, Debug)]
pub struct SystemParams {
    pub load: LoadParams,
    pub cables: Vec<CableParams>,
    pub quadrotors: Vec<QuadrotorParams>,
    pub gravity: Vector3<f64>,
}

impl SystemState {
    /// Kinematic constraint (Eq. 5): pᵢ = p + R(q)ρᵢ - lᵢsᵢ
    pub fn compute_quadrotor_position_from_load(
        load: &LoadState,
        cable: &CableState,
        cable_params: &CableParams,
        attachment_point: &Vector3<f64>,
    ) -> Vector3<f64>;

    /// Full system dynamics
    pub fn dynamics(&self, params: &SystemParams, input: &SystemInput) -> SystemStateDot;
}

/// OCP input: u = [γ₁, λ₁, ..., γₙ, λₙ]
#[derive(Clone, Debug)]
pub struct SystemInput {
    pub cable_angular_snaps: Vec<Vector3<f64>>,  // γᵢ
    pub cable_tension_accels: Vec<f64>,          // λᵢ
}
```

### control/ 모듈

```rust
// control/mod.rs
pub mod position;
pub mod attitude;
pub mod indi;
pub mod trajectory_tracker;

// control/indi.rs
/// INDI Low-Level Controller (Supplementary Methods Eq. S7-S10)
pub struct INDIController {
    pub g1: Matrix4<f64>,             // Control effectiveness (thrust/torque)
    pub g2: Matrix4<f64>,             // Rotor inertia effect
    pub inertia: Matrix3<f64>,        // J
    pub filter_cutoff: f64,           // Low-pass filter cutoff [Hz]

    // Internal state
    prev_rotor_cmd: Vector4<f64>,
    filtered_angular_accel: Vector3<f64>,
    filtered_torque: Vector3<f64>,
}

impl INDIController {
    /// Compute rotor speed commands
    pub fn compute(
        &mut self,
        thrust_cmd: f64,
        angular_accel_cmd: Vector3<f64>,
        gyro_measurement: Vector3<f64>,
        rotor_speeds: Vector4<f64>,
        dt: f64,
    ) -> Vector4<f64>;
}

// control/trajectory_tracker.rs
/// Trajectory tracking controller (Eq. 15)
pub struct TrajectoryTracker {
    pub kp: Matrix3<f64>,             // Position gain
    pub kv: Matrix3<f64>,             // Velocity gain
    pub indi: INDIController,
}

impl TrajectoryTracker {
    pub fn track(
        &mut self,
        state: &QuadrotorState,
        reference: &TrajectoryPoint,
        imu: &ImuMeasurement,
        dt: f64,
    ) -> MotorCommands;
}
```

### simulation/ 모듈

```rust
// simulation/mod.rs
pub mod simulator;
pub mod config;
pub mod sensors;

// simulation/simulator.rs
pub struct Simulator {
    pub state: SystemState,
    pub params: SystemParams,
    pub config: SimConfig,
}

impl Simulator {
    pub fn new(params: SystemParams, config: SimConfig) -> Self;

    /// Step simulation by dt
    pub fn step(&mut self, input: &SystemInput, dt: f64);

    /// Get sensor readings (with optional noise)
    pub fn get_imu(&self, quad_idx: usize) -> ImuMeasurement;

    /// Get current state (for visualization)
    pub fn get_viz_state(&self) -> VizState;
}

// simulation/config.rs
#[derive(Clone, Debug)]
pub struct SimConfig {
    pub dt: f64,                      // Integration timestep [s]
    pub integrator: IntegratorType,   // RK4, Euler, etc.
    pub sensor_noise: bool,
    pub imu_noise_params: ImuNoiseParams,
}
```

## caslo-viz 상세 구조

```rust
// crates/caslo-viz/src/main.rs
use bevy::prelude::*;
use caslo_core::{Simulator, SimConfig, SystemParams};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(CasloVizPlugin)
        .run();
}

// crates/caslo-viz/src/app.rs
pub struct CasloVizPlugin;

impl Plugin for CasloVizPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<SimulatorResource>()
            .add_systems(Startup, setup_scene)
            .add_systems(Update, (
                step_simulation,
                update_quadrotor_transforms,
                update_load_transform,
                update_cable_meshes,
                update_ui,
            ));
    }
}

// crates/caslo-viz/src/rendering/mod.rs
pub mod quadrotor;
pub mod load;
pub mod cable;
pub mod trajectory;

// Quadrotor 3D model component
#[derive(Component)]
pub struct QuadrotorViz {
    pub index: usize,
}

// Load 3D model component
#[derive(Component)]
pub struct LoadViz;

// Cable line rendering
#[derive(Component)]
pub struct CableViz {
    pub index: usize,
}
```

## 의존성 (Cargo.toml)

```toml
# Workspace root Cargo.toml
[workspace]
resolver = "2"
members = ["crates/*"]

[workspace.dependencies]
nalgebra = "0.33"
serde = { version = "1.0", features = ["derive"] }
thiserror = "1.0"

# caslo-core/Cargo.toml
[package]
name = "caslo-core"
version = "0.1.0"
edition = "2021"

[dependencies]
nalgebra = { workspace = true }
serde = { workspace = true }
thiserror = { workspace = true }

[dev-dependencies]
approx = "0.5"

# caslo-viz/Cargo.toml
[package]
name = "caslo-viz"
version = "0.1.0"
edition = "2021"

[dependencies]
caslo-core = { path = "../caslo-core" }
bevy = "0.14"
bevy_egui = "0.28"
nalgebra = { workspace = true }
```

## 구현 로드맵

### Phase 1: Core Math & Dynamics (1주)
- [ ] quaternion operations
- [ ] SO(3) utilities
- [ ] LoadState + dynamics
- [ ] CableState + kinematics
- [ ] QuadrotorState + dynamics
- [ ] SystemState + kinematic constraints
- [ ] RK4 integrator

### Phase 2: Basic Simulation (3-4일)
- [ ] Simulator struct
- [ ] Simple hover test (no control)
- [ ] Verify dynamics correctness

### Phase 3: Bevy Visualization (1주)
- [ ] Basic scene setup
- [ ] Quadrotor 3D model (simple box/drone mesh)
- [ ] Load 3D model
- [ ] Cable line rendering
- [ ] Camera controls
- [ ] Basic UI (state display)

### Phase 4: Controllers (1주)
- [ ] Position controller (Eq. 15)
- [ ] Attitude controller
- [ ] INDI low-level controller
- [ ] Trajectory tracker

### Phase 5: State Estimation (1주)
- [ ] EKF implementation
- [ ] Load-cable state estimator
- [ ] Kabsch-Umeyama initializer

### Phase 6: MPC Planner (2주)
- [ ] OCP formulation
- [ ] Constraint implementation
- [ ] NLP solver integration (ipopt-rs or custom SQP)

### Phase 7: Full Integration (1주)
- [ ] Complete simulation loop
- [ ] Performance optimization
- [ ] Example scenarios
