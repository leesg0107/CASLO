# CASLO Architecture Comparison: Paper vs Implementation

This document compares the control architecture from the paper "Agile and cooperative aerial manipulation of a cable-suspended load" (Science Robotics, 2025) with the current CASLO implementation.

---

## Table of Contents
1. [Paper Architecture Overview](#1-paper-architecture-overview)
2. [Current CASLO Implementation](#2-current-caslo-implementation)
3. [Component-by-Component Comparison](#3-component-by-component-comparison)
4. [Gap Analysis](#4-gap-analysis)
5. [Implementation Roadmap](#5-implementation-roadmap)

---

## 1. Paper Architecture Overview

### 1.1 System Overview (Fig. 8 from paper)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      CENTRALIZED (Off-board @ 10Hz)                     │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              Online Kinodynamic Motion Planner (OCP)              │ │
│  │  ┌─────────────┐    ┌──────────────────────────────────────────┐ │ │
│  │  │   xinit     │───▶│     ACADOS SQP-RTI Solver                │ │ │
│  │  │  (from EKF  │    │  - State: Eq.1 (load-cable dynamics)     │ │ │
│  │  │   + resamp) │    │  - Input: γᵢ (angular snap), λᵢ (tension │ │ │
│  │  └─────────────┘    │           acceleration)                  │ │ │
│  │                      │  - Constraints: Eq.8-12                  │ │ │
│  │                      └──────────────────────────────────────────┘ │ │
│  │                                      │                            │ │
│  │                                      ▼                            │ │
│  │  ┌──────────────────────────────────────────────────────────────┐ │ │
│  │  │                    Kinematic Constraint (Eq. 5)              │ │ │
│  │  │              pᵢ = p + R(q)ρᵢ - lᵢsᵢ                          │ │ │
│  │  │         (Convert cable states → quadrotor trajectories)      │ │ │
│  │  └──────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              Load-Cable State Estimator (EKF)                     │ │
│  │  - Estimates: load pose, twist, cable directions                 │ │
│  │  - Inputs: quadrotor positions, velocities, IMU (accelerometer)  │ │
│  │  - Cable directions from: Eq.14 (IMU-based estimation)           │ │
│  │  - Initialization: Algorithm S1 (Kabsch-Umeyama)                 │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WiFi: Receding-horizon trajectories
                                    │       [pᵢ, vᵢ, v̇ᵢ, v⃛ᵢ] over 2 sec
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUADROTOR (On-board @ 300Hz)                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    Time-based Sampler                             │ │
│  │  - Linear interpolation between trajectory nodes                  │ │
│  │  - Continues sampling until new trajectory arrives                │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼ Single reference point             │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │           Trajectory-Tracking Controller (Eq. 15)                 │ │
│  │                                                                   │ │
│  │   Tᵢ,des·zᵢ,des/mᵢ = Kp(pᵢ,ref - pᵢ) + Kv(vᵢ,ref - vᵢ)          │ │
│  │                       + v̇ᵢ,ref + fext/mᵢ                         │ │
│  │                                                                   │ │
│  │   where fext = mᵢaᵢ,filtered - fᵢ,filtered (external force est.) │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼ Tᵢ,des, zᵢ,des                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │           Tilt-Prioritized Attitude Controller                    │ │
│  │  - Generates αᵢ,des from zᵢ,des, reference jerk, zero yaw rate   │ │
│  │  - Reference: Brescianini & D'Andrea (2018)                       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼ αᵢ,des (angular acceleration cmd) │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              INDI Low-Level Controller (Eq. S7-S10)               │ │
│  │                                                                   │ │
│  │   Rotor model:  [T; τ] = G₁u²ₘ + G₂u̇ₘ             (Eq. S7)       │ │
│  │   INDI torque:  τdes = τf + J(αdes - ω̇f)          (Eq. S9)       │ │
│  │   Rotor cmd:    [Tdes; τdes] = G₁u²c + Δt⁻¹G₂(uc - uc,k-1) (S8)  │ │
│  │                                                                   │ │
│  │   Key features:                                                   │ │
│  │   - Uses filtered gyroscope (ω̇f) and rotor speeds (uf)           │ │
│  │   - Compensates for unmodeled external torques                    │ │
│  │   - Sensor-based adaptive control                                 │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼ [ω₁, ω₂, ω₃, ω₄] rotor speeds     │
│                               ESC (DShot)                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Equations Summary

| Equation | Description | Location |
|----------|-------------|----------|
| **Eq. 1** | Load-cable state definition | Paper p.11 |
| **Eq. 2** | Load dynamics (6-DOF) | Paper p.11 |
| **Eq. 3** | Cable dynamics (3rd order) | Paper p.11 |
| **Eq. 4** | Quadrotor dynamics | Paper p.11 |
| **Eq. 5** | Kinematic constraint | Paper p.11 |
| **Eq. 6** | OCP formulation | Paper p.12 |
| **Eq. 8-12** | Path constraints (thrust, tension, collision, obstacle) | Paper p.13 |
| **Eq. 15** | Trajectory-tracking controller | Paper p.14 |
| **Eq. S1-S3** | Quadrotor jerk/angular jerk derivatives | Supp. p.2 |
| **Eq. S4-S6** | Angular velocity reference generation | Supp. p.2-3 |
| **Eq. S7-S10** | INDI controller | Supp. p.4 |

### 1.3 State Definition (Eq. 1)

```
x = [p, v, q, ω, s₁, r₁, ṙ₁, r̈₁, t₁, ṫ₁, ..., sₙ, rₙ, ṙₙ, r̈ₙ, tₙ, ṫₙ]ᵀ

Dimensions for n cables:
- Load: 13 (p:3, v:3, q:4, ω:3)
- Per cable: 11 (s:3, r:3, ṙ:3, r̈:3... wait, let me recalculate)

Actually from Eq. 3:
- sᵢ ∈ S² (direction, 3D but constrained to unit sphere)
- rᵢ ∈ ℝ³ (angular velocity)
- ṙᵢ ∈ ℝ³ (angular acceleration)
- r̈ᵢ - this is NOT in state, γᵢ = r⃛ᵢ is the INPUT
- tᵢ ∈ ℝ (tension)
- ṫᵢ ∈ ℝ (tension rate)

Per cable state: 3 + 3 + 3 + 1 + 1 = 11 dimensions
Total: 13 + 11n dimensions
```

### 1.4 Control Input (Eq. 3)

```
Control inputs (what MPC optimizes):
- γᵢ = r⃛ᵢ ∈ ℝ³  : angular snap (3rd derivative of cable direction)
- λᵢ = ẗᵢ ∈ ℝ   : tension acceleration (2nd derivative of tension)

Per cable: 4 control inputs
Total: 4n control inputs
```

---

## 2. Current CASLO Implementation

### 2.1 Implementation Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      CENTRALIZED (caslo-planner)                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              MPC Solver (ACADOS via Python codegen)               │ │
│  │  - State: 3rd order dynamics (matches paper)                      │ │
│  │  - Constraints: Thrust, tension, collision avoidance              │ │
│  │  - Output: Cable states (directions, tensions, derivatives)       │ │
│  │  Status: ✅ Implemented (caslo_ocp.py)                            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              Load-Cable State Estimator (EKF)                     │ │
│  │  Status: ✅ Implemented (load_estimator.rs)                       │ │
│  │  - Kabsch-Umeyama initialization: ✅                              │ │
│  │  - EKF prediction/update: ✅                                      │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              Kinematic Constraint (Eq. 5)                         │ │
│  │  Status: ✅ Implemented (constraint.rs)                           │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ ❌ PROBLEM: Currently sending cable
                                    │    states directly, not quadrotor
                                    │    trajectories!
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SIMULATION (visualize_sim.rs)                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              ❌ WRONG: Arbitrary PID Controller                   │ │
│  │                                                                   │ │
│  │   // Current code (INCORRECT):                                    │ │
│  │   tension_error = desired_tensions[i] - cable.tension;            │ │
│  │   tension_rate_des = tension_error * 50.0;  // arbitrary gain     │ │
│  │   tension_accel = tension_rate_error * 100.0;                     │ │
│  │                                                                   │ │
│  │   dir_error = cable.direction.cross(&desired_directions[i]);      │ │
│  │   angular_jerk = ... // arbitrary PID gains                       │ │
│  │                                                                   │ │
│  │   This is NOT how the paper works!                                │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              INDI Controller (indi.rs)                            │ │
│  │  Status: ⚠️ Partially implemented (attitude only)                 │ │
│  │  - Eq. S9 (τdes = τf + J(αdes - ω̇f)): ✅                         │ │
│  │  - NOT connected to simulation loop                               │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 File Locations

| Component | File | Status |
|-----------|------|--------|
| **Dynamics** | | |
| Load dynamics (Eq. 2) | `caslo-core/src/dynamics/load.rs` | ✅ Correct |
| Cable dynamics (Eq. 3) | `caslo-core/src/dynamics/cable.rs` | ✅ Correct |
| System dynamics | `caslo-core/src/dynamics/system.rs` | ✅ Correct |
| Quadrotor dynamics (Eq. 4) | `caslo-core/src/dynamics/quadrotor.rs` | ✅ Implemented |
| **Kinematics** | | |
| Kinematic constraint (Eq. 5) | `caslo-core/src/kinematics/constraint.rs` | ✅ Correct |
| **Control** | | |
| INDI controller (Eq. S9) | `caslo-core/src/control/indi.rs` | ⚠️ Partial |
| Trajectory tracking (Eq. 15) | `caslo-core/src/control/quadrotor_tracker.rs` | ✅ Implemented (미연결) |
| Attitude controller | `caslo-core/src/control/attitude.rs` | ✅ Implemented (미연결) |
| Trajectory sampler | `caslo-core/src/control/trajectory.rs` | ✅ Implemented (미연결) |
| **Estimation** | | |
| EKF estimator | `caslo-core/src/estimation/load_estimator.rs` | ✅ Implemented |
| Kabsch-Umeyama init | `caslo-core/src/estimation/load_estimator.rs` | ✅ Implemented |
| **Planner** | | |
| OCP formulation (Eq. 6) | `caslo-planner/codegen/caslo_ocp.py` | ✅ Implemented |
| Constraints (Eq. 8-12) | `caslo-planner/src/constraints.rs` | ✅ Implemented |
| **Simulation** | | |
| Main loop | `caslo-planner/examples/visualize_sim.rs` | ❌ Wrong controller |

---

## 3. Component-by-Component Comparison

### 3.1 Motion Planner (OCP)

| Aspect | Paper | CASLO | Match |
|--------|-------|-------|-------|
| State space (Eq. 1) | 13 + 11n dims | 13 + 11n dims | ✅ |
| Control input | γᵢ, λᵢ | γᵢ, λᵢ | ✅ |
| Solver | ACADOS SQP-RTI | ACADOS SQP-RTI | ✅ |
| Horizon | 2 sec, 20 nodes | 2 sec, 20 nodes | ✅ |
| Non-equidistant intervals | Yes | Yes | ✅ |
| Thrust constraints (Eq. 8) | Yes | Yes | ✅ |
| Tension constraints (Eq. 10) | Yes | Yes | ✅ |
| Collision avoidance (Eq. 11) | Yes | Yes | ✅ |
| Obstacle avoidance (Eq. 12) | Yes | Yes | ✅ |
| **Output conversion** | Cable → Quad traj via Eq.5 | ❌ Not done | ❌ |

### 3.2 Trajectory Tracking Controller

| Aspect | Paper | CASLO | Match |
|--------|-------|-------|-------|
| **Architecture** | Quadrotor tracks position trajectory | ✅ `QuadrotorTracker` 구현됨 | ✅ (미연결) |
| Reference type | pᵢ,ref, vᵢ,ref, v̇ᵢ,ref from Eq.5 | ✅ `QuadrotorTrajectoryRef` 타입 있음 | ✅ (미연결) |
| External force compensation | fext = mᵢaᵢ - fᵢ (IMU-based) | ✅ `ExternalForce` 구조체 있음 | ✅ (미연결) |
| Position control (Eq. 15) | PD + feedforward | ✅ `quadrotor_tracker.rs`에 구현됨 | ✅ (미연결) |
| Time-based sampler | 300Hz interpolation | ✅ `Trajectory::sample()` 있음 | ✅ (미연결) |

**중요 발견**: `quadrotor_tracker.rs`에 Eq. 15 구현이 **이미 있음**! 그러나 `visualize_sim.rs`에서 사용하지 않고 임의 PID 사용 중.

### 3.3 INDI Low-Level Controller

| Aspect | Paper | CASLO | Match |
|--------|-------|-------|-------|
| Rotor model (Eq. S7) | G₁, G₂ matrices | Not implemented | ❌ |
| Torque computation (Eq. S9) | τdes = τf + J(αdes - ω̇f) | ✅ Implemented | ✅ |
| Rotor command (Eq. S8) | Numerical solve | Not implemented | ❌ |
| Filtered measurements | ωf, uf | Not connected | ⚠️ |
| **Integration** | Used in control loop | Not connected | ❌ |

### 3.4 State Estimator

| Aspect | Paper | CASLO | Match |
|--------|-------|-------|-------|
| EKF for load pose | Yes | Yes | ✅ |
| Cable direction from IMU (Eq. 14) | s̃ᵢ = (mᵢaᵢ - Tᵢzᵢ - fa,i)/‖...‖ | Not implemented | ❌ |
| Kabsch-Umeyama init (Alg. S1) | Yes | Yes | ✅ |
| Spring-damper tension model | Yes | Yes | ✅ |

---

## 4. Gap Analysis

### 4.1 Critical Gaps (Causing Crashes)

#### Gap 1: 구현된 컴포넌트가 시뮬레이션에 연결되지 않음 (핵심 문제!)
**상황**: `QuadrotorTracker` (Eq. 15), `AttitudeController`, `Trajectory` 등이 이미 구현되어 있음
**문제**: `visualize_sim.rs`에서 이 컴포넌트들을 **사용하지 않고** 임의 PID 사용 중

```
구현된 컴포넌트 (미연결):
  ✅ QuadrotorTracker    (caslo-core/src/control/quadrotor_tracker.rs)
  ✅ AttitudeController  (caslo-core/src/control/attitude.rs)
  ✅ Trajectory + sample (caslo-core/src/control/trajectory.rs)
  ✅ INDI (부분)         (caslo-core/src/control/indi.rs)

현재 visualize_sim.rs:
  ❌ 위 컴포넌트 무시하고 임의 PID 사용 → Crash!
```

#### Gap 2: MPC 출력 → Quadrotor Trajectory 변환 누락
**Problem**: MPC output (cable states) not converted to quadrotor trajectories.
**Solution**: Eq.5의 시간 미분 (S1-S3)을 사용하여 quad trajectory 생성

```
필요한 변환:
  MPC 출력: [s, r, ṙ, r̈, t, ṫ] (케이블 상태)
        ↓ Eq. 5 + 미분
  Quad Trajectory: [p, v, a, j] (드론 위치/속도/가속도/jerk)
```

#### Gap 3: 컴포넌트 연결 누락
현재 `visualize_sim.rs`의 제어 흐름을 다음과 같이 수정 필요:

```
현재 (Wrong):
  MPC → Cable states → 임의 PID → Crash!

수정 필요 (Correct):
  MPC → Cable states → Eq.5 변환 → QuadrotorTracker → AttitudeController → INDI → Rotors
```

### 4.2 Medium Priority Gaps

| Gap | Impact | Effort |
|-----|--------|--------|
| Tilt-prioritized attitude controller | Smooth attitude transitions | Medium |
| INDI rotor allocation (Eq. S8) | Accurate motor commands | Medium |
| IMU-based cable direction (Eq. 14) | Better estimation | Low |

### 4.3 Low Priority Gaps

| Gap | Impact | Effort |
|-----|--------|--------|
| Filtered measurements for INDI | Noise rejection | Low |
| Aerodynamic drag model | High-speed accuracy | Low |

---

## 5. Implementation Roadmap

### Phase 1: 기존 컴포넌트 연결 (Critical) ⭐

**목표**: 이미 구현된 컴포넌트들을 시뮬레이션에 연결

1. **MPC 출력 → Quadrotor Trajectory 변환 구현**
   - Location: `caslo-core/src/kinematics/constraint.rs`
   - 추가 필요: `quadrotor_velocity()`, `quadrotor_acceleration()`, `quadrotor_jerk()`
   - Eq. S1 (supplementary materials) 사용

2. **visualize_sim.rs 수정**
   - 기존 임의 PID 제거
   - 이미 구현된 `QuadrotorTracker`, `AttitudeController` 사용
   ```rust
   // 현재 (잘못됨):
   // tension_error = desired - actual;  // 임의 PID

   // 수정 후:
   let quad_ref = constraint.to_quadrotor_trajectory(&mpc_output);
   let tracker_output = quadrotor_tracker.compute(&state, &quad_ref, &ext_force, dt);
   let attitude_output = attitude_controller.compute(&tracker_output);
   ```

3. **Trajectory Sampler 연결**
   - `Trajectory::sample()` 사용하여 300Hz 보간

### Phase 2: Complete INDI Implementation

1. **Rotor Model (Eq. S7)**
   - Add G₁, G₂ matrices
   - Thrust/torque from rotor speeds

2. **Rotor Allocation (Eq. S8)**
   - Numerical solver for rotor commands
   - Bounds checking

3. **Filtered Measurements**
   - Low-pass filter for gyroscope
   - Synchronized filtering for ωf and uf

### Phase 3: Improve Estimation

1. **IMU-based Cable Direction (Eq. 14)**
   - Estimate cable direction from accelerometer
   - Requires thrust and drag models

2. **Aerodynamic Drag Model (Eq. 13)**
   - Thrust coefficient identification
   - Drag coefficient matrix

---

## Appendix A: Key Equations Reference

### A.1 Load Dynamics (Eq. 2)
```
ṗ = v
v̇ = -(1/m)Σtᵢsᵢ + g
q̇ = (1/2)Λ(q)[0; ω]
Jω̇ = -ω×Jω + Σtᵢ(R(q)ᵀsᵢ × ρᵢ)
```

### A.2 Cable Dynamics (Eq. 3)
```
ṡᵢ = rᵢ × sᵢ
r⃛ᵢ = γᵢ  (control input)
ẗᵢ = λᵢ  (control input)
```

### A.3 Kinematic Constraint (Eq. 5)
```
pᵢ = p + R(q)ρᵢ - lᵢsᵢ
```

### A.4 Quadrotor Jerk (Eq. S1)
```
v⃛ᵢ = v⃛ + R(q){ω×[ω̇×ρᵢ + ω×(ω×ρᵢ)] + ω̈×ρᵢ + ω̇×(ω×ρᵢ) + ω×(ω̇×ρᵢ)}
     - lᵢ{r̈ᵢ×sᵢ + 2ṙᵢ×(rᵢ×sᵢ) + rᵢ×(ṙᵢ×sᵢ) + rᵢ×[rᵢ×(rᵢ×sᵢ)]}
```

### A.5 Trajectory Tracking (Eq. 15)
```
Tᵢ,des·zᵢ,des/mᵢ = Kp(pᵢ,ref - pᵢ) + Kv(vᵢ,ref - vᵢ) + v̇ᵢ,ref + fext/mᵢ
```

### A.6 INDI Torque (Eq. S9)
```
τdes = τf + J(αdes - ω̇f)
```

---

## Appendix B: Data Flow Comparison

### Paper Data Flow
```
Load Reference Pose
        │
        ▼
┌───────────────────┐
│  Motion Planner   │ (OCP @ 10Hz)
│  Output: X* =     │
│  [cable states    │
│   over horizon]   │
└───────────────────┘
        │
        ▼ Kinematic Constraint (Eq. 5 + derivatives)
┌───────────────────┐
│  Quad Trajectory  │
│  [pᵢ, vᵢ, v̇ᵢ,    │
│   v⃛ᵢ] per quad   │
└───────────────────┘
        │
        ▼ WiFi to each quadrotor
┌───────────────────┐
│  Time Sampler     │ (interpolate @ 300Hz)
│  Single ref point │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Trajectory Track  │ (Eq. 15)
│ Tᵢ,des, zᵢ,des   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Attitude Control  │ (tilt-prioritized)
│ αᵢ,des           │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ INDI Low-Level    │ (Eq. S9)
│ [ω₁,ω₂,ω₃,ω₄]    │
└───────────────────┘
        │
        ▼
     Motors
```

### Current CASLO Data Flow (UPDATED - 2026-01-10)
```
Load Reference Pose
        │
        ▼
┌───────────────────┐
│  Motion Planner   │ (OCP @ 10Hz or Fallback)
│  Output: X* =     │
│  [cable states    │
│   over horizon]   │
└───────────────────┘
        │
        ▼ ✅ KinematicConstraint (Eq.5 + S1)
┌───────────────────┐
│  Kinematic        │
│  Constraint       │
│  Cable → Quad     │
│  Trajectory       │
└───────────────────┘
        │
        ▼ QuadrotorTrajectoryRef
┌───────────────────┐
│ QuadrotorTracker  │ (Eq. 15)
│ - Position ctrl   │
│ - Attitude ctrl   │
│ - INDI            │
└───────────────────┘
        │
        ▼ TrackerOutput (thrust, direction)
┌───────────────────┐
│ Cable Control     │
│ - γᵢ (ang. jerk)  │
│ - λᵢ (ten. accel) │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  System Dynamics  │
│  (load + cables)  │
└───────────────────┘
```

---

## Appendix C: 구현 현황 요약 (2026-01-10 업데이트)

### ✅ 구현 및 연결 완료
| 컴포넌트 | 파일 | 비고 |
|---------|------|------|
| ACADOS OCP | `caslo_ocp.py` | 3차 동역학, 제약조건 완료 |
| Load/Cable Dynamics | `load.rs`, `cable.rs` | Eq. 2, 3 구현 |
| Kinematic Constraint | `constraint.rs` | **Eq. 5 + S1 미분** (p, v, a, j 계산) |
| EKF Estimator | `load_estimator.rs` | Kabsch-Umeyama 포함 |
| QuadrotorTracker | `quadrotor_tracker.rs` | **Eq. 15 - visualize_sim에 연결됨** |
| AttitudeController | `attitude.rs` | QuadrotorTracker 내부에서 사용 |
| Trajectory Sampler | `trajectory.rs` | 보간 기능 |
| INDI | `indi.rs` | QuadrotorTracker에 통합됨 |

### ⚠️ 추가 개선 필요
| 컴포넌트 | 설명 |
|---------|------|
| INDI Rotor Model | Eq. S7, S8 (G₁, G₂ 행렬) |
| Filtered Measurements | ωf, uf 필터링 |

### 📝 변경 내역 (2026-01-10)
- `visualize_sim.rs` 수정: 논문 기반 제어 아키텍처로 전환
  - `KinematicConstraint` 사용하여 MPC 출력 → Quadrotor Trajectory 변환
  - `QuadrotorTracker` (Eq. 15) 사용하여 trajectory tracking
  - 임의 PID 제거, 논문 흐름 구현

---

*Document created: 2026-01-10*
*Last updated: 2026-01-10*
