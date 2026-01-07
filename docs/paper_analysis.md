# CASLO: Cable-Suspended Load Control System
## 논문 분석: "Agile and Cooperative Aerial Manipulation of a Cable-Suspended Load"

### 논문 정보
- **제목**: Agile and Cooperative Aerial Manipulation of a Cable-Suspended Load
- **저자**: Sihao Sun, Xuerui Wang, Dario Sanalitro, Antonio Franchi, Marco Tognon, Javier Alonso-Mora
- **출판**: Science Robotics, Volume 10, Issue 107 (2025년 10월 29일)
- **DOI**: 10.1126/scirobotics.adu8015

---

## 1. 시스템 개요

### 1.1 문제 정의
여러 대의 쿼드로터가 케이블로 매달린 화물(load)을 협력하여 고속/고가속으로 조작하는 문제.

**핵심 성과:**
- 기존 방법 대비 **8배 이상** 높은 가속도 달성 (8 m/s² vs 0.5 m/s²)
- 최대 속도 5 m/s 이상
- 좁은 통로 고속 통과 가능
- 화물에 센서 부착 불필요

### 1.2 기존 방법의 한계 (Cascaded Control)
1. **Timescale separation 가정 실패**: 고속 비행 시 화물 동역학이 쿼드로터만큼 빠름
2. **정확한 동역학 모델 필요**: 화물 질량/관성 오차에 민감
3. **화물 센서 의존**: Motion capture 마커, 카메라 등 필요

### 1.3 제안된 프레임워크 (Trajectory-based)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Load Reference Pose                           │
│                    (Polynomial trajectory)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           Online Kinodynamic Motion Planner (10 Hz)              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Optimal Control Problem (OCP)                          │    │
│  │  - Load-cable 전체 시스템 동역학                          │    │
│  │  - 제약조건: 추력한계, 케이블장력, 충돌회피, 장애물회피      │    │
│  │  - Solver: ACADOS (SQP + RTI)                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│              Receding-horizon trajectories (2s horizon)          │
└─────────────────────────────────────────────────────────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Quadrotor 1   │ │   Quadrotor 2   │ │   Quadrotor N   │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │Time Sampler │ │ │ │Time Sampler │ │ │ │Time Sampler │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│       │         │ │       │         │ │       │         │
│       ▼         │ │       ▼         │ │       ▼         │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │    INDI     │ │ │ │    INDI     │ │ │ │    INDI     │ │
│ │ Controller  │ │ │ │ Controller  │ │ │ │ Controller  │ │
│ │  (300 Hz)   │ │ │ │  (300 Hz)   │ │ │ │  (300 Hz)   │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘ └─────────────────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│              Load-Cable State Estimator (EKF)                    │
│  - 입력: 쿼드로터 위치, 속도, IMU (가속도계)                       │
│  - 출력: 화물 pose, twist, 케이블 방향                            │
│  - 초기화: Iterative Kabsch-Umeyama Algorithm                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 수학적 모델링

### 2.1 좌표계 정의
- **F_I**: 관성 좌표계 (Inertial frame)
- **F_L**: 화물 고정 좌표계 (Load-fixed frame)
- **F_i**: i번째 쿼드로터 고정 좌표계 (Quadrotor-fixed frame)

### 2.2 Load-Cable Dynamic Model

#### 상태 벡터 (State Vector)
```
x = [p, v, q, ω, s₁, r₁, ṙ₁, r̈₁, t₁, ṫ₁, ..., sₙ, rₙ, ṙₙ, r̈ₙ, tₙ, ṫₙ]ᵀ
```

| 변수 | 타입 | 설명 |
|------|------|------|
| p | ℝ³ | 화물 위치 (inertial frame) |
| v | ℝ³ | 화물 속도 |
| q | S³ | 화물 자세 (unit quaternion) |
| ω | ℝ³ | 화물 각속도 (load frame) |
| sᵢ | S² | 케이블 방향 단위벡터 (쿼드로터→화물) |
| rᵢ | ℝ³ | 케이블 각속도 |
| tᵢ | ℝ≥0 | 케이블 장력 |

#### 화물 동역학 (Load Dynamics) - Equation 2

**병진 운동:**
```
ṗ = v

v̇ = -1/m · Σⁿᵢ₌₁ tᵢsᵢ + g
```

**회전 운동:**
```
q̇ = 1/2 · Λ(q) · [0; ω]

J·ω̇ = -ω × (J·ω) + Σⁿᵢ₌₁ tᵢ · (R(q)ᵀsᵢ × ρᵢ)
```

여기서:
- m: 화물 질량
- J ∈ ℝ³ˣ³: 화물 관성 텐서
- g = [0, 0, -9.81]ᵀ: 중력 벡터
- Λ(q): 쿼터니언 곱셈 연산자
- R(q) ∈ SO(3): 쿼터니언으로부터 회전 행렬
- ρᵢ ∈ ℝ³: i번째 케이블 부착점 (load frame 기준)

#### 케이블 동역학 (Cable Kinematics) - Equation 3

```
ṡᵢ = rᵢ × sᵢ        (케이블 방향 변화율)

r⃛ᵢ = γᵢ            (케이블 각속도의 3차 미분 = angular snap)

ẗᵢ = λᵢ            (케이블 장력의 2차 미분)
```

**OCP 입력**: u = [γ₁, λ₁, ..., γₙ, λₙ]

> **핵심 설계**: γᵢ, λᵢ를 OCP 입력으로 사용하여 쿼드로터 궤적이 jerk까지 **C³ smooth**하도록 보장 (Supplementary Methods의 Proposition 1)

### 2.3 Quadrotor Dynamic Model - Equation 4

#### 상태 벡터
```
xᵢ = [pᵢ, vᵢ, qᵢ, ωᵢ]ᵀ
```

| 변수 | 타입 | 설명 |
|------|------|------|
| pᵢ | ℝ³ | 쿼드로터 CoG 위치 |
| vᵢ | ℝ³ | 쿼드로터 속도 |
| qᵢ | S³ | 쿼드로터 자세 (unit quaternion) |
| ωᵢ | ℝ³ | 쿼드로터 각속도 (body frame) |

#### 동역학 방정식

**병진 운동:**
```
ṗᵢ = vᵢ

v̇ᵢ = 1/mᵢ · (Tᵢzᵢ + tᵢsᵢ + fₐ,ᵢ) + g
```

**회전 운동:**
```
q̇ᵢ = 1/2 · Λ(qᵢ) · [0; ωᵢ]

Jᵢ·ω̇ᵢ = -ωᵢ × (Jᵢ·ωᵢ) + τᵢ + τₐ,ᵢ
```

여기서:
- mᵢ: 쿼드로터 질량
- Jᵢ ∈ ℝ³ˣ³: 쿼드로터 관성 텐서
- Tᵢ ∈ ℝ≥0: 총 추력 (collective thrust)
- zᵢ ∈ S²: 추력 방향 (body z축)
- fₐ,ᵢ ∈ ℝ³: 공기역학적 항력
- τᵢ ∈ ℝ³: 제어 토크
- τₐ,ᵢ ∈ ℝ³: 공기역학적 토크

### 2.4 Kinematic Constraints - Equation 5

케이블이 팽팽할 때 (taut cable), 쿼드로터와 화물 위치 관계:

```
pᵢ = p + R(q)·ρᵢ - lᵢ·sᵢ
```

여기서:
- lᵢ: 케이블 길이
- ρᵢ: 화물 frame 기준 케이블 부착점

#### 쿼드로터 추력 계산 - Equation 9

kinematic constraint의 2차 미분으로부터:
```
Tᵢ(x) = ‖(v̇ᵢ(x) - g)·mᵢ - tᵢsᵢ - fₐ,ᵢ‖
```

---

## 3. Online Kinodynamic Motion Planner (OCP)

### 3.1 OCP 정식화 - Equation 6

```
minimize    J = Σᵏ⁼⁰ᴺ⁻¹ (‖xₖ - xₖ,ref‖²_Q + ‖uₖ - uₖ,ref‖²_R) + ‖xₙ - xₙ,ref‖²_P

subject to  x₀ = xᵢₙᵢₜ                    (초기 조건)
            xₖ₊₁ = f(xₖ, uₖ)              (시스템 동역학)
            h(xₖ₊₁, uₖ) ≤ 0               (불평등 제약조건)
            k ∈ {0, ..., N}
```

### 3.2 Cost Function

```
J = Σₖ [
    w_p · ‖p - p_ref‖²                   // 위치 추적
  + w_q · ‖q ⊖ q_ref‖²                   // 자세 추적
  + w_v · ‖v - v_ref‖²                   // 속도 추적
  + w_ω · ‖ω - ω_ref‖²                   // 각속도 추적
  + w_γ · Σᵢ ‖γᵢ‖²                       // 케이블 각 snap 최소화
  + w_λ · Σᵢ λᵢ²                         // 장력 변화율 최소화
]
```

### 3.3 Path Constraints

#### (1) 추력 제한 - Equation 8
```
0 ≤ Tᵢ,min ≤ Tᵢ(x) ≤ Tᵢ,max
```

#### (2) 케이블 장력 (tautness) - Equation 10
```
0 < tmin ≤ tᵢ ≤ tmax
```

#### (3) 쿼드로터 간 최소 거리 - Equation 11
```
0 < dmin ≤ ‖pᵢ(x) - pⱼ(x)‖,  ∀i ≠ j
```

#### (4) 장애물 회피 - Equation 12
```
d²o,min ≤ (pₖ(x) - pₒ)ᵀ · C · (pₖ(x) - pₒ)
```
여기서:
- pₖ(x): 제어점 (쿼드로터 CoG 또는 화물 모서리)
- pₒ: no-fly zone 중심
- C ∈ ℝ³ˣ³: 대각행렬 (no-fly zone 모양 결정)

### 3.4 구현 세부사항

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| Horizon (N) | 20 nodes | 비등간격 discretization |
| Horizon time | ~2 s | |
| Planner frequency | 10 Hz | |
| Solver | ACADOS | SQP + RTI scheme |
| Average CPU time | 15.3 ms | Intel Core i7-13700H |

#### 비등간격 Discretization
- 시간 간격이 horizon을 따라 선형적으로 증가
- 가까운 미래: 높은 정밀도
- 먼 미래: 낮은 정밀도 (horizon 연장)

### 3.5 Initial State 처리

OCP의 초기 상태 `xᵢₙᵢₜ` 구성:
1. **EKF로부터**: 화물 pose, twist, 케이블 방향
2. **이전 trajectory resampling**: 케이블 rate, 장력 및 고차 미분값

> **핵심**: 이전 trajectory를 resampling하여 연속적인 reference 전환 보장

---

## 4. INDI Trajectory-Tracking Controller

### 4.1 제어 구조

```
Position Loop  →  Velocity Loop  →  Attitude Loop  →  Rate Loop  →  INDI
  (p_ref)           (v_ref)          (z_des)          (ω_des)       (τ_des)
     │                 │                │                │             │
     └─────────────────┴────────────────┴────────────────┴─────────────┘
                              PD Controller (Eq. 15)
```

### 4.2 Thrust Command - Equation 15

```
Tᵢ,des·zᵢ,des / mᵢ = Kₚ(pᵢ,ref - pᵢ) + Kᵥ(vᵢ,ref - vᵢ) + v̇ᵢ,ref + fₑₓₜ/mᵢ
```

여기서:
- Kₚ, Kᵥ ∈ ℝ³ˣ³: positive definite 게인 행렬
- fₑₓₜ: 외부 힘 (케이블 장력 + 공기역학적 항력 + 바람)

#### 외부 힘 추정 (IMU 기반)
```
fₑₓₜ = mᵢ · aᵢ,filtered - fᵢ,filtered
```
- aᵢ,filtered: 필터링된 가속도계 측정값 (bias 제거)
- fᵢ,filtered: 현재 추력 벡터 (같은 필터 적용)

### 4.3 Tilt-Prioritized Attitude Controller

z_des로부터 angular acceleration command α_des 생성:
- Reference jerk 사용
- Zero yaw rate reference (heading 고정)

### 4.4 INDI Low-Level Controller (Supplementary Methods)

#### 로터 속도 → 추력/토크 매핑 - Equation S7
```
[T ]     [u₁²]
[τ ] = G₁[u₂²] + G₂·u̇
         [u₃²]
         [u₄²]
```

여기서:
- G₁: 추력/토크 effectiveness matrix (공기역학 계수 기반)
- G₂: 로터 관성에 의한 yaw 토크 matrix

#### INDI 제어 법칙 - Equation S8, S9

원하는 토크:
```
τ_des = τ_f + J · (α_des - ω̇_f)
```

로터 속도 명령 (수치적 풀이):
```
[T_des]       [u_c²]
[τ_des] = G₁ [    ] + Δt⁻¹ · G₂ · (u_c - u_c,k-1)
```

> **INDI의 핵심**: 센서 기반 적응 제어. 외부 토크 교란(공기역학, CoG 오차 등)을 자동 보상

---

## 5. Load-Cable State Estimator (EKF)

### 5.1 EKF 상태 벡터
```
x̂ = [p, v, q, ω, p₁, v₁, ..., pₙ, vₙ]ᵀ
```

### 5.2 Process Model
- 화물 동역학 (Eq. 2)
- 쿼드로터 동역학 (Eq. 4)
- 케이블 방향: kinematic constraint (Eq. 5)로부터 계산
- 케이블 장력: spring-damper 모델
  ```
  tᵢ = k_stiff · dᵢ + k_damp · ḋᵢ
  ```
  여기서 dᵢ = ‖pᵢ - R(q)ρᵢ - p‖

### 5.3 Measurement Model

측정 벡터:
```
ỹ = [s̃₁, p̃₁, ṽ₁, ..., s̃ₙ, p̃ₙ, ṽₙ]ᵀ
```

#### 케이블 방향 측정 (가속도계로부터) - Equation 14
```
s̃ᵢ = (mᵢaᵢ - T̄ᵢzᵢ - f̄ₐ,ᵢ) / ‖mᵢaᵢ - T̄ᵢzᵢ - f̄ₐ,ᵢ‖
```

여기서:
- aᵢ = v̇ - g: bias 제거된 가속도계 측정값
- T̄ᵢ: 추력 모델 (Eq. 13): T̄ᵢ = Σⱼ cₜω²ⱼ,ᵢ
- f̄ₐ,ᵢ: 항력 모델 (Eq. 13): f̄ₐ,ᵢ = R(qᵢ)DₐR(qᵢ)ᵀvᵢ

### 5.4 초기화: Iterative Kabsch-Umeyama Algorithm

쿼드로터 위치들로부터 화물 pose 추정:
1. 케이블 방향 초기 추정: sᵢ = [0, 0, -1]ᵀ
2. 케이블 연결점 계산: cᵢ = pᵢ + sᵢlᵢ
3. SVD를 통한 rotation 추정
4. 화물 위치 추정
5. 케이블 방향 업데이트
6. 수렴할 때까지 반복

---

## 6. 실험 결과 요약

### 6.1 실험 설정
- 쿼드로터: 3대 (0.6 kg each, Agilicious 기반)
- 화물: 1.4 kg 바구니
- 케이블 길이: 1 m
- 최대 추력: 20 N

### 6.2 성능 비교 (Table 1)

| Trajectory | v_max | a_max | Geometric | NMPC | **Ours** |
|------------|-------|-------|-----------|------|----------|
| Slow | 1 m/s | 0.5 m/s² | 0.032 m | 0.036 m | **0.102 m** |
| Medium | 2 m/s | 2 m/s² | 0.135 m | 0.159 m | **0.093 m** |
| Medium Plus | 2 m/s | 4 m/s² | Crash | Crash | **0.117 m** |
| Fast | 5 m/s | 8 m/s² | Crash | Crash | **0.197 m** |

### 6.3 Robustness

| 조건 | 결과 |
|------|------|
| ±50% 질량 오차 | 안정적 추적 |
| ±50% 관성 오차 | 안정적 추적 |
| 10% CoG 오차 | ~5° 자세 오차 증가 |
| 5 m/s 바람 | 안정적 동작 |
| 43% 질량 오차 (basketball) | 0.225 m RMSE |

### 6.4 Scalability

| 쿼드로터 수 | CPU time |
|-------------|----------|
| 3 | ~15 ms |
| 4 | ~25 ms |
| 9 | ~100 ms (10Hz 한계) |

---

## 7. Rust 구현을 위한 핵심 모듈

### 7.1 모듈 구조

```
caslo/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   │
│   ├── math/                          # 수학 유틸리티
│   │   ├── mod.rs
│   │   ├── linalg.rs                  # 선형대수 (nalgebra wrapper)
│   │   ├── quaternion.rs              # 쿼터니언 연산
│   │   │   - Λ(q) 곱셈 연산자
│   │   │   - q → R(q) 변환
│   │   │   - slerp, exp, log
│   │   ├── so3.rs                     # SO(3) 회전 연산
│   │   │   - exp/log map
│   │   │   - axis-angle 변환
│   │   └── s2.rs                      # S² 단위구 연산
│   │       - 단위벡터 정규화
│   │       - 구면 보간
│   │
│   ├── dynamics/                       # 동역학 모델
│   │   ├── mod.rs
│   │   ├── load.rs                    # 화물 동역학 (Eq. 2)
│   │   │   - LoadState: {p, v, q, ω}
│   │   │   - LoadParams: {m, J}
│   │   │   - load_dynamics(state, forces, torques)
│   │   │
│   │   ├── cable.rs                   # 케이블 동역학 (Eq. 3)
│   │   │   - CableState: {s, r, ṙ, r̈, t, ṫ}
│   │   │   - cable_kinematics(state, γ, λ)
│   │   │
│   │   ├── quadrotor.rs               # 쿼드로터 동역학 (Eq. 4)
│   │   │   - QuadState: {p, v, q, ω}
│   │   │   - QuadParams: {m, J, arm_length, k_thrust, k_torque}
│   │   │   - quadrotor_dynamics(state, T, τ)
│   │   │   - rotor_to_thrust_torque(ω_motors) → (T, τ)
│   │   │
│   │   ├── system.rs                  # 전체 시스템 (Eq. 1)
│   │   │   - SystemState: {load, cables[], quads[]}
│   │   │   - kinematic_constraint(load, cable, quad) (Eq. 5)
│   │   │   - system_dynamics(state, u)
│   │   │
│   │   └── aerodynamics.rs            # 공기역학 모델
│   │       - drag_force(v, R, D_a) (Eq. 13)
│   │       - drag_torque(ω)
│   │
│   ├── planning/                       # MPC 플래너
│   │   ├── mod.rs
│   │   ├── ocp.rs                     # OCP 정의 (Eq. 6)
│   │   │   - OCPConfig: {N, horizon_time, weights}
│   │   │   - cost_function(x, x_ref, u, u_ref)
│   │   │   - path_constraints(x)
│   │   │
│   │   ├── constraints.rs             # 제약조건 (Eq. 8-12)
│   │   │   - thrust_constraint(x)
│   │   │   - tautness_constraint(x)
│   │   │   - collision_constraint(x)
│   │   │   - obstacle_constraint(x, obstacles)
│   │   │
│   │   ├── solver.rs                  # NLP 솔버 인터페이스
│   │   │   - SQP solver
│   │   │   - OSQP for QP subproblems
│   │   │
│   │   ├── trajectory.rs              # 궤적 표현
│   │   │   - Trajectory: 시간별 state 시퀀스
│   │   │   - interpolate(t) → state
│   │   │   - sample_reference(t) → (p, v, a, j)
│   │   │
│   │   └── reference.rs               # Reference 생성
│   │       - polynomial_trajectory(waypoints)
│   │       - figure_eight(params)
│   │
│   ├── control/                        # 온보드 제어기
│   │   ├── mod.rs
│   │   ├── position.rs                # Position controller
│   │   │   - compute_thrust_direction(p, v, p_ref, v_ref, a_ref)
│   │   │
│   │   ├── attitude.rs                # Tilt-prioritized attitude controller
│   │   │   - compute_angular_accel(z_des, q, ω, j_ref)
│   │   │
│   │   ├── indi.rs                    # INDI low-level (Eq. S7-S10)
│   │   │   - INDIController
│   │   │   - compute_rotor_speeds(T_des, τ_des)
│   │   │   - update_measurements(ω_gyro, ω_motors)
│   │   │
│   │   └── trajectory_tracker.rs      # 통합 추적 제어기
│   │       - TrajectoryTracker
│   │       - track(trajectory, current_state, imu) → motor_cmds
│   │
│   ├── estimation/                     # 상태 추정
│   │   ├── mod.rs
│   │   ├── ekf.rs                     # Extended Kalman Filter
│   │   │   - EKF<State, Measurement>
│   │   │   - predict(dt)
│   │   │   - update(measurement)
│   │   │
│   │   ├── load_estimator.rs          # 화물-케이블 추정기
│   │   │   - LoadCableEstimator
│   │   │   - estimate_cable_direction_from_imu(a, T, z, f_drag)
│   │   │   - fuse(quad_states[], imu_data[])
│   │   │
│   │   └── initializer.rs             # Kabsch-Umeyama 초기화
│   │       - kabsch_umeyama(quad_positions, attachment_points)
│   │
│   ├── filters/                        # 신호 처리
│   │   ├── mod.rs
│   │   ├── lowpass.rs                 # Low-pass filter
│   │   └── differentiator.rs          # 수치 미분기
│   │
│   └── simulation/                     # 시뮬레이션
│       ├── mod.rs
│       ├── world.rs                   # 시뮬레이션 환경
│       ├── integrator.rs              # RK4 적분기
│       └── sensors.rs                 # 센서 모델 (IMU 노이즈 등)
```

### 7.2 핵심 의존성

```toml
[dependencies]
nalgebra = "0.32"              # 선형대수
osqp = "0.6"                   # QP 솔버
num-traits = "0.2"             # 수치 traits
serde = { version = "1.0", features = ["derive"] }  # 직렬화

[dev-dependencies]
approx = "0.5"                 # 테스트용 근사 비교
plotters = "0.3"               # 시각화
```

### 7.3 구현 우선순위

1. **Phase 1: 수학 기반**
   - quaternion, SO(3), S² 연산
   - 선형대수 유틸리티

2. **Phase 2: 동역학 모델**
   - Load dynamics
   - Cable kinematics
   - Quadrotor dynamics
   - Kinematic constraints

3. **Phase 3: 제어기**
   - INDI low-level controller
   - Trajectory tracking controller

4. **Phase 4: 추정기**
   - EKF 구현
   - Load-cable state estimator

5. **Phase 5: 플래너**
   - OCP 정식화
   - SQP 솔버 인터페이스
   - Constraint handling

6. **Phase 6: 시뮬레이션**
   - 통합 시뮬레이션
   - 성능 검증

---

## 8. 핵심 수식 요약 (Quick Reference)

### Load Dynamics (Eq. 2)
```
v̇ = -1/m · Σᵢ tᵢsᵢ + g
J·ω̇ = -ω × Jω + Σᵢ tᵢ(Rᵀsᵢ × ρᵢ)
```

### Kinematic Constraint (Eq. 5)
```
pᵢ = p + R(q)ρᵢ - lᵢsᵢ
```

### Thrust from OCP (Eq. 9)
```
Tᵢ = ‖(v̇ᵢ - g)mᵢ - tᵢsᵢ - fₐ,ᵢ‖
```

### INDI Control (Eq. S9)
```
τ_des = τ_f + J(α_des - ω̇_f)
```

### Cable Direction from IMU (Eq. 14)
```
s̃ᵢ = (mᵢaᵢ - T̄ᵢzᵢ - f̄ₐ,ᵢ) / ‖...‖
```

### Position Controller (Eq. 15)
```
T_des·z_des/m = Kₚ(p_ref - p) + Kᵥ(v_ref - v) + a_ref + fₑₓₜ/m
```

---

## 9. References

1. Sun et al., "Agile and Cooperative Aerial Manipulation of a Cable-Suspended Load," Science Robotics, 2025
2. Tal & Karaman, "Accurate Tracking of Aggressive Quadrotor Trajectories using INDI and Differential Flatness"
3. Sun et al., "A Comparative Study of Nonlinear MPC and Differential-Flatness-Based Control for Quadrotor Agile Flight"
4. Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors"
5. ACADOS: https://github.com/acados/acados
