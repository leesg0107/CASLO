#!/usr/bin/env python3
"""
CASLO OCP Code Generation using ACADOS

Generates optimized C code for the cable-suspended load motion planner
based on the formulation from:
"Agile and Cooperative Aerial Manipulation of a Cable-Suspended Load"
(Sun et al., Science Robotics, 2025)

Usage:
    python caslo_ocp.py --num-quads 3 --output-dir ../generated

Dependencies:
    - acados (https://github.com/acados/acados)
    - casadi
    - numpy
"""

import argparse
import os
import numpy as np
from typing import Optional

# ACADOS imports
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca


def create_caslo_model(num_quads: int = 3) -> AcadosModel:
    """
    Create the ACADOS model for the cable-suspended load system.

    State vector (13 + 11*n) - Eq. 1 from paper:
        x = [p, v, q, ω, s₁, r₁, ṙ₁, t₁, ṫ₁, ..., sₙ, rₙ, ṙₙ, tₙ, ṫₙ]

        - p_L: load position (3)
        - v_L: load velocity (3)
        - q_L: load orientation quaternion (4)
        - omega_L: load angular velocity (3)
        - For each cable i:
            - s_i: cable direction unit vector (3)
            - r_i: cable angular velocity (3)
            - r_dot_i: cable angular acceleration ṙᵢ (3)
            - t_i: cable tension (1)
            - t_dot_i: cable tension rate ṫᵢ (1)

    Control vector (4*n) - Eq. 3:
        u = [γ₁, λ₁, ..., γₙ, λₙ]
        - γᵢ = r̈ᵢ: cable angular jerk (control input) (3)
        - λᵢ = ẗᵢ: tension acceleration (1)
    """
    model = AcadosModel()
    model.name = f"caslo_{num_quads}quad"

    # System parameters (will be set at runtime)
    # Load parameters
    m_L = ca.SX.sym('m_L')           # Load mass
    J_L = ca.SX.sym('J_L', 3, 3)     # Load inertia tensor
    g = 9.81                          # Gravity

    # Per-cable parameters
    l = ca.SX.sym('l', num_quads)     # Cable lengths
    rho = ca.SX.sym('rho', 3, num_quads)  # Attachment points in body frame

    # Collect parameters
    p = ca.vertcat(
        m_L,
        ca.reshape(J_L, -1, 1),
        l,
        ca.reshape(rho, -1, 1)
    )

    # === State Variables (Eq. 1) ===
    # Load states
    p_L = ca.SX.sym('p_L', 3)         # Position
    v_L = ca.SX.sym('v_L', 3)         # Velocity
    q_L = ca.SX.sym('q_L', 4)         # Quaternion (w, x, y, z)
    omega_L = ca.SX.sym('omega_L', 3) # Angular velocity (body frame)

    # Cable states (per quadrotor) - Eq. 1 from paper
    # Each cable: sᵢ(3) + rᵢ(3) + ṙᵢ(3) + r̈ᵢ(3) + tᵢ(1) + ṫᵢ(1) = 14 DOF
    # Note: The paper uses 3rd-order cable model for smooth quadrotor trajectories
    s = ca.SX.sym('s', 3, num_quads)            # Cable directions sᵢ
    r = ca.SX.sym('r', 3, num_quads)            # Cable angular velocities rᵢ
    r_dot = ca.SX.sym('r_dot', 3, num_quads)    # Cable angular accelerations ṙᵢ
    r_ddot = ca.SX.sym('r_ddot', 3, num_quads)  # Cable angular jerks r̈ᵢ (NEW!)
    t = ca.SX.sym('t', num_quads)               # Cable tensions tᵢ
    t_dot = ca.SX.sym('t_dot', num_quads)       # Cable tension rates ṫᵢ

    # Assemble state vector (Eq. 1 layout from paper)
    # x = [p, v, q, ω, s_all, r_all, ṙ_all, r̈_all, t_all, ṫ_all]
    # This 3rd-order cable model ensures smooth quadrotor jerk references
    x = ca.vertcat(
        p_L, v_L, q_L, omega_L,
        ca.reshape(s, -1, 1),        # All sᵢ (3*n)
        ca.reshape(r, -1, 1),        # All rᵢ (3*n)
        ca.reshape(r_dot, -1, 1),    # All ṙᵢ (3*n)
        ca.reshape(r_ddot, -1, 1),   # All r̈ᵢ (3*n) - NEW!
        t,                            # All tᵢ (n)
        t_dot                         # All ṫᵢ (n)
    )

    # === Control Variables (Eq. 3 from paper) ===
    # γᵢ = r⃛ᵢ: angular SNAP (control input - 3rd derivative of cable angular velocity)
    # λᵢ = ẗᵢ: tension acceleration (2nd derivative of tension)
    # Using snap as control ensures smooth jerk-level quadrotor references
    gamma = ca.SX.sym('gamma', 3, num_quads)  # Angular snap γᵢ = r⃛ᵢ (control input)
    lam = ca.SX.sym('lambda', num_quads)      # Tension acceleration λᵢ = ẗᵢ

    u = ca.vertcat(
        ca.reshape(gamma, -1, 1),
        lam
    )

    # === Dynamics (Eq. 2-3 from paper) ===

    # Quaternion to rotation matrix
    def quat_to_rot(q):
        """Convert quaternion (w,x,y,z) to rotation matrix."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        return ca.vertcat(
            ca.horzcat(1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)),
            ca.horzcat(2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)),
            ca.horzcat(2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2))
        )

    # Quaternion derivative from angular velocity
    def quat_derivative(q, omega):
        """q_dot = 0.5 * Lambda(q) @ [0; omega]"""
        w, x, y, z = q[0], q[1], q[2], q[3]
        omega_quat = ca.vertcat(0, omega)

        # Lambda matrix for quaternion multiplication
        Lambda = ca.vertcat(
            ca.horzcat(w, -x, -y, -z),
            ca.horzcat(x,  w, -z,  y),
            ca.horzcat(y,  z,  w, -x),
            ca.horzcat(z, -y,  x,  w)
        )
        return 0.5 * Lambda @ omega_quat

    # Skew-symmetric matrix
    def skew(v):
        return ca.vertcat(
            ca.horzcat(0, -v[2], v[1]),
            ca.horzcat(v[2], 0, -v[0]),
            ca.horzcat(-v[1], v[0], 0)
        )

    # Cross product
    def cross(a, b):
        return ca.vertcat(
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        )

    # Rotation matrix (load body to world)
    R_L = quat_to_rot(q_L)

    # === Load Dynamics (Eq. 2) ===
    # v_dot = -1/m * sum(t_i * s_i) + g*e_z
    # omega_dot = J^{-1} * (-omega x J*omega + sum(t_i * (R^T s_i) x rho_i))

    # Gravity vector in Z-UP coordinate frame (positive Z is up)
    # This matches the Rust simulation coordinate convention
    g_vec = ca.vertcat(0, 0, -g)

    # Sum cable forces
    cable_force_sum = ca.SX.zeros(3)
    for i in range(num_quads):
        cable_force_sum += t[i] * s[:, i]

    v_L_dot = -cable_force_sum / m_L + g_vec

    # Sum cable torques (in body frame)
    cable_torque_sum = ca.SX.zeros(3)
    for i in range(num_quads):
        s_body = R_L.T @ s[:, i]  # Cable direction in body frame
        cable_torque_sum += t[i] * cross(s_body, rho[:, i])

    # Angular dynamics
    J_omega = J_L @ omega_L
    omega_cross_J_omega = cross(omega_L, J_omega)

    # J * omega_dot = -omega x J*omega + tau
    # Solve for omega_dot (assuming J is diagonal for simplicity in codegen)
    # In actual implementation, use proper matrix inverse
    J_inv = ca.inv(J_L)
    omega_L_dot = J_inv @ (-omega_cross_J_omega + cable_torque_sum)

    # Quaternion derivative
    q_L_dot = quat_derivative(q_L, omega_L)

    # === Cable Kinematics (Eq. 3 from paper) ===
    # 3rd-order cable model for smooth quadrotor jerk references:
    # ṡᵢ = rᵢ × sᵢ           (cable direction derivative)
    # ṙᵢ = r_dot_i           (from state - angular acceleration)
    # r̈̇ᵢ = r_ddot_i          (from state - angular jerk)
    # r⃛ᵢ = γᵢ                (control input - angular SNAP)
    # ṫᵢ = t_dot_i           (from state - tension rate)
    # ẗᵢ = λᵢ                (control input - tension acceleration)

    s_dot_list = []
    for i in range(num_quads):
        s_i_dot = cross(r[:, i], s[:, i])
        s_dot_list.append(s_i_dot)

    # Assemble state derivative (Eq. 1 layout from paper)
    # State: [p, v, q, ω, s(all), r(all), ṙ(all), r̈(all), t, ṫ]
    # Derivative: [v, v̇, q̇, ω̇, ṡ(all), ṙ(all), r̈(all), r⃛(all)=γ, ṫ, ẗ=λ]
    x_dot = ca.vertcat(
        v_L,                          # ṗ = v
        v_L_dot,                      # v̇ from Eq. 2
        q_L_dot,                      # q̇
        omega_L_dot,                  # ω̇ from Eq. 2
        ca.vertcat(*s_dot_list),      # ṡᵢ = rᵢ × sᵢ
        ca.reshape(r_dot, -1, 1),     # ṙᵢ (from state - angular acceleration)
        ca.reshape(r_ddot, -1, 1),    # r̈ᵢ (from state - angular jerk)
        ca.reshape(gamma, -1, 1),     # r⃛ᵢ = γᵢ (control - angular SNAP)
        t_dot,                        # ṫᵢ (from state)
        lam                           # ẗᵢ = λᵢ (control)
    )

    # === Explicit ODE ===
    model.f_expl_expr = x_dot
    model.f_impl_expr = x_dot - ca.SX.sym('x_dot', x.size1())
    model.x = x
    model.xdot = ca.SX.sym('x_dot', x.size1())
    model.u = u
    model.p = p

    return model


def create_caslo_ocp(
    num_quads: int = 3,
    horizon_time: float = 2.0,
    num_nodes: int = 20,
    non_uniform: bool = True
) -> AcadosOcp:
    """
    Create the ACADOS OCP for the cable-suspended load system.

    Args:
        num_quads: Number of quadrotors
        horizon_time: Prediction horizon [s]
        num_nodes: Number of shooting nodes
        non_uniform: Use non-uniform time discretization

    Returns:
        Configured AcadosOcp object
    """
    ocp = AcadosOcp()

    # Model
    model = create_caslo_model(num_quads)
    ocp.model = model

    # Dimensions
    nx = model.x.size1()
    nu = model.u.size1()
    np_param = model.p.size1()

    ocp.dims.N = num_nodes

    # Time discretization
    if non_uniform:
        # Linear increase in dt (higher resolution near beginning)
        n = num_nodes
        ratio = 2.0
        dt_0 = horizon_time / (n + (ratio - 1.0) * n * (n - 1.0) / (2.0 * (n - 1.0)))
        delta = (ratio - 1.0) * dt_0 / (n - 1.0)
        time_steps = np.array([dt_0 + k * delta for k in range(num_nodes)])
    else:
        time_steps = np.ones(num_nodes) * horizon_time / num_nodes

    ocp.solver_options.time_steps = time_steps
    ocp.solver_options.tf = horizon_time

    # === Cost Function (Eq. 6) ===
    # J = sum_k (||x_k - x_ref||^2_Q + ||u_k - u_ref||^2_R) + ||x_N - x_N_ref||^2_P

    # Cost type: NONLINEAR_LS (for flexibility)
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # Residual dimensions
    # Track: position (3), velocity (3), orientation (3 for axis-angle error), angular vel (3)
    # + control smoothness terms
    ny = 12 + 4 * num_quads  # tracking + controls
    ny_e = 12  # terminal: only state tracking

    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e

    # Reference (will be set at runtime)
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # === Cost Residual Definition ===
    # Extract state components
    p_L = model.x[0:3]
    v_L = model.x[3:6]
    q_L = model.x[6:10]
    omega_L = model.x[10:13]

    # Reference symbols (will be set via parameters or yref)
    p_ref = ca.SX.sym('p_ref', 3)
    v_ref = ca.SX.sym('v_ref', 3)
    q_ref = ca.SX.sym('q_ref', 4)
    omega_ref = ca.SX.sym('omega_ref', 3)

    # Quaternion error (axis-angle representation)
    def quat_error(q, q_ref):
        """Compute axis-angle error between quaternions."""
        # q_error = q_ref^{-1} * q
        # For small errors, this gives approximately the axis-angle error
        q_ref_conj = ca.vertcat(q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])

        # Quaternion multiplication: q_ref_conj * q
        w1, x1, y1, z1 = q_ref_conj[0], q_ref_conj[1], q_ref_conj[2], q_ref_conj[3]
        w2, x2, y2, z2 = q[0], q[1], q[2], q[3]

        q_err = ca.vertcat(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )

        # For small angles: error ≈ 2 * [x, y, z]
        return 2.0 * q_err[1:4]

    # Cost residual: tracking errors + control magnitude
    # NOTE: For proper quaternion error, we would compute q_err = q_ref^(-1) * q
    # But this requires the reference quaternion in the cost expression.
    # As a simplification for tracking with slow yaw rotation (0.25 rad/s),
    # we use the quaternion vector part which approximates small angle errors.
    # For aggressive orientation tracking, this should be replaced with
    # proper quaternion error computation.
    #
    # Alternative: Set orientation weight to 0 for position-only tracking
    y = ca.vertcat(
        p_L,     # Position (3)
        v_L,     # Velocity (3)
        ca.vertcat(q_L[1], q_L[2], q_L[3]),  # Quaternion vector part (3)
        omega_L, # Angular velocity (3)
        model.u  # Controls for regularization (4*n)
    )

    y_e = ca.vertcat(
        p_L,     # Position (3)
        v_L,     # Velocity (3)
        ca.vertcat(q_L[1], q_L[2], q_L[3]),  # Quaternion vector part (3)
        omega_L  # Angular velocity (3)
    )

    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = y_e

    # Weight matrices (tuned for aggressive Figure-8 tracking)
    # IMPORTANT: Condition number κ(W) = max(W)/min(W) should be < 10,000
    # for numerical stability. Previous weights had κ ≈ 1,000,000 causing MINSTEP.

    # Position tracking - most important
    W_p = np.diag([50.0, 50.0, 100.0])  # Max weight = 100
    # Velocity tracking
    W_v = np.diag([5.0, 5.0, 10.0])     # Good damping
    # Orientation tracking - disabled for now (quaternion cost formulation issue)
    # The simplified quaternion vector part comparison doesn't work well
    # for non-identity reference orientations
    W_q = np.diag([0.0, 0.0, 0.0])      # Disabled - focus on position tracking
    # Angular velocity tracking
    W_omega = np.diag([0.0, 0.0, 0.0])  # Disabled

    # Control weights (regularization to prevent aggressive control)
    # CRITICAL: Increased 10-100x to reduce condition number
    # Old: κ = 100/0.0001 = 1,000,000 (causes MINSTEP)
    # New: κ = 100/0.01 = 10,000 (acceptable for double precision)
    W_gamma = 0.01 * np.eye(3 * num_quads)    # Increased 10x from 0.001
    W_lambda = 0.01 * np.eye(num_quads)       # Increased 100x from 0.0001

    W = np.zeros((ny, ny))
    W[0:3, 0:3] = W_p
    W[3:6, 3:6] = W_v
    W[6:9, 6:9] = W_q
    W[9:12, 9:12] = W_omega
    W[12:12+3*num_quads, 12:12+3*num_quads] = W_gamma
    W[12+3*num_quads:, 12+3*num_quads:] = W_lambda

    ocp.cost.W = W

    # Terminal cost (higher weight for position/velocity only)
    W_e = np.zeros((ny_e, ny_e))
    W_e[0:3, 0:3] = 5.0 * W_p   # Terminal position weight
    W_e[3:6, 3:6] = 5.0 * W_v   # Terminal velocity weight
    W_e[6:9, 6:9] = 0.0 * W_q   # Disabled (same as stage cost)
    W_e[9:12, 9:12] = 0.0 * W_omega  # Disabled
    ocp.cost.W_e = W_e

    # === Constraints ===

    # Control bounds (Eq. 9 - implied from smoothness)
    # Control is now SNAP (4th derivative of cable angle), not jerk
    # IMPORTANT: Control bounds must be consistent with state bounds
    # If max_angular_jerk_state = 500 rad/s³ and dt = 0.1s,
    # then max_angular_snap should satisfy: snap * dt < jerk_bound
    # To prevent state-dependent infeasibility: snap_max ≤ jerk_max / dt
    # But we need margin for the QP solver, so use 50% of theoretical max
    max_angular_snap = 200.0      # rad/s⁴ - reduced from 1000 to prevent infeasibility
    max_tension_accel = 500.0     # N/s² - reduced from 1000 for consistency

    lbu = np.concatenate([
        -max_angular_snap * np.ones(3 * num_quads),
        -max_tension_accel * np.ones(num_quads)
    ])
    ubu = np.concatenate([
        max_angular_snap * np.ones(3 * num_quads),
        max_tension_accel * np.ones(num_quads)
    ])

    ocp.constraints.lbu = lbu
    ocp.constraints.ubu = ubu
    ocp.constraints.idxbu = np.arange(nu)

    # State bounds - tension must be positive (Eq. 10)
    tension_min = 0.5
    tension_max = 50.0

    # State layout (Eq. 1 from paper with 3rd-order cable model):
    # 13 (load) + 3*n (s) + 3*n (r) + 3*n (ṙ) + 3*n (r̈) + n (t) + n (ṫ) = 13 + 14*n
    # s:    13 : 13 + 3*n
    # r:    13 + 3*n : 13 + 6*n
    # ṙ:    13 + 6*n : 13 + 9*n
    # r̈:   13 + 9*n : 13 + 12*n
    # t:    13 + 12*n : 13 + 13*n
    # ṫ:    13 + 13*n : 13 + 14*n
    s_start = 13
    r_start = 13 + 3 * num_quads
    r_dot_start = 13 + 6 * num_quads
    r_ddot_start = 13 + 9 * num_quads
    tension_start = 13 + 12 * num_quads
    tension_rate_start = 13 + 13 * num_quads

    # Build state constraint vectors
    # Constrained states: r, ṙ, r̈, t, ṫ
    # Relaxed for aggressive maneuvers (Figure-8 at 5 m/s, 8 m/s²)
    max_angular_velocity = 20.0       # rad/s - allow more cable swing
    max_angular_acceleration = 100.0  # rad/s² - allow aggressive dynamics
    max_angular_jerk_state = 500.0    # rad/s³ - angular jerk bound in state
    max_tension_rate = 200.0          # N/s - allow faster tension changes

    # Indices of constrained states
    # r: 3*n values, ṙ: 3*n values, r̈: 3*n values, t: n values, ṫ: n values = 11*n total
    num_constrained = 3 * num_quads + 3 * num_quads + 3 * num_quads + num_quads + num_quads  # r + ṙ + r̈ + t + ṫ

    lbx = np.zeros(num_constrained)
    ubx = np.zeros(num_constrained)
    idxbx = np.zeros(num_constrained, dtype=int)

    idx = 0
    # Angular velocity r bounds
    for i in range(3 * num_quads):
        idxbx[idx] = r_start + i
        lbx[idx] = -max_angular_velocity
        ubx[idx] = max_angular_velocity
        idx += 1

    # Angular acceleration ṙ bounds
    for i in range(3 * num_quads):
        idxbx[idx] = r_dot_start + i
        lbx[idx] = -max_angular_acceleration
        ubx[idx] = max_angular_acceleration
        idx += 1

    # Angular jerk r̈ bounds
    for i in range(3 * num_quads):
        idxbx[idx] = r_ddot_start + i
        lbx[idx] = -max_angular_jerk_state
        ubx[idx] = max_angular_jerk_state
        idx += 1

    # Tension t bounds (Eq. 10 from paper)
    for i in range(num_quads):
        idxbx[idx] = tension_start + i
        lbx[idx] = tension_min
        ubx[idx] = tension_max
        idx += 1

    # Tension rate ṫ bounds
    for i in range(num_quads):
        idxbx[idx] = tension_rate_start + i
        lbx[idx] = -max_tension_rate
        ubx[idx] = max_tension_rate
        idx += 1

    ocp.constraints.lbx = lbx
    ocp.constraints.ubx = ubx
    ocp.constraints.idxbx = idxbx

    # === Nonlinear Path Constraints (Eq. 8, 11, 12 from paper) ===
    # These require computing quadrotor positions from the kinematic constraint (Eq. 5):
    # p_i = p + R(q) * rho_i - l_i * s_i

    # We need to add:
    # 1. Thrust constraints (Eq. 8): T_i,min ≤ T_i(x) ≤ T_i,max
    # 2. Inter-quadrotor collision avoidance (Eq. 11): d_min ≤ ||p_i - p_j||
    # 3. Obstacle avoidance (Eq. 12): no-fly zones

    # For inter-quadrotor collision avoidance (Eq. 11)
    # Number of constraint pairs: n*(n-1)/2
    num_collision_pairs = num_quads * (num_quads - 1) // 2

    # Minimum distance between quadrotors
    # Note: With attachment radius 0.3m and cable length 1.0m, hover separation is ~0.52m
    # During aggressive maneuvers (Figure-8), quads can get closer.
    # Using 0.2m is very conservative (about half the hover separation)
    # Original paper value (0.8m) causes infeasibility
    d_min = 0.2  # Reduced for aggressive maneuvers

    # Build collision avoidance constraints
    # h(x) = d_min^2 - ||p_i - p_j||^2 ≤ 0 (reformulated for solver)
    if num_collision_pairs > 0:
        # Extract state components for constraint computation
        p_L_constr = model.x[0:3]
        q_L_constr = model.x[6:10]

        # Quaternion to rotation matrix
        def quat_to_rot_constr(q):
            w, x, y, z = q[0], q[1], q[2], q[3]
            return ca.vertcat(
                ca.horzcat(1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)),
                ca.horzcat(2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)),
                ca.horzcat(2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2))
            )

        R_L_constr = quat_to_rot_constr(q_L_constr)

        # Compute quadrotor positions using kinematic constraint (Eq. 5)
        # p_i = p + R(q) * rho_i - l_i * s_i
        # Note: We use symbolic parameters for rho and l
        quad_positions = []
        for i in range(num_quads):
            s_i = model.x[s_start + 3*i : s_start + 3*i + 3]
            # rho_i from parameters (index 10 + i*3 : 10 + i*3 + 3)
            rho_idx = 10 + num_quads + i * 3  # After m_L(1) + J_L(9) + l(n)
            rho_i = model.p[rho_idx : rho_idx + 3]
            # l_i from parameters
            l_i = model.p[10 + i]  # After m_L(1) + J_L(9)

            p_i = p_L_constr + R_L_constr @ rho_i - l_i * s_i
            quad_positions.append(p_i)

        # Collision avoidance constraints
        collision_constraints = []
        pair_idx = 0
        for i in range(num_quads):
            for j in range(i + 1, num_quads):
                # ||p_i - p_j||^2 - d_min^2 ≥ 0
                diff = quad_positions[i] - quad_positions[j]
                dist_sq = ca.dot(diff, diff)
                # Constraint: d_min^2 - dist_sq ≤ 0 (i.e., dist_sq ≥ d_min^2)
                collision_constraints.append(d_min**2 - dist_sq)
                pair_idx += 1

        if collision_constraints:
            h_collision = ca.vertcat(*collision_constraints)

            # Set up nonlinear constraints
            ocp.model.con_h_expr = h_collision
            ocp.constraints.lh = -1e9 * np.ones(num_collision_pairs)  # No lower bound
            ocp.constraints.uh = np.zeros(num_collision_pairs)  # h ≤ 0

    # === Initial State Constraint ===
    ocp.constraints.x0 = np.zeros(nx)
    # Set reasonable initial guess
    ocp.constraints.x0[6] = 1.0  # Quaternion w = 1 (identity)
    # Cable directions (s) start at index 13
    # Z-UP coordinate frame: cable pointing down = -z
    for i in range(num_quads):
        ocp.constraints.x0[s_start + 3*i + 2] = -1.0  # Cable pointing down (Z-UP: -z is down)
    # r, ṙ, r̈ are all initialized to zero (default - equilibrium)
    # Tensions (t) start at index 13 + 12*n (updated for 3rd-order model)
    for i in range(num_quads):
        ocp.constraints.x0[tension_start + i] = 5.0  # Initial tension
    # ṫ initialized to zero (default)

    # === Solver Options ===
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'  # Explicit Runge-Kutta
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1

    # Use limited-iteration SQP for balance between speed and robustness
    # Full SQP (100 iter) is too slow (~13ms), RTI (1 iter) is unstable with bad warm-start
    # Using 5 iterations provides ~3-5ms solve time with good convergence
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 5  # Limited iterations for speed
    ocp.solver_options.qp_solver_iter_max = 30  # Reduced QP iterations
    ocp.solver_options.tol = 1e-3  # Relaxed tolerance for faster convergence

    # Add regularization for numerical stability
    # CRITICAL: Higher values needed to prevent MINSTEP/NaN errors
    ocp.solver_options.regularize_method = 'PROJECT'
    ocp.solver_options.reg_epsilon = 1e-3  # Further increased for stability

    # Levenberg-Marquardt regularization (helps with ill-conditioned QPs)
    # Higher value prevents MINSTEP/NaN during aggressive maneuvers
    ocp.solver_options.levenberg_marquardt = 1e-2  # Increased 10x for robustness

    # Globalization: line search helps with convergence
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.alpha_min = 0.01  # Minimum step size
    ocp.solver_options.alpha_reduction = 0.5  # Step reduction factor

    # Code generation options
    ocp.code_export_directory = 'caslo_ocp_generated'

    # === Default Parameter Values ===
    # Parameters: m_L (1) + J_L (9) + l (n) + rho (3*n) = 10 + 4*n
    # For n=3: 22 parameters
    default_params = []

    # Load mass
    default_params.append(0.3)  # m_L [kg]

    # Load inertia (3x3 diagonal, flattened row-major)
    J_L_default = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
    default_params.extend(J_L_default)

    # Cable lengths
    for _ in range(num_quads):
        default_params.append(1.0)  # l [m]

    # Attachment points (regular polygon pattern)
    # Paper uses ~0.3m radius triangular configuration
    radius = 0.3  # 30cm from center (matching paper)
    for i in range(num_quads):
        angle = 2.0 * np.pi * i / num_quads
        default_params.append(radius * np.cos(angle))  # x
        default_params.append(radius * np.sin(angle))  # y
        default_params.append(0.0)                      # z

    ocp.parameter_values = np.array(default_params)

    return ocp


def generate_solver(
    num_quads: int = 3,
    output_dir: str = '../generated',
    horizon_time: float = 2.0,
    num_nodes: int = 20
) -> AcadosOcpSolver:
    """
    Generate the ACADOS solver C code.

    Args:
        num_quads: Number of quadrotors
        output_dir: Output directory for generated code
        horizon_time: Prediction horizon [s]
        num_nodes: Number of shooting nodes

    Returns:
        AcadosOcpSolver instance (also generates C code)
    """
    ocp = create_caslo_ocp(num_quads, horizon_time, num_nodes)

    # Set output directory
    ocp.code_export_directory = os.path.join(output_dir, f'caslo_{num_quads}quad')

    # Generate solver
    solver = AcadosOcpSolver(ocp, json_file=os.path.join(output_dir, f'caslo_{num_quads}quad.json'))

    print(f"Generated ACADOS solver for {num_quads}-quadrotor system")
    print(f"  State dimension: {ocp.model.x.size1()}")
    print(f"  Control dimension: {ocp.model.u.size1()}")
    print(f"  Horizon: {horizon_time}s, {num_nodes} nodes")
    print(f"  Output directory: {ocp.code_export_directory}")

    return solver


def main():
    parser = argparse.ArgumentParser(description='Generate CASLO ACADOS solver')
    parser.add_argument('--num-quads', type=int, default=3, help='Number of quadrotors')
    parser.add_argument('--output-dir', type=str, default='../generated', help='Output directory')
    parser.add_argument('--horizon', type=float, default=2.0, help='Horizon time [s]')
    parser.add_argument('--nodes', type=int, default=20, help='Number of shooting nodes')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate solver
    solver = generate_solver(
        num_quads=args.num_quads,
        output_dir=args.output_dir,
        horizon_time=args.horizon,
        num_nodes=args.nodes
    )

    print("\nSolver generation complete!")
    print("To use in Rust:")
    print("  1. Run build.rs to compile the generated C code")
    print("  2. Use bindgen to generate Rust FFI bindings")
    print("  3. Call solver functions through the safe wrapper")


if __name__ == '__main__':
    main()
