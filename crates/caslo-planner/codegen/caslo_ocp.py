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

    State vector (13 + 7*n):
        - p_L: load position (3)
        - v_L: load velocity (3)
        - q_L: load orientation quaternion (4)
        - omega_L: load angular velocity (3)
        - For each cable i:
            - s_i: cable direction unit vector (3)
            - r_i: cable angular velocity (3)
            - t_i: cable tension (1)

    Control vector (4*n):
        - For each cable i:
            - gamma_i: cable angular jerk (3)
            - lambda_i: tension acceleration (1)
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

    # === State Variables ===
    # Load states
    p_L = ca.SX.sym('p_L', 3)         # Position
    v_L = ca.SX.sym('v_L', 3)         # Velocity
    q_L = ca.SX.sym('q_L', 4)         # Quaternion (w, x, y, z)
    omega_L = ca.SX.sym('omega_L', 3) # Angular velocity (body frame)

    # Cable states (per quadrotor)
    s = ca.SX.sym('s', 3, num_quads)  # Cable directions
    r = ca.SX.sym('r', 3, num_quads)  # Cable angular velocities
    t = ca.SX.sym('t', num_quads)     # Cable tensions

    # Assemble state vector
    x = ca.vertcat(
        p_L, v_L, q_L, omega_L,
        ca.reshape(s, -1, 1),
        ca.reshape(r, -1, 1),
        t
    )

    # === Control Variables ===
    gamma = ca.SX.sym('gamma', 3, num_quads)  # Angular jerk
    lam = ca.SX.sym('lambda', num_quads)      # Tension acceleration

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

    # === Cable Kinematics (Eq. 3) ===
    # s_dot = r x s
    # r_dot = gamma
    # t_dot = lambda

    s_dot_list = []
    for i in range(num_quads):
        s_i_dot = cross(r[:, i], s[:, i])
        s_dot_list.append(s_i_dot)

    # Assemble state derivative
    x_dot = ca.vertcat(
        v_L,          # p_L_dot = v_L
        v_L_dot,      # v_L_dot from Eq. 2
        q_L_dot,      # q_L_dot
        omega_L_dot,  # omega_L_dot from Eq. 2
        ca.vertcat(*s_dot_list),  # s_dot = r x s
        ca.reshape(gamma, -1, 1), # r_dot = gamma
        lam           # t_dot = lambda
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

    # For now, use simple error formulation (will refine with parameters)
    # Stage cost residual: tracking errors + control magnitude
    y = ca.vertcat(
        p_L,     # Position (compare with ref)
        v_L,     # Velocity
        ca.vertcat(q_L[1], q_L[2], q_L[3]),  # Quaternion vector part (simplified)
        omega_L, # Angular velocity
        model.u  # Controls (penalize magnitude)
    )

    y_e = ca.vertcat(
        p_L,
        v_L,
        ca.vertcat(q_L[1], q_L[2], q_L[3]),
        omega_L
    )

    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = y_e

    # Weight matrices
    # Position tracking
    W_p = np.diag([10.0, 10.0, 20.0])
    # Velocity tracking
    W_v = np.diag([1.0, 1.0, 2.0])
    # Orientation tracking
    W_q = np.diag([5.0, 5.0, 5.0])
    # Angular velocity tracking
    W_omega = np.diag([0.5, 0.5, 0.5])
    # Control weights
    W_gamma = 0.01 * np.eye(3 * num_quads)  # Angular jerk
    W_lambda = 0.001 * np.eye(num_quads)     # Tension acceleration

    W = np.zeros((ny, ny))
    W[0:3, 0:3] = W_p
    W[3:6, 3:6] = W_v
    W[6:9, 6:9] = W_q
    W[9:12, 9:12] = W_omega
    W[12:12+3*num_quads, 12:12+3*num_quads] = W_gamma
    W[12+3*num_quads:, 12+3*num_quads:] = W_lambda

    ocp.cost.W = W

    # Terminal cost (higher weight)
    W_e = np.zeros((ny_e, ny_e))
    W_e[0:3, 0:3] = 10.0 * W_p
    W_e[3:6, 3:6] = 10.0 * W_v
    W_e[6:9, 6:9] = 10.0 * W_q
    W_e[9:12, 9:12] = 10.0 * W_omega
    ocp.cost.W_e = W_e

    # === Constraints ===

    # Control bounds (Eq. 9 - implied from smoothness)
    max_angular_jerk = 100.0
    max_tension_accel = 500.0

    lbu = np.concatenate([
        -max_angular_jerk * np.ones(3 * num_quads),
        -max_tension_accel * np.ones(num_quads)
    ])
    ubu = np.concatenate([
        max_angular_jerk * np.ones(3 * num_quads),
        max_tension_accel * np.ones(num_quads)
    ])

    ocp.constraints.lbu = lbu
    ocp.constraints.ubu = ubu
    ocp.constraints.idxbu = np.arange(nu)

    # State bounds - tension must be positive (Eq. 10)
    tension_min = 0.5
    tension_max = 50.0

    # Tension states are at indices: 13 + 7*num_quads - num_quads : 13 + 7*num_quads
    # Actually: 13 (load) + 3*n (s) + 3*n (r) + n (t) = 13 + 7*n
    # Tension is at indices: 13 + 6*n : 13 + 7*n
    tension_start = 13 + 6 * num_quads
    tension_end = 13 + 7 * num_quads

    lbx = tension_min * np.ones(num_quads)
    ubx = tension_max * np.ones(num_quads)
    ocp.constraints.lbx = lbx
    ocp.constraints.ubx = ubx
    ocp.constraints.idxbx = np.arange(tension_start, tension_end)

    # Nonlinear constraints: inter-quadrotor collision avoidance (Eq. 11)
    # and obstacle avoidance (Eq. 12)
    # These require computing quadrotor positions from the kinematic constraint

    # For now, set up placeholder for nonlinear constraints
    # Will be populated based on obstacle configuration

    # === Initial State Constraint ===
    ocp.constraints.x0 = np.zeros(nx)
    # Set reasonable initial guess
    ocp.constraints.x0[6] = 1.0  # Quaternion w = 1 (identity)
    for i in range(num_quads):
        ocp.constraints.x0[13 + 3*i + 2] = -1.0  # Cable pointing down
        ocp.constraints.x0[tension_start + i] = 5.0  # Initial tension

    # === Solver Options ===
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'  # Explicit Runge-Kutta
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1

    # RTI for real-time iteration
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.tol = 1e-6

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
    radius = 0.1  # 10cm from center
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
