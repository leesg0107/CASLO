//! 3D Visualization of Cable-Suspended Load System
//!
//! Shows the motion of quadrotors carrying a load via cables,
//! demonstrating obstacle avoidance with no-fly zones.
//!
//! - Load: Blue box (below)
//! - Quadrotors: Red/Green/Yellow spheres (above)
//! - Cables: White lines
//! - No-fly zones: Red transparent spheres
//!
//! Controls:
//! - Mouse drag: Rotate view
//! - Scroll: Zoom
//! - Space: Pause/Resume
//! - R: Reset

use kiss3d::light::Light;
use kiss3d::window::Window;
use kiss3d::scene::SceneNode;
use kiss3d::nalgebra::{Point3, Translation3, UnitQuaternion, Vector3};

use std::f32::consts::PI;

/// Obstacle (no-fly zone)
struct Obstacle {
    center: Vector3<f32>,
    radius: f32,
    node: SceneNode,
}

fn main() {
    let mut window = Window::new("CASLO - Obstacle Avoidance Visualization");
    window.set_light(Light::StickToCamera);
    window.set_background_color(0.05, 0.05, 0.1);

    // Create load (blue box) - hangs below drones
    let mut load = window.add_cube(0.15, 0.15, 0.08);
    load.set_color(0.2, 0.4, 0.9);

    // Create quadrotors (colored spheres)
    let num_quads = 3;
    let quad_colors = [
        (0.9, 0.2, 0.2),  // Red
        (0.2, 0.9, 0.2),  // Green
        (0.9, 0.9, 0.2),  // Yellow
    ];

    let mut quads: Vec<SceneNode> = Vec::new();
    for i in 0..num_quads {
        let mut quad = window.add_sphere(0.06);
        quad.set_color(quad_colors[i].0, quad_colors[i].1, quad_colors[i].2);
        quads.push(quad);
    }

    // Create obstacles (no-fly zones) - paper scenario
    let mut obstacles: Vec<Obstacle> = Vec::new();

    // Obstacle 1: In the middle of the path
    let mut obs1 = window.add_sphere(0.3);
    obs1.set_color(1.0, 0.2, 0.2);
    obs1.set_local_translation(Translation3::new(0.8, 0.5, 1.8));
    obstacles.push(Obstacle {
        center: Vector3::new(0.8, 0.5, 1.8),
        radius: 0.3,
        node: obs1,
    });

    // Obstacle 2: Another obstacle
    let mut obs2 = window.add_sphere(0.25);
    obs2.set_color(1.0, 0.3, 0.1);
    obs2.set_local_translation(Translation3::new(1.2, 0.8, 2.2));
    obstacles.push(Obstacle {
        center: Vector3::new(1.2, 0.8, 2.2),
        radius: 0.25,
        node: obs2,
    });

    // Obstacle 3: Vertical cylinder-like (represented as sphere)
    let mut obs3 = window.add_sphere(0.2);
    obs3.set_color(1.0, 0.4, 0.1);
    obs3.set_local_translation(Translation3::new(0.4, 0.9, 2.0));
    obstacles.push(Obstacle {
        center: Vector3::new(0.4, 0.9, 2.0),
        radius: 0.2,
        node: obs3,
    });

    // System parameters
    let cable_length = 0.8_f32;
    let attachment_radius = 0.08_f32;

    // Trajectory: Start -> avoid obstacles -> End
    // Paper scenario: navigating through no-fly zones
    let start_pos = Vector3::new(0.0_f32, 0.0, 1.5);
    let end_pos = Vector3::new(1.8_f32, 1.2, 2.0);

    // Waypoints to avoid obstacles (simplified path planning result)
    let waypoints = vec![
        Vector3::new(0.0, 0.0, 1.5),      // Start
        Vector3::new(0.3, 0.2, 1.6),      // Move up slightly
        Vector3::new(0.5, 0.3, 2.3),      // Go above obstacle 1
        Vector3::new(0.9, 0.4, 2.5),      // Clear obstacle 1
        Vector3::new(1.3, 0.6, 2.6),      // Above obstacle 2
        Vector3::new(1.6, 1.0, 2.3),      // Coming down
        Vector3::new(1.8, 1.2, 2.0),      // End
    ];

    let duration = 6.0_f32;

    // Animation state
    let mut time = 0.0_f32;
    let mut paused = false;
    let speed = 0.8_f32;

    // Camera setup - looking at the scene with Z up
    // Position camera to look from side/above
    let eye = Point3::new(4.0, -2.0, 3.0);
    let at = Point3::new(0.9, 0.6, 2.0);

    println!("=== CASLO Obstacle Avoidance Visualization ===");
    println!("Paper scenario: Navigating through no-fly zones\n");
    println!("Controls:");
    println!("  Mouse drag: Rotate view");
    println!("  Scroll: Zoom");
    println!("  Space: Pause/Resume");
    println!("  R: Reset animation");
    println!("");
    println!("Scene:");
    println!("  Blue box: Load (payload)");
    println!("  Colored spheres: Quadrotors");
    println!("  Red spheres: No-fly zones (obstacles)");
    println!("  Blue line: Planned trajectory");
    println!("");

    while window.render_with_camera(&mut kiss3d::camera::ArcBall::new(eye, at)) {
        // Handle input
        for event in window.events().iter() {
            match event.value {
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::Space, kiss3d::event::Action::Press, _) => {
                    paused = !paused;
                    println!("{}", if paused { "Paused" } else { "Resumed" });
                }
                kiss3d::event::WindowEvent::Key(kiss3d::event::Key::R, kiss3d::event::Action::Press, _) => {
                    time = 0.0;
                    println!("Reset");
                }
                _ => {}
            }
        }

        // Update time
        if !paused {
            time += 1.0 / 60.0 * speed;
            if time > duration + 2.0 {
                time = 0.0;  // Loop animation
            }
        }

        // Compute position along waypoint path
        let t_normalized = (time / duration).clamp(0.0, 1.0);
        let load_pos = interpolate_waypoints(&waypoints, t_normalized);

        // Compute velocity for load tilt
        let dt = 0.01;
        let t_next = ((time + dt) / duration).clamp(0.0, 1.0);
        let next_pos = interpolate_waypoints(&waypoints, t_next);
        let velocity = (next_pos - load_pos) / dt;

        // Update load position
        load.set_local_translation(Translation3::new(load_pos.x, load_pos.y, load_pos.z));

        // Slight tilt based on velocity (load swings slightly)
        let tilt_x = (velocity.y * 0.05).clamp(-0.2, 0.2);
        let tilt_y = (-velocity.x * 0.05).clamp(-0.2, 0.2);
        let load_rotation = UnitQuaternion::from_euler_angles(tilt_x, tilt_y, 0.0);
        load.set_local_rotation(load_rotation);

        // Update quadrotor positions (above the load)
        for (i, quad) in quads.iter_mut().enumerate() {
            let angle = 2.0 * PI * i as f32 / num_quads as f32;
            let attachment = Vector3::new(
                attachment_radius * angle.cos(),
                attachment_radius * angle.sin(),
                0.0,
            );

            // Cable direction points DOWN (from quad to load)
            // So quad is ABOVE the attachment point
            let swing = 0.05 * (time * 2.0 + i as f32 * 0.5).sin();
            let cable_dir = Vector3::new(
                swing * angle.cos() - velocity.x * 0.02,
                swing * angle.sin() - velocity.y * 0.02,
                -1.0,  // Points down
            ).normalize();

            // Attachment point on load (in world frame)
            let attach_world = load_pos + load_rotation * attachment;

            // Quadrotor is above: attachment - cable_length * cable_dir
            // Since cable_dir.z is negative, subtracting makes quad go UP
            let quad_pos = attach_world - cable_length * cable_dir;

            quad.set_local_translation(Translation3::new(quad_pos.x, quad_pos.y, quad_pos.z));

            // Draw cable
            let p1 = Point3::new(attach_world.x, attach_world.y, attach_world.z);
            let p2 = Point3::new(quad_pos.x, quad_pos.y, quad_pos.z);
            window.draw_line(&p1, &p2, &Point3::new(0.9, 0.9, 0.9));
        }

        // Draw planned trajectory
        let steps = 100;
        for i in 0..steps {
            let t1 = i as f32 / steps as f32;
            let t2 = (i + 1) as f32 / steps as f32;
            let p1 = interpolate_waypoints(&waypoints, t1);
            let p2 = interpolate_waypoints(&waypoints, t2);

            // Color: bright ahead, dim behind
            let brightness = if t1 < t_normalized { 0.3 } else { 0.8 };
            window.draw_line(
                &Point3::new(p1.x, p1.y, p1.z),
                &Point3::new(p2.x, p2.y, p2.z),
                &Point3::new(0.3, 0.5, brightness),
            );
        }

        // Draw start marker (green)
        draw_marker(&mut window, start_pos, Point3::new(0.0, 1.0, 0.0), 0.08);

        // Draw end marker (cyan)
        draw_marker(&mut window, end_pos, Point3::new(0.0, 1.0, 1.0), 0.08);

        // Draw ground plane grid
        draw_ground(&mut window, 0.0);

        // Draw coordinate axes at origin
        let origin = Point3::new(0.0, 0.0, 0.0);
        window.draw_line(&origin, &Point3::new(0.5, 0.0, 0.0), &Point3::new(1.0, 0.0, 0.0)); // X: red
        window.draw_line(&origin, &Point3::new(0.0, 0.5, 0.0), &Point3::new(0.0, 1.0, 0.0)); // Y: green
        window.draw_line(&origin, &Point3::new(0.0, 0.0, 0.5), &Point3::new(0.0, 0.0, 1.0)); // Z: blue (up)

        // Draw obstacle boundaries (wireframe circles)
        for obs in &obstacles {
            draw_sphere_wireframe(&mut window, obs.center, obs.radius, Point3::new(1.0, 0.3, 0.3));
        }
    }
}

/// Interpolate position along waypoints using Catmull-Rom spline
fn interpolate_waypoints(waypoints: &[Vector3<f32>], t: f32) -> Vector3<f32> {
    if waypoints.is_empty() {
        return Vector3::zeros();
    }
    if waypoints.len() == 1 {
        return waypoints[0];
    }

    let t = t.clamp(0.0, 1.0);
    let n = waypoints.len() - 1;
    let segment_t = t * n as f32;
    let segment = (segment_t as usize).min(n - 1);
    let local_t = segment_t - segment as f32;

    // Simple linear interpolation between waypoints
    // (Could use Catmull-Rom for smoother curves)
    let p0 = waypoints[segment];
    let p1 = waypoints[segment + 1];

    // Smooth step for nicer motion
    let smooth_t = local_t * local_t * (3.0 - 2.0 * local_t);

    p0 + (p1 - p0) * smooth_t
}

/// Draw a marker at position
fn draw_marker(window: &mut Window, pos: Vector3<f32>, color: Point3<f32>, size: f32) {
    let p = Point3::new(pos.x, pos.y, pos.z);

    window.draw_line(
        &Point3::new(p.x - size, p.y, p.z),
        &Point3::new(p.x + size, p.y, p.z),
        &color,
    );
    window.draw_line(
        &Point3::new(p.x, p.y - size, p.z),
        &Point3::new(p.x, p.y + size, p.z),
        &color,
    );
    window.draw_line(
        &Point3::new(p.x, p.y, p.z - size),
        &Point3::new(p.x, p.y, p.z + size),
        &color,
    );
}

/// Draw ground plane
fn draw_ground(window: &mut Window, z: f32) {
    let size = 2.5_f32;
    let step = 0.5_f32;
    let color = Point3::new(0.2, 0.2, 0.25);

    let mut x = -size;
    while x <= size {
        window.draw_line(
            &Point3::new(x, -size, z),
            &Point3::new(x, size, z),
            &color,
        );
        window.draw_line(
            &Point3::new(-size, x, z),
            &Point3::new(size, x, z),
            &color,
        );
        x += step;
    }
}

/// Draw wireframe sphere (obstacle boundary)
fn draw_sphere_wireframe(window: &mut Window, center: Vector3<f32>, radius: f32, color: Point3<f32>) {
    let segments = 16;

    // Draw horizontal circles
    for ring in 0..3 {
        let z_offset = (ring as f32 - 1.0) * radius * 0.7;
        let r = (radius * radius - z_offset * z_offset).sqrt();

        for i in 0..segments {
            let a1 = 2.0 * PI * i as f32 / segments as f32;
            let a2 = 2.0 * PI * (i + 1) as f32 / segments as f32;

            let p1 = Point3::new(
                center.x + r * a1.cos(),
                center.y + r * a1.sin(),
                center.z + z_offset,
            );
            let p2 = Point3::new(
                center.x + r * a2.cos(),
                center.y + r * a2.sin(),
                center.z + z_offset,
            );
            window.draw_line(&p1, &p2, &color);
        }
    }

    // Draw vertical arcs
    for i in 0..4 {
        let angle = PI * i as f32 / 4.0;
        for j in 0..segments {
            let a1 = PI * j as f32 / segments as f32;
            let a2 = PI * (j + 1) as f32 / segments as f32;

            let p1 = Point3::new(
                center.x + radius * a1.sin() * angle.cos(),
                center.y + radius * a1.sin() * angle.sin(),
                center.z + radius * a1.cos(),
            );
            let p2 = Point3::new(
                center.x + radius * a2.sin() * angle.cos(),
                center.y + radius * a2.sin() * angle.sin(),
                center.z + radius * a2.cos(),
            );
            window.draw_line(&p1, &p2, &color);
        }
    }
}
