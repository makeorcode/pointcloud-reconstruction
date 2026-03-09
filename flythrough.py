#!/usr/bin/env python3
"""
First-person flythrough viewer for dense point clouds.
Uses raw OpenGL + GLFW for true FPS camera controls.

Controls:
    W/S           - Move forward/backward
    A/D           - Strafe left/right
    Space/LCtrl   - Move up/down
    Left-click    - Hold and drag to look around
    Mid/Right-click - Hold and drag to pan
    Scroll        - Adjust move speed
    +/-           - Point size
    C             - Cycle color mode
    E             - Toggle Eye-Dome Lighting (EDL)
    L / Shift+L   - EDL strength up/down
    R             - Reset to center
    P             - Print camera position
    1-5           - Speed presets
    X             - Filter outliers (statistical)
    F12           - Save / overwrite original PCD
    Esc           - Quit

    Scene Rotation (rotates the entire point cloud):
    F5            - Rotate scene +90° around X
    F6            - Rotate scene -90° around X
    F7            - Rotate scene +90° around Y
    F8            - Rotate scene -90° around Y
    F9            - Rotate scene +90° around Z
    F10           - Rotate scene -90° around Z
    F1            - Reset scene rotation

Usage:
    python flythrough.py accumulated.pcd
    python flythrough.py deskewed.npz
"""

import sys
import math
import ctypes
import numpy as np
from pathlib import Path

import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shaders
from OpenGL.GLU import *
try:
    from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_9_BY_15
    _GLUT_AVAILABLE = True
except Exception:
    _GLUT_AVAILABLE = False


# ── GLSL Shaders ─────────────────────────────────────────────────

# Point vertex shader: distance-based size attenuation + pass-through color
POINT_VERT = """
#version 120
uniform float u_point_size;
uniform float u_near;
uniform float u_far;
varying vec3 v_color;
varying float v_point_size;
void main() {
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    v_color = gl_Color.rgb;

    // Distance attenuation: size inversely proportional to eye distance
    float dist = length((gl_ModelViewMatrix * gl_Vertex).xyz);
    float att_size = u_point_size * 4.0 / max(dist, 0.1);
    att_size = clamp(att_size, 1.0, 64.0);
    gl_PointSize = att_size;
    v_point_size = att_size;
}
"""

# Point fragment shader: circular splat with smooth edge
POINT_FRAG = """
#version 120
varying vec3 v_color;
varying float v_point_size;
void main() {
    // Circular splat: discard outside radius
    vec2 coord = gl_PointCoord - vec2(0.5);
    float r2 = dot(coord, coord);
    if (r2 > 0.25) discard;  // outside unit circle

    // Smooth edge falloff
    float alpha = 1.0 - smoothstep(0.15, 0.25, r2);
    gl_FragColor = vec4(v_color, alpha);
}
"""

# EDL fullscreen vertex shader
EDL_VERT = """
#version 120
varying vec2 v_uv;
void main() {
    v_uv = gl_MultiTexCoord0.xy;
    gl_Position = gl_Vertex;
}
"""

# EDL fullscreen fragment shader
EDL_FRAG = """
#version 120
uniform sampler2D u_color_tex;
uniform sampler2D u_depth_tex;
uniform vec2 u_screen_size;
uniform float u_edl_strength;
uniform float u_edl_radius;
uniform float u_near;
uniform float u_far;

varying vec2 v_uv;

float linearize_depth(float d) {
    // Convert from [0,1] depth buffer to linear eye-space depth
    return (2.0 * u_near * u_far) / (u_far + u_near - (2.0 * d - 1.0) * (u_far - u_near));
}

float edl_response(vec2 uv) {
    float depth = linearize_depth(texture2D(u_depth_tex, uv).r);
    float log_d = log2(max(depth, 0.001));

    // Sample 8 neighbors
    vec2 pixel = u_edl_radius / u_screen_size;
    float sum = 0.0;
    sum += max(0.0, log_d - log2(max(linearize_depth(texture2D(u_depth_tex, uv + vec2( pixel.x, 0.0)).r), 0.001)));
    sum += max(0.0, log_d - log2(max(linearize_depth(texture2D(u_depth_tex, uv + vec2(-pixel.x, 0.0)).r), 0.001)));
    sum += max(0.0, log_d - log2(max(linearize_depth(texture2D(u_depth_tex, uv + vec2(0.0,  pixel.y)).r), 0.001)));
    sum += max(0.0, log_d - log2(max(linearize_depth(texture2D(u_depth_tex, uv + vec2(0.0, -pixel.y)).r), 0.001)));
    sum += max(0.0, log_d - log2(max(linearize_depth(texture2D(u_depth_tex, uv + vec2( pixel.x,  pixel.y)).r), 0.001)));
    sum += max(0.0, log_d - log2(max(linearize_depth(texture2D(u_depth_tex, uv + vec2(-pixel.x,  pixel.y)).r), 0.001)));
    sum += max(0.0, log_d - log2(max(linearize_depth(texture2D(u_depth_tex, uv + vec2( pixel.x, -pixel.y)).r), 0.001)));
    sum += max(0.0, log_d - log2(max(linearize_depth(texture2D(u_depth_tex, uv + vec2(-pixel.x, -pixel.y)).r), 0.001)));

    return sum / 8.0;
}

void main() {
    vec4 color = texture2D(u_color_tex, v_uv);
    float depth_val = texture2D(u_depth_tex, v_uv).r;

    // Skip background (depth = 1.0)
    if (depth_val >= 0.9999) {
        gl_FragColor = color;
        return;
    }

    float response = edl_response(v_uv);
    float shade = exp(-response * u_edl_strength * 300.0);
    gl_FragColor = vec4(color.rgb * shade, 1.0);
}
"""


def _compile_shader_program(vert_src, frag_src):
    """Compile and link a GLSL shader program."""
    vert = gl_shaders.compileShader(vert_src, GL_VERTEX_SHADER)
    frag = gl_shaders.compileShader(frag_src, GL_FRAGMENT_SHADER)
    return gl_shaders.compileProgram(vert, frag)


def _create_fbo(w, h):
    """Create an FBO with color and depth textures."""
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    # Color texture
    color_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, color_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0)

    # Depth texture
    depth_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0)

    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    if status != GL_FRAMEBUFFER_COMPLETE:
        print(f"WARNING: FBO incomplete (status={status}), EDL disabled")
        return None, None, None

    return fbo, color_tex, depth_tex


def height_colors(z):
    if z.max() - z.min() < 1e-6:
        return np.ones((len(z), 3), dtype=np.float32) * 0.5
    z_norm = (z - z.min()) / (z.max() - z.min())
    colors = np.zeros((len(z), 3), dtype=np.float32)
    colors[:, 0] = np.clip(4.0 * z_norm - 2.0, 0, 1)
    colors[:, 1] = np.where(z_norm < 0.5,
                            np.clip(4.0 * z_norm, 0, 1),
                            np.clip(4.0 - 4.0 * z_norm, 0, 1))
    colors[:, 2] = np.clip(2.0 - 4.0 * z_norm, 0, 1)
    return colors


def distance_colors(pts):
    dist = np.linalg.norm(pts, axis=1)
    if dist.max() - dist.min() < 1e-6:
        return np.ones((len(pts), 3), dtype=np.float32) * 0.5
    d_norm = (dist - dist.min()) / (dist.max() - dist.min())
    colors = np.zeros((len(pts), 3), dtype=np.float32)
    colors[:, 0] = d_norm
    colors[:, 1] = 1.0 - np.abs(d_norm - 0.5) * 2
    colors[:, 2] = 1.0 - d_norm
    return colors


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def kabsch_transform(src, dst):
    """Compute rigid transform (R, t) aligning src points to dst points using SVD.

    Args:
        src: Nx3 array of source points
        dst: Nx3 array of destination points (same N)

    Returns:
        4x4 transformation matrix
    """
    assert src.shape == dst.shape and src.shape[0] >= 3
    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    s = src - centroid_src
    d = dst - centroid_dst
    H = s.T @ d
    U, S, Vt = np.linalg.svd(H)
    # Correct reflection
    det = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(det)])
    R = Vt.T @ sign_matrix @ U.T
    t = centroid_dst - R @ centroid_src
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_points(input_path):
    input_path = Path(input_path)
    if input_path.suffix == '.npz':
        data = np.load(str(input_path), allow_pickle=True)
        clouds = list(data['points'])
        all_pts = []
        for c in clouds:
            valid = np.isfinite(c).all(axis=1)
            pts = c[valid]
            dist = np.linalg.norm(pts, axis=1)
            pts = pts[(dist < 100.0) & (dist > 0.3)]
            all_pts.append(pts)
        points = np.vstack(all_pts).astype(np.float32)
    elif input_path.suffix in ('.pcd', '.ply'):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(input_path))
        points = np.asarray(pcd.points, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported format: {input_path.suffix}")
    print(f"Loaded {len(points):,} points from {input_path}")
    return points


class FPSViewer:
    # Distinct colors for compare mode (up to 4 clouds)
    COMPARE_COLORS = [
        np.array([0.2, 0.6, 1.0], dtype=np.float32),   # blue
        np.array([1.0, 0.4, 0.2], dtype=np.float32),   # orange
        np.array([0.2, 1.0, 0.4], dtype=np.float32),   # green
        np.array([1.0, 0.2, 0.8], dtype=np.float32),   # magenta
    ]

    def __init__(self, point_clouds, cloud_names=None, cloud_paths=None):
        """point_clouds: list of np arrays, one per file."""
        self.cloud_list = point_clouds
        self.cloud_names = cloud_names or [f"Cloud {i+1}" for i in range(len(point_clouds))]
        self.cloud_paths = cloud_paths or [None] * len(point_clouds)
        self.cloud_visible = [True] * len(point_clouds)
        self.compare_mode = len(point_clouds) > 1

        # Combine all for centroid / single-cloud fallback
        self.original_points = np.vstack(point_clouds)
        self.n_points = len(self.original_points)

        # Per-cloud original points (for scene rotation)
        self.original_clouds = [c.copy() for c in point_clouds]

        # Scene rotation matrix (applied to point cloud before rendering)
        # Default: F7, F7, F7, F6 equivalent (Y+90, Y+90, Y+90, X-90)
        self.scene_rot = np.array([[0, -1, 0],
                                    [0, 0, -1],
                                    [1, 0, 0]], dtype=np.float64)
        self._apply_scene_rotation()

        # Camera - Z-up, start at centroid looking along +X
        self.yaw = 0.0    # rotation in XY plane (around Z)
        self.pitch = 0.0   # tilt up/down from horizontal

        self.move_speed = 0.10
        self.mouse_sensitivity = 0.003
        self.point_size = 0.5

        # Color
        self.color_mode = 0
        if self.compare_mode:
            self.color_names = ['compare', 'height', 'distance', 'white']
        else:
            self.color_names = ['height', 'distance', 'white']
        self._build_colors()

        # Mouse state
        self.mouse_look = False
        self.mouse_pan = False
        self.last_mx = 0.0
        self.last_my = 0.0

        # Pick mode (for stitching)
        self.pick_mode = False
        self.pick_cloud = 0          # alternates: 0 = cloud 1, 1 = cloud 2
        self.picks = [[], []]        # picks[i] = list of (x,y,z) in rotated coords
        self.aligned = False         # True after G alignment

        # Grab mode (interactive cloud alignment)
        self.grab_mode = False
        self.grab_transform = np.eye(4, dtype=np.float64)
        self.grab_cloud_backup = None  # snapshot of original_clouds[1] on enter
        self.mouse_grab = False
        self.mouse_grab_rotate = False

        # Text input mode (for merge filename)
        self.text_input_active = False
        self.text_input_buffer = ""
        self.text_input_prompt = ""
        self.text_input_callback = None

        # Rendering
        self.edl_enabled = True
        self.edl_strength = 1.0
        self.edl_radius = 1.5
        self.near_plane = 0.1
        self.far_plane = 500.0
        self._fbo = None
        self._fbo_w = 0
        self._fbo_h = 0
        self._color_tex = None
        self._depth_tex = None
        self._point_shader = None
        self._edl_shader = None
        self._quad_vbo = None

        # HUD
        self.show_help = True   # start with help visible
        self.toast_msg = ""
        self.toast_time = 0.0

        # Window
        self.width = 1600
        self.height = 900

    def _apply_scene_rotation(self):
        """Apply scene rotation to points, recompute centroid, re-upload if needed."""
        self.rotated_clouds = []
        for c in self.original_clouds:
            rotated = (c.astype(np.float64) @ self.scene_rot.T).astype(np.float32)
            self.rotated_clouds.append(rotated)
        self.points = np.vstack(self.rotated_clouds)
        centroid = np.mean(self.points, axis=0)
        self.cam_pos = centroid.copy().astype(np.float64)
        self._points_dirty = True

    def _apply_grab_transform(self):
        """Apply grab_transform to cloud 2, refresh display without resetting camera."""
        T = self.grab_transform
        orig = self.grab_cloud_backup.astype(np.float64)
        transformed = (T[:3, :3] @ orig.T).T + T[:3, 3]
        self.original_clouds[1] = transformed.astype(np.float32)

        # Refresh scene rotation and colors (but don't reset camera)
        self.rotated_clouds = []
        for c in self.original_clouds:
            rotated = (c.astype(np.float64) @ self.scene_rot.T).astype(np.float32)
            self.rotated_clouds.append(rotated)
        self.points = np.vstack(self.rotated_clouds)
        self.n_points = len(self.points)
        self._points_dirty = True
        self._build_colors()

    def _build_colors(self):
        if self.compare_mode and self.color_mode == 0:
            # Compare mode: each cloud gets a solid color
            parts = []
            for i, cloud in enumerate(self.rotated_clouds):
                color = self.COMPARE_COLORS[i % len(self.COMPARE_COLORS)]
                parts.append(np.tile(color, (len(cloud), 1)))
            self.colors = np.vstack(parts).astype(np.float32)
        elif self.color_names[self.color_mode] == 'height':
            self.colors = height_colors(self.points[:, 2])  # Z is up
        elif self.color_names[self.color_mode] == 'distance':
            self.colors = distance_colors(self.points)
        else:
            self.colors = np.ones((self.n_points, 3), dtype=np.float32) * 0.85
        self._colors_dirty = True

    def _get_forward(self):
        # Z-up: yaw rotates in XY plane, pitch tilts toward Z
        return np.array([
            math.cos(self.pitch) * math.cos(self.yaw),
            math.cos(self.pitch) * math.sin(self.yaw),
            math.sin(self.pitch),
        ], dtype=np.float64)

    def _get_right(self):
        # Right is perpendicular to forward in XY plane
        return np.array([
            math.cos(self.yaw - math.pi / 2),
            math.sin(self.yaw - math.pi / 2),
            0.0,
        ], dtype=np.float64)

    def run(self):
        if not glfw.init():
            print("Failed to initialize GLFW")
            return

        glfw.window_hint(glfw.SAMPLES, 4)
        window = glfw.create_window(self.width, self.height,
                                     "Point Cloud Flythrough", None, None)
        if not window:
            glfw.terminate()
            return

        glfw.make_context_current(window)
        glfw.swap_interval(1)  # vsync

        # Initialize GLUT for bitmap text rendering
        if _GLUT_AVAILABLE:
            try:
                glutInit()
            except Exception:
                pass

        # Callbacks
        glfw.set_mouse_button_callback(window, self._mouse_button_cb)
        glfw.set_cursor_pos_callback(window, self._cursor_pos_cb)
        glfw.set_scroll_callback(window, self._scroll_cb)
        glfw.set_key_callback(window, self._key_cb)
        glfw.set_char_callback(window, self._char_cb)
        glfw.set_framebuffer_size_callback(window, self._resize_cb)

        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.05, 0.05, 0.08, 1.0)

        # Enable point sprites for circular splats
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)

        # Compile shaders
        try:
            self._point_shader = _compile_shader_program(POINT_VERT, POINT_FRAG)
            self._edl_shader = _compile_shader_program(EDL_VERT, EDL_FRAG)
        except Exception as e:
            print(f"WARNING: Shader compilation failed ({e}), using fixed pipeline")
            self._point_shader = None
            self._edl_shader = None
            self.edl_enabled = False

        # Create fullscreen quad VBO for EDL pass
        quad = np.array([
            -1, -1, 0, 0, 0,
             1, -1, 0, 1, 0,
             1,  1, 0, 1, 1,
            -1,  1, 0, 0, 1,
        ], dtype=np.float32)
        self._quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        # Create FBO for EDL
        w_fb, h_fb = glfw.get_framebuffer_size(window)
        self._init_fbo(w_fb, h_fb)

        # Upload point data to VBOs - one pair per cloud
        self.vbo_pos_list = []
        self.vbo_col_list = []
        offset = 0
        for i, cloud in enumerate(self.rotated_clouds):
            vbo_p = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_p)
            glBufferData(GL_ARRAY_BUFFER, cloud.nbytes, cloud, GL_STATIC_DRAW)
            self.vbo_pos_list.append(vbo_p)

            n = len(cloud)
            cloud_colors = self.colors[offset:offset + n]
            vbo_c = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_c)
            glBufferData(GL_ARRAY_BUFFER, cloud_colors.nbytes, cloud_colors, GL_STATIC_DRAW)
            self.vbo_col_list.append(vbo_c)
            offset += n
        self._points_dirty = False
        self._colors_dirty = False

        glPointSize(self.point_size)

        print("\n=== First-Person Point Cloud Flythrough ===")
        print(f"  {self.n_points:,} points")
        print(f"  Controls:")
        print(f"    W/A/S/D         - Move")
        print(f"    Space/LCtrl     - Up/down")
        print(f"    Left-click drag - Look around")
        print(f"    Mid/Right drag  - Pan")
        print(f"    Scroll          - Adjust speed")
        print(f"    1-5             - Speed presets")
        print(f"    +/-             - Point size")
        print(f"    C               - Cycle color")
        print(f"    R               - Reset position")
        print(f"    P               - Print position")
        print(f"  Scene Rotation:")
        print(f"    F5/F6           - Rotate scene ±90° around X")
        print(f"    F7/F8           - Rotate scene ±90° around Y")
        print(f"    F9/F10          - Rotate scene ±90° around Z")
        print(f"    F1              - Reset scene rotation")
        print(f"    Esc             - Quit")
        print()

        last_time = glfw.get_time()

        while not glfw.window_should_close(window):
            now = glfw.get_time()
            dt = now - last_time
            last_time = now

            self._process_movement(window, dt)
            self._render(window)

            glfw.swap_buffers(window)
            glfw.poll_events()

        # Cleanup GPU resources
        for vbo in self.vbo_pos_list + self.vbo_col_list:
            glDeleteBuffers(1, [vbo])
        if self._quad_vbo:
            glDeleteBuffers(1, [self._quad_vbo])
        if self._point_shader:
            glDeleteProgram(self._point_shader)
        if self._edl_shader:
            glDeleteProgram(self._edl_shader)
        if self._fbo is not None:
            glDeleteFramebuffers(1, [self._fbo])
            glDeleteTextures([self._color_tex, self._depth_tex])
        glfw.terminate()

    def _upload_colors(self):
        offset = 0
        for i, cloud in enumerate(self.rotated_clouds):
            n = len(cloud)
            cloud_colors = self.colors[offset:offset + n]
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col_list[i])
            glBufferData(GL_ARRAY_BUFFER, cloud_colors.nbytes, cloud_colors, GL_STATIC_DRAW)
            offset += n
        self._colors_dirty = False

    def _upload_points(self):
        for i, cloud in enumerate(self.rotated_clouds):
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos_list[i])
            glBufferData(GL_ARRAY_BUFFER, cloud.nbytes, cloud, GL_STATIC_DRAW)
        self._points_dirty = False

    def _init_fbo(self, w, h):
        """Create or resize the FBO for EDL rendering."""
        if w == self._fbo_w and h == self._fbo_h and self._fbo is not None:
            return
        # Clean up old FBO
        if self._fbo is not None:
            glDeleteFramebuffers(1, [self._fbo])
            glDeleteTextures([self._color_tex, self._depth_tex])
        self._fbo, self._color_tex, self._depth_tex = _create_fbo(w, h)
        self._fbo_w = w
        self._fbo_h = h
        if self._fbo is None:
            self.edl_enabled = False
            print("WARNING: FBO creation failed, EDL disabled")

    def _process_movement(self, window, dt):
        speed = self.move_speed * dt * 60.0
        forward = self._get_forward()
        right = self._get_right()
        up = np.array([0.0, 0.0, 1.0])  # Z-up

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.cam_pos += forward * speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.cam_pos -= forward * speed
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.cam_pos -= right * speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.cam_pos += right * speed
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            self.cam_pos += up * speed
        if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            self.cam_pos -= up * speed

    def _setup_camera(self, w, h):
        """Set up projection and modelview matrices."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(70.0, w / h, self.near_plane, self.far_plane)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        forward = self._get_forward()
        target = self.cam_pos + forward
        up = np.array([0.0, 0.0, 1.0])
        gluLookAt(
            self.cam_pos[0], self.cam_pos[1], self.cam_pos[2],
            target[0], target[1], target[2],
            up[0], up[1], up[2]
        )

    def _draw_points(self):
        """Draw all visible point clouds."""
        if self._points_dirty:
            self._upload_points()
        if self._colors_dirty:
            self._upload_colors()

        use_shader = self._point_shader is not None

        if use_shader:
            glUseProgram(self._point_shader)
            glUniform1f(glGetUniformLocation(self._point_shader, "u_point_size"),
                        self.point_size)
            glUniform1f(glGetUniformLocation(self._point_shader, "u_near"),
                        self.near_plane)
            glUniform1f(glGetUniformLocation(self._point_shader, "u_far"),
                        self.far_plane)
        else:
            glPointSize(self.point_size)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        for i, cloud in enumerate(self.rotated_clouds):
            if not self.cloud_visible[i]:
                continue
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos_list[i])
            glVertexPointer(3, GL_FLOAT, 0, None)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col_list[i])
            glColorPointer(3, GL_FLOAT, 0, None)
            glDrawArrays(GL_POINTS, 0, len(cloud))

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        if use_shader:
            glUseProgram(0)

    def _draw_edl_quad(self, w, h):
        """Fullscreen EDL post-processing pass."""
        glUseProgram(self._edl_shader)

        # Bind color and depth textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._color_tex)
        glUniform1i(glGetUniformLocation(self._edl_shader, "u_color_tex"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self._depth_tex)
        glUniform1i(glGetUniformLocation(self._edl_shader, "u_depth_tex"), 1)

        glUniform2f(glGetUniformLocation(self._edl_shader, "u_screen_size"),
                     float(w), float(h))
        glUniform1f(glGetUniformLocation(self._edl_shader, "u_edl_strength"),
                     self.edl_strength)
        glUniform1f(glGetUniformLocation(self._edl_shader, "u_edl_radius"),
                     self.edl_radius)
        glUniform1f(glGetUniformLocation(self._edl_shader, "u_near"),
                     self.near_plane)
        glUniform1f(glGetUniformLocation(self._edl_shader, "u_far"),
                     self.far_plane)

        # Draw fullscreen quad
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)

        glBindBuffer(GL_ARRAY_BUFFER, self._quad_vbo)
        stride = 5 * 4  # 5 floats * 4 bytes
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, stride, None)
        glClientActiveTexture(GL_TEXTURE0)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(12))

        glDrawArrays(GL_QUADS, 0, 4)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)

        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        glUseProgram(0)
        glActiveTexture(GL_TEXTURE0)

    def _render(self, window):
        w, h = glfw.get_framebuffer_size(window)
        if h == 0:
            h = 1

        use_edl = (self.edl_enabled and self._edl_shader is not None
                   and self._fbo is not None)

        # If using EDL, render scene to FBO first
        if use_edl:
            self._init_fbo(w, h)  # resize if needed
            glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._setup_camera(w, h)
        self._draw_points()

        # Draw pick markers
        if self.pick_mode or any(len(p) > 0 for p in self.picks):
            saved_size = self.point_size
            glPointSize(15.0)
            if self._point_shader:
                glUseProgram(0)  # fixed pipeline for markers
            glDisable(GL_DEPTH_TEST)
            glBegin(GL_POINTS)
            for pt in self.picks[0]:
                glColor3f(0.0, 1.0, 1.0)
                glVertex3f(pt[0], pt[1], pt[2])
            for pt in self.picks[1]:
                glColor3f(1.0, 1.0, 0.0)
                glVertex3f(pt[0], pt[1], pt[2])
            glEnd()
            glEnable(GL_DEPTH_TEST)
            glPointSize(saved_size)

        if use_edl:
            # Unbind FBO, apply EDL to screen
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, w, h)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._draw_edl_quad(w, h)

        # Draw 3D axis indicator at world origin
        self._draw_axis_world()

        # Draw HUD: crosshair + axis indicator in corner
        self._draw_hud(w, h)

    def _draw_axis_world(self):
        """Draw RGB XYZ axis lines at world origin."""
        glDisable(GL_DEPTH_TEST)
        glLineWidth(3.0)
        axis_len = 2.0

        glBegin(GL_LINES)
        # X - Red
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(axis_len, 0, 0)
        # Y - Green
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, axis_len, 0)
        # Z - Blue
        glColor3f(0.3, 0.3, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, axis_len)
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glLineWidth(1.0)

    def _set_toast(self, msg):
        """Show a brief message on screen."""
        self.toast_msg = msg
        self.toast_time = glfw.get_time()

    def _pick_point(self, window, screen_x, screen_y):
        """Ray-cast from screen click and find nearest point in the active pick cloud."""
        if not self.compare_mode or len(self.cloud_list) < 2:
            self._set_toast("Need 2 clouds for picking")
            return

        w, h = glfw.get_framebuffer_size(window)
        # Get window size for coordinate scaling (framebuffer may differ on HiDPI)
        win_w, win_h = glfw.get_window_size(window)
        scale_x = w / win_w
        scale_y = h / win_h
        # Convert to OpenGL viewport coords (Y flipped, scaled to framebuffer)
        gl_x = screen_x * scale_x
        gl_y = h - screen_y * scale_y

        # Read current matrices
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)

        # Unproject near and far points to get a ray
        near = gluUnProject(gl_x, gl_y, 0.0, modelview, projection, viewport)
        far = gluUnProject(gl_x, gl_y, 1.0, modelview, projection, viewport)

        ray_origin = np.array(near, dtype=np.float64)
        ray_dir = np.array(far, dtype=np.float64) - ray_origin
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # Search in the active pick cloud
        cloud_idx = self.pick_cloud
        pts = self.rotated_clouds[cloud_idx]
        if len(pts) == 0:
            return

        # Vectorized point-to-ray distance: ||(P - O) x D|| / ||D||
        diff = pts.astype(np.float64) - ray_origin
        cross = np.cross(diff, ray_dir)
        dists = np.linalg.norm(cross, axis=1)

        # Also compute distance along ray (to prefer closer points)
        along_ray = np.dot(diff, ray_dir)
        # Only consider points in front of camera
        mask = along_ray > 0
        if not mask.any():
            self._set_toast("No points in view")
            return

        dists[~mask] = np.inf
        best_idx = np.argmin(dists)
        best_dist = dists[best_idx]

        # Threshold: scale with move speed (proxy for scene scale)
        threshold = max(0.5, self.move_speed * 5.0)
        if best_dist > threshold:
            self._set_toast(f"No point near cursor (dist={best_dist:.2f})")
            return

        picked_pt = pts[best_idx].copy()
        self.picks[cloud_idx].append(picked_pt)

        cloud_name = self.cloud_names[cloud_idx] if cloud_idx < len(self.cloud_names) else f"Cloud {cloud_idx+1}"
        n_pairs = min(len(self.picks[0]), len(self.picks[1]))
        self._set_toast(f"Picked {cloud_name} #{len(self.picks[cloud_idx])} "
                        f"({n_pairs} pair{'s' if n_pairs != 1 else ''} ready)")
        print(f"  Pick: {cloud_name} [{picked_pt[0]:.2f}, {picked_pt[1]:.2f}, {picked_pt[2]:.2f}]")

        # Alternate to the other cloud
        self.pick_cloud = 1 - self.pick_cloud

    def _align_clouds(self):
        """Align cloud 2 onto cloud 1 using picked correspondences + ICP."""
        n_pairs = min(len(self.picks[0]), len(self.picks[1]))
        if n_pairs < 3:
            self._set_toast(f"Need 3+ pairs, have {n_pairs}")
            return

        self._set_toast("Aligning... (computing)")
        print("\n=== Alignment ===")

        # Get corresponding points (in rotated/display coordinates)
        src_pts = np.array(self.picks[1][:n_pairs], dtype=np.float64)  # cloud 2
        dst_pts = np.array(self.picks[0][:n_pairs], dtype=np.float64)  # cloud 1

        # Undo scene rotation to get original coordinates
        # rotated = original @ scene_rot.T  =>  original = rotated @ scene_rot
        src_orig = src_pts @ self.scene_rot
        dst_orig = dst_pts @ self.scene_rot

        # Kabsch: initial rigid transform from correspondences
        T_init = kabsch_transform(src_orig, dst_orig)
        print(f"  SVD initial guess from {n_pairs} pairs")

        # Apply initial transform to cloud 2 original points for ICP
        import open3d as o3d

        cloud2_orig = self.original_clouds[1].astype(np.float64)
        # Apply T_init: p' = R @ p + t
        cloud2_transformed = (T_init[:3, :3] @ cloud2_orig.T).T + T_init[:3, 3]

        # Build Open3D point clouds for ICP refinement
        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(cloud2_transformed)
        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(self.original_clouds[0].astype(np.float64))

        # ICP refinement (using run_icp from icp_merge)
        from icp_merge import run_icp
        voxel_size = 0.10
        result = run_icp(pcd_src, pcd_tgt, voxel_size)

        # Compose: T_final = T_icp @ T_init
        T_final = result.transformation @ T_init
        fitness = result.fitness
        rmse = result.inlier_rmse

        print(f"  ICP fitness={fitness:.3f}  RMSE={rmse:.4f}m")

        # Apply final transform to original cloud 2
        self.original_clouds[1] = ((T_final[:3, :3] @ cloud2_orig.T).T + T_final[:3, 3]).astype(np.float32)

        # Re-apply scene rotation and refresh
        self._apply_scene_rotation()
        self._build_colors()

        # Clear picks
        self.picks = [[], []]
        self.pick_cloud = 0
        self.aligned = True

        self._set_toast(f"Aligned! fitness={fitness:.3f} RMSE={rmse:.4f}m  [M to merge]")
        print(f"  Alignment complete\n")

    def _merge_clouds(self, filename="merged"):
        """Merge all clouds into one and save to <filename>.pcd."""
        import open3d as o3d

        # Combine all original clouds
        all_pts = np.vstack([c.astype(np.float64) for c in self.original_clouds])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)

        # Downsample slightly to clean overlapping regions
        pcd_down = pcd.voxel_down_sample(0.02)
        n_final = len(pcd_down.points)

        # Ensure .pcd extension
        if not filename.lower().endswith('.pcd'):
            filename += '.pcd'
        out_path = filename
        o3d.io.write_point_cloud(out_path, pcd_down)

        # Replace viewer state with single merged cloud
        merged_pts = np.asarray(pcd_down.points, dtype=np.float32)
        self.original_clouds = [merged_pts]
        self.cloud_list = [merged_pts]
        self.cloud_names = ["merged"]
        self.cloud_visible = [True]
        self.compare_mode = False

        self._apply_scene_rotation()

        self.color_names = ['height', 'distance', 'white']
        self.color_mode = 0
        self._build_colors()

        # Rebuild VBOs
        for vbo in self.vbo_pos_list + self.vbo_col_list:
            glDeleteBuffers(1, [vbo])
        self.vbo_pos_list = []
        self.vbo_col_list = []
        offset = 0
        for i, cloud in enumerate(self.rotated_clouds):
            vbo_p = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_p)
            glBufferData(GL_ARRAY_BUFFER, cloud.nbytes, cloud, GL_STATIC_DRAW)
            self.vbo_pos_list.append(vbo_p)

            n = len(cloud)
            cloud_colors = self.colors[offset:offset + n]
            vbo_c = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_c)
            glBufferData(GL_ARRAY_BUFFER, cloud_colors.nbytes, cloud_colors, GL_STATIC_DRAW)
            self.vbo_col_list.append(vbo_c)
            offset += n

        self.n_points = len(merged_pts)
        self._points_dirty = False
        self._colors_dirty = False
        self.picks = [[], []]
        self.pick_mode = False
        self.aligned = False

        self._set_toast(f"Merged {n_final:,} points -> {out_path}")
        print(f"  Merged {n_final:,} points -> {out_path}")

    def _filter_outliers(self):
        """Statistical outlier removal on all clouds."""
        import open3d as o3d

        self._set_toast("Filtering outliers...")
        total_before = sum(len(c) for c in self.original_clouds)
        total_after = 0

        for i in range(len(self.original_clouds)):
            pts = self.original_clouds[i].astype(np.float64)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)

            # Statistical outlier removal: 20 neighbors, 2.0 std ratio
            pcd_clean, inlier_idx = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0)

            cleaned = np.asarray(pcd_clean.points, dtype=np.float32)
            self.original_clouds[i] = cleaned
            total_after += len(cleaned)

            name = self.cloud_names[i] if i < len(self.cloud_names) else f"Cloud {i+1}"
            removed = len(pts) - len(cleaned)
            print(f"  {name}: {len(pts):,} -> {len(cleaned):,} ({removed:,} removed)")

        # Update viewer state
        self.cloud_list = [c.copy() for c in self.original_clouds]
        self._apply_scene_rotation()
        self._build_colors()

        # Rebuild VBOs
        for vbo in self.vbo_pos_list + self.vbo_col_list:
            glDeleteBuffers(1, [vbo])
        self.vbo_pos_list = []
        self.vbo_col_list = []
        offset = 0
        for i, cloud in enumerate(self.rotated_clouds):
            vbo_p = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_p)
            glBufferData(GL_ARRAY_BUFFER, cloud.nbytes, cloud, GL_STATIC_DRAW)
            self.vbo_pos_list.append(vbo_p)

            n = len(cloud)
            cloud_colors = self.colors[offset:offset + n]
            vbo_c = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_c)
            glBufferData(GL_ARRAY_BUFFER, cloud_colors.nbytes, cloud_colors, GL_STATIC_DRAW)
            self.vbo_col_list.append(vbo_c)
            offset += n

        self.n_points = total_after
        self._points_dirty = False
        self._colors_dirty = False

        removed = total_before - total_after
        self._set_toast(f"Filtered: {total_before:,} -> {total_after:,} ({removed:,} outliers removed)")
        print(f"  Total: {total_before:,} -> {total_after:,} ({removed:,} removed)")

    def _save_overwrite(self):
        """Save current cloud(s) back to their original file(s), overwriting."""
        import open3d as o3d

        saved = []
        for i in range(len(self.original_clouds)):
            path = self.cloud_paths[i] if i < len(self.cloud_paths) else None
            if path is None:
                continue

            pts = self.original_clouds[i].astype(np.float64)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)

            o3d.io.write_point_cloud(path, pcd)
            name = self.cloud_names[i] if i < len(self.cloud_names) else f"Cloud {i+1}"
            saved.append(f"{name} ({len(pts):,} pts)")
            print(f"  Saved {path} ({len(pts):,} points)")

        if saved:
            self._set_toast(f"Saved: {', '.join(saved)}")
        else:
            self._set_toast("No file paths to save to")

    def _draw_text(self, x, y, text, r=1.0, g=1.0, b=1.0):
        """Draw bitmap text at pixel coordinates (x, y from bottom-left)."""
        if not _GLUT_AVAILABLE:
            return
        glColor3f(r, g, b)
        glRasterPos2f(x, y)
        for ch in text:
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(ch))

    def _draw_hud(self, w, h):
        """Draw crosshair, axis indicator, help overlay, and status bar."""
        # Switch to 2D pixel-coordinate overlay
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, w, 0, h, -1, 1)  # origin bottom-left, pixel coords
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        # Crosshair (center of screen) - yellow in pick mode
        cx, cy = w / 2, h / 2
        size = 10
        if self.pick_mode:
            glColor3f(1.0, 1.0, 0.0)
            size = 15
        else:
            glColor3f(1.0, 1.0, 1.0)
        glLineWidth(2.0 if self.pick_mode else 1.0)
        glBegin(GL_LINES)
        glVertex2f(cx - size, cy)
        glVertex2f(cx + size, cy)
        glVertex2f(cx, cy - size)
        glVertex2f(cx, cy + size)
        glEnd()
        glLineWidth(1.0)

        line_h = 18  # line height in pixels

        # Status bar at bottom
        edl_tag = "EDL" if self.edl_enabled else "edl"
        status = (f"Speed: {self.move_speed:.2f}  |  "
                  f"Pt Size: {self.point_size:.1f}  |  "
                  f"Color: {self.color_names[self.color_mode]}  |  "
                  f"{edl_tag}  |  "
                  f"Points: {self.n_points:,}")
        if self.compare_mode:
            vis_parts = []
            for i, name in enumerate(self.cloud_names):
                tag = "ON" if self.cloud_visible[i] else "off"
                vis_parts.append(f"{name}:{tag}")
            status += "  |  " + "  ".join(vis_parts)
        status += "  |  [H] Help"
        self._draw_text(10, 10, status, 0.7, 0.7, 0.7)

        # Pick mode status line
        if self.pick_mode:
            n_pairs = min(len(self.picks[0]), len(self.picks[1]))
            cloud_name = self.cloud_names[self.pick_cloud] if self.pick_cloud < len(self.cloud_names) else f"Cloud {self.pick_cloud+1}"
            pick_status = (f"PICK MODE: Click to pick {cloud_name}  |  "
                           f"Pairs: {n_pairs}/3+  |  "
                           f"Cloud 1: {len(self.picks[0])} picks  "
                           f"Cloud 2: {len(self.picks[1])} picks  |  "
                           f"[G] Align  [U] Undo  [T] Exit pick mode")
            self._draw_text(10, 30 + 18, pick_status, 1.0, 1.0, 0.3)
        elif self.grab_mode:
            grab_status = (f"GRAB MODE: Drag to move  |  "
                           f"Shift+drag to rotate  |  "
                           f"Scroll fwd/back  |  "
                           f"[I] Run ICP  [Tab] Exit")
            self._draw_text(10, 30 + 18, grab_status, 0.3, 1.0, 0.5)

        # Toast message (fades after 2 seconds)
        now = glfw.get_time()
        if self.toast_msg and (now - self.toast_time) < 2.0:
            alpha = min(1.0, 2.0 - (now - self.toast_time))
            self._draw_text(10, 30, self.toast_msg, 1.0 * alpha, 1.0 * alpha, 0.3 * alpha)

        # Text input bar
        if self.text_input_active:
            # Draw background bar
            bar_h = 28
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(0.1, 0.1, 0.3, 0.9)
            glBegin(GL_QUADS)
            glVertex2f(0, 55)
            glVertex2f(w, 55)
            glVertex2f(w, 55 + bar_h)
            glVertex2f(0, 55 + bar_h)
            glEnd()
            glDisable(GL_BLEND)
            # Blinking cursor
            cursor = "|" if int(now * 2) % 2 == 0 else " "
            input_text = f"{self.text_input_prompt}{self.text_input_buffer}{cursor}    [Enter] Confirm  [Esc] Cancel"
            self._draw_text(10, 62, input_text, 1.0, 1.0, 1.0)

        # Help overlay
        if self.show_help:
            # Semi-transparent background
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            panel_x = 10
            panel_w = 380
            panel_lines = 30
            if self.compare_mode:
                panel_lines += 2 + len(self.cloud_list)
                panel_lines += 8  # grab align section
                panel_lines += 8  # pick align section
            panel_h = panel_lines * line_h + 20
            panel_y = h - panel_h - 10

            glColor4f(0.0, 0.0, 0.0, 0.7)
            glBegin(GL_QUADS)
            glVertex2f(panel_x, panel_y)
            glVertex2f(panel_x + panel_w, panel_y)
            glVertex2f(panel_x + panel_w, panel_y + panel_h)
            glVertex2f(panel_x, panel_y + panel_h)
            glEnd()
            glDisable(GL_BLEND)

            # Draw help text (top-down, so start from top of panel)
            tx = panel_x + 10
            ty = panel_y + panel_h - 25

            def hline(text, r=0.9, g=0.9, b=0.9):
                nonlocal ty
                self._draw_text(tx, ty, text, r, g, b)
                ty -= line_h

            hline("=== Controls ===", 0.3, 0.8, 1.0)
            hline("")
            hline("Movement:", 1.0, 0.8, 0.3)
            hline("  W/S           Forward / Back")
            hline("  A/D           Strafe Left / Right")
            hline("  Space/LCtrl   Up / Down")
            hline("  Scroll        Adjust Speed")
            hline("  1-5           Speed Presets")
            hline("")
            hline("Camera:", 1.0, 0.8, 0.3)
            hline("  Left-click    Hold + Drag to Look")
            hline("  Mid/Right     Hold + Drag to Pan")
            hline("")
            hline("Display:", 1.0, 0.8, 0.3)
            hline("  +/-           Point Size")
            hline("  C             Cycle Color Mode")
            edl_state = "ON" if self.edl_enabled else "OFF"
            hline(f"  E             Toggle EDL ({edl_state})")
            hline(f"  L / Shift+L   EDL Strength ({self.edl_strength:.1f})")
            hline("  R             Reset Position")
            hline("  P             Print Position")
            hline("")
            hline("Scene Rotation:", 1.0, 0.8, 0.3)
            hline("  F5/F6         Rotate +/-90  X")
            hline("  F7/F8         Rotate +/-90  Y")
            hline("  F9/F10        Rotate +/-90  Z")
            hline("  F1            Reset Rotation")
            if self.compare_mode:
                hline("")
                hline("Compare Mode:", 1.0, 0.8, 0.3)
                color_labels = ['BLUE', 'ORANGE', 'GREEN', 'MAGENTA']
                for i, name in enumerate(self.cloud_names):
                    vis = "ON" if self.cloud_visible[i] else "OFF"
                    cl = color_labels[i % len(color_labels)]
                    hline(f"  F{i+2}  [{cl}] {name}: {vis}")
                hline("")
                hline("Grab Align:", 1.0, 0.8, 0.3)
                hline("  Tab           Toggle Grab Mode")
                hline("  Drag          Move cloud 2")
                hline("  Shift+drag    Rotate cloud 2")
                hline("  Scroll        Push/pull cloud 2")
                hline("  I             Run ICP")
                hline("")
                hline("Pick Align:", 1.0, 0.8, 0.3)
                hline("  T             Toggle Pick Mode")
                hline("  Left-click    Pick Point (alternates)")
                hline("  U             Undo Last Pick")
                hline("  G             Align (3+ pairs)")
                hline("  M             Merge & Save")
            hline("")
            hline("Tools:", 1.0, 0.8, 0.3)
            hline("  X             Filter Outliers")
            hline("  F12           Save (Overwrite PCD)")
            hline("")
            hline("  H  Hide Help  |  Esc  Quit", 0.5, 0.5, 0.5)

        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        # Small 3D axis indicator in bottom-right corner
        axis_size = 100
        margin = 10
        glViewport(w - axis_size - margin, margin, axis_size, axis_size)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective(45, 1.0, 0.1, 10.0)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        forward = self._get_forward()
        cam = forward * 3.0
        up = np.array([0.0, 0.0, 1.0])
        gluLookAt(cam[0], cam[1], cam[2], 0, 0, 0, up[0], up[1], up[2])

        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # X - Red
        glColor3f(1.0, 0.2, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)
        # Y - Green
        glColor3f(0.2, 1.0, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)
        # Z - Blue
        glColor3f(0.4, 0.4, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glLineWidth(1.0)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        # Restore full viewport
        glViewport(0, 0, w, h)

    def _mouse_button_cb(self, window, button, action, mods):
        mx, my = glfw.get_cursor_pos(window)
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                if self.grab_mode:
                    shift = (mods & glfw.MOD_SHIFT) != 0
                    if shift:
                        self.mouse_grab_rotate = True
                    else:
                        self.mouse_grab = True
                    self.last_mx, self.last_my = mx, my
                elif self.pick_mode:
                    self._pick_point(window, mx, my)
                else:
                    self.mouse_look = True
                    self.last_mx, self.last_my = mx, my
            elif action == glfw.RELEASE:
                self.mouse_look = False
                self.mouse_grab = False
                self.mouse_grab_rotate = False
        elif button in (glfw.MOUSE_BUTTON_MIDDLE, glfw.MOUSE_BUTTON_RIGHT):
            if action == glfw.PRESS:
                self.mouse_pan = True
                self.last_mx, self.last_my = mx, my
            elif action == glfw.RELEASE:
                self.mouse_pan = False

    def _cursor_pos_cb(self, window, xpos, ypos):
        dx = xpos - self.last_mx
        dy = ypos - self.last_my
        self.last_mx = xpos
        self.last_my = ypos

        if self.mouse_grab:
            # Translate cloud 2 in the view plane
            right = self._get_right()
            cam_up = np.cross(self._get_forward(), right)
            cam_up = cam_up / np.linalg.norm(cam_up)
            pan_speed = self.move_speed * 0.01

            # Undo scene rotation to get world-space direction
            inv_rot = self.scene_rot.T
            world_right = inv_rot @ right
            world_up = inv_rot @ cam_up

            offset = -world_right * dx * pan_speed + world_up * dy * pan_speed
            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = offset
            self.grab_transform = T @ self.grab_transform
            self._apply_grab_transform()
        elif self.mouse_grab_rotate:
            # Rotate cloud 2 around its centroid
            rot_speed = 0.005
            yaw_angle = -dx * rot_speed
            pitch_angle = -dy * rot_speed

            # Rotation axes in world space (undo scene rotation)
            inv_rot = self.scene_rot.T
            cam_up_world = inv_rot @ np.array([0.0, 0.0, 1.0])
            cam_right_world = inv_rot @ self._get_right()

            # Rotation matrices
            cy, sy = np.cos(yaw_angle), np.sin(yaw_angle)
            cp, sp = np.cos(pitch_angle), np.sin(pitch_angle)

            # Rodrigues rotation around cam_up_world (yaw)
            K = np.array([[0, -cam_up_world[2], cam_up_world[1]],
                          [cam_up_world[2], 0, -cam_up_world[0]],
                          [-cam_up_world[1], cam_up_world[0], 0]])
            R_yaw = np.eye(3) + sy * K + (1 - cy) * (K @ K)

            # Rodrigues rotation around cam_right_world (pitch)
            K = np.array([[0, -cam_right_world[2], cam_right_world[1]],
                          [cam_right_world[2], 0, -cam_right_world[0]],
                          [-cam_right_world[1], cam_right_world[0], 0]])
            R_pitch = np.eye(3) + sp * K + (1 - cp) * (K @ K)

            R = R_pitch @ R_yaw

            # Rotate around cloud centroid
            centroid = np.mean(self.grab_cloud_backup.astype(np.float64), axis=0)
            # Apply current transform to centroid to get current center
            c = self.grab_transform[:3, :3] @ centroid + self.grab_transform[:3, 3]

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = c - R @ c
            self.grab_transform = T @ self.grab_transform
            self._apply_grab_transform()
        elif self.mouse_look:
            self.yaw -= dx * self.mouse_sensitivity
            self.pitch -= dy * self.mouse_sensitivity
            # Clamp pitch to avoid flipping (just under ±90°)
            self.pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, self.pitch))
        elif self.mouse_pan:
            right = self._get_right()
            cam_up = np.cross(self._get_forward(), right)
            cam_up = cam_up / np.linalg.norm(cam_up)
            pan_speed = self.move_speed * 0.01
            self.cam_pos -= right * dx * pan_speed
            self.cam_pos += cam_up * dy * pan_speed

    def _scroll_cb(self, window, xoff, yoff):
        if self.grab_mode:
            # Push/pull cloud 2 along camera forward direction
            inv_rot = self.scene_rot.T
            world_fwd = inv_rot @ self._get_forward()
            dist = self.move_speed * 0.5 * (1 if yoff > 0 else -1)
            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = world_fwd * dist
            self.grab_transform = T @ self.grab_transform
            self._apply_grab_transform()
            return
        factor = 1.2 if yoff > 0 else 0.8
        self.move_speed = max(0.01, min(10.0, self.move_speed * factor))

    def _rotate_scene(self, axis, angle_deg):
        """Rotate the scene by angle_deg around the given axis ('x', 'y', or 'z')."""
        angle = math.radians(angle_deg)
        if axis == 'x':
            R = rot_x(angle)
        elif axis == 'y':
            R = rot_y(angle)
        else:
            R = rot_z(angle)
        self.scene_rot = R @ self.scene_rot
        self._apply_scene_rotation()
        self._build_colors()
        self.yaw = 0.0
        self.pitch = 0.0
        self._set_toast(f"Scene rotated {angle_deg:+d} deg around {axis.upper()}")

    def _char_cb(self, window, codepoint):
        """Handle character input for text entry mode."""
        if not self.text_input_active:
            return
        ch = chr(codepoint)
        # Allow alphanumeric, underscore, hyphen, dot, spaces
        if ch.isprintable():
            self.text_input_buffer += ch

    def _key_cb(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return

        # Text input mode intercepts keys
        if self.text_input_active:
            if key == glfw.KEY_ENTER:
                # Confirm input
                cb = self.text_input_callback
                buf = self.text_input_buffer.strip()
                self.text_input_active = False
                self.text_input_buffer = ""
                self.text_input_callback = None
                if cb and buf:
                    cb(buf)
            elif key == glfw.KEY_ESCAPE:
                # Cancel input
                self.text_input_active = False
                self.text_input_buffer = ""
                self.text_input_callback = None
                self._set_toast("Merge cancelled")
            elif key == glfw.KEY_BACKSPACE:
                self.text_input_buffer = self.text_input_buffer[:-1]
            return

        if key == glfw.KEY_ESCAPE:
            if self.grab_mode:
                # Revert cloud 2 to pre-grab state
                self.original_clouds[1] = self.grab_cloud_backup.copy()
                self.grab_mode = False
                self.grab_cloud_backup = None
                self.rotated_clouds = []
                for c in self.original_clouds:
                    rotated = (c.astype(np.float64) @ self.scene_rot.T).astype(np.float32)
                    self.rotated_clouds.append(rotated)
                self.points = np.vstack(self.rotated_clouds)
                self._points_dirty = True
                self._build_colors()
                self._set_toast("Grab mode cancelled — cloud reverted")
                return
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_H:
            self.show_help = not self.show_help
        elif key == glfw.KEY_C:
            self.color_mode = (self.color_mode + 1) % len(self.color_names)
            self._build_colors()
            self._set_toast(f"Color: {self.color_names[self.color_mode]}")
        elif key == glfw.KEY_R:
            centroid = np.mean(self.points, axis=0)
            self.cam_pos = centroid.copy().astype(np.float64)
            self.yaw = 0.0
            self.pitch = 0.0
            self._set_toast("Reset to centroid")
        elif key == glfw.KEY_P:
            msg = (f"Pos: [{self.cam_pos[0]:.2f}, {self.cam_pos[1]:.2f}, {self.cam_pos[2]:.2f}]  "
                   f"Yaw: {math.degrees(self.yaw):.1f}  Pitch: {math.degrees(self.pitch):.1f}")
            self._set_toast(msg)
            print(msg)
        elif key == glfw.KEY_EQUAL:
            step = 0.1 if self.point_size < 1.0 else 0.5
            self.point_size = min(self.point_size + step, 20.0)
            self._set_toast(f"Point size: {self.point_size:.1f}")
        elif key == glfw.KEY_MINUS:
            step = 0.1 if self.point_size <= 1.0 else 0.5
            self.point_size = max(self.point_size - step, 0.1)
            self._set_toast(f"Point size: {self.point_size:.1f}")
        elif key == glfw.KEY_1:
            self.move_speed = 0.05; self._set_toast(f"Speed: {self.move_speed:.2f}")
        elif key == glfw.KEY_2:
            self.move_speed = 0.15; self._set_toast(f"Speed: {self.move_speed:.2f}")
        elif key == glfw.KEY_3:
            self.move_speed = 0.3; self._set_toast(f"Speed: {self.move_speed:.2f}")
        elif key == glfw.KEY_4:
            self.move_speed = 0.6; self._set_toast(f"Speed: {self.move_speed:.2f}")
        elif key == glfw.KEY_5:
            self.move_speed = 1.5; self._set_toast(f"Speed: {self.move_speed:.2f}")
        # Scene rotation keys
        elif key == glfw.KEY_F5:
            self._rotate_scene('x', 90)
        elif key == glfw.KEY_F6:
            self._rotate_scene('x', -90)
        elif key == glfw.KEY_F7:
            self._rotate_scene('y', 90)
        elif key == glfw.KEY_F8:
            self._rotate_scene('y', -90)
        elif key == glfw.KEY_F9:
            self._rotate_scene('z', 90)
        elif key == glfw.KEY_F10:
            self._rotate_scene('z', -90)
        elif key == glfw.KEY_F1:
            self.scene_rot = np.eye(3, dtype=np.float64)
            self._apply_scene_rotation()
            self._build_colors()
            self.yaw = 0.0
            self.pitch = 0.0
            self._set_toast("Scene rotation reset")
        # Grab mode keys
        elif key == glfw.KEY_TAB:
            if self.compare_mode and len(self.cloud_list) >= 2:
                self.grab_mode = not self.grab_mode
                if self.grab_mode:
                    self.pick_mode = False  # exit pick mode
                    self.grab_transform = np.eye(4, dtype=np.float64)
                    self.grab_cloud_backup = self.original_clouds[1].copy()
                    self._set_toast("Grab mode ON — drag to move, Shift+drag to rotate, I for ICP")
                else:
                    self._set_toast("Grab mode OFF")
            else:
                self._set_toast("Need 2 clouds for grab mode")
        elif key == glfw.KEY_I:
            if self.grab_mode:
                self._set_toast("Running ICP...")
                print("\n=== Grab Mode ICP ===")
                import open3d as o3d
                from icp_merge import run_icp

                cloud2 = self.original_clouds[1].astype(np.float64)
                cloud1 = self.original_clouds[0].astype(np.float64)

                pcd_src = o3d.geometry.PointCloud()
                pcd_src.points = o3d.utility.Vector3dVector(cloud2)
                pcd_tgt = o3d.geometry.PointCloud()
                pcd_tgt.points = o3d.utility.Vector3dVector(cloud1)

                result = run_icp(pcd_src, pcd_tgt, 0.10)
                fitness = result.fitness
                rmse = result.inlier_rmse
                print(f"  ICP fitness={fitness:.3f}  RMSE={rmse:.4f}m")

                # Apply ICP refinement to cloud 2
                T_icp = result.transformation
                cloud2_aligned = (T_icp[:3, :3] @ cloud2.T).T + T_icp[:3, 3]
                self.original_clouds[1] = cloud2_aligned.astype(np.float32)

                # Exit grab mode
                self.grab_mode = False
                self.grab_cloud_backup = None

                # Refresh display
                self.rotated_clouds = []
                for c in self.original_clouds:
                    rotated = (c.astype(np.float64) @ self.scene_rot.T).astype(np.float32)
                    self.rotated_clouds.append(rotated)
                self.points = np.vstack(self.rotated_clouds)
                self.n_points = len(self.points)
                self._points_dirty = True
                self._build_colors()
                self.aligned = True

                self._set_toast(f"ICP done! fitness={fitness:.3f} RMSE={rmse:.4f}m  [M to merge]")
                print(f"  Alignment complete\n")
        # Stitching keys
        elif key == glfw.KEY_T:
            if self.compare_mode and len(self.cloud_list) >= 2:
                self.pick_mode = not self.pick_mode
                if self.pick_mode:
                    self._set_toast("Pick mode ON — click to pick corresponding points")
                else:
                    self._set_toast("Pick mode OFF")
            else:
                self._set_toast("Need 2 clouds for stitching")
        elif key == glfw.KEY_G:
            if self.compare_mode:
                self._align_clouds()
            else:
                self._set_toast("Need 2 clouds to align")
        elif key == glfw.KEY_M:
            if len(self.cloud_list) >= 2:
                self.text_input_active = True
                self.text_input_buffer = "merged"
                self.text_input_prompt = "Save as: "
                self.text_input_callback = self._merge_clouds
            else:
                self._set_toast("Need 2+ clouds to merge")
        elif key == glfw.KEY_E:
            if self._edl_shader is not None and self._fbo is not None:
                self.edl_enabled = not self.edl_enabled
                self._set_toast(f"EDL: {'ON' if self.edl_enabled else 'OFF'}")
            else:
                self._set_toast("EDL not available (shader failed)")
        elif key == glfw.KEY_L:
            # Adjust EDL strength
            if mods & glfw.MOD_SHIFT:
                self.edl_strength = max(0.1, self.edl_strength - 0.2)
            else:
                self.edl_strength = min(5.0, self.edl_strength + 0.2)
            self._set_toast(f"EDL strength: {self.edl_strength:.1f}")
        elif key == glfw.KEY_U:
            # Undo last pick
            if self.pick_mode:
                # Undo goes back to the previous cloud
                prev_cloud = 1 - self.pick_cloud
                if len(self.picks[prev_cloud]) > 0:
                    removed = self.picks[prev_cloud].pop()
                    self.pick_cloud = prev_cloud
                    self._set_toast(f"Undid pick from {self.cloud_names[prev_cloud]}")
                else:
                    self._set_toast("No picks to undo")
        # Cloud visibility toggles (compare mode)
        elif key == glfw.KEY_F2 and len(self.cloud_list) >= 1:
            self.cloud_visible[0] = not self.cloud_visible[0]
            state = "ON" if self.cloud_visible[0] else "OFF"
            self._set_toast(f"{self.cloud_names[0]}: {state}")
        elif key == glfw.KEY_F3 and len(self.cloud_list) >= 2:
            self.cloud_visible[1] = not self.cloud_visible[1]
            state = "ON" if self.cloud_visible[1] else "OFF"
            self._set_toast(f"{self.cloud_names[1]}: {state}")
        elif key == glfw.KEY_F4 and len(self.cloud_list) >= 3:
            self.cloud_visible[2] = not self.cloud_visible[2]
            state = "ON" if self.cloud_visible[2] else "OFF"
            self._set_toast(f"{self.cloud_names[2]}: {state}")
        # Filter and save
        elif key == glfw.KEY_X:
            print("\n=== Statistical Outlier Filter ===")
            self._filter_outliers()
        elif key == glfw.KEY_F12:
            print("\n=== Save (Overwrite) ===")
            self._save_overwrite()

    def _resize_cb(self, window, w, h):
        self.width = w
        self.height = h
        # FBO will be resized on next render via _init_fbo check


def main():
    if len(sys.argv) < 2:
        print("Usage: python flythrough.py <file1.pcd> [file2.pcd] [file3.pcd ...]")
        print("  Multiple files: compare mode with distinct colors")
        print("  F2/F3/F4 to toggle each cloud's visibility")
        sys.exit(1)

    input_files = [f for f in sys.argv[1:] if not f.startswith('-')]
    clouds = []
    names = []
    for f in input_files:
        pts = load_points(f)
        if len(pts) == 0:
            print(f"WARNING: No points loaded from {f}, skipping.")
            continue
        clouds.append(pts)
        names.append(Path(f).stem)

    if len(clouds) == 0:
        print("ERROR: No points loaded from any file.")
        sys.exit(1)

    if len(clouds) > 1:
        print(f"\nCompare mode: {len(clouds)} clouds")
        color_labels = ['BLUE', 'ORANGE', 'GREEN', 'MAGENTA']
        for i, name in enumerate(names):
            print(f"  F{i+2}: [{color_labels[i % len(color_labels)]}] {name}")

    viewer = FPSViewer(clouds, names, cloud_paths=input_files[:len(clouds)])
    viewer.run()


if __name__ == '__main__':
    main()
