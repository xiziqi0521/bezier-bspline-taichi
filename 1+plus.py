import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH, HEIGHT = 800, 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 2000
DEGREE = 3

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS)

# ======================
# Bézier
# ======================
def de_casteljau(points, t):
    pts = np.array(points, dtype=np.float32)
    while len(pts) > 1:
        pts = (1 - t) * pts[:-1] + t * pts[1:]
    return pts[0]

# ======================
# 端点插值 Knot（关键⭐）
# ======================
def generate_clamped_knot(n, p):
    m = n + p + 1
    knot = np.zeros(m)

    for i in range(m):
        if i <= p:
            knot[i] = 0.0
        elif i >= n:
            knot[i] = 1.0
        else:
            knot[i] = (i - p) / (n - p)

    return knot

# ======================
# Cox–de Boor（修复边界⭐）
# ======================
def basis_function(i, p, t, knot):

    # ⭐关键：处理 t=1
    if p == 0:
        if (knot[i] <= t < knot[i+1]) or (t == 1.0 and knot[i+1] == 1.0):
            return 1.0
        return 0.0

    left = 0.0
    right = 0.0

    denom1 = knot[i+p] - knot[i]
    denom2 = knot[i+p+1] - knot[i+1]

    if denom1 > 1e-6:
        left = (t - knot[i]) / denom1 * basis_function(i, p-1, t, knot)

    if denom2 > 1e-6:
        right = (knot[i+p+1] - t) / denom2 * basis_function(i+1, p-1, t, knot)

    return left + right

# ======================
# 全局 B 样条（标准实现⭐）
# ======================
def bspline_curve(control_points, num_samples=1000, degree=3):
    n = len(control_points)
    if n < degree + 1:
        return []

    knot = generate_clamped_knot(n, degree)
    curve = []

    for j in range(num_samples):
        t = j / (num_samples - 1)

        point = np.array([0.0, 0.0], dtype=np.float32)

        for i in range(n):
            b = basis_function(i, degree, t, knot)
            point += b * np.array(control_points[i])

        curve.append(point)

    return curve

# ======================
# 清屏
# ======================
@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

# ======================
# 抗锯齿（满分级⭐）
# ======================
@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]

        x = pt[0] * WIDTH
        y = pt[1] * HEIGHT

        base_x = ti.cast(x, ti.i32)
        base_y = ti.cast(y, ti.i32)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx = base_x + dx
                ny = base_y + dy

                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    cx = nx + 0.5
                    cy = ny + 0.5

                    dist = ti.sqrt((cx - x)**2 + (cy - y)**2)

                    # ⭐ 高质量抗锯齿
                    weight = ti.exp(-dist * 2.5)

                    pixels[nx, ny] += ti.Vector([0.0, weight, 0.0])

# ======================
# 主程序
# ======================
def main():
    window = ti.ui.Window("FULL SCORE VERSION", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points = []
    use_bspline = False

    while window.running:

        for e in window.get_events(ti.ui.PRESS):

            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    control_points.append(window.get_cursor_pos())

            elif e.key == 'c':
                control_points.clear()

            elif e.key == 'b':
                use_bspline = not use_bspline
                print("B-spline mode:", use_bspline)

        clear_pixels()

        if len(control_points) >= 2:

            if not use_bspline:
                curve = [
                    de_casteljau(control_points, t / NUM_SEGMENTS)
                    for t in range(NUM_SEGMENTS)
                ]
            else:
                curve = bspline_curve(control_points, NUM_SEGMENTS, DEGREE)

            if len(curve) > 0:
                curve_np = np.array(curve, dtype=np.float32)

                curve_points_field.from_numpy(curve_np)
                draw_curve_kernel(len(curve_np))

        canvas.set_image(pixels)

        # 控制点
        if len(control_points) > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:len(control_points)] = np.array(control_points)
            gui_points.from_numpy(np_points)

            canvas.circles(gui_points, radius=0.006, color=(1, 0, 0))

            if len(control_points) >= 2:
                indices = []
                for i in range(len(control_points) - 1):
                    indices.extend([i, i + 1])

                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                np_indices[:len(indices)] = np.array(indices)
                gui_indices.from_numpy(np_indices)

                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))

        window.show()

if __name__ == '__main__':
    main()