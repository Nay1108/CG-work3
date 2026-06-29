import taichi as ti
import numpy as np

# 使用 GPU 后端
ti.init(arch=ti.gpu)

# 常量定义
WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000

# --- GPU 缓冲区 ---
# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# GUI 绘制数据缓冲池
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# 曲线坐标缓冲区
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)


# ==================== 数学算法实现 ====================

def de_casteljau(points, t):
    """
    贝塞尔曲线：De Casteljau 算法
    points: 控制点列表
    t: 参数 [0, 1]
    """
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)


def bspline_3(points, t):
    """
    均匀三次 B 样条曲线
    points: 4个控制点 [P0, P1, P2, P3]
    t: 参数 [0, 1]
    """
    if len(points) != 4:
        raise ValueError("B-spline segment requires exactly 4 points")
    
    p0, p1, p2, p3 = points[0], points[1], points[2], points[3]
    t2 = t * t
    t3 = t2 * t
    
    # 使用均匀三次 B 样条基矩阵
    x = (1.0 / 6.0) * (t3 * (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) +
                       t2 * (3 * p0[0] - 6 * p1[0] + 3 * p2[0]) +
                       t * (-3 * p0[0] + 3 * p2[0]) +
                       (p0[0] + 4 * p1[0] + p2[0]))
    
    y = (1.0 / 6.0) * (t3 * (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) +
                       t2 * (3 * p0[1] - 6 * p1[1] + 3 * p2[1]) +
                       t * (-3 * p0[1] + 3 * p2[1]) +
                       (p0[1] + 4 * p1[1] + p2[1]))
    
    return [x, y]


def compute_bezier_curve(control_points, num_segments):
    """计算贝塞尔曲线的所有采样点"""
    points = np.zeros((num_segments + 1, 2), dtype=np.float32)
    for i in range(num_segments + 1):
        t = i / num_segments
        points[i] = de_casteljau(control_points, t)
    return points


def compute_bspline_curve(control_points, num_segments):
    """
    计算均匀三次 B 样条曲线的所有采样点
    需要至少 4 个控制点
    """
    n = len(control_points)
    if n < 4:
        return np.zeros((num_segments + 1, 2), dtype=np.float32)
    
    # 总共有 n-3 段
    segments = n - 3
    points_per_segment = num_segments // segments + 1
    
    all_points = []
    
    for seg in range(segments):
        # 取 4 个控制点
        p0 = control_points[seg]
        p1 = control_points[seg + 1]
        p2 = control_points[seg + 2]
        p3 = control_points[seg + 3]
        segment_pts = [p0, p1, p2, p3]
        
        # 采样该段
        for i in range(points_per_segment):
            t = i / points_per_segment
            if t <= 1.0:
                all_points.append(bspline_3(segment_pts, t))
    
    # 确保包含最后一个点
    if len(all_points) > 0:
        last_seg = segments - 1
        p0 = control_points[last_seg]
        p1 = control_points[last_seg + 1]
        p2 = control_points[last_seg + 2]
        p3 = control_points[last_seg + 3]
        all_points.append(bspline_3([p0, p1, p2, p3], 1.0))
    
    result = np.array(all_points, dtype=np.float32)
    
    # 插值到固定数量
    if len(result) < num_segments + 1:
        final = np.zeros((num_segments + 1, 2), dtype=np.float32)
        for i in range(num_segments + 1):
            idx = i * (len(result) - 1) / num_segments
            idx0 = int(idx)
            idx1 = min(idx0 + 1, len(result) - 1)
            frac = idx - idx0
            final[i] = result[idx0] * (1 - frac) + result[idx1] * frac
        return final
    
    # 如果点数太多，均匀采样
    if len(result) > num_segments + 1:
        final = np.zeros((num_segments + 1, 2), dtype=np.float32)
        for i in range(num_segments + 1):
            idx = i * (len(result) - 1) / num_segments
            idx0 = int(idx)
            idx1 = min(idx0 + 1, len(result) - 1)
            frac = idx - idx0
            final[i] = result[idx0] * (1 - frac) + result[idx1] * frac
        return final
    
    return result


# ==================== GPU 内核 ====================

@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def draw_curve_kernel_aliased(n: ti.i32, color_r: ti.f32, color_g: ti.f32, color_b: ti.f32):
    """基础绘制：直接点亮单个像素（有锯齿）"""
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([color_r, color_g, color_b])


@ti.kernel
def draw_curve_kernel_antialiased(n: ti.i32, color_r: ti.f32, color_g: ti.f32, color_b: ti.f32):
    """
    反走样绘制：使用 3x3 邻域加权混合
    颜色会根据距离进行衰减
    """
    for i in range(n):
        pt = curve_points_field[i]
        # 浮点坐标
        fx = pt[0] * WIDTH
        fy = pt[1] * HEIGHT
        
        # 最近的整数像素坐标
        cx = ti.cast(fx, ti.i32)
        cy = ti.cast(fy, ti.i32)
        
        # 遍历 3x3 邻域
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px = cx + dx
                py = cy + dy
                
                # 边界检查
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    # 像素中心坐标
                    center_x = ti.cast(px, ti.f32) + 0.5
                    center_y = ti.cast(py, ti.f32) + 0.5
                    
                    # 计算距离
                    dist = ti.sqrt((fx - center_x) * (fx - center_x) + 
                                  (fy - center_y) * (fy - center_y))
                    
                    # 使用高斯权重（效果更平滑）
                    weight = ti.exp(-dist * dist * 1.5)
                    # 或者使用线性衰减：
                    # weight = ti.max(0.0, 1.0 - dist / 1.5)
                    
                    # 加权混合颜色
                    current = pixels[px, py]
                    target = ti.Vector([color_r * weight, color_g * weight, color_b * weight])
                    # 使用最大值混合（适合线条绘制）
                    pixels[px, py] = ti.max(current, target)


# ==================== 主程序 ====================

def main():
    # 创建窗口
    window = ti.ui.Window("Interactive Curve Editor", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    
    # 状态变量
    control_points = []
    curve_mode = 'bezier'  # 'bezier' 或 'bspline'
    antialiasing_enabled = True
    
    # 颜色配置
    colors = {
        'bezier': (0.0, 1.0, 0.0),      # 绿色
        'bspline': (0.0, 0.6, 1.0)      # 蓝色
    }
    
    print("=" * 50)
    print("Interactive Curve Editor")
    print("=" * 50)
    print("Controls:")
    print("  - Left Click: Add control point")
    print("  - Key 'C': Clear all control points")
    print("  - Key 'B': Switch between Bezier and B-Spline mode")
    print("  - Key 'A': Toggle anti-aliasing ON/OFF")
    print("  - Key 'ESC' or close window: Exit")
    print("=" * 50)
    print(f"Current mode: {curve_mode.upper()}")
    print(f"Anti-aliasing: {'ON' if antialiasing_enabled else 'OFF'}")
    print("=" * 50)
    
    # 帧计数器（用于控制台输出频率）
    frame_count = 0
    
    while window.running:
        # --- 处理事件 ---
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    current_count = len(control_points)
                    print(f"[+] Added point {current_count}: ({pos[0]:.3f}, {pos[1]:.3f})")
                    
            elif e.key == 'c':
                control_points = []
                print("[x] Canvas cleared")
                
            elif e.key == 'b':
                if curve_mode == 'bezier':
                    curve_mode = 'bspline'
                    print(f"[~] Switched to B-SPLINE mode (needs ≥ 4 points)")
                else:
                    curve_mode = 'bezier'
                    print(f"[~] Switched to BEZIER mode (needs ≥ 2 points)")
                    
            elif e.key == 'a':
                antialiasing_enabled = not antialiasing_enabled
                print(f"[~] Anti-aliasing: {'ON' if antialiasing_enabled else 'OFF'}")
        
        # --- 清空画面 ---
        clear_pixels()
        
        # --- 计算并绘制曲线 ---
        current_count = len(control_points)
        color = colors[curve_mode]
        
        if current_count >= 2:
            curve_points_np = None
            
            if curve_mode == 'bezier':
                # 贝塞尔曲线：需要 ≥ 2 个点
                if current_count >= 2:
                    curve_points_np = compute_bezier_curve(control_points, NUM_SEGMENTS)
                    
            else:  # bspline
                # B 样条曲线：需要 ≥ 4 个点
                if current_count >= 4:
                    curve_points_np = compute_bspline_curve(control_points, NUM_SEGMENTS)
                elif current_count >= 2 and current_count < 4:
                    # 如果点数在2-3之间，显示提示（但只显示一次）
                    if frame_count % 60 == 0:
                        print("[i] B-Spline needs at least 4 points")
            
            # 绘制曲线
            if curve_points_np is not None and len(curve_points_np) > 0:
                # 确保数组大小正确
                if len(curve_points_np) != NUM_SEGMENTS + 1:
                    # 重采样到固定数量
                    resampled = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
                    for i in range(NUM_SEGMENTS + 1):
                        idx = i * (len(curve_points_np) - 1) / NUM_SEGMENTS
                        idx0 = int(idx)
                        idx1 = min(idx0 + 1, len(curve_points_np) - 1)
                        frac = idx - idx0
                        resampled[i] = curve_points_np[idx0] * (1 - frac) + curve_points_np[idx1] * frac
                    curve_points_np = resampled
                
                # 上传到 GPU
                curve_points_field.from_numpy(curve_points_np)
                
                # 选择绘制内核
                if antialiasing_enabled:
                    draw_curve_kernel_antialiased(NUM_SEGMENTS + 1, color[0], color[1], color[2])
                else:
                    draw_curve_kernel_aliased(NUM_SEGMENTS + 1, color[0], color[1], color[2])
        
        # --- 显示画面 ---
        canvas.set_image(pixels)
        
        # --- 绘制控制点和控制多边形 ---
        if current_count > 0:
            # 控制点
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.008, color=(1.0, 0.0, 0.0))
            
            # 控制多边形（灰线连接）
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.4, 0.4, 0.4))
        
        # --- 更新窗口 ---
        window.show()
        
        # 增加帧计数器
        frame_count += 1
        
        # 每60帧显示一次状态信息（避免刷屏）
        if frame_count % 60 == 0:
            mode_str = "Bezier" if curve_mode == 'bezier' else "B-Spline"
            aa_str = "ON" if antialiasing_enabled else "OFF"
            print(f"\r[{mode_str}] AA:{aa_str} | Points:{current_count}/{MAX_CONTROL_POINTS}", end="", flush=True)


if __name__ == '__main__':
    main()