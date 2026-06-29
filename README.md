# 计算机图形学实验报告

## 王赛楠 202411998177 计算机科学与技术

## 实验三 贝塞尔曲线与B样条曲线的交互式绘制系统

## 一、实验目标

### 1.1 核心目标

- 理解贝塞尔曲线（Bézier Curve）的几何意义与数学原理
- 掌握De Casteljau算法的递归插值思想与代码实现
- 理解光栅化的基础概念，掌握像素缓冲区（Frame Buffer）的直接操作
- 掌握现代图形界面中的鼠标交互与事件处理机制

### 1.2 进阶目标（选做内容）

- 实现反走样（Anti-aliasing）技术，消除曲线边缘的锯齿现象
- 实现均匀三次B样条曲线（B-Spline Curve），并与贝塞尔曲线进行对比分析

***

## 二、实验环境

| 项目   | 配置                |
| ---- | ----------------- |
| 编程语言 | Python 3.10+      |
| 图形框架 | Taichi (GGUI)     |
| 计算后端 | GPU (CUDA/Vulkan) |
| IDE  | Trae              |
| 操作系统 | Windows 11        |

### 2.1 核心依赖

```python
import taichi as ti
import numpy as np
```

***

## 三、实验原理

### 3.1 贝塞尔曲线与De Casteljau算法

贝塞尔曲线由一组控制点定义，通过参数 $t \in \[0, 1]$ 描述曲线上的点。De Casteljau算法通过递归线性插值计算曲线上任意一点：

**算法步骤**：

1. 给定 $n$ 个控制点 $P\_0, P\_1, \dots, P\_{n-1}$
2. 对于参数 $t$，在相邻点之间进行线性插值：
   $$P'_i = (1-t)P\_i + tP_{i+1}, \quad i = 0, 1, \dots, n-2$$
3. 对新生成的 $n-1$ 个点重复上述操作
4. 递归直至只剩下一个点，该点即为曲线在参数 $t$ 处的位置

**代码实现**：

```python
def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)
```

### 3.2 均匀三次B样条曲线

B样条曲线通过分段多项式基函数实现局部控制。对于均匀三次B样条，每4个相邻控制点定义一段曲线。

**分段绘制逻辑**：

- 若有 $n$ 个控制点（$n \ge 4$），则曲线由 $n-3$ 段拼接而成
- 每段独立计算，保证 $C^2$ 连续性

### 3.3 反走样技术

走样现象源于像素坐标的整数截断。反走样通过对亚像素精度坐标进行加权采样来缓解。

***

## 四、核心功能实现

### 4.1 GPU绘制内核

**基础绘制（有锯齿）**：

```python
@ti.kernel
def draw_curve_kernel_aliased(n: ti.i32, color_r: ti.f32, color_g: ti.f32, color_b: ti.f32):
    for i in range(n):  # GPU并行循环
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([color_r, color_g, color_b])
```

**反走样绘制**：

```python
@ti.kernel
def draw_curve_kernel_antialiased(n: ti.i32, color_r: ti.f32, color_g: ti.f32, color_b: ti.f32):
    for i in range(n):
        fx = pt[0] * WIDTH
        fy = pt[1] * HEIGHT
        cx = ti.cast(fx, ti.i32)
        cy = ti.cast(fy, ti.i32)
        
        # 3x3邻域加权混合
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px, py = cx + dx, cy + dy
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    center_x = ti.cast(px, ti.f32) + 0.5
                    center_y = ti.cast(py, ti.f32) + 0.5
                    dist = ti.sqrt((fx - center_x)**2 + (fy - center_y)**2)
                    weight = ti.exp(-dist * dist * 1.5)
                    target = ti.Vector([color_r * weight, color_g * weight, color_b * weight])
                    pixels[px, py] = ti.max(pixels[px, py], target)
```

### 4.2 交互事件处理

```python
for e in window.get_events(ti.ui.PRESS):
    if e.key == ti.ui.LMB:  # 鼠标左键添加控制点
        if len(control_points) < MAX_CONTROL_POINTS:
            pos = window.get_cursor_pos()
            control_points.append(pos)
    elif e.key == 'c':  # C键清空
        control_points = []
    elif e.key == 'b':  # B键切换曲线模式
        curve_mode = 'bspline' if curve_mode == 'bezier' else 'bezier'
    elif e.key == 'a':  # A键切换反走样
        antialiasing_enabled = not antialiasing_enabled
```

***

## 五、实验结果与分析

### 5.1 功能验证

| 功能模块  | 测试用例   | 预期结果   | 实际结果 |
| ----- | ------ | ------ | ---- |
| 控制点添加 | 左键点击5次 | 显示5个红点 | ✅ 通过 |
| 贝塞尔曲线 | 3个控制点  | 绿色平滑曲线 | ✅ 通过 |
| B样条曲线 | 6个控制点  | 蓝色分段曲线 | ✅ 通过 |
| 模式切换  | 按B键    | 曲线颜色切换 | ✅ 通过 |
| 清空画布  | 按C键    | 所有元素消失 | ✅ 通过 |
| 反走样   | 按A键    | 边缘更平滑  | ✅ 通过 |

### 5.2 视觉效果对比

#### 贝塞尔曲线 vs B样条曲线

| 特性   | 贝塞尔曲线           | B样条曲线       |
| ---- | --------------- | ----------- |
| 控制方式 | 全局控制            | 局部控制        |
| 控制点数 | $n$ 个点 = $n-1$阶 | 固定3次（4个点/段） |
| 修改影响 | 整条曲线            | 局部区域        |

**观察结论**：

- 贝塞尔曲线：移动任意控制点，整条曲线形状都会改变，适合整体造型设计
- B样条曲线：移动中间控制点只影响附近几段，适合局部精细调整

***

## 六、运行录屏
![gif](https://work3-ezgif.com-video-to-gif-converter.gif)
