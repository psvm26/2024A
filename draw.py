# 绘图
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np

# 添加字体路径
font_path = "SimHei.ttf"  # 替换为实际的字体文件路径
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def draw_rect(b, points, fig_name):
    # 创建一个图形和坐标轴
    fig, ax = plt.subplots()

    # 绘制半径为 0.45 的圆，边界和填充用淡色
    circle = patches.Circle((0, 0), radius=4.5, edgecolor='gray', facecolor='gray', alpha=0.5)
    ax.add_patch(circle)
    # 设置螺线的参数
    theta = np.linspace(0, 32 * np.pi, 1000)  # 生成一系列角度，范围从 0 到 32*pi
    phi = b * theta         # 根据公式计算螺线的半径

    # 将极坐标转换为笛卡尔坐标
    x = phi * np.cos(theta)
    y = phi * np.sin(theta)  

    # 绘制螺线
    ax.plot(x, y, color='lightblue', label='螺线')

    # 创建一个多边形并添加到坐标轴
    for p in points:
        polygon = patches.Polygon(p, closed=True, linewidth=1, edgecolor='black', facecolor='none', zorder=3)
        ax.add_patch(polygon)
    # 设置坐标轴的显示范围，确保矩形在图形中完整显示
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.xlabel('x轴(m)')
    plt.ylabel('y轴(m)')

    # 显示图形
    plt.gca().set_aspect('equal', adjustable='box')  # 保持长宽比
    plt.savefig(f'figure/{fig_name}')

def draw_luo():
    # 创建一个图形和坐标轴
    fig, ax = plt.subplots()
     # 设置螺线的参数
    b = 1.7 / (2 * np.pi)
    theta = np.linspace(0, 32 * np.pi, 1000)  # 生成一系列角度，范围从 0 到 32*pi
    phi = b * theta         # 根据公式计算螺线的半径

    # 将极坐标转换为笛卡尔坐标
    x = phi * np.cos(theta)
    y = phi * np.sin(theta)  

    # 绘制螺线
    ax.plot(x, y, color='r', label='螺线1')

    x = phi * np.cos(theta + np.pi)
    y = phi * np.sin(theta + np.pi)
    ax.plot(x, y, color='b', label='螺线2')
    # 设置坐标轴的显示范围
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # 显示图形
    plt.gca().set_aspect('equal', adjustable='box')  # 保持长宽比
    plt.savefig(f'figure/luo.pdf')

def draw_track(file_name, theta1, theta2, A, B, C, D, R, thetaC_begin, thetaC_end, thetaD_begin, thetaD_end, points = [], T = [], title = None):
    fig, ax = plt.subplots()

    circle = patches.Circle((0, 0), radius=4.5, edgecolor='gray', facecolor='gray', alpha=0.5)
    ax.add_patch(circle)

     # 设置螺线的参数
    b = 1.7 / (2 * np.pi)

    thetas1 = np.linspace(theta1, 32 * np.pi, 1000)  # 生成一系列角度，范围从 0 到 32*pi
    phi1 = b * thetas1         # 根据公式计算螺线的半径
    x1 = phi1 * np.cos(thetas1)
    y1 = phi1 * np.sin(thetas1)  
    ax.plot(x1, y1, color='lightblue', label='螺线1', zorder=1)

    thetas2 = np.linspace(theta2, 32 * np.pi, 1000)  # 生成一系列角度，范围从 0 到 32*pi
    phi2 = b * thetas2         # 根据公式计算螺线的半径
    x2 = phi2 * np.cos(thetas2 + np.pi)
    y2 = phi2 * np.sin(thetas2 + np.pi)  
    ax.plot(x2, y2, color='lightgreen', label='螺线2', zorder=1)

    #线段
    ax.plot([A[0], C[0]], [A[1], C[1]], linestyle='--', color='r')
    ax.plot([C[0], D[0]], [C[1], D[1]], linestyle='--', color='r')
    ax.plot([B[0], D[0]], [B[1], D[1]], linestyle='--', color='r')

    # 圆C
    arc1 = patches.Arc((C[0], C[1]), 4*R, 4*R, angle=0, theta1=thetaC_begin, theta2=thetaC_end, edgecolor='r')
    ax.add_patch(arc1)

    # 圆D
    arc2 = patches.Arc((D[0], D[1]), 2*R, 2*R, angle=0, theta1=thetaD_begin, theta2=thetaD_end, edgecolor='r')
    ax.add_patch(arc2)

    # 散点
    if len(points) != 0:
        x_coords, y_coords = zip(*points)
        ax.scatter(x_coords, y_coords, s=3, c='black')

    # 龙舟
    if len(T) != 0:
        i = 0
        for t in T:
            if i == 0:
                polygon = patches.Polygon(t, closed=True, linewidth=1, edgecolor='red', facecolor='none', zorder=3)
            elif i == len(T)-1:
                polygon = patches.Polygon(t, closed=True, linewidth=1, edgecolor='blue', facecolor='none', zorder=3)
            else:
                polygon = patches.Polygon(t, closed=True, linewidth=1, edgecolor='black', facecolor='none', zorder=3)
            i += 1
            ax.add_patch(polygon)

    # 设置坐标轴的显示范围
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    plt.xlabel('x轴(m)')
    plt.ylabel('y轴(m)')
    if title != None:
        plt.title(title)
    # plt.grid(True)

    # 显示图形
    plt.gca().set_aspect('equal', adjustable='box')  # 保持长宽比
    plt.savefig(f'figure/{file_name}')

if __name__ == '__main__':
    draw_luo()