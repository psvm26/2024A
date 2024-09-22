# 问题2——绘制最小盘入的示意图
import numpy as np
from scipy.optimize import minimize
from draw import draw_rect

def is_separating_axis(polygon1, polygon2, axis):
    # 计算在给定轴上的投影
    def project_polygon(polygon, axis):
        min_proj = float('inf')
        max_proj = float('-inf')
        for point in polygon:
            projection = np.dot(point, axis)
            min_proj = min(min_proj, projection)
            max_proj = max(max_proj, projection)
        return min_proj, max_proj

    # 计算两个多边形在该轴上的投影
    min1, max1 = project_polygon(polygon1, axis)
    min2, max2 = project_polygon(polygon2, axis)

    # 检查投影是否重叠
    return max1 < min2 or max2 < min1

def polygons_overlap(polygon1, polygon2):
    # 获取两个多边形的所有边的法向量（即潜在分离轴）
    def get_axes(polygon):
        axes = []
        for i in range(len(polygon)):
            # 当前点和下一个点
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            # 计算边的向量
            edge = np.array(p2) - np.array(p1)
            # 计算边的法向量 (垂直)
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
        return axes

    # 获取两个多边形的分离轴
    axes1 = get_axes(polygon1)
    axes2 = get_axes(polygon2)

    # 对每一个轴进行分离轴检测
    for axis in axes1 + axes2:
        if is_separating_axis(polygon1, polygon2, axis):
            return False  # 如果找到分离轴，两个多边形不重叠

    return True  # 没有找到分离轴，两个多边形重叠

def cal_vertex(x0, y0, x1, y1, l):
    d = 0.15  # 半宽度
    x_m , y_m = (x0 + x1) / 2, (y0 + y1) / 2  # 中点坐标
    ne_x =  (x0 - x1) / np.sqrt((x0-x1) ** 2 + (y0 - y1) ** 2)
    ne_y =  (y0 - y1) / np.sqrt((x0-x1) ** 2 + (y0 - y1) ** 2)  #矩形方向向量
    nev_x = ne_y
    nev_y = -ne_x
    x, y = [], []
    x.append(x_m + l / 2 * ne_x + d * nev_x)
    y.append(y_m + l / 2 * ne_y + d * nev_y)
    x.append(x_m + l / 2 * ne_x - d * nev_x)
    y.append(y_m + l / 2 * ne_y - d * nev_y)
    x.append(x_m - l / 2 * ne_x - d * nev_x)
    y.append(y_m - l / 2 * ne_y - d * nev_y)
    x.append(x_m - l / 2 * ne_x + d * nev_x)
    y.append(y_m - l / 2 * ne_y + d * nev_y)
    return [(x[0],y[0]),(x[1],y[1]),(x[2],y[2]),(x[3],y[3])]

def curve(b, theta):
    x = b * theta * np.cos(theta)
    y = b * theta * np.sin(theta)
    return x, y

# 定义方程
def equation(theta, theta0, L):
    return (theta0 ** 2 + (theta0 + theta) ** 2 
        - 2 * theta0 * (theta0 + theta) * np.cos(theta) - L * np.pi ** 2)
    
def objective(theta, theta0, L):
    return np.abs(equation(theta, theta0, L))


if __name__ == '__main__':
    thetas = np.load('file/thetas_4124_4125.npy')  # 龙头位置
    # print(thetas)
    is_crash = False  # 是否碰撞
    crash = []
    res = 0
    res_i = 0
    res_T = []
    res_theta = 0
    b = 0.55 / (2 * np.pi)
    # b = 1.7 / 2 / (2 * np.pi)

    for theta_0 in thetas:
        theta0 = theta_0
        print(np.where(thetas == theta_0)[0][0])
        flag_T0 = False  # 是否装填龙头和第一节
        T_end = []
        T0 = []
        x0, y0 = curve(b, theta0)
        for i in range(2, 225):
            l = 3.41 if i == 2 else 2.2
            # L = 10.4 ** 2 if i == 2 else 36 
            L = (2.86 / (b*np.pi)) ** 2 if i == 2 else (1.65 / (b*np.pi)) ** 2
            # 使用 minimize 方法解方程
            result = minimize(objective, x0 = 0.2, args=(theta0, L), bounds=[(0, 2 * np.pi)],method='Nelder-Mead')
            theta_solution = result.x[0]
            # print(f"Calculated theta_solution: {round(theta_solution, 3)}")
            # print(objective(theta_solution, theta0, L))
            if theta0 + theta_solution >= 32 * np.pi:
                break
            else:
                theta0 += theta_solution
                x1, y1 = curve(b, theta0)
                T = cal_vertex(x0, y0, x1, y1, l)
                T_end.append(T)
                if not flag_T0:
                    T0.append(T)
                    if len(T0) == 2:
                        flag_T0 = True
                else:
                    if i == 4:
                        is_crash = polygons_overlap(T, T0[0])
                    else:
                        is_crash = polygons_overlap(T, T0[0]) or polygons_overlap(T, T0[1])
                    if is_crash:
                        res = np.where(thetas == theta_0)[0][0]
                        res_i = i
                        res_T = T0
                        res_T.append(T)
                        res_theta = theta_0
                        break
                if theta0 > theta_0 + 6 * np.pi:
                    break
            x0, y0 = x1, y1
        crash.append(is_crash)
        if is_crash:
            draw_rect(b, T_end, 'question2.pdf')
            break
    print(crash)
    print(res, res_i)
    print(res_T)
    print(res_theta)
    res_x, res_y = curve(b, res_theta)
    print(np.sqrt(res_x ** 2 + res_y ** 2))


