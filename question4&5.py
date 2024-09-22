# 问题4——计算掉头时把手位置 问题5——计算最大速度
from question1_1 import objective_1
from question4 import objective_R, get_track
from question2 import cal_vertex, polygons_overlap
from draw import draw_track
from save_to_excel import save_to_excel
from question_func import curve
import numpy as np
from scipy.optimize import minimize
from math import *

# 定义方程
def equation_5(theta2, L, theta1, b):
    return (L - b * (theta2/2 * np.sqrt(1 + theta2**2) +
                     (1/2) * np.log(theta2 + np.sqrt(1 + theta2**2)) -
                     theta1/2 * np.sqrt(1 + theta1**2) -
                     (1/2) * np.log(theta1 + np.sqrt(1 + theta1**2))))

def objective_5(theta1, L, theta2, b):
    return np.abs(equation_5(theta1, L, theta2, b))

def cal_coord(A, O, R, theta, clock = True):
    '''
    计算圆上点的坐标
    clock = True表示顺时针
    '''
    n_x = A[0] - O[0]
    n_y = A[1] - O[1]
    alpha = np.arctan2(n_y, n_x) - theta if clock else np.arctan2(n_y, n_x) + theta
    return O[0] + R * np.cos(alpha), O[1] + R * np.sin(alpha)

def cal_luo_theta(b, theta_0, L, begin = True):
    '''
    已知弧长和起始/终点角度，求终点/起始角度
    begin = True 已知起始角度，否则已知最终角度
    '''
    if begin:
        result = minimize(objective_1, x0=95, args=(L, theta_0, b), method='L-BFGS-B')
    else:
        result = minimize(objective_5, x0=95, args=(L, theta_0, b), method='L-BFGS-B')
    theta_solution = result.x[0]
    return theta_solution

def cal_xy(theta, mode, b, A, B, C, D, R):
    x, y = 0, 0
    if mode == 1:
        x, y = curve(theta, b)
    elif mode == 2:
        x, y = cal_coord(A, C, 2 * R, theta, True)
    elif mode == 3:
        x, y = cal_coord(B, D, R, theta, True)
    else:
        x, y = curve(theta, b)
        x, y = -x, -y
    return [x, y]

def cal_head_theta(b, theta1, theta2, v, begin_second = -100, end_second = 100):
    begin_second = - begin_second
    theta = []
    coord = []
    A, B, C, D, R, arc_Len, thetaC_begin, thetaC_end, thetaD_begin, thetaD_end = get_track(theta1, theta2)
    thetaC_max = np.radians(thetaC_end-thetaC_begin)
    thetaD_max = np.radians(thetaD_end-thetaD_begin)
    begin_theta = cal_luo_theta(b, theta1, begin_second * v, False)
    end_theta = cal_luo_theta(b, theta2, end_second * v - arc_Len, False)
    # 第一段曲线
    for i in range(0, begin_second):
        result_theta = cal_luo_theta(b, begin_theta, i * v, True)
        theta.append([result_theta, 1])
    # 第二,三段曲线
    thetaC = 0
    thetaD = 0
    flag_CD = False
    theta.append([thetaC, 2])
    for i in range(0, floor(arc_Len / v)):
        if thetaC < thetaC_max and not flag_CD:
            theta.append([thetaC, 2])
            thetaC += v / (2 * R)
        else:
            if not flag_CD:
                flag_CD = True
                begin_len = v - 2 * R * (thetaC_max + v / (2 * R) - thetaC) 
                thetaD += begin_len / R
            else:
                thetaD += v / R
            theta.append([thetaD, 3])
    #第四段曲线：
    len = arc_Len - floor(arc_Len / v) * v
    for i in range(0, end_second - floor(arc_Len / v)):
        result_theta = cal_luo_theta(b, theta2, len + i * v, False)
        theta.append([result_theta, 4])
    # for i in np.arange(end_second - floor(arc_Len / v)-4, end_second - floor(arc_Len / v)-2,0.01):
    #     result_theta = cal_luo_theta(b, theta2, len + i * v, False)
    #     theta.append([result_theta, 4])
    # 将theta转换成坐标
    for t in theta:
        coord.append(cal_xy(t[0], t[1], b, A, B, C, D, R))

    return begin_theta, end_theta, theta, coord

def cal_direction(theta, mode, b, A, B, C, D, R):
    '''计算方向向量'''
    nx, xy = 0, 0
    M = cal_xy(theta, mode, b, A, B, C, D, R)
    if mode == 1 or mode == 4:
        nx = 1 / np.sqrt(1 + theta ** 2) * (-np.cos(theta) + theta * np.sin(theta))
        ny = 1 / np.sqrt(1 + theta ** 2) * (-np.sin(theta) - theta * np.cos(theta))
        if mode == 4:
            nx, ny = -nx, -ny
    else:
        if mode == 2:
            nx = 1 / np.sqrt((M[0] - C[0]) ** 2 + (M[1] - C[1]) ** 2) * (M[0] - C[0])
            ny = 1 / np.sqrt((M[0] - C[0]) ** 2 + (M[1] - C[1]) ** 2) * (M[1] - C[1])
            nx, ny = ny, -nx
        else:
            nx = 1 / np.sqrt((M[0] - D[0]) ** 2 + (M[1] - D[1]) ** 2) * (M[0] - D[0])
            ny = 1 / np.sqrt((M[0] - D[0]) ** 2 + (M[1] - D[1]) ** 2) * (M[1] - D[1])
            nx, ny = -ny, nx
    return M, [nx, ny]
        

def cal_v(theta_new, mode_new, theta_now, mode_now, v_now, b, A, B, C, D, R):
    '''计算速度,返回速度和坐标'''
    now, n_now = cal_direction(theta_now, mode_now, b, A, B, C, D, R)
    new, n_new = cal_direction(theta_new, mode_new, b, A, B, C, D, R)
    alpha = [now[0] - new[0], now[1] - new[1]]
    v_new = v_now * (n_now[0] * alpha[0] + n_now[1] * alpha[1]) / (n_new[0] * alpha[0] + n_new[1] * alpha[1])
    return new[0], new[1], np.abs(v_new)
    # return new[0], new[1], v_new


def cal_dis(theta_new, mode_new, theta_now, mode_now, b, A, B, C, D, R):
    now = cal_xy(theta_now, mode_now, b, A, B, C, D, R)
    new = cal_xy(theta_new, mode_new, b, A, B, C, D, R)
    return np.sqrt((now[0] - new[0]) ** 2 + (now[1] - new[1]) ** 2)

def equation_dis(theta_new, mode_new, theta_now, mode_now, b, A, B, C, D, R, L):
    return L - cal_dis(theta_new, mode_new, theta_now, mode_now, b, A, B, C, D, R)

def objective_dis(theta_new, mode_new, theta_now, mode_now, b, A, B, C, D, R, L):
    return np.abs(equation_dis(theta_new, mode_new, theta_now, mode_now, b, A, B, C, D, R, L))

if __name__ == '__main__':
    b = 1.7 / (2 * np.pi)
    theta1 = 13.02410994 
    theta2 = 11.42576913
    v = 1
    A, B, C, D, R, arc_Len, thetaC_begin, thetaC_end, thetaD_begin, thetaD_end = get_track(theta1, theta2)
    thetaC_max = np.radians(thetaC_end-thetaC_begin)
    thetaD_max = np.radians(thetaD_end-thetaD_begin)
    # begin_theta, end_theta, theta, coord = cal_head_theta(b, theta1, theta2, v)
    begin_theta, end_theta, theta, coord = cal_head_theta(b, theta1, theta2, v, begin_second=-100, end_second=100)
    mode_theta = [0, 0, 0, 0, theta2]
    points = []  # 把手位置
    res_xy = np.zeros((224*2, 201))
    res_v = np.zeros((224, 201))
    # res_T = []  # 龙舟长方形
    # res_v = []
    # res_xy = np.zeros((7*2, 5))
    # res_v = np.zeros((7, 5))
    j = 0
    is_crash = False
    # use_i = [0, 1, 51, 101, 151, 201, 223]
    # theta_5 = [theta[0], theta[50], theta[100], theta[150], theta[200]]
    # use_time = [-100, -50, 0, 50, 100]
    # v_max = 0

    # 计算坐标和速度
    for head_theta in theta:
    # for head_theta in theta_5:
        print(j)
        res_T = []  # 龙舟长方形
        # res_v = []
        theta0, mode0 = head_theta
        theta_now, mode_now = theta0, mode0
        v_now = v
        x_now, y_now = cal_xy(theta_now, mode_now, b, A, B, C, D, R)
        # res_xy[0][j] = round(x_now, 6)
        # res_xy[1][j] = round(y_now, 6)
        # res_v[0][j] = round(v_now, 6)
        points.append([x_now, y_now])
        # 碰撞检验
        T0 = []
        flag_T0 = False
        for i in range(1, 224):
            flag_minue_mode = False
            # print(i)
            L = 3.41 if i == 1 else 2.2  # 板长
            l = 2.86 if i == 1 else 1.65  # 孔距
            if mode_now == 4 and theta_now-theta2 > np.pi:
                mode_new = mode_now
            elif mode_now != 1 and cal_dis(theta_now ,mode_now, mode_theta[mode_now], mode_now, b, A, B, C, D, R) < l:
                mode_new = mode_now - 1
                if cal_dis(theta_now ,mode_now, mode_theta[mode_new], mode_new, b, A, B, C, D, R) < l and mode_new != 1:
                    mode_new -= 1
                flag_minue_mode = True
            else:
                mode_new = mode_now
            if not flag_minue_mode:
                mode_bound = [[(0, 0)], [(theta_now, theta_now + np.pi)], [(0, theta_now)], [(0, theta_now)], [(theta_now - np.pi, theta_now)]]
                x0 = theta_now
            else:
                mode_bound = [[(0, 0)], [(theta1, theta1 + np.pi)], [(0, thetaC_max)], [(0, thetaD_max)], [(0,0)]]
                mode_begin = [0, theta1, thetaC_max, thetaD_max, 0]
                x0 = mode_begin[mode_new]
            result = minimize(objective_dis, x0=x0,bounds=mode_bound[mode_new], args=(mode_new, theta_now, mode_now, b, A, B, C, D, R, l), method='L-BFGS-B')
            loss = objective_dis(result.x[0], mode_new, theta_now, mode_now, b, A, B, C, D, R, l)
            theta_new = result.x[0]
            if  loss > 1e-3:
                # print(f'loss{j}:'+str(loss))
                result = minimize(objective_dis, x0=0,bounds=mode_bound[mode_new], args=(mode_new, theta_now, mode_now, b, A, B, C, D, R, l), method='L-BFGS-B')
                loss_new = objective_dis(result.x[0], mode_new, theta_now, mode_now, b, A, B, C, D, R, l)
                if loss_new < loss:
                    theta_new = result.x[0]
                else:
                    mode_new += 1
                    result = minimize(objective_dis, x0=0,bounds=mode_bound[mode_new], args=(mode_new, theta_now, mode_now, b, A, B, C, D, R, l), method='L-BFGS-B')
                    loss_new = objective_dis(result.x[0], mode_new, theta_now, mode_now, b, A, B, C, D, R, l)
                    if loss_new < loss:
                        theta_new = result.x[0]
            # print(theta_new, objective_dis(theta_new, mode_new, theta_now, mode_now, b, A, B, C, D, R, l))
            x_new, y_new, v_new = cal_v(theta_new, mode_new, theta_now, mode_now, v_now, b, A, B, C, D, R)
            T = cal_vertex(x_now, y_now, x_new, y_new, L)
            # 添加龙头和第一节
            if not flag_T0:
                T0.append(T)
                if len(T0) == 2:
                    flag_T0 = True
            else:
                if i == 3:
                    is_crash = polygons_overlap(T, T0[0])
                else:
                    is_crash = polygons_overlap(T, T0[0]) or polygons_overlap(T, T0[1])
            if is_crash:
                break
            res_T.append(T)
            points.append([x_new, y_new])
            # res_v.append(v_new)
            res_xy[i*2][j] = x_new
            res_xy[i*2+1][j] = y_new
            res_v[i][j] = v_new
            # if i in use_i:
            #     res_xy[use_i.index(i)*2][j] = x_new
            #     res_xy[use_i.index(i)*2+1][j] = y_new
            #     res_v[use_i.index(i)][j] = v_new
            x_now, y_now , v_now= x_new, y_new, v_new
            theta_now, mode_now = theta_new, mode_new
        # question5:求最大速度
        # print(max(res_v))
        # if max(res_v) > v_max:
        #     draw_track(f'question5/track_of_max_v.pdf',theta1, theta2, A, B, C, D, R, thetaC_begin, thetaC_end, thetaD_begin, thetaD_end, T=res_T, title=f'最大速度时板凳龙所处位置')
        #     v_max = max(res_v)
        #     print('new_max:'+str(v_max))
        j += 1
        if is_crash:
            # draw_track('track_of_crash.pdf',theta1, theta2, A, B, C, D, R, thetaC_begin, thetaC_end, thetaD_begin, thetaD_end, T = res_T)
            break
    
    print(R, arc_Len)
    print(res_v)

    # 保存为表格
    rows = np.arange(1, 222)
    # rows = [1, 51, 101, 151, 201]
    vrow_names = ['龙头(m/s)']
    xyrow_names = ['龙头x(m)','龙头y(m)']
    for row in rows:
        vrow_names.append(f'第{row}节龙身(m/s)')
        xyrow_names.append(f'第{row}节龙身x(m)')
        xyrow_names.append(f'第{row}节龙身y(m)')
    vrow_names.append('龙尾(m/s)')
    vrow_names.append('龙尾(后)(m/s)')
    xyrow_names.append('龙尾x(m)')
    xyrow_names.append('龙尾y(m)')
    xyrow_names.append('龙尾(后)x(m)')
    xyrow_names.append('龙尾(后)xy(m)')

    column_names = []
    # use_time = [-100, -50, 0, 50, 100]
    for t in np.arange(1, 101):
        column_names.append(f'-{101-t}s')
    column_names.append('0s')
    for t in np.arange(1, 101):
        column_names.append(f'{t}s')
    # for t in use_time:
    #     column_names.append(f'{t}s')
    # print(column_names)

    save_to_excel(res_xy, xyrow_names, column_names, 'result4.xlsx', '位置')
    save_to_excel(res_v, vrow_names, column_names, 'result4.xlsx', '速度')

    # head_points = []
    # for t in theta:
    #     head_points.append(cal_xy(t[0], t[1], b, A, B, C, D, R))
    # 绘图
    # draw_track('track_of_head.pdf',theta1, theta2, A, B, C, D, R, thetaC_begin, thetaC_end, thetaD_begin, thetaD_end, points=head_points)
    # draw_track(f'track_of_all_test_1.pdf',theta1, theta2, A, B, C, D, R, thetaC_begin, thetaC_end, thetaD_begin, thetaD_end, T=res_T, title=f'max_v轨迹图')
    
    
    
