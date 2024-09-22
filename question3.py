# 问题3——二分法求最小螺距
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from question2 import polygons_overlap, cal_vertex, objective
from question_func import cal_theta, curve
from draw import draw_rect

b_range = (4.5 / (32 * np.pi), 0.55 / (2 * np.pi))
# thetas = np.load('file/thetas_all.npy')[300:]

if __name__ == '__main__':
    b_max = b_range[1]
    b_min = b_range[0]
    res_b = 0
    j = 0
    while(np.abs(b_max-b_min) > 1e-5):
    # for b in np.arange(b_min, b_max, 0.005):
        b = (b_min + b_max) / 2
        is_crash = False  # 是否碰撞
        thetas_0 = cal_theta(b) 
        # theta0 = theta_0
        print('------------------------------------------')
        print(b * 2 * np.pi)
        for theta_0 in thetas_0:
            theta0 = theta_0
            T0 = []
            T_res = []
            flag_T0 = False
            x0, y0 = curve(theta0, b)
            print(np.sqrt(x0 ** 2 + y0 ** 2))
            for i in range(2, 225):
                l = 3.41 if i == 2 else 2.2
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
                    x1, y1 = curve(theta0, b)
                    T = cal_vertex(x0, y0, x1, y1, l)
                    T_res.append(T)
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
                            draw_rect(b, T_res,f'qusetion3.pdf')
                            # j += 1
                            res_theta = theta_0
                            break
                    if theta0 > theta_0 + 3 * np.pi:
                        break
                x0, y0 = x1, y1
            if is_crash:
                print('crash')
                break
        if is_crash:
            res_b = b
            b_min = b
        else:
            b_max = b
        print('num:' + str(len(T_res)))
        if np.abs(b_max-b_min) < 1e-5:
            draw_rect(b, T_res,f'qusetion3.pdf')
    print('res:' + str((b_min + b_max) / 2 * (2 * np.pi)))
    # print('res:' + str(res_b))


