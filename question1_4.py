# 问题1——7个把手的速度
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from save_to_excel import save_to_excel

max_x = 0.55 * 16
thetas = [32 * np.pi, 93.465, 85.819, 77.422, 67.995, 57.032]
result_arr = np.zeros((7,6))
use_i = [1, 2, 52, 102, 152, 202, 224]

def curve(theta):
    b = 0.55 / (2 * np.pi)
    x = b * theta * np.cos(theta)
    y = b * theta * np.sin(theta)
    return x, y

def cal_v(theta0, theta1, v0):
    b = 0.55 / (2 * np.pi)
    x0, y0 = curve(theta0)
    x1, y1 = curve(theta1)
    x, y = x0 - x1, y0 - y1
    k = v0 * np.sqrt(1+theta1 ** 2) / np.sqrt(1+theta0 ** 2)  * (((-np.cos(theta0) + theta0 * np.sin(theta0))* x + (-np.sin(theta0)-theta0*np.cos(theta0))*y)
        / ((-np.cos(theta1) + theta1 * np.sin(theta1))* x + (-np.sin(theta1)-theta1*np.cos(theta1))*y))
    return np.abs(k)

def cal_v_1(theta0, l, v0):
    b = 0.55 / (2 * np.pi)
    x0, y0 = curve(theta0)
    x1, y1 = max_x, np.sqrt(l ** 2 - (max_x - x0) ** 2) + y0
    x, y = x0 - x1, y0 - y1
    v = v0 /  np.sqrt(1+theta0 ** 2) * ((np.cos(theta0) - theta0*np.sin(theta0))* x + (np.sin(theta0) + theta0*np.cos(theta0))*y) / y
    return np.abs(v)
    

# 定义方程
def equation(theta, theta0, L):
    return (theta0 ** 2 + (theta0 + theta) ** 2 
        - 2 * theta0 * (theta0 + theta) * np.cos(theta) - L * np.pi ** 2)
    
def objective(theta, theta0, L):
    return np.abs(equation(theta, theta0, L))

j = 0
res = []
for theta_0 in thetas:
    theta0 = theta_0
    flag = True if theta_0 != thetas[0] else False
    v = 1
    result_arr[0][j] = 1
    for i in range(2, 225):
        l = 2.86 if i == 2 else 1.65
        L = 10.4 ** 2 if i == 2 else 36 
        if flag:
            # 使用 minimize 方法解方程
            result = minimize(objective, x0 = 0.2, args=(theta0, L), method='L-BFGS-B')
            theta_solution = result.x[0]
            print(f"Calculated theta_solution: {round(theta_solution, 3)}")
            # if theta0 + theta_solution >= 32 * np.pi:
            #     flag = False
            #     v = cal_v_1(theta0, l, v)
            # else:
            v = cal_v(theta0, theta0+theta_solution, v)
            theta0 += theta_solution
        else:
            v = v
        if i in use_i:
            result_arr[(use_i.index(i))][j] = round(v, 6)
    j += 1
    
print(result_arr)

# 创建行名和列名
rows = [1, 51, 101, 151, 201]
row_names = ['龙头(m/s)']
for row in rows:
    row_names.append(f'第{row}节龙身(m/s)')
row_names.append('龙尾(m/s)')
column_names = ['0s', '60s', '120s', '180s', '240s', '300s']

save_to_excel(result_arr, row_names, column_names, 'question1_1(全螺旋).xlsx', '速度')

