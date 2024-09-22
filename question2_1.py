# 问题2——计算能盘入的最小角度
import numpy as np
import pandas as pd
from save_to_excel import save_to_excel
from scipy.optimize import minimize

max_x = 0.55 * 16

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

def cal_v_1(theta0, v0):
    b = 0.55 / (2 * np.pi)
    x0, y0 = curve(theta0)
    x1, y1 = max_x, np.sqrt(l ** 2 - (max_x - x0) ** 2) + y0
    x, y = x0 - x1, y0 - y1
    v = v0 /  np.sqrt(1+theta0 ** 2) * ((np.cos(theta0) - theta0*np.sin(theta0))* x + (np.sin(theta0) + theta0*np.cos(theta0))*y) / y
    return abs(v)

# 定义方程
def equation(theta, theta0, L):
    return (theta0 ** 2 + (theta0 + theta) ** 2 
        - 2 * theta0 * (theta0 + theta) * np.cos(theta) - L * np.pi ** 2)
    
def objective(theta, theta0, L):
    return np.abs(equation(theta, theta0, L))

theta0 = np.load('file/thetas_4124_4125.npy')[7]
# result_arr = np.zeros((224,3))
result_arr = np.zeros((7,3))
loss = []
use_i = [1, 2, 52, 102, 152, 202, 224]
# print(theta0)

flag = True
x, y = curve(theta0)
v = 1
result_arr[0][0] = round(x, 6)
result_arr[0][1] = round(y, 6)
result_arr[0][2] = round(v, 6)
for i in range(2, 225):
    l = 2.86 if i == 2 else 1.65
    L = 10.4 ** 2 if i == 2 else 36 
    if flag:
        # 使用 minimize 方法解方程
        result = minimize(objective, x0 = 0.2, args=(theta0, L), method='L-BFGS-B')
        theta_solution = result.x[0]
        print(f"Calculated theta_solution: {round(theta_solution, 3)}")
        loss.append(objective(theta_solution, theta0, L))
        if theta0 + theta_solution >= 32 * np.pi:
            flag = False
            x, y = curve(theta0)
            v = cal_v_1(theta0, v)
            # print(x,y)
            y = np.sqrt(l ** 2 - (max_x - x) ** 2) + y
        else:
            v = cal_v(theta0, theta0+theta_solution, v)
            theta0 += theta_solution
            x, y = curve(theta0)
    else:
        x = max_x
        y += l
        v = v
    if i in use_i:
        result_arr[use_i.index(i)][0] = round(x, 6)
        result_arr[use_i.index(i)][1] = round(y, 6)
        result_arr[use_i.index(i)][2] = round(v, 6)
    # result_arr[i-1][0] = round(x, 6)
    # result_arr[i-1][1] = round(y, 6)
    # result_arr[i-1][2] = round(v, 6)


print(result_arr)
print(sorted(loss)[-10:])
# 创建行名和列名
# rows = np.arange(1, 222)
rows = [1, 51, 101, 151, 201]
row_names = ['龙头']
for row in rows:
    row_names.append(f'第{row}节龙身')
# row_names.append('龙尾')
row_names.append('龙尾(后)')
column_names = ['横坐标x(m)', '纵坐标y(m)', '速度(m/s)']

save_to_excel(result_arr, row_names, column_names, 'question2_1.xlsx', 'sheet1')