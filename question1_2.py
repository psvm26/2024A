# 问题1:7个把手的位置
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from save_to_excel import save_to_excel

max_x = 0.55 * 16
thetas = [32 * np.pi, 93.465, 85.819, 77.422, 67.995, 57.032]
result_arr = np.zeros((14,6))
use_i = [1, 2, 52, 102, 152, 202, 224]

def curve(theta):
    b = 0.55 / (2 * np.pi)
    x = b * theta * np.cos(theta)
    y = b * theta * np.sin(theta)
    return x, y

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
    flag = True
    x, y = curve(theta0)
    result_arr[0][j] = round(x, 6)
    result_arr[1][j] = round(y, 6)
    for i in range(2, 225):
        l = 2.86 if i == 2 else 1.65
        L = 10.4 ** 2 if i == 2 else 36 
        if flag:
            # 使用 minimize 方法解方程
            # result = minimize(objective, x0 = 0.2, args=(theta0, L), method='Nelder-Mead')
            result = minimize(objective, x0 = 0.2, args=(theta0, L), method='L-BFGS-B')
            theta_solution = result.x[0]
            # print(f"Calculated theta_solution: {round(theta_solution, 3)}")
            res.append(objective(theta_solution, theta0, L))
            if theta0 + theta_solution >= 32 * np.pi:
                flag = False
                x, y = curve(theta0)
                print(x,y)
                y = np.sqrt(l ** 2 - (max_x - x) ** 2) + y
            else:
                theta0 += theta_solution
                x, y = curve(theta0)
        else:
            x = max_x
            y += l
        if i in use_i:
            result_arr[(use_i.index(i))*2][j] = round(x, 6)
            result_arr[(use_i.index(i))*2+1][j] = round(y, 6)
    j += 1
    
print(result_arr)
print(sorted(res)[-10:])
# 创建行名和列名
rows = [1, 51, 101, 151, 201]
row_names = ['龙头x(m)', '龙头y(m)']
for row in rows:
    row_names.append(f'第{row}节龙身x(m)')
    row_names.append(f'第{row}节龙身y(m)')
row_names.append('龙尾x(m)')
row_names.append('龙尾y(m)')
column_names = ['0s', '60s', '120s', '180s', '240s', '300s']

save_to_excel(result_arr, row_names, column_names, 'question1_1.xlsx', '位置')

