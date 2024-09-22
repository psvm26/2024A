# 要使用的函数
from scipy.optimize import minimize
import numpy as np

thetas = []
losses = []
# 定义方程
def equation_1(theta1, L, theta2, b):
    return (L - b * (theta2/2 * np.sqrt(1 + theta2**2) +
                     (1/2) * np.log(theta2 + np.sqrt(1 + theta2**2)) -
                     theta1/2 * np.sqrt(1 + theta1**2) -
                     (1/2) * np.log(theta1 + np.sqrt(1 + theta1**2))))

def objective_1(theta1, L, theta2, b):
    return np.abs(equation_1(theta1, L, theta2, b))

def curve(theta, b):
    x = b * theta * np.cos(theta)
    y = b * theta * np.sin(theta)
    return x, y

def cal_R(theta, b):
    x, y = curve(theta, b)
    return np.sqrt(x ** 2 + y ** 2)

def cal_theta(b):
    theta2 = 32 * np.pi  # 例子中的角度
    L = 0  
    res_theta = []
    for i in range(0, 450):
    # 使用 minimize 方法解方程
        result = minimize(objective_1, x0=95, args=(L, 32 * np.pi, b), method='L-BFGS-B')
        theta1_solution = result.x[0]
        res_theta.append(theta1_solution)
        if cal_R(theta1_solution, b) < 4.5:
            return res_theta[-30:]
        L += 1
    
if __name__ == '__main__':
    print(thetas)
    print(sorted(losses)[-10:])
    np.save('file/thetas_4124_4125.npy', thetas)

