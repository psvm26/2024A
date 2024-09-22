# 问题1:求龙头位置
from scipy.optimize import minimize
import numpy as np

# 定义方程
def equation_1(theta1, L, theta2, b):
    return (L - b * (theta2/2 * np.sqrt(1 + theta2**2) +
                     (1/2) * np.log(theta2 + np.sqrt(1 + theta2**2)) -
                     theta1/2 * np.sqrt(1 + theta1**2) -
                     (1/2) * np.log(theta1 + np.sqrt(1 + theta1**2))))

def objective_1(theta1, L, theta2, b):
    return np.abs(equation_1(theta1, L, theta2, b))

if __name__ == '__main__':
    thetas = []
    losses = []
    theta2 = 32 * np.pi  # 例子中的角度
    L = 0
    b = 0.55 / (2 * np.pi) 
    # b = 1.7 / (2 * np.pi)
    # print(-1*equation_1(30*np.pi,0, theta2, b))
    for i in range(0, 300):
        # 使用 brentq 方法解方程
        bounds = [(0, theta2)]  # theta1 的界限

    # 使用 minimize 方法解方程
        result = minimize(objective_1, x0=95, args=(L, 32*np.pi, b), method='L-BFGS-B')
        theta1_solution = result.x[0]
        # print(f"Calculated theta1: {round(theta1_solution, 3)}")
        # print(-1 * equation(theta1_solution,L, 32*np.pi, b))
        losses.append(-1 * equation_1(theta1_solution,L, 32*np.pi, b))
        thetas.append(theta1_solution)
        theta2 = theta1_solution
        L += 1
    
if __name__ == '__main__':
    print(thetas)
    print(sorted(losses)[-10:])
    np.save('file/thetas.npy', thetas)

