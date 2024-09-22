# 问题4——计算最优掉头曲线
import numpy as np
from question_func import curve
from scipy.optimize import minimize
from draw import draw_track


# 定义方程
def equation_R(R, x_A, y_A, n_Ax, n_Ay, x_B, y_B, n_Bx, n_By):
    x_C, y_C = x_A + 2 * R * n_Ax, y_A + 2 * R * n_Ay 
    x_D, y_D = x_B + R * n_Bx, y_B + R * n_By 
    return (3 * R - np.sqrt((x_C - x_D) ** 2 + (y_C - y_D) ** 2))
    
def objective_R(R, x_A, y_A, n_Ax, n_Ay, x_B, y_B, n_Bx, n_By):
    return np.abs(equation_R(R, x_A, y_A, n_Ax, n_Ay, x_B, y_B, n_Bx, n_By))

def get_track(theta1, theta2, cal = False):
    b = 1.7 / (2 * np.pi)

    x_A, y_A = curve(theta1, b)
    n_Ax = 1 / np.sqrt(1 + theta1 ** 2) * (-theta1 * np.sin(theta1) + np.cos(theta1))
    n_Ay = 1 / np.sqrt(1 + theta1 ** 2) * (theta1 * np.cos(theta1) + np.sin(theta1))
    n_Ax, n_Ay = -n_Ay, n_Ax

    x_B, y_B = curve(theta2, b)
    x_B, y_B = -x_B, -y_B
    n_Bx = 1 / np.sqrt(1 + theta2 ** 2) * (theta2 * np.sin(theta2) - np.cos(theta2))
    n_By = 1 / np.sqrt(1 + theta2 ** 2) * (-theta2 * np.cos(theta2) - np.sin(theta2))
    n_Bx, n_By = -n_By, n_Bx

    result = minimize(objective_R, x0 = 4.5, args=(x_A, y_A, n_Ax, n_Ay, x_B, y_B, n_Bx, n_By),
                       bounds=[(0, 4.5)],method='Nelder-Mead')
    R = result.x[0]
    # print(R)
    # print(objective_R(R, x_A, y_A, n_Ax, n_Ay, x_B, y_B, n_Bx, n_By))

    x_C, y_C = x_A + 2 * R * n_Ax, y_A + 2 * R * n_Ay 
    x_D, y_D = x_B + R * n_Bx, y_B + R * n_By 

    thetaC_begin = np.degrees(np.arctan2(y_D - y_C, x_D - x_C))
    thetaC_end = np.degrees(np.arctan2(y_A - y_C, x_A - x_C))
    thetaD_begin = np.degrees(np.arctan2(y_C - y_D, x_C - x_D))
    thetaD_end = np.degrees(np.arctan2(y_B - y_D, x_B - x_D))
    if thetaC_end < thetaC_begin:
        thetaC_end += 360
    if thetaD_end < thetaD_begin:
        thetaD_end += 360

    arc_Len = 2 * R * np.radians(thetaC_end - thetaC_begin) + R * np.radians(thetaD_end - thetaD_begin)

    A = [x_A, y_A]
    B = [x_B, y_B]
    C = [x_C, y_C]
    D = [x_D, y_D]

    l = np.sqrt((x_B - x_C) ** 2 + (y_B - y_C) ** 2 - (2*R) ** 2)

    # alpha = np.arccos(1-1.65**2/(2*(2*R)**2))
    # l1 = 2*R*alpha + R * np.radians(thetaD_end - thetaD_begin)
    # print(l1)

    if cal:
        return R, arc_Len, thetaD_end - thetaD_begin, l
    else:
        return A, B, C, D, R, arc_Len, thetaC_begin, thetaC_end, thetaD_begin, thetaD_end


def cal_Larc_Len(b, theta1, theta2):
    return  b * (theta2/2 * np.sqrt(1 + theta2**2) +
            (1/2) * np.log(theta2 + np.sqrt(1 + theta2**2)) -
            theta1/2 * np.sqrt(1 + theta1**2) -
            (1/2) * np.log(theta1 + np.sqrt(1 + theta1**2)))

def cal_Carc_Len(R, theta1, theta2):
    return R * (theta2 - theta1)



if __name__ == '__main__':
    theta1 = 13.02410994 
    theta2 = 11.42576913
    A, B, C, D, R, arc_Len, thetaC_begin, thetaC_end, thetaD_begin, thetaD_end = get_track(theta1,theta2)
    print(arc_Len)
    draw_track('track.pdf', theta1, theta2, A, B, C, D, R, thetaC_begin, thetaC_end, thetaD_begin, thetaD_end, title='轨迹图')
