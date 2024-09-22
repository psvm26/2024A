# 问题4——遗传算法计算最优theta1.theta2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sko.GA import GA
from question4 import get_track

def schaffer(p):
    theta1, theta2 = p

    R, arc_Len, theta, l = get_track(theta1, theta2, cal=True)
    if (theta < 180 and l < 2.86) or (theta >= 180 and R < 1.43) or np.abs(theta1 - theta2) > np.pi:
        return np.inf  # 如果违反约束，返回无穷大
    return arc_Len

max_theta = 4.5 * 2 * np.pi/ 1.7 
ga = GA(func=schaffer, n_dim=2, size_pop=50, max_iter=800, prob_mut=0.001, lb=[0, 0], ub=[max_theta, max_theta], precision=1e-7)
best_theta, best_fit = ga.run()
print('best_theta:', best_theta, '\n', 'best_arc:', best_fit)
print(get_track(best_theta[0], best_theta[1], cal=True))

