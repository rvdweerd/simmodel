import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def PlotPerformanceCharts(algos,performance_metrics):
    num_iter = performance_metrics['e_returns'][algos[0].__name__].shape[1]
    for algo in algos:
        metrics_mean = np.mean(performance_metrics['e_returns'][algo.__name__], axis=0)
        metrics_std = np.std(performance_metrics['e_returns'][algo.__name__], axis=0)
        xs = np.arange(num_iter)
        ys = metrics_mean
        stds = metrics_std
        lower_std = np.array([val - std for val, std in zip(ys, stds)])
        upper_std = np.array([val + std for val, std in zip(ys, stds)])
        #plt.figure()
        global_min=np.min(performance_metrics['e_returns'][algo.__name__])
        global_max=np.max(performance_metrics['e_returns'][algo.__name__])
        lower_std=np.clip(lower_std,global_min,global_max)
        upper_std=np.clip(upper_std,global_min,global_max)
        plt.plot(xs, ys, label=algo.__name__)
        plt.fill_between(xs, lower_std, upper_std, alpha=0.4)
        plt.legend()
    plt.xlabel("Episode")
    plt.title("Episode returns")
    plt.ylim((-8,6))
    plt.show()

def PlotGridValues(algos,env,Q_table):
    if env.type == 'graph': return
    cols = env.cols
    rows = env.rows
    vmin = env.final_reward * -1.2
    vmax = env.final_reward

    for algo in algos:
        # Vanilla
        V_table = np.max(Q_table[algo.__name__], axis=1).reshape(rows, cols)
        #plt.imshow(V_table[::-1][:], vmin=vmin, vmax=vmax, cmap='seismic_r')
        plt.imshow(V_table[::-1][:], vmin=vmin, vmax=vmax, cmap='PiYG')
        plt.title("State values:"+algo.__name__)
        #plt.colorbar()
        plt.show()


        # First Vanilla
        plt.xlim((0, cols))
        plt.ylim((0, rows))
        plt.title("Greedy policy:"+algo.__name__)
        for i in range(cols):
            for j in range(rows):
                state_number = j*cols+i
                state_color = env.get_state_color(state_number)
                plt.gca().add_patch(Rectangle((i, j), 1, 1, linewidth=1,
                                                edgecolor='r', facecolor=state_color))
                max_actions,  = np.where(
                    Q_table[algo.__name__][state_number] == np.max(Q_table[algo.__name__][state_number]))
                if state_color in ["white", "blue"]:
                    if 1 in max_actions:
                        plt.arrow(i+0.5, j+0.5, 0.45, 0, width=0.04,
                                    length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
                    if 2 in max_actions:
                        plt.arrow(i+0.5, j+0.5, 0, -0.45, width=0.04,
                                    length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
                    if 3 in max_actions:
                        plt.arrow(i+0.5, j+0.5, -0.45, 0, width=0.04,
                                    length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
                    if 0 in max_actions:
                        plt.arrow(i+0.5, j+0.5, 0, 0.45, width=0.04,
                                    length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
        plt.show()
        #input('key')