import matplotlib.pyplot as plt
import numpy as np


def plot_car_results(x_opt, u_opt, L, h):
    """
    Plot car trajectory (with orientation shadows + steering indicators),
    first-order controls, and finite-difference acceleration / steering rate.

    Returns the matplotlib Figure.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

    car_length = L
    car_width = L * 0.6
    steps_to_show = np.arange(0, len(x_opt), max(1, len(x_opt) // 20))
    n_shadows = len(steps_to_show)

    for i, k in enumerate(steps_to_show):
        p1, p2, theta = x_opt[k]
        corners = np.array([
            [ car_length/2,  car_width/2],
            [ car_length/2, -car_width/2],
            [-car_length/2, -car_width/2],
            [-car_length/2,  car_width/2],
            [ car_length/2,  car_width/2],
        ])
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
        rotated = (R @ corners.T).T + np.array([p1, p2])
        alpha = 0.15 + 0.7 * (i + 1) / n_shadows
        ax1.fill(rotated[:, 0], rotated[:, 1], color='dodgerblue',
                 alpha=alpha, edgecolor='k', linewidth=0.7)

        if k < len(u_opt):
            phi = u_opt[k, 1]
            front_center = (np.array([p1, p2]) +
                            (car_length / 2) * np.array([np.cos(theta), np.sin(theta)]))
            steer_tip = (front_center +
                         (car_length / 3) * np.array([np.cos(theta + phi), np.sin(theta + phi)]))
            ax1.plot([front_center[0], steer_tip[0]],
                     [front_center[1], steer_tip[1]],
                     color='crimson', linewidth=1, alpha=alpha + 0.1)

    ax1.plot(x_opt[0, 0],  x_opt[0, 1],  'go', markersize=10, label='Start')
    ax1.plot(x_opt[-1, 0], x_opt[-1, 1], 'ro', markersize=10, label='Goal')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 1)
    ax1.set_aspect('equal')

    k_steps = np.arange(len(u_opt))
    ax2.plot(k_steps, u_opt[:, 0], 'b-', linewidth=2, label='Speed (m/s)')
    ax2.plot(k_steps, u_opt[:, 1], 'r-', linewidth=2, label='Steering Angle (rad)')
    ax2.set_xlabel('$k$', fontsize=16)
    ax2.set_ylabel('$u_k$', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)

    acceleration  = np.diff(u_opt[:, 0]) / h
    steering_rate = np.diff(u_opt[:, 1]) / h
    k_steps_diff = np.arange(len(acceleration))
    ax3.plot(k_steps_diff, acceleration,  'g-', linewidth=2, label='Acceleration (m/s²)')
    ax3.plot(k_steps_diff, steering_rate, 'm-', linewidth=2, label='Steering Rate (rad/s)')
    ax3.set_xlabel('$k$', fontsize=16)
    ax3.set_ylabel(r'$\frac{u_k - u_{k-1}}{h}$', fontsize=18)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)

    plt.tight_layout()
    return fig
