from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib import animation, pylab


g = 9.81


def init_plot(R):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.axis("off")

    time_text = ax.text(
        0.45, 0.95, "", transform=ax.transAxes, family="Consolas", fontsize=13
    )

    return fig, ax, time_text


def create_subplots(ax, n):
    colors = pylab.get_cmap("rainbow")(np.linspace(0, 1, n))
    pends, tracks = [], []
    for i in range(n):
        pend, = ax.plot([], [], marker="o", lw=2, c=colors[i], zorder=i)
        track, = ax.plot([], [], lw=0.5, c=colors[i], zorder=i)
        pends.append(pend)
        tracks.append(track)

    return pends, tracks


def derivs(state, t, l1, l2, m1, m2):
    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (m1 + m2) * l1 - m2 * l1 * cos(delta) * cos(delta)
    dydx[1] = (
        m2 * l1 * state[1] * state[1] * sin(delta) * cos(delta)
        + m2 * g * sin(state[2]) * cos(delta)
        + m2 * l2 * state[3] * state[3] * sin(delta)
        - (m1 + m2) * g * sin(state[0])
    ) / den1

    dydx[2] = state[3]

    den2 = (l2 / l1) * den1
    dydx[3] = (
        -m2 * l2 * state[3] * state[3] * sin(delta) * cos(delta)
        + (m1 + m2) * g * sin(state[0]) * cos(delta)
        - (m1 + m2) * l1 * state[1] * state[1] * sin(delta)
        - (m1 + m2) * g * sin(state[2])
    ) / den2

    return dydx


def solve_ode(state, t, l1, l2, m1, m2):
    y = integrate.odeint(derivs, state, t, args=(l1, l2, m1, m2))
    x1, y1 = l1 * sin(y[:, 0]), -l1 * cos(y[:, 0])
    x2, y2 = l2 * sin(y[:, 2]) + x1, -l2 * cos(y[:, 2]) + y1

    return np.column_stack([x1, y1, x2, y2])


def get_positions(n, t, l1, l2, m1, m2, th1, av1, th2, av2, perturbation):
    positions = np.zeros(len(t), dtype=[(f"P{i}", "float32", 4) for i in range(n)])
    for i in range(n):
        state = np.radians([th1, av1, th2 + i * perturbation, av2])  # initial state
        positions[f"P{i}"] = solve_ode(state, t, l1, l2, m1, m2)

    return positions


def animate(i, positions, pends, tracks, tail, time_text):
    time_text.set_text("{:.1f} s".format(i * dt))
    for j in range(n):
        pos = positions[f"P{j}"]
        thisx = [0, pos[i, 0], pos[i, 2]]
        thisy = [0, pos[i, 1], pos[i, 3]]

        pends[j].set_data(thisx, thisy)
        tracks[j].set_data(pos[max(0, i - tail) : i, 2], pos[max(0, i - tail) : i, 3])

    return pends + tracks + [time_text]


if __name__ == "__main__":
    # INPUT
    l1 = 1.0  # length of pendulum 1 in m
    l2 = 1.0  # length of pendulum 2 in m
    m1 = 1.0  # mass of pendulum 1 in kg
    m2 = 1.0  # mass of pendulum 2 in kg
    th1, av1 = 120.0, 0.0  # the initial angles in degrees
    th2, av2 = 130.0, 0.0  # the initial angular velocities in degrees per second
    n = 12  # number of pendulums
    perturbation = 1e-6
    T, dt = 18, 0.02  # duration of the motion and the timestep length in seconds
    tail = 100  # number of previous pendulum positions to be drawn

    # BODY
    t = np.arange(0, T, dt)
    positions = get_positions(n, t, l1, l2, m1, m2, th1, av1, th2, av2, perturbation)

    fig, ax, time_text = init_plot(l1 + l2)
    pends, tracks = create_subplots(ax, n)
    ani = animation.FuncAnimation(
        fig,
        animate,
        range(1, len(t)),
        fargs=[positions, pends, tracks, tail, time_text],
        interval=dt * 1000,
        blit=True,
        repeat_delay=100,
    )

    plt.show()
