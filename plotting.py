from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

def plot_moving_average(data, window=500):
    cs = np.cumsum(np.insert(data,0,0))
    ma = (cs[window:] - cs[:-window]) / window
    plt.plot(ma)
    plt.show()

def animate(functions):
    # print([id(f) for f in functions])

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    X = np.arange(0, 1, 0.05)
    Y = np.arange(0, 1, 0.05)
    X, Y = np.meshgrid(X, Y)

    def update(num, data, surf):
        ax.cla()
        set_axes()
        surf = ax.plot_surface(X,Y,data[num])

    def set_axes():
        ax.set_zlim(0.0, 1.0)

    # print(functions[0](0,0))
    # print(functions[1](0,0))

    data = np.array([f(X,Y) for f in functions])
    surf = ax.plot_surface(X,Y,data[0])
    set_axes()

    ani = animation.FuncAnimation(fig, update, len(functions), fargs=(data,surf), interval=20)

    plt.show()
