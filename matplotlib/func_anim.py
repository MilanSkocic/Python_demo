import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def init():
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-2, 2)
    line.set_xdata(t)
    line.set_ydata(y)
    return line,

def update(frame, *args):

    t, = args
    f = frame+1
    line.set_ydata(np.sin(t*f))

    return line,
    

fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([],[], ls="-")

t = np.linspace(0,10, 1000)
f = 1.0
y = np.sin(t*f)

s = animation.FuncAnimation(fig, update, init_func=init, fargs=(t,), blit=True, 
                            frames=100, interval=100, repeat=False)

s.save("./test.gif", dpi=150)
