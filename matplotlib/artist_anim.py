
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

ims = []
t = np.linspace(0,10, 1000)

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(1,10):
    line = ax.plot(t, np.sin(t*i), color="C0", label=i)
    ims.append(line)

s = animation.ArtistAnimation(fig, ims, blit=True, interval=100)
s.save("./test2.gif", dpi=150)