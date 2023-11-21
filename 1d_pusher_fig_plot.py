import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

PSCALE = 0.1
OSCALE = 1

pusher = np.arange(48, 50, PSCALE / 2)
obstacle = np.arange(0, 100, OSCALE)

block = np.zeros((int(10 / (10 * PSCALE / 4)),int(100 / OSCALE)))
for i, p in enumerate(pusher):
    for j, o in enumerate(obstacle):
        after_push = 50.0
        if p < 50:
            after_push = min(max(p + 2, 50), 52)
        after_obstacle = after_push
        if after_push != 5 and o > 50.5:
            after_obstacle = min(after_push, o - 0.5)
        block[i,j] = after_obstacle
        print(p,o,after_push, after_obstacle)
print(block.shape)

fig, ax = plt.subplots()

B, D = np.meshgrid(pusher, obstacle)
c = ax.pcolormesh(obstacle, pusher, block, vmin=50, vmax=52)
c.set_edgecolor('face')


# plt.contourf(pusher, obstacle, block)
ax.axis([0, 100, 48, 50])
plt.colorbar(c, ax=ax)
plt.xlabel('obstacle')
plt.ylabel('pusher')
plt.show()
fig.set_size_inches(3*3,1.2*3)
plt.savefig("1d_pusher.png")