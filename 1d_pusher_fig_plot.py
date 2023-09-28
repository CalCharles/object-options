import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

SCALE = 0.1

pusher = np.arange(0, 10, SCALE)
obstacle = np.arange(0, 10, SCALE)

block = np.zeros((int(10 / SCALE),int(10 / SCALE)))
for i, p in enumerate(pusher):
    for j, o in enumerate(obstacle):
        after_push = 5.0
        if p < 5:
            after_push = min(max(p + 2, 5), 7)
        after_obstacle = after_push
        if after_push != 5 and o > 5.5:
            after_obstacle = min(after_push, o - 0.5)
        block[j,i] = after_obstacle
        print(p,o,after_push, after_obstacle)
print(block.shape)

fig, ax = plt.subplots()

B, D = np.meshgrid(pusher, obstacle)
c = ax.pcolormesh(pusher, obstacle, block, vmin=5, vmax=7)
c.set_edgecolor('face')


# plt.contourf(pusher, obstacle, block)
ax.axis([0, 10, 0, 10])
plt.colorbar(c, ax=ax)
plt.xlabel('pusher')
plt.ylabel('obstacle')
plt.show()
plt.savefig("1d_pusher.svg")