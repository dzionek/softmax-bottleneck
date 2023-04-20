import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

BIGGER_SIZE = 15

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
_, ax = plt.subplots()

X = np.arange(0, 10, 0.1)
Y = np.arange(0, 10, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.logaddexp(X, Y)

contour = ax.contour(X, Y, Z, cmap=cm.coolwarm)
ax.clabel(contour, inline=True, fontsize=BIGGER_SIZE)

plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()


# plt.show()
plt.savefig('lse_contour.pdf')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.invert_xaxis()
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()

# plt.show()
plt.savefig('lse_3d.pdf')