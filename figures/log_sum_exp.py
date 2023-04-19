import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

_, ax = plt.subplots()

X = np.arange(0, 10, 0.1)
Y = np.arange(0, 10, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.logaddexp(X, Y)

contour = ax.contour(X, Y, Z, cmap=cm.coolwarm)
ax.clabel(contour, inline=True, fontsize=10)

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