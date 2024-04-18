import numpy as np
import matplotlib.pyplot as plt

COLORS = [
    "#2678DB",
    "#DB8926",
    "#DE3232"
]

beta = np.array([0, 15, 30, 45, 60, 75, 90])

alpha = np.array([0, 28, 56, 88, 118, 150, 178])

plt.figure(figsize=(16, 10), dpi=400)
ax = plt.axes()

x_tick = 20
x_min = np.floor(np.amin(beta) / x_tick) * x_tick
x_max = np.ceil(np.amax(beta) / x_tick) * x_tick

y_tick = 40
y_min = np.floor(np.amin(alpha) / y_tick) * y_tick
y_max = np.ceil(np.amax(alpha) / y_tick) * y_tick

plt.plot(
    [x_min, x_max],
    [x_min * 2, x_max * 2],
    c=COLORS[0], linewidth=2,
    label=f"Теоретическая зависимость поворота пл-ти поляризации $α(β)$"
)

plt.errorbar(
    beta,
    alpha,
    xerr=5.0,
    yerr=2.0,
    color=COLORS[1], marker='s', markersize=8, linewidth=0,
    ecolor=COLORS[2], elinewidth=2,
    label=f"Экспериментальные точки"
)

plt.legend(fontsize=18)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(np.arange(x_min, x_max + x_tick / 20, x_tick), fontsize=16)
plt.yticks(np.arange(y_min, y_max + y_tick / 20, y_tick), fontsize=16)
ax.xaxis.set_minor_locator(plt.MultipleLocator(x_tick / 4))
ax.yaxis.set_minor_locator(plt.MultipleLocator(y_tick / 4))

ax.grid(which="major", c="#696969", linestyle="-", linewidth=2, alpha=0.6)
ax.grid(which="minor", c="#696969", linestyle="--", linewidth=1, alpha=0.6)

plt.title(f"Зависимость поворота пл-ти поляризации от ориентации пластинки $λ/2$ $α(β)$",
          fontsize=22, pad=22)
plt.xlabel("Угол поворота главных направлений пластинки $λ/2$ $β$, °",
           fontsize=20, labelpad=10)
plt.ylabel("Поворот плоскости поляризации $α$, °",
           fontsize=20, labelpad=10)

plt.savefig("polarization_pics/alpha-beta.png")

plt.legend(fontsize=18)
