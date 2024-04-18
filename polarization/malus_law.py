import numpy as np
import matplotlib.pyplot as plt

COLORS = [
    "#35C987",
    "#C93577",
    "#DE3232"
]

POWER_DETECTOR_RELATIVE_ERR = 0.011


def malus_law(p_0, theta):
    return p_0 * np.cos(np.radians(theta)) ** 2


d_theta = np.array([90, 86, 80, 70, 60, 50, 40, 30, 20, 10, 0],
                   dtype=np.float64)

d_theta_err = np.full_like(d_theta, 2,
                           dtype=np.float64)

d_theta_space = np.linspace(0, 90, 1000,
                            dtype=np.float64)

power = np.array([0.270, 4.70, 27.9, 104, 218, 359, 506, 655, 749, 823, 846],
                 dtype=np.float64)

power_err_m = np.array([10e-3, 10e-2, 10e-1] + [10e-0] * 8, dtype=np.float64)
power_err_i = power * POWER_DETECTOR_RELATIVE_ERR
power_err = np.sqrt(power_err_m ** 2 + power_err_i ** 2)

power_space = malus_law(power[-1], d_theta_space)

plt.figure(figsize=(16, 10), dpi=400)
ax = plt.axes()

x_tick = 10
x_min = np.floor(np.amin(d_theta_space) / x_tick) * x_tick
x_max = np.ceil(np.amax(d_theta_space) / x_tick) * x_tick

y_tick = 200
y_min = np.floor(np.amin(power_space) / y_tick) * y_tick
y_max = np.ceil(np.amax(power_space) / y_tick) * y_tick

plt.plot(
    d_theta_space,
    power_space,
    c=COLORS[0], linewidth=2,
    label=f"Теоретическая кривая мощности излучения $P(Δθ)$"
)

plt.errorbar(
    d_theta,
    power,
    xerr=d_theta_err,
    yerr=power_err,
    color=COLORS[1], marker='s', markersize=8, linewidth=0,
    ecolor=COLORS[2], elinewidth=2,
    label=f"Экспериментальные точки"
)

plt.legend(fontsize=18)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(np.arange(x_min, x_max + x_tick / 20, x_tick), fontsize=16)
plt.yticks(np.arange(y_min, y_max + y_tick / 20, y_tick), fontsize=16)
ax.xaxis.set_minor_locator(plt.MultipleLocator(x_tick / 5))
ax.yaxis.set_minor_locator(plt.MultipleLocator(y_tick / 5))

ax.grid(which="major", c="#696969", linestyle="-", linewidth=2, alpha=0.6)
ax.grid(which="minor", c="#696969", linestyle="--", linewidth=1, alpha=0.6)

plt.title(f"Зависимость мощности излучения от положения поляризатора $P(Δθ)$",
          fontsize=22, pad=22)
plt.xlabel("Угол между плоскостью поляризации и разрешённым направлением $Δθ$, °",
           fontsize=20, labelpad=10)
plt.ylabel("Мощность излучения на детекторе $P$, мкВт",
           fontsize=20, labelpad=10)

plt.savefig("polarization_pics/p-d_theta.png")

plt.legend(fontsize=18)
