import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

COLORS = [
    "#D2D43A",
    "#3C3AD4",
    "#E49435"
]


def power(alpha, alpha_0, power_max, power_min):
    d_alpha = np.radians(alpha - alpha_0)
    return power_max * np.sin(d_alpha) ** 2 + power_min * np.cos(d_alpha) ** 2


def power_jacobian(alpha, alpha_0, power_max, power_min):
    d_alpha = np.radians(alpha - alpha_0)
    return np.stack(
        [
            (np.pi / 180) * np.sin(2 * d_alpha) * (power_min - power_max),
            np.sin(d_alpha) ** 2,
            np.cos(d_alpha) ** 2
        ],
        axis=1
    )


power_data = np.array([864, 698, 645, 718, 886, 1115, 1403, 1464, 1700, 1666, 1491, 1217, 929],
                      dtype=np.float64)
alpha_data = np.arange(0, 181, 15,
                       dtype=np.float64)

filtered_power = power_data[~np.isin(power_data, 1464)]
filtered_alpha = alpha_data[~np.isin(alpha_data, 105)]

power_err = power_data * 0.011
filtered_power_err = filtered_power * 0.011
alpha_err = np.full_like(alpha_data, 2.5)

parameters_0 = np.array([0, 1700, 650],
                        dtype=np.float64)

result = opt.curve_fit(power,
                       filtered_alpha,
                       filtered_power,
                       p0=parameters_0,
                       sigma=filtered_power_err,
                       absolute_sigma=False,
                       jac=power_jacobian)

result_err = np.sqrt(np.diag(result[1]))

alpha_0_opt = np.array([result[0][0], result_err[0]])
power_max_opt = np.array([result[0][1], result_err[1]])
power_min_opt = np.array([result[0][2], result_err[2]])

print(f"alpha_0: {alpha_0_opt[0]}±{alpha_0_opt[1]}\n"
      f"power_max: {power_max_opt[0]}±{power_max_opt[1]}\n"
      f"power_min: {power_min_opt[0]}±{power_min_opt[1]}\n")

plt.figure(figsize=(16, 10), dpi=400)
ax = plt.axes()

x_tick = 30
x_min = np.floor(np.amin(alpha_data) / x_tick) * x_tick - x_tick / 3
x_max = np.ceil(np.amax(alpha_data) / x_tick) * x_tick + x_tick / 3

alpha_space = np.linspace(x_min, x_max, 1000)
power_space = power(alpha_space, alpha_0_opt[0], power_max_opt[0], power_min_opt[0])

y_tick = 250
y_min = np.floor(np.amin(power_space) / y_tick) * y_tick
y_max = np.ceil(np.amax(power_space) / y_tick) * y_tick

plt.plot(
    alpha_space,
    power_space,
    c=COLORS[0], linewidth=3,
    label=f"Оптимизированная модель $P(α)$"
)

plt.errorbar(
    alpha_data,
    power_data,
    xerr=alpha_err,
    yerr=power_err,
    color=COLORS[1], marker='s', markersize=8, linewidth=0,
    ecolor=COLORS[2], elinewidth=3,
    label=f"Экспериментальные точки"
)

plt.plot(
    105,
    1464,
    color="red", marker='x', markersize=16, linewidth=0,
    zorder=10
)

plt.legend(fontsize=18)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(np.arange(0, x_max + x_tick / 20, x_tick), fontsize=16)
plt.yticks(np.arange(y_min, y_max + y_tick / 20, y_tick), fontsize=16)
ax.xaxis.set_minor_locator(plt.MultipleLocator(x_tick / 3))
ax.yaxis.set_minor_locator(plt.MultipleLocator(y_tick / 5))

ax.grid(which="major", c="#696969", linestyle="-", linewidth=1, alpha=0.6)
ax.grid(which="minor", c="#696969", linestyle="--", linewidth=0.5, alpha=0.6)

plt.title(f"Зависимость мощности света от положения анализатора $P(α)$",
          fontsize=22, pad=22)
plt.xlabel("Угол поворота анализатора $α$, °",
           fontsize=20, labelpad=10)
plt.ylabel("Мощность светового излучения $P$, мкВт",
           fontsize=20, labelpad=10)

plt.savefig("polarization_pics/power-alpha_x_plate.png")

plt.legend(fontsize=18)
