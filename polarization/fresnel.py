import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

COLORS = [
    "#0000FF",  # s-pol-line
    "#BF00FF",  # s-pol-dots
    "#FF0000",  # p-pol-line
    "#FFBF00",  # p-pol-dots
    "#F27474",  # errors
]


def r_s(theta_i, n_21) -> np.ndarray:
    cos_theta_i = np.cos(theta_i)
    sqr = np.sqrt(n_21 ** 2 - np.sin(theta_i) ** 2)
    num = cos_theta_i - sqr
    den = cos_theta_i + sqr

    return num / den


def r_p(theta_i, n_21) -> np.ndarray:
    n_cos_theta_i = n_21 ** 2 * np.cos(theta_i)
    sqr = np.sqrt(n_21 ** 2 - np.sin(theta_i) ** 2)
    num = n_cos_theta_i - sqr
    den = n_cos_theta_i + sqr

    return - num / den


def power_s(theta_i, n_21, power_s_0, noise) -> np.ndarray:
    return power_s_0 * r_s(theta_i, n_21) ** 2 + noise


def power_p(theta_i, n_21, power_p_0, noise) -> np.ndarray:
    return power_p_0 * r_p(theta_i, n_21) ** 2 + noise


def power_s_jacobi(theta_i, n_21, power_s_0, noise) -> np.ndarray:
    cos_theta_i = np.cos(theta_i)
    num = - 2 * n_21 * cos_theta_i
    sqr = np.sqrt(n_21 ** 2 - np.sin(theta_i) ** 2)
    den = sqr * (cos_theta_i + sqr) ** 2
    r_s0 = r_s(theta_i, n_21)

    return np.stack(
        [
            power_s_0 * 2 * r_s0 * num / den,
            r_s0 ** 2,
            np.ones_like(theta_i)
        ],
        axis=1
    )


def power_p_jacobi(theta_i, n_21, power_p_0, noise) -> np.ndarray:
    cos_theta_i = np.cos(theta_i)
    sq_sin_theta_i = np.sin(theta_i) ** 2
    sq_n_21 = n_21 ** 2
    num = 2 * n_21 * cos_theta_i * (- sq_n_21 + 2 * sq_sin_theta_i)
    sqr = np.sqrt(sq_n_21 - sq_sin_theta_i)
    den = sqr * (cos_theta_i * sq_n_21 + sqr) ** 2
    r_p0 = r_p(theta_i, n_21)

    return np.stack(
        [
            power_p_0 * 2 * r_p0 * num / den,
            r_p0 ** 2,
            np.ones_like(theta_i)
        ],
        axis=1
    )


theta_i_s_data = np.array(
    [
        15, 30, 35, 40, 45, 50, 55, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
        70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85
    ],
    dtype=np.float64
)

theta_i_s_err = np.full_like(theta_i_s_data, 1.0)

power_s_data = np.array(
    [
        70.2, 84.9, 93.9, 98.5, 111.9, 136.1, 158.5, 190.2, 192.8, 198.2, 211.4, 215.2, 220.0, 234.8, 245.1,
        254.7, 263.3, 280.1, 288.4, 302, 320, 337, 351, 368, 383, 401, 426, 439, 477, 499, 519, 552, 581
    ],
    dtype=np.float64
)

power_s_err = power_s_data * 0.011

parameters_s_0 = np.array([2.0, 1000.0, 0.0],
                          dtype=np.float64)

result_s = opt.curve_fit(power_s,
                         np.radians(theta_i_s_data),
                         power_s_data,
                         p0=parameters_s_0,
                         sigma=power_s_err,
                         absolute_sigma=False,
                         jac=power_s_jacobi)

result_err_s = np.sqrt(np.diag(result_s[1]))

n_21_s_opt = np.array([result_s[0][0], result_err_s[0]])
power_s_0_opt = np.array([result_s[0][1], result_err_s[1]])
noise_s_opt = np.array([result_s[0][2], result_err_s[2]])

print(f"n_21_s: {n_21_s_opt[0]}±{n_21_s_opt[1]}\n"
      f"power_s_0: {power_s_0_opt[0]}±{power_s_0_opt[1]}\n"
      f"noise_s: {noise_s_opt[0]}±{noise_s_opt[1]}\n")

plt.figure(figsize=(16, 10), dpi=400)
ax = plt.axes()

x_tick = 15
x_min = 0
x_max = 90

theta_i_space = np.linspace(x_min, x_max, 1000)
power_s_space = power_s(np.radians(theta_i_space), n_21_s_opt[0], power_s_0_opt[0], noise_s_opt[0])

y_tick = 200
y_min = np.floor(np.amin(power_s_space) / y_tick) * y_tick
y_max = np.ceil(np.amax(power_s_space) / y_tick) * y_tick

plt.plot(
    theta_i_space,
    power_s_space,
    c=COLORS[0], linewidth=3,
    label=f"Оптимизированная модель $P_s(θ_i)$"
)

plt.errorbar(
    theta_i_s_data,
    power_s_data,
    xerr=theta_i_s_err,
    yerr=power_s_err,
    color=COLORS[1], marker='s', markersize=6, linewidth=0,
    ecolor=COLORS[4], elinewidth=2,
    label=f"Экспериментальные точки"
)

plt.legend(fontsize=18)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(np.arange(0, x_max + x_tick / 20, x_tick), fontsize=16)
plt.yticks(np.arange(y_min, y_max + y_tick / 20, y_tick), fontsize=16)
ax.xaxis.set_minor_locator(plt.MultipleLocator(x_tick / 3))
ax.yaxis.set_minor_locator(plt.MultipleLocator(y_tick / 4))

ax.grid(which="major", c="#696969", linestyle="-", linewidth=1, alpha=0.6)
ax.grid(which="minor", c="#696969", linestyle="--", linewidth=0.5, alpha=0.6)

plt.title(f"Зависимость мощности отражённого s-пол. света от угла падения $P_s(θ_i)$",
          fontsize=22, pad=22)
plt.xlabel("Угол падения света $θ_i$, °",
           fontsize=20, labelpad=10)
plt.ylabel("Мощность отражённого света $P_s$, мкВт",
           fontsize=20, labelpad=10)

plt.savefig("polarization_pics/power-theta_i_s.png")

theta_i_p_data = np.array(
    [
        15, 30, 45, 50, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 66, 67, 68, 69, 70, 80
    ],
    dtype=np.float64
)

theta_i_p_err = np.full_like(theta_i_p_data, 1.0)

power_p_data = np.array(
    [
        127, 103.5, 57.3, 39.1, 21.38, 19.25, 16.78, 13.94, 11.44, 9.91, 7.97,
        6.86, 6.55, 6.71, 7.47, 9.66, 13.14, 14.36, 19.19, 24.66, 236.1
    ],
    dtype=np.float64
)

power_p_err = power_p_data * 0.011

parameters_p_0 = np.array([2.0, 1000.0, 6.5],
                          dtype=np.float64)

result_p = opt.curve_fit(power_p,
                         np.radians(theta_i_p_data),
                         power_p_data,
                         p0=parameters_p_0,
                         sigma=power_p_err,
                         absolute_sigma=False,
                         jac=power_p_jacobi)

result_err_p = np.sqrt(np.diag(result_p[1]))

n_21_p_opt = np.array([result_p[0][0], result_err_p[0]])
power_p_0_opt = np.array([result_p[0][1], result_err_p[1]])
noise_p_opt = np.array([result_p[0][2], result_err_p[2]])
brewster_angle = opt.minimize(fun=power_p,
                              x0=np.radians(63.0),
                              args=(n_21_p_opt[0], power_p_0_opt[0], noise_p_opt[0]))

print(f"n_21_p: {n_21_p_opt[0]}±{n_21_p_opt[1]}\n"
      f"power_p_0: {power_p_0_opt[0]}±{power_p_0_opt[1]}\n"
      f"noise_p: {noise_p_opt[0]}±{noise_p_opt[1]}\n"
      f"brewster's angle: {np.degrees(brewster_angle.x[0])}\n")

plt.figure(figsize=(16, 10), dpi=400)
ax = plt.axes()

x_tick = 15
x_min = 0
x_max = 90

power_p_space = power_p(np.radians(theta_i_space), n_21_p_opt[0], power_p_0_opt[0], noise_p_opt[0])

y_tick = 400
y_min = np.floor(np.amin(power_p_space) / y_tick) * y_tick
y_max = np.ceil(np.amax(power_p_space) / y_tick) * y_tick

plt.plot(
    theta_i_space,
    power_p_space,
    c=COLORS[2], linewidth=3,
    label=f"Оптимизированная модель $P_p(θ_i)$"
)

plt.errorbar(
    theta_i_p_data,
    power_p_data,
    xerr=theta_i_p_err,
    yerr=power_p_err,
    color=COLORS[3], marker='s', markersize=6, linewidth=0,
    ecolor=COLORS[4], elinewidth=2,
    label=f"Экспериментальные точки"
)

plt.plot(
    [np.degrees(brewster_angle.x[0])] * 2,
    [y_min, y_max],
    c=COLORS[2], linewidth=2, linestyle="--",
    label=f"Угол Брюстера $θ_B = {np.degrees(brewster_angle.x[0]):.1f}°$"
)

plt.legend(fontsize=18)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(np.arange(0, x_max + x_tick / 20, x_tick), fontsize=16)
plt.yticks(np.arange(y_min, y_max + y_tick / 20, y_tick), fontsize=16)
ax.xaxis.set_minor_locator(plt.MultipleLocator(x_tick / 3))
ax.yaxis.set_minor_locator(plt.MultipleLocator(y_tick / 4))

ax.grid(which="major", c="#696969", linestyle="-", linewidth=1, alpha=0.6)
ax.grid(which="minor", c="#696969", linestyle="--", linewidth=0.5, alpha=0.6)

plt.title(f"Зависимость мощности отражённого p-пол. света от угла падения $P_p(θ_i)$",
          fontsize=22, pad=22)
plt.xlabel("Угол падения света $θ_i$, °",
           fontsize=20, labelpad=10)
plt.ylabel("Мощность отражённого света $P_p$, мкВт",
           fontsize=20, labelpad=10)

plt.savefig("polarization_pics/power-theta_i_p.png")

R_s_data = (power_s_data - noise_s_opt[0]) / power_s_0_opt[0]
R_p_data = (power_p_data - noise_p_opt[0]) / power_p_0_opt[0]

plt.figure(figsize=(16, 10), dpi=400)
ax = plt.axes()

x_tick = 15
x_min = 0
x_max = 90

R_s_space = (power_s_space - noise_s_opt[0]) / power_s_0_opt[0]
R_p_space = (power_p_space - noise_p_opt[0]) / power_p_0_opt[0]

y_tick = 0.25
y_min = 0
y_max = 1

plt.plot(
    theta_i_space,
    R_s_space,
    c=COLORS[0], linewidth=3,
    label=f"Коэффициент отражения $R_s(θ_i)$\n"
          f"Оценка $n_{{21s}} = {n_21_s_opt[0]:.2f}±{np.ceil(n_21_s_opt[1] * 100.) / 100.:.2f}$"
)

plt.plot(
    theta_i_s_data,
    R_s_data,
    color=COLORS[1], marker='o', markersize=6, linewidth=0
)

plt.plot(
    theta_i_space,
    R_p_space,
    c=COLORS[2], linewidth=3,
    label=f"Коэффициент отражения $R_p(θ_i)$\n"
          f"Оценка $n_{{21p}} = {n_21_p_opt[0]:.2f}±{np.ceil(n_21_p_opt[1] * 100.) / 100.:.2f}$"
)

plt.plot(
    theta_i_p_data,
    R_p_data,
    color=COLORS[3], marker='o', markersize=6, linewidth=0
)

plt.plot(
    [np.degrees(brewster_angle.x[0])] * 2,
    [y_min, y_max],
    c=COLORS[2], linewidth=2, linestyle="--",
    label=f"Угол Брюстера $θ_B = {np.degrees(brewster_angle.x[0]):.1f}°$"
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

plt.title(f"Зависимость коэффициента отражения s- и p- света от угла падения $R(θ_i)$",
          fontsize=22, pad=22)
plt.xlabel("Угол падения света $θ_i$, °",
           fontsize=20, labelpad=10)
plt.ylabel("Коэффициент отражения $R$",
           fontsize=20, labelpad=10)

plt.savefig("polarization_pics/R-theta_i.png")
