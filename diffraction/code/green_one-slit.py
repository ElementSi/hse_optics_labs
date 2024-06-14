import numpy as np
import scipy as sp
import glob
import os
import matplotlib.pyplot as plt

import diffractoin_common as dc

MODEL_SCALE_FACTOR = 10
INTENSITY_THRESHOLD = 210

INIT_ESTIMATES = np.array(
    [
        [0.191] * 20,
        [305, 380, 385, 370, 375, 410, 415, 415, 370, 370, 380, 350, 380, 490, 530, 530, 520, 550, 550, 550],
        [1.6e4, 5587, 3394, 2493, 1427, 1197, 813, 879, 593, 498, 550, 400, 450, 1735, 2163, 2659, 2598, 2379, 2586,
         4793],
        [8, 5, 2, 2, 1, 2, 5, 2, 3, 3, 3, 8, 5, 3, 3, 11, 11, 30, 42, 52]
    ],
    dtype=np.float64
)

file_pattern = os.path.join(os.path.dirname(__file__), '..', 'data', 'green_one-slit_series', '* mm.txt')
file_list = sorted(glob.glob(file_pattern), key=lambda x: int(os.path.basename(x).split()[0]))

num_files = len(file_list)
num_cols = 4
num_rows = (num_files + num_cols - 1) // num_cols

plt.figure(figsize=(48, num_rows * 12))

b_vals = []

for file_id, file_name in enumerate(file_list):
    data = np.loadtxt(file_name, delimiter='\t', dtype=np.float64)
    intensity = data[:, 3]

    pixel_id = np.arange(intensity.size)

    high_intensity_indices = np.where(intensity > INTENSITY_THRESHOLD)[0]

    high_intensity_indices = dc.delete_excess(high_intensity_indices)
    zero_bounds = np.array([high_intensity_indices[0], high_intensity_indices[-1] + 1])

    intensity[high_intensity_indices] = 0

    R0 = float(os.path.basename(file_name).split()[0])

    params = sp.optimize.minimize(
        fun=lambda x:
        dc.loss_function(
            dc.one_slit_intensity(
                pixel_id,
                zero_bounds,
                dc.SENSOR_RES,
                dc.G_WAVELENGTH,
                R0,
                INTENSITY_THRESHOLD,
                *x
            ),
            intensity
        ),
        x0=np.array(
            [
                INIT_ESTIMATES[0][file_id],
                INIT_ESTIMATES[1][file_id] * dc.SENSOR_RES,
                INIT_ESTIMATES[2][file_id],
                INIT_ESTIMATES[3][file_id]
            ]
        ),
        jac=lambda x:
        dc.loss_jac(
            dc.one_slit_intensity(
                pixel_id,
                zero_bounds,
                dc.SENSOR_RES,
                dc.G_WAVELENGTH,
                R0,
                INTENSITY_THRESHOLD,
                *x
            ),
            intensity,
            dc.one_slit_intensity_jac(
                pixel_id,
                zero_bounds,
                dc.SENSOR_RES,
                dc.G_WAVELENGTH,
                R0,
                INTENSITY_THRESHOLD,
                *x
            )
        ),
        bounds=np.array(
            [(0.180, 0.198), (0, 640), (0, None), (0, INIT_ESTIMATES[3][file_id] + 5)]
        )
    ).x

    plt.subplot(num_rows, num_cols, file_id + 1)

    intensity_error = dc.filter_large_v(np.full_like(intensity, params[2] / 10 ** (dc.SN_RATIO / 20)),
                                        intensity,
                                        zero_bounds,
                                        INTENSITY_THRESHOLD)

    plt.errorbar(
        pixel_id,
        intensity,
        yerr=intensity_error,
        fmt='none',
        ecolor=dc.COLORS[2],
        alpha=0.6,
        elinewidth=0.6,
        zorder=0,
    )

    plt.scatter(
        pixel_id,
        intensity,
        s=32,
        color=dc.COLORS[0],
        zorder=1,
    )

    pixel_space = np.linspace(pixel_id[0], pixel_id[-1], pixel_id.size * MODEL_SCALE_FACTOR, endpoint=False)
    scaled_zero_bounds = zero_bounds * MODEL_SCALE_FACTOR

    plt.plot(
        pixel_space[:scaled_zero_bounds[0]],
        dc.one_slit_intensity(pixel_space,
                              scaled_zero_bounds,
                              dc.SENSOR_RES,
                              dc.G_WAVELENGTH,
                              R0,
                              INTENSITY_THRESHOLD,
                              *params
                              )[:scaled_zero_bounds[0]],
        linewidth=4,
        color=dc.COLORS[1],
        alpha=0.8,
        label=fr"$I_{{{file_id + 1}}}$ = $I($"
              f"$b$ = {params[0]:.3f}, "
              f"$p_0$ = {params[1] / dc.SENSOR_RES:.1f}, "
              f"$I_0$ = {params[2]:.0f}, "
              f"$I_n$ = {params[3]:.0f}"
              f"$)$",
        zorder=2,
    )

    plt.plot(
        pixel_space[scaled_zero_bounds[1]:],
        dc.one_slit_intensity(pixel_space,
                              scaled_zero_bounds,
                              dc.SENSOR_RES,
                              dc.G_WAVELENGTH,
                              R0,
                              INTENSITY_THRESHOLD,
                              *params
                              )[scaled_zero_bounds[1]:],
        linewidth=4,
        color=dc.COLORS[1],
        alpha=0.8,
        zorder=2,
    )

    plt.plot(
        np.array([zero_bounds[0] + 1, zero_bounds[1] - 1]),
        np.array([INTENSITY_THRESHOLD - 0.5] * 2),
        linewidth=4,
        linestyle='--',
        color=dc.COLORS[1],
        alpha=0.8,
        zorder=2,
    )

    plt.title(f'$R_0$ = {float(os.path.basename(file_name).split()[0]) + dc.DIST_0} mm', fontsize=30, pad=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.xticks(np.arange(0, 640, 100), fontsize=20)
    plt.yticks(np.arange(0, 256 + 1, 32), fontsize=20)
    plt.xlim(0, 640)
    plt.ylim(0, 256)
    plt.grid(True)

    b_vals.append(params[0])

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.2, wspace=0.2)

plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'pics', 'green_one-slit_series_plots.png'))

print(f"b = ({np.mean(b_vals):.4f}±{np.sqrt(np.var(b_vals)):.4f})mm")
