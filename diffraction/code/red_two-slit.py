import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt

import diffractoin_common as dc

INTENSITY_THRESHOLD = 224
R0_1585mm = 1585
R0_290mm = 290
MODEL_SCALE_FACTOR = 100

file_name_red_1585mm = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'two-slit',
    'red_two_big_slit_1585mm.txt'
)

data_1585mm = np.loadtxt(file_name_red_1585mm, delimiter='\t', dtype=np.float64)
intensity_red_1585mm = data_1585mm[:, 3]
rgb_red_1585mm = data_1585mm[:, 0:3].astype(dtype=np.int64)

low_intensity_indices_1585mm = np.where(intensity_red_1585mm <= INTENSITY_THRESHOLD)[0]

plt.figure(figsize=(12, 12))

plt.scatter(dc.pixel_id[low_intensity_indices_1585mm],
            intensity_red_1585mm[low_intensity_indices_1585mm],
            color=dc.COLORS[3],
            s=32,
            zorder=1)

params_1585mm = sp.optimize.minimize(
    fun=lambda x:
    dc.loss_function(
        dc.two_slit_intensity(
            dc.pixel_id[low_intensity_indices_1585mm],
            dc.SENSOR_RES,
            dc.R_WAVELENGTH,
            R0_1585mm,
            *x
        ),
        intensity_red_1585mm[low_intensity_indices_1585mm]
    ),
    x0=np.array(
        [
            0.1,
            2,
            330 * dc.SENSOR_RES,
            300,
            20
        ]
    ),
    bounds=np.array(
        [(0.1, 0.3), (0.15, 2.0), (300 * dc.SENSOR_RES, 400 * dc.SENSOR_RES), (0, None), (0, 50)]
    )
).x

pixel_space = np.linspace(dc.pixel_id[0], dc.pixel_id[-1], dc.pixel_id.size * MODEL_SCALE_FACTOR, endpoint=False)
low_intensity_space_id_1585mm = np.where(dc.two_slit_intensity(pixel_space,
                                                        dc.SENSOR_RES,
                                                        dc.R_WAVELENGTH,
                                                        R0_1585mm,
                                                        *params_1585mm
                                                        ) <= INTENSITY_THRESHOLD)[0]

intensity_error_1585mm = params_1585mm[3] / 10 ** (dc.SN_RATIO / 20)

plt.errorbar(
    dc.pixel_id[low_intensity_indices_1585mm],
    intensity_red_1585mm[low_intensity_indices_1585mm],
    yerr=intensity_error_1585mm,
    fmt='none',
    ecolor=dc.COLORS[2],
    alpha=0.6,
    elinewidth=0.6,
    zorder=0,
)

plt.plot(
    pixel_space[low_intensity_space_id_1585mm],
    dc.two_slit_intensity(pixel_space[low_intensity_space_id_1585mm],
                          dc.SENSOR_RES,
                          dc.R_WAVELENGTH,
                          R0_1585mm,
                          *params_1585mm
                          ),
    linewidth=4,
    color=dc.COLORS[1],
    alpha=0.8,
    label=r"$I_{2slit}$ = $I($"
          f"$b$ = {params_1585mm[0]:.3f}, "
          f"$d$ = {params_1585mm[1]:.3f}, "
          f"$p_0$ = {params_1585mm[2] / dc.SENSOR_RES:.1f}, "
          f"$I_0$ = {params_1585mm[3]:.0f}, "
          f"$I_n$ = {params_1585mm[4]:.0f}"
          f"$)$",
    zorder=2,
)

plt.title(f'$R_0$ = 1585 mm', fontsize=30, pad=20)
plt.legend(loc='upper right', fontsize=20)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
plt.xticks(np.arange(0, 640, 100), fontsize=20)
plt.yticks(np.arange(0, 256 + 1, 32), fontsize=20)
plt.xlim(0, 640)
plt.ylim(0, 256)
plt.grid(True)

plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'pics', 'red_two-slit_1585mm_plot.png'))

fig = plt.figure(figsize=(12, 12))
plt.title(f'$R_0$ = {R0_1585mm + dc.DIST_0} mm', fontsize=30, pad=20)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
ax = fig.add_subplot(111)
ax.tick_params(axis='x', labelsize=20)
ax.yaxis.set_visible(False)
ax.imshow(np.broadcast_to(rgb_red_1585mm, (640, 640, 3)), aspect='auto', origin='lower')

plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'pics', 'red_two-slit_1585mm_img.png'))

file_name_red_290mm = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'two-slit',
    'red_two-slit_290mm.txt'
)

data_290mm = np.loadtxt(file_name_red_290mm, delimiter='\t', dtype=np.float64)
intensity_red_290mm = data_290mm[:, 3]
rgb_red_290mm = data_290mm[:, 0:3].astype(dtype=np.int64)

low_intensity_indices_290mm = np.where(intensity_red_290mm <= INTENSITY_THRESHOLD)[0]

params_290mm = sp.optimize.minimize(
    fun=lambda x:
    dc.loss_function(
        dc.two_slit_intensity(
            dc.pixel_id[low_intensity_indices_290mm],
            dc.SENSOR_RES,
            dc.R_WAVELENGTH,
            R0_290mm,
            *x
        ),
        intensity_red_290mm[low_intensity_indices_290mm]
    ),
    x0=np.array(
        [
            0.17,
            0.96,
            300 * dc.SENSOR_RES,
            1000,
            10
        ]
    ),
    bounds=np.array(
        [(0.16, 0.2), (0.80, 1.0), (290 * dc.SENSOR_RES, 310 * dc.SENSOR_RES), (800, None), (0, 16)]
    )
).x

low_intensity_space_id_290mm = np.where(dc.two_slit_intensity(pixel_space,
                                                        dc.SENSOR_RES,
                                                        dc.R_WAVELENGTH,
                                                        R0_290mm,
                                                        *params_290mm
                                                        ) <= INTENSITY_THRESHOLD)[0]

intensity_error_290mm = params_290mm[3] / 10 ** (dc.SN_RATIO / 20)

plt.figure(figsize=(12, 12))

plt.scatter(dc.pixel_id[low_intensity_indices_290mm],
            intensity_red_290mm[low_intensity_indices_290mm],
            color=dc.COLORS[3],
            s=32,
            zorder=1)

plt.errorbar(
    dc.pixel_id[low_intensity_indices_290mm],
    intensity_red_290mm[low_intensity_indices_290mm],
    yerr=intensity_error_290mm,
    fmt='none',
    ecolor=dc.COLORS[2],
    alpha=0.6,
    elinewidth=0.6,
    zorder=0,
)

plt.plot(
    pixel_space[low_intensity_space_id_290mm],
    dc.two_slit_intensity(pixel_space[low_intensity_space_id_290mm],
                          dc.SENSOR_RES,
                          dc.R_WAVELENGTH,
                          R0_290mm,
                          *params_290mm
                          ),
    linewidth=4,
    color=dc.COLORS[1],
    alpha=0.8,
    label=r"$I_{2slit}$ = $I($"
          f"$b$ = {params_290mm[0]:.3f}, "
          f"$d$ = {params_290mm[1]:.3f}, "
          f"$p_0$ = {params_290mm[2] / dc.SENSOR_RES:.1f}, "
          f"$I_0$ = {params_290mm[3]:.0f}, "
          f"$I_n$ = {params_290mm[4]:.0f}"
          f"$)$",
    zorder=2,
)

plt.title(f'$R_0$ = {R0_290mm + dc.DIST_0} mm', fontsize=30, pad=20)
plt.legend(loc='upper right', fontsize=20)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
plt.xticks(np.arange(0, 640, 100), fontsize=20)
plt.yticks(np.arange(0, 256 + 1, 32), fontsize=20)
plt.xlim(0, 640)
plt.ylim(0, 256)
plt.grid(True)

plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'pics', 'red_two-slit_290mm_plot.png'))

fig = plt.figure(figsize=(12, 12))
plt.title(f'$R_0$ = 290 mm', fontsize=30, pad=20)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
ax = fig.add_subplot(111)
ax.tick_params(axis='x', labelsize=20)
ax.yaxis.set_visible(False)
ax.imshow(np.broadcast_to(rgb_red_290mm, (640, 640, 3)), aspect='auto', origin='lower')

plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'pics', 'red_two-slit_290mm_img.png'))
