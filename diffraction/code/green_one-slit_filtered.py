import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt

import diffractoin_common as dc

R0 = -4.4
INTENSITY_THRESHOLD = 257
MODEL_SCALE_FACTOR = 10
pixel_id = np.arange(640)

file_name_3g = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'green_one-slit_filtered',
    'green_one-slit_3glasses.txt'
)

data_3g = np.loadtxt(file_name_3g, delimiter='\t', dtype=np.float64)
intensity_3g = data_3g[:, 3]
rgb_3g = data_3g[:, 0:3].astype(dtype=np.int64)

zero_bounds = np.zeros(2, dtype=np.int64)

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
        intensity_3g
    ),
    x0=np.array(
        [
            0.1906,
            303 * dc.SENSOR_RES,
            3427,
            2.5
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
        intensity_3g,
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
        [(0.180, 0.198), (0, 640), (0, None), (0, 100)]
    )
).x

print(params)

plt.figure(figsize=(12, 12))

intensity_error = dc.filter_large_v(np.full_like(intensity_3g, params[2] / 10 ** (dc.SN_RATIO / 20)),
                                    intensity_3g,
                                    zero_bounds,
                                    INTENSITY_THRESHOLD)

plt.errorbar(
    pixel_id,
    intensity_3g,
    yerr=intensity_error,
    fmt='none',
    ecolor=dc.COLORS[2],
    alpha=0.6,
    elinewidth=0.6,
    zorder=0,
)

plt.scatter(pixel_id,
            intensity_3g,
            color=dc.COLORS[0],
            s=32,
            zorder=1)

pixel_space = np.linspace(pixel_id[0], pixel_id[-1], pixel_id.size * MODEL_SCALE_FACTOR, endpoint=False)
plt.plot(pixel_space,
         dc.one_slit_intensity(pixel_space,
                               zero_bounds,
                               dc.SENSOR_RES,
                               dc.G_WAVELENGTH,
                               R0,
                               INTENSITY_THRESHOLD,
                               *params
                               ),
         linewidth=4,
         color=dc.COLORS[1],
         alpha=0.8,
         zorder=2,
         )

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
plt.xticks(np.arange(0, 640, 100), fontsize=20)
plt.yticks(np.arange(0, 256 + 1, 32), fontsize=20)
plt.xlim(0, 640)
plt.ylim(0, 256)
plt.grid(True)

plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'pics', 'green_one-slit_filtered_plot.png'))

fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
ax = fig.add_subplot(111)
ax.tick_params(axis='x', labelsize=20)
ax.yaxis.set_visible(False)
cax = ax.imshow(np.broadcast_to(rgb_3g, (640, 640, 3)), aspect='auto', origin='lower')

plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'pics', 'green_one-slit_filtered_img.png'))
