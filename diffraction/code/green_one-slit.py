import numpy as np
import scipy as sp
import glob
import os
import matplotlib.pyplot as plt

SENSOR_RES = 0.0056  # mm/pixel
WAVELENGTH = 0.000532  # mm - green laser
MODEL_SCALE_FACTOR = 10
INTENSITY_THRESHOLD = 210
DIST_0 = 56

INIT_ESTIMATES = np.array(
    [
        [0.19] * 20,
        [305, 380, 385, 370, 375, 410, 415, 415, 370, 370, 380, 350, 380, 490, 530, 530, 520, 550, 550, 550],
        [16000, 6000, 3400, 2500, 1600, 1200, 1000, 1000, 600, 500, 550, 400, 450, 1800, 2200, 2800, 2600, 2400, 2650, 4800],
        [8, 5, 2, 2, 2, 2, 5, 5, 5, 5, 5, 8, 5, 5, 8, 10, 10, 30, 35, 50]
    ],
    dtype=np.float64
)

SCALE = np.array(
    [
        0.1,
        1,
        1000,
        10
    ]
)


def delete_excess(array):
    max_length = 0
    start_index = 0
    end_index = 0

    current_length = 1
    current_start = 0

    for i in range(1, len(array)):
        if array[i] == array[i - 1] + 1:
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                start_index = current_start
                end_index = i - 1
            current_length = 1
            current_start = i

    if current_length > max_length:
        start_index = current_start
        end_index = len(array) - 1

    return array[start_index: end_index + 1]


def filter_large_v(array, sample, bounds, threshold):
    array[bounds[0]: bounds[1]] = 0
    temp_array = array.copy()
    temp_array[sample > threshold] = 0
    array[sample > threshold] = max(temp_array)

    return array


def one_slit_intensity(pix, f_bounds, res, lmbd, r0, b, x0, intens0, noise):

    alpha = np.pi * (b / lmbd) * ((pix * res - x0) / (r0 + DIST_0))

    with np.errstate(divide='ignore', invalid='ignore'):
        sin_alpha = np.sin(alpha)
        alpha_safe = np.where(alpha == 0, np.nan, alpha)
        sinc_alpha = np.where(alpha == 0, 1, sin_alpha / alpha_safe)
        sinc_alpha_squared = sinc_alpha ** 2

    intens = intens0 * sinc_alpha_squared + noise

    intens = filter_large_v(intens, intens, f_bounds, INTENSITY_THRESHOLD)

    return intens


def one_slit_intensity_jac(pix, f_bounds, res, lmbd, r0, b, x0, intens0, noise):
    alpha = np.pi * (b / lmbd) * ((pix * res - x0) / (r0 + DIST_0))

    with np.errstate(divide='ignore', invalid='ignore'):
        sin_alpha = np.sin(alpha)
        alpha_safe = np.where(alpha == 0, np.nan, alpha)
        sinc_alpha = np.where(alpha == 0, 1, sin_alpha / alpha_safe)
        sinc_alpha_squared = sinc_alpha ** 2
        cos_alpha = np.cos(alpha)
        d_alpha_db = np.pi * (pix * res - x0) / (lmbd * (r0 + DIST_0))
        d_alpha_dx0 = -np.pi * b / (lmbd * (r0 + DIST_0))

    intens = intens0 * sinc_alpha_squared

    beta = 2 * intens0 * sinc_alpha * (cos_alpha * alpha - sin_alpha) / alpha_safe ** 2

    d_intensity_db = beta * d_alpha_db
    d_intensity_dx0 = beta * d_alpha_dx0
    d_intensity_dintens0 = sinc_alpha_squared
    d_intensity_dnoise = np.ones_like(alpha)

    d_intensity_db = filter_large_v(d_intensity_db, intens, f_bounds, INTENSITY_THRESHOLD)
    d_intensity_dx0 = filter_large_v(d_intensity_dx0, intens, f_bounds, INTENSITY_THRESHOLD)
    d_intensity_dintens0 = filter_large_v(d_intensity_dintens0, intens, f_bounds, INTENSITY_THRESHOLD)
    d_intensity_dnoise = filter_large_v(d_intensity_dnoise, intens, f_bounds, INTENSITY_THRESHOLD)

    return np.array([d_intensity_db, d_intensity_dx0, d_intensity_dintens0, d_intensity_dnoise])


def loss_function(x, expected_x):
    return np.sum((x - expected_x) ** 2)


def loss_jac(x, expected_x, jac):
    grad = 2 * np.sum((x - expected_x) * jac, axis=1)
    grad[0] /= 100000
    grad[1] /= 10000
    grad[3] /= 10
    return grad


file_pattern = os.path.join(os.path.dirname(__file__), '..', 'data', 'green_one-slit_series', '* mm.txt')
file_list = sorted(glob.glob(file_pattern), key=lambda x: int(os.path.basename(x).split()[0]))

num_files = len(file_list)
num_cols = 4
num_rows = (num_files + num_cols - 1) // num_cols

plt.figure(figsize=(24, num_rows * 6))

b_vals = []

for i, file_name in enumerate(file_list):
    data = np.loadtxt(file_name, delimiter='\t', dtype=np.float64)
    intensity = data[:, 3]

    pixel_id = np.arange(len(intensity))

    high_intensity_indices = np.where(intensity > INTENSITY_THRESHOLD)[0]

    high_intensity_indices = delete_excess(high_intensity_indices)
    zero_bounds = np.array([high_intensity_indices[0], high_intensity_indices[-1]])

    intensity[high_intensity_indices] = 0

    r0 = float(os.path.basename(file_name).split()[0])

    params = sp.optimize.minimize(
        fun=lambda x:
        loss_function(
            one_slit_intensity(
                pixel_id,
                zero_bounds,
                SENSOR_RES,
                WAVELENGTH,
                r0,
                *x
            ),
            intensity
        ),
        x0=np.array(
            [
                INIT_ESTIMATES[0][i],
                INIT_ESTIMATES[1][i] * SENSOR_RES,
                INIT_ESTIMATES[2][i],
                INIT_ESTIMATES[3][i]
            ]
        ),
        jac=lambda x:
        loss_jac(
            one_slit_intensity(
                pixel_id,
                zero_bounds,
                SENSOR_RES,
                WAVELENGTH,
                r0,
                *x
            ),
            intensity,
            one_slit_intensity_jac(
                pixel_id,
                zero_bounds,
                SENSOR_RES,
                WAVELENGTH,
                r0,
                *x
            )
        ),
        bounds=np.array(
            [(0.180, 0.192), (0, 640), (0, None), (0, INIT_ESTIMATES[3][i] + 5)]
        )
    ).x

    pixel_space = np.linspace(pixel_id[0], pixel_id[-1], pixel_id.size * MODEL_SCALE_FACTOR)

    plt.subplot(num_rows, num_cols, i + 1)
    plt.scatter(pixel_id, intensity)
    plt.plot(pixel_space,
             one_slit_intensity(pixel_space, zero_bounds * MODEL_SCALE_FACTOR, SENSOR_RES, WAVELENGTH, r0, *params),
             color='r',
             label=f"b = {params[0]:.3f}, pix0 = {params[1] / SENSOR_RES:.2f}, intens0 = {params[2]:.0f}, noise = {params[3]:.0f}")
    plt.title(f'$R_0$ = {float(os.path.basename(file_name).split()[0]) + DIST_0} mm')
    plt.legend(loc='upper right', fontsize=10)
    plt.xlim(0, 640)
    plt.ylim(0, 255)
    plt.grid(True)

    b_vals.append(params[0])

plt.tight_layout()

plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'pics', 'green_one-slit_series_plots.png'))

print(f"b = ({np.mean(b_vals):.6f}±{np.var(b_vals):.6f})mm")
