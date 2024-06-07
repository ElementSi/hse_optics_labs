import numpy as np
import scipy as sp
import glob
import os
import matplotlib.pyplot as plt

SENSOR_RES = 0.0056  # mm/pixel
WAVELENGTH = 0.000532  # mm - green laser
MODEL_SCALE_FACTOR = 10
INTENSITY_THRESHOLD = 210


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


def one_slit_intensity(pix, f_bounds, res, lmbd, r0, b, x0, intens0):
    alpha = np.pi * (b / lmbd) * ((pix * res - x0) / r0)

    intens = intens0 * (np.sin(alpha) / alpha) ** 2

    intens[f_bounds[0]: f_bounds[1]] = 0

    return intens


def one_slit_intensity_jac(pix, f_bounds, res, lmbd, r0, b, x0, intens0):
    alpha_b_der = (np.pi / lmbd) * ((pix * res - x0) / r0)
    alpha_x0_der = - (np.pi * b) / (lmbd * r0)

    alpha = b * alpha_b_der
    sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)

    beta = np.empty_like(alpha)
    for i in range(sin_alpha.size):
        if alpha[i] != 0:
            beta[i] = sin_alpha[i] / alpha[i]
        else:
            beta[i] = 1.0

    grad = np.array(
        [
            2 * beta * alpha_b_der * (cos_alpha * alpha - sin_alpha) / (alpha ** 2),
            2 * beta * alpha_x0_der * (cos_alpha * alpha - sin_alpha) / (alpha ** 2),
            beta ** 2
        ]
    )

    grad[:, f_bounds[0]: f_bounds[1]] = 0

    return grad


def loss_function(expected_x, x):
    return np.sum((expected_x - x) ** 2)


def loss_jac(expected_x, x, jac):
    return 2 * np.sum(expected_x - x) * np.sum(jac, axis=1)


file_pattern = os.path.join(os.path.dirname(__file__), '..', 'data', 'green_one-slit_series', '* mm.txt')
file_list = sorted(glob.glob(file_pattern), key=lambda x: int(os.path.basename(x).split()[0]))

num_files = len(file_list)
num_cols = 4
num_rows = (num_files + num_cols - 1) // num_cols

sample_params = []

plt.figure(figsize=(20, num_rows * 5))

for i, file_name in enumerate(file_list):
    data = np.loadtxt(file_name, delimiter='\t', dtype=np.float64)
    intensity = data[:, 3]

    pixel_id = np.arange(len(intensity))

    high_intensity_indices = np.where(intensity > INTENSITY_THRESHOLD)[0]

    high_intensity_indices = delete_excess(high_intensity_indices)
    zero_bounds = np.array([high_intensity_indices[0], high_intensity_indices[-1]])

    intensity[high_intensity_indices] = 0

    r_0 = float(os.path.basename(file_name).split()[0])

    params = sp.optimize.minimize(
        fun=lambda x:
        loss_function(
            one_slit_intensity(
                pixel_id,
                zero_bounds,
                SENSOR_RES,
                WAVELENGTH,
                r_0,
                *x
            ),
            intensity
        ),
        x0=np.array(
            [
                0.1,
                400 * SENSOR_RES,
                1000
            ]
        ),
        jac=lambda x:
        loss_jac(
            one_slit_intensity(
                pixel_id,
                zero_bounds,
                SENSOR_RES,
                WAVELENGTH,
                r_0,
                *x
            ),
            intensity,
            one_slit_intensity_jac(
                pixel_id,
                zero_bounds,
                SENSOR_RES,
                WAVELENGTH,
                r_0,
                *x
            )
        )
    ).x

    sample_params.append(params)

    pixel_space = np.linspace(pixel_id[0], pixel_id[-1], pixel_id.size * MODEL_SCALE_FACTOR)

    plt.subplot(num_rows, num_cols, i + 1)
    plt.scatter(pixel_id, intensity)
    plt.plot(pixel_space,
             one_slit_intensity(pixel_space, zero_bounds * MODEL_SCALE_FACTOR, SENSOR_RES, WAVELENGTH, r_0, *params),
             color='r')
    plt.title(f'{os.path.basename(file_name)}')
    plt.grid(True)

plt.tight_layout()

plt.show()

print(sample_params)
