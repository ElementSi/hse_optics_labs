import numpy as np

COLORS = [
    "#80D04B",  # green laser (532nm)
    "#235299",  # cyan-blue (green laser triadic 1)
    "#D04B80",  # mystic (green laser triadic 2)
]

SENSOR_RES = 0.0056  # mm/pixel
SN_RATIO = 45.0  # dB -
G_WAVELENGTH = 0.000532  # mm - green laser
DIST_0 = 56  # mm - dist from z0 to screen


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
                end_index = i
            current_length = 1
            current_start = i

    if current_length > max_length:
        start_index = current_start
        end_index = len(array)

    return array[start_index: end_index]


def filter_large_v(array, sample, bounds, threshold):
    array[bounds[0]: bounds[1]] = 0
    temp_array = array.copy()
    temp_array[sample > threshold] = 0
    array[sample > threshold] = max(temp_array)

    return array


def one_slit_intensity(pix, f_bounds, res, lmbd, r0, threshold, b, x0, intens0, noise):
    alpha = np.pi * (b / lmbd) * ((pix * res - x0) / (r0 + DIST_0))

    with np.errstate(divide='ignore', invalid='ignore'):
        sin_alpha = np.sin(alpha)
        alpha_safe = np.where(alpha == 0, np.nan, alpha)
        sinc_alpha = np.where(alpha == 0, 1, sin_alpha / alpha_safe)
        sinc_alpha_squared = sinc_alpha ** 2

    intens = intens0 * sinc_alpha_squared + noise

    intens = filter_large_v(intens, intens, f_bounds, threshold)

    return intens


def one_slit_intensity_jac(pix, f_bounds, res, lmbd, r0, threshold, b, x0, intens0, noise):
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

    d_intensity_db = filter_large_v(d_intensity_db, intens, f_bounds, threshold)
    d_intensity_dx0 = filter_large_v(d_intensity_dx0, intens, f_bounds, threshold)
    d_intensity_dintens0 = filter_large_v(d_intensity_dintens0, intens, f_bounds, threshold)
    d_intensity_dnoise = filter_large_v(d_intensity_dnoise, intens, f_bounds, threshold)

    return np.array([d_intensity_db, d_intensity_dx0, d_intensity_dintens0, d_intensity_dnoise])


def loss_function(x, expected_x):
    return 1 / 2 * np.sum((x - expected_x) ** 2)


def loss_jac(x, expected_x, jac):
    return np.sum((x - expected_x) * jac, axis=1)
