import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def process_data(data):
    result = np.mean(data)
    return result


file_pattern = os.path.join('../data', 'green_one-slit_series', '* mm.txt')
file_list = sorted(glob.glob(file_pattern), key=lambda x: int(os.path.basename(x).split()[0]))

num_files = len(file_list)
num_cols = 4
num_rows = (num_files + num_cols - 1) // num_cols

plt.figure(figsize=(20, num_rows * 5))

for i, file_name in enumerate(file_list):
    data = np.loadtxt(file_name, delimiter='\t')
    intensity = data[:, 3]
    pixel_id = np.arange(len(intensity))

    plt.subplot(num_rows, num_cols, i + 1)
    plt.scatter(pixel_id, intensity, label=os.path.basename(file_name))
    plt.title(f'{os.path.basename(file_name)}')
    plt.grid(True)

plt.tight_layout()

plt.show()
