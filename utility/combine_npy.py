import pandas as pd
import numpy as np
import os

input_paths = ["/shares/perception-working/minh/vox2/vox2_npy_3fps_au",\
               "/shares/perception-working/minh/vox2/vox2_npy_3fps_gp"]

output_path = "/shares/perception-working/minh/vox2/vox2_npy_3fps_augp"

for file in os.listdir(input_paths[0]):
    data1 = np.load(os.path.join(input_paths[0], file))
    data2 = np.load(os.path.join(input_paths[1], file))
    data_combine = np.concatenate((data1, data2), axis=1)
    path_out = os.path.join(output_path, file)
    np.save(path_out, data_combine)