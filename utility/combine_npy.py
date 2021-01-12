import pandas as pd
import numpy as np
import os
import sys

mode = "table"
if(mode == "create"):
    input_paths = ["/shares/perception-working/minh/vox2/vox2_npy_3fps_au",\
                   "/shares/perception-working/minh/vox2/vox2_npy_3fps_gp"]
    
    output_path = "/shares/perception-working/minh/vox2/vox2_npy_3fps_augp"
    
    for file in os.listdir(input_paths[0]):
        data1 = np.load(os.path.join(input_paths[0], file))
        data2 = np.load(os.path.join(input_paths[1], file))
        data_combine = np.concatenate((data1, data2), axis=1)
        path_out = os.path.join(output_path, file)
        np.save(path_out, data_combine)
else:
    table_file = sys.argv[1]
    output_file = sys.argv[2]
    df = pd.read_csv(table_file)
    header = list(df.columns)
    data = df.values
    output = []
    for i in range(data.shape[0]):
        current_row = data[i]
        current_path = current_row[1]
        current_path = current_path.replace('vox2_npy_3fps_gp', 'vox2_npy_3fps_augp')
        current_length = current_row[2]
        try:
            data = np.load(current_path)
        except:
            continue
        output.append([current_path, current_length, 'None'])
    pd.DataFrame(output).to_csv(output_file, header=header[1:])