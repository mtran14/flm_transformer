import pandas as pd
import numpy as np
import os

#process at 5fps so ::5
#299:316
input_dir = "/shares/perception-working/minh/openface_voxceleb2_flm_comp/"
output_path = "/shares/perception-working/minh/all_aus.csv"
output = None
for file in os.listdir(input_dir):
    current_data = pd.read_csv(os.path.join(input_dir, file)).values[::5]
    current_aus = current_data[:,299:316]
    if(output is None):
        output = current_aus
    else:
        output = np.concatenate((output, current_aus),axis=0)
pd.DataFrame(output).to_csv(output_path, header=None, index=False)
print("Done merging files")
