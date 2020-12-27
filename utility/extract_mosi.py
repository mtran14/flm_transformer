import pandas as pd
import sys
sys.path.append("/shares/perception-working/minh/CMU-MultimodalSDK")
import mmsdk
from mmsdk import mmdatasdk
import numpy as np
import os

mosi_path = "/shares/perception-working/minh/openface_mosi/"
output_path = "/shares/perception-working/minh/openface_mosi_segmented/"
dataset=mmdatasdk.mmdataset(mosi_path)
num_features = 713
pseudo_header = []
for i in range(num_features):
    pseudo_header.append("ft_" + str(i))
    
all_output = []
for user_id in dataset['Opinion Segment Labels'].keys():
    #query data
    try:
        current_intervals = np.array(dataset['Opinion Segment Labels'][user_id]['intervals'])
        current_scores = np.array(dataset['Opinion Segment Labels'][user_id]['features'])
        current_openface = np.array(dataset['OpenFace_2'][user_id]['features'])
    except:
        continue
    index = 0
    time_stamp_col = current_openface[:,1]
    for i in range(current_intervals.shape[0]):
        interval = current_intervals[i]
        current_score = current_scores[i][0]
        bool_arr = np.logical_and(time_stamp_col >= interval[0], time_stamp_col <= interval[1])
        current_segment = current_openface[bool_arr]
        output_file_name = user_id +  str(index) + ".csv"
        current_output_path = os.path.join(output_path,output_file_name)
        pd.DataFrame(current_segment).to_csv(current_output_path, header=pseudo_header, index=False)  
        all_output.append([output_file_name, current_score])
        index += 1
    print("Done ", user_id)
pd.DataFrame(all_output).to_csv("../data/mosi_segment_scores.csv", header=None, index=False)
print("here")
