import os
import sys

#--input_path ../data/openface_vox_sample/ --output_path ../data/openface_npy/ --chunk 3 --n_chunk 16
input_path = "/shares/perception-working/minh/openface_voxceleb2/"
output_path = "/shares/perception-working/minh/vox2/vox2_npy/"
#input_path = "../data/openface_vox_sample/"
#output_path = "../data/openface_npy/"
window_size = 5
step_size = 2
downsamping_rate = 1

current_chunk = sys.argv[2]
n_chunk = sys.argv[1]

cmd = "taskset --cpu-list " + current_chunk + \
    " python preprocess_openface.py --input_path " +input_path + \
    " --output_path " + output_path + " --chunk " + current_chunk + " --n_chunk " + n_chunk + \
    " --dr " + str(downsamping_rate) + " --window_size " + str(window_size) + " --step_size " + str(step_size) 
os.system(cmd) 
print("Finish chunk ", current_chunk)
