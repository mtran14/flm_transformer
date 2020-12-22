import os
import sys
import pandas as pd
import pickle
import argparse
from scipy.optimize import curve_fit
import math
import numpy as np
import matplotlib.pyplot as plt
#############################
# HELPER FUNCTIONS #
#############################
def sliding_window(data, sliding_size, step_size):
    start = 0
    N = data.shape[0]
    output_src = []
    while(start < N):
        if(start+sliding_size <= N): 
            output_src.append(np.arange(start, start + sliding_size))
        start += step_size
    output_src = np.array(output_src)
    return np.mean(data[output_src], axis=1)

def function1(x, A, B): 
    return A*x + B


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def rotate_frame(frame_input):
    new_X = []
    new_Y = []
    origin = (0,0)
    for i in range(68):
        qx, qy = rotate(origin, (frame_input[i],frame_input[i+68]), math.pi)
        new_X.append(qx)
        new_Y.append(qy) 
    
    new_X = np.array(new_X)
    new_X -= new_X.min()
    new_Y = np.array(new_Y)
    new_Y -= new_Y.min()  
    output = list(new_X) + list(new_Y)
    return output

#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():
    
    parser = argparse.ArgumentParser(description='preprocess arguments for any dataset.')
    parser.add_argument('--input_path', default='./data/', type=str, help='Path to store output', required=False)
    parser.add_argument('--output_path', default='./data/', type=str, help='Path to store output', required=False)
    #save for later use including gaze, au, etc
    #parser.add_argument('--feature_type', default='fbank', type=str, help='Feature type ( mfcc / fbank / mel / linear )', required=False)
    parser.add_argument('--window_size', default=5, type=int, help='Window size (frames)', required=False)
    parser.add_argument('--step_size', default=2, type=int, help='Step size (frames)', required=False)
    parser.add_argument('--chunk', default=-1, type=int, help='Decides the chunk to work on', required=True)
    parser.add_argument('--n_chunk', default=-1, type=int, help='Total number of chunks', required=True)
    parser.add_argument('--nose_al', default=1, type=int, help='Align nose or not', required=False)
    parser.add_argument('--dr', default=1, type=int, help='Downsampling rate', required=False)
    args = parser.parse_args()
    return args

#######################
# OPENFACE PREPROCESS #
#######################
def openface_preprocess(args):
    input_path = args.input_path
    file_list = os.listdir(input_path)
    chunk_size = len(file_list) // args.n_chunk
    current_files = file_list[chunk_size * args.chunk : chunk_size * (args.chunk+1)]
    overall_file_outname = "train-clean-schz_chunk_"+str(args.chunk) + ".csv"
    overall_clean_data = []
    for file in current_files:
        file_path = os.path.join(input_path, file)
        data = pd.read_csv(file_path).values
        data = data[::args.dr]
        drops = []
        for k in range(data.shape[0]):
            for element in data[k]:
                if(isinstance(element, str)):
                    drops.append(k)
        data = np.delete(data, drops, axis=0)
        if(data.shape[0] == 0):
            continue
        data = sliding_window(data, args.window_size, args.step_size)
        output = []
        for j in range(data.shape[0]):
            X = []
            y = []
            if(data[j][2] < 0.9): #3
                continue
            if(args.nose_al == 1):
                for i in [27,28,29,30]: #5 and 73
                    x_val_index = 16 + i
                    y_val_index = 84 + i           
                    X.append(data[j][x_val_index])
                    y.append(data[j][y_val_index])
                #try:
                popt, pcov = curve_fit(function1, X ,y)
                x1, y1 = 0 , popt[1]
                x2, y2 = 1 , popt[0] + popt[1]
                rotate_angle = 3*math.pi/2 - math.atan2(y2-y1, x2-x1)
                
                new_X = []
                new_Y = []
                for i in range(68):
                    x_val_index = 16 + i
                    y_val_index = 84 + i  
                    origin = (0,0)
                    qx, qy = rotate(origin, (data[j][x_val_index],data[j][y_val_index]), rotate_angle)
                    new_X.append(qx)
                    new_Y.append(qy)
                
                new_X = np.array(new_X)
                new_Y = np.array(new_Y)
                new_X -= min(new_X)
                new_Y -= min(new_Y)
                
                
                scale_factor = 300 / max(max(new_X), max(new_Y))
                
                new_X = np.array(new_X)*scale_factor
                new_Y = np.array(new_Y)*scale_factor
                
                new_row = list(new_X)+list(new_Y)
                if(new_Y[8] > 150):
                    new_row = rotate_frame(new_row)  
                new_row = np.array(new_row)
                nr_X = new_row[0:68]
                nr_Y = new_row[68:136]
                nr_X -= np.mean(nr_X)
                nr_X /= 300
                nr_Y -= np.mean(nr_Y)
                nr_Y /= 300
                new_row = list(nr_X) + list(nr_Y)
                output.append(new_row)
            else:
                new_X = []
                new_Y = []
                for i in range(68):
                    x_val_index = 16 + i
                    y_val_index = 84 + i
                    new_X.append(data[j][x_val_index])
                    new_Y.append(data[j][y_val_index]) 
                new_X -= min(new_X)
                new_Y -= min(new_Y)
                scale_factor = 300 / max(max(new_X), max(new_Y))
                new_X = np.array(new_X)*scale_factor
                new_Y = np.array(new_Y)*scale_factor
                new_row = list(new_X)+list(new_Y) 
                new_row = np.array(new_row)
                nr_X = new_row[0:68]
                nr_Y = new_row[68:136]
                nr_X -= np.mean(nr_X)
                nr_X /= 300
                nr_Y -= np.mean(nr_Y)
                nr_Y /= 300
                new_row = list(nr_X) + list(nr_Y)
                output.append(new_row) 
        file_out = os.path.join(args.output_path, file.split(".")[0]+".npy")
        output = np.array(output)
        np.save(file_out, output)
        overall_clean_data.append([file_out, output.shape[0]])
    pd.DataFrame(overall_clean_data).to_csv(os.path.join(args.output_path+"/../", overall_file_outname), header=None, index=False)
    print('All done, saved at', args.output_path, 'exit.')


########
# MAIN #
########
def main():

    # get arguments
    args = get_preprocess_args()

    # Preprocessing Data & Make Data Table
    openface_preprocess(args)


if __name__ == '__main__':
    main()