import pandas as pd
import sys
import mmsdk
from mmsdk import mmdatasdk
import numpy as np
import csv

openface_path = "/shares/perception-working/minh/openface_mosi/"
dataset=mmdatasdk.mmdataset(openface_path)

#dataset.keys() = 'OpenFace_2' and 'Opinion Segment Labels'
#user id = dataset['OpenFace_2].keys()
#dataset['Opinion Segment Labels'][user_id][intervals][i] ~ [features][i]

