import pandas as pd
import numpy as np
import os
import random

input_path = "/shares/perception-working/minh/vox2/chunk_tables_au/"
concat_table = {}
total_tokens = 0
for file in os.listdir(input_path):
    current_path = os.path.join(input_path, file)
    data = pd.read_csv(current_path, header=None).values
    for row in data:
        if(row[1] >= 5):
            concat_table[row[0]] = row[1]
            if(isinstance(row[1],str)):
                print("here")
            total_tokens += row[1]
       
print("Total tokens: ", total_tokens)
    
test_dict = {}
test_size = 4000
dev_size = 4000
dev_dict = {}
for i in range(dev_size):
    rm_key = random.choice(list(concat_table.keys()))
    rm_val = concat_table.pop(rm_key)
    assert rm_key not in concat_table.keys()
    dev_dict[rm_key] = rm_val
    
for i in range(test_size):
    rm_key = random.choice(list(concat_table.keys()))
    rm_val = concat_table.pop(rm_key)
    assert rm_key not in concat_table.keys()
    test_dict[rm_key] = rm_val  

sorted_dict_train = {k: v for k, v in sorted(concat_table.items(), key=lambda item: item[1], reverse=True)} 
sorted_dict_dev = {k: v for k, v in sorted(dev_dict.items(), key=lambda item: item[1], reverse=True)} 
sorted_dict_test = {k: v for k, v in sorted(test_dict.items(), key=lambda item: item[1], reverse=True)} 

output_train = []
for i in sorted_dict_train.keys():
    row_info = [i, sorted_dict_train[i], 'None']
    output_train.append(row_info)
    
output_dev = []
for i in sorted_dict_dev.keys():
    row_info = [i, sorted_dict_dev[i], 'None']
    output_dev.append(row_info)
    
output_test = []
for i in sorted_dict_test.keys():
    row_info = [i, sorted_dict_test[i], 'None']
    output_test.append(row_info)    
    
headers = ['file_path', 'length', 'label']
pd.DataFrame(output_train).to_csv("../data/train-table_au.csv", header=headers)
pd.DataFrame(output_dev).to_csv("../data/dev-table_au.csv", header=headers)
pd.DataFrame(output_test).to_csv("../data/test-table_au.csv", header=headers)
print("Finish creating table files, exit.")
    