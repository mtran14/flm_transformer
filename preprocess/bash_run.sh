#!/bin/bash
for value in {1..32}
do
python preprocess_openface.py --input_path=/shares/perception-working/minh/openface_voxceleb2/ --output_path=/shares/perception-working/minh/vox2/vox2_npy_3fps/ --chunk=$value --n_chunk=32 --window_size=10 --step_size=10 &
done
