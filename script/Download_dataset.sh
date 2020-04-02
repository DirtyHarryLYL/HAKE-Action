#!/bin/bash

# ---------------HICO-DET Dataset------------------
echo "Downloading Dataset"

mkdir -Data/
URL_HICO_DET=http://napoli18.eecs.umich.edu/public_html/data/hico_20160224_det.tar.gz

wget -N $URL_HICO_DET -P Data/
tar -xvzf Data/hico_20160224_det.tar.gz -C Data/
rm Data/hico_20160224_det.tar.gz

echo "Downloading training data..."

python script/Download_data.py 1uhtAg00EGZelc6mhEowQ3vwxWN1YBmXC Data/Trainval_GT_all_part.pkl
python script/Download_data.py 1cqaJxzFRrMENTYGcuZFYy8JRuTP_qAEq Data/Trainval_Neg_all_part.pkl
python script/Download_data.py 1KnQvUklj8p_cld1FagsUK6BEUeHqBtpR Data/Test_all_part.pkl
python script/Download_data.py 1hkakTuYB_3C4GCbZgSm2pH7AHbhmEr5y Data/ava_train_all_fixed.pkl
python script/Download_data.py 1cYR2rI8H78sY7exv-L3HXYhXqp7HQdUU Data/ava_val_fixed.pkl
python script/Download_data.py 1ew-uIrD_ZFmO5RFsuMV2sBEqWWVtXkHH Weights/pasta_full.zip
python script/Download_data.py 1h0InSjxCffLuyoXEPbLIzOH0KVWUuUJc Weights/pasta_AVA.zip
python script/Download_data.py 1l12Dr217NeTnbBA5e-9lLetMFmDv7eI9 Weights/pasta_HICO-DET.zip
python script/Download_data.py 1_O8eo1fmJtVzb_W5Y4emg_eE7LTJKPGR Results.tar.gz
