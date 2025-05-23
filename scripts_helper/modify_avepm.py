import os
import numpy as np
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Modify AVE-PM for S-PM')
parser.add_argument('--selected_csv_file', type=str, default='selected_csv_file.csv', help='the selected csv file')
parser.add_argument('--mapping_file', type=str, default='data', help='the directory of data')
parser.add_argument('--src_feat_path', type=str, default='', help='the original feature path')
parser.add_argument('--dst_feat_path', type=str, default='', help='the modified feature path')

args = parser.parse_args()
selected_csv_file = args.selected_csv_file
selected_data_csv = pd.read_csv(selected_csv_file)
mapping_file = args.mapping_file
mapping_csv = pd.read_csv(mapping_file)
src_feat_path = args.src_feat_path
dst_feat_path = args.dst_feat_path

if not os.path.exists(dst_feat_path):
    os.makedirs(dst_feat_path)

look_up_table = {}
name_table = {}
for _, row in mapping_csv.iterrows():
    look_up_table[row['old_index']] = row['index']
    name_table[row['old_index']] = row['category']

def process_row(row):
    onset = round(row['onset'])
    offset = round(row['offset'])
    sample_id = row['sample_id']
    old_label = row['label']
    file_path = os.path.join(src_feat_path, f"{sample_id}.pkl")
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    new_label = look_up_table[old_label]
    one_hot = np.zeros((10, len(look_up_table)))
    one_hot[onset:offset, new_label] = 1
    data['onehot_labels'] = one_hot
    dst_file_path = os.path.join(dst_feat_path, f"{sample_id}.pkl")
    with open(dst_file_path, 'wb') as f:
        pickle.dump(data, f)

    return {
        'category': name_table[old_label],
        'video_id': sample_id,
        'onset': onset,
        'offset': offset,
        'old_index': old_label
    }

def main():
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_row, selected_data_csv.to_dict('records')), total=len(selected_data_csv)))

if __name__ == '__main__':
    main()