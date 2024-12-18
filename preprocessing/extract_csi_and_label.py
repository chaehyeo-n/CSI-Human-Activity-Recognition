import numpy as np
import os
import glob
from .merge_input_and_annotation import merge_csi_label

def extract_csi_by_label(raw_folder, label, labels, win_len, thrshd, step, save=False):
    """
    Returns all the samples (X,y) of "label" in the entire dataset
    Args:
        raw_folder: The path of Dataset folder
        label    :  str, could be one of labels
        labels   :  list of str, ['sitdown', 'standup']
        save     :  boolean, choose whether save the numpy array 
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """

    # validate the label
    label = label.lower()
    if not label in labels:
        raise ValueError("The label {} should be among 'sitdown','standup'".format(labels))
    
    data_path_pattern = os.path.join(raw_folder, 'input_*' + label + '*.csv')
    input_csv_files = sorted(glob.glob(data_path_pattern))
    annot_csv_files = [os.path.basename(fname).replace('input_', 'annotation_') for fname in input_csv_files]
    annot_csv_files = [os.path.join(raw_folder, fname) for fname in annot_csv_files]

    feature = []
    index = 0
    for csi_file, label_file in zip(input_csv_files, annot_csv_files):
        index += 1
        if not os.path.exists(label_file):
            print('Warning! Label File {} doesn\'t exist.'.format(label_file))
            continue
        feat_arr, _ = merge_csi_label(csi_file, label_file, win_len=win_len, thrshd=thrshd, step=step)
        feat_arr_flattened = feat_arr.reshape(feat_arr.shape[0], -1)
        feature.append(feat_arr_flattened)
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100, label))
        
    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed("X_{}_win_{}_thrshd_{}percent_step_{}.npz".format(
            label, win_len, int(thrshd*100), step), feat_arr)
    # one hot
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label