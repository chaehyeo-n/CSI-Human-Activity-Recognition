import numpy as np
import csv

def merge_csi_label(csifile, labelfile, win_len, thrshd, step):
    """
    Merge CSV files into a Numpy Array  X,  csi amplitude feature
    Returns Numpy Array X, Shape(Num, Win_Len, 90)
    Args:
        csifile  :  str, csv file containing CSI data
        labelfile:  str, csv file with activity label 
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    activity = []
    with open(labelfile, 'r') as labelf:
        reader = csv.reader(labelf)
        for line in reader:
            label  = line[0]
            if label == 'NoActivity':
                activity.append(0)
            else:
                activity.append(1)
    activity = np.array(activity)

    csi = []
    with open(csifile, 'r') as csif:
        reader = csv.reader(csif)
        for line in reader:
            # extract the amplitude only
            line_array = np.array([float(v) for v in line[1:91]])
            csi.append(line_array[np.newaxis, ...])
    csi = np.concatenate(csi, axis=0)

    assert(csi.shape[0] == activity.shape[0])

    # create a sliding window
    index = 0
    feature = []
    labels = []
    while index + win_len <= csi.shape[0]:
        cur_activity = activity[index:index + win_len]
        if np.sum(cur_activity) < thrshd * win_len:
            index += step
            continue
        cur_feature = csi[index:index + win_len, :].reshape(-1)
        feature.append(cur_feature)
        labels.append(1 if np.sum(cur_activity) > 0 else 0)
        index += step

    return np.array(feature), np.array(labels)