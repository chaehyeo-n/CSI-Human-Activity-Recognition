from preprocessing.extract_csi_and_label import extract_csi_by_label

def extract_csi(raw_folder, labels, win_len, thrshd, step, save=False):
    """
    Return List of Array in the format of [X_label1, y_label1, X_label2, y_label2]
    Args:
        raw_folder:  the folder path of raw CSI csv files, input_* annotation_*
        labels    :  all the labels existing in the folder
        save      :  boolean, choose whether save the numpy array 
        win_len   :  integer, window length
        thrshd    :  float,  determine if an activity is strong enough inside a window
        step      :  integer, sliding window by step
    """
    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label(raw_folder, label, labels, save, win_len, thrshd, step)
        feature_arr_flattened = feature_arr.reshape(feature_arr.shape[0], -1)
        ans.append(feature_arr_flattened)
        ans.append(label_arr)
    return tuple(ans)