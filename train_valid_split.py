import numpy as np

def train_valid_split(numpy_tuple, train_portion, seed):
    """
    Returns Train and Valid Datset with the format of (x_train, y_train, x_valid, y_valid),
    where x_train and y_train are shuffled randomly.

    Args:
        numpy_tuple  : tuple of numpy array: (x_sitdown, x_standup)
        train_portion: float, range (0,1)
        seed         : random seed
    """
    np.random.seed(seed=seed)
    x_train = []
    x_valid = []
    y_valid = []
    y_train = []

    for i, x_arr in enumerate(numpy_tuple):
        x_arr_flattened = x_arr.reshape(x_arr.shape[0], -1)
        index = np.random.permutation(x_arr_flattened.shape[0])
        split_len = int(train_portion * x_arr_flattened.shape[0])
        x_train.append(x_arr_flattened[index[:split_len], ...])
        tmpy = np.zeros((split_len, len(numpy_tuple)))
        tmpy[:, i] = 1
        y_train.append(tmpy)

        x_valid.append(x_arr_flattened[index[split_len:], ...])
        tmpy = np.zeros((x_arr_flattened.shape[0] - split_len, len(numpy_tuple)))
        tmpy[:, i] = 1
        y_valid.append(tmpy)

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    index = np.random.permutation(x_train.shape[0])
    x_train = x_train[index, ...]
    y_train = y_train[index, ...]
    return x_train, y_train, x_valid, y_valid