import numpy as np
import pickle
import os.path as osp
from sklearn.model_selection import StratifiedKFold

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def load_data(name):
    with open(osp.join('data', name), 'rb') as file:
        data = pickle.load(file)
    file.close()
    return data


def load_training_data():
    training_data = load_data('training_data.pkl')
    training_labels = load_data('training_labels.pkl')

    # For classification tasks, prefer -1/+1 labels rather than 0/1 labels
    training_labels = [-1 if x == 0 else x for x in training_labels]

    return np.array(training_data), np.array(training_labels)


def load_test_data():
    testing_data = load_data('test_data.pkl')
    return testing_data


def split_data(n_splits=3, return_indices = False):
    training_data, training_labels = load_training_data()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    spl = list(splitter.split(training_data, training_labels))
    index_splits = [spl[j][1] for j in range(n_splits)]

    split_list = [(training_data[curr_idx], training_labels[curr_idx]) for curr_idx in index_splits]
    if return_indices:
        return split_list, index_splits
    return split_list


if __name__ == '__main__':
    train_data, train_labels = load_training_data()
    split = split_data()

    test_data = load_test_data()

    assert len(train_data) == len(train_labels)
    print(f'train data length: {len(train_data)}')
    print(f'test data length: {len(test_data)}')
    print("Showing a sequence/graph")
    ex_graph = train_data[0]
    print(ex_graph.number_of_nodes())
    print(ex_graph.number_of_edges())
    print(ex_graph.edges)
    for i in range(len(ex_graph.nodes)):
        print(ex_graph.nodes[i])
    print(type(ex_graph.edges))
