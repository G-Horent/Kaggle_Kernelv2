import numpy as np
from data import load_training_data, load_test_data


def normalize_kernels(path):
    ker = np.load(path)
    ker_normalized = np.zeros_like(ker)

    for i in range(ker.shape[0]):
        for j in range(ker.shape[1]):
            ker_normalized[i, j] = ker[i, j] / (np.sqrt(ker[i, i] * ker[j, j] + 1e-8))

    np.save(path.replace('.npy', '_normalized.npy'), ker_normalized)


if __name__ == '__main__':
    # kernel = np.load('saved/walk_kernel_3_subset_0.npy')
    # print(kernel.shape)
    # print(np.max(kernel))
    # print(np.min(kernel))
    # normalize_kernels('saved/walk_kernel_3_subset_0.npy')
    # kernel = np.load('saved/walk_kernel_3_subset_0.npy')
    # print(np.linalg.eigh(kernel))
    # print(kernel)
    # print(np.max(kernel))
    # print(np.min(kernel))
    train_data, train_labels = load_training_data()
    test_data = load_test_data()

    max_deg_train = 0
    max_deg_test = 0
    for graph in train_data:
        degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)
        dmax = max(degree_sequence)
        if dmax > max_deg_train:
            max_deg_train = dmax

    for graph in test_data:
        degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)
        dmax = max(degree_sequence)
        if dmax > max_deg_test:
            max_deg_test = dmax

    print(max_deg_train)
    print(max_deg_test)
