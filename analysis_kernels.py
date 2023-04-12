import numpy as np
from data import load_training_data, load_test_data


def normalize_kernels(path):
    ker = np.load(path)

    # ker_normalized = np.zeros_like(ker)
    # for i in range(ker.shape[0]):
    #     for j in range(ker.shape[1]):
    #         ker_normalized[i, j] = ker[i, j] / (np.sqrt(ker[i, i] * ker[j, j] + 1e-8))
    
    ker_normalized = normalize(ker, eps=1e-8)

    np.save(path.replace('.npy', '_normalized.npy'), ker_normalized)

def normalize(K, eps=0):
    K_diag = K[np.arange(K.shape[0]), np.arange(K.shape[0])]
    K_norm = K/np.sqrt(K_diag[:,None] * K_diag[None,:] + eps)
    return K_norm

def psdfy(K, eps = 0):
    w, v = np.linalg.eigh(K)
    w_pos_part = np.maximum(eps, w)
    K_psd = (v*w_pos_part)@v.T
    K_psd = (K_psd + K_psd.T)/2
    return K_psd

def correct_kernel(K, slow_kernel, sub_graphs1, sub_graphs2 = None, tol=0.01, stop_cond=10):
    """Correct absurd values with a slower using surer kernel. Starts with nan, then higher values in absolute values.

    Args:
        K (ndarray): Kernel matrix to correct
        slow_kernel (Kernel): Kernel object, with option fast=False
        sub_graphs1 (list): List of (processed) graphs corresponding to rows
        sub_graphs2 (list, optional): List of (processed) graphs corresponding to columns. If None, assumed to be equal to sub_graphs1. Defaults to None.
        tol (float, optional): When a correction leads to a relative difference < tol, then we consider the initial value was good. Defaults to 0.01.
        stop_cond (int, optional): Number of consecutive times the initial value must be good to decide to stop the algorithm. Defaults to 10.

    Returns:
        ndarray: Corrected kernel matrix
        list: list of old values replaced (in the corresponding order)
        list: list of new values old values have been replaced by (in the corresponding order)
        list: list of difference new - old values (in the corresponding order)
    """ 
    n = K.shape[0]
    K_correct = np.copy(K)
    inner = sub_graphs2 is None
    if inner: sub_graphs2 = sub_graphs1
    assert K.shape[0] == len(sub_graphs1)
    assert K.shape[1] == len(sub_graphs2)

    idx_big = np.argsort(np.abs(K).flatten(order='C'))[::-1]
    # nan are at the beginning
    i_big = idx_big//n
    j_big = idx_big%n

    if inner:
        # To remove the lower triangular part
        of_interest = j_big >= i_big
        i_big = i_big[of_interest]
        j_big = j_big[of_interest]
    
    k_old_list = []
    k_cor_list = []
    k_diff_list = []
    is_tol_list = []
    for q in range(len(i_big)):
        i, j = i_big[q], j_big[q]
        k_cor, info = slow_kernel.kernel_eval(sub_graphs1[i], sub_graphs2[j])
        k_old = K[i, j]
        k_diff = k_cor - k_old
        is_tol = not(np.isnan(k_old)) and (np.abs(k_diff) < tol*np.abs(k_old))

        if q%100 == 0 or is_tol: 
            print(f"{q}: {k_cor} {k_old} {k_diff}              ", end="\r")

        k_old_list.append(k_old)
        k_cor_list.append(k_cor)
        k_diff_list.append(k_diff)
        is_tol_list.append(is_tol)

        # if is_tol:
        #     cntn = input(f"{q} {k_cor} {k_old} {k_diff}")
        #     if cntn=="stop":
        #         break
        
        K_correct[i, j] = k_cor
        if inner:
            K_correct[j, i] = k_cor
        
        if np.all(is_tol_list[-stop_cond:]): 
            break

    return K_correct, k_old_list, k_cor_list, k_diff_list

if __name__ == '__main__':
    kernel = np.load('saved/walk_kernel_3_subset_0.npy')
    print(kernel.shape)
    print(np.max(kernel))
    print(np.min(kernel))
    normalize_kernels('saved/walk_kernel_3_subset_0.npy')
    kernel = np.load('saved/walk_kernel_3_subset_0.npy')
    print(np.linalg.eigh(kernel))
    print(kernel)
    print(np.max(kernel))
    print(np.min(kernel))


    # train_data, train_labels = load_training_data()
    # test_data = load_test_data()
    #
    # max_deg_train = 0
    # max_deg_test = 0
    # for graph in train_data:
    #     degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)
    #     dmax = max(degree_sequence)
    #     if dmax > max_deg_train:
    #         max_deg_train = dmax
    #
    # for graph in test_data:
    #     degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)
    #     dmax = max(degree_sequence)
    #     if dmax > max_deg_test:
    #         max_deg_test = dmax
    #
    # print(max_deg_train)
    # print(max_deg_test)
