import numpy as np
from data import load_training_data, load_test_data, split_data
from utils import predictions_to_csv
from kernel_methods import KernelSVM, KernelLogisticRegression, get_kernel
from datetime import datetime
import os


def ensemble_predictions(num_splits, kernel_name='KernelRBF'):
    train_splits = split_data(n_splits=num_splits)
    test_data = load_test_data()

    print("Starting ensemble predictions")
    now = datetime.now()
    path_exp = 'saved/experiment_' + now.strftime("%m%d_%H%M%S")

    os.makedirs(path_exp)

    prediction_final = np.zeros((num_splits, len(test_data)))

    for idx, (train_data, train_labels) in enumerate(train_splits):
        print(f"Treating split {idx}")
        model = KernelSVM(lmbd=0.00001, kernel_name=kernel_name, precomputed_kernel=False, n=3, save_path=path_exp)
        model.fit(train_data, train_labels)
        predictions = model.predict(test_data)
        prediction_final[idx, :] = predictions

    pred = np.sum(prediction_final, axis=0)
    now = datetime.now()
    predictions_to_csv(
        f'submissions/submission_splits_{num_splits}_{kernel_name}_' + now.strftime("%m%d_%H%M%S") + '.csv', pred)


if __name__ == '__main__':
    ensemble_predictions(num_splits=6, kernel_name='Kernel_nwalk')
