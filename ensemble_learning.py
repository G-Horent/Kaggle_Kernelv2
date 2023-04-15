import numpy as np
from data import load_training_data, load_test_data, split_data
from utils import predictions_to_csv
from kernel_methods import KernelSVM, get_kernel
from datetime import datetime
import os


def ensemble_predictions(args):
    num_splits = args.splits
    kernel_name = args.kernel
    train_splits = split_data(n_splits=num_splits)
    test_data = load_test_data()

    params = {'n': args.n, 'sigma': args.sigma, 'h': args.h}

    print("Starting ensemble predictions")

    prediction_final = np.zeros((num_splits, len(test_data)))

    for idx, (train_data, train_labels) in enumerate(train_splits):
        print(f"Treating split {idx}")
        model = KernelSVM(lmbd=args.lmbd, kernel_name=kernel_name, balanced=False, precomputed_kernel=False, save_kernel=args.save, **params)
        model.fit(train_data, train_labels)
        print('Finished fitting, starting evaluation')
        auc1 = model.score(train_splits[(idx+1)%num_splits][0], train_splits[(idx+1)%num_splits][1])
        auc2 = model.score(train_splits[(idx+2)%num_splits][0], train_splits[(idx+2)%num_splits][1])

        print(f'AUC split {(idx+1)%num_splits} : {auc1}')
        print(f'AUC split {(idx+2)%num_splits} : {auc2}')
        predictions = model.predict(test_data)
        prediction_final[idx, :] = predictions

    pred = np.sum(prediction_final, axis=0)
    now = datetime.now()
    predictions_to_csv(
        f'submissions/submission_splits_{num_splits}_{kernel_name}_' + now.strftime("%m%d_%H%M%S") + '.csv', pred)


def single_prediction(args):
    kernel_name = args.kernel

    train_data, train_labels = load_training_data()
    test_data = load_test_data()

    model = KernelSVM(lmbd=0.0001, kernel_name=kernel_name, balanced=False, precomputed_kernel=False, save_kernel=args.save,
                      h=3)
    model.fit(train_data, train_labels)
    print('Finished fitting, starting evaluation')

    predictions = model.predict(test_data)
    print('Finished evaluation, saving...')
    now = datetime.now()
    predictions_to_csv(
        f'submissions/submission_all_{kernel_name}_' + now.strftime("%m%d_%H%M%S") + '.csv', predictions)


