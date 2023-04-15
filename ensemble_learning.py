import numpy as np
from data import load_training_data, load_test_data, split_data
from utils import predictions_to_csv
from kernel_methods import KernelSVM, get_kernel
from datetime import datetime
import os


def ensemble_predictions(args, explicit_name=False):
    num_splits = args.splits
    kernel_name = args.kernel
    train_splits = split_data(n_splits=num_splits)
    test_data = load_test_data()

    params = {'n': args.n, 'sigma': args.sigma, 'h': args.h, 'rwlam': args.rwlam}
    prediction_final = np.zeros((num_splits, len(test_data)))

    print("Starting ensemble predictions")
    for idx, (train_data, train_labels) in enumerate(train_splits):

        print(f"Treating split {idx}")
        model = KernelSVM(lmbd=args.lmbd, kernel_name=kernel_name, balanced=args.balanced, precomputed_kernel=False, save_kernel=args.save, **params)
        model.fit(train_data, train_labels)

        print('Finished fitting, starting evaluation')
        auc1 = model.score(train_splits[(idx+1)%num_splits][0], train_splits[(idx+1)%num_splits][1])
        auc2 = model.score(train_splits[(idx+2)%num_splits][0], train_splits[(idx+2)%num_splits][1])
        print(f'AUC split {(idx+1)%num_splits} : {auc1}')
        print(f'AUC split {(idx+2)%num_splits} : {auc2}')

        predictions = model.predict(test_data)
        prediction_final[idx, :] = predictions

    pred = np.sum(prediction_final, axis=0)
    if explicit_name:
        now = datetime.now().strftime("%m%d_%H%M%S")
        predictions_to_csv(
            f'submissions/submission_splits_{num_splits}_{kernel_name}_{now}.csv', pred)
    else:
        predictions_to_csv("test_pred.csv", pred) # Required by the instructions


def single_prediction(args, explicit_name=False):
    kernel_name = args.kernel

    train_data, train_labels = load_training_data()
    test_data = load_test_data()

    params = {'n': args.n, 'sigma': args.sigma, 'h': args.h, 'rwlam': args.rwlam}

    print("Starting fitting the whole dataset")
    model = KernelSVM(lmbd=0.0001, kernel_name=kernel_name, balanced=args.balanced, precomputed_kernel=False, save_kernel=args.save, **params)
    model.fit(train_data, train_labels)
    print('Finished fitting, starting evaluation')

    predictions = model.predict(test_data)
    print('Finished evaluation, saving...')
    if explicit_name:
        now = datetime.now().strftime("%m%d_%H%M%S")
        predictions_to_csv(
            f'submissions/submission_all_{kernel_name}_{now}.csv', predictions)
    else:
        predictions_to_csv("test_pred.csv", predictions)


