import os
from argparse import ArgumentParser
from ensemble_learning import ensemble_predictions, single_prediction

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--kernel', type=str, default='KernelWLSubtree', help="Kernel used")
    args.add_argument('--splits', type=int, default=0, help="Number of splits. If 0, uses the entire dataset")
    args.add_argument('--lmbd', type=float, default=0.0001, help='Regularization parameter')
    args.add_argument('--n', type=int, default=3, help='Length of walk for walk-kernel')
    args.add_argument('--sigma', type=float, default=1.0, help='sigma for RBF kernel')
    args.add_argument('--balanced', action='store_true', help='Used the balanced version of the SVM')
    args.add_argument('--h', type=int, default=3, help='Number of iterations for the WL algorithm')
    args.add_argument('--save', action='store_true', help='Save the kernel')
    args = args.parse_args()

    if args.save:
        os.makedirs('saved', exist_ok=True)

    os.makedirs("submissions", exist_ok=True)

    if args.splits == 0:
        single_prediction(args)
    else:
        ensemble_predictions(args)
