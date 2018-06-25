import argparse
import os
import pickle

from lightgbm import LGBMClassifier

parser = argparse.ArgumentParser()

parser.add_argument("--fold_id", type=int, default=0)
parser.add_argument("--load_path", type=str, default='../preprocessed_data/non_stratified/')
parser.add_argument("--save_path", type=str, default='../model/non_stratified/')

parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--num_leaves", type=int, default=20)
parser.add_argument("--colsample_bytree", type=float, default=0.1)
parser.add_argument("--subsample", type=float, default=0.1)
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--reg_alpha", type=float, default=0.00001)
parser.add_argument("--reg_lambda", type=float, default=0.00001)
parser.add_argument("--min_split_gain", type=float, default=0.00001)
parser.add_argument("--min_child_weight", type=int, default=1)

args = parser.parse_args()


def fit():
    [train_x, train_y, valid_x, valid_y] = pickle.load(
        open(args.load_path + 'fold_' + str(args.fold_id) + '.p', 'rb'))

    clf = LGBMClassifier(
        nthread=4,
        n_estimators=args.n_estimators,
        learning_rate=0.02,
        num_leaves=args.num_leaves,
        colsample_bytree=args.colsample_bytree,
        subsample=args.subsample,
        max_depth=args.max_depth,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        min_split_gain=args.min_split_gain,
        min_child_weight=args.min_child_weight,
        silent=-1,
        verbose=-1,
    )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric='auc', verbose=100, early_stopping_rounds=200)

    save_dir = args.save_path

    all_keys = vars(args).keys()
    useful_key = [item for item in all_keys if item not in ['load_path', 'save_path']]

    for key in useful_key:
        save_dir += key + '_' + str(vars(args)[key]) + '/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pickle.dump(clf, open(save_dir + 'model.pkl', 'wb'))


if __name__ == '__main__':
    fit()
