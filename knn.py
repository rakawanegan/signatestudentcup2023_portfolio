from sklearn.neighbors import KNeighborsRegressor as KNN
import pandas as pd
from pathlib import Path
from lib.preprocess import Preprocess
from lib.postprocess import Estimater
from lib.local_utils import run_command, generate_experiment_memo
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
import numpy as np
import datetime
import os


MODELNAME = "knn"
date = datetime.datetime.now().strftime('%m%d%H%M')
dirname = f"results/{MODELNAME}_{date}"
os.makedirs(dirname)
run_command(f"cp {Path(__file__).resolve()} {dirname}/main.py")
run_command(f"cp -r lib {dirname}/")

#load data
SEED = 314
k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)
ppr = Preprocess(SEED)
x, y, x_test = ppr.get_data()

k = 5
use_col = ["year", "odometer",]

foldsocrelist = list()
models = list()
pprs = list()
for i, (train_index, eval_index) in enumerate(k_fold.split(x, y)):
    x_train, x_eval = x.loc[train_index,:].copy(), x.loc[eval_index,:].copy()
    y_train, y_eval = y.loc[train_index,:].copy(), y.loc[eval_index,:].copy()
    ppr = Preprocess(SEED)
    x_train = ppr.fit_transform(x_train, y_train)[use_col].fillna(-100)
    x_eval = ppr.transform(x_eval)[use_col].fillna(-100)
    pprs.append(ppr)

    model = KNN(k)
    model.fit(x_train.values, y_train.values)

    eval_pred = model.predict(x_eval.values)
    score = mean_absolute_percentage_error(y_eval, eval_pred)
    foldsocrelist.append(score)
    print(f"mape score {i}: {score:.7f}")

    eval_pred = np.array(eval_pred).reshape(-1)
    y_eval = np.array(y_eval).reshape(-1)
    evals = pd.concat( [pd.Series(eval_pred), pd.Series(y_eval)], names=['predict', 'actual'], axis=1)
    evals.to_csv(f"{dirname}/eval{i}.csv", header=False)
    estimater = Estimater(ppr, model, use_col)
    estimater.dump(f"{dirname}/estimater{i}.pkl")
    models.append(model)

ppr = pprs[np.argmin(foldsocrelist)]
model = models[np.argmin(foldsocrelist)]
x_test = ppr.transform(x_test)[use_col].fillna(-100)
y_pred = model.predict(x_test.values)
submission = pd.DataFrame(y_pred, columns=['Price'], index=x_test.index)
submission.to_csv(f"{dirname}/submission.csv", header=False)

print(f"mape score: {np.mean(foldsocrelist):.7f}")
content = {
    "MODELNAME": MODELNAME,
    "SEED": SEED,
    "use_col": use_col,
    "k": k,
    "eval_mape mean": np.mean(foldsocrelist),
    "ecal_mape each": "\n".join([f"- {i}: {score}" for i,score in enumerate(foldsocrelist)]),
    "input_future": "\n".join([f'- {col}' for col in x.columns]),
}

generate_experiment_memo(dirname, content)
