from pathlib import Path
from lib.preprocess import Preprocess
from lib.postprocess import Estimater
from lib.local_utils import run_command, generate_experiment_memo
# !pip install pytorch-tabnet
# ref to https://github.com/dreamquark-ai/tabnet
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.metrics import Metric
import torch
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
import os
import datetime
import numpy as np

SEED = 314
MAX_EPOCH = 1000
BATCH_SIZE = 1024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
MODELNAME = "pretrain_tabnet"
date = datetime.datetime.now().strftime('%m%d%H%M')
dirname = f"results/{MODELNAME}_{date}"
os.makedirs(dirname)
run_command(f"cp {Path(__file__).resolve()} {dirname}/main.py")
run_command(f"cp -r lib {dirname}/")


#load data
ppr = Preprocess(SEED)
x, y, x_test = ppr.get_data()
k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)

class mape(Metric):
    def __init__(self):
        self._name = "mape"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred)


foldscorelist = list()
models = list()
pprs = list()
for i, (train_index, eval_index) in enumerate(k_fold.split(x, y)):
    x_train, x_eval = x.loc[train_index,:].copy(), x.loc[eval_index,:].copy()
    y_train, y_eval = y.loc[train_index,:].copy(), y.loc[eval_index,:].copy()
    ppr = Preprocess(SEED)
    x_train = ppr.fit_transform(x_train, y_train, is_impute=True)
    x_eval = ppr.transform(x_eval)
    pprs.append(ppr)

    premodel = TabNetPretrainer(
        n_d=51,
        n_a=51,
        n_steps=4,
        gamma=1.5,
        lambda_sparse=0,
        n_independent=2,
        n_shared=5,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=6e-3),
        mask_type="entmax",
    )
    
    premodel.fit(
        X_train=x_train.values,
        eval_set=[x_eval.values],
        pretraining_ratio=0.3,
        num_workers=os.cpu_count(),
    )

    model = TabNetRegressor(
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params=dict(T_max=50),
        seed=SEED,
    )

    model.fit(
        X_train=x_train.values,
        y_train=y_train.values,
        eval_set=[(x_eval.values, y_eval.values)],
        eval_name=["valid"],
        eval_metric=[mape],
        max_epochs=MAX_EPOCH,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        drop_last=False,
        loss_fn=torch.nn.functional.mse_loss,
        from_unsupervised=premodel
    )

    models.append(model)
    x_eval = torch.tensor(x_eval.values, dtype=torch.float32)
    eval_pred = model.predict(x_eval)
    score = mean_absolute_percentage_error(y_eval, eval_pred)
    print(f"mape score {i}: {score:.7f}")
    foldscorelist.append(score)
    eval_pred = np.array(eval_pred).reshape(-1)
    y_eval = np.array(y_eval).reshape(-1)
    evals = pd.concat( [pd.Series(eval_pred), pd.Series(y_eval)], names=['predict', 'actual'], axis=1)
    evals.to_csv(f"{dirname}/eval{i}.csv", header=False)
    estimater = Estimater(ppr, model)
    estimater.dump(f"{dirname}/estimater{i}.pkl")


ppr = pprs[np.argmin(foldscorelist)]
model = models[np.argmin(foldscorelist)]
test_index = x_test.index
x_test = ppr.transform(x_test)
x_test = torch.tensor(x_test.values, dtype=torch.float32)

model.predict(x_test)
y_pred = model.predict(x_test)
submission = pd.DataFrame(y_pred, columns=['Price'], index=test_index)
submission.to_csv(f"{dirname}/submission.csv", header=False)

content = {
    "MODELNAME": MODELNAME,
    "MAX_EPOCH": MAX_EPOCH,
    "BATCH_SIZE": BATCH_SIZE,
    "SEED": SEED,
    "mape mean": np.mean(foldscorelist),
    "foldscorelist": foldscorelist,
    "input_future": "\n".join([f'- {col}' for col in x_train.columns]),
}

generate_experiment_memo(dirname, content)
