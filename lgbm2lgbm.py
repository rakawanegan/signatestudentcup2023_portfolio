import optuna.integration.lightgbm as lgb
from lightgbm import Dataset
import pandas as pd
from pathlib import Path
from lib.preprocess import Preprocess
from lib.postprocess import Estimater
from lib.local_utils import run_command, generate_experiment_memo
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import numpy as np
import os
import datetime


SEED = 314
MODELNAME = "lgbm2lgbm"
date = datetime.datetime.now().strftime('%m%d%H%M')
dirname = f"results/{MODELNAME}_{date}"
os.makedirs(dirname)
run_command(f"cp {Path(__file__).resolve()} {dirname}/main.py")
run_command(f"cp -r lib {dirname}/")

#load data
ppr = Preprocess(SEED)
x, y, x_test = ppr.get_data()

x_train0, x_eval0, y_train0, y_eval0 = train_test_split(x, y, test_size=0.2, random_state=SEED)
x_train0 = ppr.fit_transform(x_train0, y_train0, is_ohe=True, is_dropregion=False)
x_eval0 = ppr.transform(x_eval0)

# y_train0["price"] /= x_train0["year"]
# y_eval0["price"] /= x_eval0["year"]

lgb_train = Dataset(x_train0.values, y_train0.values.ravel())
lgb_eval = Dataset(x_eval0.values, y_eval0.values.ravel(), reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'mape',
    'metric': 'mape',
    'seed': SEED,
    'learning_rate': 0.01,
    'force_row_wise': True,
        }

pre_model = lgb.train(
        params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=1000,
        optuna_seed=SEED,
        callbacks=[
                lgb.early_stopping(stopping_rounds=250, verbose=True),
                ]
        )

pre_model_predict = pd.Series(pre_model.predict(ppr.transform(x_test.copy()).values), index=x_test.index)

importance = pd.DataFrame(
    pre_model.feature_importance(importance_type='gain'),
    index=x_train0.columns,
    columns=['importance']
)
importance.sort_values(by='importance', ascending=False).to_csv(f"{dirname}/first_importance.csv")

topimportance = importance.loc[importance["importance"]!=0,:].sort_values(by='importance', ascending=False).head(180)
del x_train0, x_eval0, y_train0, y_eval0, lgb_train, lgb_eval, pre_model, importance

k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)
foldscorelist = list()
models = list()
pprs = list()
use_columns_list = list()
y_pred_ensambles = list()
for i, (train_index, eval_index) in enumerate(k_fold.split(x, y)):
    x_train, x_eval = x.loc[train_index,:].copy(), x.loc[eval_index,:].copy()
    y_train, y_eval = y.loc[train_index,:].copy(), y.loc[eval_index,:].copy()
    ppr = Preprocess(SEED)
    x_train = ppr.fit_transform(x_train, y_train, is_ohe=True, is_dropregion=False)
    x_eval = ppr.transform(x_eval)
    pprs.append(ppr)

    # y_train["price"] /= x_train["year"]
    # y_eval["price"] /= x_eval["year"]

    use_columns = list(set(topimportance.index) & set(x_train.columns) & set(x_eval.columns))
    use_columns_list.append(use_columns)
    x_train = x_train.loc[:,use_columns]
    x_eval = x_eval.loc[:,use_columns]

    lgb_train = Dataset(x_train.values, y_train.values.ravel())
    lgb_eval = Dataset(x_eval.values, y_eval.values.ravel(), reference=lgb_train)

    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'mape',
            'metric': 'mape',
            'seed': SEED,
            'force_row_wise': True,
            }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=10000,
        optuna_seed=SEED,
        callbacks=[
                lgb.reset_parameter(learning_rate=[0.005]*200+[0.001]*9800),
                lgb.early_stopping(stopping_rounds=250, verbose=True),
                ]
    )

    importance = pd.DataFrame(
        model.feature_importance(importance_type='gain'),
        index=x_train.columns,
        columns=['importance']
    )
    importance.to_csv(f"{dirname}/importance{i}.csv")

    eval_pred = model.predict(x_eval.values)
    score = mean_absolute_percentage_error(y_eval, eval_pred)
    foldscorelist.append(score)
    print(f"mape score: {score:.7f}")
    eval_pred = np.array(eval_pred).reshape(-1)
    y_eval = np.array(y_eval).reshape(-1)
    evals = pd.concat( [pd.Series(eval_pred), pd.Series(y_eval)], names=['predict', 'actual'], axis=1)
    evals.to_csv(f"{dirname}/eval{i}.csv", header=False)
    estimater = Estimater(ppr, model, use_columns)
    # estimater.dump(f"{dirname}/estimater{i}.pkl")
    models.append(model)

    y_pred_ensambles.append(pd.Series(model.predict(ppr.transform(x_test.copy()).loc[:,use_columns].values), index=x_test.index))

all_pred_ensamble = y_pred_ensambles.copy()
all_pred_ensamble.append(pre_model_predict)
all_pred_ensamble = pd.concat(all_pred_ensamble, axis=1).mean(axis=1)
all_pred_ensamble = pd.DataFrame(all_pred_ensamble, columns=['Price'], index=x_test.index)
# all_pred_ensamble["Price"] *= x_test["year"]
all_pred_ensamble.to_csv(f"{dirname}/submission_allensamble.csv", header=False)

y_pred_ensamble = pd.concat(y_pred_ensambles, axis=1).mean(axis=1)
y_pred_ensamble = pd.DataFrame(y_pred_ensamble, columns=['Price'], index=x_test.index)
# y_pred_ensamble["Price"] *= x_test["year"]
y_pred_ensamble.to_csv(f"{dirname}/submission_foldensamble.csv", header=False)

ppr = pprs[np.argmin(foldscorelist)]
model = models[np.argmin(foldscorelist)]
use_columns = use_columns_list[np.argmin(foldscorelist)]
x_test = ppr.transform(x_test).loc[:,use_columns]
y_pred = model.predict(x_test.values)

submission = pd.DataFrame(y_pred, columns=['Price'], index=x_test.index)
# submission["Price"] *= x_test["year"]
submission.to_csv(f"{dirname}/submission.csv", header=False)

print(f"mape score: {np.mean(foldscorelist):.7f}")
content = {
    "MODELNAME": MODELNAME,
    "SEED": SEED,
    "eval_mape mean": np.mean(foldscorelist),
    "ecal_mape each": "\n".join([f"- {i}: {score}" for i,score in enumerate(foldscorelist)]),
    "input_future": "\n".join([f'- {col}' for col in x_train.columns]),
}

generate_experiment_memo(dirname, content)