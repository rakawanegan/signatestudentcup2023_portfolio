import pandas as pd
import glob
import os
import numpy as np
import datetime
from sklearn.metrics import mean_absolute_percentage_error
from pathlib import Path
from lib.local_utils import run_command, generate_experiment_memo


MODELNAME = "ensamble"
date = datetime.datetime.now().strftime('%m%d%H%M')
dirname = f"results/{MODELNAME}_{date}"
os.makedirs(dirname)
run_command(f"cp {Path(__file__).resolve()} {dirname}/main.py")

# ignore_list = ["ensamble",]

# score_list = list()
# for i in range(5):
#     eval_files = glob.glob(f'results/*/eval{i}.csv')
#     eval_files = [eval_file for eval_file in eval_files if not any(ignore in eval_file for ignore in ignore_list)]
#     eval_dfs = list()

#     for eval in eval_files:
#         eval_dfs.append(pd.read_csv(eval, header=None, names=["idx", "predict", "actual"], index_col="idx"))
#     eval_df = pd.concat(eval_dfs,axis=1)["predict"]
#     eval_actual = pd.concat(eval_dfs,axis=1)["actual"]
#     eval = pd.DataFrame(eval_df.mean(axis=1))
#     score = mean_absolute_percentage_error(eval, eval_actual.values[:,0])
#     print(f"mape score {i}: {score:.7f}")
#     score_list.append(score)
# eval = pd.DataFrame(eval_df.mean(axis=1), columns=['predict'])

test_dfs = list()
# test_files = glob.glob('results/*/submission_foldensamble.csv')
test_files = ["lgbm2lgbm_08192015/submission_allensamble.csv", "lgbm2lgbm4highprice_08232117/submission_foldensamble.csv", "lgbm2lgbm4highprice_08231616/submission_foldensamble.csv"]
test_files = [ f"results/{file_name}" for file_name in test_files]
# test_files = [test_file for test_file in test_files if not any(ignore in test_file for ignore in ignore_list)]

for test in test_files:
    test_dfs.append(pd.read_csv(test, header=None, names=["idx",'predict']))
test_df = pd.concat(test_dfs,axis=1)["predict"]


submit = pd.DataFrame(test_df.mean(axis=1), columns=['predict'])
submit['predict']=submit['predict']
submit.index = test_dfs[0]["idx"]

submit.to_csv(f"{dirname}/submission.csv", header=False)

content = {
    "MODELNAME": MODELNAME,
    "ensamble": "mean",
    # "score mean": np.mean(score_list),
    # "score each": "\n".join([f'- {i}: {score}' for i, score in enumerate(score_list)]),
    # "ensamble_list": "\n".join([f'- {eval.split("/")[1]}' for eval in eval_files]),
}

generate_experiment_memo(dirname, content)