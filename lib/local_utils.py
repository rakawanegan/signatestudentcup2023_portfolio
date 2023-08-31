import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate
import configparser
import subprocess
import joblib
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import seaborn as sns


def run_command(command):
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed with error: {e.stderr}")
        return None


def read_config(path,name):
    config = configparser.ConfigParser()
    config.read(path)
    config_dict = dict(config[name])
    type_dict = {"int":int,"float":float,"str":str}
    for key,value in config_dict.items():
        type_, value = value.split(" ")
        config_dict[key] = type_dict[type_](value)
    return config_dict


def send_email(subject:str, body:str) -> bool:
    dic = read_config("./config.ini","gmail")
    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    smtpobj.starttls()
    smtpobj.login(dic["adress"], dic["password"])

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = dic["adress"]
    msg['To'] = dic["to"]
    msg['Date'] = formatdate()

    smtpobj.send_message(msg)
    smtpobj.close()
    return True

def create_experiment_memo(dir, content):
    file_name = os.path.join(dir, "experiment_description.md")

    with open(file_name, "w") as f:
        f.write(content)

def generate_experiment_memo(dir:str, experiment_info:dict):
    memo_content = f"# Lab Notebook\n\n"

    for key, value in experiment_info.items():
        memo_content += f"\n## {key}\n{value}\n"

    create_experiment_memo(dir, memo_content)

def make_pricerelationdf(feature:pd.DataFrame, y_train:pd.Series, is_show=False):
    col = feature.columns
    if len(col) != 2:
        raise ValueError("feature must have 2 columns")
    pricerelationdf = feature.copy()
    pricerelationdf["price"] = y_train
    pricerelationdf = pricerelationdf.groupby(col).mean().reset_index().pivot(index=col[0], columns=col[1], values="price")
    if is_show:
        plt.figure(figsize=(20, 10))
        sns.heatmap(pricerelationdf, annot=True, fmt=".0f", cmap="Blues")
    return pricerelationdf

def make_sizerelationdf(feature:pd.DataFrame, is_show=False):
    cols = feature.columns
    if len(cols) != 2:
        raise ValueError("feature must have 2 columns")
    sizerelationdf = feature.groupby(cols[0])[cols[1]].value_counts().unstack().fillna(0)
    if is_show:
        plt.figure(figsize=(20, 10))
        sns.heatmap(sizerelationdf, annot=True, fmt=".0f", cmap="Blues")
    return sizerelationdf