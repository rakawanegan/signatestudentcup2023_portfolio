import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
import unicodedata
import joblib


class Preprocess:
    def __init__(self, SEED=314):
        self.SEED = SEED
        self.si = SimpleImputer(strategy="mean")
        self.mms = MinMaxScaler()
        self.eps = 1e-10

    def get_data(self, is_highprice=False):
        x_test = pd.read_csv('data/official_data/test.csv', index_col="id")
        x_train = pd.read_csv('data/official_data/train.csv', index_col="id").drop(columns=['price'])
        y_train = pd.read_csv('data/official_data/train.csv', index_col="id", usecols=["id", "price"])
        if is_highprice:
            x_train = pd.read_csv("data/additional_data/highprice/x_train.csv", index_col="id")
            y_train = pd.read_csv("data/additional_data/highprice/y_train.csv", index_col="id")
        return x_train, y_train, x_test

    def _convert_cylinders(self, x):
        return re.sub("[^0-9]", "", x) if x!="other" else np.NaN

    def _strnormalize(self, s):
        error_alphabet_dict = joblib.load("data/additional_data/error_alphabet_dict.pkl")
        s = unicodedata.normalize("NFKC", s).lower()
        for k,v in error_alphabet_dict.items():
            if k in s:
                s = s.replace(k,v)
        return s

    def _yearconjecture(self, x):
        return x if x<=2023 else x-1000

    def _odometerconjecture(self, x):
        return -1*x if x < -1 else (x//10 if x > 1e+6 else x)

    def _calc_odometer(self,x):
        return min(max((2023 - x) * self.odometerperyear, self.odometerlimitmin), self.odometerlimitmax)

    def _fit(self, x:pd.DataFrame, y:pd.DataFrame, score_by="mean", is_dropregion=True, is_impute=False, is_scale=False, is_ohe=True, is_emb=True):
        self.condition2score = dict()
        for condtion in x["condition"].unique():
            self.condition2score[condtion] = y[x["condition"] == condtion]["price"].median()
            if score_by == "mean":
                self.condition2score[condtion] = y[x["condition"] == condtion]["price"].mean()

        self.title_status2score = dict()
        for title_status in x["title_status"].unique():
            self.title_status2score[title_status] = y[x["title_status"] == title_status]["price"].median()
            if score_by == "mean":
                self.title_status2score[title_status] = y[x["title_status"] == title_status]["price"].mean()

        self.size2score = joblib.load("data/additional_data/size_sort_dict.pkl")
        rs_df = x[["region", "state"]][x.region.isin(x["region"].unique())].drop(x[x.state.isna()].index, axis=0)
        self.region2state = dict(zip(rs_df.region, rs_df.state))
        self.region2state["ashtabula"] = "oh"
        self.region2state["northwest KS"] = "ka"
        self.region2state["southern WV"] = "wv"
        self.duplicate_state_dict = {'ks':'ka','dc':'wa'}
        self.odometerperyear = (x.loc[x.odometer.notna(),"odometer"] / (2023 - x.loc[x.odometer.notna(),"year"])).mean()
        # self.odometerlimitmax = x.odometer.max()
        self.odometerlimitmax = x.odometer.quantile(0.95)
        self.odometerlimitmin = x.loc[x.odometer > 0,"odometer"].min()
        x["size"] = x["size"].map(self.size2score)
        self.x_type = x["type"].unique()
        self.x_size = x["size"].unique()
        for cat1 in self.x_type:
            for cat2 in self.x_size:
                x.loc[((x["type"] == cat1) & (x["size"] == cat2)), "type_size"] = f"{cat1}_{cat2}"
        self.cylinders_fill_dict = x.groupby("cylinders")["type_size"].value_counts().unstack().fillna(0).apply(lambda x: x / x.sum(), axis=0).idxmax(axis=0).to_dict()
        self.cylinders_fill_dict = {k:self._convert_cylinders(v) for k,v in self.cylinders_fill_dict.items()}

        for cat1 in x["manufacturer"].unique():
            for cat2 in x["type"].unique():
                x.loc[((x["manufacturer"] == cat1) & (x["type"] == cat2)), "manufacturer_type"] = f"{cat1}_{cat2}"
        self.fuel_fill_dict = x.groupby("fuel")["manufacturer_type"].value_counts().unstack().fillna(0).apply(lambda x: x / x.sum(), axis=0).idxmax(axis=0).to_dict()

        self.categorylist = ['manufacturer', 'fuel', 'transmission', 'drive', 'size', 'type']
        self.odometerratiodict = dict()
        self.yearratiodict = dict()
        for category in self.categorylist:
            self.odometerratiodict[f"odometerper{category}"] = x.groupby(category)["odometer"].mean()
            self.yearratiodict[f"yearper{category}"] = x.groupby(category)["year"].mean()

        self.is_dropregion = is_dropregion
        self.is_impute = is_impute
        self.is_scale = is_scale
        self.is_ohe = is_ohe
        self.is_emb = is_emb

        self.pibot_feature = joblib.load("data/additional_data/feature_handle_dict.pkl")

        country2manufacturer = joblib.load("data/additional_data/manufacture2country_dict.pkl")
        self.manufacturer2country = {v:k for k,vs in country2manufacturer.items() for v in vs}
        return self

    def _base_transform(self, x:pd.DataFrame):
        x["cylinders"] = x["cylinders"].map(self._convert_cylinders)
        x.loc[((x.condition=="new")|(x.condition=="like new"))&(x.title_status=="salvage"),"condition"] = "salvage"
        x["size"] = x["size"].map(self.size2score)
        x["condition_score"] = x["condition"].map(self.condition2score)
        x["title_status_score"] = x["title_status"].map(self.title_status2score)
        x.loc[x.state.isna(),"state"] = x.loc[x.state.isna(),"region"].map(self.region2state)
        x["state"] = x["state"].map(lambda s: self.duplicate_state_dict[s] if s in self.duplicate_state_dict.keys() else s)
        for cat1 in self.x_type:
            for cat2 in self.x_size:
                x.loc[((x["type"] == cat1) & (x["size"] == cat2)), "type_size"] = f"{cat1}_{cat2}"
        x.loc[x.cylinders.isna(),"cylinders"] = x.loc[x.cylinders.isna(),"type_size"].map(self.cylinders_fill_dict)
        x["cylinders"] = x["cylinders"].astype(float)
        if self.is_dropregion:
            x = x.drop(columns=["region"])
        else:
            x["region"] = x["region"].map(self._strnormalize)
        x["manufacturer"] = x["manufacturer"].map(self._strnormalize)
        for cat1 in x["manufacturer"].unique():
            for cat2 in x["type"].unique():
                x.loc[((x["manufacturer"] == cat1) & (x["type"] == cat2)), "manufacturer_type"] = f"{cat1}_{cat2}"
        x.loc[x.fuel.isna(),"fuel"] = x.loc[x.fuel.isna(),"manufacturer"].map(self.fuel_fill_dict)
        x["manufacturer_country"] = x["manufacturer"].map(self.manufacturer2country)
        for key,value in self.pibot_feature.items():
            x[f"is_{value}"] = x[key].apply(lambda x: 1 if x == value else 0)
        x["year"] = x["year"].map(self._yearconjecture)
        x["odometer"] = x["odometer"].map(self._odometerconjecture)
        x.loc[x.odometer==-1,"odometer"] = x.loc[x.odometer==-1,"year"].map(self._calc_odometer)
        x["odometerpercar-age"] = x["odometer"] / (2023 - x["year"] + self.eps)
        x["log-odmeter"] = np.log(x["odometer"] + self.eps)
        x["odometer-inverse"] = 1 / (x["odometer"] + self.eps)
        x["car-age-inverse"] = 1 / (2023 - x["year"] + self.eps)
        x.loc[:,"truck_diesel"] = 0
        x.loc[(x.type=="truck")&(x.fuel=="diesel"),"truck_diesel"] = 1
        for category in self.categorylist:
            x[f"odometerper{category}"] = x["odometer"] / x[category].map(self.odometerratiodict[f"odometerper{category}"])
            x[f"yearper{category}"] = x["year"] / x[category].map(self.yearratiodict[f"yearper{category}"])
        if self.is_impute:
            x.loc[x.title_status.isna(),["title_status","type"]] = "dropped" # it includes meaning?
        return x

    def transform(self, x:pd.DataFrame):
        x = x.copy()
        x = self._base_transform(x)
        object_columns = x.select_dtypes(include=object).columns
        if self.is_emb==False:
            return x
        if self.is_ohe:
            x = pd.get_dummies(x, columns=object_columns)
            x = x.reindex(columns=self.ohe_columns, fill_value=0)
            bool_columns = x.select_dtypes(include=bool).columns
            x[bool_columns] = x[bool_columns].astype(float)
        else:
            x[object_columns] = self.oe.transform(x[object_columns])
        if self.is_impute:
            x = pd.DataFrame(self.si.transform(x), columns=x.columns)
        if self.is_scale:
            x = pd.DataFrame(self.mms.transform(x), columns=x.columns)
        return x

    def fit_transform(self, x:pd.DataFrame, y:pd.DataFrame, score_by="mean", is_dropregion=True, is_impute=False, is_scale=False, is_ohe=True, is_emb=True):
        x = x.copy()
        self._fit(x, y, score_by, is_dropregion, is_impute, is_scale, is_ohe)
        x = self._base_transform(x)
        object_columns = x.select_dtypes(include=object).columns
        if self.is_emb==False:
            return x
        if self.is_ohe:
            self.ohe_columns = pd.get_dummies(x, columns=object_columns).columns
            x = pd.get_dummies(x, columns=object_columns)
            x = x.reindex(columns=self.ohe_columns, fill_value=0)
            bool_columns = x.select_dtypes(include=bool).columns
            x[bool_columns] = x[bool_columns].astype(float)
        else:
            self.oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
            x[object_columns] = self.oe.fit_transform(x[object_columns])
        if self.is_impute:
            x = pd.DataFrame(self.si.fit_transform(x), columns=x.columns)
        if self.is_scale:
            x = pd.DataFrame(self.mms.fit_transform(x), columns=x.columns)
        return x
