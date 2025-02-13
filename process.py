from IPython.display import Markdown, display

import logging

from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from category_encoders import TargetEncoder
import pandas as pd
import numpy as np

np.random.seed(42)
rng = np.random.default_rng(42)

import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    # OneHotEncoder,
    LabelEncoder,
    FunctionTransformer,
    RobustScaler,
)
from sklearn.compose import make_column_transformer

from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    ArbitraryNumberImputer,
    CategoricalImputer,
)
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection,
    MRMR,
)

from feature_engine.scaling import MeanNormalizationScaler
from feature_engine.encoding import OneHotEncoder

from feature_engine.preprocessing import MatchVariables, MatchCategories

from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.selection import RecursiveFeatureElimination
from scipy.stats import ks_2samp

from debug_pipeline import DebugPipeline

from pathlib import Path

data_path = Path("./data")


class DataProcessor(TransformerMixin):

    def __init__(self, data_path=Path("./data")):
        logging.debug("Initialization Complete")

    def create_processing_pipeline(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return Pipeline(
            steps=[
                ("imputer", self._create_imputer_pipeline(X, y)),
                ("debug_imputer", DebugPipeline("Imputer")),
                ("cat", self._create_categorical_pipeline(X, y)),
                ("debug_cat", DebugPipeline("Categorical")),
                ("selector", self._create_feature_selector_pipeline(X, y)),
                ("debug_selector", DebugPipeline("Feature Selector")),
                ("match_variables", MatchVariables()),
                ("match_categories", MatchCategories(missing_values="ignore")),
                ("fix_match_categories", CategoricalImputer()),
                ("debug_match", DebugPipeline("Match Variables")),
                ("encoder", self._create_encoder_pipeline(X, y)),
            ],
            verbose=True,
        )

    def _create_imputer_pipeline(self, X, y=None):

        MissingIndicator = AddMissingIndicator(
            variables=["LotFrontage", "GarageYrBlt", "MasVnrArea"]
        )
        NoneImputer = CategoricalImputer(
            variables=[
                "PoolQC",
                "Fence",
                "Alley",
                "MiscFeature",
                "FireplaceQu",
                "GarageFinish",
                "GarageQual",
                "GarageCond",
                "GarageType",
                "BsmtExposure",
                "BsmtCond",
                "BsmtQual",
                "BsmtFinType2",
                "BsmtFinType1",
                "Electrical",
            ],
            fill_value="None",
        )
        OtherImputer = CategoricalImputer(variables=["MasVnrType"], fill_value="Other")
        ZeroImputer = ArbitraryNumberImputer(
            variables=["LotFrontage", "GarageYrBlt", "MasVnrArea"], arbitrary_number=0
        )

        # CatchAllImputer = ColumnTransformer(
        #     transformers=[
        #         (
        #             "none_imputer",
        #             SimpleImputer(strategy="constant", fill_value=None),
        #             cat_features,
        #         ),
        #         ("mean_imputer", SimpleImputer(strategy="mean"), num_features),
        #     ],
        #     verbose=True,
        #     verbose_feature_names_out=False,
        # )

        pipeline = Pipeline(
            steps=[
                ("missing_indicator", MissingIndicator),
                ("none_imputer", NoneImputer),
                ("other_imputer", OtherImputer),
                ("debug", DebugPipeline()),
                ("zero_imputer", ZeroImputer),
                # ("catch_all_imputer", CatchAllImputer),
                ("catch_all_mean_imputer", MeanMedianImputer("mean")),
                ("catch_all_missing_imputer", CategoricalImputer()),
            ],
            verbose=True,
        )

        # if df is not None:
        #     pipeline.add("catch_all_imputer", CatchAllImputer)

        # logging.debug(f"Imputer Pipeline: {pipeline}")
        return pipeline

    def _create_categorical_pipeline(self, X, y=None):
        cat_transformer = FunctionTransformer(
            func=self._categorical_preprocess, validate=False
        )

        return Pipeline(
            steps=[
                ("categorical_processor", cat_transformer),
                # ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ],
            verbose=True,
        )

    def _categorical_preprocess(self, X, y=None):
        logging.info("Preprocessing Categorical Features")
        # logging.debug(f"Original Columns: {X.columns}")
        X["MSZoning"] = pd.Categorical(X["MSZoning"], ordered=False)
        # X["Street"] = pd.Categorical(
        #     X["Street"], categories=self.label_encode_order(X, "Street"), ordered=True
        # )
        # X["Alley"] = pd.Categorical(
        #     X["Alley"], categories=self.label_encode_order(X, "Alley"), ordered=True
        # )
        # X["LotShape"] = pd.Categorical(
        #     X["LotShape"],
        #     categories=self.label_encode_order(X, "LotShape"),
        #     ordered=True,
        # )
        # X["LandContour"] = pd.Categorical(
        #     X["LandContour"],
        #     categories=self.label_encode_order(X, "LandContour"),
        #     ordered=True,
        # )
        # X["Utilities"] = pd.Categorical(
        #     X["Utilities"],
        #     categories=self.label_encode_order(X, "Utilities"),
        #     ordered=True,
        # )
        X["LotConfig"] = pd.Categorical(X["LotConfig"], ordered=False)
        # X["LandSlope"] = pd.Categorical(
        #     X["LandSlope"],
        #     categories=self.label_encode_order(X, "LandSlope"),
        #     ordered=True,
        # )

        # X["Neighborhood"] = pd.Categorical(
        #     X["Neighborhood"],
        #     categories=self.label_encode_order(X, "Neighborhood"),
        #     ordered=True,
        # )

        # create NearHighVolStreet ordered categorical feature
        X["NearHighVolStreet"] = "None"
        X.loc[
            (X.Condition1 == "Feedr") | (X.Condition2 == "Feedr"), "NearHighVolStreet"
        ] = "Feedr"
        X.loc[
            (X.Condition1 == "Artery") | (X.Condition2 == "Artery"),
            "NearHighVolStreet",
        ] = "Artery"

        # X["NearHighVolStreet"] = pd.Categorical(
        #     X["NearHighVolStreet"],
        #     categories=self.label_encode_order(X, "NearHighVolStreet"),
        #     ordered=True,
        # )

        # create NearPosFeature ordered categorical feature
        X["NearPosFeature"] = "Normal"
        X.loc[(X.Condition1 == "PosN") | (X.Condition2 == "PosN"), "NearPosFeature"] = (
            "Pos"
        )
        X.loc[(X.Condition1 == "PosA") | (X.Condition2 == "PosA"), "NearPosFeature"] = (
            "Pos"
        )

        # X["NearPosFeature"] = pd.Categorical(
        #     X["NearPosFeature"],
        #     categories=self.label_encode_order(X, "NearPosFeature"),
        #     ordered=True,
        # )

        # create NearRailRoad ordered categorical feature
        X["NearRailRoad"] = "None"
        X.loc[(X.Condition1 == "RRNn") | (X.Condition2 == "RRNn"), "NearRailRoad"] = (
            "RRNn"
        )
        X.loc[(X.Condition1 == "RRAn") | (X.Condition2 == "RRAn"), "NearRailRoad"] = (
            "RRAn"
        )
        X.loc[(X.Condition1 == "RRAe") | (X.Condition2 == "RRAe"), "NearRailRoad"] = (
            "RRAe"
        )
        X.loc[(X.Condition1 == "RRNe") | (X.Condition2 == "RRNe"), "NearRailRoad"] = (
            "RRNe"
        )

        # TODO: Play with this ordering
        # X["NearRailRoad"] = pd.Categorical(
        #     X["NearRailRoad"],
        #     categories=self.label_encode_order(X, "NearRailRoad"),
        #     ordered=True,
        # )

        # drop Condition1 and Condition2 variables as we have extracted the information
        X = X.drop(["Condition1", "Condition2"], axis=1)

        # TODO: play with this as ordered or in multiple variables
        X["BldgType"] = pd.Categorical(X["BldgType"], ordered=False)

        # X["HouseStyle"] = pd.Categorical(
        #     X["HouseStyle"],
        #     categories=self.label_encode_order(X, "HouseStyle"),
        #     ordered=True,
        # )

        X["OverallQual"] = pd.Categorical(
            X["OverallQual"], categories=np.arange(1, 11), ordered=True
        )
        X["OverallCond"] = pd.Categorical(
            X["OverallCond"], categories=np.arange(1, 11), ordered=True
        )

        return X

    def _create_feature_selector_pipeline(self, X, y=None):

        mrmr = MRMR(
            method="RFCQ",
            # max_features=40,
            regression=True,
            scoring="neg_mean_squared_error",
        )

        rfr = RecursiveFeatureElimination(
            estimator=RandomForestRegressor(),
            scoring="neg_mean_squared_error",
            threshold=0.01,
            cv=5,
        )

        return Pipeline(
            steps=[
                ("drop_constant", DropConstantFeatures(tol=0.99)),
                ("drop_duplicate", DropDuplicateFeatures()),
                # (
                #     "smart_correlated",
                #     SmartCorrelatedSelection(
                #         selection_method="variance",
                #         threshold=0.8,
                #         # estimator="rf",
                #     ),
                # ),
                ("mrmr", mrmr),
                ("rfr", rfr),
            ],
            verbose=True,
        )

    def _create_encoder_pipeline(self, X, y=None):

        target_encoder_vars = [
            "Street",
            "Alley",
            "LotShape",
            "LandContour",
            "Utilities",
            "LandSlope",
            "Neighborhood",
            "NearHighVolStreet",
            "NearPosFeature",
            "NearRailRoad",
            "HouseStyle",
        ]

        target_encoder_vars = [
            col
            for col in target_encoder_vars
            if col in X.select_dtypes(include=["object", "category"]).columns
        ]

        onehot_vars = [
            col
            for col in X.select_dtypes(include=["object", "category"]).columns
            if col not in target_encoder_vars
        ]

        class MyTargetEncoder(TargetEncoder):
            def fit(self, X, y=None, **kwargs):
                self.cols = [col for col in self.cols if col in X.columns]
                return super().fit(X, y, **kwargs)

            def transform(self, X, y=None, **kwargs):

                self.cols = [col for col in self.cols if col in X.columns]
                return super().transform(X, y=None, **kwargs)

            def fit_transform(self, X, y=None):
                return super().fit_transform(X, y)

        scaler = MeanNormalizationScaler()
        onehot = OneHotEncoder(drop_last=True)  # , variables=onehot_vars)

        return Pipeline(
            steps=[
                # (
                #     "target_encoder",
                #     MyTargetEncoder(cols=target_encoder_vars, verbose=10),
                # ),
                ("onehot", onehot),
                ("scaler", scaler),
            ],
            verbose=True,
        )

    def metric(self, y_true, y_pred, as_scoring=False):
        y_true = list(y_true)
        y_pred = list(y_pred)

        _metric = np.sqrt(np.mean((np.log(y_true) - np.log(y_pred)) ** 2))
        return -_metric if as_scoring else _metric

    def inverse_metric(self, y_true, y_pred):
        return 1 / self.metric(y_true, y_pred)

    def fit(self, X, y=None):
        # X = X.drop("SalePrice", axis=1, errors="ignore")

        logging.debug(f"FIT Columns: {len(X.columns)}, {X.columns}")
        assert y is not None, "y cannot be None"
        self.orig_df = pd.concat([X, y], axis=1)

        self.pipeline = self.create_processing_pipeline(X, y)
        return self.pipeline.fit(X, y)
        # return self

    def transform(self, X, y=None):
        # X = X.drop("SalePrice", axis=1, errors="ignore")

        logging.debug(f"TRANSFORM Columns: {len(X.columns)}, {X.columns}")
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):

        return self.fit(X, y).transform(X)

    def ks_compare_distributions(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        drop_cols: str | list[str] = ["Id", "SalePrice"],
        plot=False,
    ):
        if drop_cols:
            df1 = df1.drop(drop_cols, axis=1, errors="ignore")
            df2 = df2.drop(drop_cols, axis=1, errors="ignore")

        ks_results = [
            ks_2samp(df1[col], df2[col]) for col in df1.columns if col in df2.columns
        ]
        ks_results = pd.DataFrame(
            ks_results, index=df1.columns, columns=["statistic", "pvalue"]
        )
        ks_results["significant_05"] = ks_results["pvalue"] < 0.05
        ks_results["significant_10"] = ks_results["pvalue"] < 0.1

        if plot:
            self._ks_plot(df1, df2, ks_results)

        return ks_results

    def _ks_plot(self, df1, df2, ks_results):
        significant_cols = ks_results[ks_results["significant_10"]].index.tolist()
        display(Markdown("### Kolmogorov-Smirnov Test Results"))
        display(ks_results.loc[significant_cols].sort_values("pvalue"))

        for col in significant_cols:
            fig, ax = plt.subplots()
            sns.histplot(df1[col], ax=ax, color="blue", alpha=0.5, label="df1")
            sns.histplot(df2[col], ax=ax, color="red", alpha=0.5, label="df2")
            ax.set_title(
                f"{col} 2SKS Test p-value: {ks_results.loc[col, 'pvalue']:.4f}"
            )
            ax.legend()
            plt.show()
        return None

    def get_nan_cols(self, df, do_pipeline=True, fit=True):
        if do_pipeline:
            if fit:
                self.fit(df)
            df = self.transform(df)

        nans = df.isna().sum().sort_values(ascending=False)
        return nans[nans > 0]

    def label_encode_order(
        self, df, col, label="SalePrice", aslist=True, ascending=True
    ):
        """
        Encodes a categorical column based on the mean value of another column (Typically the dependent variable).

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        col (str): The name of the column to encode.
        label (str, optional): The name of the column to calculate the mean values. Default is "SalePrice".
        aslist (bool, optional): If True, returns the encoded values as a list. If False, returns a Series. Default is True.
        ascending (bool, optional): If True, sorts the mean values in ascending order. If False, sorts in descending order. Default is True.

        Returns:
        list or pd.Series: The encoded values as a list or Series, depending on the value of aslist.
        """

        assert col in df.columns, f"Column '{col}' not in DataFrame columns"

        if self.orig_df is None and col in df.columns and label in df.columns:
            self.orig_df = df

        if label not in df.columns:
            df = self.orig_df

        values = df.groupby(col)[label].mean().round(2).sort_values(ascending=ascending)

        if aslist:
            values = values.index.to_list()

        return values
