from sklearn.base import BaseEstimator, TransformerMixin
from IPython.display import display


class DebugPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    # self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # display(f"Before transformation: \n{X.head()}")
        print(f"\n\nDebugging: {self.name}")
        print(f"Debug Shape: {X.shape}\n")
        # print(
        #     f"Numberic Columns: {X.select_dtypes(include='number').columns.to_list()}"
        # )
        # print(
        #     f"Categorical Columns: {X.select_dtypes(include=['object', 'category']).columns.to_list()}"
        # )
        print(f"Debug nans: \n{X.isnull().sum().sort_values(ascending=False)[:5]}")
        return X
