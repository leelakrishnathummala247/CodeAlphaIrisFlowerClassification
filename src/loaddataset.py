from sklearn.datasets import load_iris
import pandas as pd
import os

def load_dataset():

    iris = load_iris()

    data = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    data["species"] = iris.target

    os.makedirs("dataset", exist_ok=True)

    data.to_csv(
        "dataset/iris.csv",
        index=False
    )

    print("Dataset saved successfully.")

    return data