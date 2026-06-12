import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from src.loaddataset import load_dataset
from src.trainmodel import train_model
from src.evaluatemodel import evaluate_model
from src.predict import predict_species

load_dataset()

data = pd.read_csv(
    "dataset/iris.csv"
)

os.makedirs(
    "outputs",
    exist_ok=True
)

sns.pairplot(
    data,
    hue="species"
)

plt.savefig(
    "outputs/pairplot.png"
)

plt.close()

train_model()

evaluate_model()

predict_species()