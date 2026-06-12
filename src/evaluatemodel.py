import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():

    data = pd.read_csv("dataset/iris.csv")

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    with open("models/irismodel.pkl", "rb") as file:
        model = pickle.load(file)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("Accuracy :", accuracy)

    report = classification_report(
        y_test,
        predictions
    )

    os.makedirs("outputs", exist_ok=True)

    with open(
        "outputs/classificationreport.txt",
        "w"
    ) as file:
        file.write(report)

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d"
    )

    plt.title("Confusion Matrix")

    plt.savefig(
        "outputs/confusionmatrix.png"
    )

    plt.close()

    print("Evaluation completed.")