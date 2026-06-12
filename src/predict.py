import pickle
import numpy as np

def predict_species():

    with open(
        "models/irismodel.pkl",
        "rb"
    ) as file:
        model = pickle.load(file)

    sample_data = np.array(
        [[5.1, 3.5, 1.4, 0.2]]
    )

    prediction = model.predict(
        sample_data
    )

    flower_names = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    print(
        "Predicted Flower :",
        flower_names[prediction[0]]
    )