import pandas as pd


def build_input_frame(sepal_length, sepal_width, petal_length, petal_width):
    data = {
        "sepal length (cm)": [sepal_length],
        "sepal width (cm)": [sepal_width],
        "petal length (cm)": [petal_length],
        "petal width (cm)": [petal_width],
    }
    return pd.DataFrame(data)
