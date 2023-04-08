import pandas as pd


def predictions_to_csv(filename, predictions):
    ids = list(range(1, 2001))
    df = pd.DataFrame({"Id": ids, "Prediction": predictions})
    df.to_csv(filename, sep=',', index=False)
