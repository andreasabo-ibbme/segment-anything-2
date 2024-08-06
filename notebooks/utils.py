import pandas as pd
import numpy as np
from icecream import ic


def read_points(input_file, image_name):
    df = pd.read_csv(input_file)
    df.rename(
        columns={"Unnamed: 2": "image_name", "combined_scorer": "combined_scorer.0"},
        inplace=True,
    )
    row = df[df["image_name"] == image_name]

    cols = [col for col in df.columns.to_list() if "combined_scorer" in col]
    points = []
    for col in cols:

        col_num = int(col.split(".")[-1])
        if col_num % 2 == 1:
            continue

        # Take this point and the one after it
        points.append(
            
                [
                    float(row[f"combined_scorer.{col_num}"].values[0]),
                    float(row[f"combined_scorer.{col_num+ 1}"].values[0]),
                ]
        )

    points = np.array(points)
    return points
