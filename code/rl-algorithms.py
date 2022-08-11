# get wikipedia table using pandas and print as latex
import pandas as pd
import numpy as np

URL = "https://en.wikipedia.org/wiki/Reinforcement_learning"

if __name__ == "__main__":

    # read in table
    df = pd.read_html(URL)[1]

    # remove column "Description"
    df = df.drop(columns=["Description"])

    print(df.to_latex(escape=False,index=False))
    
    # remove columns
