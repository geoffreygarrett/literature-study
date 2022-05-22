import pandas as pd

URL = "https://en.wikipedia.org/wiki/Deep_space_bands"

dfs = pd.read_html(URL)

print(dfs[0].to_latex(index=False))
