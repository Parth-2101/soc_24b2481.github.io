import pandas as pd
poky=pd.read_csv ('Pandas/pokemon_data.csv')
print(poky.groupby(['Type 1']).mean(numeric_only=True).sort_values('Defense'))
