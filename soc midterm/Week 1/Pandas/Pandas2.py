import pandas as pd
poky=pd.read_csv('Pandas/pokemon_data.csv')
poky.to_csv('Pandas/pokemon_data.csv' , index=False)