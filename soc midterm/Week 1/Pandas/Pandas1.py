import pandas as pd
poky=pd.read_csv('Pandas/pokemon_data.csv')
print(poky.columns)
print(poky['Name'][0:7])
print(poky.iloc[1:4])
print(poky.iloc[2,1])
for index,row in poky.iterrows():
    print(index,row['Name'])
print(poky.loc[poky['Type 1']=="Fire"])
