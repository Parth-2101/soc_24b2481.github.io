import pandas as pd
poky=pd.read_csv('Pandas/pokemon_data.csv')
new_poky= poky.loc[(poky['Type 1']== 'Grass') & (poky['Type 2']=='Poison') & (poky['HP']>=70)]
new_poky.to_csv('Pandas/modified_poky.csv' , index=False)
new_poky= new_poky.reset_index(drop=True)
print(poky.loc[poky['Name'].str.contains('Mega')])