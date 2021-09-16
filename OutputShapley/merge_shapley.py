import pandas as pd
import os

list_files=[x for x in os.listdir() if x[-3:]=='csv']

df_shapley=pd.DataFrame()

for f in list_files:
    print(f)
    temp=pd.read_csv(f,index_col=0)
    df_shapley=df_shapley.append(temp,ignore_index=True)

df_shapley.to_csv('../DataShapley.csv')
