import pandas as pd

df = pd.read_csv('D:/Data Sets/penguins_size.csv')

df.dropna(inplace=True) #removing Null values

df.drop('island', axis=1, inplace=True)