import pandas as pd

df = pd.read_csv('D:/Data Sets/penguins_size.csv')

df.dropna(inplace=True) #removing Null values

df.drop('island', axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

for col in df.select_dtypes(include='object'):

df[col] = enc.fit_transform(df[col])

#train test split

from sklearn.model_selection import train_test_split

y = df.species

df.drop('species', axis=1,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, y,test_size=0.15)



#model train

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_pred, y_test)

print(f'The accuracy of model is {acc}')

#save model

from joblib import dump

dump(model,'penguin_model')


from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
model = load('penguin_model')

class my_input(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: int

@app.post('/predict/')
async def main(input: my_input):
    data = input.dict()
    data_ = [[data['culmen_length_mm'], data['culmen_depth_mm'], data['flipper_length_mm'], data['body_mass_g'], data['sex']]]
    species = model.predict(data_)[0]
    probability = model.predict_proba(data_).max()
    if species == 0:
        species_name = 'Adelie'
    elif species == 1:
        species_name = 'Chinstrap'
    else:
        species_name = 'Gentoo'
    return {
        'prediction': species_name,
        'probability': float(probability)
}