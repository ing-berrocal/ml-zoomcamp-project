import pickle


with open('model/model_life_expectancy_edaix.pkl', 'rb') as f:
    pipeline, model_estimator, columns, country, status = pickle.load(f)

import numpy as np
import pandas as pd
df_concrete = pd.read_csv('./data/Life_Expectancy_Data.csv')

# %%
# eliminar espacios y pasar nombres a minusculas de las columnas
df_concrete.columns = df_concrete.columns.str.strip().str.lower().str.replace(' ', '_')


# %%
df_train = df_concrete[(df_concrete['year'] < 2015)]
df_val = df_concrete[(df_concrete['year'] >= 2015)]


# %%
df_train_cl = df_val.dropna(subset=['life_expectancy'])
df_y = df_train_cl['life_expectancy']

# %%
df_train_cl['country_idx'] = df_train_cl['country'].map(country)
df_train_cl['status_idx'] = df_train_cl['status'].map(status)

# %%
y_test = df_train_cl.life_expectancy
X_test = df_train_cl.drop(columns='life_expectancy')


# %%
df_numeric = X_test.select_dtypes(include=[np.number])

# %%
# llenamos columnas nullas con la media de cada columna
df_numeric_imputed = pd.DataFrame(pipeline.fit_transform(df_numeric), columns=df_numeric.columns)


# %%
y_predict = model_estimator.predict(df_numeric_imputed[columns])

# %%
df_numeric_imputed.loc[0,columns].to_json()

# %%
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print(f'R2 score: {r2}')

json = df_numeric_imputed.iloc[0,:].to_json()

print(json)