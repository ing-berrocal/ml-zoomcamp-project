
import numpy as np
import pandas as pd

df_concrete = pd.read_csv('./data/Life_Expectancy_Data.csv')
df_concrete.columns = df_concrete.columns.str.strip().str.lower().str.replace(' ', '_')
df_cleaned_target = df_concrete.dropna(subset=['life_expectancy'])
df_cleaned_target.select_dtypes(include=['object']).nunique()

# %%
# Seleccionar las columnas numericas y generar una matrix de correlacion
numeric_cols = df_cleaned_target.select_dtypes(include=[np.number]).columns

countries = df_cleaned_target.country.unique()
status = df_cleaned_target.status.unique()

country_mapping = {country: idx for idx, country in enumerate(countries)}
status_mapping = {status: idx for idx, status in enumerate(status)}
df_cleaned_target['country_idx'] = df_cleaned_target['country'].map(country_mapping)
df_cleaned_target['status_idx'] = df_cleaned_target['status'].map(status_mapping)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

imputer = SimpleImputer()
steps = [('imputation', SimpleImputer())]
pipeline = Pipeline(steps)

# %%
df_numeric = df_cleaned_target[['life_expectancy', 'adult_mortality',
       'infant_deaths', 'alcohol', 'percentage_expenditure', 'hepatitis_b',
       'measles', 'bmi', 'under-five_deaths', 'polio', 'total_expenditure',
       'diphtheria', 'hiv/aids', 'gdp', 'population', 'thinness__1-19_years',
       'thinness_5-9_years', 'income_composition_of_resources', 'schooling']]


# llenamos columnas nullas con la media de cada columna
df_numeric_imputed = pd.DataFrame(pipeline.fit_transform(df_numeric), columns=df_numeric.columns)

# 
df_data = pd.concat([df_cleaned_target[['country_idx', 'status_idx','year']].reset_index(drop=True), df_numeric_imputed.reset_index(drop=True)], axis=1)


df_train = df_data[(df_data['year'] < 2015)]
df_val = df_data[(df_data['year'] >= 2015)]


from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


X = df_train.drop('life_expectancy',axis=1)
y = df_train['life_expectancy']

# %%
tscv = TimeSeriesSplit(5)

# %%
# import Linear

model = LinearRegression()  
scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')

print(scores)

# %%
# radomforestregressor

model = RandomForestRegressor()  

scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')

print(scores)

# %%
# import xgboost
model = xgb.XGBRegressor()  

scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')

print(scores)

# %% [markdown]
# ## Results
# 
# Best models are XGBRegressor and RandomForestRegressor

# %%
# train_test_split

X_train = df_train[(df_train['year'] < 2014)].drop('life_expectancy', axis=1)
X_test = df_train[(df_train['year'] >= 2014)].drop('life_expectancy', axis=1)
y_train = df_train[(df_train['year'] < 2014)]['life_expectancy']
y_test = df_train[(df_train['year'] >= 2014)]['life_expectancy']


# %%
n_estmimators = [10,20,35,50, 100, 200, 300, 400, 500]
max_depth = [10,15,20,25,30]

bagging_results = []

for n in n_estmimators:
    for depth in max_depth:
        model = RandomForestRegressor(n_estimators=n, max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Random Forest Regressor with {n} estimators and max depth {depth}: R² score = {score}")
        bagging_results.append({'estimators':n, 'depth':depth, 'score':score})

# %%
# generar dataframe desde bagging_results
df_bagging_results = pd.DataFrame(bagging_results)

# %%
df_bagging_results.pivot(index='estimators', columns='depth', values='score')

# %%
# visualizar el mejor resultado
random_forest_best_result = df_bagging_results.loc[df_bagging_results['score'].idxmax()]
print("Mejor score", random_forest_best_result['score'])
print("Estimators:", random_forest_best_result['estimators'], "Depth:", random_forest_best_result['depth'])

# %% [markdown]
# ## Gradient

# %%
# Boosting
from sklearn.ensemble import GradientBoostingRegressor

# %%
n_estmimators = [50, 100, 200, 300, 400, 500]
learning_rates = [0.01, 0.05, 0.1, 0.2]
max_depths = [5, 7, 8]

gradboosting_results = []

for n in n_estmimators:
    for lr in learning_rates:
        for depth in max_depths:
            model = GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"Gradient Boosting Regressor with {n} estimators, learning rate {lr}, and max depth {depth}: R² score = {score}")
            gradboosting_results.append({'estimators':n, 'learning_rate':lr, 'depth':depth, 'score':score})

# %%
# generar dataframe desde gradboosting_results
df_gradboosting_results = pd.DataFrame(gradboosting_results)

# %%
df_gradboosting_results.pivot(index='estimators', columns=['depth', 'learning_rate'], values='score')  

# %%
# visualizar el mejor resultado
gradient_boosting_best_result = df_gradboosting_results.loc[df_gradboosting_results['score'].idxmax()]
print("Mejor score", gradient_boosting_best_result['score'])
print("Estimators:", gradient_boosting_best_result['estimators'], "Depth:", gradient_boosting_best_result['depth'], "Learning Rate:", gradient_boosting_best_result['learning_rate'])

# %%
#AdaBoost
from sklearn.ensemble import AdaBoostRegressor

# %%
n_estmimators = [100, 200, 300, 400, 500]
learning_rates = [0.01, 0.05, 0.1, 0.2]
max_depths = [3, 5, 7]

adaboost_results = []

for n in n_estmimators:
    for lr in learning_rates:
        for depth in max_depths:
            model = AdaBoostRegressor(n_estimators=n, learning_rate=lr, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"AdaBoost Regressor with {n} estimators, learning rate {lr}, max depth {depth}: R² score = {score}")
            adaboost_results.append({'estimators':n, 'learning_rate':lr, 'depth':depth, 'score':score})

# %%
# generar dataframe desde adaboost_results
df_adaboost_results = pd.DataFrame(adaboost_results)

# %%
df_adaboost_results.pivot(index='estimators', columns=['depth', 'learning_rate'], values='score')  

# %%

adaboost_best_result = df_adaboost_results.loc[df_adaboost_results['score'].idxmax()]
print("Mejor score", adaboost_best_result['score'])
print("Estimators:", adaboost_best_result['estimators'], "Depth:", adaboost_best_result['depth'], "Learning Rate:", adaboost_best_result['learning_rate'])

# %% [markdown]
# ## XBoost

# %%
# xboost
n_estmimators = [10,20,35,50, 100, 200, 300, 400, 500]
learning_rates = [0.01, 0.05, 0.1, 0.2]
max_depths = [3, 5, 7]

xgboost_results = []

for n in n_estmimators:
    for lr in learning_rates:
        for depth in max_depths:
            model = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = lr, max_depth = depth, n_estimators = n)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"XGBoost Regressor with {n} estimators, learning rate {lr}, and max depth {depth}: R² score = {score}")
            xgboost_results.append({'estimators':n, 'learning_rate':lr, 'depth':depth, 'score':score})

# %%
df_xgboost_results = pd.DataFrame(xgboost_results)

# %%
df_xgboost_results.pivot(index='estimators', columns=['depth', 'learning_rate'], values='score')  

# %%
# visualizar el mejor resultado
xgb_best_result = df_xgboost_results.loc[df_xgboost_results['score'].idxmax()]
print("Mejor score", xgb_best_result['score'])
print("Estimators:", xgb_best_result['estimators'], "Depth:", xgb_best_result['depth'], "Learning Rate:", xgb_best_result['learning_rate'])

# %% [markdown]
# ## Model

# %%
#model = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = 0.2, max_depth = 5, n_estimators = 500)
#model = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = 0.1, max_depth = 7, n_estimators = 35)
#400.0 Depth: 7.0 Learning Rate: 0.1
model = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = 0.1, max_depth = 7, n_estimators = 400)


# %%
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

# %%
model.score(X_train, y_train)

# %%
model.score(X_test, y_test)

# %% [markdown]
# ## Feautering

# %%
model.feature_importances_


# %%
model.get_booster().get_score(importance_type='total_gain')

# %%
model.get_booster().get_score(importance_type='gain')

# %%
pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# %% [markdown]
# #data

# %%
from sklearn.feature_selection import RFE

model_estimator = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = 0.1, max_depth = 7, n_estimators = 400)
rfe = RFE(estimator=model_estimator, n_features_to_select=0.75, verbose=1)
rfe.fit(X_train, y_train)

# %%
X_train.columns[rfe.support_]

# %%
X_train.iloc[:,rfe.support_]

# %%
# xboost
n_estmimators = [35,50, 100, 200, 300, 400, 500]
learning_rates = [0.01, 0.05, 0.1, 0.2]
max_depths = [3, 5, 7]

xgboost_results = []

for n in n_estmimators:
    for lr in learning_rates:
        for depth in max_depths:
            model = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = lr, max_depth = depth, n_estimators = n)
            model.fit(X_train.iloc[:,rfe.support_], y_train)
            score = model.score(X_test.iloc[:,rfe.support_], y_test)
            print(f"XGBoost Regressor with {n} estimators, learning rate {lr}, and max depth {depth}: R² score = {score}")
            xgboost_results.append({'estimators':n, 'learning_rate':lr, 'depth':depth, 'score':score})

# %%
df_xgboost_results = pd.DataFrame(xgboost_results)

# %%
df_xgboost_results.pivot(index='estimators', columns=['depth', 'learning_rate'], values='score')  

# %%
# visualizar el mejor resultado
xgb_best_result = df_xgboost_results.loc[df_xgboost_results['score'].idxmax()]
print("Mejor score", xgb_best_result['score'])
print("Estimators:", xgb_best_result['estimators'], "Depth:", xgb_best_result['depth'], "Learning Rate:", xgb_best_result['learning_rate'])

# %% [markdown]
# - Before RFE score 0.9503033037843287
# - Estimators: 400.0 Depth: 7.0 Learning Rate: 0.1
# - Attributes : 21
# 
# ------
# 
# - After RFE score 0.9517411454297142
# - Estimators: 500.0 Depth: 7.0 Learning Rate: 0.1
# - Attributes : 15

# %%
X_test.iloc[9,rfe.support_].to_json()

# %%
y_test[0:10]

# %%
model_estimator.fit(X_train.iloc[:,rfe.support_],y_train)

# %%
columns = X_train.iloc[:,rfe.support_].columns

# %%
model_estimator.score(X_train[columns],y_train)

# %%
model_estimator.score(X_test[columns],y_test)

# %% [markdown]
# Better score

# %%
y_predict = model_estimator.predict(X_test[columns])

# %% [markdown]
# ## Save Model

# %%
#selected
columns

# %%
#import pi
import pickle

# %%
with open('model/model_life_expectancy_edaix.pkl', 'wb') as f_out:
    pickle.dump((pipeline, model_estimator,columns.values, country_mapping, status_mapping), f_out)

# %%



