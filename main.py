from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import pandas as pd

dataframe = pd.read_csv('dataset/Housing.csv')

X = dataframe.drop('price', axis=1)
y = dataframe['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=25)

# model = DecisionTreeRegressor()
model = GradientBoostingRegressor()
# model = RandomForestRegressor()
# model = Lasso()
# model = Ridge()
# model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

# print(pred)

rmse = np.sqrt(mean_squared_error(y_test, pred))
joblib.dump(model, 'HousingPrediction.pkl')
score = r2_score(y_test, pred)
print(rmse)
print(score)
