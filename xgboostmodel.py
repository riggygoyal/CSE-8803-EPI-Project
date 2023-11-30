import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('260_weeks_data.csv')

# Filter data for region 10
df_region_10 = df[df['Region'] == 10]
df_region_10 = df_region_10[::-1]

# Use 'Week of' as the datetime index
df_region_10.loc[:, 'Week of'] = pd.to_datetime(df_region_10['Week of'])
df_region_10.set_index('Week of', inplace=True)

# Feature engineering: You may need to add additional features based on your dataset
# For simplicity, we will use the average temperature as the only feature
X = df_region_10[['Avg. temperature']]
y = df_region_10['Regional Illnesses']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define the parameter grid for XGBoost
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Create an XGBoost regressor
xgb_model = XGBRegressor(objective='reg:squarederror')

# Use GridSearchCV to perform grid search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Create and train the XGBoost model with the best hyperparameters
best_model = XGBRegressor(objective='reg:squarederror', **best_params)
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE with Grid Search: {rmse}')

# Plot the actual vs predicted values for the last weeks
plt.plot(np.array(df_region_10.index[-len(y_test):]), np.array(y_test), label='Actual')
plt.plot(np.array(df_region_10.index[-len(y_test):]), y_pred, label='Predicted')
plt.xlabel('Week of')
plt.ylabel('Regional Illnesses')
plt.title('XGBoost Model with Grid Search - Actual vs Predicted')
plt.legend()
plt.savefig('XGBoost_Plot_Region10.png')