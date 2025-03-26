import pandas as pd  # Helps us work with tables (like Excel)
import numpy as np  # Helps with math calculations
import seaborn as sns  # Helps create nice charts
import matplotlib.pyplot as plt  # Helps draw graphs
from sklearn.model_selection import train_test_split  # Helps split data for training
from sklearn.linear_model import LinearRegression  # Our ML model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Measures how good the model is
df=df_numerized.copy()

x=df[['budget','Correct_year','runtime']]
y=df['gross']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape,y_train.shape)
print(x_train.isnull().sum())
print(y_train.isnull().sum())
x_train=x_train.fillna(0)
y_train=y_train.fillna(0)
print(x_train.isnull().sum())
print(y_train.isnull().sum())
print(np.isinf(x_train).sum())
print(np.isinf(y_train).sum())
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
mae = mean_absolute_error(y_test,y_pred)
mse= mean_squared_error(y_test,y_pred)
r2= r2_score(y_test,y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
# Convert categorical columns to numeric
for col in ['director', 'writer', 'company']:
    df_numerized[col] = df_numerized[col].astype('category').cat.codes

# Fill missing values with the mean (for numeric columns only)
df_numerized = df_numerized.fillna(df_numerized.mean())

df_numerized = df_numerized.fillna(df_numerized.mean())
print(x_train.isnull().sum())
x_train = x_train.fillna(x_train.mean())  # Replace NaNs with column means again
print(x_train.isnull().sum())
print(y_train.isnull().sum())
print(x_train.isnull().sum())
print(x_test.isnull().sum())
x_train = x_train.fillna(0)
x_test = x_test.fillna(0)
print(x_train.dtypes)
x_train = x_train.apply(pd.to_numeric, errors='coerce')
x_test = x_test.apply(pd.to_numeric, errors='coerce')
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
y_train = y_train.values.ravel()
print(x_train.head())
print(y_train[:5])  # This prints the first 5 values

model.fit(x_train, y_train)
# Predict values
y_pred = model.predict(x_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)
print(df.columns)

x = df_numerized[['budget', 'runtime', 'Correct_year', 'director', 'writer', 'company', 'score', 'votes']]
y = df_numerized['gross']
x_train = x_train.fillna(0)
x_test = x_test.fillna(0)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
# Predict values
y_pred = model.predict(x_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)
# Keep only the strong predictors
x_new = df_numerized[['budget', 'votes', 'score', 'runtime']]  
y = df_numerized['gross']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Optimized Model - MAE:", mae)
print("Optimized Model - MSE:", mse)
print("Optimized Model - R² Score:", r2)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(PolynomialFeatures(degree=2),LinearRegression())
poly_model.fit(x_train,y_train)
y_pred_poly = poly_model.predict(x_test)
mae_poly = mean_absolute_error(y_test,y_pred_poly)
mse_poly = mean_squared_error(y_test,y_pred_poly)
r2_poly = r2_score(y_test,y_pred_poly)

print("Polynomial Model - MAE:", mae_poly)
print("Polynomial Model - MSE:", mse_poly)
print("Polynomial Model - R² Score:", r2_poly)

poly_model_3 = make_pipeline(PolynomialFeatures(degree=3),LinearRegression())
poly_model_3.fit(x_train,y_train)
y_pred_poly_3 = poly_model_3.predict(x_test)
mae_poly_3 = mean_absolute_error(y_test,y_pred_poly)
mse_poly_3 = mean_squared_error(y_test,y_pred_poly)
r2_poly_3 = r2_score(y_test,y_pred_poly)

print("Polynomial Model - MAE:", mae_poly_3)
print("Polynomial Model - MSE:", mse_poly_3)
print("Polynomial Model - R² Score:", r2_poly_3)
from sklearn.ensemble import RandomForestRegressor

# Create the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(x_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(x_test)

# Evaluate performance
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print results
print("Random Forest Model - MAE:", mae_rf)
print("Random Forest Model - MSE:", mse_rf)
print("Random Forest Model - R² Score:", r2_rf)
df = df.dropna()  # You can also use df.dropna()

# Step 3: Compute correlation of numerical features with 'gross'
correlation_matrix = df.corr(numeric_only=True)
correlation_with_gross = correlation_matrix["gross"].sort_values(ascending=False)

# Step 4: Select top 3 features (excluding 'gross' itself)
top_features = correlation_with_gross.index[1:4]  # Get top 3 features

# Step 5: Prepare dataset for training
x = df[top_features]  # Select only top features
y = df["gross"]

# Step 6: Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 7: Train Model
model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(x_train, y_train)

# Step 8: Predict and Evaluate Model
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print Results
print("Top 3 Features:", list(top_features))
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
rf=RandomForestRegressor(random_state=42)
param_grid={
    'n_estimators':[100,200,300],
    'max_depth':[10,20,30],
    'min_samples_split' :[2,5,4],
    'min_samples_leaf':[1,2,4]
}
grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)
grid_search.fit(x_train,y_train)
print("Best Hyperparameters:",grid_search.best_params_)
best_rf= RandomForestRegressor(n_estimators=300,
                               max_depth=10,
                               min_samples_leaf=2,
                               min_samples_split=5)
best_rf.fit(x_train,y_train)
y_pred_rf = best_rf.predict(x_test)
mae = mean_absolute_error(y_test,y_pred_rf)
mse= mean_squared_error(y_test,y_pred_rf)
r2= r2_score(y_test,y_pred_rf)

print("Optimized Model - MAE:", mae)
print("Optimized Model - MSE:", mse)
print("Optimized Model - R² Score:", r2)
import pandas as pd
import matplotlib.pyplot as plt

# Get feature importance from the trained model
feature_importances = pd.Series(best_rf.feature_importances_, index=x.columns)

# Sort and visualize
feature_importances.sort_values(ascending=True).plot(kind='barh', figsize=(8,5), color='skyblue')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Predicting Gross Revenue")
plt.show()
import xgboost 
print("imported Xboost Succesfully !")
from xgboost import XGBRegressor
Xgb_Model = XGBRegressor(n_estimators=300 , max_depth=10,learning_rate=0.1,random_state=42)
Xgb_Model.fit(x_train,y_train)
y_pred_xgb=Xgb_Model.predict(x_test)
r2_xgb= r2_score(y_test,y_pred_xgb)
print("XGBoost R² Score:", r2_xgb)
param_grid_xgb = {
    'n_estimators': [100, 300, 500],  # More trees
    'max_depth': [5, 10, 15],  # Try different tree depths
    'learning_rate': [0.01, 0.1, 0.2],  # Test different learning rates
    'subsample': [0.8, 1.0],  # Use 80%-100% of data per tree
    'colsample_bytree': [0.8, 1.0]  # Use 80%-100% of features per tree
}

xgb = XGBRegressor(random_state=42)

grid_search_xgb = GridSearchCV(
    estimator=xgb, 
    param_grid=param_grid_xgb, 
    cv=5, 
    n_jobs=-1, 
    verbose=2
)

grid_search_xgb.fit(x_train, y_train)

# Get the best parameters
print("Best Hyperparameters for XGBoost:", grid_search_xgb.best_params_)
# Train Optimized XGBoost Model
best_xgb = XGBRegressor(
    colsample_bytree=1.0,
    learning_rate=0.01,
    max_depth=5,
    n_estimators=300,
    subsample=0.8,
    random_state=42
)

best_xgb.fit(x_train, y_train)

# Make Predictions
y_pred_xgb_best = best_xgb.predict(x_test)

# Evaluate Performance
r2_xgb_best = r2_score(y_test, y_pred_xgb_best)
print("Optimized XGBoost R² Score:", r2_xgb_best)
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_xgb_best, label="XGBoost Predictions", color="red", alpha=0.6)
plt.scatter(y_test, best_rf.predict(x_test), label="Random Forest Predictions", color="blue", alpha=0.3)
plt.plot(y_test, y_test, color="black", linestyle="dashed", label="Perfect Prediction")
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.legend()
plt.title("Comparison of XGBoost vs Random Forest Predictions")
plt.show()

param_grid_xgb_advanced = {
    'n_estimators': [300, 500, 700],  # Try more trees
    'max_depth': [3, 5, 7, 10],  # Control tree complexity
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Learning speed
    'subsample': [0.7, 0.8, 1.0],  # Percentage of data per tree
    'colsample_bytree': [0.7, 0.8, 1.0],  # Percentage of features per tree
    'min_child_weight': [1, 3, 5]  # Minimum sum of instance weight needed in a child node
}
xgb = XGBRegressor(random_state=42)

grid_search_xgb_advanced = GridSearchCV(
    estimator=xgb, 
    param_grid=param_grid_xgb_advanced, 
    cv=5, 
    n_jobs=-1, 
    verbose=2
)

grid_search_xgb_advanced.fit(x_train, y_train)

# Get the best parameters
print("Best Hyperparameters for XGBoost (Advanced):", grid_search_xgb_advanced.best_params_)
best_xgb_advanced = XGBRegressor(
     colsample_bytree= 1.0,
     learning_rate = 0.01,
     max_depth = 5 , 
     min_child_weight = 5 , 
     n_estimators = 300 , 
     subsample = 0.07,
    random_state=42
)

best_xgb_advanced.fit(x_train, y_train)

# Make Predictions
y_pred_xgb_advanced = best_xgb_advanced.predict(x_test)

# Evaluate the Model
r2_xgb_advanced = r2_score(y_test, y_pred_xgb_advanced)
print("Optimized XGBoost R² Score (Advanced):", r2_xgb_advanced)
print(x_train.shape)
print(x_train.isnull().sum())
print(x_test.isnull().sum())
param_grid_xgb_balanced = {
    'n_estimators': [300, 400],  # Keep 300-400 trees
    'max_depth': [5, 7],  # Keep max depth lower (not too complex)
    'learning_rate': [0.01, 0.05],  # Keep it small for stability
    'subsample': [0.8, 1.0],  # Keep 80%-100% data usage
    'colsample_bytree': [0.8, 1.0],  # Use 80%-100% of features per tree
    'min_child_weight': [1, 3]  # Regularize with min samples per leaf
}
grid_search_xgb_balanced = GridSearchCV(
    estimator=XGBRegressor(random_state=42), 
    param_grid=param_grid_xgb_balanced, 
    cv=5, 
    n_jobs=-1, 
    verbose=2
)

grid_search_xgb_balanced.fit(x_train, y_train)

# Get best parameters
print("Best Hyperparameters (Balanced):", grid_search_xgb_balanced.best_params_)
best_xgb_balanced = XGBRegressor(
    colsample_bytree=1.0,
    learning_rate=0.01,
    max_depth=5,
    min_child_weight=3,
    n_estimators=300,
    subsample=0.8,
    random_state=42
)

best_xgb_balanced.fit(x_train, y_train)

# Make Predictions
y_pred_xgb_balanced = best_xgb_balanced.predict(x_test)

# Evaluate Performance
r2_xgb_balanced = r2_score(y_test, y_pred_xgb_balanced)
print("Optimized XGBoost R² Score (Balanced):", r2_xgb_balanced)

import joblib
joblib.dump(best_xgb,"optimized_xgboost_model.pkl")
print ("Model saved successfully! 🎉")

