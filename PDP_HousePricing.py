import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

# Load data and create model
df_house = pd.read_csv('kc_house_data.csv')
X_house = df_house[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'yr_built']]
y_house = df_house['price']

rf_house = RandomForestRegressor(n_estimators=20, random_state=42)
rf_house.fit(X_house, y_house)

# Create PDP plots
# 1D
PartialDependenceDisplay.from_estimator(
    rf_house,
    X_house,               
    features=['bedrooms', 'bathrooms', 'sqft_living','floors'],
    n_cols=4,
    grid_resolution=30,
)
plt.show()

#2D, for bathroom, bedroom and floors
feature_pairs = [
    ('bathrooms', 'bedrooms'),
    ('bathrooms', 'floors'),
    ('bedrooms', 'floors')
]

fig, ax = plt.subplots(1, 3, figsize=(18, 6)) 

PartialDependenceDisplay.from_estimator(
    rf_house,
    X_house,
    features=feature_pairs,
    kind='average',
    grid_resolution=30,
    ax=ax
)

plt.tight_layout()
plt.show()

# 2D, sqft_living (because it has higher pricing values)
feature_pairs = [
    ('sqft_living', 'bathrooms'),
    ('sqft_living', 'bedrooms'),
    ('sqft_living', 'floors')
]

fig, ax = plt.subplots(1, 3, figsize=(18, 6)) 

PartialDependenceDisplay.from_estimator(
    rf_house,
    X_house,
    features=feature_pairs,
    kind='average',
    grid_resolution=30,
    ax=ax
)

plt.tight_layout()
plt.show()

