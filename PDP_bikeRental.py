import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

# from previous practica
def preprocess_linear(df):
    data = df[['workingday', 'holiday']]
    # one hot encoding for season
    season_encoded = pd.get_dummies(df['season'], prefix='season', drop_first=True, dtype=int)
    data = pd.concat([data, season_encoded], axis=1)
    # MISTY if weathersit=2, RAIN if weathersit=3 or 4
    data['MISTY'] = df['weathersit'].apply(lambda x: 1 if x == 2 else 0)
    data['RAIN'] = df['weathersit'].apply(lambda x: 1 if x == 3 or x == 4 else 0)
    # Denormalize temp, hum, windspeed
    data['temp'] = df['temp'] * 39 + 8
    data['hum'] = df['hum'] * 100
    data['windspeed'] = df['windspeed'] * 67
    # Create day_since_2011
    data['day_since_2011'] = (pd.to_datetime(df['dteday']) - pd.to_datetime('2011-01-01')).dt.days

    return data, df['cnt']

# Build model
df_day = pd.read_csv('day.csv')
X, y = preprocess_linear(df_day)

rf = RandomForestRegressor(n_estimators=20, random_state=42)
rf.fit(X, y)

# Plot PDP
PartialDependenceDisplay.from_estimator(
    rf,
    X,               
    features=['day_since_2011', 'temp', 'hum', 'windspeed'],
    n_cols=4,
    grid_resolution=30,
)
plt.show()

# 2D PDP for temperature and humidity
PartialDependenceDisplay.from_estimator(
    rf,
    X,
    features=[('temp', 'hum')],
    kind='average',
    grid_resolution=30
)
plt.show()
