import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

# Used to create a sequence of previous temepratures for predictions
def create_sequences(df, seq_length):
  sequences = []
  for i in range(seq_length - 1):
    sequences.append([0]*seq_length)
  sequences = np.array(sequences)

  for i in range(len(df) - seq_length + 1):
    sequence = df.iloc[i:i+seq_length].values
    sequences = np.concatenate((sequences, sequence.T), axis=0)
  return sequences

# Define the model
def create_model(shape=(1,1,1), units=50, dropout_rate=0.2, learning_rate=0.001, activation='tanh'):
    model = Sequential()
    model.add(LSTM(units, input_shape=(shape[1], shape[2])))
    model.add(Dense(1, activation='linear'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics = RootMeanSquaredError())
    return model

# Load your time series data
df = pd.read_csv('sample_data/260_weeks_data.csv')

# Filter data for region 10
df_region_10 = df[df['Region'] == 10]
df_region_10 = df_region_10[::-1]
df_region_10.replace(r'^\s*$', np.nan, regex=True)
df_region_10 = df_region_10.fillna(0)

# Use 'Week of' as the datetime index
df_region_10.loc[:, 'Week of'] = pd.to_datetime(df_region_10['Week of'])
df_region_10.set_index('Week of', inplace=True)

# Create X, y data
sequence_length = 30
sequences = create_sequences(df_region_10[['Avg. temperature']], sequence_length)
X = sequences[:, :, np.newaxis]
y = np.expand_dims(df_region_10['Regional Illnesses'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Grid search for best hyperparameters

units = [16, 32, 64, 128]
dropout_rates = [0.2, 0.5]
learning_rates = [0.001, 0.01]
activations = ['tanh', 'relu']

best_rmse = 10000
params = []
best_predictions = np.array([])

for unit in units:
  for dropout_rate in dropout_rates:
    for learning_rate in learning_rates:
      for activation in activations:
        model = create_model(shape=X_train.shape,
                          units=unit,
                          dropout_rate=dropout_rate,
                          learning_rate=learning_rate,
                          activation=activation
                           )
        # Train the model
        model.fit(X_train, y_train, epochs=50)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        if rmse < best_rmse:
          best_rmse = rmse
          params = [unit, dropout_rate, learning_rate, activation]
          best_predictions = predictions

print(f'Test RMSE with Grid Search: {best_rmse}')

# Plot predictions against actual values
plt.plot(np.array(df_region_10.index[-len(y_test):]), np.array(y_test), label='Actual')
plt.plot(np.array(df_region_10.index[-len(y_test):]), best_predictions, label='Predicted')
plt.xlabel('Week of')
plt.ylabel('Regional Illnesses')
plt.legend()
