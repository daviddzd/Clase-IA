import pandas as pd

df = pd.read_excel('/content/Consulta_20241113-093525023.xlsx')
df.head(20)

# Seleccionar la fila 8 como encabezado (índice 7)
new_header = df.iloc[8]
new_header

# Crear un nuevo DataFrame desde la fila 17 en adelante
df = df[17:]
df

# Asignar el nuevo encabezado
df.columns = new_header
df.head()

# Reiniciar el índice para que comience desde 0 en el nuevo DataFrame
df.reset_index(drop=True, inplace=True)
df.head()

# Convertir la columna 'fecha' a tipo datetime
df['Fecha'] = pd.to_datetime(df['Título'])
df.head()

df.info()

# Filtrar por el año 2023
_df = df[(df['Fecha'] >= '01-01-2014') & (df['Fecha'] <= '12-31-2024')]
_df.head()

_df.tail()

import matplotlib.pyplot as plt

# dates = pd.to_datetime(_df["Fecha"]).apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
dates = _df["Fecha"].tolist()

# Grafica el "Índice Nacional de Precios al consumidor, variación anual"
plt.figure(figsize=(16, 10))
plt.plot(dates,_df["Índice Nacional de Precios al consumidor, variación anual"], color='blue', label="Variación anual")
plt.title("Índice Nacional de Precios al Consumidor - variación anual")
plt.xlabel("Tiempo")
plt.ylabel("Variación anual")
plt.xticks(rotation=90)
plt.legend()
plt.show()

X = _df.drop(columns=["Índice Nacional de Precios al consumidor, variación anual", "Título", "Fecha"])
y = _df["Índice Nacional de Precios al consumidor, variación anual"].values

X.shape

y.shape

X = _df.drop(columns=["Índice Nacional de Precios al consumidor, variación anual", "Título", "Fecha"])
y = _df["Índice Nacional de Precios al consumidor, variación anual"].values

import numpy as np

window_size = 12
X_rnn = []
y_rnn = []
# Construye secuencias temporales
for i in range(window_size, len(X_scaled)):
    X_rnn.append(X_scaled[i - window_size:i])
    y_rnn.append(y_scaled[i])

X_rnn, y_rnn = np.array(X_rnn), np.array(y_rnn)

split = int(len(X) * 0.8)
X_train, y_train = X_rnn[:split], y_rnn[:split]
X_test, y_test = X_rnn[split:], y_rnn[split:]

X_train.shape

y_test.shape

window_size

X_train.shape[2]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN, GRU

model = Sequential([
  SimpleRNN(units=60, return_sequences=True, input_shape=(window_size, X_train.shape[2])),
  SimpleRNN(units=30, return_sequences=True),
  SimpleRNN(units=15, return_sequences=False),
  Dense(units=1)
])

from tensorflow.keras.optimizers import Adam

# Compilar el modelo
learning_rate = 0.001
adam_optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, batch_size=1, epochs=10, validation_data=(X_test, y_test))

# Predice en el conjunto de prueba y desescala los datos
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_original = scaler.inverse_transform(y_test)

# Muestra algunas predicciones y valores reales
print("Predicciones:", predictions.flatten()[:5])
print("Valores reales:", y_test_original.flatten()[:5])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calcula el MAE
mae = mean_absolute_error(y_test, predictions)
print(f"Error Absoluto Medio (MAE): {mae}")

# Calcula el MSE
mse = mean_squared_error(y_test, predictions)
print(f"Error Cuadrático Medio (MSE): {mse}")

# Calcula el RMSE
rmse = np.sqrt(mse)
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse}")

# Calcula el Coeficiente de Determinación (R²)
r2 = r2_score(y_test, predictions)
print(f"Coeficiente de Determinación (R²): {r2}")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Gráfica de los valores reales y las predicciones del conjunto de prueba
valid = _df[split:]
valid = valid.reset_index(drop=True)
valid['Predictions'] = np.nan
valid.loc[window_size:, 'Predictions'] = predictions

dates_valid = pd.to_datetime(valid['Fecha']).apply(lambda x: x.strftime('%Y-%m-%d')).tolist()

plt.figure(figsize=(16, 8))
plt.title('Modelo RNN para Predicción de la inflación')
plt.xlabel('Fecha')
plt.ylabel('Inflacion')
plt.plot(dates_valid, valid[["Índice Nacional de Precios al consumidor, variación anual", 'Predictions']])
plt.legend(['Valor Real', 'Predicciones'], loc='lower right')
plt.xticks(rotation=90)
plt.show()

