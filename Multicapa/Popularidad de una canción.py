"""
Este es un ejemplo sencillo de regresión utilizando una red neuronal multicapa para predecir 
la popularidad de una canción en función de sus características acústicas.
modulos utilizados:
- pandas: para manipulación y análisis de datos.
- tensorflow y keras: para construir y entrenar la red neuronal.
- numpy: para operaciones numéricas.
"""

import pandas as pd #para manipulación y análisis de datos

# Cargar el conjunto de datos desde una URL
url = "https://raw.githubusercontent.com/mevangelista-alvarado/datasets/refs/heads/main/spotify_songs.csv"
# El archivo CSV contiene miles de canciones con varias características
df = pd.read_csv(url)

print(df.head())

# Seleccionar características (features)
features = [
    'danceability', 'energy', 'key', 'loudness',
    'mode', 'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms',
]
X = df[features].values

# Target numérico
y = df['popularity'].values 

# Dividimos los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, random_state=42
)

# Escalamos las características para mejorar el rendimiento del modelo
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definimos nuestra red neuronal multicapa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Creamos el optimizador Adam con una tasa de aprendizaje personalizada
from tensorflow.keras.optimizers import Adam

# Tasa de aprendizaje deseada
learning_rate = 0.001
adam_optimizer = Adam(learning_rate=learning_rate)

model.compile(
    optimizer=adam_optimizer, 
    loss='mse', 
    metrics=['mae'],
)

# Entrenamos el modelo
history = model.fit(
    X_train, 
    y_train,
    validation_split=0.2,
    epochs=50, 
    batch_size=50,
)

# Graficamos la función de pérdida
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Función de pérdida durante el entrenamiento')
plt.show()

# Evaluamos el modelo en el conjunto de prueba
loss, mae = model.evaluate(X_test, y_test)
print(f"MAE en el conjunto test: {mae}")

# Predecimos y comparamos con los valores reales
import pandas as pd 

predictions = model.predict(X_test)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})
print(comparison.head())

# seleccionamos una canción para predecir su popularidad
nombre_cancion = "Beso"

canciones_df = df[df['track_name'].str.contains(nombre_cancion, case=False, na=False)]

print(f"Canciones encontradas:")
canciones_df[['track_name', 'artists', 'album_name']].head()

# indice a selecionar
i = 0
cancion = canciones_df.iloc[i]
X_input = cancion[features].values.reshape(1, -1)
X_input = scaler.transform(X_input)

prediccion = model.predict(X_input)[0][0]
print(f"Canción: {cancion['track_name']} - {cancion['artists']}")
print(f"Popularidad real: {cancion['popularity']}")
print(f"Predicción: {prediccion:.2f}")

# Calculamos métricas adicionales para evaluar el rendimiento del modelo
from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, predictions)
print(f'R²: {r2}')

mse = mean_squared_error(y_test, predictions)
print(f'MSE: {mse}')
