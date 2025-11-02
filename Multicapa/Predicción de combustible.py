"""
Este programa nos ayudará a predecir el consumo de combustible utilizando una red neuronal multicapa, modulos utlizados:
- ucimlrepo: para obtener el conjunto de datos de consumo de combustible.
- pandas: para manipulación y análisis de datos.
- tensorflow y keras: para construir y entrenar la red neuronal.
- numpy: para operaciones numéricas.
"""

from ucimlrepo import fetch_ucirepo #inastalado via pip
  
# Cargar el conjunto de datos de consumo de combustible
auto_mpg = fetch_ucirepo(id=9) 
  
# Exploramos nuestros datos 
X = auto_mpg.data.features 

y = auto_mpg.data.targets  

print(X.head())

print(y.head())

X.info()
y.info()

# Las siguientes lineas de codigo borran las filas con valores faltantes
import pandas as pd 
df = pd.concat([X, y], axis=1).dropna()

# Verificamos que no haya valores faltantes
df.info()

# Definimos X e y
X = df.drop('mpg', axis=1)
y = df['mpg']

#Ahora dividimos los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, random_state=1
)

print(X_train.shape)
print(X_test.shape)

# Escalamos las características para mejorar el rendimiento del modelo
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construimos la red neuronal multicapa
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Definir el modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  
    Dense(32, activation='relu'),                                   
    Dense(16, activation='relu'), 
    Dense(1)                                                        
])

# Compilar el modelo con el optimizador Adam y una tasa de aprendizaje personalizada
from tensorflow.keras.optimizers import Adam

# Tasa de aprendizaje deseada
learning_rate = 0.001
adam_optimizer = Adam(learning_rate=learning_rate)

model.compile(
    optimizer=adam_optimizer,
    loss='mean_squared_error',
    metrics=['root_mean_squared_error'],
)

# Entrenamos el modelo
history = model.fit(
    X_train, y_train,
    epochs=5, batch_size=1, 
    validation_data=(X_test, y_test)
)

#Graficamos la función de pérdida
import matplotlib.pyplot as plt

# Graficar la función de pérdida
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Función de pérdida durante el entrenamiento')
plt.show()

# Evaluamos el modelo en el conjunto de prueba
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Mean Absolute Error: {test_mae:.2f}')

# Comparamos las predicciones con los valores reales
predictions = model.predict(X_test)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})
print(comparison.head())

# Calculamos métricas adicionales para evaluar el rendimiento del modelo
from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, predictions)
print(f'R²: {r2}')

mse = mean_squared_error(y_test, predictions)
print(f'MSE: {mse}')