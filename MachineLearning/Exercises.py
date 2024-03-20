##########################################################################################

def operaciones_matematicas(a, b):
    return [a+b,a-b,a*b,a/b,a%b]

# Prueba la función con valores específicos
a = 10
b = 3

resultado = operaciones_matematicas(a, b)
print("Suma:", resultado[0])
print("Resta:", resultado[1])
print("Multiplicación:", resultado[2])
print("División:", resultado[3])
print("Residuo:", resultado[4])

##########################################################################################

def calcular_suma_y_promedio(lista_numeros):
    suma = 0
    promedio = 0
    if len(lista_numeros) > 0:
        suma = sum(lista_numeros)
        promedio = suma / len(lista_numeros)
    return {"suma": suma, "promedio": promedio}


# Pruebas
numeros = [1, 2, 3, 4, 5]
resultado = calcular_suma_y_promedio(numeros)
print("Suma:", resultado["suma"])
print("Promedio:", resultado["promedio"])

##########################################################################################

def contar_frecuencia(lista):
    contador = {}
    for elemento in lista:
        if elemento not in contador:
            contador[elemento] = 1
        else:
            contador[elemento] += 1
    return contador

# Ejemplo de uso
elementos = [1, 2, 2, 3, 1, 2, 4, 5, 4]
resultado = contar_frecuencia(elementos)
print(resultado)

##########################################################################################

def aplicar_funcion_y_filtrar(lista, valor_umbral):
    resultado = list(filter(lambda x: x > valor_umbral, map(lambda y: y**2, lista)))
    return resultado

# Ejemplo de uso
numeros = [1, 2, 3, 4, 5]
valor_umbral = 3
resultado = aplicar_funcion_y_filtrar(numeros, valor_umbral)
print(resultado)

##########################################################################################

import numpy as np

def generar_numeros_enteros_aleatorios(N, minimo, maximo):
    return np.random.randint(minimo,maximo,N)

# Ejemplo de uso
N = 5
minimo = 1
maximo = 10
resultado = generar_numeros_enteros_aleatorios(N, minimo, maximo)
print(resultado)

##########################################################################################

import numpy as np

def generar_secuencia_numerica(minimo, maximo, paso):
    return np.arange(minimo,maximo,paso)

# Ejemplo de uso
minimo = 0
maximo = 10
paso = 2
resultado = generar_secuencia_numerica(minimo, maximo, paso)
print(resultado)

##########################################################################################

import pandas as pd

def calcular_promedio(dataframe):
    return dataframe.mean()

# Ejemplo de uso
data = {
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
}

df = pd.DataFrame(data)
resultado = calcular_promedio(df)
print(resultado)

##########################################################################################

import pandas as pd

def seleccionar_datos(dataframe, criterio):
    return dataframe.query(criterio)

# Ejemplo de uso
data = {
    'nombre': ['Alice', 'Bob', 'Charlie', 'David'],
    'edad': [20, 22, 18, 25],
    'calificaciones': [90, 88, 75, 95]
}

df = pd.DataFrame(data)
criterio = 'edad > 18'
resultado = seleccionar_datos(df, criterio)
print(resultado)

##########################################################################################

import pandas as pd
import numpy as np

def rellenar_con_media(dataframe, columna):
    if columna not in dataframe.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    media = np.mean(dataframe[columna].dropna())
    dataframe[columna].fillna(media, inplace=True)

    return dataframe


# Ejemplo de uso
data = {
    'nombre': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'edad': [20, None, 18, 25, None],
    'calificaciones': [90, 88, None, None, 95]
}

df = pd.DataFrame(data)
columna = 'calificaciones'

resultado = rellenar_con_media(df, columna)
print(resultado)

##########################################################################################

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Función de regresión lineal
def regresion_ventas(datos):
    # Separamos los datos en variables independientes (X) y variable dependiente (y)
    X = datos[['TV', 'Radio', 'Periodico']]
    y = datos['Ventas']

    # Creamos un modelo de regresión lineal
    modelo_regresion = LinearRegression()

    # Entrenamos el modelo con los datos
    modelo_regresion.fit(X, y)

    return modelo_regresion

# Ejemplo de uso con datos reales
data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
    'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
    'Periodico': [69.2, 45.1, 69.3, 58.5, 58.4],
    'Ventas': [22.1, 10.4, 9.3, 18.5, 12.9]
}
df = pd.DataFrame(data)
modelo_regresion = regresion_ventas(df)

# Estimaciones de ventas para nuevos datos de inversión en publicidad
nuevos_datos = pd.DataFrame({'TV': [200, 60, 30], 'Radio': [40, 20, 10], 'Periodico': [50, 10, 5]})
estimaciones_ventas = modelo_regresion.predict(nuevos_datos)

print("Estimaciones de Ventas:")
print(estimaciones_ventas)


##########################################################################################

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Función de regresión logística
def regresion_logistica(datos):
    # Ajusta un modelo de regresión logística
    modelo_regresion_logistica = LogisticRegression()
    modelo_regresion_logistica.fit(datos.drop('Enfermedad', axis=1), datos['Enfermedad'])
    return modelo_regresion_logistica

# Ejemplo de uso con datos de pacientes
data = {
    'Edad': [50, 35, 65, 28, 60],
    'Colesterol': [180, 150, 210, 130, 190],
    'Enfermedad': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
modelo_regresion_logistica = regresion_logistica(df)

# Estimaciones de clasificación binaria para nuevos datos
nuevos_datos = pd.DataFrame({'Edad': [45, 55], 'Colesterol': [170, 200]})
estimaciones_clasificacion = modelo_regresion_logistica.predict(nuevos_datos)
print("Estimaciones de Clasificación:")
print(estimaciones_clasificacion)

##########################################################################################

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Función de clasificación KNN
def knn_clasificacion(datos, k=3):
    # Separar las características (X) y las etiquetas (y)
    X = datos.drop('Tipo', axis=1)
    y = datos['Tipo']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear el modelo KNN
    modelo_knn = KNeighborsClassifier(n_neighbors=k)
    
    # Entrenar el modelo con los datos de entrenamiento
    modelo_knn.fit(X_train, y_train)
    
    # Realizar predicciones con los datos de prueba
    estimaciones_clasificacion = modelo_knn.predict(X_test)
    
    # Calcular la precisión del modelo
    precision = accuracy_score(y_test, estimaciones_clasificacion)
    
    return precision, estimaciones_clasificacion
    
# Ejemplo de uso con el conjunto de datos Iris
data = pd.read_csv('iris.csv')  # Reemplaza 'iris.csv' con tu archivo de datos
modelo_knn = knn_clasificacion(data, k=3)
 
# Estimaciones de clasificación para nuevas muestras
nuevas_muestras = pd.DataFrame({
    'LargoSepalo': [5.1, 6.0, 4.4],
    'AnchoSepalo': [3.5, 2.9, 3.2],
    'LargoPetalo': [1.4, 4.5, 1.3],
    'AnchoPetalo': [0.2, 1.5, 0.2]
})
 
estimaciones_clasificacion = modelo_knn.predict(nuevas_muestras)
print("Estimaciones de Clasificación:")
print(estimaciones_clasificacion)

##########################################################################################

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Función de clasificación KNN
def knn_clasificacion(datos, k=3):
    # Separar las características (X) y las etiquetas (y)
    X = datos.drop('Name', axis=1)
    y = datos['Name']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear el modelo KNN
    modelo_knn = KNeighborsClassifier(n_neighbors=k)
    
    # Entrenar el modelo con los datos de entrenamiento
    modelo_knn.fit(X_train, y_train)
    
    return modelo_knn

# Ejemplo de uso con el conjunto de datos Iris
data = pd.read_csv('iris.csv')   # Reemplaza 'iris.csv' con tu archivo de datos
modelo_knn = knn_clasificacion(data, k=3)
 
# Estimaciones de clasificación para nuevas muestras
nuevas_muestras = pd.DataFrame({
    'sepal_length': [5.1, 6.0, 4.4],
    'sepal_width': [3.5, 2.9, 3.2],
    'petal_length': [1.4, 4.5, 1.3],
    'petal_width': [0.2, 1.5, 0.2]
})
 
estimaciones_clasificacion = modelo_knn.predict(nuevas_muestras.values)
print("Estimaciones de Clasificación:")
print(estimaciones_clasificacion)

##########################################################################################

# Como programador de Python, completa la función [] para obtener el siguiente resultado [] en base al siguiente código [] en base al siguiente enunciado []