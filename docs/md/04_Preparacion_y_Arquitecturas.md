# 5. PREPARACIÓN Y PREPROCESAMIENTO DE DATOS

---

## 5.1. Inspección y Limpieza de Datos

Para garantizar la integridad y calidad del proceso de aprendizaje, se llevó a cabo una auditoría exhaustiva y un control de calidad sobre el archivo fuente `dataset_pm25.csv`. A fin de no asumir el dataset como una "caja negra" y justificar rigurosamente cada conclusión, la inspección se estructuró en dos niveles metodológicos complementarios: una **auditoría manual y paramétrica en hoja de cálculo (Excel / OnlyOffice)** y una **verificación computacional mediante script (Python / Pandas)**.

### 5.1.1. Metodología de Auditoría en Hoja de Cálculo (Excel)
Se importó el conjunto de datos completo en un libro de cálculo estructurado (`Auditoria_Estadistica_Dataset_PM25.xlsx`), organizándolo en dos hojas de trabajo:
1. **Hoja `Dataset_PM25`:** Aloja la totalidad de los datos crudos, abarcando desde la fila 2 hasta la fila 41.731 (exactamente **41.730 registros observacionales**) distribuidos a lo largo de 11 columnas (columnas `A` a `K`).
2. **Hoja `Auditoria_Estadistica`:** Contiene una matriz de control donde cada métrica descriptiva fue calculada dinámicamente mediante **fórmulas nativas de Excel** referenciadas sobre el rango absoluto de cada variable (`Dataset_PM25!X2:X41731`):

* **Verificación de Completitud (Registros Válidos):** Se utilizó la función `=CONTAR(Dataset_PM25!X2:X41731)` para cada columna, confirmando que cada una de las 11 variables contiene exactamente 41.730 valores numéricos procesables.
* **Detección de Datos Faltantes (Nulos / Vacíos):** Se evaluó la ausencia de información mediante la función `=CONTAR.BLANCO(Dataset_PM25!X2:X41731)`. El resultado arrojó **0 celdas vacías** en todas las columnas. Adicionalmente, mediante filtros condicionales y `=CONTAR.SI()`, se comprobó que no existieran cadenas de texto sustitutas de error (tales como `NA`, `null` o espacios).
* **Tendencia Central:** Calculada mediante `=PROMEDIO(Dataset_PM25!X2:X41731)` para la media aritmética y `=MEDIANA(Dataset_PM25!X2:X41731)` para el valor central ordenado.
* **Dispersión:** Calculada mediante la fórmula de desviación estándar muestral `=DESVEST.M(Dataset_PM25!X2:X41731)`.
* **Rango y Distribución Cuartílica:** Se determinaron los extremos mediante `=MIN()` y `=MAX()`, y los percentiles de corte mediante `=PERCENTIL.INC(Dataset_PM25!X2:X41731; 0,25)` para el primer cuartil ($Q_1$) y `=PERCENTIL.INC(Dataset_PM25!X2:X41731; 0,75)` para el tercer cuartil ($Q_3$).
* **Dispersión Intercuartílica y Amplitud:** Se incorporaron fórmulas de diferencia para el Rango Intercuartílico ($IQR = Q_3 - Q_1$) y el Rango Total ($Max - Min$).

A continuación se presenta la evidencia directa obtenida de la hoja de auditoría procesada:

![Figura 1. Matriz de auditoría estadística y control de calidad implementada en hoja de cálculo sobre las 41.730 observaciones del Beijing PM2.5 Dataset](../img/Captura_Auditoria_Estadistica_Excel.png)
*Figura 1. Matriz paramétrica de auditoría estadística y control de calidad implementada en hoja de cálculo sobre las 41.730 observaciones del Beijing PM2.5 Dataset (`Auditoria_Estadistica_Dataset_PM25.xlsx`).*

> **Nota de trazabilidad para revisión docente:**  
> El libro de cálculo con las fórmulas vivas y el rango completo de datos se encuentra disponible para verificación en el repositorio local del proyecto bajo la ruta:  
> `docs/Auditoria_Estadistica_Dataset_PM25.xlsx`  
> En la hoja `Auditoria_Estadistica`, cada celda evalúa de forma dinámica el rango `Dataset_PM25!A2:K41731`, demostrando la validez empírica de los valores reportados sin manipulación estática.

### 5.1.2. Verificación Programática en Python y Google Colab
De forma paralela y para asegurar reproducibilidad algorítmica e interactiva, se implementó un cuaderno de trabajo en Google Colab denominado `01_auditoria_y_verificacion_dataset_pm25.ipynb`. Mediante la librería Pandas se procesó el tensor completo de 41.730 filas, validando tipos de datos, comprobando la ausencia absoluta de valores nulos (NaN = 0) y generando una matriz estadística descriptiva formateada con estilos ejecutivos.

```python
import pandas as pd

# Carga e inspección de estructura
df = pd.read_csv('data/datasets/dataset_pm25.csv')
print("Dimensiones:", df.shape)          # Retorna: (41730, 11)
print("Valores nulos:\n", df.isnull().sum())  # Retorna: 0 para todas las columnas
print("Tipos de datos:\n", df.dtypes)     # Retorna: float64 e int64 en todas
```

> **Nota de acceso y reproducibilidad interactiva:**  
> El cuaderno de ejecución en Google Colab con la carga del dataset, el filtrado de nulos y las tablas estilizadas interactivas se encuentra disponible públicamente para revisión en el siguiente enlace:  
> [Google Colab — Auditoría Estadística y Control de Calidad Beijing PM2.5](https://colab.research.google.com/drive/1eXevX6U_6He5snGHMdMV9cGOJR5pyI5L?usp=sharing)  
> Asimismo, el archivo fuente ejecutable `.ipynb` se encuentra archivado en el repositorio local del proyecto bajo la ruta: `notebooks/01_auditoria_y_verificacion_dataset_pm25.ipynb`.

Ambos procedimientos (Excel y Python/Colab) convergieron en los mismos resultados exactos, consolidando la siguiente tabla paramétrica de control de calidad:

| Variable | Columna Excel | Rango Evaluado | Muestras ($N$) | Vacíos (NaN) | Media ($\mu$) | Desv. Est. ($s$) | Mínimo | Mediana ($Q_2$) | Máximo |
| :--- | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `TEMP` | Columna A | `A2:A41731` | 41.730 | 0 | 12,41 °C | 12,17 °C | −19,00 °C | 14,00 °C | 42,00 °C |
| `PRES` | Columna B | `B2:B41731` | 41.730 | 0 | 1016,43 hPa | 10,30 hPa | 991,00 hPa | 1016,00 hPa | 1046,00 hPa |
| `DEWP` | Columna C | `C2:C41731` | 41.730 | 0 | 1,76 °C | 14,43 °C | −40,00 °C | 2,00 °C | 28,00 °C |
| `Iws` | Columna D | `D2:D41731` | 41.730 | 0 | 23,79 m/s | 49,49 m/s | 0,45 m/s | 5,37 m/s | 565,49 m/s |
| `hour` | Columna E | `E2:E41731` | 41.730 | 0 | 11,50 | 6,92 | 0 | 12 | 23 |
| `day` | Columna F | `F2:F41731` | 41.730 | 0 | 15,68 | 8,78 | 1 | 16 | 31 |
| `pm25_t` | Columna G | `G2:G41731` | 41.730 | 0 | 98,66 μg/m³ | 92,05 μg/m³ | 0,00 μg/m³ | 72,00 μg/m³ | 994,00 μg/m³ |
| `pm25_t-1` | Columna H | `H2:H41731` | 41.730 | 0 | 98,66 μg/m³ | 92,05 μg/m³ | 0,00 μg/m³ | 72,00 μg/m³ | 994,00 μg/m³ |
| `pm25_t-2` | Columna I | `I2:I41731` | 41.730 | 0 | 98,67 μg/m³ | 92,05 μg/m³ | 0,00 μg/m³ | 72,00 μg/m³ | 994,00 μg/m³ |
| `pm25_t-3` | Columna J | `J2:J41731` | 41.730 | 0 | 98,67 μg/m³ | 92,05 μg/m³ | 0,00 μg/m³ | 72,00 μg/m³ | 994,00 μg/m³ |
| `pm25_t+24` | Columna K | `K2:K41731` | 41.730 | 0 | 98,59 μg/m³ | 92,07 μg/m³ | 0,00 μg/m³ | 72,00 μg/m³ | 994,00 μg/m³ |

### 5.1.3. Conclusiones de la Inspección
1. **Completitud Absoluta:** Al comprobarse de forma documental que no existen valores nulos ni registros corruptos, no fue necesario implementar técnicas de imputación (como interpolación lineal, sustitución por media o eliminación de filas), preservando la integridad de las 41.730 horas consecutivas.
2. **Heterogeneidad de Escalas:** Se confirma la disparidad de magnitudes: la presión atmosférica (`PRES`) ronda magnitudes de 1000 hPa, mientras que el viento acumulado (`Iws`) presenta picos asimétricos de hasta 565,49 m/s frente a una mediana baja de 5,37 m/s. Esto demuestra empíricamente por qué es obligatorio aplicar escalamiento previo a la red neuronal.

---

## 5.2. Partición del Conjunto de Datos en Entrenamiento y Prueba

El conjunto total de 41.730 observaciones temporales se dividió en dos subconjuntos independientes: un conjunto de **entrenamiento** (*training set*) para la calibración sináptica de los pesos y sesgos mediante Backpropagation, y un conjunto de **prueba** (*testing set*) reservado exclusivamente para la validación final de la capacidad predictiva.

**Proporción adoptada:** Partición estándar 80% / 20%, distribuida en:
* Conjunto de entrenamiento: **33.384 observaciones** (filas 2 a 33.385 en la hoja de cálculo).
* Conjunto de prueba: **8.346 observaciones** (filas 33.386 a 41.731 en la hoja de cálculo).

### Justificación de la Partición Temporal Cronológica
En problemas de series temporales continuas, **no es metodológicamente admisible aplicar una división aleatoria** (`random_split` o muestreo aleatorio uniforme). La concentración de material particulado responde a una fuerte dependencia autorregresiva secuencial; si se mezclaran aleatoriamente las observaciones, el modelo aprendería a predecir instantes utilizando información de horas futuras respecto a ese punto, incurriendo en una severa fuga temporal de datos (*temporal data leakage*). 

Por lo tanto, la partición se fijó de manera estrictamente cronológica por corte de índice: las primeras 33.384 horas conforman el pasado y presente del entrenamiento, mientras que las últimas 8.346 horas (aproximadamente los últimos 11 meses del registro) actúan como el futuro ciego sobre el cual se evalúa la generalización del modelo.

---

## 5.3. Normalización y Escalamiento de las Variables

### 5.3.1. Selección de la Técnica de Escalamiento
Dado que en las capas ocultas se implementa la función de activación tangente hiperbólica (`tansig`), cuya imagen natural es el intervalo continuo $(-1, +1)$, se seleccionó la **Normalización Min-Max simétrica acotada en $[-1, +1]$**:

$$x'_j = 2 \cdot \left( \frac{x_j - x_{j,\min}}{x_{j,\max} - x_{j,\min}} \right) - 1$$

Esta técnica mapea linealmente el mínimo de la serie a $-1$ y el máximo a $+1$. La ventaja técnica fundamental frente a la normalización estándar en $[0, 1]$ radica en que las entradas centradas en cero garantizan gradientes con signos balanceados durante la retropropagación, evitando sesgos sistemáticos en la actualización de los pesos sinápticos.

### 5.3.2. Prevención Rigurosa del Data Leakage
Un principio fundamental de la ingeniería de datos en aprendizaje automático exige que **el conjunto de prueba no intervenga bajo ningún concepto en el cálculo de los parámetros de preprocesamiento**. Si los valores $x_{\min}$ y $x_{\max}$ se obtuvieran sobre la totalidad de las 41.730 filas, el modelo recibiría información indirecta acerca de los extremos del conjunto de prueba.

Para evitar esta anomalía, el procedimiento se implementó estrictamente de la siguiente forma:
1. Se calcularon los vectores de mínimos ($\mathbf{x}_{\min}^{train}$) y máximos ($\mathbf{x}_{\max}^{train}$) empleando **únicamente las 33.384 muestras de entrenamiento**.
2. Con dichos parámetros fijos se transformó el conjunto de entrenamiento:
   $$\mathbf{X}_{train}^{norm} = 2 \cdot \frac{\mathbf{X}_{train} - \mathbf{x}_{\min}^{train}}{\mathbf{x}_{\max}^{train} - \mathbf{x}_{\min}^{train}} - 1$$
3. El conjunto de prueba se proyectó utilizando exactamente los **mismos parámetros aprendidos del entrenamiento**:
   $$\mathbf{X}_{test}^{norm} = 2 \cdot \frac{\mathbf{X}_{test} - \mathbf{x}_{\min}^{train}}{\mathbf{x}_{\max}^{train} - \mathbf{x}_{\min}^{train}} - 1$$

Este mismo protocolo se aplicó sobre la variable objetivo $y = pm25_{t+24}$, normalizándola en el espacio $[-1, +1]$ para el entrenamiento. En la fase de evaluación, toda predicción $\hat{y}^{norm}$ es desnormalizada a su escala real de $\mu g/m^3$ mediante la ecuación inversa:

$$\hat{y} = \frac{(\hat{y}^{norm} + 1) \cdot (y_{\max}^{train} - y_{\min}^{train})}{2} + y_{\min}^{train}$$

asegurando que las métricas finales MSE y RMSE se expresen en las unidades físicas del problema.

---

---

# 6. DISEÑO Y COMPARACIÓN DE ARQUITECTURAS

---

## 6.1. Arquitectura de Referencia (Benchmark: 10:10:1)

La arquitectura de referencia corresponde al modelo base sugerido en la guía docente. Representa una estructura clásica de una sola capa oculta dimensionada con el mismo número de neuronas que variables de entrada:

**Topología:** $10 \xrightarrow{\text{tansig}} 10 \xrightarrow{\text{purelin}} 1$

| Capa | Tipo de Capa | Neuronas | Función de Activación | Matriz de Pesos ($\mathbf{W}$) | Vector de Sesgos ($\mathbf{b}$) | Subtotal Parámetros |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| $l=0$ | Entrada | 10 | Identidad (Distribución) | — | — | — |
| $l=1$ | Capa Oculta 1 | 10 | Tangente Hiperbólica (`tansig`) | $\mathbf{W}^{(1)} \in \mathbb{R}^{10 \times 10}$ | $\mathbf{b}^{(1)} \in \mathbb{R}^{10 \times 1}$ | $100 + 10 = 110$ |
| $l=2$ | Capa de Salida | 1 | Lineal (`purelin`) | $\mathbf{W}^{(2)} \in \mathbb{R}^{1 \times 10}$ | $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times 1}$ | $10 + 1 = 11$ |

**Total de Parámetros Entrenables:**
$$N_{params}^{ref} = 110 + 11 = \mathbf{121 \text{ parámetros}}$$

*(Ver figura: `docs/img/Arquitectura_Referencia_10_10_1.jpg`)*

**Comportamiento esperado:** Al poseer únicamente 121 coeficientes libres, es un modelo computacionalmente liviano con bajo riesgo de sobreajuste, pero con una frontera de transformación potencialmente rígida ante la interacción simultánea de variables meteorológicas y dinámicas autorregresivas.

---

## 6.2. Arquitectura Propia Propuesta (10:16:8:1)

Como propuesta propia, se diseñó una arquitectura más profunda orientada bajo el concepto de **embudo de abstracción progresiva** (*funnel topology*), estructurada con dos capas ocultas intermedias:

**Topología:** $10 \xrightarrow{\text{tansig}} 16 \xrightarrow{\text{tansig}} 8 \xrightarrow{\text{purelin}} 1$

| Capa | Tipo de Capa | Neuronas | Función de Activación | Matriz de Pesos ($\mathbf{W}$) | Vector de Sesgos ($\mathbf{b}$) | Subtotal Parámetros |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| $l=0$ | Entrada | 10 | Identidad (Distribución) | — | — | — |
| $l=1$ | Capa Oculta 1 | 16 | Tangente Hiperbólica (`tansig`) | $\mathbf{W}^{(1)} \in \mathbb{R}^{16 \times 10}$ | $\mathbf{b}^{(1)} \in \mathbb{R}^{16 \times 1}$ | $160 + 16 = 176$ |
| $l=2$ | Capa Oculta 2 | 8 | Tangente Hiperbólica (`tansig`) | $\mathbf{W}^{(2)} \in \mathbb{R}^{8 \times 16}$ | $\mathbf{b}^{(2)} \in \mathbb{R}^{8 \times 1}$ | $128 + 8 = 136$ |
| $l=3$ | Capa de Salida | 1 | Lineal (`purelin`) | $\mathbf{W}^{(3)} \in \mathbb{R}^{1 \times 8}$ | $\mathbf{b}^{(3)} \in \mathbb{R}^{1 \times 1}$ | $8 + 1 = 9$ |

**Total de Parámetros Entrenables:**
$$N_{params}^{prop} = 176 + 136 + 9 = \mathbf{321 \text{ parámetros}}$$

*(Ver figura: `docs/img/Arquitectura_Propia_10_16_8_1.jpg`)*

### Justificación Técnica de la Propuesta
1. **Expansión Inicial ($10 \to 16$):** Las 10 variables de entrada presentan relaciones altamente no lineales (por ejemplo, la interacción entre viento `Iws` y punto de rocío `DEWP` en la dispersión de aerosoles). Al expandir las entradas a 16 neuronas en la primera capa oculta, la red proyecta el vector de características a un espacio de mayor dimensionalidad donde las complejas interacciones atmosféricas pueden ser desacopladas con mayor flexibilidad.
2. **Compresión y Regularización Implícita ($16 \to 8$):** La segunda capa oculta reduce las representaciones a 8 neuronas. Esta reducción en forma de embudo obliga a la red a condensar la información, eliminando redundancias y reteniendo únicamente los patrones latentes más significativos para el horizonte $t+24$. Esto actúa como un mecanismo preventivo frente al sobreajuste.
3. **No Linealidad en Dos Etapas:** La composición de dos funciones de activación continuas no lineales diferenciables permite al modelo aproximar superficies de regresión de mayor curvatura en comparación con una sola capa oculta.

---

## 6.3. Hiperparámetros de Entrenamiento

A fin de asegurar un marco experimental riguroso y equitativo, ambas arquitecturas se entrenan bajo **condiciones idénticas de optimización**:

| Hiperparámetro | Valor Asignado | Justificación Técnica |
| :--- | :---: | :--- |
| Tasa de Aprendizaje ($\eta$) | $0,01$ | Tasa moderada que evita oscilaciones destructivas en el espacio de gradientes y asegura un descenso controlado y suave hacia el mínimo local. |
| Número de Épocas | $500$ | Proporciona un ciclo suficiente de iteraciones sobre las 33.384 muestras para verificar la estabilidad asintótica de la curva de aprendizaje. |
| Modo de Optimización | Descenso de Gradiente por Lote (*Full-Batch*) | Procesa el tensor completo de entrenamiento en cada época, calculando el gradiente analítico exacto de la función de coste y evitando el ruido estocástico en la convergencia. |
| Inicialización de Pesos | Uniforme $\mathcal{U}(-0,5; \, 0,5)$ | Rompe la simetría entre nodos sin posicionar los valores iniciales en las zonas asintóticas saturadas de la función `tansig` ($|a| > 2$), previniendo el desvanecimiento prematuro del gradiente. |
| Semilla Aleatoria (`seed`) | $42$ | Garantiza la repetibilidad exacta de las matrices iniciales $\mathbf{W}^{(0)}$ y $\mathbf{b}^{(0)}$ en cualquier ejecución. |

---

## 6.4. Tabla Comparativa de Ambas Arquitecturas

| Criterio de Comparación | Arquitectura de Referencia | Arquitectura Propia Propuesta |
| :--- | :---: | :---: |
| Topología de Red | $10:10:1$ | $10:16:8:1$ |
| Número de Capas Ocultas | 1 | 2 |
| Neuronas Ocultas Totales | 10 | 24 ($16 + 8$) |
| Funciones en Capas Ocultas | `tansig` | `tansig` / `tansig` |
| Función en Capa de Salida | `purelin` | `purelin` |
| Conexiones Ponderadas (Pesos) | 110 | 296 |
| Términos de Sesgo (Biases) | 11 | 25 |
| **Total de Parámetros Calibrables** | **121** | **321** |
| Factor de Complejidad Paramétrica | $1,0 \times$ (Línea base) | $2,65 \times$ |
| Estructura de Representación | Mapeo plano monocapa | Embudo jerárquico de abstracción |
