# 3. FUNDAMENTACIÓN TEÓRICA

---

## 3.1. Limitación del Perceptrón Simple y Necesidad de Capas Ocultas

El Perceptrón Simple opera como un **discriminador lineal**: genera una salida a partir de la combinación ponderada de sus entradas aplicada a una función de activación escalón. Su frontera de decisión es siempre un hiperplano en $\mathbb{R}^n$, descrito por la ecuación:

$$\mathbf{W}^T \mathbf{x} + b = 0$$

Esta restricción implica que el Perceptrón Simple solo puede resolver problemas **linealmente separables**, es decir, aquellos en que existe al menos un hiperplano capaz de separar todas las muestras de clase positiva de las de clase negativa. En 1969, Minsky y Papert demostraron formalmente que la función lógica **XOR** no satisface esta condición:

| $x_1$ | $x_2$ | $y_{XOR}$ |
| :---: | :---: | :---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Al graficar estos cuatro puntos en el plano $\mathbb{R}^2$, las dos clases ($y=0$ y $y=1$) se distribuyen en esquinas opuestas del cuadrado unitario: ninguna recta los separa. Esta limitación no es particular del XOR; es extensible a toda función que requiera una frontera curva, multimodal o segmentada. Resolver estos problemas exige la incorporación de **transformaciones no lineales en capas intermedias**, dando origen a las Redes Neuronales Multicapa.

---

## 3.2. Arquitectura de la Red Neuronal Multicapa (MLP)

Una red MLP se organiza en $L$ capas secuenciales:

* **Capa de entrada ($l=0$):** Recibe el vector de características $\mathbf{x} \in \mathbb{R}^n$. En este trabajo, $n = 10$ (las diez variables del dataset de Beijing). No aplica transformación alguna; solo distribuye las entradas hacia la primera capa oculta.
* **Capas ocultas ($l = 1, \ldots, L-1$):** Cada capa contiene un número $n_l$ de neuronas. Cada neurona recibe el vector de activación de la capa anterior, calcula una combinación lineal ponderada (denominada **entrada neta**) y la pasa por una función de activación no lineal.
* **Capa de salida ($l = L$):** Produce la predicción final $\hat{y}$. Al tratarse de un problema de **regresión continua** (predecir una concentración real de PM2.5), esta capa utiliza una función de activación lineal.

### Propagación Hacia Adelante (*Forward Pass*)

Para cada capa $l$ y cada patrón de entrada $\mathbf{x}$, la activación se calcula en dos pasos:

**Paso 1 — Entrada neta (combinación lineal):**

$$\mathbf{a}^{(l)} = \mathbf{W}^{(l)} \cdot \mathbf{o}^{(l-1)} + \mathbf{b}^{(l)}$$

donde:
- $\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ es la **matriz de pesos** de la capa $l$. El elemento $W_{ij}^{(l)}$ representa el peso de la conexión desde la neurona $j$ de la capa $l-1$ hacia la neurona $i$ de la capa $l$.
- $\mathbf{b}^{(l)} \in \mathbb{R}^{n_l}$ es el **vector de sesgos** (*bias*) de la capa $l$.
- $\mathbf{o}^{(l-1)} \in \mathbb{R}^{n_{l-1}}$ es el **vector de salidas** (activaciones) de la capa anterior. Para la capa de entrada: $\mathbf{o}^{(0)} = \mathbf{x}$.

**Paso 2 — Función de activación:**

$$\mathbf{o}^{(l)} = f^{(l)}\!\left(\mathbf{a}^{(l)}\right)$$

donde $f^{(l)}$ es la función de activación de la capa $l$, aplicada elemento a elemento. La elección de $f^{(l)}$ es distinta para las capas ocultas y para la capa de salida, según se detalla en la sección 3.3.

El proceso se repite capa tras capa, propagando la información hacia adelante hasta obtener la predicción en la capa de salida:

$$\hat{y} = o^{(L)} = f^{(L)}\!\left(\mathbf{W}^{(L)} \cdot \mathbf{o}^{(L-1)} + \mathbf{b}^{(L)}\right)$$

---

## 3.3. Funciones de Activación

La elección de la función de activación por capa es una decisión de diseño central en el MLP. A continuación se presentan las tres funciones relevantes para este trabajo.

### 3.3.1. Sigmoide Logística (`logsig`)

$$f(a) = \sigma(a) = \frac{1}{1 + e^{-a}}$$

- **Rango de salida:** $(0, \, 1)$
- **Derivada:** $f'(a) = \sigma(a)\,\bigl(1 - \sigma(a)\bigr) = \mathbf{o}\,(1 - \mathbf{o})$
- **Uso:** Capas ocultas. Introduce la no linealidad necesaria para aprender fronteras complejas.
- **Limitación:** Sufre el problema del **gradiente desvanecido** (*vanishing gradient*) para valores de $a$ muy grandes o muy pequeños, donde $f'(a) \approx 0$ y el aprendizaje se ralentiza en capas profundas.

### 3.3.2. Tangente Hiperbólica (`tansig`)

$$f(a) = \tanh(a) = \frac{e^{a} - e^{-a}}{e^{a} + e^{-a}}$$

- **Rango de salida:** $(-1, \, +1)$
- **Derivada:** $f'(a) = 1 - \tanh^2(a) = 1 - \mathbf{o}^2$
- **Uso:** Capas ocultas. Al estar centrada en cero, sus gradientes son más simétricos que los de la sigmoide logística, lo que generalmente acelera la convergencia.
- **Relación con `logsig`:** $\tanh(a) = 2\,\sigma(2a) - 1$; es una versión escalada y desplazada de la sigmoide logística.

### 3.3.3. Función Lineal o Identidad (`purelin`)

$$f(a) = a$$

- **Rango de salida:** $(-\infty, \, +\infty)$
- **Derivada:** $f'(a) = 1$
- **Uso:** **Capa de salida en problemas de regresión.** Al no acotar la salida, permite que la red produzca cualquier valor real, lo que es indispensable para predecir la concentración de PM2.5 ($\mu g/m^3$), cuyo rango no está limitado a $(0,1)$ ni a $(-1,+1)$.

> **Regla de diseño fundamental:** Las capas ocultas deben usar funciones de activación **no lineales** (`logsig` o `tansig`) para que la red tenga capacidad de aproximar funciones complejas. La capa de salida usa `purelin` en regresión para no restringir el rango de la predicción.

---

## 3.4. Algoritmo de Retropropagación del Error (*Backpropagation*)

El algoritmo de *Backpropagation* es el mecanismo de aprendizaje que ajusta los parámetros $\{\mathbf{W}^{(l)}, \mathbf{b}^{(l)}\}$ de todas las capas de la red para minimizar el error entre la predicción $\hat{y}$ y el valor real $y$. Opera en dos fases por cada patrón o lote de patrones de entrenamiento.

### 3.4.1. Función de Coste: Error Cuadrático Medio (MSE)

Para $N$ muestras de entrenamiento, la función de coste que se minimiza es:

$$\mathcal{L} = \text{MSE} = \frac{1}{N} \sum_{k=1}^{N} \left(y_k - \hat{y}_k\right)^2$$

El objetivo del entrenamiento es encontrar los parámetros que minimizan $\mathcal{L}$, lo cual se realiza mediante **descenso del gradiente**: actualizar los pesos en la dirección opuesta al gradiente de la función de coste.

### 3.4.2. Fase 1 — Propagación Hacia Adelante

Se ejecuta el *forward pass* completo (Sección 3.2) para obtener la predicción $\hat{y}$ de la red con los parámetros actuales.

### 3.4.3. Fase 2 — Retropropagación del Gradiente

Se calcula el error en la salida y se propaga hacia atrás, capa por capa, usando la **regla de la cadena**.

**Gradiente local en la capa de salida ($l = L$):**

$$\boldsymbol{\delta}^{(L)} = -\left(y - \hat{y}\right) \cdot f'^{(L)}\!\left(\mathbf{a}^{(L)}\right)$$

Como la capa de salida usa `purelin`, $f'^{(L)}(a) = 1$, por lo que:

$$\boldsymbol{\delta}^{(L)} = -(y - \hat{y}) = \hat{y} - y$$

**Gradiente local en las capas ocultas ($l = L-1, \ldots, 1$):**

Aplicando la regla de la cadena hacia atrás:

$$\boldsymbol{\delta}^{(l)} = \left(\mathbf{W}^{(l+1)}\right)^T \cdot \boldsymbol{\delta}^{(l+1)} \odot f'^{(l)}\!\left(\mathbf{a}^{(l)}\right)$$

donde $\odot$ denota el producto elemento a elemento (*Hadamard*). El término $\left(\mathbf{W}^{(l+1)}\right)^T \cdot \boldsymbol{\delta}^{(l+1)}$ representa la contribución al error que la capa $l+1$ "devuelve" a la capa $l$, ponderada por los pesos que las conectan.

Las derivadas de las funciones de activación en términos de las salidas ya calculadas son:
- Para `logsig`: $f'(a^{(l)}) = o^{(l)}\,(1 - o^{(l)})$
- Para `tansig`: $f'(a^{(l)}) = 1 - (o^{(l)})^2$

### 3.4.4. Actualización de Parámetros

Con los gradientes locales calculados en todas las capas, los pesos y sesgos se actualizan mediante la regla del descenso del gradiente con tasa de aprendizaje $\eta$:

$$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \cdot \boldsymbol{\delta}^{(l)} \cdot \left(\mathbf{o}^{(l-1)}\right)^T$$

$$\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \cdot \boldsymbol{\delta}^{(l)}$$

Este proceso —*forward pass*, cálculo del error, *backward pass* y actualización— se repite para cada época de entrenamiento. Una **época** corresponde a un pase completo sobre todo el conjunto de entrenamiento.

### 3.4.5. Hiperparámetros del Entrenamiento

- **Tasa de aprendizaje ($\eta$):** Controla el tamaño del paso en cada actualización. Un $\eta$ muy grande puede causar divergencia (el error oscila o aumenta); uno muy pequeño produce una convergencia lenta. Es el hiperparámetro de mayor impacto en la dinámica del entrenamiento.
- **Número de épocas:** Define cuántas veces pasa el algoritmo sobre el conjunto de entrenamiento completo. Se selecciona observando cuándo la curva de pérdida deja de descender significativamente.

---

## 3.5. Normalización de Datos y Prevención del *Data Leakage*

Las entradas del dataset de Beijing presentan **escalas radicalmente distintas**: `PRES` opera en el rango de cientos de hPa, mientras que `hour` varía entre 0 y 23, y `Iws` (velocidad del viento acumulada) puede tomar valores decimales pequeños. Operar directamente sobre estas escalas provoca que las neuronas sean mucho más sensibles a las variables de mayor magnitud, distorsionando el aprendizaje y produciendo gradientes mal condicionados. La normalización previa es, por tanto, **indispensable**.

### 3.5.1. Normalización Min-Max al Rango $[0, 1]$

$$x'_j = \frac{x_j - x_{j,\min}}{x_{j,\max} - x_{j,\min}}$$

Comprime cada variable $x_j$ a $[0,1]$. Es compatible con la función de activación `logsig` en capas ocultas, cuya salida también opera en $(0,1)$.

### 3.5.2. Estandarización Z-Score

$$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

Centra cada variable en media cero y escala según su desviación estándar, produciendo una distribución con $\mu' = 0$ y $\sigma' = 1$. Es preferible cuando los datos presentan distribuciones aproximadamente gaussianas o cuando se usa `tansig` (centrada en cero) en capas ocultas.

### 3.5.3. Ajuste Exclusivo sobre el Conjunto de Entrenamiento

El punto más crítico del preprocesamiento es que los parámetros del normalizador ($x_{\min}$, $x_{\max}$, $\mu$, $\sigma$) **deben calcularse únicamente sobre el conjunto de entrenamiento** y, posteriormente, aplicarse tanto al conjunto de entrenamiento como al de prueba:

```python
# Correcto — sin data leakage
scaler.fit(X_train)          # Aprende parámetros solo de train
X_train_norm = scaler.transform(X_train)
X_test_norm  = scaler.transform(X_test)  # Aplica los mismos parámetros
```

### 3.5.4. El Problema del *Data Leakage*

Si los parámetros de normalización se calculan sobre el conjunto completo (train + test) antes de la partición, el modelo obtiene información implícita sobre el comportamiento estadístico del conjunto de prueba durante el entrenamiento. Esto genera estimaciones de rendimiento artificialmente optimistas que no reflejan la capacidad real de generalización del modelo sobre datos nunca vistos. En series temporales como la de PM2.5, el *data leakage* es especialmente crítico porque mezclaría estadísticas de observaciones futuras con el entrenamiento del modelo.

---

## 3.6. Métricas de Desempeño en Regresión: MSE y RMSE

### 3.6.1. Error Cuadrático Medio (MSE)

$$\text{MSE} = \frac{1}{N} \sum_{k=1}^{N} \left(y_k - \hat{y}_k\right)^2$$

- Penaliza errores grandes de forma cuadrática: un error doble produce una penalización cuádruple. Esto lo hace sensible a valores atípicos (*outliers*) y útil para detectarlos.
- Sus unidades son $(\mu g/m^3)^2$, lo que dificulta la interpretación directa en la escala original del fenómeno.

### 3.6.2. Raíz del Error Cuadrático Medio (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{k=1}^{N} \left(y_k - \hat{y}_k\right)^2}$$

- Sus unidades son $\mu g/m^3$, **la misma escala que la variable objetivo**. Esto permite interpretar directamente el error promedio de la predicción: un RMSE de, por ejemplo, 30 $\mu g/m^3$ indica que la predicción de PM2.5 se desvía en promedio 30 $\mu g/m^3$ del valor real.
- Es la métrica principal de comparación entre las dos arquitecturas diseñadas en este trabajo.

> **Criterio de comparación:** La arquitectura que produzca el **menor RMSE sobre el conjunto de prueba** ofrece la mayor precisión predictiva. El conjunto de prueba nunca participa en el entrenamiento ni en la normalización, por lo que su error es un estimador honesto de la capacidad de generalización del modelo.
