# 4. DEFINICIÓN DEL PROBLEMA Y CASO DE ESTUDIO

---

## 4.1. Descripción del Problema Ambiental

El **material particulado fino PM2.5** hace referencia al conjunto de partículas sólidas y líquidas en suspensión en la atmósfera cuyo diámetro aerodinámico es inferior a 2,5 micrómetros. A esta escala, las partículas son lo suficientemente pequeñas para penetrar las vías respiratorias superiores e inferiores y depositarse directamente en los alvéolos pulmonares, desde donde pueden pasar al torrente sanguíneo. La Organización Mundial de la Salud (OMS) clasifica el PM2.5 como el contaminante atmosférico de mayor riesgo para la salud pública a nivel global, vinculándolo a enfermedades cardiovasculares, accidentes cerebrovasculares, cáncer de pulmón e infecciones respiratorias agudas.

La ciudad de **Beijing, China**, ha experimentado históricamente episodios extremos de contaminación por PM2.5, agravados por su geografía (cuenca rodeada de montañas que limitan la dispersión atmosférica), por la alta densidad vehicular y por la quema de carbón en calefacción residencial e industria. La anticipación de la concentración de PM2.5 con un horizonte de **24 horas** permite a las autoridades sanitarias emitir alertas preventivas, restringir el tráfico vehicular, limitar actividades al aire libre y activar protocolos de emergencia ambiental con suficiente antelación para resultar efectivos.

El problema de predicción de PM2.5 presenta tres retos analíticos que impiden resolverlo con modelos lineales simples:
* **Multivariabilidad:** La concentración de PM2.5 está determinada simultáneamente por múltiples factores meteorológicos y por la propia inercia temporal de la serie.
* **No linealidad:** Las interacciones entre temperatura, presión, humedad y viento no son lineales ni aditivas; sus efectos combinados sobre la dispersión de partículas responden a dinámicas atmosféricas complejas.
* **Dependencia temporal:** La concentración de PM2.5 en el instante $t$ guarda correlación estadística con sus valores en instantes anteriores ($t-1$, $t-2$, $t-3$), lo que introduce retardos autorregresivos que deben modelarse explícitamente.

Estas características convierten a la predicción de PM2.5 en un caso de estudio idóneo para las Redes Neuronales Multicapa como aproximadores de funciones no lineales multivariables.

---

## 4.2. Descripción del Dataset: *Beijing PM2.5 Dataset*

El conjunto de datos utilizado en este trabajo es una versión preprocesada del **Beijing PM2.5 Dataset**, ampliamente empleado como banco de pruebas en la literatura de predicción de calidad del aire con métodos de aprendizaje automático. El archivo `dataset_pm25.csv` contiene **11 columnas**: 10 variables de entrada (*features*) y 1 variable objetivo (*target*).

### Tabla de Variables del Dataset

| Variable | Tipo | Rango / Unidad | Rol | Justificación Técnica |
| :--- | :---: | :---: | :---: | :--- |
| `TEMP` | Continua | $°C$ | Entrada meteorológica | La temperatura modula la velocidad de las reacciones fotoquímicas en la atmósfera y afecta la estabilidad vertical del aire, determinando si los contaminantes se acumulan o se dispersan verticalmente. |
| `PRES` | Continua | $hPa$ | Entrada meteorológica | Una presión atmosférica elevada se asocia a condiciones anticiclónicas estables que favorecen la acumulación de contaminantes en las capas bajas. Presiones bajas, en cambio, promueven la mezcla turbulenta y la dispersión. |
| `DEWP` | Continua | $°C$ | Entrada meteorológica | El punto de rocío es un indicador directo de la humedad relativa del aire. Niveles altos de humedad favorecen la formación de aerosoles secundarios y la higroscopicidad de las partículas, elevando el PM2.5 efectivo medido. |
| `Iws` | Continua | $m/s$ | Entrada meteorológica | La velocidad acumulada del viento es el principal factor de ventilación y dispersión de contaminantes: a mayor velocidad, mayor transporte y dilución de PM2.5; a menor velocidad, mayor acumulación local. |
| `hour` | Discreta | $0 - 23$ | Entrada temporal | El ciclo diurno governa la intensidad del tráfico vehicular (hora pico mañana/tarde), la actividad industrial y la intensidad de la radiación solar que reactiva procesos fotoquímicos de formación de partículas secundarias. |
| `day` | Discreta | $1 - 365$ | Entrada temporal | Captura la estacionalidad anual: en invierno, el uso intensivo de calefacción a base de carbón en Beijing eleva significativamente las emisiones de PM2.5; en verano, las condiciones de mayor radiación y precipitación tienden a reducirlas. |
| `pm25_t` | Continua | $\mu g/m^3$ | Entrada autorregresiva | Nivel de PM2.5 en el instante actual $t$. Es el regresor autorregresivo de orden 0: la concentración presente es el predictor más directo de la concentración futura a corto plazo. |
| `pm25_t-1` | Continua | $\mu g/m^3$ | Entrada autorregresiva | Retardo temporal de 1 hora: captura la **inercia a corto plazo** de la serie. Si la concentración fue alta en $t-1$, es probable que siga siendo alta en $t$ y en $t+24$. |
| `pm25_t-2` | Continua | $\mu g/m^3$ | Entrada autorregresiva | Retardo temporal de 2 horas: permite al modelo identificar **tendencias** de ascenso o descenso sostenido en la contaminación. |
| `pm25_t-3` | Continua | $\mu g/m^3$ | Entrada autorregresiva | Retardo temporal de 3 horas: incorpora **memoria autorregresiva** de mayor profundidad, útil para detectar patrones de acumulación prolongada o dispersión gradual. |
| **`pm25_t+24`** | **Continua** | $\mu g/m^3$ | **Salida objetivo (target)** | **Concentración de PM2.5 a predecir, 24 horas hacia el futuro desde el instante $t$. Es la única variable objetivo del problema de regresión.** |

---

## 4.3. Definición Formal del Problema

El objetivo es aprender una función de mapeo no lineal $f: \mathbb{R}^{10} \to \mathbb{R}$ que, dado un vector de entrada:

$$\mathbf{x} = \bigl[\,\text{TEMP},\;\text{PRES},\;\text{DEWP},\;\text{Iws},\;\text{hour},\;\text{day},\;pm25_t,\;pm25_{t-1},\;pm25_{t-2},\;pm25_{t-3}\,\bigr]^T \in \mathbb{R}^{10}$$

produzca una predicción continua:

$$\hat{y} = f(\mathbf{x}) \approx pm25_{t+24} \in \mathbb{R}^+$$

Se trata de un problema de **regresión no lineal multivariable**. La elección de la Red Neuronal Multicapa (MLP) como modelo está plenamente justificada por el Teorema Universal de Aproximación (Cybenko, 1989): una red MLP con al menos una capa oculta y funciones de activación no lineales diferenciables puede aproximar $f$ con error arbitrariamente pequeño, siempre que disponga de suficientes neuronas y datos de entrenamiento representativos.

El proceso de aprendizaje consiste en encontrar el conjunto de parámetros $\Theta = \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)}\}_{l=1}^{L}$ que minimiza el Error Cuadrático Medio sobre el conjunto de entrenamiento:

$$\Theta^* = \arg\min_\Theta \frac{1}{N_{train}} \sum_{k=1}^{N_{train}} \left(y_k - f(\mathbf{x}_k;\, \Theta)\right)^2$$

y que, al aplicarse sobre el conjunto de prueba (datos nunca vistos durante el entrenamiento), mantiene un error bajo como evidencia de **capacidad de generalización**.
