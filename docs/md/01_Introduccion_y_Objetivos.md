# 1. INTRODUCCIÓN

Las Redes Neuronales Artificiales (RNA) representan uno de los enfoques más poderosos y versátiles dentro de la Inteligencia Artificial moderna. Su inspiración biológica, fundamentada en el funcionamiento del sistema nervioso humano, dio origen a modelos capaces de aprender representaciones complejas del mundo a partir de datos. Entre estos modelos, el **Perceptrón Simple** —propuesto por Frank Rosenblatt en 1958— constituyó el primer clasificador neuronal con capacidad de aprendizaje adaptativo; sin embargo, su operación quedó estrictamente acotada a problemas linealmente separables: aquellos donde una frontera de decisión rectilínea (un hiperplano) es suficiente para dividir las clases en el espacio de entradas. En 1969, Minsky y Papert demostraron formalmente que el Perceptrón Simple es incapaz de resolver funciones tan básicas como la disyunción exclusiva (XOR), impulsando así el desarrollo de arquitecturas con mayor capacidad expresiva.

Las **Redes Neuronales Multicapa** (*Multilayer Perceptron*, MLP) superan esta limitación incorporando capas intermedias —denominadas capas ocultas— entre la entrada y la salida. Cada capa oculta aplica transformaciones no lineales sobre el vector de activación de la capa anterior, construyendo representaciones jerarquizadas y progresivamente más abstractas del espacio de características. El resultado es un modelo cuya potencia expresiva queda garantizada por el **Teorema Universal de Aproximación** (Cybenko, 1989; Hornik, 1991): cualquier función continua definida sobre un conjunto compacto puede ser aproximada con precisión arbitraria por una red MLP con al menos una capa oculta y funciones de activación no lineales diferenciables. El mecanismo que hace posible el aprendizaje en estas arquitecturas profundas es el algoritmo de **Retropropagación del Error** (*Backpropagation*), que propaga el gradiente del error desde la capa de salida hacia las capas internas mediante la regla de la cadena del cálculo diferencial, actualizando iterativamente todos los pesos sinápticos y sesgos de la red.

En este trabajo se aplican estos fundamentos a un problema concreto de alta relevancia ambiental y epidemiológica: la **predicción de la concentración de material particulado fino PM2.5 en la ciudad de Beijing, China, con un horizonte temporal de 24 horas**. El material particulado PM2.5 —partículas de diámetro aerodinámico inferior a 2,5 micrómetros— representa uno de los contaminantes atmosféricos más peligrosos para la salud humana, asociado a enfermedades respiratorias y cardiovasculares crónicas. Anticipar su concentración con un día de antelación permite a autoridades de salud pública y a la población en general tomar decisiones preventivas oportunas. El conjunto de datos empleado integra retardos autorregresivos de la propia serie de PM2.5 ($pm25_t$, $pm25_{t-1}$, $pm25_{t-2}$, $pm25_{t-3}$), variables meteorológicas (temperatura, presión, punto de rocío, velocidad del viento) e índices temporales (hora del día, día del año), conformando un vector de **10 entradas** y una única salida continua ($pm25_{t+24}$). Se diseñan, entrenan y evalúan comparativamente **dos arquitecturas MLP**: una de referencia, con topología estándar sugerida por el docente, y una propia con justificación razonada de sus decisiones de diseño. Toda decisión metodológica que se aparte de los procedimientos explicados en clase queda explícitamente fundamentada en este documento.

---

# 2. OBJETIVOS

## 2.1. Objetivo General

Diseñar, implementar y evaluar comparativamente dos Redes Neuronales Multicapa (MLP) para la predicción de la concentración de PM2.5 con horizonte temporal de 24 horas, mediante el algoritmo de retropropagación del error aplicado sobre el *Beijing PM2.5 Dataset*, con el propósito de analizar la influencia de la arquitectura sobre la capacidad de aproximación no lineal y demostrar el proceso completo de aprendizaje supervisado en redes de múltiples capas.

## 2.2. Objetivos Específicos

1. Analizar los fundamentos teóricos y matemáticos de la arquitectura MLP: propagación hacia adelante, funciones de activación diferenciables y el algoritmo de retropropagación del error como optimización por descenso del gradiente.

2. Inspeccionar, limpiar y preparar el *Beijing PM2.5 Dataset*, aplicando una partición rigurosa en conjuntos de entrenamiento y prueba con normalización ajustada exclusivamente sobre el conjunto de entrenamiento, previniendo la fuga de información (*data leakage*).

3. Diseñar dos arquitecturas MLP diferenciadas —una de referencia y una propia—, justificando la elección de topología, número de neuronas por capa y funciones de activación en cada una.

4. Implementar el entrenamiento supervisado de ambas arquitecturas mediante *Backpropagation*, registrando la evolución del error cuadrático medio (MSE) por época hasta la convergencia.

5. Evaluar cuantitativamente el desempeño de cada modelo sobre el conjunto de prueba mediante las métricas MSE y RMSE, y representar gráficamente el PM2.5 real frente al predicho en escala original ($\mu g/m^3$).

6. Comparar ambas arquitecturas, identificar la influencia de cada variable de entrada sobre la predicción, diagnosticar la presencia de sobreajuste y formular conclusiones fundamentadas en los resultados experimentales.
