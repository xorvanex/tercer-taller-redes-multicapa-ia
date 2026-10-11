# TERCER TALLER: REDES NEURONALES MULTICAPA (MLP)
### Predicción de Material Particulado Fino (PM2.5 a 24 horas) mediante Perceptrón Multicapa y Backpropagation Analítico

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Licencia-MIT-green?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Universidad-De%20Cartagena-red?style=for-the-badge" alt="Unicartagena"/>
  <img src="https://img.shields.io/badge/Asignatura-Inteligencia%20Artificial-blue?style=for-the-badge" alt="Asignatura"/>
  <a href="https://colab.research.google.com/drive/1eXevX6U_6He5snGHMdMV9cGOJR5pyI5L?usp=sharing" target="_blank">
    <img src="https://img.shields.io/badge/Google%20Colab-Auditor%C3%ADa%20Interactiva-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab"/>
  </a>
  <img src="https://img.shields.io/badge/Dataset-Beijing%20PM2.5%20(41.730)-teal?style=for-the-badge" alt="Dataset"/>
</p>

---

## Enlaces Directos a los Entregables Principales

* **Informe Técnico Oficial (PDF):** [`docs/Taller_RNA_MLP_IA_PALMERA_CAMARGO.pdf`](docs/Taller_RNA_MLP_IA_PALMERA_CAMARGO.pdf)
* **Documento Editable (Word):** [`docs/Taller_RNA_MLP_IA_PALMERA_CAMARGO.docx`](docs/Taller_RNA_MLP_IA_PALMERA_CAMARGO.docx)
* **Libro de Auditoría y Control de Calidad (Excel):** [`docs/Auditoria_Estadistica_Dataset_PM25.xlsx`](docs/Auditoria_Estadistica_Dataset_PM25.xlsx)
* **Cuaderno Interactivo en la Nube:** [Google Colab — Auditoría Beijing PM2.5](https://colab.research.google.com/drive/1eXevX6U_6He5snGHMdMV9cGOJR5pyI5L?usp=sharing)
* **Dataset Oficial Preprocesado:** [`data/datasets/dataset_pm25.csv`](data/datasets/dataset_pm25.csv)

---

## Guía Rápida de Instalación y Ejecución

Siga estos pasos para configurar el entorno y ejecutar la solución en cualquier sistema operativo (Windows, Linux o macOS):

### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/xorvanex/tercer-taller-redes-multicapa-ia.git
cd tercer-taller-redes-multicapa-ia
```

### Paso 2: Crear y Activar un Entorno Virtual
En Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
En Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias
```bash
pip install -r requirements.txt
```

---

## Modos de Ejecución del Proyecto

El sistema ofrece dos modalidades de ejecución a través del punto de entrada [`src/main.py`](src/main.py):

### Modo 1: Interfaz Gráfica Interactiva de Escritorio (GUI) — Recomendado
Inicia el panel de control interactivo construido en **CustomTkinter** y lienzos vectoriales de **Matplotlib**:
```bash
python src/main.py --gui
```
**Características de la GUI:**
1. **Topología y Relaciones:** Tres vistas conmutables:
   * *Grafo Sináptico Nítido:* Visualización de nodos circulares con sinapsis coloreadas por intensidad y filtro de enfoque para aislar la influencia de cualquier variable de entrada individual.
   * *Diagrama de Bloques (estilo MATLAB):* Esquema modular con bloques de suma, funciones `tansig` y salida lineal `purelin`.
   * *Mapa de Calor de Pesos ($W^{(1)}$):* Matriz numérica de pesos de la primera capa oculta.
2. **Curvas de Aprendizaje:** Monitor en tiempo real del progreso de entrenamiento durante las 500 épocas, con barra de progreso y guía conceptual de hiperparámetros.
3. **Evaluación y Métricas:** Tarjetas dinámicas de KPIs cuantitativos (MSE, RMSE, MAE) en escala normalizada y física real ($\mu\text{g/m}^3$), cálculo de ganancia del modelo propio y gráfico de 200 horas continuas de predicción vs. valor real.
4. **Simulador en Tiempo Real:** Entrada interactiva de parámetros meteorológicos y retardos de contaminación, proyección de trayectoria temporal ($t-3 \to t \to t+24$) y clasificación del riesgo ambiental bajo los estándares de la OMS / EPA.
5. **Botón de Restablecimiento Universal:** Permite regresar toda la interfaz a sus condiciones iniciales con un solo clic.

---

### Modo 2: Consola Automatizada / Benchmark Reproducible (CLI)
Ejecuta el pipeline completo de entrenamiento sobre el tensor de 33.384 muestras, imprime la auditoría cuantitativa y exporta automáticamente las figuras científicas en alta resolución (300 DPI) a la carpeta `docs/img/`:
```bash
python src/main.py --cli
```

---

## Resumen Ejecutivo de Resultados Experimentales

Evaluación sobre las **8.346 observaciones no vistas del conjunto de prueba** (últimos 11 meses de la serie temporal):

| Métrica de Desempeño | Modelo de Referencia (10:10:1) | Modelo Propio Propuesto (10:16:8:1) | Ganancia / Mejora Relativa |
| :--- | :---: | :---: | :---: |
| Topología de Capas | $10 \to 10 \to 1$ | $10 \to 16 \to 8 \to 1$ | $+1$ capa oculta de abstracción |
| Parámetros Libres ($N_{params}$) | 121 | 321 | $+165,3\%$ capacidad representacional |
| Tiempo de Entrenamiento (500 épocas) | 6,79 s | 21,13 s | $+14,34$ s sobre CPU (Full-Batch) |
| **MSE Entrenamiento (Normalizado)** | 0,035462 | 0,031615 | $-10,85\%$ de error cuadrático |
| **MSE Prueba (Normalizado)** | 0,034532 | 0,031509 | **$-8,75\%$ de error cuadrático** |
| **MSE Prueba (Escala Real)** | **8.529,83 μg²/m⁶** | **7.782,96 μg²/m⁶** | **$-746,87$ unidades cuadradas** |
| **RMSE Prueba (Escala Real)** | **92,36 μg/m³** | **88,22 μg/m³** | **$-4,14$ μg/m³ (Mejora del 4,48%)** |
| **MAE Prueba (Escala Real)** | **65,41 μg/m³** | **61,73 μg/m³** | **$-3,68$ μg/m³** |

<p align="center">
  <img src="docs/img/Curvas_Aprendizaje_Comparativas.png" alt="Curvas de Aprendizaje Comparativas" width="85%"/>
</p>
<p align="center">
  <em>Figura 1. Evolución del Error Cuadrático Medio normalizado (MSE) durante 500 épocas sobre el conjunto de entrenamiento y de prueba. La ausencia de divergencia confirma generalización sin sobreajuste.</em>
</p>

<p align="center">
  <img src="docs/img/Comparativa_Prediccion_Real_vs_Predicha.png" alt="Comparativa Temporal Real vs Predicha" width="85%"/>
</p>
<p align="center">
  <em>Figura 2. Seguimiento temporal de concentración de PM2.5 real vs. predicciones a 24 horas en una ventana de 200 horas consecutivas de prueba.</em>
</p>

---

## Descripción del Problema y Dataset

El objetivo del taller es predecir la concentración de material particulado fino **PM2.5 con 24 horas de antelación ($t+24$)** en la ciudad de Beijing, China, a partir de registros meteorológicos, temporales e inerciales.

Se procesa el dataset oficial con **41.730 registros continuos y 11 variables**:

| Variable | Tipo | Unidad / Rango | Rol en el Modelo | Justificación Técnica |
| :--- | :---: | :---: | :---: | :--- |
| `TEMP` | Continua | °C | Entrada | Modula la velocidad fotoquímica y la estabilidad atmosférica vertical. |
| `PRES` | Continua | hPa | Entrada | Condiciona fenómenos anticiclónicos de acumulación o turbulencia de dispersión. |
| `DEWP` | Continua | °C | Entrada | Indicador de humedad relativa y condensación de aerosoles secundarios. |
| `Iws` | Continua | m/s | Entrada | Velocidad acumulada del viento; factor fundamental de ventilación y transporte. |
| `hour` | Discreta | 0 – 23 | Entrada | Ciclo diurno humano (horas pico vehiculares, emisiones industriales y radiación solar). |
| `day` | Discreta | 1 – 31 | Entrada | Periodicidad mensual y estacional de emisiones urbanas. |
| `pm25_t` | Continua | μg/m³ | Entrada | Concentración de PM2.5 en el instante actual $t$ (regresor de orden 0). |
| `pm25_t-1` | Continua | μg/m³ | Entrada | Retardo autorregresivo de 1 hora previa (inercia inmediata). |
| `pm25_t-2` | Continua | μg/m³ | Entrada | Retardo autorregresivo de 2 horas previas (captura de tendencia sostenida). |
| `pm25_t-3` | Continua | μg/m³ | Entrada | Retardo autorregresivo de 3 horas previas (memoria de acumulación de masa). |
| **`pm25_t+24`** | **Continua** | **μg/m³** | **Salida Objetivo** | **Concentración de PM2.5 proyectada a 24 horas hacia el futuro.** |

---

## Control de Calidad y Preprocesamiento

1. **Auditoría Paramétrica en Excel / OnlyOffice:**  
   Libro [`docs/Auditoria_Estadistica_Dataset_PM25.xlsx`](docs/Auditoria_Estadistica_Dataset_PM25.xlsx) con fórmulas dinámicas sobre el rango absoluto `A2:K41731`:
   * 41.730 registros válidos verificados (`=CONTAR`).
   * 0 celdas vacías o nulas (`=CONTAR.BLANCO`).
2. **Partición Temporal Cronológica (80% / 20%):**  
   * **Entrenamiento:** 33.384 observaciones (primeras 33.384 horas consecutivas).
   * **Prueba:** 8.346 observaciones (últimas 8.346 horas consecutivas).
   * *Justificación:* Al tratarse de una serie temporal, mezclar aleatoriamente las muestras (*shuffle*) introduciría **fuga temporal de datos (*temporal data leakage*)**, entrenando al modelo con información del futuro.
3. **Normalización Min-Max Simétrica en $[-1, +1]$:**  
   $$x'_j = 2 \cdot \left( \frac{x_j - x_{j,\min}^{train}}{x_{j,\max}^{train} - x_{j,\min}^{train}} \right) - 1$$
   * Parámetros calculados **exclusivamente sobre el conjunto de entrenamiento** para evitar fuga de información.
   * Intervalo centrado en cero, optimizado para la activación `tansig` en capas ocultas.

---

## Arquitecturas Neuronales Comparadas

* **Modelo de Referencia (Benchmark 10:10:1):**  
  Topología con una sola capa oculta:
  $$10 \xrightarrow{\text{tansig}} 10 \xrightarrow{\text{purelin}} 1$$
  $$N_{params} = (10 \times 10 + 10) + (10 \times 1 + 1) = 110 + 11 = \mathbf{121 \text{ parámetros}}$$
* **Modelo Propio Propuesto (Embudo 10:16:8:1):**  
  Topología con dos capas ocultas en estructura de compresión progresiva:
  $$10 \xrightarrow{\text{tansig}} 16 \xrightarrow{\text{tansig}} 8 \xrightarrow{\text{purelin}} 1$$
  $$N_{params} = (10 \times 16 + 16) + (16 \times 8 + 8) + (8 \times 1 + 1) = 176 + 136 + 9 = \mathbf{321 \text{ parámetros}}$$
  *Justificación:* La capa de 16 neuronas expande la dimensionalidad para capturar interacciones no lineales entre clima y contaminación previa; la capa de 8 neuronas contrae la representación latente actuando como un filtro de regularización implícita frente al ruido.

---

## Estructura del Directorio del Proyecto

```text
tercer-taller-redes-multicapa-ia/
├── data/
│   ├── datasets/
│   │   └── dataset_pm25.csv             # Dataset oficial preprocesado (41.730 registros)
│   └── enunciados/
│       ├── Actividad RNA (MLP).docx     # Directrices y rúbrica oficial del docente
│       ├── Informacion_actividad.txt    # Requisitos extraídos
│       ├── Publicacion_Classroom.png    # Evidencia de publicación en Google Classroom
│       └── Rubrica Octubre 2026.pdf     # Rúbrica de evaluación (8 puntos)
│
├── docs/
│   ├── Auditoria_Estadistica_Dataset_PM25.xlsx # Libro Excel con fórmulas dinámicas
│   ├── Taller_RNA_MLP_IA_PALMERA_CAMARGO.docx  # Documento técnico formal estructurado
│   ├── Taller_RNA_MLP_IA_PALMERA_CAMARGO.pdf   # Informe técnico final en PDF para entrega
│   ├── img/                             # Figuras científicas en alta resolución
│   └── md/                              # Memorias técnicas en Markdown
│
├── notebooks/
│   └── 01_auditoria_y_verificacion_dataset_pm25.ipynb # Cuaderno interactivo (Colab/Jupyter)
│
├── src/
│   ├── __init__.py                      # Inicializador de paquete
│   ├── data_loader.py                   # Carga, partición 80/20 y escalador Min-Max [-1, 1]
│   ├── mlp.py                           # MLP vectorizado en NumPy con Backpropagation analítico
│   ├── app_gui.py                       # Interfaz gráfica moderna en CustomTkinter (4 pestañas)
│   └── main.py                          # Punto de entrada dual (--cli y --gui)
│
├── .gitignore                           # Exclusiones de control de versiones
├── LICENSE                              # Licencia MIT académica
├── README.md                            # Documentación principal del repositorio
└── requirements.txt                     # Dependencias exactas del proyecto
```

---

## Información Institucional y Académica

* **Institución:** Universidad de Cartagena
* **Facultad:** Facultad de Ingeniería
* **Programa:** Ingeniería de Sistemas — Semestre VII
* **Asignatura:** Inteligencia Artificial (Segundo Corte)
* **Docente:** Manuel Alejandro Ospina Alarcón
* **Autores:**
  * **Dago David Palmera Navarro** — Código: `0222321003`
  * **Julián David Camargo Padilla** — Código: `0222320016`
* **Ponderación:** 8 puntos de la calificación definitiva del corte
* **Licencia:** MIT License
