# MLflow Project & Cookiecutter Data Science

El proyecto implementa una arquitectura modular basada en el estándar **Cookiecutter Data Science (CCDS)**, empaquetado como módulo de Python (`heart_ccds`) e integrado con **MLflow Tracking**, **MLflow Projects** y **Model Registry**.

---

## 1. Preparar el ambiente (solo la primera vez)

Activa el entorno virtual del proyecto (o crea uno nuevo):

```bash
virtualenv <env-name> --python=python3.13
source <env-path>/bin/activate
pip install -r requirements.txt
```

Instala las dependencias y registra el paquete local en modo editable:

```bash
pip install -r requirements.txt
```

En cada terminal nueva, activa el ambiente y define el experimento de MLflow:

```bash
source <env-path>/bin/activate
export MLFLOW_EXPERIMENT_NAME=an-example-name
```

---

## 2. Correr el proyecto

### Opción A: Usando MLflow Projects

```bash
mlflow run . --env-manager local
```

Variación de hiperparámetros para Random Forest Classifier:

```bash
mlflow run . -P n_estimators=200 -P max_depth=10 --env-manager local
mlflow run . -P n_estimators=300 -P max_depth=15 --env-manager local
```

Al terminar se imprimen las métricas calculadas y el URI del modelo:
```text
==================================================
Training finished successfully!
MLflow Run ID: 371defb74e1946ebafcb82f4e526b346
Model URI: runs:/371defb74e1946ebafcb82f4e526b346/model
Metrics:
  - accuracy: 0.8804
  - precision: 0.8846
  - recall: 0.9020
  - f1_score: 0.8932
==================================================
```

---

## 3. Ver los resultados

Inicia la interfaz de usuario de MLflow:

```bash
mlflow ui
```

Abre http://127.0.0.1:5000 en tu navegador, selecciona varios runs y da clic en **Compare** para analizar cómo varían las métricas (`accuracy`, `precision`, `recall`, `f1_score`) en función de `n_estimators`, `max_depth`, `min_samples_split` y `min_samples_leaf`.

No hace falta tener la interfaz abierta mientras corres los entrenamientos: puedes abrirla después para revisar el historial completo.

---

## 4. Estructura del proyecto

```text
heart-ccds/
├── MLproject              # Entry points y parámetros que puede ejecutar MLflow
├── Makefile               # Atajos de terminal (`make data`, `make train`, `make lint`)
├── README.md              # Documentación general y guía de uso del proyecto
├── pyproject.toml         # Configuración del paquete editable y herramientas de linting
├── requirements.txt       # Dependencias de Python reproducibles
├── main.py                # Script CLI principal para entrenamiento y registro
├── data/
│   ├── raw/               # heart.csv (dataset original inmutable)
│   ├── interim/           # interim.csv (datos verificados y limpios)
│   └── processed/         # train.csv y test.csv (particiones estratificadas listas para modelar)
├── notebooks/             # Notebooks de Jupyter con el flujo interactivo modularizado
├── reports/
│   ├── figures/           # Gráficas generadas del EDA (histogramas, boxplots, distribuciones)
│   └── predictions.csv    # Predicciones generadas por el paso de inferencia
└── heart_ccds/            # Código fuente modular del paquete Python
    ├── __init__.py        # Inicializador del paquete
    ├── config.py          # Rutas del proyecto, columnas de features y constantes globales
    ├── dataset.py         # Paso load: validación y limpieza inicial del dataset
    ├── features.py        # Paso features: ColumnTransformer (PowerTransformer + OneHotEncoder) y split
    ├── plots.py           # Funciones reutilizables para visualización y EDA
    └── modeling/
        ├── __init__.py
        ├── train.py       # Paso train: entrenamiento del pipeline y logging en MLflow
        └── predict.py     # Paso predict: carga de artefactos MLflow, evaluación e inferencia
```

El modelo entrenado es un pipeline integral de Scikit-learn:
- **Numéricas (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`):** `PowerTransformer(method='yeo-johnson')` para estabilizar varianza y normalizar sesgos.
- **Categóricas (`Sex`, `ChestPainType`, `FastingBS`, `RestingECG`, `ExerciseAngina`, `ST_Slope`):** `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- **Clasificador:** `RandomForestClassifier` optimizado con hiperparámetros configurables vía CLI.
