# course-approval-ml

Machine learning project to predict whether a student **passes a subject**, covering the full data
preparation pipeline: ETL, exploratory data analysis, cleaning, encoding, balancing and dimensionality
reduction with PCA.

Built as part of a graduation seminar on Machine Learning.

## Objective

Build a clean, model-ready dataset from raw academic records and analyze the factors that influence
course approval. The final prepared dataset is intended for a downstream binary classification model
(`Aprobo_encoded`: 0 = No, 1 = Yes).

## Dataset

`data/AprobacionCurso2026.csv` — synthetic academic records for one course approval cycle (2026). Each row
represents a student with the following features:

| Variable | Description |
| --- | --- |
| `ID_Estudiante` | Student identifier (removed) |
| `Edad` | Age |
| `Fecha_Inscripcion` | Enrollment date (transformed) |
| `Genero` | Gender (removed) |
| `Estilo_Aprendizaje` | Learning style (removed) |
| `Carrera` | Degree program (one-hot encoded) |
| `Signo_Zodiacal` | Zodiac sign (removed) |
| `Examen_Admision` | Admission exam score |
| `Horas_Estudio_Semanal` | Weekly study hours |
| `Asistencia_Pct` | Attendance percentage |
| `Faltas` | Number of absences (removed, redundant) |
| `Promedio_Parciales` | Average of partial exams |
| `Motivacion` | Motivation level |
| `Nota_Final` | Final grade (removed — data leakage) |
| `Aprobo` | Target: passed (No/Sí) |

## Pipeline

The notebook `notebooks/data_preparation.ipynb` implements the following steps:

1. **Load & initial exploration** — structure, dtypes and basic statistics.
2. **Type conversion** — `object → category`, `float → int`, dates to `datetime`.
3. **Feature engineering** — `Antiguedad` (years since enrollment).
4. **Column cleaning** — drop irrelevant features and columns with data leakage (`Nota_Final`).
5. **Correlation analysis** — Pearson correlation matrix and heatmap.
6. **Profiling report** — HTML report generated with `ydata-profiling`.
7. **Categorical analysis** — value counts, `Carrera` vs. `Aprobo`, Chi-square independence test.
8. **Encoding** — One-Hot Encoding for `Carrera`, Label Encoding for the target.
9. **Outliers** — replaced with `NaN` using the IQR criterion.
10. **Imputation** — missing values filled with `KNNImputer`.
11. **Scaling** — `MinMaxScaler` to the [0, 1] range.
12. **Class balancing** — `RandomUnderSampler` from `imbalanced-learn`.
13. **Dimensionality reduction** — PCA, keeping 3 components plus the target.

## Repository structure

```
course-approval-ml/
├── data/
│   └── AprobacionCurso2026.csv             # raw dataset
└── notebooks/
    └── data_preparation.ipynb              # ETL + data preparation pipeline
```

## How to run

Open the notebook in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blandevv/course-approval-ml/blob/main/notebooks/data_preparation.ipynb)

Then upload `AprobacionCurso2026.csv` to the Colab environment (or mount your Google Drive) and run
all cells in order. Dependencies (`ydata-profiling`, `imbalanced-learn`) are installed automatically
at the top of the notebook.

## Requirements

- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- scipy
- ydata-profiling
- imbalanced-learn

## Next steps

- Train a classification model (e.g. Logistic Regression, Random Forest) on the final dataset.
- Evaluate with metrics suited to imbalanced data.
