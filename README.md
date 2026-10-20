# TerraVista AI

### Explainable housing-price estimation with a reproducible machine-learning pipeline

TerraVista AI is a production-style Flask web application that turns the classic Boston Housing dataset into an interactive, explainable regression experience. Users can submit the dataset's 13 structured property and neighbourhood features, receive a model-generated estimate, inspect held-out evaluation metrics, and review the model's coefficient-level context.

> **Important:** This project is an educational demonstration of supervised learning and web deployment. Its historical data must not be used as a source of current property valuations, lending decisions, eligibility decisions, or decisions about people and communities.

![TerraVista AI interface](Screenshot.png)

## Why this project stands out

- **End-to-end ML product:** data loading, model training, serialization, inference, evaluation, and presentation are connected in one runnable application.
- **Reproducible experimentation:** an explicit 80/20 train-test split with `random_state=42` makes the baseline evaluation repeatable.
- **Explainability built in:** the model-notes experience exposes MAE, RMSE, R², actual-versus-predicted values, and linear coefficients instead of presenting a prediction as an opaque answer.
- **Human-centred interface:** responsive editorial design, accessible focus states, inline validation, reduced-motion support, and a session-scoped reading history.
- **Deployment-ready structure:** Flask application entry point, declared runtime dependencies, Gunicorn support, and environment-based secret configuration.

## Core capabilities

| Capability | Implementation |
| --- | --- |
| Interactive estimation | A validated form for all 13 model features |
| Regression inference | Serialized `LinearRegression` model loaded from `model.pkl` |
| Model evaluation | MAE, RMSE, R², test-set chart data, and coefficient context |
| Recent readings | The five latest estimates stored in the user's Flask session |
| JSON access | `GET /api/insights` exposes evaluation and explainability data |
| Responsive UI | Server-rendered Jinja templates, semantic HTML, CSS, and vanilla JavaScript |
| Production serving | Gunicorn command included for Render-style deployments |

## System architecture

```text
boston.csv
    │
    ├── model.py ── train/test split ── LinearRegression ── model.pkl
    │
    └── app.py ── load model + evaluate baseline
                    │
                    ├── HTML views: /, /estimate, /model-notes, /recent
                    ├── POST /predict: validate → infer → store session reading
                    └── GET /api/insights: metrics + chart data + coefficients
```

### Machine-learning workflow

1. Load the 506-row dataset from `boston.csv`.
2. Separate the 13 predictors from the `medv` target.
3. Split the data into training and held-out test partitions using `test_size=0.2` and `random_state=42`.
4. Fit an ordinary least-squares `LinearRegression` estimator on the training partition.
5. Serialize the trained estimator to `model.pkl`.
6. Recreate the deterministic evaluation split in the Flask application and expose the resulting metrics.
7. Validate incoming values against the observed feature ranges before inference.

The target is expressed in thousands of 1970s-era dollars. The UI preserves that unit and formats estimates as `$X.XXk`.

## Technology stack

- **Backend:** Python, Flask, Gunicorn
- **Machine learning:** scikit-learn `LinearRegression`
- **Data processing:** pandas, NumPy
- **Frontend:** Jinja2, semantic HTML, CSS, vanilla JavaScript
- **Runtime artifacts:** CSV training data and serialized pickle model

## Project structure

```text
.
├── app.py                    # Flask routes, validation, inference, and insights API
├── model.py                  # Reproducible training and model serialization script
├── boston.csv                # Historical dataset
├── model.pkl                 # Serialized trained LinearRegression estimator
├── requirements.txt          # Runtime dependencies
├── templates/                # Jinja page and partial templates
└── static/                   # JavaScript and modular CSS assets
```

## Local development

### Prerequisites

- Python 3.10 or newer
- PowerShell on Windows, or an equivalent shell on macOS/Linux

### Install and run

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python model.py
python app.py
```

Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in a browser.

If PowerShell execution policy prevents activation, invoke the virtual-environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe model.py
.\.venv\Scripts\python.exe app.py
```

### Production-style serving (Unix/WSL)

```powershell
python -m gunicorn --bind 0.0.0.0:8000 app:app
```

Gunicorn is intended for Unix-like deployment environments. On native Windows, use `python app.py` for local development or deploy through a Linux-based service such as Render.

Set a real secret outside local development:

```powershell
$env:FLASK_SECRET_KEY = "replace-with-a-long-random-value"
```

The application falls back to a development key only when `FLASK_SECRET_KEY` is not supplied.

## Application routes

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | `GET` | Landing page and product overview |
| `/estimate` | `GET` | Prediction workspace |
| `/predict` | `POST` | Validate form data and generate an estimate |
| `/model-notes` | `GET` | Evaluation metrics and model context |
| `/recent` | `GET` | Session-scoped recent readings |
| `/clear-history` | `POST` | Clear the current session's reading history |
| `/api/insights` | `GET` | Return metrics, chart rows, and coefficients as JSON |

## Evaluation and interpretation

The model-notes page reports:

- **Mean Absolute Error (MAE):** average absolute prediction error in target units.
- **Root Mean Squared Error (RMSE):** error metric that weights larger residuals more heavily.
- **R²:** proportion of held-out target variance explained by the linear model.
- **Coefficient context:** the fitted linear contribution associated with each input feature.

These are baseline regression diagnostics, not accuracy percentages or guarantees of real-world performance. The model is intentionally simple so its behaviour can be inspected and discussed.

## Data ethics and limitations

The Boston Housing dataset is historical and has well-documented ethical limitations. In particular, its `B` feature encodes a problematic racial variable. TerraVista keeps the original feature set for reproducibility, labels the feature for careful interpretation, and does not present the application as a fair or contemporary valuation system.

The dataset also represents a historical market context rather than today's housing conditions. Predictions are educational estimates, not appraisals, investment advice, or market forecasts. A responsible production system would require contemporary data, documented provenance, bias assessment, uncertainty intervals, monitoring, access controls, and domain review.

## Deployment

This project can be deployed as a Python web service on platforms such as Render:

1. Push the project directory to a GitHub repository.
2. Create a web service connected to the repository.
3. Use the following commands:

   ```text
   Build command: pip install -r requirements.txt
   Start command: gunicorn app:app
   ```

4. Configure `FLASK_SECRET_KEY` with a long, randomly generated value.
5. Ensure `app.py`, `model.pkl`, `boston.csv`, `templates/`, and `static/` are included in the deployed service.

## Roadmap

- Add automated unit and integration tests for validation, inference, routes, and API output.
- Package preprocessing and the estimator as a versioned scikit-learn pipeline.
- Add uncertainty intervals and prediction monitoring.
- Introduce model and dataset version metadata.
- Replace the historical source with a contemporary, documented, ethically reviewed dataset.
- Add privacy-reviewed persistence only if saved explorations become a supported product feature.

## Author

[Monisha Krishnamoorthy](https://github.com/monishakrishnamoorthy7)
