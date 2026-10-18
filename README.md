# Telco Customer Churn - Logistic Regression + FastAPI

Predicts whether a telecom customer will churn, explains the drivers of churn using odds ratios, and serves the model through a FastAPI service.

## Files

| File                 | Role                                                                                          |
| -------------------- | --------------------------------------------------------------------------------------------- |
| `model/all_names.py` | column groups (binary / one-hot / numeric) and the allowed values of every categorical column |
| `model/parser.py`    | loads the csv, fixes the blank `TotalCharges` quirk, gives the class distribution             |
| `model/models.py`    | encoding + scaling, `LogisticRegression(class_weight='balanced')`, 80/20 split, saving        |
| `model/evaluate.py`  | classification report, confusion matrix and ROC plots, odds ratio interpretation              |
| `model/train.py`     | runs everything end to end, saves the model and the figures to `model_output/`                |
| `api.py`             | FastAPI service: `POST /predict`, `GET /health`                                               |

The csv stays in the project root. `model_output/` is created by training and holds `churn_model.pkl`, `confusion_matrix.png` and `roc_curve.png`.

## Run

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt   # or: python -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate

python -m model.train                        # train, evaluate, save the model (run from the project root)
uvicorn api:app --host 0.0.0.0 --port 8000   # serve it; docs at http://localhost:8000/docs
```

Or with make + docker:

```bash
make install # create the venv and install the dependencies
make train   # train, save the model and figures
make build   # retrain, then build the docker image telco-churn:latest
make run     # serve the image on http://localhost:8080
```

The image only serves the model: it has `api.py`, `model/all_names.py`, `model/models.py`, `model_output/churn_model.pkl` and the packages in `requirements-api.txt`, but not the csv or the plotting libraries.

## Example request

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
  "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
  "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.5, "TotalCharges": 190.0
}'
# -> {"will_churn": true, "churn_probability": 0.8955}
```

## Data set

Data can be downloaded from kaggle from this link: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/code).

After downloading, update file name in the code base.
