# Real-Time Fraud Detection & Risk Scoring System

A beginner-friendly ML project that predicts whether a transaction is fraudulent and exposes the model through a FastAPI endpoint.

## Dataset
`data/transactions.csv` contains 2,000 synthetic transactions for learning and development.

> Important: this dataset is synthetic and should not be used for real financial decisions.

## Setup
### Windows
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python src/train.py
```

## Test a prediction
```bash
python src/predict.py
```

## Run API
```bash
uvicorn api.main:app --reload
```

Open:
`http://127.0.0.1:8000/docs`

## Project flow
Transaction -> preprocessing/features -> ML model -> fraud probability -> risk decision.
