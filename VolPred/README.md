"""
VolPred - BRENT Volatility Prediction System
Quick Start & Architecture Guide
"""

# ==============================================================================
# FIXED ISSUES
# ==============================================================================

ISSUE RESOLVED:
- Error: "IntervalTrigger.__init__() got an unexpected keyword argument 'hour'"
- Root Cause: APScheduler's IntervalTrigger doesn't support hour parameter
- Solution: Use IntervalTrigger(days=10) for repeating 10-day intervals

NEW FILES CREATED:
✅ predict/__init__.py          (Python package init)
✅ data/__init__.py             (Python package init)

UPDATED FILES:
✅ utils/scheduler.py           (Fixed IntervalTrigger)
✅ requirements.txt             (Added numpy, fixed python-dotenv)


# ==============================================================================
# PROJECT ARCHITECTURE
# ==============================================================================

FastAPI Application (main.py)
    ├── Startup Event: Initialize DB + Scheduler
    ├── 9 REST Endpoints (health, predictions, metrics, chart data)
    └── Shutdown Event: Graceful scheduler shutdown

Background Scheduler (utils/scheduler.py)
    ├── Job 1: DAILY @ 18:00 UTC (daily_predict_task)
    │   ├── Load trained GARCH model
    │   ├── Forecast next-day volatility
    │   ├── Save prediction to SQLite DB
    │   └── Update chart_data.json file
    │
    └── Job 2: EVERY 10 DAYS (retrain_task_10d)
        ├── Delete stale JSON
        ├── Fetch fresh BRENT data from Alpha Vantage
        ├── Retrain GARCH model
        ├── Save model + metadata
        └── Evaluate metrics (MSE, MAE, RMSE)

Database (predict/db.py + model.py)
    └── SQLite @ model/volatility.db
        ├── actual_values    → BRENT prices + volatility
        └── predicted_values → Model forecasts

Services (predict/services.py)
    ├── save_actual_value()
    ├── save_predicted_value()
    ├── get_historical_data(days)
    ├── create_chart_data_json()     ← Frontend chart ready
    ├── save_chart_data_to_file()
    └── metrics()                    ← Performance metrics


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

1. GET /health
   └─ Health check

2. GET /api/v1/predictions/latest
   └─ Last prediction

3. GET /api/v1/predictions/history?days=7
   └─ Historical predictions

4. GET /api/v1/data/historical?days=90
   └─ Historical BRENT prices (90 days default)

5. GET /api/v1/chart-data?days=90
   └─ ⭐ FRONTEND CHART DATA (prices + predictions as JSON)

6. GET /api/v1/chart-data/download?days=90
   └─ Download chart JSON file

7. GET /api/v1/metrics?days=30
   └─ Model performance (MSE, MAE, RMSE)

8. GET /api/v1/scheduler/status
   └─ Job statuses and next run times

9. GET /api/v1/model/metadata
   └─ Last retraining info


# ==============================================================================
# RUNNING THE APPLICATION
# ==============================================================================

1. Install dependencies:
   pip install -r requirements.txt

2. Ensure API key is set:
   export API_KEY="your_alphavantage_key"
   # OR add to .env file: API_KEY=your_key

3. Run the API server:
   python main.py
   # OR
   uvicorn main:app --reload

4. Access the API:
   - API: http://localhost:8000/api/v1/...
   - Docs: http://localhost:8000/docs (Swagger UI)
   - ReDoc: http://localhost:8000/redoc

5. Get chart data for frontend:
   curl http://localhost:8000/api/v1/chart-data?days=90 | python -m json.tool


# ==============================================================================
# DIRECTORY STRUCTURE
# ==============================================================================

VolPred/
├── main.py                   ← FastAPI application
├── requirements.txt
├── .env                      ← API_KEY here
│
├── data/
│   ├── __init__.py          ✅ NEW
│   └── market.py            ← CommData, Alpha Vantage client
│
├── predict/
│   ├── __init__.py          ✅ NEW
│   ├── db.py                ← SQLAlchemy setup
│   ├── model.py             ← ORM models
│   └── services.py          ← Business logic
│
├── utils/
│   ├── __init__.py
│   ├── config.py            ← Configuration
│   ├── logger.py            ← Logging setup
│   ├── scheduler.py         ← APScheduler ✅ FIXED
│   └── tasks.py             ← Job functions
│
├── model/                   ← Generated files
│   ├── BRENT_daily.json
│   ├── BRENT_daily_garch.joblib
│   ├── chart_data.json      ← Frontend chart
│   ├── retraining_metadata.json
│   └── volatility.db        ← SQLite database
│
└── logs/
    └── tasks.log            ← Application logs


# ==============================================================================
# FRONTEND INTEGRATION
# ==============================================================================

Chart data available as JSON at: /api/v1/chart-data?days=90

Response structure:
{
  "status": "success",
  "data": {
    "metadata": {
      "generated_at": "2026-04-04T19:00:00Z",
      "historical_days": 90,
      "commodity": "BRENT"
    },
    "actual_values": [
      {"date": "2026-01-06T...", "value": 82.5, "volatility": 1.23},
      ...
    ],
    "predictions": [
      {"date": "2026-04-04T...", "predicted_vol": 1.45, "predicted_variance": 2.1},
      ...
    ],
    "summary": {
      "total_data_points": 90,
      "total_predictions": 7,
      "date_range": {...}
    }
  }
}

Use with Chart.js, D3, or any charting library!


# ==============================================================================
# TROUBLESHOOTING
# ==============================================================================

Issue: "No module named 'predict'"
Fix: Ensure predict/__init__.py exists ✅ (created)

Issue: "No module named 'data'"
Fix: Ensure data/__init__.py exists ✅ (created)

Issue: "API_KEY not found"
Fix: Add API_KEY to .env file or export as environment variable

Issue: "Database locked"
Fix: Normal SQLite behavior, retries happen automatically

Issue: "Scheduler not running"
Fix: Check logs in logs/tasks.log for errors

