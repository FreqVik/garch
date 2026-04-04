"""Background task functions for data fetching, model training, and predictions."""
import os
import json
from datetime import datetime, timezone
import joblib
import pandas as pd
from arch import arch_model

from utils.config import (
    MODEL_JSON_FILE,
    MODEL_JOBLIB_FILE,
    METADATA_FILE,
    API_FUNCTION,
    API_INTERVAL,
    API_OUTPUTSIZE,
    GARCH_P,
    GARCH_Q,
    TRAIN_TEST_SPLIT,
)
from utils.logger import setup_logger
from data.market import CommData
from predict.services import PredictionService

logger = setup_logger("background_tasks")


def daily_predict_task() -> dict:
    """
    Daily prediction task: Load trained model and forecast next-day volatility.
    Saves prediction to database and updates chart data.
    
    Returns:
        Dictionary with forecast results or error status
    """
    try:
        logger.info("Starting daily prediction task")
        
        # Check if model exists
        if not MODEL_JOBLIB_FILE.exists():
            logger.warning(f"Model file not found at {MODEL_JOBLIB_FILE}. Skipping prediction.")
            return {"status": "skipped", "reason": "model_not_found"}
        
        # Load trained model
        model = joblib.load(MODEL_JOBLIB_FILE)
        logger.info("Loaded trained GARCH model")
        
        # Forecast next-day volatility (1-step ahead)
        forecast = model.forecast(horizon=1)
        next_day_variance = forecast.variance.iloc[-1, 0]
        next_day_volatility = next_day_variance ** 0.5
        
        # Save prediction to database
        service = PredictionService()
        prediction = service.save_predicted_value(
            date=datetime.now(timezone.utc),
            predicted_vol=float(next_day_volatility),
            predicted_variance=float(next_day_variance)
        )
        logger.info(f"Saved prediction to database: id={prediction.id}")
        
        # Update chart data JSON file
        chart_file = service.save_chart_data_to_file(days=90)
        logger.info(f"Updated chart data file: {chart_file}")
        
        result = {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "next_day_volatility_forecast": float(next_day_volatility),
            "next_day_variance_forecast": float(next_day_variance),
            "saved_to_db": True,
            "chart_data_updated": True
        }
        
        logger.info(f"Daily prediction completed: volatility = {next_day_volatility:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Daily prediction task failed: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}


def retrain_task_10d() -> dict:
    """
    Retraining task: Delete stale JSON, fetch fresh data, retrain GARCH model.
    
    Returns:
        Dictionary with retraining results or error status
    """
    try:
        logger.info("Starting 10-day retraining task")
        
        # Step 1: Delete old JSON file
        if MODEL_JSON_FILE.exists():
            os.remove(MODEL_JSON_FILE)
            logger.info(f"Deleted stale JSON file: {MODEL_JSON_FILE}")
        
        # Step 2: Fetch fresh data from API
        comm_data = CommData(function=API_FUNCTION, interval=API_INTERVAL)
        logger.info(f"Fetching fresh data from Alpha Vantage ({API_FUNCTION}, {API_INTERVAL})")
        
        df = comm_data.fetch_data(outputsize=API_OUTPUTSIZE)
        logger.info(f"Fetched {len(df)} data points")
        
        if df.empty:
            raise ValueError("Fetched dataset is empty")
        
        # Step 3: Retrain model
        logger.info("Retraining GARCH model")
        train_df, test_df = comm_data.train_test_split(df, test_size=TRAIN_TEST_SPLIT)
        
        close = train_df["value"]
        returns = close.pct_change().dropna() * 100
        
        if returns.empty:
            raise ValueError("Not enough valid data to fit GARCH model")
        
        model = arch_model(
            returns,
            vol="GARCH",
            p=GARCH_P,
            q=GARCH_Q,
            dist="normal"
        )
        fitted_model = model.fit(disp="off")
        logger.info("GARCH model fitted successfully")
        
        # Step 4: Evaluate on test set
        metrics = comm_data.metrics(fitted_model, test_df)
        logger.info(f"Model metrics - MSE: {metrics['MSE']:.6f}, MAE: {metrics['MAE']:.6f}")
        
        # Step 5: Save model
        joblib.dump(fitted_model, MODEL_JOBLIB_FILE)
        logger.info(f"Saved trained model to {MODEL_JOBLIB_FILE}")
        
        # Step 6: Save metadata
        metadata = {
            "last_retrain_timestamp": datetime.utcnow().isoformat(),
            "data_points_used": len(df),
            "training_samples": len(train_df),
            "test_samples": len(test_df),
            "metrics": metrics,
            "garch_config": {"p": GARCH_P, "q": GARCH_Q},
            "test_split_ratio": TRAIN_TEST_SPLIT,
        }
        
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {METADATA_FILE}")
        
        result = {
            "status": "success",
            "timestamp": metadata["last_retrain_timestamp"],
            "data_points": len(df),
            "metrics": metrics,
        }
        
        logger.info("10-day retraining task completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Retraining task failed: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}


def get_retraining_metadata() -> dict:
    """
    Load retraining metadata from disk.
    
    Returns:
        Dictionary with metadata or empty dict if not found
    """
    try:
        if METADATA_FILE.exists():
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load metadata: {str(e)}")
    
    return {}
