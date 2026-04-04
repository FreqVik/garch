"""Prediction service for managing database operations and predictions."""
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import json
from pathlib import Path
import joblib

from .model import ActualValue, PredictedValue
from .db import get_db, get_db_context


class PredictionService:
    """Service layer for prediction and data management."""
    
    def __init__(self):
        """Initialize the service."""
        pass

    def save_actual_value(self, date: datetime, value: float, volatility: float) -> ActualValue:
        """
        Save an actual value to the database.
        
        Args:
            date: Timestamp of the value
            value: BRENT price value
            volatility: Actual volatility
            
        Returns:
            ActualValue instance
        """
        with get_db_context() as db:
            actual = ActualValue(date=date, value=value, volatility=volatility)
            db.add(actual)
            db.flush()
            return actual

    def save_predicted_value(self, date: datetime, predicted_vol: float, 
                           predicted_variance: float) -> PredictedValue:
        """
        Save a predicted volatility to the database.
        
        Args:
            date: Prediction timestamp
            predicted_vol: Predicted volatility (standard deviation)
            predicted_variance: Predicted variance
            
        Returns:
            PredictedValue instance
        """
        with get_db_context() as db:
            predicted = PredictedValue(
                date=date,
                predicted_vol=predicted_vol,
                predicted_variance=predicted_variance
            )
            db.add(predicted)
            db.flush()
            return predicted

    def get_historical_data(self, days: int = 90) -> List[Dict]:
        """
        Get historical actual values for the last N days.
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            List of dictionaries with actual values
        """
        db = get_db()
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            values = db.query(ActualValue).filter(
                ActualValue.date >= cutoff_date
            ).order_by(ActualValue.date.asc()).all()
            
            return [
                {
                    "date": v.date.isoformat(),
                    "value": v.value,
                    "volatility": v.volatility
                }
                for v in values
            ]
        finally:
            db.close()

    def get_predictions(self, days: int = 7) -> List[Dict]:
        """
        Get recent predictions for the last N days.
        If no predictions exist in database, generate from trained model.
        
        Args:
            days: Number of days of predictions to retrieve
            
        Returns:
            List of dictionaries with predictions
        """
        db = get_db()
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            predictions = db.query(PredictedValue).filter(
                PredictedValue.date >= cutoff_date
            ).order_by(PredictedValue.date.asc()).all()
            
            # If predictions exist in database, return them
            if predictions:
                return [
                    {
                        "date": p.date.isoformat(),
                        "predicted_vol": p.predicted_vol,
                        "predicted_variance": p.predicted_variance
                    }
                    for p in predictions
                ]
            
            # If no predictions in database, generate from trained model
            return self._generate_predictions_from_model()
            
        finally:
            db.close()

    def _generate_predictions_from_model(self) -> List[Dict]:
        """
        Fallback: Generate predictions directly from trained GARCH model.
        Used when no predictions exist in database.
        
        Returns:
            List with single prediction from model, or empty if model not found
        """
        try:
            from utils.config import MODEL_JOBLIB_FILE
            
            if not MODEL_JOBLIB_FILE.exists():
                return []
            
            # Load the trained model
            model = joblib.load(MODEL_JOBLIB_FILE)
            
            # Generate 1-step-ahead forecast
            forecast = model.forecast(horizon=1)
            variance = forecast.variance.iloc[-1, 0]
            volatility = variance ** 0.5
            
            return [
                {
                    "date": datetime.now(timezone.utc).isoformat(),
                    "predicted_vol": float(volatility),
                    "predicted_variance": float(variance),
                    "source": "model_forecast"
                }
            ]
        except Exception as e:
            # Return empty list if model loading or forecasting fails
            return []

    def create_chart_data_json(self, days: int = 90) -> Dict:
        """
        Create JSON data for frontend chart (last N days of data + recent predictions).
        
        Args:
            days: Number of days of historical data to include
            
        Returns:
            Dictionary with historical and prediction data
        """
        actual_data = self.get_historical_data(days=days)
        predictions = self.get_predictions(days=7)
        
        chart_data = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "historical_days": days,
                "commodity": "BRENT",
                "currency": "USD"
            },
            "actual_values": actual_data,
            "predictions": predictions,
            "summary": {
                "total_data_points": len(actual_data),
                "total_predictions": len(predictions),
                "date_range": {
                    "start": actual_data[0]["date"] if actual_data else None,
                    "end": actual_data[-1]["date"] if actual_data else None
                }
            }
        }
        
        return chart_data

    def get_exchange_data_chart(self, days: int = 30) -> Dict:
        """
        Fetch fresh data from exchange (Alpha Vantage) and format for line chart plotting.
        
        Args:
            days: Number of days of data to fetch (30 = 1 month)
            
        Returns:
            Chart-ready dictionary with dates and price/volatility series
        """
        try:
            from data.market import CommData
            from utils.config import API_FUNCTION, API_INTERVAL
            
            # Fetch fresh data from Alpha Vantage
            comm_data = CommData(function=API_FUNCTION, interval=API_INTERVAL)
            df = comm_data.fetch_data(outputsize="full")
            
            # Limit to last N days
            df = df.tail(days).copy()
            
            if df.empty:
                return {"error": "No data available from exchange"}
            
            # Extract dates and prices
            dates = df.index.strftime("%Y-%m-%d").tolist()
            prices = df["value"].tolist()
            
            # Calculate volatility from daily returns
            returns = df["value"].pct_change().dropna() * 100
            volatility = (returns.rolling(window=5).std()).tolist()
            
            # Pad volatility array to match length
            while len(volatility) < len(prices):
                volatility.insert(0, None)
            
            chart_data = {
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "Alpha Vantage Exchange",
                    "commodity": "BRENT",
                    "interval": API_INTERVAL,
                    "currency": "USD",
                    "days": len(dates)
                },
                "chart": {
                    "dates": dates,
                    "series": [
                        {
                            "name": "BRENT Price (USD)",
                            "data": prices,
                            "type": "line",
                            "color": "#1f77b4"
                        },
                        {
                            "name": "Volatility (%)",
                            "data": volatility,
                            "type": "line",
                            "color": "#ff7f0e"
                        }
                    ]
                },
                "summary": {
                    "date_range": {
                        "start": dates[0] if dates else None,
                        "end": dates[-1] if dates else None
                    },
                    "price_range": {
                        "min": min(prices),
                        "max": max(prices),
                        "current": prices[-1] if prices else None
                    },
                    "volatility_range": {
                        "min": min([v for v in volatility if v is not None]) if volatility else None,
                        "max": max([v for v in volatility if v is not None]) if volatility else None
                    }
                }
            }
            
            return chart_data
            
        except Exception as e:
            return {"error": f"Failed to fetch exchange data: {str(e)}"}

    def save_chart_data_to_file(self, days: int = 90, output_dir: Optional[Path] = None) -> Path:
        """
        Save chart data to a JSON file.
        
        Args:
            days: Number of days of historical data to include
            output_dir: Directory to save the file (default: model/)
            
        Returns:
            Path to the saved file
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "model"
        
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "chart_data.json"
        
        chart_data = self.create_chart_data_json(days=days)
        
        with open(output_file, "w") as f:
            json.dump(chart_data, f, indent=2)
        
        return output_file

    def metrics(self, start_date: Optional[datetime] = None, 
                end_date: Optional[datetime] = None) -> Dict:
        """
        Calculate metrics (MSE, MAE, RMSE) between actual and predicted values.
        
        Args:
            start_date: Start date for metrics calculation
            end_date: End date for metrics calculation
            
        Returns:
            Dictionary with calculated metrics
        """
        db = get_db()
        try:
            query_actual = db.query(ActualValue)
            query_pred = db.query(PredictedValue)
            
            if start_date:
                query_actual = query_actual.filter(ActualValue.date >= start_date)
                query_pred = query_pred.filter(PredictedValue.date >= start_date)
            
            if end_date:
                query_actual = query_actual.filter(ActualValue.date <= end_date)
                query_pred = query_pred.filter(PredictedValue.date <= end_date)
            
            actuals = query_actual.order_by(ActualValue.date.asc()).all()
            predictions = query_pred.order_by(PredictedValue.date.asc()).all()
            
            if not actuals or not predictions:
                return {"error": "Insufficient data for metrics calculation"}
            
            # Match predictions to actual values by date
            actual_vols = [a.volatility for a in actuals]
            pred_vols = [p.predicted_vol for p in predictions]
            
            # Use minimum length for comparison
            min_len = min(len(actual_vols), len(pred_vols))
            actual_vols = actual_vols[:min_len]
            pred_vols = pred_vols[:min_len]
            
            # Calculate metrics
            import numpy as np
            actual_arr = np.array(actual_vols)
            pred_arr = np.array(pred_vols)
            
            errors = pred_arr - actual_arr
            mse = float((errors ** 2).mean())
            mae = float(np.abs(errors).mean())
            rmse = float(np.sqrt(mse))
            
            return {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "data_points": min_len,
                "period": {
                    "start": actuals[0].date.isoformat(),
                    "end": actuals[-1].date.isoformat()
                }
            }
        finally:
            db.close()