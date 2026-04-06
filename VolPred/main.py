"""FastAPI application with background task scheduler."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from pathlib import Path
import json

from utils.scheduler import init_scheduler, shutdown_scheduler, get_scheduler
from utils.logger import setup_logger
from predict.services import PredictionService
from utils.tasks import get_retraining_metadata
from dashboard.db import init_db as init_dashboard_db
from dashboard.routes import router as dashboard_router

logger = setup_logger("main")

# Create FastAPI app
app = FastAPI(
    title="BRENT Volatility Predictor",
    description="GARCH model-based BRENT crude oil volatility forecasting",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
prediction_service = PredictionService()

# Include dashboard routes
app.include_router(dashboard_router)


@app.on_event("startup")
async def startup_event():
    """Initialize database and scheduler on startup."""
    try:
        logger.info("Starting up application")
        
        # Initialize dashboard database (includes all tables)
        init_dashboard_db()
        logger.info("Dashboard database initialized")
        
        # Initialize and start scheduler
        init_scheduler()
        logger.info("Background task scheduler started")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown scheduler gracefully."""
    try:
        logger.info("Shutting down application")
        shutdown_scheduler()
        logger.info("Background task scheduler stopped")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}", exc_info=True)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "BRENT Volatility Predictor"
    }


@app.get("/api/v1/predictions/latest")
async def get_latest_prediction():
    """Get the latest volatility prediction."""
    try:
        predictions = prediction_service.get_predictions(days=1)
        
        if not predictions:
            raise HTTPException(status_code=404, detail="No predictions found")
        
        return {
            "status": "success",
            "data": predictions[-1] if predictions else None
        }
    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/predictions/history")
async def get_prediction_history(days: int = 7):
    """Get prediction history for the last N days."""
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
        
        predictions = prediction_service.get_predictions(days=days)
        
        return {
            "status": "success",
            "days": days,
            "count": len(predictions),
            "data": predictions
        }
    except Exception as e:
        logger.error(f"Error fetching prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/historical")
async def get_historical_data(days: int = 90):
    """Get historical BRENT price data for the last N days."""
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
        
        data = prediction_service.get_historical_data(days=days)
        
        return {
            "status": "success",
            "commodity": "BRENT",
            "days": days,
            "count": len(data),
            "data": data
        }
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/chart-data")
async def get_chart_data(days: int = 90):
    """
    Get combined chart data: historical prices + predictions.
    Used by frontend for displaying charts.
    
    Returns data for the last N days with recent predictions.
    """
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
        
        chart_data = prediction_service.create_chart_data_json(days=days)
        
        return {
            "status": "success",
            "data": chart_data
        }
    except Exception as e:
        logger.error(f"Error creating chart data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/exchange-data/chart")
async def get_exchange_chart(days: int = 30):
    """
    Get fresh exchange data (Alpha Vantage) formatted for line chart plotting.
    Includes BRENT prices and calculated volatility for the last N days.
    
    Default: 30 days (1 month)
    
    Returns:
        Chart-ready JSON with dates and data series for line plotting
    """
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
        
        chart_data = prediction_service.get_exchange_data_chart(days=days)
        
        if "error" in chart_data:
            raise HTTPException(status_code=500, detail=chart_data["error"])
        
        return {
            "status": "success",
            "data": chart_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching exchange chart: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/chart-data/download")
async def download_chart_data(days: int = 90):
    """Download chart data as JSON file."""
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
        
        chart_file = prediction_service.save_chart_data_to_file(days=days)
        
        if not chart_file.exists():
            raise HTTPException(status_code=404, detail="Chart data file not found")
        
        return FileResponse(
            path=chart_file,
            filename="chart_data.json",
            media_type="application/json"
        )
    except Exception as e:
        logger.error(f"Error downloading chart data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics")
async def get_metrics(days: int = 30):
    """
    Get model performance metrics (MSE, MAE, RMSE) for the last N days.
    """
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
        
        from datetime import timedelta
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        metrics = prediction_service.metrics(start_date=start_date, end_date=end_date)
        
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scheduler/status")
async def get_scheduler_status():
    """Get current scheduler status and job information."""
    try:
        scheduler = get_scheduler()
        jobs = scheduler.get_jobs()
        
        job_status = [
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            }
            for job in jobs
        ]
        
        return {
            "status": "success",
            "scheduler_running": scheduler.scheduler.running,
            "jobs": job_status
        }
    except Exception as e:
        logger.error(f"Error fetching scheduler status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model/metadata")
async def get_model_metadata():
    """Get metadata about the last model retraining."""
    try:
        metadata = get_retraining_metadata()
        
        if not metadata:
            raise HTTPException(status_code=404, detail="No metadata found")
        
        return {
            "status": "success",
            "metadata": metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting BRENT Volatility Predictor API server")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
