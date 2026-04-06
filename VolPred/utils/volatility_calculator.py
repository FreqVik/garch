"""
Utility to calculate volatility from price data and generate frontend-ready JSON.
"""
import json
import os
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path


class VolatilityCalculator:
    """Calculate rolling volatility from price data and export to JSON."""
    
    def __init__(self, function: str = "BRENT", interval: str = "daily"):
        self.function = function
        self.interval = interval
        self.project_root = Path(__file__).parent.parent
        self.model_dir = self.project_root / "model"
    
    def read_price_json(self, json_path: str) -> pd.DataFrame:
        """
        Read price data from JSON file and convert to DataFrame.
        
        Args:
            json_path: Path to the JSON file with price data
            
        Returns:
            DataFrame with datetime index and 'value' column (prices as float)
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "data" not in data or not isinstance(data["data"], list):
            raise ValueError("Expected JSON format with 'data' list")
        
        df = pd.DataFrame(data["data"])
        if "date" not in df.columns or "value" not in df.columns:
            raise ValueError("Expected 'date' and 'value' columns in data")
        
        # Convert to proper types
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        # Remove rows with missing or invalid data
        df = df.dropna(subset=["date", "value"])
        df = df.set_index("date").sort_index()
        
        return df
    
    def calculate_rolling_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling volatility of returns.
        
        Args:
            df: DataFrame with 'value' column (prices)
            window: Rolling window size in days (default 20 days ≈ 1 month)
            
        Returns:
            DataFrame with 'volatility' column
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Calculate returns (percentage change)
        returns = df["value"].pct_change() * 100  # Convert to percentage
        
        # Calculate rolling standard deviation
        volatility = returns.rolling(window=window).std()
        
        result = df[["value"]].copy()
        result["volatility"] = volatility
        
        return result
    
    def generate_volatility_json(
        self,
        price_json_path: str,
        output_json_path: str = None,
        window: int = 20
    ) -> str:
        """
        Generate volatility JSON file from price data.
        
        Args:
            price_json_path: Path to input price JSON file
            output_json_path: Path to output volatility JSON (default: model/volatility.json)
            window: Rolling window for volatility calculation
            
        Returns:
            Path to the generated JSON file
        """
        if output_json_path is None:
            output_json_path = str(self.model_dir / "volatility.json")
        
        # Read price data
        price_df = self.read_price_json(price_json_path)
        
        # Calculate volatility
        vol_df = self.calculate_rolling_volatility(price_df, window=window)
        
        # Reset index to get date as a column
        vol_df = vol_df.reset_index()
        vol_df.columns = ["date", "price", "volatility"]
        
        # Format for JSON output
        data_list = []
        for idx, row in vol_df.iterrows():
            data_list.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "price": float(row["price"]),
                "volatility": float(row["volatility"]) if pd.notna(row["volatility"]) else None
            })
        
        # Create output JSON structure
        output_data = {
            "name": f"{self.function} Volatility",
            "interval": self.interval,
            "unit": "percentage",
            "volatility_type": "rolling_standard_deviation",
            "window_days": window,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": data_list
        }
        
        # Write to file
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        
        return output_json_path
    
    def get_latest_price_json(self) -> str:
        """Find the most recently modified price JSON file in model directory."""
        json_files = [
            os.path.join(self.model_dir, f)
            for f in os.listdir(self.model_dir)
            if f.endswith(".json") and "volatility" not in f
        ]
        
        if not json_files:
            raise FileNotFoundError(f"No price JSON files found in {self.model_dir}")
        
        return max(json_files, key=os.path.getmtime)


if __name__ == "__main__":
    import sys
    
    calculator = VolatilityCalculator(function="BRENT", interval="daily")
    
    # Use provided path or find latest
    if len(sys.argv) > 1:
        price_json_path = sys.argv[1]
    else:
        price_json_path = calculator.get_latest_price_json()
        print(f"Using price data from: {price_json_path}")
    
    # Generate volatility JSON
    output_path = calculator.generate_volatility_json(price_json_path)
    print(f"✓ Volatility JSON generated: {output_path}")
    
    # Print summary stats
    vol_df = calculator.read_price_json(price_json_path)
    vol_df = calculator.calculate_rolling_volatility(vol_df, window=20)
    
    print(f"\nPrice data summary:")
    print(f"  Date range: {vol_df.index.min().date()} to {vol_df.index.max().date()}")
    print(f"  Total records: {len(vol_df)}")
    print(f"  Price range: ${vol_df['value'].min():.2f} - ${vol_df['value'].max():.2f}")
    print(f"\nVolatility summary:")
    print(f"  Current volatility: {vol_df['volatility'].iloc[-1]:.2f}%")
    print(f"  Average volatility: {vol_df['volatility'].mean():.2f}%")
    print(f"  Min volatility: {vol_df['volatility'].min():.2f}%")
    print(f"  Max volatility: {vol_df['volatility'].max():.2f}%")
