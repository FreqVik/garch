import httpx
import pandas as pd
from datetime import datetime, timedelta
from arch import arch_model
import os
import json
import joblib
from dotenv import load_dotenv

load_dotenv()

class CommData:

    def __init__(self, function: str, interval: str):
        self.function = function
        self.interval = interval
        self.api_key = os.getenv("API_KEY")
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_data(self, outputsize: str = "full") -> pd.DataFrame:
        """
        Fetches commodity data from Alpha Vantage API.
        Saves to a json file and then creates a dataframe from it.
        """
        params = {
            "function": self.function,
            "interval": self.interval,
            "outputsize": outputsize,
            "apikey": self.api_key
        }
        response = httpx.get(self.base_url, params=params)
        data = response.json()

        project_root = os.path.dirname(os.path.dirname(__file__))
        model_dir = os.path.join(project_root, "model")
        os.makedirs(model_dir, exist_ok=True)
        output_file = os.path.join(model_dir, f"{self.function}_{self.interval}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return self._json_to_df(data)

    def _json_to_df(self, data: dict) -> pd.DataFrame:
        """
        Normalizes supported API response shapes into a time-indexed dataframe.
        """
        if "data" in data and isinstance(data["data"], list):
            df = pd.DataFrame(data["data"])
            if "date" not in df.columns or "value" not in df.columns:
                raise ValueError("JSON 'data' list must contain 'date' and 'value' fields")
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            return df

        time_series_key = f"Time Series ({self.interval})"
        if time_series_key in data:
            df = pd.DataFrame(data[time_series_key]).T
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            return df

        raise ValueError("Unsupported JSON format: expected 'data' list or time-series structure")

    def read_json_to_df(self, json_path: str) -> pd.DataFrame:
        """
        Reads a saved JSON file from disk and returns a dataframe with a
        datetime index and a single value column.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "data" not in data or not isinstance(data["data"], list):
            raise ValueError("Expected a JSON file with a top-level 'data' list")

        df = pd.DataFrame(data["data"])
        if "date" not in df.columns or "value" not in df.columns:
            raise ValueError("JSON 'data' list must contain 'date' and 'value' fields")

        df = df[["date", "value"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df = df.set_index("date").sort_index()
        return df

    def _resolve_json_path(self, json_filename: str = None) -> str:
        """
        Resolves which JSON file to use from the model directory.
        If json_filename is not provided, picks the most recently modified .json file.
        """
        project_root = os.path.dirname(os.path.dirname(__file__))
        model_dir = os.path.join(project_root, "model")

        if json_filename:
            input_file = os.path.join(model_dir, json_filename)
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"No JSON data file found at {input_file}")
            return input_file

        json_files = [
            os.path.join(model_dir, file_name)
            for file_name in os.listdir(model_dir)
            if file_name.endswith(".json")
        ]
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {model_dir}")

        return max(json_files, key=os.path.getmtime)

    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Splits a time-ordered dataframe into train and test sets.
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        split_index = int(len(df) * (1 - test_size))
        if split_index <= 0 or split_index >= len(df):
            raise ValueError("test_size leaves no data for train or test")

        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()
        return train_df, test_df

    def createModel(self, test_size: float = 0.2, json_filename: str = None):
        """
        Creates a GARCH model
        """
        input_file = self._resolve_json_path(json_filename)

        df = self.read_json_to_df(input_file)
        train_df, _ = self.train_test_split(df, test_size=test_size)
        close = train_df["value"]

        returns = close.pct_change().dropna() * 100

        if returns.empty:
            raise ValueError("Not enough valid close-price data to fit GARCH model")

        model = arch_model(returns, vol="GARCH", p=1, q=1, dist="normal")
        fitted_model = model.fit(disp="off")

        project_root = os.path.dirname(os.path.dirname(__file__))
        model_dir = os.path.join(project_root, "model")
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, f"{self.function}_{self.interval}_garch.joblib")
        joblib.dump(fitted_model, model_file)

        return fitted_model

    def metrics(self, model, test_df: pd.DataFrame):
        """
        Evaluates the GARCH model on the test set and returns performance metrics.
        """
        if test_df.empty:
            raise ValueError("Test DataFrame is empty")

        close = test_df["value"]
        returns = close.pct_change().dropna() * 100

        if returns.empty:
            raise ValueError("Not enough valid close-price data in test set to evaluate model")

        # Rolling one-step-ahead variance forecasts over the test window.
        train_returns = pd.Series(model.model.y).dropna().reset_index(drop=True)
        test_returns = returns.reset_index(drop=True)

        predicted_variance = []
        history = train_returns.copy()
        for observed_return in test_returns:
            rolling_model = arch_model(history, vol="GARCH", p=1, q=1, dist="normal")
            rolling_fit = rolling_model.fit(disp="off")
            next_var = rolling_fit.forecast(horizon=1).variance.iloc[-1, 0]
            predicted_variance.append(next_var)
            history = pd.concat([history, pd.Series([observed_return])], ignore_index=True)

        actual_variance = (test_returns ** 2).to_numpy()
        pred_variance = pd.Series(predicted_variance).to_numpy()

        mse = float(((pred_variance - actual_variance) ** 2).mean())
        mae = float((pd.Series(pred_variance - actual_variance).abs()).mean())

        return {"MSE": mse, "MAE": mae}


if __name__ == "__main__":
    comm_data = CommData(function="BRENT", interval="daily")
    json_path = comm_data._resolve_json_path()
    df = comm_data.read_json_to_df(json_path)
    train_df, test_df = comm_data.train_test_split(df)
    model = comm_data.createModel()
    metrics = comm_data.metrics(model, test_df)
    print(metrics)
    print(model.summary())
    #print(comm_data.metrics(model, test_df))
    #model = comm_data.createModel()
    #print(model.summary())
