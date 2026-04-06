"""
This module contains the service layer
for the dashboard:
    1. reads from a json file containing values of brent crude oil volatility
    2. displays the values in a chart on the dashboard
    3. displays the predicted value for the next day on the dashboard
    4. updates the json file with new data every interval (e.g., daily) and displays the updated chart on the dashboard
    5. below the chart display the ongoing performance metrics of the model (e.g., MSE, MAE) based on the latest test set evaluation

"""

import logging
import os
from typing import Dict

