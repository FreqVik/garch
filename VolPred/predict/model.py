from sqlalchemy import Column, Integer, DateTime, Float
from datetime import datetime, timezone
from .db import Base


class ActualValue(Base):
    __tablename__ = 'actual_values'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    value = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)

    def __repr__(self):
        return f"<ActualValue(id={self.id}, date={self.date}, value={self.value}, volatility={self.volatility})>"


class PredictedValue(Base):
    __tablename__ = 'predicted_values'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    predicted_vol = Column(Float, nullable=False)
    predicted_variance = Column(Float, nullable=True)

    def __repr__(self):
        return f"<PredictedValue(id={self.id}, date={self.date}, predicted_vol={self.predicted_vol})>"