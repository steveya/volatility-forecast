import torch
import pandas as pd
from .base import ensure_timestamp
from .dataset import PriceVolume
from .dataloader import TingleEoDDataLoader
import pandas_market_calendars as mcal


class ReturnDataManager:
    def get_data(
        self, universe, start_date, end_date, calendar=mcal.get_calendar("NYSE")
    ):
        offset_start_date = ensure_timestamp(start_date) - calendar.holidays()
        adj_price = PriceVolume.CLOSE.get_data(
            TingleEoDDataLoader(universe), offset_start_date, end_date
        )
        tensor = torch.tensor(adj_price.to_numpy())
        return torch.diff(torch.log(tensor), axis=0)


class OffsetReturnDataManager:
    def __init__(self, lag=1):
        self.lag = lag
        super().__init__()

    def get_data(
        self, universe, start_date, end_date, calendar=mcal.get_calendar("NYSE")
    ):
        offset_start_date = ensure_timestamp(start_date) - calendar.holidays()
        offset_end_date = ensure_timestamp(end_date) - calendar.holidays()
        tensor = ReturnDataManager().get_data(
            universe, offset_start_date, offset_end_date, calendar
        )
        return tensor


class LagAbsReturnDataManager:
    def __init__(self, lag=1):
        self.lag = lag
        super().__init__()

    def get_data(
        self, universe, start_date, end_date, calendar=mcal.get_calendar("NYSE")
    ):
        tensor = OffsetReturnDataManager(lag=self.lag).get_data(
            universe, start_date, end_date, calendar
        )
        return torch.abs(tensor)


class LagSquareReturnDataManager:
    def __init__(self, lag=1):
        self.lag = lag
        super().__init__()

    def get_data(
        self, universe, start_date, end_date, calendar=mcal.get_calendar("NYSE")
    ):
        tensor = OffsetReturnDataManager(lag=self.lag).get_data(
            universe, start_date, end_date, calendar
        )
        return tensor**2


class OneStepAheadSquareReturnDataManager:
    def get_data(
        self, universe, start_date, end_date, calendar=mcal.get_calendar("NYSE")
    ):
        tensor = LagSquareReturnDataManager(lag=-1).get_data(
            universe, start_date, end_date, calendar
        )
        return tensor
