import numpy as np
import pandas as pd
from .base import ensure_timestamp
from .dataset import PriceVolume
from .dataloader import TiingoEoDDataLoader
import pandas_market_calendars as mcal


def get_closest_next_business_day(date, calendar):
    custom_bday = pd.offsets.CustomBusinessDay(calendar=calendar)
    return date - custom_bday + custom_bday


def get_closest_prev_business_day(date, calendar):
    custom_bday = pd.offsets.CustomBusinessDay(calendar=calendar)
    return date + custom_bday - custom_bday


class ReturnDataManager:
    def get_data(
        self, universe, start_date, end_date, calendar=mcal.get_calendar("NYSE")
    ):
        custom_bday = pd.offsets.CustomBusinessDay(calendar=calendar)
        offset_start_date = (
            get_closest_next_business_day(
                ensure_timestamp(start_date), calendar=calendar
            )
            - custom_bday
        )
        offset_end_date = get_closest_prev_business_day(
            ensure_timestamp(end_date), calendar=calendar
        )
        adj_price = PriceVolume.CLOSE.get_data(
            TiingoEoDDataLoader(universe), offset_start_date, offset_end_date
        )
        data = adj_price.to_numpy()
        return np.diff(np.log(data), axis=0)


class OffsetReturnDataManager:
    def __init__(self, lag):
        self.lag = lag
        super().__init__()

    def get_data(
        self, universe, start_date, end_date, calendar=mcal.get_calendar("NYSE")
    ):
        custom_bday = pd.offsets.CustomBusinessDay(calendar=calendar)
        offset_start_date = (
            get_closest_next_business_day(
                ensure_timestamp(start_date), calendar=calendar
            )
            - custom_bday * self.lag
        )
        offset_end_date = (
            get_closest_prev_business_day(ensure_timestamp(end_date), calendar=calendar)
            - custom_bday * self.lag
        )
        data = ReturnDataManager().get_data(
            universe, offset_start_date, offset_end_date, calendar
        )
        return data


class LagReturnDataManager:
    def __init__(self, lag=1):
        self.lag = lag
        super().__init__()

    def get_data(
        self, universe, start_date, end_date, calendar=mcal.get_calendar("NYSE")
    ):
        return OffsetReturnDataManager(lag=self.lag).get_data(
            universe, start_date, end_date, calendar
        )

    def __repr__(self) -> str:
        return f"LagReturnDataManager(lag={self.lag})"


class LagAbsReturnDataManager:
    def __init__(self, lag=1):
        self.lag = lag
        super().__init__()

    def get_data(
        self, universe, start_date, end_date, calendar=mcal.get_calendar("NYSE")
    ):
        data = OffsetReturnDataManager(lag=self.lag).get_data(
            universe, start_date, end_date, calendar
        )
        return np.abs(data)

    def __repr__(self) -> str:
        return f"LagAbsReturnDataManager(lag={self.lag})"


class LagSquareReturnDataManager:
    def __init__(self, lag=1):
        self.lag = lag
        super().__init__()

    def get_data(
        self, universe, start_date, end_date, calendar=mcal.get_calendar("NYSE")
    ):
        data = OffsetReturnDataManager(lag=self.lag).get_data(
            universe, start_date, end_date, calendar
        )
        return data**2

    def __repr__(self) -> str:
        return f"LagSquareReturnDataManager(lag={self.lag})"


class SquareReturnDataManager:
    def get_data(
        self, universe, start_date, end_date, calendar=mcal.get_calendar("NYSE")
    ):
        data = OffsetReturnDataManager(lag=0).get_data(
            universe, start_date, end_date, calendar
        )
        return data**2

    def __repr__(self) -> str:
        return "SquareReturnDataManager()"
