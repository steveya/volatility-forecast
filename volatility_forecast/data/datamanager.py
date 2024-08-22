import numpy as np
import pandas as pd
from typing import Type, NoReturn, List
from .base import ensure_timestamp, DateLike
from .dataset import PriceVolume
from .dataloader import TiingoEoDDataLoader
from .date_util import get_closest_next_business_day, get_closest_prev_business_day
import pandas_market_calendars as mcal


class ReturnDataManager:
    def __init__(self, data_loader_type: Type = TiingoEoDDataLoader) -> NoReturn:
        self.data_loader_type = data_loader_type

    def get_data(
        self,
        universe: List[str],
        start_date: DateLike,
        end_date: DateLike,
        calendar=mcal.get_calendar("NYSE"),
    ) -> pd.DataFrame:
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
            self.data_loader_type(universe), offset_start_date, offset_end_date
        )
        data = adj_price.to_numpy()
        date = adj_price.index
        return np.diff(np.log(data), axis=0), date


class OffsetReturnDataManager:
    def __init__(self, lag, data_loader_type: Type = TiingoEoDDataLoader) -> NoReturn:
        self.lag = lag
        self.data_loader_type = data_loader_type
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
        data, date = ReturnDataManager(data_loader_type=self.data_loader_type).get_data(
            universe, offset_start_date, offset_end_date, calendar
        )
        return data, date


class LagReturnDataManager:
    def __init__(
        self, lag: int = 1, data_loader_type: Type = TiingoEoDDataLoader
    ) -> NoReturn:
        self.lag = lag
        self.data_loader_type = data_loader_type
        super().__init__()

    def get_data(
        self,
        universe: List[str],
        start_date: DateLike,
        end_date: DateLike,
        calendar=mcal.get_calendar("NYSE"),
    ) -> pd.DataFrame:
        return OffsetReturnDataManager(
            lag=self.lag, data_loader_type=self.data_loader_type
        ).get_data(universe, start_date, end_date, calendar)

    def __repr__(self) -> str:
        return f"LagReturnDataManager(lag={self.lag})"


class LagAbsReturnDataManager:
    def __init__(
        self, lag: int = 1, data_loader_type: Type = TiingoEoDDataLoader
    ) -> NoReturn:
        self.lag = lag
        self.data_loader_type = data_loader_type
        super().__init__()

    def get_data(
        self,
        universe: List[str],
        start_date: DateLike,
        end_date: DateLike,
        calendar=mcal.get_calendar("NYSE"),
    ) -> pd.DataFrame:
        data, date = OffsetReturnDataManager(
            lag=self.lag, data_loader_type=self.data_loader_type
        ).get_data(universe, start_date, end_date, calendar)
        return np.abs(data), date

    def __repr__(self) -> str:
        return f"LagAbsReturnDataManager(lag={self.lag})"


class LagSquareReturnDataManager:
    def __init__(
        self, lag: int = 1, data_loader_type: Type = TiingoEoDDataLoader
    ) -> NoReturn:
        self.lag = lag
        self.data_loader_type = data_loader_type
        super().__init__()

    def get_data(
        self,
        universe: List[str],
        start_date: DateLike,
        end_date: DateLike,
        calendar=mcal.get_calendar("NYSE"),
    ) -> pd.DataFrame:
        data, date = OffsetReturnDataManager(
            lag=self.lag, data_loader_type=self.data_loader_type
        ).get_data(universe, start_date, end_date, calendar)
        return data**2, date

    def __repr__(self) -> str:
        return f"LagSquareReturnDataManager(lag={self.lag})"


class SquareReturnDataManager:
    def __init__(self, data_loader_type: Type = TiingoEoDDataLoader) -> NoReturn:
        self.data_loader_type = data_loader_type

    def get_data(
        self,
        universe: List[str],
        start_date: DateLike,
        end_date: DateLike,
        calendar=mcal.get_calendar("NYSE"),
    ) -> pd.DataFrame:
        data, date = OffsetReturnDataManager(
            lag=0, data_loader_type=self.data_loader_type
        ).get_data(universe, start_date, end_date, calendar)
        return data**2, date

    def __repr__(self) -> str:
        return "SquareReturnDataManager()"
