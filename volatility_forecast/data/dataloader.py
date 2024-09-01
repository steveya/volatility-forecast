import os
import requests
import pandas as pd
from typing import List, Any, NoReturn, Mapping, Optional
from functools import lru_cache
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import pandas_market_calendars as mcal

from .base import DataLoader, DataField, DateLike
from .date_util import get_closest_next_business_day, get_closest_prev_business_day
from .persistence import persist_data, load_data_from_db


def get_tiingo_api():
    load_dotenv()
    return os.getenv("TIINGO_API")


class TiingoEoDDataLoader(DataLoader):
    def __init__(self, tickers: List[str]) -> NoReturn:
        self.tickers = tickers
        super().__init__()

    @staticmethod
    def convert_name(field_name: str) -> str:
        return f"adj{field_name.capitalize()}"

    def _get_data(
        self,
        fields: List[DataField],
        start_date: DateLike,
        end_date: DateLike,
    ) -> Mapping[DataField, pd.DataFrame]:
        data = {}
        data_cache = self._get_tingle_data(start_date, end_date)

        from .dataset import PriceVolume

        for field in fields:
            if field not in PriceVolume.fields:
                raise NotImplemented(f"Field {field} is not supported")

            field_name = self.convert_name(field.name)
            if field_name not in data_cache:
                raise ValueError(f"Field {field_name} is not in the yFinance downloads")

            data[field] = data_cache.loc[start_date:end_date, field_name]

        return data

    @lru_cache(maxsize=5)
    def _get_tingle_data(
        self, start_date: DateLike, end_date: DateLike
    ) -> pd.DataFrame:
        headers = {"Content-Type": "application/json"}

        def make_request(url):
            try:
                response = requests.get(url, headers=headers)
                return response.json()
            except requests.RequestException as e:
                return {"error": str(e)}

        def fetch_all(urls):
            results = {}
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_key = {
                    executor.submit(make_request, url): key for key, url in urls.items()
                }
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        data = future.result()
                        results[key] = data
                    except Exception as exc:
                        print(f"{key} generated an exception: {exc}")
            return results

        api_key = get_tiingo_api()
        sdate_str = start_date.strftime("%Y-%m-%d")
        edate_str = end_date.strftime("%Y-%m-%d")

        url_template = "https://api.tiingo.com/tiingo/daily/{}/prices?startDate={}&endDate={}&token={}"
        urls = {}
        for ticker in self.tickers:
            urls[ticker] = url_template.format(ticker, sdate_str, edate_str, api_key)

        all_data = fetch_all(urls)

        def process_response(response):
            df = pd.DataFrame(response).set_index("date")
            df.index = pd.to_datetime(df.index).normalize()

            return df

        combined_data = pd.concat(
            [process_response(value) for value in all_data.values()],
            axis=1,
            keys=all_data.keys(),
        )
        combined_data = combined_data.swaplevel(0, 1, axis=1).sort_index(axis=1)
        # combined_data.index = combined_data.index.tz_convert(None)
        return combined_data


class TiingoEodDataLoaderProd(DataLoader):
    def __init__(self, tickers: List[str]) -> NoReturn:
        self.tickers = tickers
        super().__init__()

    @staticmethod
    def convert_name(field_name):
        return f"adj{field_name.capitalize()}"

    def _get_data(
        self, fields: List[DataField], start_date: DateLike, end_date: DateLike
    ) -> Mapping[DataField, pd.DataFrame]:
        assert (
            get_closest_next_business_day(start_date, mcal.get_calendar("NYSE"))
            == start_date
        )
        assert (
            get_closest_prev_business_day(end_date, mcal.get_calendar("NYSE"))
            == end_date
        )

        data = {}
        data_cache = self._get_tingle_data(start_date, end_date)

        from .dataset import PriceVolume

        for field in fields:
            if field not in PriceVolume.fields:
                raise NotImplemented(f"Field {field} is not supported")

            field_name = self.convert_name(field.name)
            if field_name not in data_cache:
                raise ValueError(f"Field {field_name} is not in the Tiingo data")

            data[field] = data_cache.loc[start_date:end_date, field_name]

        return data

    # @lru_cache(maxsize=5)
    def _get_tingle_data(
        self, start_date: DateLike, end_date: DateLike
    ) -> pd.DataFrame:
        combined_data = {}

        for ticker in self.tickers:
            # Try to load data from the database first
            db_data = load_data_from_db(ticker, start_date, end_date)

            if db_data.empty:
                # If no data in database, fetch from Tiingo
                tiingo_data = self._fetch_tiingo_data(ticker, start_date, end_date)

                # Persist the newly fetched data
                persist_data(tiingo_data, ticker)

                ticker_data = tiingo_data
            else:
                # Check if we have all the data we need
                db_start = db_data.index.min()
                db_end = db_data.index.max()

                missing_ranges = []

                # Check for missing data at the start
                if db_start > start_date:
                    missing_ranges.append(
                        (
                            get_closest_next_business_day(
                                start_date, mcal.get_calendar("NYSE")
                            ),
                            get_closest_prev_business_day(
                                db_start, mcal.get_calendar("NYSE")
                            ),
                        )
                    )

                # Check for missing data at the end
                if db_end < end_date:
                    missing_ranges.append(
                        (
                            get_closest_next_business_day(
                                db_end, mcal.get_calendar("NYSE")
                            ),
                            get_closest_prev_business_day(
                                end_date, mcal.get_calendar("NYSE")
                            ),
                        )
                    )

                # Fetch missing data if any
                for miss_start, miss_end in missing_ranges:
                    missing_data = self._fetch_tiingo_data(ticker, miss_start, miss_end)
                    db_data = pd.concat([db_data, missing_data])
                    persist_data(missing_data, ticker)

                ticker_data = db_data

            # Add to combined data
            combined_data[ticker] = ticker_data

        combined_data = pd.concat(
            combined_data.values(), axis=1, keys=combined_data.keys()
        )
        combined_data = combined_data.swaplevel(0, 1, axis=1).sort_index(axis=1)

        return combined_data

    def _fetch_tiingo_data(
        self, ticker: str, start_date: DateLike, end_date: DateLike
    ) -> pd.DataFrame:
        headers = {"Content-Type": "application/json"}
        api_key = get_tiingo_api()
        sdate_str = start_date.strftime("%Y-%m-%d")
        edate_str = end_date.strftime("%Y-%m-%d")

        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={sdate_str}&endDate={edate_str}&token={api_key}"

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

        df = pd.DataFrame(data).set_index("date")
        df.index = pd.to_datetime(df.index)
        return df


class PriceVolumeDatabaseLoader(DataLoader):
    def __init__(self, tickers: List[str]) -> NoReturn:
        self.tickers = tickers
        super().__init__()

    def _get_data(
        self,
        fields: List[DataField],
        start_date: Optional[DateLike],
        end_date: Optional[DateLike],
    ) -> Mapping[DataField, pd.DataFrame]:
        data = {}

        from .dataset import PriceVolume

        for ticker in self.tickers:
            db_data = load_data_from_db(ticker, start_date, end_date)
            for field in fields:
                if field not in PriceVolume.fields:
                    raise NotImplemented(f"Field {field} is not supported")

                field_name = field.name.lower()
                if field_name not in db_data.columns:
                    raise ValueError(f"Field {field_name} is not in the database data")

                if field not in data:
                    data[field] = pd.DataFrame()
                data[field][ticker] = db_data[field_name]

        return data
