import os
import requests
import pandas as pd
from typing import List, Any
from functools import lru_cache
from dotenv import load_dotenv

from .base import DataLoader, DataField
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_tiingo_api():
    load_dotenv()
    return os.getenv("TIINGO_API")


class TiingoEoDDataLoader(DataLoader):
    def __init__(self, tickers: List[str]) -> None:
        self.tickers = tickers
        super().__init__()

    @staticmethod
    def convert_name(field_name):
        return f"adj{field_name.capitalize()}"

    def _get_data(
        self, fields: List[DataField], start_date: Any | None, end_date: Any | None
    ) -> Any:
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
    def _get_tingle_data(self, start_date, end_date):
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
        combined_data.index = combined_data.index.tz_convert(None)
        return combined_data
