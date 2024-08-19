import pandas as pd
from .base import DateLike
from .database import PriceVolumeData, get_session


def persist_data(data: pd.DataFrame, ticker: str) -> None:
    session = get_session()
    try:
        for date, row in data.iterrows():
            price_volume = PriceVolumeData(
                date=date,
                ticker=ticker,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                adjOpen=row["adjOpen"],
                adjHigh=row["adjHigh"],
                adjLow=row["adjLow"],
                adjClose=row["adjClose"],
                adjVolume=row["adjVolume"],
            )
            session.add(price_volume)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def load_data_from_db(
    ticker: str, start_date: DateLike, end_date: DateLike
) -> pd.DataFrame:
    session = get_session()
    try:
        query = (
            session.query(PriceVolumeData)
            .filter(
                PriceVolumeData.ticker == ticker,
                PriceVolumeData.date >= start_date,
                PriceVolumeData.date <= end_date,
            )
            .order_by(PriceVolumeData.date)
        )

        data = pd.DataFrame(
            [
                {
                    "date": item.date,
                    "open": item.open,
                    "high": item.high,
                    "low": item.low,
                    "close": item.close,
                    "volume": item.volume,
                    "adjOpen": item.adjOpen,
                    "adjHigh": item.adjHigh,
                    "adjLow": item.adjLow,
                    "adjClose": item.adjClose,
                    "adjVolume": item.adjVolume,
                }
                for item in query
            ]
        )

        if data.empty:
            return data

        if "date" in data:
            data = data.set_index("date")
            data.index = pd.to_datetime(data.index).normalize()
        else:
            raise ValueError("DataFrame must have a 'date' column")

        return data
    finally:
        session.close()
