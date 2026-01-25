import pandas as pd
from datetime import date as date_type, datetime
from typing import Optional
from .base import DateLike
from .database import (
    PriceVolumeData,
    get_session,
    set_session_override,
    is_session_override,
)


def _to_date(value: DateLike) -> date_type:
    if isinstance(value, pd.Timestamp):
        return value.tz_localize(None).date() if value.tzinfo else value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date_type):
        return value
    return pd.Timestamp(value).date()


def persist_data(
    data: pd.DataFrame, ticker: str, *, session: Optional[object] = None
) -> None:
    if session is not None:
        set_session_override(session)
    session = session or get_session()
    close_on_exit = not is_session_override(session)
    try:
        for date, row in data.iterrows():
            date_val = _to_date(date)
            price_volume = PriceVolumeData(
                date=date_val,
                ticker=ticker,
                open=row.get("open"),
                high=row.get("high"),
                low=row.get("low"),
                close=row.get("close"),
                volume=row.get("volume"),
                adjOpen=row.get("adjOpen", row.get("open")),
                adjHigh=row.get("adjHigh", row.get("high")),
                adjLow=row.get("adjLow", row.get("low")),
                adjClose=row.get("adjClose", row.get("close")),
                adjVolume=row.get("adjVolume", row.get("volume")),
            )
            session.add(price_volume)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        if close_on_exit:
            session.close()


def load_data_from_db(
    ticker: str,
    start_date: DateLike,
    end_date: DateLike,
    *,
    session: Optional[object] = None
) -> pd.DataFrame:
    if session is not None:
        set_session_override(session)
    session = session or get_session()
    close_on_exit = not is_session_override(session)
    try:
        start_val = _to_date(start_date)
        end_val = _to_date(end_date)
        query = (
            session.query(PriceVolumeData)
            .filter(
                PriceVolumeData.ticker == ticker,
                PriceVolumeData.date >= start_val,
                PriceVolumeData.date <= end_val,
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
        if close_on_exit:
            session.close()
