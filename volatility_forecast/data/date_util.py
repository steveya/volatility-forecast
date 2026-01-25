import pandas as pd


def get_closest_next_business_day(
    date: pd.Timestamp, calendar: pd.tseries.holiday.AbstractHolidayCalendar
) -> pd.Timestamp:
    custom_bday = pd.offsets.CustomBusinessDay(calendar=calendar)
    ts = pd.Timestamp(date)
    result = ts - custom_bday + custom_bday
    if isinstance(date, pd.Timestamp):
        return result
    return result.date()


def get_closest_prev_business_day(
    date: pd.Timestamp, calendar: pd.tseries.holiday.AbstractHolidayCalendar
) -> pd.Timestamp:
    custom_bday = pd.offsets.CustomBusinessDay(calendar=calendar)
    ts = pd.Timestamp(date)
    result = ts + custom_bday - custom_bday
    if isinstance(date, pd.Timestamp):
        return result
    return result.date()
