import pandas as pd
import pandas_market_calendars as mcal

from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

import unittest
from unittest.mock import patch
from volatility_forecast.data.dataloader import (
    TiingoEoDDataLoader,
    PriceVolumeDatabaseLoader,
    TiingoEodDataLoaderProd,
)
from volatility_forecast.data.base import Field, DataSet
from volatility_forecast.data.date_util import (
    get_closest_next_business_day,
    get_closest_prev_business_day,
)
from volatility_forecast.data.dataset import PriceVolume
from volatility_forecast.data.persistence import persist_data, load_data_from_db
from volatility_forecast.data.database import engine, Base


class TestTingleEoDDataLoader(unittest.TestCase):
    def setUp(self):
        self.tickers = (
            "AAPL",
            "MSFT",
        )
        self.loader = TiingoEoDDataLoader(self.tickers)
        self.fields = [PriceVolume.CLOSE, PriceVolume.VOLUME]

    def test_convert_name(self):
        self.assertEqual(self.loader.convert_name("CLOSE"), "adjClose")
        self.assertEqual(self.loader.convert_name("VOLUME"), "adjVolume")


class TestPriceVolume(unittest.TestCase):
    def setUp(self):
        self.tickers = (
            "AAPL",
            "MSFT",
        )
        self.loader = TiingoEoDDataLoader(self.tickers)
        self.fields = [PriceVolume.CLOSE, PriceVolume.VOLUME]

    def test_columns_exist(self):
        self.assertTrue(hasattr(PriceVolume, "OPEN"))
        self.assertTrue(hasattr(PriceVolume, "HIGH"))
        self.assertTrue(hasattr(PriceVolume, "LOW"))
        self.assertTrue(hasattr(PriceVolume, "CLOSE"))
        self.assertTrue(hasattr(PriceVolume, "VOLUME"))

    def test_get_data(self):
        data = PriceVolume.CLOSE.get_data(self.loader, "2021-01-01", "2021-01-10")
        self.assertEqual(len(data), 5)


class TestField(unittest.TestCase):
    def test_field_initialization(self):
        field = Field(float, doc="Test field")
        self.assertEqual(field.dtype, float)
        self.assertEqual(field.doc, "Test field")
        self.assertEqual(field.metadata, {})

    def test_field_binding(self):
        field = Field(float)
        bound_field = field.bind("test_field")
        self.assertEqual(bound_field.name, "test_field")
        self.assertEqual(bound_field.metadata, {})


class TestDataSet(unittest.TestCase):
    def test_get_field(self):
        class TestDataSet(DataSet):
            TEST_FIELD = Field(float)

        field = TestDataSet.get_field("TEST_FIELD")
        self.assertEqual(field.name, "TEST_FIELD")
        self.assertEqual(field.dtype, float)


class TestLoaderAndPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create tables
        Base.metadata.create_all(engine)

        cls.tickers = (
            "SPY",
            "AAPL",
        )
        cls.end_date = get_closest_prev_business_day(
            datetime.now().date() - timedelta(days=5),
            mcal.get_calendar("NYSE"),
        )
        cls.start_date = get_closest_next_business_day(
            cls.end_date - timedelta(days=30),
            mcal.get_calendar("NYSE"),
        )
        cls.loader = TiingoEodDataLoaderProd(cls.tickers)

    def setUp(self):
        # This method will be called before each test
        pass

    def tearDown(self):
        # This method will be called after each test
        pass

    def test_fetch_and_persist_data(self):
        data = self.loader.get_data(
            fields=[PriceVolume.CLOSE, PriceVolume.VOLUME],
            start_date=self.start_date,
            end_date=self.end_date,
        )

        self.assertIsNotNone(data)
        self.assertEqual(len(data), 2)  # CLOSE and VOLUME
        self.assertIn(PriceVolume.CLOSE, data)
        self.assertIn(PriceVolume.VOLUME, data)

        for field, field_data in data.items():
            self.assertIsInstance(field_data, pd.DataFrame)
            self.assertEqual(len(field_data.columns), len(self.tickers))

    def test_data_in_database(self):
        for ticker in self.tickers:
            db_data = load_data_from_db(ticker, self.start_date, self.end_date)
            self.assertIsNotNone(db_data)
            self.assertFalse(db_data.empty)
            self.assertTrue("adjClose" in db_data.columns)
            self.assertTrue("adjVolume" in db_data.columns)

    def test_fetch_from_database(self):
        # Fetch data twice
        data1 = self.loader.get_data(
            fields=[PriceVolume.CLOSE, PriceVolume.VOLUME],
            start_date=self.start_date,
            end_date=self.end_date,
        )

        data2 = self.loader.get_data(
            fields=[PriceVolume.CLOSE, PriceVolume.VOLUME],
            start_date=self.start_date,
            end_date=self.end_date,
        )

        # Verify that the data is the same
        for field in [PriceVolume.CLOSE, PriceVolume.VOLUME]:
            pd.testing.assert_frame_equal(data1[field], data2[field])

    def test_fetch_new_data(self):
        new_end_date = self.end_date + timedelta(days=1)
        new_data = self.loader.get_data(
            fields=[PriceVolume.CLOSE, PriceVolume.VOLUME],
            start_date=self.start_date,
            end_date=new_end_date,
        )

        for field, field_data in new_data.items():
            self.assertTrue(field_data.index.max().date() >= self.end_date)


class TestDatabaseDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use an in-memory SQLite database for testing
        cls.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(cls.engine)
        cls.Session = sessionmaker(bind=cls.engine)

        # Set up test data
        cls.tickers = (
            "AAPL",
            "GOOGL",
        )

        cls.end_date = get_closest_prev_business_day(
            datetime.now().date() - timedelta(days=5),
            mcal.get_calendar("NYSE"),
        )
        cls.start_date = get_closest_next_business_day(
            cls.end_date - timedelta(days=30),
            mcal.get_calendar("NYSE"),
        )

        # Create some test data
        cls.test_data = {}
        for ticker in cls.tickers:
            dates = pd.date_range(cls.start_date, cls.end_date)
            cls.test_data[ticker] = pd.DataFrame(
                {
                    "open": range(len(dates)),
                    "high": range(len(dates)),
                    "low": range(len(dates)),
                    "close": range(len(dates)),
                    "volume": range(1000, 1000 + len(dates)),
                },
                index=dates,
            )
            persist_data(cls.test_data[ticker], ticker, session=cls.Session())

        cls.loader = PriceVolumeDatabaseLoader(cls.tickers)

    def setUp(self):
        # Create a new session for each test
        self.session = self.Session()

    def tearDown(self):
        # Close the session after each test
        self.session.close()

    def test_get_data_single_field(self):
        data = self.loader.get_data(
            fields=[PriceVolume.CLOSE],
            start_date=self.start_date,
            end_date=self.end_date,
        )

        self.assertIn(PriceVolume.CLOSE, data)
        self.assertIsInstance(data[PriceVolume.CLOSE], pd.DataFrame)
        self.assertEqual(len(data[PriceVolume.CLOSE].columns), len(self.tickers))
        self.assertEqual(
            len(data[PriceVolume.CLOSE]), (self.end_date - self.start_date).days + 1
        )

    def test_get_data_multiple_fields(self):
        data = self.loader.get_data(
            fields=[PriceVolume.CLOSE, PriceVolume.VOLUME],
            start_date=self.start_date,
            end_date=self.end_date,
        )

        self.assertIn(PriceVolume.CLOSE, data)
        self.assertIn(PriceVolume.VOLUME, data)
        self.assertEqual(len(data), 2)

    def test_get_data_date_range(self):
        mid_date = self.start_date + timedelta(days=5)
        data = self.loader.get_data(
            fields=[PriceVolume.CLOSE], start_date=mid_date, end_date=self.end_date
        )

        self.assertEqual(
            len(data[PriceVolume.CLOSE]), (self.end_date - mid_date).days + 1
        )

    def test_get_data_single_ticker(self):
        single_ticker_loader = PriceVolumeDatabaseLoader([self.tickers[0]])
        data = single_ticker_loader.get_data(
            fields=[PriceVolume.CLOSE],
            start_date=self.start_date,
            end_date=self.end_date,
        )

        self.assertEqual(len(data[PriceVolume.CLOSE].columns), 1)
        self.assertEqual(data[PriceVolume.CLOSE].columns[0], self.tickers[0])

    def test_get_data_non_existent_ticker(self):
        non_existent_loader = PriceVolumeDatabaseLoader(["NON_EXISTENT"])
        data = non_existent_loader.get_data(
            fields=[PriceVolume.CLOSE],
            start_date=self.start_date,
            end_date=self.end_date,
        )

        self.assertTrue(data[PriceVolume.CLOSE].empty)

    def test_get_data_non_existent_date_range(self):
        future_start = self.end_date + timedelta(days=1)
        future_end = future_start + timedelta(days=5)
        data = self.loader.get_data(
            fields=[PriceVolume.CLOSE], start_date=future_start, end_date=future_end
        )

        self.assertTrue(data[PriceVolume.CLOSE].empty)


if __name__ == "__main__":
    unittest.main()
