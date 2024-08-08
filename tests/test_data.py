import unittest
from unittest.mock import patch
from volatility_forecast.data.dataloader import TingleEoDDataLoader
from volatility_forecast.data.base import Field, DataSet
from volatility_forecast.data.dataset import PriceVolume
import pandas as pd


class TestTingleEoDDataLoader(unittest.TestCase):
    def setUp(self):
        self.tickers = (
            "AAPL",
            "MSFT",
        )
        self.loader = TingleEoDDataLoader(self.tickers)
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
        self.loader = TingleEoDDataLoader(self.tickers)
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


if __name__ == "__main__":
    unittest.main()
