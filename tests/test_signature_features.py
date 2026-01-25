"""
Test suite for signature-based features using sktime's SignatureTransformer.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from volatility_forecast.features.signature_features import SignatureFeaturesTemplate
from volatility_forecast.sources.simulated_garch import SimulatedGARCHSource
from volatility_forecast.pipeline import build_default_ctx
from alphaforge.data.context import DataContext
from alphaforge.features.template import SliceSpec


class TestSignatureFeaturesTemplate(unittest.TestCase):
    """Test the SignatureFeaturesTemplate with sktime SignatureTransformer."""

    def setUp(self):
        """Set up test context and data."""
        self.ctx = build_default_ctx()
        self.entity = "TEST_ENTITY"
        self.source_name = "simulated_garch"

        # Add simulated GARCH source
        sim_source = SimulatedGARCHSource(
            entity_id=self.entity,
            n_periods=2500,
            random_state=42,
            mu=0.0,
            omega=0.00001,
            alpha=0.1,
            beta=0.85,
            eta=4.0,
            shock_prob=0.005,
        )
        self.ctx.sources[self.source_name] = sim_source

        self.template = SignatureFeaturesTemplate()

    def test_template_initialization(self):
        """Test that template initializes correctly."""
        self.assertEqual(self.template.name, "signature")
        self.assertEqual(self.template.version, "2.0")
        self.assertIn("lags", self.template.param_space)
        self.assertIn("sig_level", self.template.param_space)
        self.assertIn("augmentation_list", self.template.param_space)

    def test_param_space(self):
        """Test parameter space definitions."""
        params = self.template.param_space

        # Check lags parameter
        self.assertEqual(params["lags"].type, "int")
        self.assertEqual(params["lags"].default, 5)

        # Check sig_level parameter
        self.assertEqual(params["sig_level"].type, "int")
        self.assertEqual(params["sig_level"].default, 2)

        # Check augmentation_list parameter
        self.assertEqual(params["augmentation_list"].type, "categorical")
        self.assertEqual(params["augmentation_list"].default, "all")
        self.assertIn("none", params["augmentation_list"].choices)
        self.assertIn("time", params["augmentation_list"].choices)
        self.assertIn("basepoint", params["augmentation_list"].choices)

    def test_transform_basic(self):
        """Test basic signature feature transform."""
        params = {
            "lags": 5,
            "sig_level": 2,
            "source": self.source_name,
            "table": "market.ohlcv",
            "price_col": "close",
            "augmentation_list": "all",
        }

        slice_spec = SliceSpec(
            entities=[self.entity],
            start=pd.Timestamp("2021-01-01", tz="UTC"),
            end=pd.Timestamp("2021-12-31", tz="UTC"),
        )

        result = self.template.transform(self.ctx, params, slice_spec, state=None)

        # Check result structure
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.X)
        self.assertIsNotNone(result.catalog)

        # Check that features were created
        self.assertGreater(len(result.X.columns), 0)
        self.assertEqual(len(result.X.columns), len(result.catalog))

        # Check that features have no NaN values
        self.assertFalse(result.X.isnull().any().any())

        # Check metadata
        self.assertEqual(result.meta["template"], "signature")
        self.assertEqual(result.meta["version"], "2.0")

    def test_augmentation_all(self):
        """Test signature features with all augmentations."""
        params = {
            "lags": 5,
            "sig_level": 2,
            "source": self.source_name,
            "table": "market.ohlcv",
            "price_col": "close",
            "augmentation_list": "all",
        }

        slice_spec = SliceSpec(
            entities=[self.entity],
            start=pd.Timestamp("2021-01-01", tz="UTC"),
            end=pd.Timestamp("2021-06-30", tz="UTC"),
        )

        result = self.template.transform(self.ctx, params, slice_spec, state=None)

        # Check that features were created
        self.assertGreater(len(result.X.columns), 0)

        # Check catalog contains augmentation info
        for idx, row in result.catalog.iterrows():
            self.assertEqual(row["augmentation_list"], "all")
            self.assertEqual(row["family"], "signature")
            self.assertEqual(row["sig_level"], 2)

    def test_augmentation_none(self):
        """Test signature features with no augmentations."""
        params = {
            "lags": 5,
            "sig_level": 2,
            "source": self.source_name,
            "table": "market.ohlcv",
            "price_col": "close",
            "augmentation_list": "none",
        }

        slice_spec = SliceSpec(
            entities=[self.entity],
            start=pd.Timestamp("2021-01-01", tz="UTC"),
            end=pd.Timestamp("2021-06-30", tz="UTC"),
        )

        result = self.template.transform(self.ctx, params, slice_spec, state=None)

        # Check that features were created
        self.assertGreater(len(result.X.columns), 0)

        # Verify catalog
        for idx, row in result.catalog.iterrows():
            self.assertEqual(row["augmentation_list"], "none")

    def test_augmentation_time(self):
        """Test signature features with time augmentation only."""
        params = {
            "lags": 5,
            "sig_level": 2,
            "source": self.source_name,
            "table": "market.ohlcv",
            "price_col": "close",
            "augmentation_list": "time",
        }

        slice_spec = SliceSpec(
            entities=[self.entity],
            start=pd.Timestamp("2021-01-01", tz="UTC"),
            end=pd.Timestamp("2021-06-30", tz="UTC"),
        )

        result = self.template.transform(self.ctx, params, slice_spec, state=None)

        # Check that features were created
        self.assertGreater(len(result.X.columns), 0)

        # Verify catalog
        for idx, row in result.catalog.iterrows():
            self.assertEqual(row["augmentation_list"], "time")

    def test_augmentation_basepoint(self):
        """Test signature features with basepoint augmentation only."""
        params = {
            "lags": 5,
            "sig_level": 2,
            "source": self.source_name,
            "table": "market.ohlcv",
            "price_col": "close",
            "augmentation_list": "basepoint",
        }

        slice_spec = SliceSpec(
            entities=[self.entity],
            start=pd.Timestamp("2021-01-01", tz="UTC"),
            end=pd.Timestamp("2021-06-30", tz="UTC"),
        )

        result = self.template.transform(self.ctx, params, slice_spec, state=None)

        # Check that features were created
        self.assertGreater(len(result.X.columns), 0)

        # Verify catalog
        for idx, row in result.catalog.iterrows():
            self.assertEqual(row["augmentation_list"], "basepoint")

    def test_different_sig_levels(self):
        """Test signature features with different signature depths."""
        for sig_level in [1, 2, 3]:
            params = {
                "lags": 5,
                "sig_level": sig_level,
                "source": self.source_name,
                "table": "market.ohlcv",
                "price_col": "close",
                "augmentation_list": "all",
            }

            slice_spec = SliceSpec(
                entities=[self.entity],
                start=pd.Timestamp("2021-01-01", tz="UTC"),
                end=pd.Timestamp("2021-03-31", tz="UTC"),
            )

            result = self.template.transform(self.ctx, params, slice_spec, state=None)

            # Check that features were created
            self.assertGreater(len(result.X.columns), 0)

            # Check that sig_level is recorded correctly
            for idx, row in result.catalog.iterrows():
                self.assertEqual(row["sig_level"], sig_level)

    def test_different_lags(self):
        """Test signature features with different lag windows."""
        for lags in [3, 5, 10]:
            params = {
                "lags": lags,
                "sig_level": 2,
                "source": self.source_name,
                "table": "market.ohlcv",
                "price_col": "close",
                "augmentation_list": "all",
            }

            slice_spec = SliceSpec(
                entities=[self.entity],
                start=pd.Timestamp("2021-01-01", tz="UTC"),
                end=pd.Timestamp("2021-03-31", tz="UTC"),
            )

            result = self.template.transform(self.ctx, params, slice_spec, state=None)

            # Check that features were created
            self.assertGreater(len(result.X.columns), 0)

            # Check that lag is recorded correctly
            for idx, row in result.catalog.iterrows():
                self.assertEqual(row["lag"], lags)

    def test_feature_consistency(self):
        """Test that the same input produces the same output."""
        params = {
            "lags": 5,
            "sig_level": 2,
            "source": self.source_name,
            "table": "market.ohlcv",
            "price_col": "close",
            "augmentation_list": "all",
        }

        slice_spec = SliceSpec(
            entities=[self.entity],
            start=pd.Timestamp("2021-01-01", tz="UTC"),
            end=pd.Timestamp("2021-06-30", tz="UTC"),
        )

        # Transform twice
        result1 = self.template.transform(self.ctx, params, slice_spec, state=None)
        result2 = self.template.transform(self.ctx, params, slice_spec, state=None)

        # Check that results are identical
        pd.testing.assert_frame_equal(result1.X, result2.X)
        pd.testing.assert_frame_equal(result1.catalog, result2.catalog)

    def test_catalog_structure(self):
        """Test that catalog has the expected structure."""
        params = {
            "lags": 5,
            "sig_level": 2,
            "source": self.source_name,
            "table": "market.ohlcv",
            "price_col": "close",
            "augmentation_list": "all",
        }

        slice_spec = SliceSpec(
            entities=[self.entity],
            start=pd.Timestamp("2021-01-01", tz="UTC"),
            end=pd.Timestamp("2021-06-30", tz="UTC"),
        )

        result = self.template.transform(self.ctx, params, slice_spec, state=None)

        # Check required columns in catalog
        required_cols = [
            "group_path",
            "family",
            "transform",
            "source_table",
            "source_col",
            "lag",
            "sig_level",
            "term",
            "augmentation_list",
        ]

        for col in required_cols:
            self.assertIn(col, result.catalog.columns)

    def test_feature_values_finite(self):
        """Test that all feature values are finite."""
        params = {
            "lags": 5,
            "sig_level": 2,
            "source": self.source_name,
            "table": "market.ohlcv",
            "price_col": "close",
            "augmentation_list": "all",
        }

        slice_spec = SliceSpec(
            entities=[self.entity],
            start=pd.Timestamp("2021-01-01", tz="UTC"),
            end=pd.Timestamp("2021-06-30", tz="UTC"),
        )

        result = self.template.transform(self.ctx, params, slice_spec, state=None)

        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(result.X.values)))

    def test_index_alignment(self):
        """Test that X and catalog indices are properly aligned."""
        params = {
            "lags": 5,
            "sig_level": 2,
            "source": self.source_name,
            "table": "market.ohlcv",
            "price_col": "close",
            "augmentation_list": "all",
        }

        slice_spec = SliceSpec(
            entities=[self.entity],
            start=pd.Timestamp("2021-01-01", tz="UTC"),
            end=pd.Timestamp("2021-06-30", tz="UTC"),
        )

        result = self.template.transform(self.ctx, params, slice_spec, state=None)

        # Check that X columns match catalog index (sets are equal)
        self.assertEqual(set(result.X.columns), set(result.catalog.index))

        # Check that they have the same length (no duplicates)
        self.assertEqual(len(result.X.columns), len(result.catalog.index))


class TestSignatureFeatureIntegration(unittest.TestCase):
    """Integration tests for signature features with different configurations."""

    def setUp(self):
        """Set up test context."""
        self.ctx = build_default_ctx()
        self.entity = "INTEGRATION_TEST"
        self.source_name = "simulated_garch"

        # Add simulated GARCH source with different parameters
        sim_source = SimulatedGARCHSource(
            entity_id=self.entity,
            n_periods=2500,
            random_state=123,
            mu=0.0,
            omega=0.00002,
            alpha=0.15,
            beta=0.80,
            eta=4.0,
            shock_prob=0.005,
        )
        self.ctx.sources[self.source_name] = sim_source

        self.template = SignatureFeaturesTemplate()

    def test_high_signature_level(self):
        """Test with higher signature level."""
        params = {
            "lags": 5,
            "sig_level": 4,
            "source": self.source_name,
            "table": "market.ohlcv",
            "price_col": "close",
            "augmentation_list": "all",
        }

        slice_spec = SliceSpec(
            entities=[self.entity],
            start=pd.Timestamp("2021-01-01", tz="UTC"),
            end=pd.Timestamp("2021-03-31", tz="UTC"),
        )

        result = self.template.transform(self.ctx, params, slice_spec, state=None)

        # Should create many more features with higher signature level
        self.assertGreater(len(result.X.columns), 10)
        self.assertTrue(np.all(np.isfinite(result.X.values)))

    def test_long_lag_window(self):
        """Test with longer lag window."""
        params = {
            "lags": 20,
            "sig_level": 2,
            "source": self.source_name,
            "table": "market.ohlcv",
            "price_col": "close",
            "augmentation_list": "all",
        }

        slice_spec = SliceSpec(
            entities=[self.entity],
            start=pd.Timestamp("2021-01-01", tz="UTC"),
            end=pd.Timestamp("2021-03-31", tz="UTC"),
        )

        result = self.template.transform(self.ctx, params, slice_spec, state=None)

        # Check features were created
        self.assertGreater(len(result.X.columns), 0)
        self.assertTrue(np.all(np.isfinite(result.X.values)))

    def test_comparison_across_augmentations(self):
        """Compare feature dimensions across different augmentations."""
        augmentation_configs = ["none", "time", "basepoint", "all"]
        results = {}

        for aug in augmentation_configs:
            params = {
                "lags": 5,
                "sig_level": 2,
                "source": self.source_name,
                "table": "market.ohlcv",
                "price_col": "close",
                "augmentation_list": aug,
            }

            slice_spec = SliceSpec(
                entities=[self.entity],
                start=pd.Timestamp("2021-01-01", tz="UTC"),
                end=pd.Timestamp("2021-03-31", tz="UTC"),
            )

            result = self.template.transform(self.ctx, params, slice_spec, state=None)
            results[aug] = result

            # All should produce features
            self.assertGreater(len(result.X.columns), 0)

        # Compare number of features (augmentation affects feature dimension)
        # "all" should generally have the most features due to both augmentations
        print(f"\nFeature counts by augmentation:")
        for aug in augmentation_configs:
            print(f"  {aug}: {len(results[aug].X.columns)} features")


if __name__ == "__main__":
    unittest.main()
