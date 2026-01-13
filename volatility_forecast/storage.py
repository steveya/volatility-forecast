from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd
import joblib


def _ts_utc(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass
class VolForecastStore:
    """
    Domain store for:
      - model registry (metadata + artifact path)
      - forecasts (point-in-time forecasts)
      - actuals (realized target series for reports)

    Uses DuckDB for metadata/tables and filesystem for artifacts.
    """

    root: str

    def __post_init__(self):
        self.root_path = Path(self.root).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.root_path / "volatility_forecast.duckdb"
        self.models_dir = self.root_path / "models"
        self.models_dir.mkdir(exist_ok=True)

        self._init_db()

    def _conn(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._conn() as con:
            con.execute(
                """
            CREATE TABLE IF NOT EXISTS models (
                model_id VARCHAR PRIMARY KEY,
                model_type VARCHAR NOT NULL,
                version INTEGER NOT NULL,
                created_utc TIMESTAMPTZ NOT NULL,
                trained_start TIMESTAMPTZ,
                trained_end TIMESTAMPTZ,
                metric_name VARCHAR,
                metric_value DOUBLE,
                params_json VARCHAR,
                artifact_path VARCHAR NOT NULL
            );
            """
            )
            con.execute(
                """
            CREATE TABLE IF NOT EXISTS forecasts (
                forecast_id VARCHAR PRIMARY KEY,
                model_id VARCHAR NOT NULL,
                ticker VARCHAR NOT NULL,
                asof_utc TIMESTAMPTZ NOT NULL,
                target_utc TIMESTAMPTZ NOT NULL,
                horizon INTEGER NOT NULL,
                yhat DOUBLE NOT NULL,
                created_utc TIMESTAMPTZ NOT NULL,
                meta_json VARCHAR
            );
            """
            )
            con.execute(
                """
            CREATE TABLE IF NOT EXISTS actuals (
                actual_id VARCHAR PRIMARY KEY,
                ticker VARCHAR NOT NULL,
                target_utc TIMESTAMPTZ NOT NULL,
                horizon INTEGER NOT NULL,
                y DOUBLE NOT NULL,
                created_utc TIMESTAMPTZ NOT NULL,
                meta_json VARCHAR
            );
            """
            )

            # Helpful indexes
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_models_type_ver ON models(model_type, version);"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_fcst_target ON forecasts(ticker, target_utc);"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_act_target ON actuals(ticker, target_utc);"
            )

    def _model_meta(model_obj: Any) -> dict:
        meta = {
            "model_class": f"{model_obj.__class__.__module__}.{model_obj.__class__.__name__}"
        }
        for k_src, k_dst in [
            ("n_features_", "n_features"),
            ("feature_schema_hash_", "feature_schema_hash"),
            ("feature_names_", "feature_names"),
        ]:
            if hasattr(model_obj, k_src):
                v = getattr(model_obj, k_src)
                if k_dst == "feature_names" and v is not None:
                    v = list(v)
                meta[k_dst] = v
        return meta

    def register_model(
        self,
        model_type: str,
        model_obj: Any,
        *,
        trained_start: Optional[pd.Timestamp] = None,
        trained_end: Optional[pd.Timestamp] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        params: Optional[dict] = None,
    ) -> str:
        trained_start = _ts_utc(trained_start) if trained_start is not None else None
        trained_end = _ts_utc(trained_end) if trained_end is not None else None

        with self._conn() as con:
            row = con.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM models WHERE model_type = ?",
                [model_type],
            ).fetchone()
            version = int(row[0])

        model_id = f"{model_type}:{version:04d}"
        artifact_dir = self.models_dir / model_type
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{model_id}.joblib"
        joblib.dump(model_obj, artifact_path)

        created = pd.Timestamp.utcnow().tz_localize("UTC")

        # ---- enrich params with model metadata automatically ----
        p = dict(params or {})
        p.update(_model_meta(model_obj))
        params_json = json.dumps(p, sort_keys=True, default=str)

        with self._conn() as con:
            con.execute(
                """
                INSERT INTO models(
                    model_id, model_type, version, created_utc,
                    trained_start, trained_end,
                    metric_name, metric_value, params_json, artifact_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    model_id,
                    model_type,
                    version,
                    created,
                    trained_start,
                    trained_end,
                    metric_name,
                    metric_value,
                    params_json,
                    str(artifact_path),
                ],
            )
        return model_id

    def load_latest_model(self, model_type: str):
        with self._conn() as con:
            row = con.execute(
                """
                SELECT artifact_path
                FROM models
                WHERE model_type = ?
                ORDER BY version DESC
                LIMIT 1
            """,
                [model_type],
            ).fetchone()
        if row is None:
            return None
        return joblib.load(row[0])

    def latest_model_id(self, model_type: str) -> Optional[str]:
        with self._conn() as con:
            row = con.execute(
                """
                SELECT model_id
                FROM models
                WHERE model_type = ?
                ORDER BY version DESC
                LIMIT 1
            """,
                [model_type],
            ).fetchone()
        return None if row is None else str(row[0])

    def model_history(self, model_type: str) -> pd.DataFrame:
        with self._conn() as con:
            return con.execute(
                """
                SELECT *
                FROM models
                WHERE model_type = ?
                ORDER BY version
            """,
                [model_type],
            ).df()

    def get_model_row(self, model_id: str) -> Optional[dict]:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM models WHERE model_id = ?",
                [model_id],
            ).fetchone()
            if row is None:
                return None
            cols = [d[0] for d in con.description]
            return dict(zip(cols, row))

    def load_model(self, model_id: str):
        with self._conn() as con:
            row = con.execute(
                "SELECT artifact_path FROM models WHERE model_id = ?",
                [model_id],
            ).fetchone()
        if row is None:
            return None
        return joblib.load(row[0])

    # ----------------------------
    # Forecasts
    # ----------------------------
    def upsert_forecast(
        self,
        *,
        model_id: str,
        ticker: str,
        asof_utc: pd.Timestamp,
        target_utc: pd.Timestamp,
        horizon: int,
        yhat: float,
        meta: Optional[dict] = None,
    ) -> str:
        asof_utc = _ts_utc(asof_utc)
        target_utc = _ts_utc(target_utc)
        created = pd.Timestamp.utcnow().tz_localize("UTC")

        # forecast_id is deterministic for idempotency
        forecast_id = f"{model_id}|{ticker}|{asof_utc.isoformat()}|{target_utc.isoformat()}|h{horizon}"
        meta_json = json.dumps(meta or {}, sort_keys=True)

        with self._conn() as con:
            con.execute(
                """
                INSERT INTO forecasts(
                    forecast_id, model_id, ticker, asof_utc, target_utc, horizon, yhat, created_utc, meta_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(forecast_id) DO UPDATE SET
                    yhat=excluded.yhat,
                    created_utc=excluded.created_utc,
                    meta_json=excluded.meta_json
            """,
                [
                    forecast_id,
                    model_id,
                    ticker,
                    asof_utc,
                    target_utc,
                    horizon,
                    float(yhat),
                    created,
                    meta_json,
                ],
            )

        return forecast_id

    def fetch_forecasts(
        self,
        *,
        ticker: str,
        start_utc: Optional[pd.Timestamp] = None,
        end_utc: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        start_utc = _ts_utc(start_utc) if start_utc is not None else None
        end_utc = _ts_utc(end_utc) if end_utc is not None else None

        where = ["ticker = ?"]
        params = [ticker]
        if start_utc is not None:
            where.append("target_utc >= ?")
            params.append(start_utc)
        if end_utc is not None:
            where.append("target_utc <= ?")
            params.append(end_utc)

        sql = f"SELECT * FROM forecasts WHERE {' AND '.join(where)} ORDER BY target_utc"
        with self._conn() as con:
            return con.execute(sql, params).df()

    # ----------------------------
    # Actuals
    # ----------------------------
    def upsert_actual(
        self,
        *,
        ticker: str,
        target_utc: pd.Timestamp,
        horizon: int,
        y: float,
        meta: Optional[dict] = None,
    ) -> str:
        target_utc = _ts_utc(target_utc)
        created = pd.Timestamp.utcnow().tz_localize("UTC")
        actual_id = f"{ticker}|{target_utc.isoformat()}|h{horizon}"
        meta_json = json.dumps(meta or {}, sort_keys=True)

        with self._conn() as con:
            con.execute(
                """
                INSERT INTO actuals(
                    actual_id, ticker, target_utc, horizon, y, created_utc, meta_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(actual_id) DO UPDATE SET
                    y=excluded.y,
                    created_utc=excluded.created_utc,
                    meta_json=excluded.meta_json
            """,
                [actual_id, ticker, target_utc, horizon, float(y), created, meta_json],
            )
        return actual_id

    def join_forecast_vs_actual(
        self,
        *,
        ticker: str,
        start_utc: Optional[pd.Timestamp] = None,
        end_utc: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        start_utc = _ts_utc(start_utc) if start_utc is not None else None
        end_utc = _ts_utc(end_utc) if end_utc is not None else None

        where = ["f.ticker = ? AND a.ticker = ?"]
        params = [ticker, ticker]
        if start_utc is not None:
            where.append("f.target_utc >= ? AND a.target_utc >= ?")
            params.extend([start_utc, start_utc])
        if end_utc is not None:
            where.append("f.target_utc <= ? AND a.target_utc <= ?")
            params.extend([end_utc, end_utc])

        sql = f"""
        SELECT
            f.ticker,
            f.model_id,
            f.asof_utc,
            f.target_utc,
            f.horizon,
            f.yhat,
            a.y AS y_actual
        FROM forecasts f
        JOIN actuals a
          ON f.ticker = a.ticker
         AND f.target_utc = a.target_utc
         AND f.horizon = a.horizon
        WHERE {' AND '.join(where)}
        ORDER BY f.target_utc
        """
        with self._conn() as con:
            return con.execute(sql, params).df()
