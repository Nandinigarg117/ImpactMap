"""Load and apply the saved commit risk model without blocking other app features."""

from dataclasses import dataclass, field
from functools import lru_cache
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn

from impactmap.feature_extractor import ExtractedFeatures

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models/model.joblib"
DEFAULT_SCHEMA_PATH = ROOT / "models/feature_schema.json"
DEFAULT_METRICS_PATH = ROOT / "models/metrics.json"
DEFAULT_TIERS_PATH = ROOT / "models/risk_tiers.json"


@dataclass
class PredictionResult:
    available: bool
    score: Optional[float] = None
    tier: Optional[str] = None
    threshold_used: Optional[float] = None
    feature_values: Dict[str, float] = field(default_factory=dict)
    missing_or_degraded_features: List[str] = field(default_factory=list)
    model_version: Optional[str] = None
    reliability_notes: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class LoadedModel:
    artifact: Dict[str, Any]
    schema: Dict[str, Any]
    warnings: List[str]


def _versions() -> Dict[str, str]:
    versions = {"scikit_learn": sklearn.__version__}
    try:
        import shap
        versions["shap"] = shap.__version__
    except Exception:
        pass
    try:
        import pandas
        versions["pandas"] = pandas.__version__
    except Exception:
        pass
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except Exception:
        pass
    return versions


@lru_cache(maxsize=4)
def _load_cached(model_path: str, schema_path: str, metrics_path: str) -> LoadedModel:
    model_file, schema_file, metrics_file = map(Path, (model_path, schema_path, metrics_path))
    if not model_file.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_file}")
    if not schema_file.is_file():
        raise FileNotFoundError(f"Feature schema not found: {schema_file}")
    artifact = joblib.load(model_file)
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or "calibrated_model" not in artifact:
        raise ValueError("Model artifact does not contain a calibrated_model")
    columns = schema.get("feature_columns")
    dtypes = schema.get("feature_dtypes")
    if not isinstance(columns, list) or not columns or not isinstance(dtypes, dict):
        raise ValueError("Feature schema is missing ordered feature_columns or feature_dtypes")
    if list(artifact.get("feature_columns", [])) != columns:
        raise ValueError("Model feature names/order do not match feature_schema.json")
    if set(columns) != set(dtypes):
        raise ValueError("Feature schema dtype names do not match feature_columns")
    warnings: List[str] = []
    if metrics_file.is_file():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        trained = metrics.get("library_versions", {})
        for name, current in _versions().items():
            expected = trained.get(name)
            if expected and expected != current:
                message = f"Library version mismatch for {name}: model metadata={expected}, runtime={current}."
                warnings.append(message)
                logger.warning(message)
    else:
        warnings.append("Training metrics metadata is missing; library compatibility could not be checked.")
    return LoadedModel(artifact=artifact, schema=schema, warnings=warnings)


def load_model(model_path: Path = DEFAULT_MODEL_PATH, schema_path: Path = DEFAULT_SCHEMA_PATH,
               metrics_path: Path = DEFAULT_METRICS_PATH) -> LoadedModel:
    """Load and validate an artifact once per path tuple; cache can be cleared in tests."""
    return _load_cached(str(model_path.resolve()), str(schema_path.resolve()), str(metrics_path.resolve()))


def _extract_features(features: Any) -> Tuple[Mapping[str, Any], List[str], List[str]]:
    if isinstance(features, ExtractedFeatures):
        return features.values, list(features.missing_or_degraded), list(features.notes)
    if isinstance(features, Mapping):
        return features, [], []
    raise TypeError("features must be ExtractedFeatures or a feature mapping")


def predict_risk(features: Any, *, model_path: Path = DEFAULT_MODEL_PATH,
                 schema_path: Path = DEFAULT_SCHEMA_PATH, metrics_path: Path = DEFAULT_METRICS_PATH,
                 tiers_path: Path = DEFAULT_TIERS_PATH) -> PredictionResult:
    """Predict a risk score; all failures are represented explicitly as unavailable."""
    values: Dict[str, float] = {}
    degraded: List[str] = []
    notes: List[str] = []
    version = None
    try:
        loaded = load_model(model_path, schema_path, metrics_path)
        artifact, schema = loaded.artifact, loaded.schema
        version = artifact.get("model_version")
        notes.extend(loaded.warnings)
        raw, degraded, extraction_notes = _extract_features(features)
        notes.extend(extraction_notes)
        columns = schema["feature_columns"]
        missing = [c for c in columns if c not in raw]
        if missing:
            return PredictionResult(False, feature_values=dict(raw), missing_or_degraded_features=missing + degraded,
                                    model_version=version, reliability_notes=notes,
                                    error=f"Missing required feature(s): {', '.join(missing)}")
        extra = [c for c in raw if c not in columns]
        if extra:
            return PredictionResult(False, feature_values=dict(raw), missing_or_degraded_features=degraded,
                                    model_version=version, reliability_notes=notes,
                                    error=f"Unexpected feature(s): {', '.join(extra)}")
        if list(raw.keys()) != columns:
            return PredictionResult(False, feature_values=dict(raw), missing_or_degraded_features=degraded,
                                    model_version=version, reliability_notes=notes,
                                    error="Input feature names/order do not match saved schema")
        for name in columns:
            val = raw[name]
            dtype = schema["feature_dtypes"][name]
            if isinstance(val, bool) or not isinstance(val, (int, float, np.number)) or not math.isfinite(float(val)):
                raise ValueError(f"Feature '{name}' must be a finite numeric value")
            if dtype.startswith("int") and float(val).is_integer() is False:
                raise ValueError(f"Feature '{name}' must match schema dtype {dtype}")
            values[name] = float(val)
        if values.get("nf", 0) == 0:
            return PredictionResult(False, feature_values=values, missing_or_degraded_features=degraded,
                                    model_version=version, reliability_notes=notes,
                                    error="Risk score unavailable: commit has zero changed source files.")
        if degraded:
            notes.append("Some history features are degraded; interpret this risk score with reduced reliability.")
        ranges = artifact.get("training_ranges")
        if ranges:
            out = [n for n in columns if n in ranges and
                   (values[n] < ranges[n]["min"] or values[n] > ranges[n]["max"])]
            if out:
                notes.append("Out-of-training-range feature values detected: " + ", ".join(out) + ".")
        else:
            notes.append("Training feature ranges are absent; out-of-range checks could not be performed.")
        frame = pd.DataFrame([[values[c] for c in columns]], columns=columns)
        model = artifact["calibrated_model"]
        score = float(model.predict_proba(frame)[0, 1])
        if not math.isfinite(score):
            raise ValueError("Model returned a non-finite risk score")
        tier_data = json.loads(Path(tiers_path).read_text(encoding="utf-8")) if Path(tiers_path).is_file() else artifact.get("risk_tiers", {})
        high = float(tier_data.get("High", artifact.get("threshold", 0.5)))
        medium = float(tier_data.get("Medium", high * 0.6))
        tier = "High" if score >= high else "Medium" if score >= medium else "Low"
        notes.append("Output is a risk score; calibration is not independently verified by metrics metadata.")
        return PredictionResult(True, score, tier, float(artifact.get("threshold", high)), values,
                                degraded, version, notes)
    except Exception as exc:
        logger.exception("Risk prediction unavailable")
        return PredictionResult(False, feature_values=values, missing_or_degraded_features=degraded,
                                model_version=version, reliability_notes=notes,
                                error=f"Risk prediction unavailable: {exc}")
