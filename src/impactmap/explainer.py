"""Local and global explanations for commit risk model predictions."""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from impactmap.predictor import DEFAULT_MODEL_PATH, DEFAULT_SCHEMA_PATH, LoadedModel, load_model

logger = logging.getLogger(__name__)
DISCLAIMER = (
    "Feature importance and SHAP describe the model's behavior, not the true cause of a bug.",
    "A high score does not prove a defect; a low score does not guarantee safety.",
    "The model was trained on ApacheJIT data covering Apache ActiveMQ, Camel, Cassandra, Flink, Groovy, Hadoop, Hadoop HDFS, Hadoop MapReduce, HBase, Hive, Ignite, Kafka, Spark, Zeppelin, and ZooKeeper projects.",
)


@dataclass
class FeatureExplanation:
    feature: str
    value: float
    contribution: float
    direction: str
    magnitude: float
    text: str


@dataclass
class ExplanationResult:
    available: bool
    local_method: Optional[str] = None
    global_method: Optional[str] = None
    features: List[FeatureExplanation] = field(default_factory=list)
    global_importance: Dict[str, float] = field(default_factory=dict)
    disclaimers: tuple = DISCLAIMER
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _shap_contributions(model: Any, frame: pd.DataFrame) -> np.ndarray:
    import shap
    explainer = shap.Explainer(model, feature_names=list(frame.columns))
    values = np.asarray(explainer(frame).values)
    if values.ndim == 3:
        values = values[0, :, -1]
    elif values.ndim == 2:
        values = values[0]
    return values.reshape(-1)


def _description(schema: dict, name: str, value: float, direction: str, magnitude: float) -> str:
    base = schema.get("feature_descriptions", {}).get(name, name)
    labels = {
        "la": f"This commit adds {value:g} lines",
        "ld": f"This commit deletes {value:g} lines",
        "nf": f"This commit touches {value:g} files",
        "nd": f"This commit touches {value:g} directories",
        "ns": f"This commit touches {value:g} subsystems",
        "ent": f"Change spread entropy is {value:g}",
        "ndev": f"The average prior developer count is {value:g}",
        "age": f"The average file age is {value:g} days",
        "nuc": f"The average prior change count is {value:g}",
        "aexp": f"The author has {value:g} prior commits",
    }
    phrase = labels.get(name, f"{name} is {value:g}")
    effect = "increased" if direction == "raises" else "decreased" if direction == "lowers" else "had little effect on"
    return f"{phrase}, a measure of {base.lower()} and {effect} estimated risk (contribution magnitude {magnitude:.4g})."


def explain_prediction(features: Any, *, loaded: Optional[LoadedModel] = None,
                       model_path: Path = DEFAULT_MODEL_PATH, schema_path: Path = DEFAULT_SCHEMA_PATH,
                       importance_path: Path = Path(__file__).resolve().parents[2] / "models/global_importance.json") -> ExplanationResult:
    """Explain feature effects. SHAP is preferred; documented approximations are fallback."""
    try:
        loaded = loaded or load_model(model_path, schema_path)
        raw = features.values if hasattr(features, "values") and not isinstance(features, dict) else features
        columns = loaded.schema["feature_columns"]
        if list(raw.keys()) != columns:
            raise ValueError("Feature names/order do not match saved schema")
        frame = pd.DataFrame([[float(raw[c]) for c in columns]], columns=columns)
        artifact = loaded.artifact
        base_model = artifact.get("base_model", artifact["calibrated_model"])
        notes: List[str] = []
        try:
            contributions = _shap_contributions(base_model, frame)
            local_method = "SHAP"
        except Exception as shap_error:
            logger.warning("SHAP explanation failed; using approximation: %s", shap_error)
            local_method = "one-feature-at-a-time baseline substitution (approximation)"
            medians = artifact.get("training_medians")
            if not medians:
                raise RuntimeError("SHAP failed and training medians are absent; local fallback cannot be computed") from shap_error
            model = artifact["calibrated_model"]
            baseline = {c: float(medians[c]) for c in columns}
            base_frame = pd.DataFrame([baseline], columns=columns)
            base_score = float(model.predict_proba(base_frame)[0, 1])
            contributions = np.zeros(len(columns), dtype=float)
            for i, name in enumerate(columns):
                changed = dict(baseline)
                changed[name] = float(raw[name])
                contributions[i] = float(model.predict_proba(pd.DataFrame([changed], columns=columns))[0, 1]) - base_score
            notes.append("Local effects use one-feature-at-a-time baseline substitution and are approximate.")
        values = []
        for name, value, contribution in zip(columns, (float(raw[c]) for c in columns), contributions):
            c = float(contribution)
            direction = "raises" if c > 1e-9 else "lowers" if c < -1e-9 else "neutral"
            values.append(FeatureExplanation(name, value, c, direction, abs(c),
                                              _description(loaded.schema, name, value, direction, abs(c))))
        global_importance = {}
        global_method = "training-time permutation importance"
        if importance_path.is_file():
            global_importance = {k: float(v) for k, v in json.loads(importance_path.read_text()).items()}
        else:
            notes.append("Global importance artifact is missing; run the training script to compute permutation importance.")
            global_method = "unavailable"
        return ExplanationResult(True, local_method, global_method, values, global_importance, notes=notes)
    except Exception as exc:
        return ExplanationResult(False, notes=["No explanation was substituted for unavailable evidence."], error=str(exc))
