"""Prediction service for CHE risk scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .repo import PredictionRepository

MODEL_DIR = Path(__file__).parent.parent / "model" / "saved_models"
MODEL_FILES: dict[str, str] = {
    "Logistic Regression": "log_reg.pkl",
    "Random Forest": "rf.pkl",
    "MLP": "mlp.pkl",
    "SVM": "svm.pkl",
    "XGBoost": "xgb.pkl",
    "GBM": "lgb.pkl",
}

# Expected input fields and basic sanity ranges.
NUMERIC_FIELDS: dict[str, tuple[float, float]] = {
    "Blood Ammonia": (0.0, 1_000.0),
    "Albumin": (0.0, 100.0),
}
CATEGORICAL_FIELDS: tuple[str, ...] = (
    "Tips",
    "HBV",
    "Splenomegaly",
    "History of Hepatic Encephalopathy",
)
REQUIRED_FIELDS: tuple[str, ...] = tuple(NUMERIC_FIELDS) + CATEGORICAL_FIELDS
_RISK_THRESHOLDS = (0.3, 0.7)


class ModelLoadError(RuntimeError):
    """Raised when model assets cannot be loaded."""


class InputValidationError(ValueError):
    """Raised when user input fails validation."""


class PredictionError(RuntimeError):
    """Raised when inference fails."""


@dataclass
class ModelArtifacts:
    models: dict[str, Any]
    scaler: Any
    encoder: Any
    val_cols: Sequence[str]
    cat_cols: Sequence[str]
    model_features: Sequence[str]


@dataclass
class PredictionResult:
    scores: dict[str, float]
    top_model: str
    top_score: float
    risk_level: str


def _load_file(path: Path) -> Any:
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")
    try:
        return joblib.load(path)
    except FileNotFoundError as exc:
        raise ModelLoadError(f"Model file missing: {path}") from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ModelLoadError(f"Failed to load {path.name}: {exc}") from exc


def load_artifacts(model_dir: Path = MODEL_DIR) -> ModelArtifacts:
    """Load models and preprocessors from disk."""
    base_dir = Path(model_dir).resolve()
    if not base_dir.exists():
        raise ModelLoadError(f"Model directory not found: {base_dir}")

    # 容错加载模型:允许部分模型加载失败
    models = {}
    failed_models = []
    for name, filename in MODEL_FILES.items():
        try:
            models[name] = _load_file(base_dir / filename)
        except ModelLoadError as e:
            failed_models.append(f"{name}: {str(e)}")
            continue

    if not models:
        raise ModelLoadError(f"所有模型加载失败: {'; '.join(failed_models)}")

    if failed_models:
        import warnings
        warnings.warn(f"部分模型加载失败(将使用其余{len(models)}个模型): {'; '.join(failed_models)}")

    scaler = _load_file(base_dir / "scaler.pkl")
    encoder = _load_file(base_dir / "encoder.pkl")
    val_cols = list(_load_file(base_dir / "val_cols.pkl"))
    cat_cols = list(_load_file(base_dir / "cat_cols.pkl"))
    model_features = list(_load_file(base_dir / "model_features.pkl"))
    return ModelArtifacts(
        models=models,
        scaler=scaler,
        encoder=encoder,
        val_cols=val_cols,
        cat_cols=cat_cols,
        model_features=model_features,
    )


@lru_cache(maxsize=1)
def _cached_artifacts(model_dir: str) -> ModelArtifacts:
    return load_artifacts(Path(model_dir))


def get_artifacts(model_dir: Path | str = MODEL_DIR) -> ModelArtifacts:
    """Return cached model artifacts, loading from disk if needed."""
    resolved = str(Path(model_dir).resolve())
    return _cached_artifacts(resolved)


def validate_input(payload: Mapping[str, Any]) -> dict[str, float | int]:
    """Ensure all required fields exist with sane values."""
    if not isinstance(payload, Mapping):
        raise InputValidationError("payload must be a mapping")

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise InputValidationError(f"missing fields: {', '.join(missing)}")

    cleaned: dict[str, float | int] = {}
    for field, (lower, upper) in NUMERIC_FIELDS.items():
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputValidationError(f"{field} must be a number")
        numeric = float(value)
        if numeric < lower or numeric > upper:
            raise InputValidationError(f"{field} must be between {lower} and {upper}")
        cleaned[field] = numeric

    for field in CATEGORICAL_FIELDS:
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputValidationError(f"{field} must be 0 or 1")
        if value not in (0, 1):
            raise InputValidationError(f"{field} must be 0 or 1")
        cleaned[field] = int(value)

    return cleaned


def _get_cat_feature_names(artifacts: ModelArtifacts) -> list[str]:
    encoder = artifacts.encoder
    if hasattr(encoder, "get_feature_names_out"):
        try:
            return list(encoder.get_feature_names_out(artifacts.cat_cols))
        except Exception:
            pass
    return list(artifacts.cat_cols)


def prepare_features(payload: Mapping[str, Any], artifacts: ModelArtifacts) -> pd.DataFrame:
    """Preprocess raw input into the feature frame expected by models."""
    df = pd.DataFrame([payload])
    try:
        input_val = df[artifacts.val_cols]
        input_cat = df[artifacts.cat_cols]
    except KeyError as exc:
        raise InputValidationError(f"input missing required columns: {exc}") from exc

    try:
        scaled_val = artifacts.scaler.transform(input_val)
        encoded_cat = artifacts.encoder.transform(input_cat)
    except Exception as exc:
        raise PredictionError(f"preprocessing failed: {exc}") from exc

    val_df = pd.DataFrame(scaled_val, columns=artifacts.val_cols)
    cat_df = pd.DataFrame(encoded_cat, columns=_get_cat_feature_names(artifacts))
    features = pd.concat([val_df, cat_df], axis=1)

    for column in artifacts.model_features:
        if column not in features.columns:
            features[column] = 0

    try:
        aligned = features[artifacts.model_features].astype(float)
    except Exception as exc:
        raise PredictionError(f"feature alignment failed: {exc}") from exc
    return aligned


def _predict_probability(model: Any, features: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        output = np.asarray(model.predict_proba(features))
        if output.ndim < 2 or output.shape[1] < 2:
            raise PredictionError("predict_proba returned unexpected shape")
        return float(output[0, -1])

    if hasattr(model, "predict"):
        output = np.asarray(model.predict(features))
        if output.size == 0:
            raise PredictionError("predict returned empty result")
        return float(output.reshape(-1)[0])

    raise PredictionError("model does not support prediction")


def _clip_probability(probability: float) -> float:
    return max(0.0, min(1.0, probability))


def predict_scores(features: pd.DataFrame, artifacts: ModelArtifacts) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, model in artifacts.models.items():
        try:
            prob = _predict_probability(model, features)
        except Exception as exc:
            raise PredictionError(f"model {name} failed: {exc}") from exc
        scores[name] = _clip_probability(prob)

    if not scores:
        raise PredictionError("no models available for prediction")
    return scores


def classify_risk(probability: float) -> str:
    low, medium = _RISK_THRESHOLDS
    if probability < low:
        return "low"
    if probability < medium:
        return "medium"
    return "high"


def persist_prediction(
    repo: PredictionRepository,
    user_id: int,
    payload: Mapping[str, Any],
    scores: Mapping[str, float],
    top_model: str,
) -> int:
    return repo.create_prediction(
        user_id=user_id,
        input_json=json.dumps(payload, sort_keys=True),
        model_scores_json=json.dumps(scores, sort_keys=True),
        top_model=top_model,
    )


def run_prediction(
    payload: Mapping[str, Any],
    artifacts: ModelArtifacts | None = None,
    repo: PredictionRepository | None = None,
    user_id: int | None = None,
) -> PredictionResult:
    cleaned = validate_input(payload)
    assets = artifacts or get_artifacts()
    features = prepare_features(cleaned, assets)
    scores = predict_scores(features, assets)
    top_model = max(scores, key=scores.get)
    top_score = scores[top_model]

    if repo is not None and user_id is not None:
        persist_prediction(repo, user_id, cleaned, scores, top_model)

    return PredictionResult(
        scores=scores,
        top_model=top_model,
        top_score=top_score,
        risk_level=classify_risk(top_score),
    )
