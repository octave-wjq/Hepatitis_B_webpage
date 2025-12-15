import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app import predict


class StubRepo:
    def __init__(self) -> None:
        self.records = []

    def create_prediction(self, user_id: int, input_json: str, model_scores_json: str, top_model: str) -> int:
        record = {
            "user_id": user_id,
            "input_json": input_json,
            "model_scores_json": model_scores_json,
            "top_model": top_model,
        }
        self.records.append(record)
        return len(self.records)


class IdentityScaler:
    def __init__(self) -> None:
        self.seen = None

    def transform(self, df: pd.DataFrame):
        self.seen = df.copy()
        return df.astype(float).values


class IdentityEncoder:
    def __init__(self) -> None:
        self.seen = None

    def transform(self, df: pd.DataFrame):
        self.seen = df.copy()
        return df.astype(float).values

    def get_feature_names_out(self, cols):
        return list(cols)


class ProbaModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.seen = None

    def predict_proba(self, features: pd.DataFrame):
        self.seen = features.copy()
        return np.array([[1 - self.probability, self.probability]])


class DirectModel:
    def __init__(self, value: float) -> None:
        self.value = value
        self.seen = None

    def predict(self, features: pd.DataFrame):
        self.seen = features.copy()
        return np.array([self.value])


class FailingModel:
    def predict_proba(self, features: pd.DataFrame):
        raise RuntimeError("boom")


class ExplodingEncoder:
    def transform(self, df: pd.DataFrame):
        raise RuntimeError("encode failed")


class NoFeatureNamesEncoder:
    def get_feature_names_out(self, cols):
        raise RuntimeError("no names")

    def transform(self, df: pd.DataFrame):
        return df.values


class BadShapeProbaModel:
    def predict_proba(self, features: pd.DataFrame):
        return np.array([0.4])


class EmptyPredictModel:
    def predict(self, features: pd.DataFrame):
        return np.array([])


@pytest.fixture()
def valid_payload():
    return {
        "Blood Ammonia": 40.0,
        "Albumin": 32.5,
        "Tips": 1,
        "HBV": 0,
        "Splenomegaly": 1,
        "History of Hepatic Encephalopathy": 0,
    }


def test_load_artifacts_missing_dir(tmp_path):
    missing = tmp_path / "absent"
    with pytest.raises(predict.ModelLoadError):
        predict.load_artifacts(missing)


def test_load_file_missing_and_wrap(monkeypatch, tmp_path):
    with pytest.raises(predict.ModelLoadError):
        predict._load_file(tmp_path / "missing.pkl")

    existing = tmp_path / "exists.pkl"
    existing.write_text("stub")

    def raise_fnf(path):
        raise FileNotFoundError("gone")

    monkeypatch.setattr(predict, "joblib", SimpleNamespace(load=raise_fnf))
    with pytest.raises(predict.ModelLoadError):
        predict._load_file(existing)


def test_load_artifacts_with_mocked_files(monkeypatch, tmp_path):
    files = list(predict.MODEL_FILES.values()) + [
        "scaler.pkl",
        "encoder.pkl",
        "val_cols.pkl",
        "cat_cols.pkl",
        "model_features.pkl",
    ]
    for filename in files:
        (tmp_path / filename).write_text("placeholder")

    mapping = {
        "log_reg.pkl": "log",
        "rf.pkl": "rf",
        "mlp.pkl": "mlp",
        "svm.pkl": "svm",
        "xgb.pkl": "xgb",
        "lgb.pkl": "lgb",
        "scaler.pkl": "scaler",
        "encoder.pkl": "encoder",
        "val_cols.pkl": ["Blood Ammonia", "Albumin"],
        "cat_cols.pkl": ["HBV"],
        "model_features.pkl": ["Blood Ammonia", "Albumin", "HBV"],
    }

    def fake_load(path: Path):
        return mapping[path.name]

    monkeypatch.setattr(predict, "joblib", SimpleNamespace(load=fake_load))

    artifacts = predict.load_artifacts(tmp_path)
    assert artifacts.models["Random Forest"] == "rf"
    assert artifacts.scaler == "scaler"
    assert artifacts.encoder == "encoder"
    assert artifacts.val_cols == ["Blood Ammonia", "Albumin"]
    assert artifacts.cat_cols == ["HBV"]
    assert artifacts.model_features[-1] == "HBV"


def test_get_artifacts_uses_cache(monkeypatch, tmp_path):
    predict._cached_artifacts.cache_clear()
    calls = []

    def fake_load(path: Path):
        calls.append(path)
        return "artifact"

    monkeypatch.setattr(predict, "load_artifacts", fake_load)
    first = predict.get_artifacts(tmp_path)
    second = predict.get_artifacts(tmp_path)

    assert first == "artifact"
    assert second == "artifact"
    assert len(calls) == 1
    predict._cached_artifacts.cache_clear()


def test_validate_input_rejects_missing_and_ranges(valid_payload):
    bad_payload = valid_payload.copy()
    bad_payload.pop("HBV")
    with pytest.raises(predict.InputValidationError):
        predict.validate_input(bad_payload)

    out_of_range = valid_payload.copy()
    out_of_range["Albumin"] = -1
    with pytest.raises(predict.InputValidationError):
        predict.validate_input(out_of_range)

    wrong_type = valid_payload.copy()
    wrong_type["Tips"] = "yes"
    with pytest.raises(predict.InputValidationError):
        predict.validate_input(wrong_type)

    bool_payload = valid_payload.copy()
    bool_payload["HBV"] = True
    with pytest.raises(predict.InputValidationError):
        predict.validate_input(bool_payload)

    numeric_type = valid_payload.copy()
    numeric_type["Albumin"] = "high"
    with pytest.raises(predict.InputValidationError):
        predict.validate_input(numeric_type)

    cat_bad_value = valid_payload.copy()
    cat_bad_value["Tips"] = 2
    with pytest.raises(predict.InputValidationError):
        predict.validate_input(cat_bad_value)

    with pytest.raises(predict.InputValidationError):
        predict.validate_input(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_get_cat_feature_names_fallback(valid_payload):
    artifacts = predict.ModelArtifacts(
        models={},
        scaler=IdentityScaler(),
        encoder=NoFeatureNamesEncoder(),
        val_cols=list(predict.NUMERIC_FIELDS.keys()),
        cat_cols=list(predict.CATEGORICAL_FIELDS),
        model_features=[],
    )
    assert predict._get_cat_feature_names(artifacts) == list(predict.CATEGORICAL_FIELDS)


def test_prepare_features_missing_columns(valid_payload):
    artifacts = predict.ModelArtifacts(
        models={},
        scaler=IdentityScaler(),
        encoder=IdentityEncoder(),
        val_cols=["Blood Ammonia", "Albumin", "Extra"],
        cat_cols=list(predict.CATEGORICAL_FIELDS),
        model_features=[],
    )
    with pytest.raises(predict.InputValidationError):
        predict.prepare_features(valid_payload, artifacts)


def test_prepare_features_preprocess_failure(valid_payload):
    artifacts = predict.ModelArtifacts(
        models={},
        scaler=IdentityScaler(),
        encoder=ExplodingEncoder(),
        val_cols=list(predict.NUMERIC_FIELDS.keys()),
        cat_cols=list(predict.CATEGORICAL_FIELDS),
        model_features=[],
    )
    with pytest.raises(predict.PredictionError):
        predict.prepare_features(valid_payload, artifacts)


def test_prepare_features_alignment_failure(valid_payload):
    class StringScaler:
        def transform(self, df: pd.DataFrame):
            return np.array([["bad", "data"]])

    artifacts = predict.ModelArtifacts(
        models={},
        scaler=StringScaler(),
        encoder=IdentityEncoder(),
        val_cols=list(predict.NUMERIC_FIELDS.keys()),
        cat_cols=list(predict.CATEGORICAL_FIELDS),
        model_features=list(predict.NUMERIC_FIELDS.keys()) + list(predict.CATEGORICAL_FIELDS),
    )
    with pytest.raises(predict.PredictionError):
        predict.prepare_features(valid_payload, artifacts)


def test_full_prediction_flow_persists(valid_payload):
    val_cols = list(predict.NUMERIC_FIELDS.keys())
    cat_cols = list(predict.CATEGORICAL_FIELDS)
    model_features = val_cols + cat_cols + ["missing_feature"]

    artifacts = predict.ModelArtifacts(
        models={"Random Forest": ProbaModel(0.25), "GBM": DirectModel(0.8)},
        scaler=IdentityScaler(),
        encoder=IdentityEncoder(),
        val_cols=val_cols,
        cat_cols=cat_cols,
        model_features=model_features,
    )
    repo = StubRepo()

    result = predict.run_prediction(valid_payload, artifacts=artifacts, repo=repo, user_id=3)

    assert result.top_model == "GBM"
    assert result.risk_level == "high"
    assert abs(result.top_score - 0.8) < 1e-6

    assert len(repo.records) == 1
    stored_input = json.loads(repo.records[0]["input_json"])
    assert stored_input["Blood Ammonia"] == pytest.approx(valid_payload["Blood Ammonia"])
    stored_scores = json.loads(repo.records[0]["model_scores_json"])
    assert "GBM" in stored_scores

    gbm_seen = artifacts.models["GBM"].seen
    assert gbm_seen is not None
    assert gbm_seen.columns.tolist() == model_features
    assert gbm_seen["missing_feature"].iloc[0] == 0.0


def test_prediction_error_bubbles_up(valid_payload):
    val_cols = list(predict.NUMERIC_FIELDS.keys())
    cat_cols = list(predict.CATEGORICAL_FIELDS)
    artifacts = predict.ModelArtifacts(
        models={"BadModel": FailingModel()},
        scaler=IdentityScaler(),
        encoder=IdentityEncoder(),
        val_cols=val_cols,
        cat_cols=cat_cols,
        model_features=val_cols + cat_cols,
    )

    with pytest.raises(predict.PredictionError):
        predict.run_prediction(valid_payload, artifacts=artifacts)


def test_classify_risk_thresholds():
    assert predict.classify_risk(0.1) == "low"
    assert predict.classify_risk(0.5) == "medium"
    assert predict.classify_risk(0.9) == "high"


def test_predict_probability_branches(valid_payload):
    features = pd.DataFrame([valid_payload])
    with pytest.raises(predict.PredictionError):
        predict._predict_probability(BadShapeProbaModel(), features)

    with pytest.raises(predict.PredictionError):
        predict._predict_probability(EmptyPredictModel(), features)

    class NoMethods:
        pass

    with pytest.raises(predict.PredictionError):
        predict._predict_probability(NoMethods(), features)


def test_predict_scores_empty_models(valid_payload):
    artifacts = predict.ModelArtifacts(
        models={},
        scaler=IdentityScaler(),
        encoder=IdentityEncoder(),
        val_cols=list(predict.NUMERIC_FIELDS.keys()),
        cat_cols=list(predict.CATEGORICAL_FIELDS),
        model_features=list(predict.NUMERIC_FIELDS.keys()) + list(predict.CATEGORICAL_FIELDS),
    )
    with pytest.raises(predict.PredictionError):
        predict.predict_scores(pd.DataFrame([valid_payload]), artifacts)


def test_persist_prediction_failure(valid_payload):
    class FailingRepo:
        def create_prediction(self, *args, **kwargs):
            raise RuntimeError("db down")

    artifacts = predict.ModelArtifacts(
        models={"RF": ProbaModel(0.2)},
        scaler=IdentityScaler(),
        encoder=IdentityEncoder(),
        val_cols=list(predict.NUMERIC_FIELDS.keys()),
        cat_cols=list(predict.CATEGORICAL_FIELDS),
        model_features=list(predict.NUMERIC_FIELDS.keys()) + list(predict.CATEGORICAL_FIELDS),
    )

    with pytest.raises(RuntimeError):
        predict.run_prediction(valid_payload, artifacts=artifacts, repo=FailingRepo(), user_id=1)
