"""Streamlit entry point with minimal authentication gate."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import streamlit as st
except ImportError:  # pragma: no cover - exercised via stub in tests
    st = None

from app import predict
from app.auth import bootstrap_user, init_session_state, is_authenticated, login_user, logout
from app.db import init_db
from app.repo import PredictionRepository, UserRepository

DATA_PATH = Path(os.getenv("APP_DB_PATH", "data/app.db"))
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _get_streamlit(st_module=None):
    module = st_module or st
    if module is None:
        raise RuntimeError("Streamlit is required but not installed")
    return module


def get_user_repo(db_path: Path | str = DATA_PATH) -> UserRepository:
    db = init_db(db_path, SCHEMA_PATH)
    return UserRepository(db)


def get_repositories(
    db_path: Path | str = DATA_PATH,
) -> tuple[UserRepository, PredictionRepository]:
    db = init_db(db_path, SCHEMA_PATH)
    return UserRepository(db), PredictionRepository(db)


def render_login(user_repo: UserRepository, st_module=None) -> bool:
    streamlit = _get_streamlit(st_module)
    init_session_state(streamlit.session_state)
    bootstrap_user(user_repo)

    if is_authenticated(streamlit.session_state):
        streamlit.success(f"已登录: {streamlit.session_state['user']}")
        if streamlit.button("退出登录"):
            logout(streamlit.session_state)
            streamlit.rerun()  # 触发页面重新加载
        return True

    streamlit.subheader("用户登录")
    username = streamlit.text_input("用户名")
    password = streamlit.text_input("密码", type="password")
    if streamlit.button("登录"):
        if login_user(username, password, streamlit.session_state, user_repo):
            streamlit.success("登录成功")
            streamlit.rerun()  # 触发页面重新加载
        else:
            streamlit.error("用户名或密码错误")
    return False


def require_auth(user_repo: UserRepository, st_module=None) -> bool:
    streamlit = _get_streamlit(st_module)
    authenticated = render_login(user_repo, streamlit)
    if authenticated:
        return True
    streamlit.stop()
    return False


def _get_user_id(session_state, user_repo: UserRepository) -> int:
    username = session_state.get("user")
    if not username:
        raise RuntimeError("未找到登录用户信息")
    user = user_repo.get_user_by_username(username)
    if user is None:
        raise RuntimeError("用户不存在或尚未初始化")
    return user.id


def _render_prediction_form(streamlit):
    streamlit.subheader("输入患者临床参数")
    col1, col2 = streamlit.columns(2)

    inputs = {
        "Blood Ammonia": col1.number_input("Blood Ammonia (血氨)", min_value=0.0, value=50.0),
        "Albumin": col2.number_input("Albumin (白蛋白)", min_value=0.0, value=35.0),
        "Tips": col1.selectbox("Tips", options=[0, 1], format_func=lambda v: f"{v}"),
        "HBV": col2.selectbox("HBV 感染", options=[0, 1], format_func=lambda v: f"{v}"),
        "Splenomegaly": col1.selectbox("Splenomegaly (脾肿大)", options=[0, 1], format_func=lambda v: f"{v}"),
        "History of Hepatic Encephalopathy": col2.selectbox(
            "肝性脑病史", options=[0, 1], format_func=lambda v: f"{v}"
        ),
    }
    return inputs


def render_prediction_ui(user_repo: UserRepository, pred_repo: PredictionRepository, st_module=None) -> None:
    streamlit = _get_streamlit(st_module)
    streamlit.header("CHE 风险预测")

    try:
        artifacts = predict.get_artifacts()
    except predict.ModelLoadError as exc:
        streamlit.error(f"模型加载失败: {exc}")
        return

    inputs = _render_prediction_form(streamlit)
    if not streamlit.button("开始预测"):
        return

    try:
        user_id = _get_user_id(streamlit.session_state, user_repo)
        result = predict.run_prediction(inputs, artifacts=artifacts, repo=pred_repo, user_id=user_id)
    except predict.InputValidationError as exc:
        streamlit.error(f"输入无效: {exc}")
        return
    except predict.ModelLoadError as exc:
        streamlit.error(f"模型加载失败: {exc}")
        return
    except predict.PredictionError as exc:
        streamlit.error(f"推理失败: {exc}")
        return

    streamlit.success(f"最高风险模型: {result.top_model}")
    streamlit.metric("风险概率", f"{result.top_score * 100:.1f}%", delta=result.risk_level)
    streamlit.caption("模型概率均为 0-1 之间的值, 风险等级依据 30%/70% 阈值判定。")

    scores_table = [{"模型": name, "概率": f"{score*100:.2f}%"} for name, score in result.scores.items()]
    streamlit.table(scores_table)


def main() -> None:
    user_repo, pred_repo = get_repositories()
    if not require_auth(user_repo):
        return

    streamlit = _get_streamlit()
    render_prediction_ui(user_repo, pred_repo, st_module=streamlit)


if __name__ == "__main__":
    main()
