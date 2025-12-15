import threading
from pathlib import Path

import pytest

from app.db import ConstraintError, DatabaseError, init_db
from app.repo import PredictionRepository, UserRepository


@pytest.fixture()
def schema_path() -> Path:
    return Path(__file__).resolve().parent.parent / "app" / "schema.sql"


@pytest.fixture()
def db(tmp_path, schema_path):
    db_path = tmp_path / "test.db"
    database = init_db(db_path, schema_path, pool_size=4)
    yield database
    database.close()


@pytest.fixture()
def repos(db):
    return UserRepository(db), PredictionRepository(db)


def test_init_db_missing_schema(tmp_path):
    bogus_schema = tmp_path / "missing.sql"
    with pytest.raises(DatabaseError):
        init_db(tmp_path / "test.db", bogus_schema)


def test_user_crud_flow(repos):
    user_repo, _ = repos
    user_id = user_repo.create_user("alice", "hash1")
    user = user_repo.get_user_by_username("alice")
    assert user is not None
    assert user.id == user_id
    assert user.password_hash == "hash1"

    assert user_repo.update_user_password("alice", "hash2") is True
    updated = user_repo.get_user_by_username("alice")
    assert updated is not None
    assert updated.password_hash == "hash2"

    assert user_repo.update_user_password("ghost", "nothing") is False


def test_duplicate_user_raises(repos):
    user_repo, _ = repos
    user_repo.create_user("bob", "pw")
    with pytest.raises(ConstraintError):
        user_repo.create_user("bob", "pw")


def test_prediction_pagination(repos):
    user_repo, pred_repo = repos
    user_id = user_repo.create_user("carol", "pw")
    for i in range(5):
        pred_repo.create_prediction(
            user_id=user_id,
            input_json=f'{{"i": {i}}}',
            model_scores_json=f'{{"m": {i}}}',
            top_model="rf",
        )

    first_page = pred_repo.list_predictions(user_id, limit=3, offset=0)
    second_page = pred_repo.list_predictions(user_id, limit=3, offset=3)

    assert len(first_page) == 3
    assert len(second_page) == 2
    assert first_page[0].id > first_page[-1].id  # ordered by id desc
    ids = {p.id for p in first_page + second_page}
    assert len(ids) == 5


def test_thread_safe_writes(repos):
    user_repo, pred_repo = repos
    user_id = user_repo.create_user("dave", "pw")

    total_threads = 5
    per_thread = 8
    barrier = threading.Barrier(total_threads)

    def worker(thread_idx: int) -> None:
        barrier.wait()
        for i in range(per_thread):
            pred_repo.create_prediction(
                user_id=user_id,
                input_json=f'{{"t": {thread_idx}, "i": {i}}}',
                model_scores_json='{"score": 0.5}',
                top_model="lgbm",
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(total_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    predictions = pred_repo.list_predictions(user_id, limit=1000)
    assert len(predictions) == total_threads * per_thread


def test_permission_error(tmp_path, schema_path):
    read_only_dir = tmp_path / "ro"
    read_only_dir.mkdir()
    read_only_dir.chmod(0o500)

    with pytest.raises(DatabaseError):
        init_db(read_only_dir / "db.sqlite", schema_path)


def test_pool_close_blocks_new_connections(db):
    db.close()
    with pytest.raises(DatabaseError):
        with db.connection():
            pass
