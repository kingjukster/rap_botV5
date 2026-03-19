"""
Pytest configuration. Disable DB for tests to avoid MySQL dependency.

lm_proposer calls load_dotenv(override=True), which overwrites RAPBOT_USE_DB
after collection. We set it early and re-apply before tests via fixture.
"""
import os

import pytest


def pytest_configure(config):
    """Disable DB before any evo_rhyme imports."""
    os.environ["RAPBOT_USE_DB"] = "0"


@pytest.fixture(autouse=True, scope="session")
def _disable_db_for_tests():
    """Re-apply RAPBOT_USE_DB=0 and clear db cache (lm_proposer may have overwritten)."""
    os.environ["RAPBOT_USE_DB"] = "0"
    try:
        import evo_rhyme.db as _db
        _db._DB_CONFIG = None
    except ImportError:
        pass
    yield
