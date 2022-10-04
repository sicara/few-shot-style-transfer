from pathlib import Path
import pytest


@pytest.fixture()
def path_to_resources():
    return Path("src/tests/resources")
