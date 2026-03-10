import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked with @pytest.mark.phoebe if phoebe is not importable."""
    try:
        import phoebe  # noqa: F401
        return
    except ImportError:
        skip_phoebe = pytest.mark.skip(reason="PHOEBE not installed")
        for item in items:
            if "phoebe" in item.keywords:
                item.add_marker(skip_phoebe)
