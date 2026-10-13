import pytest

from mesh2mesh import i18n


@pytest.fixture(autouse=True)
def _russian_by_default():
    """Tests check Russian text: run them in ru whatever the Windows language is."""
    i18n.set_lang("ru")
    yield
    i18n.set_lang("ru")


@pytest.fixture
def english():
    i18n.set_lang("en")
    yield
    i18n.set_lang("ru")
