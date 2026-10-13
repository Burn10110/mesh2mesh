"""User-facing text in two languages (ru, en).

The key is the Russian source text as written in the code; English lives in i18n_en.EN.
Russian output therefore stays byte-for-byte what it was. Placeholders use str.format
names: tr("STL: заявлено {n} треугольников", n=n).

Language: set_lang() (CLI --lang, window switch) > env M2M_LANG > Windows UI language
(Russian -> ru, anything else -> en). Module-level text that must be translated later
is marked with N_() and passed through tr() at the point of use.
"""

import locale
import os
import sys

LANGS = ("ru", "en")
_lang = None


def _system_lang():
    if sys.platform == "win32":
        try:
            import ctypes
            lid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            return "ru" if (lid & 0x3FF) == 0x19 else "en"   # LANG_RUSSIAN
        except (AttributeError, OSError):
            pass
    try:
        loc = locale.getlocale()[0] or ""
    except ValueError:
        loc = ""
    return "ru" if loc.lower().startswith(("ru", "russian")) else "en"


def normalize(lang):
    """'en', 'EN', 'en_US' -> 'en'; unknown -> None."""
    lang = (lang or "").strip().lower()[:2]
    return lang if lang in LANGS else None


def get_lang():
    global _lang
    if _lang is None:
        _lang = normalize(os.environ.get("M2M_LANG")) or _system_lang()
    return _lang


def set_lang(lang):
    """Set the language ('ru'/'en'); None or unknown -> back to env/system default."""
    global _lang
    _lang = normalize(lang)
    return get_lang()


def tr(text, **kw):
    if get_lang() == "en":
        from .i18n_en import EN
        text = EN.get(text, text)
    return text.format(**kw) if kw else text


def N_(text):
    """Mark text for translation without translating it now (tr() it where it is shown)."""
    return text


def pre_parse_lang(argv):
    """--lang X / --lang=X from argv before argparse runs (so --help is translated).
    Returns the value or None; argv is not changed."""
    for i, a in enumerate(argv):
        if a == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--lang="):
            return a.split("=", 1)[1]
    return None
