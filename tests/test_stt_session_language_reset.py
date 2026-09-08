"""A finished conversation's detected language must not reach the next one.

Pipeline units are reused across clients: `websocket_router._clean_unit` flushes the queues
so pending work "cannot be picked up by handlers and leak into the next session that claims
this unit", and `SESSION_END` is "the soft reset signal for stateful handlers".

`last_language` is exactly that kind of state. Backends consult it when detection yields
nothing, and Qwen3-ASR forces progressive requests with it, so leaving it set transcribes a
new client in the previous client's language. Only Parakeet reset it; the reset now lives on
`BaseSTTHandler` so it covers every backend, including future ones.
"""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import ExitStack
from unittest import mock

import pytest

from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

# (module, class, extra attributes process()/on_session_end() needs)
_STT_HANDLERS = [
    ("speech_to_speech.STT.parakeet_tdt_handler", "ParakeetTDTSTTHandler", {"enable_live_transcription": False}),
    ("speech_to_speech.STT.whisper_stt_handler", "WhisperSTTHandler", {}),
    ("speech_to_speech.STT.mlx_audio_whisper_handler", "MLXAudioWhisperSTTHandler", {}),
    ("speech_to_speech.STT.lightning_whisper_mlx_handler", "LightningWhisperSTTHandler", {}),
    ("speech_to_speech.STT.faster_whisper_handler", "FasterWhisperSTTHandler", {}),
    ("speech_to_speech.STT.qwen3_asr_handler", "Qwen3ASRSTTHandler", {}),
]

# Optional third-party imports stubbed so every backend stays checkable without its extra.
_STUBS = {
    "speech_to_speech.STT.faster_whisper_handler": ("faster_whisper", "WhisperModel"),
    "speech_to_speech.STT.lightning_whisper_mlx_handler": ("lightning_whisper_mlx", "LightningWhisperMLX"),
}


def _load(module_name: str, class_name: str):
    stub = _STUBS.get(module_name)
    with ExitStack() as stack:
        if stub is not None and stub[0] not in sys.modules:
            package, attribute = stub
            module = types.ModuleType(package)
            setattr(module, attribute, object)
            stack.enter_context(mock.patch.dict(sys.modules, {package: module}))
            stack.callback(sys.modules.pop, module_name, None)
        try:
            return getattr(importlib.import_module(module_name), class_name)
        except ImportError:
            return None


def _handler(cls, extra, *, start_language, last_language):
    handler = object.__new__(cls)
    handler.start_language = start_language
    handler.last_language = last_language
    for key, value in extra.items():
        setattr(handler, key, value)
    return handler


@pytest.mark.parametrize(("module_name", "class_name", "extra"), _STT_HANDLERS)
def test_detected_language_does_not_survive_the_session(module_name, class_name, extra):
    """In auto mode the next client must start from "detect", not the last detection."""
    cls = _load(module_name, class_name)
    if cls is None:
        pytest.skip(f"{module_name} requires an optional dependency")

    handler = _handler(cls, extra, start_language=None, last_language="de")

    handler.on_session_end()

    assert handler.last_language != "de", (
        f"{class_name} carried the previous session's detected language into the next one"
    )


@pytest.mark.parametrize(("module_name", "class_name", "extra"), _STT_HANDLERS)
def test_configured_language_survives_the_session(module_name, class_name, extra):
    """A user-configured language is process-level config, not per-conversation state."""
    cls = _load(module_name, class_name)
    if cls is None:
        pytest.skip(f"{module_name} requires an optional dependency")

    handler = _handler(cls, extra, start_language="de", last_language="fr")

    handler.on_session_end()

    assert handler.last_language == "de"


def test_parakeet_keeps_its_english_fallback():
    """Parakeet already reset, with its own default. That behaviour is unchanged."""
    cls = _load("speech_to_speech.STT.parakeet_tdt_handler", "ParakeetTDTSTTHandler")
    handler = _handler(cls, {"enable_live_transcription": False}, start_language=None, last_language="de")

    handler.on_session_end()

    assert handler.last_language == "en"


# --- the base-class contract --------------------------------------------------------------


def test_base_handler_declares_the_language_attributes():
    """Declared on the base so the reset works for a backend that never sets them."""
    assert BaseSTTHandler.start_language is None
    assert BaseSTTHandler.last_language is None


def test_reset_is_safe_for_a_handler_that_never_set_a_language():
    class Bare(BaseSTTHandler):
        pass

    handler = object.__new__(Bare)

    handler.on_session_end()  # must not raise

    assert handler.last_language is None


def test_reset_still_clears_revision_bookkeeping():
    """The pre-existing responsibility of on_session_end must be preserved."""

    class Bare(BaseSTTHandler):
        pass

    handler = object.__new__(Bare)
    handler._completed_final_revision_keys = {"turn_1:0": None}

    handler.on_session_end()

    assert handler._completed_final_revision_keys == {}


def test_reset_session_language_is_callable_on_its_own():
    """Exposed separately so a backend can reuse it without overriding on_session_end."""

    class Bare(BaseSTTHandler):
        pass

    handler = object.__new__(Bare)
    handler.start_language = "fr"
    handler.last_language = "ja"

    handler.reset_session_language()

    assert handler.last_language == "fr"
