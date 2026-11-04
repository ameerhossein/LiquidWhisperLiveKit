#!/usr/bin/env python3
"""`wlk` launcher with decoding tuned for BuzzASR/persian.

Why this exists
---------------
WhisperLiveKit hardcodes `beam_size=5, condition_on_previous_text=True` and
leaves faster-whisper's temperature fallback at [0.0, 0.2, ..., 1.0]. Two
problems show up with this checkpoint on streaming (partial) buffers:

1. Temperature fallback fires on nearly every update: partial utterances decode
   as repetitive text, the compression-ratio check fails, and faster-whisper
   re-decodes the segment up to six more times. Measured on a 5 s buffer:
   102 s with fallback vs 30 s at a single temperature.
2. Beam 5 latches onto repetition loops ("...به داخل آب رفت و به داخل آب رفت")
   that greedy decoding with the model-card penalties does not.

The model card recommends `num_beams=1, no_repeat_ngram_size=3,
repetition_penalty=1.2`; on a clean 4 s clip this gives byte-identical text to
beam 5 and on a mid-sentence slice it is 22 s (coherent) vs 38 s (looping).

So: single temperature, greedy decoding, model-card penalties. Everything else
(backend, VAD, timestamps, word timings) is untouched WhisperLiveKit.

Environment knobs:
  WLK_BEAM=1                  beam size (1 = model card; 5 = WhisperLiveKit default)
  WLK_TEMPERATURE=0.0         "" restores faster-whisper's fallback list
  WLK_NO_REPEAT_NGRAM=3       0 disables
  WLK_REPETITION_PENALTY=1.2  1.0 disables
  WLK_CONDITION_ON_PREVIOUS=0 reuse the previous window's tokens (WhisperLiveKit: 1)
  WLK_USE_PROMPT=0            pass committed text as initial_prompt (WhisperLiveKit: 1)
  WLK_NO_SPEECH_THRESHOLD=0.8 skip segments the model itself marks as silence
  WLK_VAD_FILTER=0            let faster-whisper strip silence inside the buffer
  WLK_VAC=0                   upstream pause splitting (1 enables it)

All CLI arguments are passed through to `wlk serve`.
"""
import os
import sys
import logging
import time
from contextvars import ContextVar

import whisperlivekit.local_agreement.backends as backends
import whisperlivekit.local_agreement.whisper_online as whisper_online

from settings import decode_options, model_options, print_settings, require_model

_SESSION_CONTEXT = ContextVar("buzzasr_session_context", default=None)


def _decode_options():
    return decode_options()


class _TunedFasterWhisperASR(backends.FasterWhisperASR):
    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel
        from whisperlivekit.model_paths import resolve_model_path
        runtime = model_options()
        reference = str(require_model(resolve_model_path(model_dir))) if model_dir else model_size
        if not reference:
            raise ValueError("A model directory or model size is required")
        print_settings(decode_options(), runtime)
        return WhisperModel(reference, download_root=cache_dir, **runtime)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        default_prompt = "1" if os.environ.get("WLK_PROFILE", "tuned") == "baseline" else "0"
        self.use_prompt = os.environ.get("WLK_USE_PROMPT", default_prompt) != "0"
        self.decode_options = _decode_options()
        self.transcription_calls = 0

    def transcribe(self, audio, init_prompt=""):
        kwargs = dict(self.decode_options)
        kwargs.update(self.transcribe_kargs)
        kwargs.update(language=self.original_language, word_timestamps=True,
                      initial_prompt=(init_prompt if self.use_prompt else
                                      _SESSION_CONTEXT.get()) or None)
        started = time.perf_counter()
        segments, _info = self.model.transcribe(audio, **kwargs)
        result = list(segments)
        self.transcription_calls += 1
        logging.getLogger(__name__).info(
            "BuzzASR inference call=%d audio_s=%.3f elapsed_s=%.3f",
            self.transcription_calls, len(audio) / 16000, time.perf_counter() - started)
        return result


def _install_session_context():
    from whisperlivekit.session_asr_proxy import SessionASRProxy
    original = SessionASRProxy.transcribe
    if getattr(original, "_buzzasr_context", False):
        return

    def transcribe(proxy, audio, init_prompt=""):
        # Preserve the upstream lock and language isolation. ContextVar is scoped
        # to this call/thread, so simultaneous sessions cannot share terminology.
        token = _SESSION_CONTEXT.set(proxy._session_context)
        try:
            return original(proxy, audio, init_prompt=init_prompt)
        finally:
            _SESSION_CONTEXT.reset(token)
    transcribe._buzzasr_context = True
    SessionASRProxy.transcribe = transcribe


_install_session_context()


def _install_persian_ui():
    import whisperlivekit
    original = whisperlivekit.get_inline_ui_html

    def html():
        # plaintext chooses direction independently for each transcript paragraph;
        # metadata retains the surrounding UI's LTR direction.
        content = original().replace("class='textcontent'", "class='textcontent' dir='auto'")
        content = content.replace('class="translation_text"', 'class="translation_text" dir="auto"')
        return content.replace("</head>", """<style>
.textcontent, .translation_text { unicode-bidi: plaintext; text-align: start; }
#speaker, #timeInfo { direction: ltr; unicode-bidi: isolate; }
</style></head>""")
    whisperlivekit.get_inline_ui_html = html


_install_persian_ui()


# backend_factory does `asr_cls = FasterWhisperASR` from its own module globals,
# so patch both the definition module and the importing module.
backends.FasterWhisperASR = _TunedFasterWhisperASR
whisper_online.FasterWhisperASR = _TunedFasterWhisperASR

from whisperlivekit.cli import main  # noqa: E402  (must come after the patch)


def configure_server_args(argv):
    # Pause splitting can force two passes on a subsecond fragment even when
    # coalescing is enabled. This checkpoint hallucinated a whole paragraph on
    # a 0.67 s fragment in the acceptance clip. Preserve continuous audio by
    # default; users can explicitly compare upstream VAC with WLK_VAC=1.
    default_vac = "1" if os.environ.get("WLK_PROFILE", "tuned") == "baseline" else "0"
    argv = list(argv)
    if argv[1:2] == ["serve"] and os.environ.get("WLK_VAC", default_vac) != "1":
        if "--no-vac" not in argv:
            argv.append("--no-vac")
    return argv


if __name__ == "__main__":
    sys.argv = configure_server_args(sys.argv)
    sys.exit(main())
