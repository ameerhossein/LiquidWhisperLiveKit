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

All CLI arguments are passed through to `wlk serve`.
"""
import os
import sys

import whisperlivekit.local_agreement.backends as backends
import whisperlivekit.local_agreement.whisper_online as whisper_online


def _decode_options():
    opts = {}
    temperature = os.environ.get("WLK_TEMPERATURE", "0.0")
    if temperature not in ("", "none", "None"):
        opts["temperature"] = float(temperature)
    ngram = int(os.environ.get("WLK_NO_REPEAT_NGRAM", "3"))
    if ngram:
        opts["no_repeat_ngram_size"] = ngram
    penalty = float(os.environ.get("WLK_REPETITION_PENALTY", "1.2"))
    if penalty != 1.0:
        opts["repetition_penalty"] = penalty
    no_speech = os.environ.get("WLK_NO_SPEECH_THRESHOLD", "0.8")
    if no_speech not in ("", "none", "None"):
        opts["no_speech_threshold"] = float(no_speech)
    return opts


class _TunedFasterWhisperASR(backends.FasterWhisperASR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam_size = int(os.environ.get("WLK_BEAM", "1"))
        # Both default to OFF here: feeding a rolling buffer's own transcript hack
        # as the prompt makes this model amplify any repetition into fluent
        # nonsense, and it decodes the committed text twice (once as prompt, once
        # as audio). Observed: update 2 was garbage with prompts on, correct off.
        self.condition_on_previous = os.environ.get("WLK_CONDITION_ON_PREVIOUS", "0") != "0"
        self.use_prompt = os.environ.get("WLK_USE_PROMPT", "0") != "0"
        self.decode_options = _decode_options()

    def transcribe(self, audio, init_prompt=""):
        """Same contract as FasterWhisperASR.transcribe, tuned decoding."""
        kwargs = dict(
            language=self.original_language,
            initial_prompt=init_prompt if self.use_prompt else None,
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=self.condition_on_previous,
            **self.decode_options,
        )
        # e.g. vad_filter added by use_vad(); never let it collide with ours.
        for key, value in self.transcribe_kargs.items():
            kwargs.setdefault(key, value)
        segments, _info = self.model.transcribe(audio, **kwargs)
        return list(segments)


# backend_factory does `asr_cls = FasterWhisperASR` from its own module globals,
# so patch both the definition module and the importing module.
backends.FasterWhisperASR = _TunedFasterWhisperASR
whisper_online.FasterWhisperASR = _TunedFasterWhisperASR

from whisperlivekit.cli import main  # noqa: E402  (must come after the patch)

if __name__ == "__main__":
    sys.exit(main())
