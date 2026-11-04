# Acceptance fixture

`verify-fa.wav` is the existing project warmup clip copied into a small tracked
fixture (16 kHz mono, 4.08 seconds). `verify-fa.txt` is its expected Persian text.
The acceptance script checks exact text after whitespace normalization and
noncollapsed word timestamps. This is a wiring/regression fixture, not a
representative accuracy benchmark or a manually aligned word-boundary reference.

The other diarization/audio fixtures in this directory predate this fix; their
historical comparison scores should be rerun after the driver corrections.
