# Prediction recorded before blind scoring

Recorded on 2026-08-23 before either critic was run on the target 420 videos.

Robometer-4B-LIBERO is expected to rank policies more closely to environment
success because it was trained with both progress and trajectory-preference
supervision, including LIBERO rollouts. The smaller critic saw only expert
LIBERO-90 videos with synthetic temporal progress labels, so it may confuse
motion or late trajectory time with task completion. The useful counter-case
is that the smaller critic uses the repository's exact 128x128 camera convention
and an identical 32-bin target, whereas Robometer may have a domain or camera
calibration mismatch.

This prediction fixes the expected winner but does not define any threshold or
permit checkpoint selection after observing ranking labels.
