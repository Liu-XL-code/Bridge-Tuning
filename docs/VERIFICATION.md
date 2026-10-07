# Verification scope

The minimal CT release passed 8 unit tests, loading of all 3 ABC YAML profiles,
and --help startup checks for all 9 pipeline command entry points on 2026-10-07.

The CT FFT/LoRA model, preprocessing, stopping, strict checkpoint loading and native
inverse-inference paths derive from the implementation checked using unit tests and
short two-stage training on locally generated synthetic volumes. Earlier checks
included feature_size=12/ROI64 and feature_size=48/ROI96 configurations. These checks
do not establish clinical performance or reproduce the full 200/300-epoch study.

This minimal release removes comparison strategies and unrelated MRI/synthetic
configuration files. Its unit tests generate disposable inputs in temporary folders;
no fixture volumes, clinical data, run results or checkpoints are distributed.

Run `python -m unittest discover -s tests -v` in the installed environment.
Run `python verify_files.py` to check distributed source hashes without importing
model dependencies. SHA256SUMS.txt excludes itself and its equivalent source manifest.

Real inference requires a compatible locally trained target checkpoint. Random
initialization is a software-debugging option, not the proposed pretrained method.
