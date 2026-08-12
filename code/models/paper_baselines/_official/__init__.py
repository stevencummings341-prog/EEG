"""Verbatim copies of the baselines' official released code.

Only ``EEGDeformer.py`` is importable and executed — it is already PyTorch.
``EEGModels_arl_keras.py`` and ``BenchmarkModels_eegnex_keras.py`` are Keras/TensorFlow and
are kept **for line-by-line verification only**; their PyTorch transcriptions live one level
up (``eegnet_official_torch.py`` / ``eegnex_official_torch.py``). Do not import the Keras
files: TensorFlow is not (and should not be) installed in the ``mi_torch_cu118`` env.
"""
