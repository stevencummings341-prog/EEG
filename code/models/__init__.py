"""模型子包。

cross-session 比较框架（当前主线）用 4 个共享统一 dict 契约的模型：
  - eegnet      : EEGNetClassifier (Lawhern 2018) —— baseline
  - deepconvnet : DeepConvNet (Schirrmeister 2017) —— baseline
  - fbcnet      : FBCNet (Mane 2021) —— baseline
  - cap_eegnet  : CAPEEGNet v1（本项目模型：encoder + 分类头 + 学习型 confidence 头）

统一契约：forward(x[B,C,T] 或 [B,1,C,T]) -> {"logits", "features", "confidence"}。
用 registry.build_model(name, ...) 构造。CAP-EEGNet 的 full 组件（多神经子模块/多源
置信度/原型/adapter/域对齐/在线）尚未实现（future work），见 cap_eegnet.py 与 docs/ROADMAP.md。
"""

from .registry import MODEL_NAMES, build_model  # noqa: F401

