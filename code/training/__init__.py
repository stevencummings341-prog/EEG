"""训练子包：训练循环、损失、优化器、checkpoint。

每次 run 必须保存：resolved config / split / checkpoint / metrics / logs，
统一放在 outputs/<run_id>/ 与 checkpoints/<run_id>/。见 20-model-training 规则。
"""
