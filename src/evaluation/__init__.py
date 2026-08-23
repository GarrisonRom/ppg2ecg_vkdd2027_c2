"""评估工具模块。

子模块:
  - metrics: 多层评估指标体系 (波形/生理/分布/泛化对比)
  - visualize: 结果可视化
"""

from .metrics import (
    aggregate_by_subject,
    evaluate_all,
    evaluate_distribution,
    evaluate_waveform,
    hr_transfer_slope,
    ood_gap,
)
from .visualize import plot_ecg_comparison, plot_training_curves

__all__ = [
    "evaluate_all",
    "evaluate_waveform",
    "evaluate_distribution",
    "aggregate_by_subject",
    "hr_transfer_slope",
    "ood_gap",
    "plot_ecg_comparison",
    "plot_training_curves",
]
