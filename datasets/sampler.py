"""沿用 AlphaMaster 的 Qlib 窗口采样，额外保留 JKP 列组。"""
import numpy as np
from qlib.data.dataset import TSDataSampler


class WindowSampler(TSDataSampler):
    """四组数据一起采样和填充，市场历史随股票历史行取样。"""

    def __init__(self, data, start, end, step_len,
                 fillna_type="ffill+bfill", dtype="float32"):
        """保留字段和分组切片，然后交给原生 Qlib sampler。"""
        if data.index.names != ["datetime", "instrument"] or not data.index.is_unique:
            raise ValueError("Expected unique (datetime, instrument) keys")
        if data.columns.nlevels != 2 or not data.columns.is_unique:
            raise ValueError("Expected unique grouped columns")
        self.columns = data.columns.copy()
        self.group_dims = {group: len(data[group].columns) for group in ("feature", "prior", "market", "label")}
        self.group_slices = {}
        offset = 0
        # 四组连续排列；去掉 prior 后，数组列顺序与 AlphaMaster 相同。
        for group, width in self.group_dims.items():
            self.group_slices[group] = slice(offset, offset + width)
            offset += width
        expected = [group for group, width in self.group_dims.items() for _ in range(width)]
        if list(self.columns.get_level_values(0)) != expected or not all(self.group_dims.values()):
            raise ValueError("Expected ordered feature, prior, market, label groups")
        # 原生 Qlib 对全部列使用同一行索引；不再单独覆盖市场历史。
        # Qlib 会消费传入的 DataFrame，以减少内存占用。
        super().__init__(data, start=start, end=end, step_len=step_len,
                         fillna_type=fillna_type, dtype=np.dtype(dtype))
        if not len(self):
            raise ValueError("Empty split")
