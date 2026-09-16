"""按样本终点日期组织 batch，参考 AlphaMaster 分组和 HVQ 加载方式。"""
import pickle

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Sampler


class DailyBatchSampler(Sampler):
    """每批给出一个交易日的全部股票位置，训练时只打乱日期顺序。"""

    def __init__(self, dataset, shuffle=False):
        """按 (日期, 股票代码) 排序并预先分组，不改变数据集底层顺序。"""
        self.shuffle = shuffle
        index = dataset.get_index()
        if index.names != ["datetime", "instrument"] or not index.is_unique:
            raise ValueError("Expected unique (datetime, instrument) keys")
        # 不假设数据按日期连续存放：先计算每个日期的真实位置。
        positions = pd.Series(np.arange(len(index)), index=index).sort_index()
        # 每组保存原数据集的位置，而非重新编号后的排序位置。
        self.daily_indices = [group.to_numpy() for _, group in positions.groupby(level="datetime", sort=True)]

    def __iter__(self):
        """依次产生同日股票的位置列表；股票、历史时间和字段顺序不打乱。"""
        order = np.arange(len(self.daily_indices))
        if self.shuffle:
            np.random.shuffle(order)
        for i in order:
            yield self.daily_indices[i].tolist()

    def __len__(self):
        """返回有样本的交易日数量，即每轮的 batch 数。"""
        return len(self.daily_indices)

    def ordered_indices(self):
        """验证/预测时与 batch 输出顺序一致的样本位置。"""
        if self.shuffle:
            raise ValueError("ordered_indices requires shuffle=False")
        # 拼接顺序与非随机加载一致，用它对齐逐批拼接的预测结果。
        return np.concatenate(self.daily_indices) if self.daily_indices else np.empty(0, dtype=int)


def load_dataset(path):
    """加载预生成的 sampler；Python 模块路径中需包含本项目。"""
    with open(path, "rb") as stream:
        return pickle.load(stream)


def init_data_loader(dataset, shuffle=False, num_workers=0):
    """创建按日 DataLoader，返回 [当天股票数, 窗口长度, 字段数] 张量。"""
    sampler = DailyBatchSampler(dataset, shuffle=shuffle)
    # batch_sampler 决定整日股票批次；无需设置固定 batch_size 或再加一层 batch。
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
