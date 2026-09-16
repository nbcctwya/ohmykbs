"""个股、JKP 和市场处理器，参考 AlphaMaster 与 HVQ 的实际实现。"""
import copy

import numpy as np
import pandas as pd
from qlib.contrib.data.handler import Alpha158, check_transform_proc
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor


class GlobalFactorMerger(Processor):
    """与 HVQ 一致，将每日 JKP 因子广播给同日所有股票。"""

    def __init__(self, factors):
        """保存以交易日期为索引、以因子名称为列的每日因子矩阵。"""
        self.factors = factors

    def __call__(self, frame):
        """按股票行的日期重复取出因子，并追加到 prior 列组。"""
        dates = frame.index.get_level_values("datetime")
        # dates 可以重复：同一天不同股票会取到相同的 13 个因子。
        prior = pd.DataFrame(self.factors.loc[dates].to_numpy(), index=frame.index,
            columns=pd.MultiIndex.from_product([["prior"], self.factors.columns]))
        return pd.concat([frame, prior], axis=1)


class Alpha158WithJKP(Alpha158):
    """原生 Alpha158 加上独立归一化的 JKP 先验。"""

    def __init__(self, factors, **kwargs):
        """先合并 JKP，再执行 YAML 中的个股和先验处理器。"""
        if factors.shape[1] != 13:
            raise ValueError("Expected exactly 13 JKP factors")
        # 合并必须在 prior 标准化之前；深拷贝避免 Qlib 修改原配置。
        kwargs["infer_processors"] = [GlobalFactorMerger(factors)] + copy.deepcopy(kwargs["infer_processors"])
        super().__init__(**kwargs)


class Alpha158USWithJKP(Alpha158WithJKP):
    """沿用 AlphaMaster 的美股 VWAP 适配，保留 VWAP0 列名。"""

    def get_feature_config(self):
        """只替换引用 VWAP 的表达式，其他公式和字段顺序不变。"""
        fields, names = super().get_feature_config()
        # 本地美股缺少 VWAP，使用 (最高价+最低价+收盘价)/3 的代理。
        fields = ["($high+$low+$close)/3/$close" if "$vwap" in field else field
                  for field in fields]
        return fields, names


def build_factor_matrix(raw, calendar, config):
    """筛选原始 JKP 收益，返回滞后一日的滚动累计收益矩阵。"""
    selected = raw.loc[(raw["location"] == config["location"])
                      & (raw["weighting"] == config["weighting"])
                      & (raw["freq"] == config["freq"])]
    # 长表转为“日期×因子”宽表；pivot 默认按因子名称排列列。
    returns = selected.pivot(index="date", columns="name", values="ret").sort_index()
    # 先对齐再 shift，滞后单位是 Qlib 交易日，当前日收益不会进入当日先验。
    returns = returns.reindex(calendar).shift(1)
    window = config["window"]
    # 等价于连续 window 日的 (1+r) 连乘减一；历史不足时保留 NaN。
    factors = np.expm1(np.log1p(returns).rolling(window, min_periods=window).sum())
    factors.columns = [f"JKP_{name}_RET{window}D" for name in factors.columns]
    if factors.shape[1] != 13:
        raise ValueError("Expected exactly 13 JKP factors")
    return factors


def market_feature_config(indices, windows):
    """按参考项目的顺序生成指数收益、收益波动和成交量表达式。"""
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Market indices must be nonempty and unique")
    if not windows or any(type(w) is not int or w <= 0 for w in windows) or len(set(windows)) != len(windows):
        raise ValueError("Market windows must be unique positive integers")
    # 每指数先输出当日收益，再对每个窗口输出四个指标：共 1+4×5=21 维。
    exprs = ["$close/Ref($close,1)-1"]
    for w in windows:
        exprs.extend([
            f"Mean($close/Ref($close,1)-1,{w})",
            f"Std($close/Ref($close,1)-1,{w})",
            f"Mean($volume,{w})/$volume",
            f"Std($volume,{w})/$volume",
        ])
    # Mask 强制读取指定指数；三个指数按配置顺序拼成 63 维。
    fields = [f'Mask({expr}, "{inst}")' for inst in indices for expr in exprs]
    return fields, list(fields)


class ValidateMarket(Processor):
    """在填零之前识别完全缺失的市场字段。"""

    def __call__(self, frame):
        """整列缺失时立即报错，避免 Fillna 掩盖缺少指数数据的问题。"""
        missing = frame.columns[frame.isna().all()]
        if len(missing):
            raise ValueError(f"Unavailable market expressions: {list(missing)}")
        return frame


class MarketDataHandler(DataHandlerLP):
    """独立处理市场特征，按参考项目的股票池展开行拟合标准化。"""

    def __init__(self, market_indices, windows, fit_start_time, fit_end_time,
                 infer_processors, **kwargs):
        """创建日频市场 loader，并为处理器注入标准化拟合日期。"""
        processors = check_transform_proc(copy.deepcopy(infer_processors),
                                          fit_start_time, fit_end_time)
        # 市场信息独立于个股标准化；不执行标签过滤或标签排名。
        super().__init__(
            data_loader={"class": "QlibDataLoader", "kwargs": {
                "config": {"market": market_feature_config(market_indices, windows)},
                "freq": "day",
            }},
            infer_processors=[ValidateMarket()] + processors,
            learn_processors=[], process_type=DataHandlerLP.PTYPE_A, **kwargs,
        )


def daily_market_frame(handler):
    """检查同日股票的市场值一致后，压缩为每日一行的市场矩阵。"""
    frame = handler.fetch(col_set=["market"], data_key=DataHandlerLP.DK_I)
    grouped = frame.groupby(level="datetime", sort=True)
    # NaN 也作为一种取值比较，不能把“有值/缺失”的不一致忽略掉。
    if (grouped.nunique(dropna=False) > 1).any().any():
        raise ValueError("Market values differ across stocks on the same date")
    daily = grouped.first()
    if not np.isfinite(daily.to_numpy()).all():
        raise ValueError("Processed market values must be finite")
    return daily
