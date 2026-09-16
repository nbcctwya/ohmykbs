"""读取数据配置，构建并保存 train/valid/test；不依赖模型。"""
import argparse
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.data import D

from .handlers import Alpha158WithJKP, Alpha158USWithJKP, MarketDataHandler, build_factor_matrix, daily_market_frame
from .sampler import WindowSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    """读取数据 YAML，检查窗口、日期划分和标准化拟合区间。"""
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    settings = config["dataset"]
    if type(config["jkp_config"]["window"]) is not int or config["jkp_config"]["window"] <= 0:
        raise ValueError("JKP window must be a positive integer")
    if type(settings["step_len"]) is not int or settings["step_len"] <= 0:
        raise ValueError("step_len must be a positive integer")
    if settings["dtype"] not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if settings["fillna_type"] not in ("none", "ffill", "ffill+bfill"):
        raise ValueError("Unsupported fillna_type")
    previous_end = None
    # 这里只检查样本日期不重叠；没有剔除未来标签终点跨界的样本。
    for split in ("train", "valid", "test"):
        start, end = map(pd.Timestamp, settings["segments"][split])
        if start > end or (previous_end is not None and start <= previous_end):
            raise ValueError("Split dates must be ordered and nonoverlapping")
        if settings["split_data_keys"][split] not in ("learn", "infer"):
            raise ValueError("Split data keys must be learn or infer")
        previous_end = end
    train_start, train_end = map(pd.Timestamp, settings["segments"]["train"])
    # 标准化参数只允许在训练期拟合，验证和测试不能参与拟合。
    for key in ("data_handler_config", "market_data_handler_config"):
        params = config[key]
        fit_start, fit_end = map(pd.Timestamp, (params["fit_start_time"], params["fit_end_time"]))
        if not train_start <= fit_start <= fit_end <= train_end:
            raise ValueError("Normalization fit must lie within train dates")
    return config


def prepare_segment(handler, market, settings, split):
    """读取一个划分及其回看历史，拼接四组数据并创建固定窗口 sampler。"""
    start, end = map(pd.Timestamp, settings["segments"][split])
    calendar = pd.DatetimeIndex(D.calendar(start_time=handler.start_time, end_time=end))
    if not len(calendar):
        raise ValueError("No trading calendar for requested split")
    # 划分起点前多取 step_len 个交易日，让首日样本也能回看历史。
    lookback = calendar[max(0, calendar.searchsorted(start) - settings["step_len"])]
    # learn 已做标签过滤和排名；infer 保留原始标签，包括缺失值。
    # 沿用 AlphaMaster：learn 的历史行也已按单目标有效性过滤。
    stock = handler.fetch(selector=slice(lookback, end), col_set=["feature", "prior", "label"],
                          data_key=settings["split_data_keys"][split]).copy()
    dates = stock.index.get_level_values("datetime")
    # 按股票行日期广播市场信息，拼接时显式保证 feature/prior/market/label 顺序。
    broadcast = pd.DataFrame(market.loc[dates].to_numpy(), index=stock.index, columns=market.columns)
    frame = pd.concat([stock.loc[:, ["feature", "prior"]], broadcast, stock.loc[:, ["label"]]], axis=1)
    if [len(frame[group].columns) for group in ("feature", "prior", "market", "label")] != [158, 13, 63, 1]:
        raise ValueError("Expected AlphaMaster + JKP layout: 158 feature + 13 prior + 63 market + 1 label")
    if not np.isfinite(frame.loc[:, ["feature", "prior", "market"]].to_numpy()).all():
        raise ValueError("Processed input features must be finite")
    labels = frame["label"].to_numpy()
    # 测试目标缺失不能补零；学习目标则必须有效，无穷值两者都不允许。
    if np.isinf(labels).any() or (settings["split_data_keys"][split] == "learn" and np.isnan(labels).any()):
        raise ValueError("Invalid labels: only infer data may contain NaN")
    # 与 AlphaMaster 一样，市场按股票行日期拼接后一起采样。
    return WindowSampler(frame, start, end, settings["step_len"],
                         settings["fillna_type"], settings["dtype"])


def build_dataset(config_path):
    """独立构建三个划分，保存 sampler 和本次使用的配置快照。"""
    config = load_config(config_path)
    output = Path(config["output_dir"]).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    # 先独占输出目录，避免覆盖；后续失败可能留下部分文件。
    output.mkdir(parents=True, exist_ok=False)  # 不覆盖已有数据。
    (output / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    init = dict(config["qlib_init"])
    # 展开用户目录并将 Qlib 运行记录放进输出目录，不读取模型配置。
    init["provider_uri"] = str(Path(init["provider_uri"]).expanduser())
    init["exp_manager"] = {"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
        "kwargs": {"uri": f"file:{output.resolve() / 'qlib_runs'}", "default_exp_name": "data_only"}}
    qlib.init(**init)
    stock_config = config["data_handler_config"]
    jkp_path = Path(config["jkp_config"]["path"]).expanduser()
    if not jkp_path.is_absolute():
        jkp_path = PROJECT_ROOT / jkp_path
    raw = pd.read_csv(jkp_path, parse_dates=["date"])
    # 因子与个股使用相同加载区间的交易日历；因子计算在 handler 之前完成。
    calendar = D.calendar(start_time=stock_config["start_time"], end_time=stock_config["end_time"])
    factors = build_factor_matrix(raw, calendar, config["jkp_config"])
    handler_class = Alpha158USWithJKP if init["region"] == "us" else Alpha158WithJKP
    # 仅美股替换 VWAP；其余个股特征、JKP 和市场计算保持共同规则。
    handler = handler_class(factors, **stock_config)
    handler._data = None  # 参考 AlphaMaster：释放不再使用的原始数据副本。
    market_handler = MarketDataHandler(**config["market_data_handler_config"])
    market = daily_market_frame(market_handler)
    del market_handler
    gc.collect()

    for split in ("train", "valid", "test"):
        # 保存的是逐日数据、索引和采样规则，而不是全部展开后的重叠窗口。
        sampler = prepare_segment(handler, market, config["dataset"], split)
        path = output / f"{split}.pkl"
        with path.open("xb") as stream:
            pickle.dump(sampler, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{split}: {len(sampler)} samples, {sampler[0].shape} -> {path}", flush=True)
        del sampler
        gc.collect()
    return output


def main():
    """命令行入口：选择数据配置，或仅校验配置而不读取真实数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        load_config(args.config)
        print(f"Configuration valid: {args.config}")
    else:
        build_dataset(args.config)


if __name__ == "__main__":
    main()
