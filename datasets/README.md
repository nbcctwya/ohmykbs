# 数据集构建

数据配置位于 `configs/data/`，独立于模型配置。数据口径沿用 AlphaMaster，仅额外增加 13 维 JKP：158 维 Alpha158、13 维 JKP、63 维市场特征和 1 列 5 日收益标签，使用 8 日窗口。中国股票使用原生 Alpha158；美股沿用 AlphaMaster 的 VWAP 适配，将 `$vwap/$close` 替换为 `($high+$low+$close)/3/$close`，保留 `VWAP0` 列名。

```text
datasets/
├── handlers.py   # Alpha158 + JKP、市场 handler
├── sampler.py    # 原生 Qlib 固定窗口采样
├── build.py      # 读取配置、构建三个划分并保存
└── loader.py     # 按日期组织训练、验证和预测 batch
```

构建流程：读取 YAML → 初始化 Qlib → 创建个股和市场 handler → 准备 train/valid/test → 保存 pickle 和配置快照。特征、标签处理、划分和窗口设置来自数据配置，不读取模型配置。

在 `ohmykbs` 目录使用 `kbs` 环境，只校验配置和运行模拟测试：

```bash
conda run -n kbs python -m datasets.build --config configs/data/csi300_dataset.yaml --validate-only
conda run -n kbs python -m datasets.build --config configs/data/sp500_dataset.yaml --validate-only
conda run -n kbs python -m unittest discover -s tests -v
```

正式构建时去掉 `--validate-only`。当前阶段没有执行真实构建。输出目录由 `output_dir` 指定，相对路径以 `ohmykbs` 为基准：

```text
datasets/processed/csi300/
├── config.yaml
├── train.pkl
├── valid.pkl
└── test.pkl
```

pickle 保存逐日数组、样本索引、字段和固定窗口规则，加载后通过 `sampler[index]` 取窗口，而不是提前展开所有重叠窗口。当前单样本为 `[8,235]`；`group_slices` 的顺序和切片为：

| 列组 | 切片 | 后续模型读取方式 |
|---|---|---|
| feature | 0:158 | 完整 8 日窗口 |
| prior | 158:171 | 窗口末日的 13 个因子 |
| market | 171:234 | 完整 8 日窗口 |
| label | 234:235 | 窗口末日的单个目标 |

未来标签不能作为模型输入。与 AlphaMaster 一样，市场信息先拼接到股票行，再由原生 Qlib 对所有列一起采样。股票缺少某日历史行时，市场历史也随该行一起填充。

JKP CSV 位于 `datasets/jkpdata/`，当前两份输入复制自 HVQ-Stock。`jkp_config` 指定路径、国家、权重、频率及滚动窗口。因子先对齐 Qlib 交易日历并滞后一日，再计算 `expm1(rolling_20_sum(log1p(ret)))`，按因子名称排序并广播给同日股票；先验独立执行 RobustZScoreNorm 和 Fillna。

目标表达式与 AlphaMaster 完全相同：`Ref($close, -5) / Ref($close, -1) - 1`，默认列名为 `LABEL0`，即 `close(t+5)/close(t+1)-1`。训练/验证再执行 CSRankNorm；测试保留原始收益。JKP 的 20 日滚动计算与输入的 8 日窗口相互独立。

训练/验证使用 `learn` 数据，测试使用 `infer` 数据并保留缺失标签。已有输出目录拒绝覆盖，失败构建可能留下部分文件。加载时需要本项目的 `datasets` 包。Qlib sampler 构造时会消费传入的 DataFrame，以减少内存占用。

当前保留原日期划分，没有追加跨边界标签过滤。测试仅使用模拟数据和 `init_data=False` 的 handler，真实数据构建及全量质量核验留待后续。

## 按日加载

在 `ohmykbs` 目录运行。构建文件后可以这样使用：

```python
from datasets.loader import load_dataset, init_data_loader

dataset = load_dataset("datasets/processed/csi300/train.pkl")
loader = init_data_loader(dataset, shuffle=True)
for batch in loader:
    # batch: [当天股票数, 8, 235]，float32 Tensor
    stock = batch[:, :, :158]
    prior = batch[:, -1, 158:171]
    market = batch[:, :, 171:234]
    targets = batch[:, -1, 234]
```

每个 batch 的样本终点日期相同。训练可打乱日期顺序，日期内部按股票代码排序；验证和预测使用 `shuffle=False`。没有固定的股票 batch size，也不丢弃股票数较少的日期。`len(loader)` 是交易日数量。

预测时索引需要跟随实际输出顺序：

```python
loader = init_data_loader(dataset, shuffle=False)
prediction_index = dataset.get_index()[loader.batch_sampler.ordered_indices()]
```

该索引对应依次拼接各 batch 预测结果的顺序，不能直接假设与 pickle 原始索引顺序一致。加载层只组织已有样本，不改变特征、标签和划分，也不修复已有跨界标签问题。`num_workers` 默认 0；需要多进程时再设置，采用 spawn 的脚本应使用 `if __name__ == "__main__":` 入口。
