"""Synthetic checks only: never initialize a real Qlib provider."""
import copy
import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import TSDataSampler

from datasets.build import build_dataset, load_config, main
from datasets.handlers import Alpha158WithJKP, Alpha158USWithJKP, GlobalFactorMerger, MarketDataHandler, build_factor_matrix, daily_market_frame, market_feature_config
from datasets.sampler import WindowSampler
from datasets.loader import DailyBatchSampler, init_data_loader

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    dates = pd.bdate_range("2020-01-01", "2020-03-10")
    index = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["datetime", "instrument"])
    columns = pd.MultiIndex.from_tuples(
        [("feature", f"F{i}") for i in range(158)]
        + [("prior", f"P{i}") for i in range(13)]
        + [("label", "LABEL0")])
    stock = pd.DataFrame(np.arange(len(index) * 172).reshape(-1, 172).astype(float), index=index, columns=columns)
    stock = stock.drop((dates[4], "BBB"))
    stock.loc[(dates[-1], "AAA"), ("label", "LABEL0")] = np.nan
    market = pd.DataFrame(np.arange(len(dates) * 63).reshape(-1, 63).astype(float), index=dates,
                          columns=pd.MultiIndex.from_product([["market"], [f"M{i}" for i in range(63)]]))
    return dates, stock, market


def combined(stock, market):
    broadcast = pd.DataFrame(market.loc[stock.index.get_level_values("datetime")].to_numpy(),
                             index=stock.index, columns=market.columns)
    return pd.concat([stock.loc[:, ["feature", "prior"]], broadcast, stock.loc[:, ["label"]]], axis=1)


class DatasetTests(unittest.TestCase):
    def test_daily_loader_handles_instrument_major_and_singleton_batches(self):
        dates, stock, market = fixture()
        dataset = WindowSampler(combined(stock, market), dates[4], dates[6], 3)
        loader = init_data_loader(dataset)
        positions = loader.batch_sampler.ordered_indices()
        index = dataset.get_index()[positions]
        self.assertTrue(index.is_monotonic_increasing)
        emitted = []
        offset = 0
        for batch in loader:
            keys = index[offset:offset + len(batch)]
            self.assertEqual(keys.get_level_values("datetime").nunique(), 1)
            self.assertEqual(tuple(batch.shape[1:]), (3, 235))
            np.testing.assert_array_equal(batch.numpy(), dataset[positions[offset:offset + len(batch)]])
            emitted.append(len(batch))
            offset += len(batch)
        self.assertEqual(emitted, [1, 2, 2])
        self.assertEqual(offset, len(dataset))

    def test_daily_shuffle_preserves_groups_and_all_samples(self):
        dates, stock, market = fixture()
        dataset = WindowSampler(combined(stock, market), dates[4], dates[10], 3)
        sampler = DailyBatchSampler(dataset, shuffle=True)
        with patch("datasets.loader.np.random.shuffle", side_effect=lambda x: x.__setitem__(slice(None), x[::-1])):
            batches = list(sampler)
        actual = dataset.get_index()
        self.assertEqual(actual[batches[0]].get_level_values("datetime")[0], dates[10])
        for positions in batches:
            self.assertEqual(actual[positions].get_level_values("datetime").nunique(), 1)
        self.assertEqual(sorted(i for positions in batches for i in positions), list(range(len(dataset))))
        with self.assertRaises(ValueError):
            sampler.ordered_indices()

    def test_alphamaster_eight_day_layout(self):
        dates, stock, market = fixture()
        sampler = WindowSampler(combined(stock, market), dates[25], dates[-1], 8)
        sample = sampler[(dates[25], "AAA")]
        self.assertEqual(sample.shape, (8, 235))
        self.assertEqual(sampler.group_dims, dict(feature=158, prior=13, market=63, label=1))
        self.assertEqual(sampler.group_slices, dict(feature=slice(0,158), prior=slice(158,171),
                                                   market=slice(171,234), label=slice(234,235)))
        np.testing.assert_array_equal(sample[-1,158:171], stock.loc[(dates[25], "AAA"), "prior"].to_numpy())

    def test_jkp_rolling_lag_and_same_date_broadcast(self):
        dates = pd.bdate_range("2020-01-01", periods=25)
        raw = pd.DataFrame([
            dict(date=date, name=f"factor{i:02}", ret=.01, location="chn", weighting="vw_cap", freq="daily")
            for date in dates for i in range(13)
        ])
        config = dict(location="chn", weighting="vw_cap", freq="daily", window=20)
        factors = build_factor_matrix(raw, dates, config)
        self.assertTrue(factors.iloc[:20].isna().all().all())
        np.testing.assert_allclose(factors.iloc[20].to_numpy(), (1.01 ** 20 - 1))
        changed = raw.copy()
        changed.loc[changed.date == dates[20], "ret"] = .9
        lagged = build_factor_matrix(changed, dates, config)
        np.testing.assert_array_equal(factors.iloc[20].to_numpy(), lagged.iloc[20].to_numpy())
        self.assertFalse(np.array_equal(factors.iloc[21].to_numpy(), lagged.iloc[21].to_numpy()))
        index = pd.MultiIndex.from_product([[dates[20]], ["AAA", "BBB"]], names=["datetime", "instrument"])
        frame = pd.DataFrame(0., index=index, columns=pd.MultiIndex.from_tuples([("feature", "F")]))
        merged = GlobalFactorMerger(factors)(frame)
        np.testing.assert_array_equal(merged["prior"].iloc[0].to_numpy(), merged["prior"].iloc[1].to_numpy())
        with self.assertRaisesRegex(ValueError, "13"):
            build_factor_matrix(raw.loc[raw.name != "factor00"], dates, config)

    def test_both_configs_and_reference_handlers(self):
        for universe in ("csi300", "sp500"):
            config = load_config(ROOT / f"configs/data/{universe}_dataset.yaml")
            factors = pd.DataFrame(columns=[f"P{i}" for i in range(13)])
            cls = Alpha158WithJKP if universe == "csi300" else Alpha158USWithJKP
            handler = cls(factors, **config["data_handler_config"], init_data=False)
            market = MarketDataHandler(**config["market_data_handler_config"], init_data=False)
            self.assertIsNotNone(market.data_loader)
            fields, names = handler.get_feature_config()
            reference = Alpha158.__new__(Alpha158).get_feature_config()
            differences = [i for i, (a, b) in enumerate(zip(fields, reference[0])) if a != b]
            self.assertEqual(differences, [] if universe == "csi300" else [names.index("VWAP0")])
            if universe == "sp500":
                self.assertEqual(fields[names.index("VWAP0")], "($high+$low+$close)/3/$close")
            labels = config["data_handler_config"]["label"]
            self.assertEqual(labels, ["Ref($close, -5) / Ref($close, -1) - 1"])
            with (ROOT.parent / f"AlphaMaster/configs/master_{universe}.yaml").open() as stream:
                original = yaml.safe_load(stream)
            self.assertEqual({k: config["qlib_init"][k] for k in original["qlib_init"]}, original["qlib_init"])
            self.assertEqual(config["dataset"]["step_len"], original["task"]["dataset"]["kwargs"]["step_len"])
            self.assertEqual(config["dataset"]["segments"], original["task"]["dataset"]["kwargs"]["segments"])
            stock_params = copy.deepcopy(config["data_handler_config"])
            stock_params["infer_processors"] = [p for p in stock_params["infer_processors"] if p.get("kwargs", {}).get("fields_group") != "prior"]
            reference_params = original["data_handler_config"]
            self.assertEqual(stock_params, reference_params)
            params = config["market_data_handler_config"]
            expressions, _ = market_feature_config(params["market_indices"], params["windows"])
            self.assertEqual(len(expressions), 63)
            spec = importlib.util.spec_from_file_location("reference_dataset", ROOT.parent / "AlphaMaster/src/alphamaster/dataset.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            reference_market = module.marketDataHandler.__new__(module.marketDataHandler)
            reference_market.market_indices = params["market_indices"]
            self.assertEqual(expressions, reference_market.get_feature_config()[0])
            self.assertEqual(names, reference[1])

    def test_native_alphamaster_sampling_and_roundtrip(self):
        dates, stock, market = fixture()
        frame = combined(stock, market)
        # AlphaMaster 把市场拼到股票行，统一使用原生 Qlib sampler。
        reference_frame = frame.loc[:, ["feature", "market", "label"]].copy()
        reference = TSDataSampler(reference_frame, dates[5], dates[-1], 8,
                                  fillna_type="ffill+bfill", dtype=np.float32)
        sampler = WindowSampler(frame.copy(), dates[5], dates[-1], 8)
        self.assertTrue(reference.get_index().equals(sampler.get_index()))
        positions = np.arange(len(sampler))
        actual = sampler[positions]
        without_jkp = np.concatenate([actual[:, :, :158], actual[:, :, 171:]], axis=-1)
        np.testing.assert_array_equal(without_jkp, reference[positions])
        self.assertEqual(actual.shape, (len(sampler), 8, 235))
        self.assertEqual(actual.dtype, np.float32)
        # BBB 缺一天股票行，其市场历史也一起填充，与 AlphaMaster 一致。
        self.assertFalse(np.array_equal(sampler[(dates[5], "AAA")][:, 171:234],
                                        sampler[(dates[5], "BBB")][:, 171:234]))
        restored = pickle.loads(pickle.dumps(sampler))
        np.testing.assert_array_equal(restored[(dates[5], "AAA")], sampler[(dates[5], "AAA")])
        self.assertTrue(np.isnan(restored[(dates[-1], "AAA")][-1, -1]))

    def test_sampler_rejects_bad_schema(self):
        dates, stock, market = fixture()
        frame = combined(stock, market)
        with self.assertRaisesRegex(ValueError, "ordered"):
            WindowSampler(frame.iloc[:, ::-1], dates[5], dates[-1], 3)
        with self.assertRaisesRegex(ValueError, "unique"):
            WindowSampler(pd.concat([frame, frame.iloc[:1]]), dates[5], dates[-1], 3)
        with self.assertRaisesRegex(ValueError, "Empty"):
            WindowSampler(frame.copy(), dates[-1] + pd.Timedelta(days=1), dates[-1] + pd.Timedelta(days=3), 3)

    def test_same_date_market_validation(self):
        dates, stock, market = fixture()
        expanded = combined(stock, market).loc[:, ["market"]]
        handler = SimpleNamespace(fetch=lambda **kwargs: expanded)
        daily = daily_market_frame(handler)
        np.testing.assert_array_equal(daily.to_numpy(), market.to_numpy())
        expanded.iloc[1, 0] += 1
        with self.assertRaisesRegex(ValueError, "differ"):
            daily_market_frame(handler)

    def test_config_rejects_overlap_and_fit_leakage(self):
        config = load_config(ROOT / "configs/data/csi300_dataset.yaml")
        overlap = copy.deepcopy(config)
        overlap["dataset"]["segments"]["valid"][0] = "2020-12-31"
        leakage = copy.deepcopy(config)
        leakage["data_handler_config"]["fit_end_time"] = "2021-01-01"
        with tempfile.TemporaryDirectory() as temp:
            for invalid in (overlap, leakage):
                path = Path(temp) / "invalid.yaml"
                path.write_text(yaml.safe_dump(invalid))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_validate_only_never_initializes_or_builds(self):
        with patch("sys.argv", ["build", "--config", str(ROOT / "configs/data/sp500_dataset.yaml"), "--validate-only"]), \
             patch("datasets.build.qlib.init", side_effect=AssertionError("real initialization")), \
             patch("datasets.build.build_dataset", side_effect=AssertionError("build forbidden")):
            main()

    def test_mock_build_serializes_splits_and_refuses_overwrite(self):
        dates, stock, market = fixture()
        config = load_config(ROOT / "configs/data/csi300_dataset.yaml")
        settings = config["dataset"]
        settings["step_len"] = 3
        settings["segments"] = {"train": ["2020-01-06", "2020-01-08"],
                                "valid": ["2020-01-09", "2020-01-13"],
                                "test": ["2020-01-14", "2020-01-16"]}
        for component in ("data_handler_config", "market_data_handler_config"):
            config[component].update(start_time="2020-01-01", end_time="2020-01-16",
                fit_start_time="2020-01-06", fit_end_time="2020-01-08")
        seen = []

        def fetch(selector, col_set, data_key):
            seen.append(data_key)
            frame = stock.loc[selector, :].copy()
            return frame.dropna() if data_key == "learn" else frame

        handler = SimpleNamespace(start_time="2020-01-01", fetch=fetch)
        with tempfile.TemporaryDirectory() as temp:
            config["output_dir"] = str(Path(temp) / "synthetic")
            path = Path(temp) / "data.yaml"
            path.write_text(yaml.safe_dump(config))
            with patch("datasets.build.qlib.init") as init, \
                 patch("datasets.build.Alpha158WithJKP", return_value=handler), \
                 patch("datasets.build.pd.read_csv", return_value=pd.DataFrame()), \
                 patch("datasets.build.build_factor_matrix", return_value=pd.DataFrame()), \
                 patch("datasets.build.MarketDataHandler", return_value=object()), \
                 patch("datasets.build.daily_market_frame", return_value=market), \
                 patch("datasets.build.D", SimpleNamespace(calendar=lambda **kwargs: dates)):
                destination = build_dataset(path)
                self.assertEqual(seen, ["learn", "learn", "infer"])
                self.assertTrue((destination / "config.yaml").exists())
                for split in ("train", "valid", "test"):
                    with (destination / f"{split}.pkl").open("rb") as stream:
                        sampler = pickle.load(stream)
                    self.assertEqual(sampler[0].shape, (3, 235))
                    self.assertEqual(sampler[0].dtype, np.float32)
                with self.assertRaises(FileExistsError):
                    build_dataset(path)
                self.assertEqual(init.call_count, 1)


if __name__ == "__main__":
    unittest.main()
