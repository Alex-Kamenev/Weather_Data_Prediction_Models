import unittest
import contextlib
import io
from unittest.mock import patch
import pandas as pd

from edapy.datainfo import DataInfo
from model_predict_py.model import Modelpredict


class DataInfoTests(unittest.TestCase):
    def test_midpoint_preserves_observed_temperatures_for_other_station(self):
        df = pd.DataFrame({"DATE": ["2020-01-01"] * 2, "TMIN": [0.0, 20.0], "TMAX": [10.0, 30.0], "TAVG": [None, 23.0]})
        DataInfo(df).half_sum_fill_in_missing("TAVG")
        self.assertEqual(df["TAVG"].tolist(), [5.0, 23.0])

    def test_one_hot_encoding_preserves_rows_with_duplicate_index(self):
        df = pd.DataFrame({"STATION": ["one", "two"], "TAVG": [10.0, 20.0]}, index=[0, 0])
        result = Modelpredict().one_hot_encode(df, "STATION")
        self.assertEqual(len(result), 2)
        self.assertEqual(result["TAVG"].tolist(), [10.0, 20.0])
        self.assertEqual(result["STATION__one"].tolist(), [True, False])

    def test_linear_regression_runs_outside_notebook(self):
        df = pd.DataFrame({"x": list(range(20)), "target": [2 * x + 1 for x in range(20)]})
        model = Modelpredict()
        with patch.object(model, "plot_scatter"), patch.object(model, "plot_kde"), contextlib.redirect_stdout(io.StringIO()):
            predicted, actual = model.linear_reg(df, "target", 0.25)
        self.assertEqual(len(predicted), 5)
        for prediction, observation in zip(predicted, actual):
            self.assertAlmostEqual(prediction, observation)


if __name__ == "__main__":
    unittest.main()
