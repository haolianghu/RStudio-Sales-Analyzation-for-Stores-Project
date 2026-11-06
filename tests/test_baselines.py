import unittest
from pathlib import Path

import pandas as pd

from reproducible_baselines import customer_classification, sales_forecast


ROOT = Path(__file__).resolve().parents[1]


class BaselineTests(unittest.TestCase):
    def test_existing_sales_data(self):
        sales = pd.read_csv(ROOT / "whole_foods.csv")
        result = sales_forecast(sales)
        self.assertEqual(result["rows"], 88)
        self.assertEqual(result["holdout_quarters"], 8)
        self.assertEqual([row["quarter"] for row in result["next_four_quarters"]], ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual([row["year"] for row in result["next_four_quarters"]], [2017.0, 2017.25, 2017.5, 2017.75])
        self.assertGreater(result["model"]["rmse"], 0)
        self.assertGreater(result["seasonal_naive"]["rmse"], 0)

    def test_existing_customer_data(self):
        purchases = pd.read_csv(ROOT / "purchases_type.csv")
        result = customer_classification(purchases)
        self.assertEqual((result["train_rows"], result["test_rows"]), (800, 200))
        for metrics in result["models"].values():
            self.assertEqual(sum(sum(row) for row in metrics["confusion_matrix"]), 200)

    def test_irregular_quarters_and_invalid_target(self):
        sales = pd.read_csv(ROOT / "whole_foods.csv")
        sales.loc[10, "year"] += 0.1
        with self.assertRaisesRegex(ValueError, "encode quarters"):
            sales_forecast(sales)
        purchases = pd.read_csv(ROOT / "purchases_type.csv")
        purchases["type"] = "one class"
        with self.assertRaisesRegex(ValueError, "two classes"):
            customer_classification(purchases)


if __name__ == "__main__":
    unittest.main()
