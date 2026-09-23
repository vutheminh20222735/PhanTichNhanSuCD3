"""Kiểm tra CV multi-model, batch predict, what-if."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "hr_analytics"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from mo_hinh.danh_gia import evaluate_all_classifiers, merge_cv_into_metrics
from mo_hinh.du_doan import predict_attrition_batch, simulate_what_if, summarize_batch_risk
from mo_hinh.huan_luyen import HAS_XGBOOST, train_classification_models


def _tiny_hr_df(n: int = 100) -> pd.DataFrame:
    rows = []
    for i in range(n):
        leave = i % 5 == 0
        rows.append({
            "EmployeeNumber": i + 1,
            "Age": 22 + (i % 30),
            "Department": "Sales" if i % 2 == 0 else "Research & Development",
            "OverTime": "Yes" if leave or i % 3 == 0 else "No",
            "MonthlyIncome": 3000 + i * 40,
            "YearsAtCompany": 1 + (i % 10),
            "JobSatisfaction": 1 if leave else 3 + (i % 2),
            "Attrition": "Yes" if leave else "No",
        })
    return pd.DataFrame(rows)


class TestCvMultiModelBatchWhatIf(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = _tiny_hr_df()
        cls.train = train_classification_models(
            cls.df, save=False, target="Attrition", run_cv=True, cv_folds=3
        )

    def test_models_include_gb_and_optional_xgb(self):
        names = set(self.train["models"].keys())
        self.assertIn("Logistic Regression", names)
        self.assertIn("Random Forest", names)
        self.assertIn("Gradient Boosting", names)
        if HAS_XGBOOST:
            self.assertIn("XGBoost", names)
        self.assertIsNotNone(self.train.get("cv_table"))
        self.assertFalse(self.train["cv_table"].empty)

    def test_evaluate_and_merge_cv(self):
        ev = evaluate_all_classifiers(
            self.train["models"], self.train["X_test"], self.train["y_test"]
        )
        merged = merge_cv_into_metrics(ev["metrics_table"], self.train["cv_table"])
        self.assertIn("CV_Recall", merged.columns)
        self.assertTrue(ev["best_model_name"] in self.train["models"])

    def test_batch_predict(self):
        best = "Random Forest"
        model = self.train["models"][best]
        scored = predict_attrition_batch(
            model,
            self.df,
            self.train["feature_columns"],
            id_col="EmployeeNumber",
            keep_cols=["Department", "OverTime"],
        )
        self.assertEqual(len(scored), len(self.df))
        self.assertIn("risk_band", scored.columns)
        summary = summarize_batch_risk(scored)
        self.assertEqual(summary["total"], len(self.df))
        self.assertGreaterEqual(summary["High"] + summary["Medium"] + summary["Low"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as w:
                scored.to_excel(w, sheet_name="Risk_List", index=False)
                pd.DataFrame([summary]).to_excel(w, sheet_name="Summary", index=False)
            self.assertTrue(path.exists() and path.stat().st_size > 0)

    def test_what_if(self):
        model = self.train["models"]["Random Forest"]
        base = self.train["X_train"].iloc[0].to_dict()
        # Đổi OverTime nếu có
        changes = {}
        if "OverTime" in base:
            changes["OverTime"] = "No" if str(base["OverTime"]) == "Yes" else "Yes"
        elif "MonthlyIncome" in base:
            changes["MonthlyIncome"] = float(base["MonthlyIncome"]) * 1.2
        else:
            col = self.train["feature_columns"][0]
            changes[col] = base[col]
        sim = simulate_what_if(model, base, changes, self.train["feature_columns"])
        self.assertIn("delta_pp", sim)
        self.assertIn("baseline", sim)
        self.assertIn("after", sim)
        self.assertTrue(sim["message"])


if __name__ == "__main__":
    unittest.main()
