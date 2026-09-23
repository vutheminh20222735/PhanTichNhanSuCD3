"""Kiểm tra SHAP + xuất Word/PDF có kèm mục giải thích."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "hr_analytics"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ket_qua.xuat_pdf import export_report_pdf
from ket_qua.xuat_word import export_report_word
from mo_hinh.giai_thich import (
    explain_global_shap,
    explain_local_shap,
    shap_section_for_report,
)
from mo_hinh.huan_luyen import train_classification_models
from ung_dung import PeopleRiskApp
from xu_ly.trang_thai_dataset import DatasetState, DatasetStatus


def _tiny_hr_df(n: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(n):
        leave = i % 5 == 0
        rows.append({
            "Age": 22 + (i % 30),
            "Department": "Sales" if i % 2 == 0 else "Research & Development",
            "OverTime": "Yes" if leave or i % 3 == 0 else "No",
            "MonthlyIncome": 3000 + i * 40,
            "YearsAtCompany": 1 + (i % 10),
            "JobSatisfaction": 1 if leave else 3 + (i % 2),
            "Attrition": "Yes" if leave else "No",
        })
    return pd.DataFrame(rows)


class TestShapExplain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = _tiny_hr_df()
        cls.train = train_classification_models(cls.df, save=False, target="Attrition")

    def test_global_and_local_shap(self):
        pipe = self.train["models"]["Random Forest"]
        global_res = explain_global_shap(
            pipe,
            self.train["X_train"],
            self.train["numeric_features"],
            self.train["categorical_features"],
            top_n=8,
            max_samples=40,
            background_size=40,
        )
        self.assertFalse(global_res["importance"].empty)
        self.assertIn("narrative", global_res)

        sample = self.train["X_train"].iloc[0].to_dict()
        local_res = explain_local_shap(
            pipe,
            sample,
            self.train["feature_columns"],
            self.train["numeric_features"],
            self.train["categorical_features"],
            self.train["X_train"],
            top_n=8,
            background_size=40,
        )
        self.assertFalse(local_res["contributions"].empty)
        self.assertTrue(local_res["narrative"])

        section = shap_section_for_report(
            global_result=global_res,
            local_result=local_res,
            prediction={"probability_pct": 55.0, "risk_band": "Medium", "prediction": "Yes"},
            model_name="Random Forest",
        )
        self.assertTrue(section["global_top"])
        self.assertTrue(section["local_top"])

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "shap.pdf"
            docx_path = Path(tmp) / "shap.docx"
            export_report_pdf(
                pdf_path,
                dataset_name="tiny",
                target="Attrition",
                n_rows=len(self.df),
                n_cols=self.df.shape[1],
                filter_note="Không lọc",
                kpis={"attrition_rate": 20.0, "total_employees": len(self.df)},
                insights=[{"title": "OT", "insight": "OverTime cao", "severity": "Cao"}],
                recommendations=[{"based_on": "OT", "recommended_action": "Giảm OT", "priority": "Cao"}],
                model_name="Random Forest",
                model_metrics={"Accuracy": 0.8, "Precision": 0.7, "Recall": 0.75, "F1": 0.72, "ROC-AUC": 0.81},
                shap_section=section,
            )
            export_report_word(
                docx_path,
                dataset_name="tiny",
                target="Attrition",
                n_rows=len(self.df),
                n_cols=self.df.shape[1],
                filter_note="Không lọc",
                kpis={"attrition_rate": 20.0, "total_employees": len(self.df)},
                insights=[{"title": "OT", "insight": "OverTime cao", "severity": "Cao"}],
                recommendations=[{"based_on": "OT", "recommended_action": "Giảm OT", "priority": "Cao"}],
                model_name="Random Forest",
                model_metrics={"Accuracy": 0.8, "Precision": 0.7, "Recall": 0.75, "F1": 0.72, "ROC-AUC": 0.81},
                shap_section=section,
            )
            self.assertTrue(pdf_path.exists() and pdf_path.stat().st_size > 0)
            self.assertTrue(docx_path.exists() and docx_path.stat().st_size > 0)

    def test_export_word_button_wires_shap(self):
        app = object.__new__(PeopleRiskApp)
        app.dataset_state = DatasetState(
            status=DatasetStatus.READY,
            name="Dataset demo",
            df=pd.DataFrame({"Attrition": [0, 1], "Age": [25, 30]}),
            target="Attrition",
        )
        app.eval_result = None
        app.shap_global = {
            "narrative": "Feature Age quan trọng.",
            "importance": pd.DataFrame({"Feature": ["Age"], "mean_|SHAP|": [0.12]}),
        }
        app.shap_local = None
        app.last_prediction = None
        app._report_payload = lambda: {
            "df": pd.DataFrame({"Attrition": [0, 1], "Age": [25, 30]}),
            "kpis": {"attrition_rate": 50.0},
            "insights": [],
            "recs": [],
            "filter_note": "Không lọc",
            "model_name": "Random Forest",
            "model_metrics": None,
            "metrics_table": None,
            "shap_section": shap_section_for_report(
                global_result=app.shap_global,
                model_name="Random Forest",
            ),
        }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.docx"
            with patch("ung_dung.filedialog.asksaveasfilename", return_value=str(out)), patch(
                "ung_dung.export_report_word"
            ) as mock_word, patch("ung_dung.messagebox.showinfo"):
                app._export_word()
                mock_word.assert_called_once()
                self.assertIn("shap_section", mock_word.call_args.kwargs)
                self.assertTrue(mock_word.call_args.kwargs["shap_section"]["global_top"])


if __name__ == "__main__":
    unittest.main()
