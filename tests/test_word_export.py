import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "hr_analytics"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ket_qua.xuat_word import export_report_word
from ung_dung import PeopleRiskApp
from xu_ly.trang_thai_dataset import DatasetState, DatasetStatus


class TestWordExport(unittest.TestCase):
    def test_export_report_word_creates_docx(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.docx"
            export_report_word(
                path,
                dataset_name="Dataset demo",
                target="attrition",
                n_rows=120,
                n_cols=14,
                filter_note="Department=Sales",
                kpis={
                    "total_employees": 120,
                    "employees_staying": 105,
                    "employees_leaving": 15,
                    "attrition_rate": 12.5,
                    "avg_age": 31.4,
                    "avg_income": 25000000,
                    "avg_years_company": 3.8,
                    "avg_job_satisfaction": 4.1,
                },
                insights=[
                    {"title": "Turnover cao", "insight": "Nhân viên mới rời đi nhiều hơn mức kỳ vọng.", "severity": "Cao"},
                    {"title": "Mức lương", "insight": "Lương dưới trung vị ở đội bán hàng thấp hơn.", "severity": "Trung bình"},
                ],
                recommendations=[
                    {"based_on": "Turnover cao", "recommended_action": "Xem xét chính sách giữ chân nhân viên mới.", "priority": "Cao"},
                ],
                model_name="RandomForest",
                model_metrics={"Accuracy": 0.86, "Precision": 0.78, "Recall": 0.74, "F1": 0.76, "ROC-AUC": 0.82},
            )
            self.assertTrue(path.exists())
            self.assertTrue(path.stat().st_size > 0)

    def test_export_pdf_uses_dataset_state(self):
        app = object.__new__(PeopleRiskApp)
        app.dataset_state = DatasetState(
            status=DatasetStatus.READY,
            name="Dataset demo",
            df=__import__("pandas").DataFrame({"attrition": [0, 1], "age": [25, 30]}),
            target="attrition",
        )
        app.eval_result = None
        app._report_payload = lambda: {
            "df": __import__("pandas").DataFrame({"attrition": [0, 1], "age": [25, 30]}),
            "kpis": {"attrition_rate": 12.5},
            "insights": [{"title": "Turnover", "insight": "Tăng", "severity": "Cao"}],
            "recs": [{"based_on": "Turnover", "recommended_action": "Giám sát", "priority": "Cao"}],
            "filter_note": "Không lọc",
            "model_name": None,
            "model_metrics": None,
            "metrics_table": None,
            "shap_section": None,
        }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.pdf"
            with patch("ung_dung.filedialog.asksaveasfilename", return_value=str(out)), patch(
                "ung_dung.export_report_pdf"
            ) as mock_pdf, patch("ung_dung.messagebox.showinfo"), patch(
                "ung_dung.messagebox.showerror"
            ):
                app._export_pdf()
                mock_pdf.assert_called_once()
                self.assertEqual(mock_pdf.call_args.kwargs["dataset_name"], "Dataset demo")

    def test_export_word_uses_dataset_state(self):
        app = object.__new__(PeopleRiskApp)
        app.dataset_state = DatasetState(
            status=DatasetStatus.READY,
            name="Dataset demo",
            df=__import__("pandas").DataFrame({"attrition": [0, 1], "age": [25, 30]}),
            target="attrition",
        )
        app.eval_result = None
        app._report_payload = lambda: {
            "df": __import__("pandas").DataFrame({"attrition": [0, 1], "age": [25, 30]}),
            "kpis": {"attrition_rate": 12.5},
            "insights": [{"title": "Turnover", "insight": "Tăng", "severity": "Cao"}],
            "recs": [{"based_on": "Turnover", "recommended_action": "Giám sát", "priority": "Cao"}],
            "filter_note": "Không lọc",
            "model_name": None,
            "model_metrics": None,
            "metrics_table": None,
        }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.docx"
            with patch("ung_dung.filedialog.asksaveasfilename", return_value=str(out)), patch(
                "ung_dung.export_report_word"
            ) as mock_word, patch("ung_dung.messagebox.showinfo"):
                app._export_word()
                mock_word.assert_called_once()
                self.assertEqual(mock_word.call_args.kwargs["dataset_name"], "Dataset demo")

    def test_export_excel_creates_xlsx(self):
        app = object.__new__(PeopleRiskApp)
        app.dataset_state = DatasetState(
            status=DatasetStatus.READY,
            name="Dataset demo",
            df=__import__("pandas").DataFrame({"attrition": [0, 1], "age": [25, 30]}),
            target="attrition",
        )
        app.eval_result = None
        app._report_payload = lambda: {
            "df": __import__("pandas").DataFrame({"attrition": [0, 1], "age": [25, 30]}),
            "kpis": {"attrition_rate": 12.5},
            "insights": [{"title": "Turnover", "insight": "Tăng", "severity": "Cao"}],
            "recs": [{"based_on": "Turnover", "recommended_action": "Giám sát", "priority": "Cao"}],
            "filter_note": "Không lọc",
            "model_name": None,
            "model_metrics": None,
            "metrics_table": None,
        }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.xlsx"
            with patch("ung_dung.filedialog.asksaveasfilename", return_value=str(out)), patch(
                "ung_dung.messagebox.showinfo"
            ):
                app._export()
                self.assertTrue(out.exists())
                self.assertGreater(out.stat().st_size, 0)

    def test_validate_dataset_for_analysis_handles_invalid_data(self):
        df = __import__("pandas").DataFrame({"attrition": [None, None], "age": [20, 30]})
        ok, message = PeopleRiskApp._validate_dataset_for_analysis(object.__new__(PeopleRiskApp), df, "attrition")
        self.assertFalse(ok)
        self.assertIn("trống", message.lower())


if __name__ == "__main__":
    unittest.main()
