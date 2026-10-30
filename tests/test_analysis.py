import unittest

import pandas as pd

from analysis import analyze_question, execute_task_plan, prepare_chart_data


class AnalyzeQuestionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "City": [
                "North", "North", "South", "South",
                "East", "East", "West", "West",
            ],
            "Product": ["A", "B", "A", "C", "D", "E", "F", "G"],
            "Quantity": [2, 1, 5, 2, 1, 3, 4, 1],
            "Unit Price": [10, 20, 10, 20, 50, 20, 15, 5],
        })

    def test_total_revenue_uses_quantity_times_unit_price(self):
        result = analyze_question("What was the total revenue?", self.frame)
        self.assertEqual(result.rows[0]["Value"], 265)

    def test_highest_revenue_city(self):
        result = analyze_question(
            "Which city had the highest revenue?", self.frame
        )
        self.assertEqual(result.rows[0]["Category"], "East")
        self.assertEqual(result.rows[0]["Value"], 110)

    def test_highest_revenue_product(self):
        result = analyze_question(
            "Which product had the highest revenue?", self.frame
        )
        self.assertEqual(result.rows[0]["Category"], "A")
        self.assertEqual(result.rows[0]["Value"], 70)

    def test_compound_hinglish_question_returns_all_requested_insights(self):
        result = analyze_question(
            "Sales aur Revenue: kul kitni kamai hui, aur kis City ya "
            "Product se sabse zyada revenue mila?",
            self.frame,
        )
        labels = {row["Insight"] for row in result.rows}
        self.assertIn("Total Revenue", labels)
        self.assertIn("Highest Revenue by City", labels)
        self.assertIn("Highest Revenue by Product", labels)
        self.assertEqual(len(result.charts), 2)

    def test_top_five_products(self):
        result = analyze_question(
            "Show the top 5 products by revenue", self.frame
        )
        self.assertEqual(len(result.rows), 5)
        self.assertEqual(result.rows[0]["Category"], "A")
        self.assertEqual(result.rows[0]["Value"], 70)

    def test_compare_revenue_across_cities(self):
        result = analyze_question(
            "Compare revenue across cities", self.frame
        )
        self.assertEqual(len(result.rows), 4)
        self.assertEqual(result.rows[0]["Category"], "East")

    def test_explicit_filters_are_applied(self):
        result = analyze_question(
            "What was total revenue in North?", self.frame
        )
        self.assertEqual(result.rows[0]["Value"], 40)

    def test_multiple_numeric_metrics_are_returned(self):
        frame = self.frame.assign(
            Revenue=self.frame.Quantity * self.frame["Unit Price"],
            Profit=[5, 10, 12, 8, 20, 15, 7, 3],
        )
        result = analyze_question(
            "Compare Revenue and Profit across cities", frame
        )
        labels = {row["Insight"] for row in result.rows}
        self.assertIn("Revenue by City", labels)
        self.assertIn("Profit by City", labels)

    def test_missing_dimension_is_reported(self):
        frame = self.frame.drop(columns=["City"])
        result = analyze_question(
            "Which city had the highest revenue?", frame
        )
        self.assertFalse(result.has_results)
        self.assertIn("city column", result.message)

    def test_local_fallback_and_duplicate_index(self):
        frame = self.frame.copy()
        frame.index = [1, 1, 2, 2, 3, 3, 4, 4]
        result = analyze_question("Total revenue", frame)
        self.assertEqual(result.rows[0]["Value"], 265)
        grouped = analyze_question(
            "Compare revenue across cities", frame
        )
        self.assertTrue(grouped.has_results)
        self.assertEqual(
            list(grouped.charts[0][1].columns), ["Category", "Value"]
        )

    def test_chart_data_avoids_index_and_column_name_collisions(self):
        frame = self.frame.copy()
        frame["Order_ID"] = range(len(frame))
        frame.index.name = "Order_ID"
        chart_data = prepare_chart_data(
            frame, "Order_ID", "Unit Price", aggregate=True
        )
        self.assertEqual(list(chart_data.columns), ["X value", "Y value"])
        same_column = prepare_chart_data(frame, "Unit Price", "Unit Price")
        self.assertEqual(list(same_column.columns), ["X value", "Y value"])

    def test_validated_plan_uses_pandas_calculations(self):
        frame = self.frame.assign(
            Revenue=self.frame.Quantity * self.frame["Unit Price"]
        )
        result = execute_task_plan(frame, [
            {
                "operation": "total",
                "column": "Revenue",
                "filters": [
                    {"column": "City", "operator": "equals", "value": "North"}
                ],
            },
            {
                "operation": "top",
                "column": "Revenue",
                "group_by": "Product",
                "limit": 2,
            },
        ])
        self.assertEqual(result.rows[0]["Value"], 40)
        self.assertEqual(result.rows[1]["Category"], "A")
        self.assertEqual(result.rows[1]["Value"], 70)


if __name__ == "__main__":
    unittest.main()
