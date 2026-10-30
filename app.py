
import json
import logging
import os
import re

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from analysis import (
    analyze_question,
    execute_task_plan,
    format_result_value,
    prepare_chart_data,
)

LOGGER = logging.getLogger(__name__)


def interpret_with_ai(question, frame):
    api_key = os.getenv("GEMINI_API_KEY", "")
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "") or api_key
    except Exception:
        pass
    if not isinstance(api_key, str) or not api_key.strip():
        return None

    try:
        from google import genai
    except ImportError as error:
        LOGGER.warning(
            "Optional question interpretation unavailable (%s)",
            type(error).__name__,
        )
        return None

    schema = [
        {"name": str(column), "type": str(frame[column].dtype)}
        for column in frame.columns
    ]
    prompt = f"""
Interpret the complete question as a list of data-analysis tasks.
Return only JSON with this shape:
{{"tasks":[{{"operation":"total","column":"Revenue","group_by":null,
 "limit":null,"filters":[]}}]}}

Allowed operations: total, mean, maximum, minimum, group_sum, group_mean,
top, compare, count.
Use exact column names from this schema. Never return calculated values.
For grouped operations, specify group_by. For top, use limit.
Filters must be objects with column, operator, and value. Operators may be
equals, greater_than, less_than, at_least, or at_most.
If any requested operation or required column cannot be mapped, return an
empty tasks list. Do not omit requested tasks.

Schema:
{json.dumps(schema, ensure_ascii=False)}

Question:
{question}
"""
    client = None
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        raw = (response.text or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return parsed.get("tasks") if isinstance(parsed, dict) else None
    except Exception as error:
        LOGGER.warning(
            "Optional question interpretation unavailable (%s)",
            type(error).__name__,
        )
        return None
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as error:
                LOGGER.warning(
                    "Optional question interpretation cleanup failed (%s)",
                    type(error).__name__,
                )


def combined_revenue_result(question, frame):
    words = set(re.findall(r"[a-z0-9]+", question.casefold()))
    asks_total = bool(words & {"total", "sum", "overall", "kul"})
    asks_highest = bool(
        words & {"highest", "most", "top", "zyada", "maximum"}
    ) or "sabse zyada" in question.casefold()
    asks_revenue = bool(
        words & {"revenue", "sales", "sale", "kamai", "earning", "income"}
    )
    if not (asks_total and asks_highest and asks_revenue):
        return None

    numeric_columns = [
        column for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    revenue_terms = {
        "revenue", "sales", "sale", "amount", "earning", "earnings",
        "income", "turnover",
    }
    measure_candidates = [
        column for column in numeric_columns
        if set(re.findall(r"[a-z0-9]+", str(column).casefold()))
        & revenue_terms
    ]

    if measure_candidates:
        measure = frame[measure_candidates[0]]
    else:
        quantity_columns = [
            column for column in numeric_columns
            if set(re.findall(r"[a-z0-9]+", str(column).casefold()))
            & {"quantity", "qty", "units"}
        ]
        price_columns = [
            column for column in numeric_columns
            if set(re.findall(r"[a-z0-9]+", str(column).casefold()))
            & {"price", "unitprice", "rate"}
        ]
        if len(quantity_columns) != 1 or len(price_columns) != 1:
            return None
        measure = frame[quantity_columns[0]] * frame[price_columns[0]]

    dimension_aliases = {
        "City": {"city", "cities", "town", "location"},
        "Product": {"product", "products", "item", "items", "sku"},
    }
    result_rows = [{
        "Insight": "Total Revenue",
        "Category": "",
        "Value": measure.sum(min_count=1),
    }]
    charts = []
    for label, aliases in dimension_aliases.items():
        if not (words & aliases):
            continue
        candidates = [
            column for column in frame.columns
            if set(re.findall(r"[a-z0-9]+", str(column).casefold()))
            & aliases
        ]
        if len(candidates) != 1:
            return None
        dimension = candidates[0]
        grouped = (
            measure.reset_index(drop=True)
            .groupby(frame[dimension].reset_index(drop=True), dropna=False)
            .sum(min_count=1)
            .dropna()
            .sort_values(ascending=False)
        )
        if grouped.empty:
            return None
        category, value = grouped.index[0], grouped.iloc[0]
        result_rows.append({
            "Insight": f"Highest Revenue {label}",
            "Category": "(Missing)" if pd.isna(category) else str(category),
            "Value": value,
        })
        charts.append((
            f"Revenue by {label}",
            pd.DataFrame({
                "Category": grouped.index.astype(str),
                "Value": grouped.to_numpy(),
            }),
        ))
    if len(result_rows) == 1:
        return None
    return result_rows, charts


st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("AI Data Analyst")
st.subheader("Automatic Dataset Summary & Visualization")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # ---------------- READ CSV ----------------

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    if df.empty or len(df.columns) == 0:
        st.warning("The uploaded CSV is empty.")
        st.stop()

    # Remove accidental leading/trailing spaces from column names.
    # If this creates duplicate names, stop rather than risk wrong results.
    cleaned_columns = [str(c).strip() for c in df.columns]

    if len(cleaned_columns) != len(set(cleaned_columns)):
        st.error(
            "Some column names become duplicates after trimming spaces. "
            "Please rename those columns in your CSV."
        )
        st.stop()

    df.columns = cleaned_columns
    df = df.reset_index(drop=True)

    # ---------------- DATASET OVERVIEW ----------------

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Duplicate Rows", int(df.duplicated().sum()))

    # ---------------- DATA PREVIEW ----------------

    st.header("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # ---------------- NUMERICAL SUMMARY ----------------

    st.header("Numerical Summary")
    numerical_df = df.select_dtypes(include="number")

    if not numerical_df.empty:
        st.dataframe(numerical_df.describe())
    else:
        st.info("No numerical columns found.")

    # ---------------- MISSING VALUES ----------------

    st.header("Missing Values")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing Percentage": (
            df.isnull().mean().values * 100
        ).round(2)
    })

    st.dataframe(missing_df, use_container_width=True)

    # ---------------- DATA TYPES ----------------

    st.header("Column Data Types")

    types_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(types_df, use_container_width=True)

    # ---------------- VISUALIZATION ----------------

    st.header("Data Visualization")

    if df.empty or not len(df.columns):
        st.info("There is no data available to visualize.")
    else:
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        date_columns = [
            column for column in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[column])
        ]
        for column in df.select_dtypes(include=["object", "string"]).columns:
            sample = df[column].dropna().astype(str).head(50)
            if not sample.empty:
                parsed_dates = pd.to_datetime(sample, errors="coerce")
                if parsed_dates.notna().mean() >= 0.8:
                    date_columns.append(column)

        categorical_columns = [
            column for column in df.columns
            if column not in numeric_columns and column not in date_columns
        ]
        bar_x_options = [
            column for column in df.columns
            if column in categorical_columns or column in date_columns
            or (
                column in numeric_columns
                and df[column].nunique(dropna=True) <= 20
            )
        ]
        line_x_options = [
            column for column in df.columns
            if column in categorical_columns or column in date_columns
            or column in numeric_columns
        ]

        chart_types = []
        if numeric_columns and bar_x_options:
            chart_types.append("Bar Chart")
        if numeric_columns and line_x_options:
            chart_types.append("Line Chart")
        if numeric_columns:
            chart_types.append("Histogram")
        if len(numeric_columns) >= 2:
            chart_types.append("Scatter Plot")
        if numeric_columns and categorical_columns:
            chart_types.append("Pie Chart")

        if not chart_types:
            st.info(
                "This dataset needs a numerical column and a suitable "
                "category column to create a chart."
            )
        else:
            chart_key = "visualization_chart_type"
            if st.session_state.get(chart_key) not in chart_types:
                st.session_state[chart_key] = chart_types[0]
            chart_type = st.selectbox(
                "Select Chart Type",
                chart_types,
                key=chart_key,
            )

            def select_column(label, options, key):
                if not options:
                    st.info(f"No suitable column is available for {label.lower()}.")
                    return None
                if st.session_state.get(key) not in options:
                    st.session_state[key] = options[0]
                return st.selectbox(label, options, key=key)

            if chart_type in {"Bar Chart", "Line Chart", "Pie Chart"}:
                y_column = select_column(
                    "Select Y-axis",
                    numeric_columns,
                    f"{chart_type}_y",
                )
                if chart_type == "Pie Chart":
                    x_options = categorical_columns
                elif chart_type == "Line Chart":
                    x_options = line_x_options
                else:
                    x_options = bar_x_options
                if y_column in x_options and y_column in numeric_columns:
                    x_options = [column for column in x_options if column != y_column]
                x_column = select_column(
                    "Select X-axis",
                    x_options,
                    f"{chart_type}_x",
                )

                if x_column is not None and y_column is not None:
                    if x_column not in df.columns or y_column not in df.columns:
                        st.info("Choose columns that are present in the uploaded data.")
                    elif y_column not in numeric_columns:
                        st.info("Choose a numerical column for the Y-axis.")
                    elif x_column == y_column:
                        st.info("Choose different columns for the X-axis and Y-axis.")
                    elif chart_type == "Pie Chart":
                        pie_data = (
                            df.groupby(x_column, dropna=False)[y_column]
                            .sum()
                            .dropna()
                        )
                        if pie_data.empty or (pie_data < 0).any() or pie_data.sum() <= 0:
                            st.info("Choose a category and non-negative values for the pie chart.")
                        else:
                            fig, ax = plt.subplots()
                            ax.pie(
                                pie_data.values,
                                labels=[
                                    "(Missing)" if pd.isna(value) else str(value)
                                    for value in pie_data.index
                                ],
                                autopct="%1.1f%%",
                            )
                            ax.set_title(f"{y_column} by {x_column}")
                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        chart_data = prepare_chart_data(
                            df,
                            x_column,
                            y_column,
                            aggregate=(chart_type == "Bar Chart"),
                        )
                        if chart_data.empty:
                            st.info("There are no usable values for this chart.")
                        elif chart_type == "Bar Chart":
                            st.bar_chart(
                                chart_data,
                                x="X value",
                                y="Y value",
                            )
                        else:
                            if x_column in date_columns:
                                chart_data = chart_data.sort_values("X value")
                            st.line_chart(
                                chart_data,
                                x="X value",
                                y="Y value",
                            )

            elif chart_type == "Histogram":
                column = select_column(
                    "Select Numerical Column",
                    numeric_columns,
                    "Histogram_column",
                )
                if column is not None and column in numeric_columns:
                    values = df[column].dropna()
                    if values.empty:
                        st.info("There are no usable values for this chart.")
                    else:
                        fig, ax = plt.subplots()
                        ax.hist(values, bins=10)
                        ax.set_xlabel(column)
                        ax.set_ylabel("Frequency")
                        ax.set_title(f"Distribution of {column}")
                        st.pyplot(fig)
                        plt.close(fig)

            elif chart_type == "Scatter Plot":
                x_column = select_column(
                    "Select X-axis",
                    numeric_columns,
                    "Scatter Plot_x",
                )
                y_options = [
                    column for column in numeric_columns
                    if column != x_column
                ]
                y_column = select_column(
                    "Select Y-axis",
                    y_options,
                    "Scatter Plot_y",
                )
                if x_column is not None and y_column is not None:
                    if (
                        x_column not in numeric_columns
                        or y_column not in numeric_columns
                    ):
                        st.info("Choose numerical columns for both scatter plot axes.")
                    else:
                        chart_data = prepare_chart_data(
                            df, x_column, y_column
                        )
                        if chart_data.empty:
                            st.info("There are no usable values for this chart.")
                        else:
                            st.scatter_chart(
                                chart_data,
                                x="X value",
                                y="Y value",
                            )

    # ---------------- ASK YOUR DATA ----------------

    st.divider()
    st.header("Ask Your Data")
    st.caption("Your data has answers. Let's uncover them.")

    question = st.text_input(
        "What would you like to know?",
        placeholder="Ask a question about your dataset...",
        key="data_question"
    )

    if st.button("Analyze", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Finding insights in your data..."):
                result = analyze_question(question, df)
                combined = combined_revenue_result(question, df)
                if combined is not None:
                    result.rows, result.charts = combined
                if not result.has_results:
                    tasks = interpret_with_ai(question, df)
                    if tasks:
                        interpreted = execute_task_plan(df, tasks)
                        if interpreted.has_results:
                            result = interpreted
                        elif interpreted.message:
                            result.message = interpreted.message

            if result.has_results:
                result_table = pd.DataFrame(result.rows)
                if "Value" in result_table:
                    result_table["Value"] = [
                        format_result_value(value, str(insight), df)
                        for value, insight in zip(
                            result_table["Value"],
                            result_table["Insight"],
                        )
                    ]
                st.dataframe(result_table, hide_index=True, use_container_width=True)
                for title, chart_data in result.charts:
                    if len(chart_data) > 1:
                        fig, ax = plt.subplots()
                        ax.barh(chart_data["Category"], chart_data["Value"])
                        ax.set_title(title)
                        ax.set_xlabel("Value")
                        ax.invert_yaxis()
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
            else:
                st.info(
                    result.message
                    or "I couldn't answer that from this dataset. Try a "
                    "different question or check the available columns."
                )

else:
    st.info("Upload a CSV file to view its analysis and ask questions.")
