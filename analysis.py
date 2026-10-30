import re
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import pandas as pd


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "give",
    "hai", "how", "in", "is", "ki", "ko", "me", "of", "on", "or", "our",
    "please", "show", "tell", "the", "to", "what", "which", "with", "you",
}
_MEASURE_ALIASES = {
    "revenue": {
        "amount", "earning", "earnings", "income", "kamai", "revenue",
        "sale", "sales", "turnover",
    },
    "quantity": {"quantity", "qty", "unit", "units", "volume"},
    "price": {
        "price", "rate", "unitprice", "sellingprice", "saleprice",
    },
}
_DIMENSION_ALIASES = {
    "city": {"city", "cities", "town", "location"},
    "product": {"product", "products", "item", "items", "sku"},
    "category": {"category", "categories", "type", "segment"},
    "region": {"region", "state", "country", "zone"},
    "customer": {"customer", "client", "buyer"},
    "date": {"date", "day", "month", "year", "time"},
}
_HIGH_WORDS = {
    "highest", "largest", "maximum", "max", "most", "top",
    "zyada", "adhik",
}
_LOW_WORDS = {"lowest", "smallest", "minimum", "min", "least"}
_TOTAL_PATTERN = re.compile(
    r"\b(total|sum|overall|kul|kitni|kitna|how much|in total)\b"
)
_COMPARE_PATTERN = re.compile(
    r"\b(compare|comparison|across|between|breakdown|distribution)\b"
)
_AVERAGE_PATTERN = re.compile(r"\b(average|avg|mean)\b")


@dataclass
class AnalysisResponse:
    rows: list[dict[str, Any]] = field(default_factory=list)
    charts: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    message: str = ""

    @property
    def has_results(self) -> bool:
        return bool(self.rows)


def _normalize(value: Any) -> str:
    return " ".join(_TOKEN_PATTERN.findall(str(value).lower()))


def _tokens(value: Any) -> set[str]:
    return set(_TOKEN_PATTERN.findall(str(value).lower())) - _STOP_WORDS


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {_normalize(phrase)} " in f" {_normalize(text)} "


def _column_match_score(question: str, column: Any) -> int:
    normalized_column = _normalize(column)
    if not normalized_column:
        return 0
    if _contains_phrase(question, normalized_column):
        return 20 + len(normalized_column.split())
    return len(_tokens(question) & _tokens(column)) * 3


def _has_revenue_intent(question: str) -> bool:
    return any(
        _contains_phrase(question, word)
        for word in (
            "revenue", "sales", "sale", "earning", "income", "kamai",
            "turnover",
        )
    )


def _measure_for_column(column: Any, kind: str) -> bool:
    name = _normalize(column).replace(" ", "")
    return any(alias in name for alias in _MEASURE_ALIASES[kind])


def _resolve_measure(
    question: str, frame: pd.DataFrame
) -> tuple[str, pd.Series] | tuple[None, None]:
    numeric_columns = [
        column for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not numeric_columns:
        return None, None

    query_tokens = _tokens(question)
    revenue_intent = _has_revenue_intent(question)
    scored = []
    for column in numeric_columns:
        column_tokens = _tokens(column)
        if revenue_intent:
            aliases = _MEASURE_ALIASES["revenue"]
            exact_metric_tokens = column_tokens & query_tokens & aliases
            score = 20 * len(exact_metric_tokens)
            if column_tokens & aliases and not exact_metric_tokens:
                score = 4
        else:
            score = _column_match_score(question, column)
            for alias_set in _MEASURE_ALIASES.values():
                if query_tokens & alias_set and column_tokens & alias_set:
                    score += 8
        scored.append((score, column))

    scored.sort(key=lambda item: item[0], reverse=True)
    if scored and scored[0][0] > 0:
        return str(scored[0][1]), frame[scored[0][1]]

    if revenue_intent:
        quantity_columns = [
            column for column in numeric_columns
            if _measure_for_column(column, "quantity")
        ]
        price_columns = [
            column for column in numeric_columns
            if _measure_for_column(column, "price")
        ]
        if len(quantity_columns) == 1 and len(price_columns) == 1:
            quantity = frame[quantity_columns[0]]
            unit_price = frame[price_columns[0]]
            return (
                f"{quantity_columns[0]} × {price_columns[0]}",
                quantity * unit_price,
            )
    return None, None


def _column_for_dimension(
    question: str, frame: pd.DataFrame, dimension: str
) -> Any | None:
    aliases = _DIMENSION_ALIASES[dimension]
    if not any(_contains_phrase(question, alias) for alias in aliases):
        return None

    candidates = []
    for column in frame.columns:
        normalized_tokens = _tokens(column)
        if normalized_tokens & aliases:
            candidates.append(column)
    if not candidates:
        return None
    candidates.sort(
        key=lambda column: _column_match_score(question, column),
        reverse=True,
    )
    return candidates[0]


def _requested_dimensions(
    question: str, frame: pd.DataFrame
) -> tuple[list[Any], list[str]]:
    found = []
    missing = []
    for dimension in _DIMENSION_ALIASES:
        if any(
            _contains_phrase(question, alias)
            for alias in _DIMENSION_ALIASES[dimension]
        ):
            column = _column_for_dimension(question, frame, dimension)
            if column is None:
                missing.append(dimension)
            elif column not in found:
                found.append(column)

    for column in frame.columns:
        normalized_column = _normalize(column)
        if (
            normalized_column
            and _contains_phrase(question, normalized_column)
            and column not in found
            and not pd.api.types.is_numeric_dtype(frame[column])
        ):
            found.append(column)
    if not found:
        group_match = re.search(
            r"\b(?:by|across|per)\s+([a-z0-9 ]+)",
            _normalize(question),
        )
        if group_match:
            group_phrase = group_match.group(1).strip()
            metric_words = set().union(*_MEASURE_ALIASES.values())
            phrase_tokens = _tokens(group_phrase)
            if phrase_tokens and not phrase_tokens <= metric_words:
                missing.append(group_phrase.split()[0])
    return found, missing


def _apply_filters(question: str, frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            normalized_column = _normalize(column)
            if not normalized_column or not _contains_phrase(question, normalized_column):
                continue
            comparison = re.search(
                rf"{re.escape(normalized_column)}\s*"
                r"(above|over|greater than|more than|below|under|less than|"
                r"at least|at most|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)",
                _normalize(question),
            )
            if comparison is None:
                continue
            operator, raw_value = comparison.groups()
            value = float(raw_value)
            if operator in {"above", "over", "greater than", "more than", ">"}:
                filtered = filtered[filtered[column] > value]
            elif operator in {"below", "under", "less than", "<"}:
                filtered = filtered[filtered[column] < value]
            elif operator in {"at least", ">="}:
                filtered = filtered[filtered[column] >= value]
            else:
                filtered = filtered[filtered[column] <= value]

    text = _normalize(question)
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].dropna().astype(str).unique()
        if len(values) > 10000:
            continue
        for value in values:
            normalized_value = _normalize(value)
            if not normalized_value or not _contains_phrase(text, normalized_value):
                continue
            if (
                _contains_phrase(text, f"in {normalized_value}")
                or _contains_phrase(text, f"for {normalized_value}")
                or _contains_phrase(text, f"where {normalized_value}")
                or re.search(
                    rf"\bwhere\s+{re.escape(_normalize(column))}\s+"
                    rf"(?:is|=)\s+{re.escape(normalized_value)}\b",
                    text,
                )
            ):
                filtered = filtered[
                    filtered[column].astype(str).str.casefold()
                    == str(value).casefold()
                ]
                break
    return filtered


def _format_category(value: Any) -> str:
    return "(Missing)" if pd.isna(value) else str(value)


def _currency_format(frame: pd.DataFrame, measure_name: str) -> str:
    text = measure_name.casefold()
    currency_markers = (
        ("₹", "₹%.2f"), ("inr", "₹%.2f"), ("rupee", "₹%.2f"),
        ("$", "$%.2f"), ("usd", "$%.2f"), ("€", "€%.2f"),
        ("eur", "€%.2f"), ("£", "£%.2f"), ("gbp", "£%.2f"),
    )
    for marker, format_string in currency_markers:
        if marker in text:
            return format_string
    column_text = " ".join(str(column).casefold() for column in frame.columns)
    for marker, format_string in currency_markers:
        if marker in column_text:
            return format_string
    for column in frame.select_dtypes(exclude="number").columns:
        samples = frame[column].dropna().astype(str).head(100)
        for marker, format_string in currency_markers:
            if samples.str.contains(re.escape(marker), case=False, regex=True).any():
                return format_string
    return "%.2f"


def format_result_value(value: Any, insight: str, frame: pd.DataFrame) -> Any:
    if not isinstance(value, Real) or isinstance(value, bool):
        return value
    if "count" in insight.casefold():
        return f"{int(value):,}" if float(value).is_integer() else f"{value:,.2f}"
    currency_format = _currency_format(frame, insight)
    if currency_format != "%.2f":
        return currency_format % value
    return f"{value:,.2f}"


def prepare_chart_data(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    aggregate: bool = False,
) -> pd.DataFrame:
    if aggregate:
        values = (
            frame[y_column].reset_index(drop=True).groupby(
                frame[x_column].reset_index(drop=True),
                dropna=False,
            ).mean()
        )
        return pd.DataFrame({
            "X value": values.index.to_numpy(),
            "Y value": values.to_numpy(),
        }).dropna()
    return pd.DataFrame({
        "X value": frame[x_column].to_numpy(),
        "Y value": frame[y_column].to_numpy(),
    }).dropna()


def _add_aggregate(
    response: AnalysisResponse,
    filtered: pd.DataFrame,
    measure: pd.Series,
    dimension: Any,
    how: str,
    title: str,
    order: str = "all",
    limit: int | None = None,
) -> None:
    grouped = (
        measure.loc[filtered.index]
        .groupby(filtered[dimension], dropna=False)
        .agg(how)
    )
    grouped = grouped.dropna().sort_values(ascending=(order == "low"))
    if limit is not None:
        grouped = grouped.head(limit)
    for category, value in grouped.items():
        response.rows.append({
            "Insight": title,
            "Category": _format_category(category),
            "Value": value,
        })
    chart_frame = pd.DataFrame({
        "Category": [_format_category(value) for value in grouped.index],
        "Value": grouped.to_numpy(),
    })
    if not chart_frame.empty:
        response.charts.append((title, chart_frame))


def _analyze_question_for_measure(
    question: str,
    frame: pd.DataFrame,
    measure_override: tuple[str, pd.Series] | None = None,
) -> AnalysisResponse:
    response = AnalysisResponse()
    frame = frame.reset_index(drop=True)
    normalized_question = _normalize(question)
    if not normalized_question:
        response.message = "Enter a question about the uploaded dataset."
        return response

    if any(
        _contains_phrase(normalized_question, phrase)
        for phrase in ("available columns", "list columns", "column names")
    ):
        response.rows = [
            {"Insight": "Available column", "Category": str(column), "Value": ""}
            for column in frame.columns
        ]
        return response

    filtered = _apply_filters(question, frame)
    measure_name, measure = (
        measure_override
        if measure_override is not None
        else _resolve_measure(question, frame)
    )
    if measure_override is not None:
        measure = measure.reset_index(drop=True)
    dimensions, missing_dimensions = _requested_dimensions(question, frame)
    if missing_dimensions:
        names = ", ".join(dict.fromkeys(missing_dimensions))
        response.message = (
            f"I couldn't find a {names} column in this dataset. "
            "Choose a dimension that is available in the uploaded file."
        )
        return response

    measure_is_requested = (
        _has_revenue_intent(question)
        or any(word in _tokens(question) for word in (
            "sales", "revenue", "amount", "price", "quantity", "qty",
            "average", "avg", "mean", "sum", "total", "profit",
        ))
        or any(_column_match_score(question, column) >= 20 for column in frame.select_dtypes(include="number").columns)
    )
    count_requested = any(
        word in _tokens(question)
        for word in ("count", "many", "records", "rows")
    )
    if count_requested and dimensions and not measure_is_requested:
        for dimension in dimensions:
            count_name = f"Count by {dimension}"
            grouped_counts = (
                filtered.groupby(dimension, dropna=False)
                .size()
                .sort_values(ascending=False)
            )
            for category, count in grouped_counts.items():
                response.rows.append({
                    "Insight": count_name,
                    "Category": _format_category(category),
                    "Value": int(count),
                })
            response.charts.append((
                count_name,
                pd.DataFrame({
                    "Category": [
                        _format_category(value)
                        for value in grouped_counts.index
                    ],
                    "Value": grouped_counts.to_numpy(),
                }),
            ))
        return response
    if measure_is_requested and measure is None:
        response.message = (
            "I couldn't identify a suitable numeric measure. Name a numeric "
            "column, or include both a quantity and unit-price column for "
            "revenue calculations."
        )
        return response

    if measure is None:
        if count_requested:
            response.rows.append({
                "Insight": "Record count",
                "Category": "",
                "Value": int(len(filtered)),
            })
            return response
        response.message = (
            "I couldn't identify the analysis you want. Try asking for a "
            "total, average, highest or lowest value, top items, or a "
            "comparison by a dataset column."
        )
        return response

    if filtered.empty:
        response.message = "No rows in this dataset match the requested filters."
        return response

    revenue_intent = _has_revenue_intent(question)
    measure_title = (
        "Revenue"
        if revenue_intent
        and (
            "×" in measure_name
            or _measure_for_column(measure_name, "revenue")
        )
        else measure_name
    )
    measure_group = measure.loc[filtered.index]
    top_match = re.search(r"\btop\s+(\d+)\b", normalized_question)
    top_limit = min(int(top_match.group(1)), 100) if top_match else None
    high_requested = bool(_tokens(question) & _HIGH_WORDS) or top_limit is not None
    low_requested = bool(_tokens(question) & _LOW_WORDS)
    compare_requested = bool(_COMPARE_PATTERN.search(normalized_question))
    total_requested = bool(_TOTAL_PATTERN.search(normalized_question))
    average_requested = bool(_AVERAGE_PATTERN.search(normalized_question))

    if total_requested:
        total = measure_group.sum(min_count=1)
        response.rows.append({
            "Insight": f"Total {measure_title}",
            "Category": "",
            "Value": total,
        })

    if dimensions:
        for dimension in dimensions:
            dimension_name = str(dimension)
            if high_requested or low_requested:
                direction = "low" if low_requested and not high_requested else "high"
                title = (
                    f"{'Lowest' if direction == 'low' else 'Highest'} "
                    f"{measure_title} by {dimension_name}"
                )
                _add_aggregate(
                    response, filtered, measure_group, dimension, "sum",
                    title, direction, top_limit or 1,
                )
            elif compare_requested:
                title = f"{measure_title} by {dimension_name}"
                _add_aggregate(
                    response, filtered, measure_group, dimension, "sum",
                    title,
                )
            else:
                title = f"{'Average' if average_requested else 'Total'} {measure_title} by {dimension_name}"
                _add_aggregate(
                    response, filtered, measure_group, dimension,
                    "mean" if average_requested else "sum",
                    title,
                )
    elif not total_requested:
        operation = (
            "mean" if average_requested
            else "max" if high_requested
            else "min" if low_requested
            else "sum" if total_requested
            else None
        )
        if operation is None:
            response.message = (
                "I couldn't determine whether you want a total, average, "
                "minimum, or maximum. Please rephrase the question."
            )
            return response
        value = getattr(measure_group, operation)()
        response.rows.append({
            "Insight": f"{operation.title()} {measure_title}",
            "Category": "",
            "Value": value,
        })

    if not response.rows:
        response.message = (
            "I understood the metric but couldn't identify a supported "
            "calculation. Please name a grouping such as a city or product."
        )
    return response


def analyze_question(question: str, frame: pd.DataFrame) -> AnalysisResponse:
    normalized_question = _normalize(question)
    numeric_columns = [
        column for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    mentioned_columns = [
        column for column in numeric_columns
        if _contains_phrase(normalized_question, _normalize(column))
    ]
    has_multiple_metrics = False
    if len(mentioned_columns) > 1:
        positions = sorted(
            (
                normalized_question.find(_normalize(column)),
                normalized_question.find(_normalize(column))
                + len(_normalize(column)),
            )
            for column in mentioned_columns
        )
        has_multiple_metrics = any(
            re.search(r"\b(and|or|aur)\b", normalized_question[left[1]:right[0]])
            for left, right in zip(positions, positions[1:])
        )

    if has_multiple_metrics:
        combined = AnalysisResponse()
        for column in mentioned_columns:
            result = _analyze_question_for_measure(
                question, frame, (str(column), frame[column])
            )
            if not result.has_results:
                return result
            combined.rows.extend(result.rows)
            combined.charts.extend(result.charts)
        return combined

    return _analyze_question_for_measure(question, frame)


def execute_task_plan(
    frame: pd.DataFrame, tasks: Any
) -> AnalysisResponse:
    response = AnalysisResponse()
    if not isinstance(tasks, list) or not tasks or len(tasks) > 12:
        response.message = "The requested analysis could not be interpreted."
        return response

    working = frame.reset_index(drop=True)
    for task in tasks:
        if not isinstance(task, dict):
            response.message = "The requested analysis could not be interpreted."
            return response

        operation = task.get("operation")
        column = task.get("column")
        group_by = task.get("group_by")
        if not isinstance(operation, str) or operation not in {
            "total", "mean", "maximum", "minimum", "group_sum",
            "group_mean", "top", "compare", "count",
        }:
            response.message = "The requested analysis could not be interpreted."
            return response

        if operation != "count":
            if (
                not isinstance(column, str)
                or column not in working.columns
                or not pd.api.types.is_numeric_dtype(working[column])
            ):
                response.message = (
                    "A numeric column needed for this analysis is not "
                    "available in the dataset."
                )
                return response

        if operation in {"group_sum", "group_mean", "top", "compare", "count"}:
            if not isinstance(group_by, str) or group_by not in working.columns:
                response.message = (
                    "A grouping column needed for this analysis is not "
                    "available in the dataset."
                )
                return response
        elif group_by is not None:
            response.message = "The requested analysis could not be interpreted."
            return response

        filters = task.get("filters", [])
        if not isinstance(filters, list) or len(filters) > 10:
            response.message = "The requested filters could not be interpreted."
            return response
        filtered = working
        for item in filters:
            if not isinstance(item, dict):
                response.message = "The requested filters could not be interpreted."
                return response
            filter_column = item.get("column")
            operator = item.get("operator")
            value = item.get("value")
            if not isinstance(filter_column, str) or filter_column not in working.columns:
                response.message = (
                    "A column needed for this filter is not available in "
                    "the dataset."
                )
                return response
            if operator == "equals":
                filtered = filtered[
                    filtered[filter_column].astype(str).str.casefold()
                    == str(value).casefold()
                ]
            elif (
                isinstance(operator, str)
                and operator in {
                    "greater_than", "less_than", "at_least", "at_most"
                }
                and pd.api.types.is_numeric_dtype(filtered[filter_column])
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                if operator == "greater_than":
                    filtered = filtered[filtered[filter_column] > value]
                elif operator == "less_than":
                    filtered = filtered[filtered[filter_column] < value]
                elif operator == "at_least":
                    filtered = filtered[filtered[filter_column] >= value]
                else:
                    filtered = filtered[filtered[filter_column] <= value]
            else:
                response.message = (
                    "A filter in this question could not be applied safely."
                )
                return response

        if filtered.empty:
            response.message = "No rows in this dataset match the requested filters."
            return response

        if operation == "count":
            counts = filtered.groupby(group_by, dropna=False).size()
            title = f"Count by {group_by}"
            for category, value in counts.items():
                response.rows.append({
                    "Insight": title,
                    "Category": _format_category(category),
                    "Value": int(value),
                })
            chart = pd.DataFrame({
                "Category": [_format_category(value) for value in counts.index],
                "Value": counts.to_numpy(),
            })
            if not chart.empty:
                response.charts.append((title, chart))
            continue

        values = filtered[column]
        if operation == "total":
            value = values.sum(min_count=1)
            response.rows.append({
                "Insight": f"Total {column}",
                "Category": "",
                "Value": value,
            })
        elif operation in {"mean", "maximum", "minimum"}:
            method = {
                "mean": "mean",
                "maximum": "max",
                "minimum": "min",
            }[operation]
            response.rows.append({
                "Insight": f"{method.title()} {column}",
                "Category": "",
                "Value": getattr(values, method)(),
            })
        else:
            how = "mean" if operation == "group_mean" else "sum"
            grouped = values.groupby(filtered[group_by], dropna=False).agg(
                how if how == "mean" else lambda series: series.sum(min_count=1)
            ).dropna()
            ascending = False
            grouped = grouped.sort_values(ascending=ascending)
            if operation == "top":
                requested_limit = task.get("limit", 5)
                if (
                    not isinstance(requested_limit, int)
                    or isinstance(requested_limit, bool)
                    or requested_limit < 1
                ):
                    response.message = "The requested ranking could not be interpreted."
                    return response
                grouped = grouped.head(min(requested_limit, 100))
                prefix = "Top"
            elif operation == "compare":
                prefix = "Comparison of"
            else:
                prefix = "Average"
            title = f"{prefix} {column} by {group_by}"
            for category, value in grouped.items():
                response.rows.append({
                    "Insight": title,
                    "Category": _format_category(category),
                    "Value": value,
                })
            chart = pd.DataFrame({
                "Category": [_format_category(value) for value in grouped.index],
                "Value": grouped.to_numpy(),
            })
            if not chart.empty:
                response.charts.append((title, chart))

    if not response.rows:
        response.message = "No results were produced for this question."
    return response
