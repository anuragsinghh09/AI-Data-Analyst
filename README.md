# AI Data Analyst

Repository: AI-Data-Analyst

AI Data Analyst is a Streamlit application for exploring CSV data through dataset summaries, interactive charts, and natural-language questions. Supported questions are analyzed locally with Pandas; Google Gemini can optionally interpret questions that the local analysis does not recognize.

## Problem Statement

Business users often struggle to extract actionable insights from raw sales data due to the complexity of data analysis tools and the need for programming knowledge. There is a need for an intuitive, interactive tool that can answer business questions and visualize data instantly.

## Project Objectives

- Enable users to upload sales data and analyze it interactively.
- Provide natural language question answering using Gemini AI.
- Generate charts and summary tables for key business metrics.
- Make data analysis accessible to non-technical users.

## Key Features

- Upload and analyze CSV sales data.
- Review row and column counts, duplicate rows, data preview, numeric summaries, missing values, and column types.
- Create bar, line, histogram, scatter, and pie charts from uploaded columns.
- Ask natural-language questions for supported totals, averages, rankings, comparisons, and groupings.
- Optionally use Gemini to interpret unsupported questions; Pandas validates and calculates the resulting operations.

## Technologies Used

- Python
- Streamlit
- Pandas
- Matplotlib
- Google Gemini API (via the `google-genai` SDK; optional)

## How the Application Works

1. User uploads a CSV file.
2. The app displays dataset diagnostics and lets the user choose columns and chart types.
3. Supported questions are parsed locally; if local analysis cannot answer a question and a Gemini key is configured, only the question and column names/types are sent to Gemini for interpretation.
4. The app validates the interpreted operations and uses Pandas to calculate results from the uploaded data.

## Project Folder Structure

```
AI-Data-Analyst/
├── app.py                # Main Streamlit application
├── analysis.py           # Data analysis logic
├── requirements.txt      # Python dependencies
├── START.bat             # Windows batch file to launch the app
├── sample_data/
│   └── sample_sales_data.csv  # Fabricated demonstration data
├── screenshots/          # Add screenshots here for documentation
├── tests/
│   └── test_analysis.py  # Unit tests for analysis logic
├── .streamlit/
│   └── secrets.toml.example  # Safe configuration template
└── .gitignore            # Git ignore rules
```

The local `app_backup.py`, `sales_data.csv`, virtual environments, caches, and real secrets are intentionally excluded from the public repository.

## Installation Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/ai-data-analyst.git
   cd ai-data-analyst
   ```
2. **Create a virtual environment (recommended):**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # Or: source .venv/bin/activate  # On macOS/Linux
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Configuring the Gemini API Key Safely

1. **Create a `.streamlit/secrets.toml` file:**
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
   - Add your actual Gemini API key in place of the placeholder.
   - **Never commit your real API key to Git or share it publicly.**

2. **Alternative:** Set `GEMINI_API_KEY` in your environment. In PowerShell, for the current terminal session, run:
   ```powershell
   $env:GEMINI_API_KEY = "your-gemini-api-key"
   ```

The app checks Streamlit secrets first and then the environment variable. Gemini is optional; the local question analyzer remains available without it. When Gemini is used, the app sends the question and dataset column names/types for interpretation, not the uploaded rows.

## Running the Application

- **With Streamlit:**
  ```sh
  streamlit run app.py
  ```
- **Or use the provided batch file (Windows):**
  ```sh
  START.bat
  ```

## Screenshots

Add screenshots to the `screenshots/` folder. Example paths:

- `screenshots/dashboard.png`
- `screenshots/visualization.png`
- `screenshots/ask-your-data.png`

## Current Limitations & Future Improvements

- Requires a valid Gemini API key for AI-powered features.
- Supports CSV files only; date-aware filtering and broader file formats are not implemented.
- Gemini interpretation requires network access and a valid API key; its interpretation may not cover every question.
- No user authentication or multi-user support.
- Future: Add broader data formats, date-aware analysis, more chart options, and deployment guidance.

---

**For questions or contributions, please open an issue or pull request on GitHub.**
