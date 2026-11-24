# Machine Learning & Data Visualization (DV) Lab Experiments

**Author:** Aviral Pandey  
**Repository:** [ML-DV-Lab-Exp---Aviral-Pandey](https://github.com/aviralp1910-bot/ML-DV-Lab-Exp---Aviral-Pandey)

---

## 📁 Repository Structure

```text
├── ML/
│   ├── .gitkeep                               # Git track placeholder for Machine Learning labs
│   └── README.md                              # ML module documentation
├── DV/
│   ├── exp1_getting_started_views.py          # Exp 1: Table Summary, Bar & Line Charts
│   ├── exp2_data_cleaning_transformation.py   # Exp 2: Missing Values, Types, & Calculated Fields
│   ├── exp3_bar_line_series.py                # Exp 3: Plotly Bar & Line Series with Tooltips
│   ├── exp4_interactive_dashboard.py          # Exp 4: Interactive Retail Dashboard (Dropdowns & Sliders)
│   ├── exp5_geo_visualization.py              # Exp 5: Choropleth & Symbol Maps (Plotly Express)
│   ├── exp6_pie_donut_charts.py               # Exp 6: Olympic Medals Pie & Donut Proportions
│   ├── exp7_heatmaps_treemaps.py              # Exp 7: Correlation Heatmap & Hierarchical Treemap
│   ├── exp8_time_series_forecasting.py        # Exp 8: Moving Averages & Holt-Winters Forecasting
│   └── outputs/                               # Generated High-Res PNGs & Interactive HTMLs
├── requirements.txt                           # Python package dependencies
├── .gitignore                                 # Git ignore patterns
└── README.md                                  # Repository documentation
```

---

## 📊 Data Visualization (DV) Lab Experiments Overview

| # | Experiment Script | Description & Methodologies | Visualization Engine | Output Artifacts |
|---|---|---|---|---|
| **1** | [`exp1_getting_started_views.py`](DV/exp1_getting_started_views.py) | **Getting Started Views:** Loads retail dataset, renders descriptive statistics summary table, category bar chart with annotations, and time-series trend line. | Matplotlib / Seaborn | `exp1_summary_table.png`<br>`exp1_bar_chart.png`<br>`exp1_line_chart.png` |
| **2** | [`exp2_data_cleaning_transformation.py`](DV/exp2_data_cleaning_transformation.py) | **Data Cleaning & Transformation:** Handles missing value imputation, snake_case column normalization, numeric/datetime type coercion, and feature engineering (Revenue, Net Profit, Margin %). | Pandas / Matplotlib | `exp2_missing_values_cleaning.png`<br>`exp2_calculated_fields.png` |
| **3** | [`exp3_bar_line_series.py`](DV/exp3_bar_line_series.py) | **Bar & Line Series with Interactive Tooltips:** Category bar chart with data labels, time-trend spline charts with range slider, and dual-axis combined Bar + Line figures. | Plotly Graph Objects / Express | `exp3_category_bar.html`<br>`exp3_trend_line.html`<br>`exp3_combined_bar_line.html` |
| **4** | [`exp4_interactive_dashboard.py`](DV/exp4_interactive_dashboard.py) | **Interactive Retail Dashboard:** Multi-panel executive dashboard featuring dynamic dropdown region filters (`updatemenus`), timeline range slider, KPI scorecards, and category breakdown. | Plotly Subplots | `exp4_retail_dashboard.html` |
| **5** | [`exp5_geo_visualization.py`](DV/exp5_geo_visualization.py) | **Geographic Visualization:** Global choropleth (filled) polygon map using ISO-3 codes and proportional symbol (bubble) map with custom projections and hover cards. | Plotly Express | `exp5_choropleth_map.html`<br>`exp5_symbol_map.html` |
| **6** | [`exp6_pie_donut_charts.py`](DV/exp6_pie_donut_charts.py) | **Pie & Donut Composition Charts:** Olympic games medal share comparison using exploded pie chart (Gold dominance) and donut chart with center KPI annotation (Total medals). | Plotly & Matplotlib | `exp6_pie_donut_charts.html`<br>`exp6_pie_donut_charts.png` |
| **7** | [`exp7_heatmaps_treemaps.py`](DV/exp7_heatmaps_treemaps.py) | **Heatmaps & Hierarchical Treemaps:** Pearson correlation matrix with triangular mask & diverging colormap (Seaborn), and Category > Subcategory > Region interactive Treemap (Plotly). | Seaborn & Plotly Express | `exp7_correlation_heatmap.png`<br>`exp7_hierarchical_treemap.html` |
| **8** | [`exp8_time_series_forecasting.py`](DV/exp8_time_series_forecasting.py) | **Time-Series Analysis & Forecasting:** Computes 7-day SMA, 30-day EMA, and Bollinger Bands. Fits statistical Holt-Winters Exponential Smoothing to forecast forward with 95% Confidence Intervals. | Statsmodels, Matplotlib & Plotly | `exp8_time_series_forecasting.png`<br>`exp8_time_series_forecasting.html` |

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites & Environment Setup
Clone this repository and create a Python virtual environment:
```bash
git clone https://github.com/aviralp1910-bot/ML-DV-Lab-Exp---Aviral-Pandey.git
cd ML-DV-Lab-Exp---Aviral-Pandey

# Optional: Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Run Experiments
All scripts are completely self-contained with built-in data generators and zero missing CSV dependencies:

```bash
# Run Experiment 1 (Basic Views)
python DV/exp1_getting_started_views.py

# Run Experiment 2 (Data Cleaning & Transformation)
python DV/exp2_data_cleaning_transformation.py

# Run Experiment 3 (Interactive Bar & Line Series)
python DV/exp3_bar_line_series.py

# Run Experiment 4 (Interactive Retail Dashboard)
python DV/exp4_interactive_dashboard.py

# Run Experiment 5 (Geographic Choropleth & Symbol Maps)
python DV/exp5_geo_visualization.py

# Run Experiment 6 (Pie & Donut Composition Charts)
python DV/exp6_pie_donut_charts.py

# Run Experiment 7 (Correlation Heatmaps & Treemaps)
python DV/exp7_heatmaps_treemaps.py

# Run Experiment 8 (Time-Series & Holt-Winters Forecasting)
python DV/exp8_time_series_forecasting.py
```

Generated static images (`.png`) and standalone interactive web dashboards (`.html`) are saved in the `DV/outputs/` directory.

---
*(C) Aviral Pandey - Data Visualization & Machine Learning Lab Coursework*
