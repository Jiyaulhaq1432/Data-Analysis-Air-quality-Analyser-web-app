import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
import glob
import os
import warnings
warnings.filterwarnings("ignore")

# ── ML imports ────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, f1_score,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beijing Air Quality Analyser",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #0d1117; }
.block-container { padding: 1.5rem 2rem; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.hero-banner {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 1.5rem;
    border: 1px solid rgba(99,202,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-title {
    font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700;
    color: #63caff; margin: 0 0 0.5rem 0; letter-spacing: -1px;
}
.hero-sub { color: #8ab4c9; font-size: 1rem; margin: 0; }

.metric-card {
    background: linear-gradient(135deg, #1a2332 0%, #1e2d40 100%);
    border-radius: 12px; padding: 1.2rem;
    border: 1px solid rgba(99,202,255,0.15);
    text-align: center; margin: 0.3rem 0;
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #63caff; font-family: 'Space Mono', monospace; }
.metric-lbl { font-size: 0.75rem; color: #8ab4c9; text-transform: uppercase; letter-spacing: 1px; }

.section-header {
    font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #63caff;
    border-left: 3px solid #63caff; padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}
.info-box {
    background: rgba(99,202,255,0.08); border: 1px solid rgba(99,202,255,0.2);
    border-radius: 10px; padding: 1rem 1.2rem; margin: 0.8rem 0;
    color: #cdd9e5; font-size: 0.9rem;
}
.stButton>button {
    background: linear-gradient(135deg, #1a6b8a, #0d4f6b); color: white;
    border: 1px solid rgba(99,202,255,0.3); border-radius: 8px;
    font-family: 'Space Mono', monospace; font-size: 0.85rem;
    padding: 0.5rem 1.2rem; width: 100%; transition: all 0.2s;
}
.stButton>button:hover { border-color: #63caff; box-shadow: 0 0 12px rgba(99,202,255,0.3); }

[data-testid="stSidebar"] {
    background: #0d1822; border-right: 1px solid rgba(99,202,255,0.15);
}
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0f3d52, #1a6b8a);
    color: #63caff; font-family: 'Space Mono', monospace; font-size: 0.8rem;
    padding: 0.3rem 0.7rem; border-radius: 20px;
    border: 1px solid rgba(99,202,255,0.3); margin: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Plot theme helper ─────────────────────────────────────────────────────────
PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,24,34,0.8)",
    font=dict(family="DM Sans", color="#cdd9e5"),
    margin=dict(l=40, r=20, t=50, b=40),
)

def style_fig(fig, title=""):
    fig.update_layout(**PLOT_THEME, title=dict(
        text=title, font=dict(family="Space Mono", size=14, color="#63caff")
    ))
    fig.update_xaxes(gridcolor="rgba(99,202,255,0.08)", linecolor="rgba(99,202,255,0.2)")
    fig.update_yaxes(gridcolor="rgba(99,202,255,0.08)", linecolor="rgba(99,202,255,0.2)")
    return fig


# ── Cached helpers ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_and_combine(file_contents: list) -> pd.DataFrame | None:
    """Load and merge multiple CSV/Excel files into one DataFrame."""
    dfs = []
    for name, content in file_contents:
        try:
            nl = name.lower()
            if nl.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
            elif nl.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(content))
            else:
                continue
            station = name.replace("PRSA_Data_", "").split("_20")[0]
            if "station" not in df.columns:
                df["station"] = station
            dfs.append(df)
        except Exception as e:
            st.warning(f"Could not read {name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame | None:
    """
    Load CSV files from the cloned GitHub repository.

    Search order (first match wins):
      1. /content/Data-Analysis-Air-quality-Analyser-web-app/Data Files/
      2. /content/Data-Analysis-Air-quality-Analyser-web-app/data/
      3. /content/Data-Analysis-Air-quality-Analyser-web-app/   (root)
      4. <folder containing app.py>/data/
      5. <folder containing app.py>/
    """
    REPO_NAME = "Data-Analysis-Air-quality-Analyser-web-app"

    search_dirs = [
        # ── Colab-specific: cloned repo paths ─────────────────────────────
        f"/content/{REPO_NAME}/Data Files",
        f"/content/{REPO_NAME}/data",
        f"/content/{REPO_NAME}",
        # ── Generic: next to app.py ────────────────────────────────────────
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data Files"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        os.path.dirname(os.path.abspath(__file__)),
    ]

    files = []
    used_dir = None
    for d in search_dirs:
        found = sorted(glob.glob(os.path.join(d, "*.csv")))
        if found:
            files    = found
            used_dir = d
            break

    if not files:
        return None

    # Show which folder was used (visible in Streamlit sidebar)
    st.sidebar.caption(f"📂 Data source: `{used_dir}` ({len(files)} file(s))")

    dfs = []
    for path in files:
        try:
            part    = pd.read_csv(path)
            name    = os.path.basename(path)
            station = name.replace("PRSA_Data_", "").split("_20")[0].replace(".csv", "")
            if "station" not in part.columns:
                part["station"] = station
            dfs.append(part)
        except Exception as e:
            st.warning(f"Could not read {os.path.basename(path)}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else None
@st.cache_data(show_spinner=False)
def load_default_data():
    """
    Finds CSVs from the cloned GitHub repo in Colab.
    Search order: Data Files/ → data/ → repo root → next to app.py
    """
    REPO_NAME   = "Data-Analysis-Air-quality-Analyser-web-app"
    search_dirs = [
        f"/content/{REPO_NAME}/Data Files",
        f"/content/{REPO_NAME}/data",
        f"/content/{REPO_NAME}",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data Files"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        os.path.dirname(os.path.abspath(__file__)),
    ]
    files, used_dir = [], None
    for d in search_dirs:
        found = sorted(glob.glob(os.path.join(d, "*.csv")))
        if found:
            files, used_dir = found, d
            break
    if not files:
        return None
    st.sidebar.caption(f"📂 Source: {used_dir}  ({len(files)} files)")
    dfs = []
    for path in files:
        try:
            part    = pd.read_csv(path)
            name    = os.path.basename(path)
            station = name.replace("PRSA_Data_","").split("_20")[0].replace(".csv","")
            if "station" not in part.columns:
                part["station"] = station
            dfs.append(part)
        except Exception as e:
            st.warning(f"Could not read {os.path.basename(path)}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None


@st.cache_data(show_spinner=False)
def preprocess(_df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline — cached so navigation stays instant."""
    df = _df.copy()
    dt_cols = [c for c in ["year", "month", "day", "hour"] if c in df.columns]
    if len(dt_cols) == 4:
        df["datetime"] = pd.to_datetime(df[dt_cols])
        df["season"] = df["month"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring",  4: "Spring",  5: "Spring",
            6: "Summer",  7: "Summer",  8: "Summer",
            9: "Autumn",  10: "Autumn", 11: "Autumn",
        })
        df["day_of_week"] = df["datetime"].dt.day_name()
        df["is_weekend"] = (df["datetime"].dt.dayofweek >= 5).astype(int)
    if "PM2.5" in df.columns:
        bins   = [0, 12, 35.4, 55.4, 150.4, 250.4, float("inf")]
        labels = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]
        df["AQI_Cat"] = pd.cut(df["PM2.5"], bins=bins, labels=labels)
    if "wd" in df.columns:
        df["wd"] = df["wd"].astype("category")
    return df


def get_numeric(df):     return df.select_dtypes(include=np.number).columns.tolist()
def get_categorical(df): return df.select_dtypes(include=["object", "category"]).columns.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem'>
        <div style='font-family:Space Mono;font-size:1.5rem;color:#63caff;'>🌫️ AQA</div>
        <div style='color:#8ab4c9;font-size:0.75rem;letter-spacing:2px;'>AIR QUALITY ANALYSER</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATE",
        ["🏠 Home", "📋 Dataset", "📍 EDA", "📊 Visualisations", "🧪 Modelling", "📝 Report"],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown(
        "<div style='color:#8ab4c9;font-size:0.75rem;font-family:Space Mono;'>UPLOAD DATA</div>",
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Upload CSV/Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown(
        "<div class='info-box'>Optionally upload your own files to override the default dataset."
        " Supports CSV &amp; Excel.</div>",
        unsafe_allow_html=True,
    )

# ── Resolve active DataFrame ──────────────────────────────────────────────────
# Priority: uploaded files > repo default data
df_raw = None
df     = None

if uploaded_files:
    # User uploaded something — use that
    file_contents = [(f.name, f.read()) for f in uploaded_files]
    df_raw = _load_and_combine(file_contents)
else:
    # Fall back to the CSV files already in the repo
    df_raw = load_default_data()

if df_raw is not None:
    df = preprocess(df_raw)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>Beijing Air Quality<br>Analysis Platform</div>
        <p class='hero-sub'>CMP7005 · Programming for Data Analysis · Cardiff Metropolitan University</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip(
        [c1, c2, c3, c4],
        [("12", "Monitoring Stations"), ("35,064", "Hourly Records/Station"),
         ("11", "Variables"), ("4 Years", "2013 – 2017")],
    ):
        col.markdown(
            f"<div class='metric-card'><div class='metric-val'>{val}</div>"
            f"<div class='metric-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-header'>📋 Project Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    This platform supports the full data analysis pipeline for the Beijing PM2.5 Air Quality
    dataset as required by <b>CMP7005 PRAC1</b>. Navigate through the five sections using the sidebar:<br><br>
    <b>📋 Dataset</b> – Inspect raw data, check data types and missing values.<br>
    <b>📍 EDA</b> – Automated Exploratory Data Analysis with statistical summaries.<br>
    <b>📊 Visualisations</b> – 10+ interactive chart types: distributions, correlations, temporal plots.<br>
    <b>🧪 Modelling</b> – Select regression or classification models, tune and evaluate performance.<br>
    <b>📝 Report</b> – Summary of findings and key insights.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🌐 Dataset Variables</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Air Quality Pollutants**
| Variable | Description |
|---|---|
| PM2.5 | Fine particulate matter (μg/m³) |
| PM10 | Coarse particulate matter (μg/m³) |
| SO2 | Sulphur dioxide (μg/m³) |
| NO2 | Nitrogen dioxide (μg/m³) |
| CO | Carbon monoxide (μg/m³) |
| O3 | Ozone (μg/m³) |
        """)
    with col2:
        st.markdown("""
**Meteorological Variables**
| Variable | Description |
|---|---|
| TEMP | Temperature (°C) |
| PRES | Atmospheric pressure (hPa) |
| DEWP | Dew point temperature (°C) |
| RAIN | Precipitation (mm) |
| WSPM | Wind speed (m/s) |
        """)

    if df is not None:
        source = "uploaded files" if uploaded_files else "default repo data"
        n_stations = df["station"].nunique() if "station" in df.columns else "?"
        st.success(
            f"✅ Data ready — **{len(df):,} rows** across **{n_stations} station(s)** "
            f"*(source: {source})*"
        )
    else:
        st.warning(
            "⚠️ No data found. Place CSV files in a `data/` folder next to app.py, "
            "or upload files using the sidebar."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset":
    st.markdown("<h2 style='color:#63caff;font-family:Space Mono'>Dataset Explorer</h2>",
                unsafe_allow_html=True)

    if df is None:
        st.markdown(
            "<div class='info-box'>⚠️ No data found. Place CSV files in the <b>data/</b> folder "
            "next to app.py, or upload files using the sidebar.</div>",
            unsafe_allow_html=True,
        )
    else:
        num_cols = get_numeric(df)

        c1, c2, c3, c4 = st.columns(4)
        for col, (val, lbl) in zip(
            [c1, c2, c3, c4],
            [(f"{len(df):,}", "Rows"), (str(df.shape[1]), "Columns"),
             (f"{df.isnull().sum().sum():,}", "Missing Values"),
             (f"{df.duplicated().sum():,}", "Duplicates")],
        ):
            col.markdown(
                f"<div class='metric-card'><div class='metric-val'>{val}</div>"
                f"<div class='metric-lbl'>{lbl}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='section-header'>Raw Data Preview</div>", unsafe_allow_html=True)
        n = st.slider("Rows to display", 5, 100, 20)
        st.dataframe(df.head(n), use_container_width=True)

        st.markdown("<div class='section-header'>Column Information</div>", unsafe_allow_html=True)
        info_df = pd.DataFrame({
            "Column":   df.columns,
            "Dtype":    df.dtypes.values,
            "Non-Null": df.count().values,
            "Null":     df.isnull().sum().values,
            "Null %":   (df.isnull().mean() * 100).round(2).values,
            "Unique":   df.nunique().values,
        })
        st.dataframe(info_df, use_container_width=True)

        st.markdown("<div class='section-header'>Statistical Summary</div>", unsafe_allow_html=True)
        st.dataframe(
            df[num_cols].describe().T.style.background_gradient(cmap="Blues"),
            use_container_width=True,
        )

        st.markdown("<div class='section-header'>Missing Values Chart</div>", unsafe_allow_html=True)
        miss = df[num_cols].isnull().mean().sort_values(ascending=False)
        miss = miss[miss > 0]
        if len(miss) > 0:
            fig = px.bar(miss, orientation="h",
                         labels={"index": "Column", "value": "Missing %"},
                         color=miss.values, color_continuous_scale="Blues")
            fig = style_fig(fig, "Missing Values (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values detected in numeric columns!")

        if "station" in df.columns:
            st.markdown("<div class='section-header'>Station Distribution</div>",
                        unsafe_allow_html=True)
            vc  = df["station"].value_counts()
            fig = px.bar(vc, labels={"index": "Station", "value": "Record Count"},
                         color=vc.values, color_continuous_scale="Blues")
            fig = style_fig(fig, "Records per Station")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>Download Cleaned Dataset</div>",
                    unsafe_allow_html=True)
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        st.download_button("⬇️ Download as CSV", buf.getvalue(), "cleaned_data.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📍 EDA":
    st.markdown("<h2 style='color:#63caff;font-family:Space Mono'>Exploratory Data Analysis</h2>",
                unsafe_allow_html=True)

    if df is None:
        st.markdown("<div class='info-box'>⚠️ No data loaded.</div>", unsafe_allow_html=True)
    else:
        num_cols = get_numeric(df)
        tab1, tab2, tab3 = st.tabs(["📐 Univariate", "📈 Bivariate", "🌐 Multivariate"])

        # ── Univariate ──────────────────────────────────────────────────────
        with tab1:
            st.markdown("<div class='section-header'>Univariate Analysis</div>",
                        unsafe_allow_html=True)
            col        = st.selectbox("Select variable", num_cols, key="uni_col")
            chart_type = st.selectbox("Chart type",
                ["Histogram", "Box Plot", "Violin", "KDE + Rug", "ECDF"], key="uni_chart")
            series = df[col].dropna()

            if chart_type == "Histogram":
                bins = st.slider("Bins", 10, 100, 30)
                fig  = px.histogram(df, x=col, nbins=bins,
                                    color="station" if "station" in df.columns else None,
                                    marginal="box",
                                    color_discrete_sequence=px.colors.sequential.Blues_r)
                fig  = style_fig(fig, f"Distribution of {col}")
            elif chart_type == "Box Plot":
                grp = "station" if "station" in df.columns else None
                fig = px.box(df, y=col, x=grp, color=grp,
                             color_discrete_sequence=px.colors.sequential.Blues_r)
                fig = style_fig(fig, f"Box Plot of {col}")
            elif chart_type == "Violin":
                grp = "station" if "station" in df.columns else None
                fig = px.violin(df, y=col, x=grp, color=grp, box=True,
                                color_discrete_sequence=px.colors.sequential.Blues_r)
                fig = style_fig(fig, f"Violin Plot of {col}")
            elif chart_type == "KDE + Rug":
                fig = px.histogram(df, x=col, marginal="rug",
                                   color_discrete_sequence=["#63caff"])
                fig = style_fig(fig, f"KDE of {col}")
            else:
                fig = px.ecdf(df, x=col,
                              color="station" if "station" in df.columns else None,
                              color_discrete_sequence=px.colors.sequential.Blues_r)
                fig = style_fig(fig, f"ECDF of {col}")

            st.plotly_chart(fig, use_container_width=True)

            desc = series.describe()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean",     f"{desc['mean']:.2f}")
            c2.metric("Std Dev",  f"{desc['std']:.2f}")
            c3.metric("Median",   f"{series.median():.2f}")
            c4.metric("Skewness", f"{series.skew():.3f}")

        # ── Bivariate ───────────────────────────────────────────────────────
        with tab2:
            st.markdown("<div class='section-header'>Bivariate Analysis</div>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            xvar = c1.selectbox("X variable", num_cols, key="biv_x")
            yvar = c2.selectbox("Y variable", [c for c in num_cols if c != xvar], key="biv_y")
            chart = st.selectbox("Chart type",
                ["Scatter", "Scatter + Trend", "Hexbin Density", "Joint KDE",
                 "Bar (mean by group)"], key="biv_chart")

            color_options = ["None"]
            for c_opt in ["station", "AQI_Cat", "season"]:
                if c_opt in df.columns:
                    color_options.append(c_opt)
            color_by  = st.selectbox("Colour by", color_options, key="biv_color")
            color_arg = None if color_by == "None" else color_by
            sample    = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df

            if chart == "Scatter":
                fig = px.scatter(sample, x=xvar, y=yvar, color=color_arg, opacity=0.5,
                                 color_discrete_sequence=px.colors.sequential.Blues_r)
                fig = style_fig(fig, f"{yvar} vs {xvar}")
            elif chart == "Scatter + Trend":
                fig = px.scatter(sample, x=xvar, y=yvar, color=color_arg,
                                 trendline="ols", opacity=0.5,
                                 color_discrete_sequence=px.colors.sequential.Blues_r)
                fig = style_fig(fig, f"{yvar} vs {xvar} with Trend")
            elif chart == "Hexbin Density":
                fig = px.density_heatmap(sample, x=xvar, y=yvar,
                                         nbinsx=40, nbinsy=40,
                                         color_continuous_scale="Blues")
                fig = style_fig(fig, f"Density: {yvar} vs {xvar}")
            elif chart == "Joint KDE":
                fig = px.density_contour(sample, x=xvar, y=yvar, color=color_arg,
                                         color_discrete_sequence=px.colors.sequential.Blues_r)
                fig.update_traces(contours_coloring="fill", contours_showlabels=True)
                fig = style_fig(fig, f"Joint KDE: {yvar} vs {xvar}")
            else:
                grp = color_arg or ("station" if "station" in df.columns else None)
                if grp:
                    agg = df.groupby(grp)[[xvar, yvar]].mean().reset_index()
                    fig = px.bar(agg, x=grp, y=yvar, color=grp,
                                 color_discrete_sequence=px.colors.sequential.Blues_r)
                    fig = style_fig(fig, f"Mean {yvar} by {grp}")
                else:
                    st.warning("No grouping variable available.")
                    fig = go.Figure()

            st.plotly_chart(fig, use_container_width=True)
            corr = df[[xvar, yvar]].dropna().corr().iloc[0, 1]
            st.markdown(f"<span class='score-badge'>Pearson r = {corr:.4f}</span>",
                        unsafe_allow_html=True)

        # ── Multivariate ────────────────────────────────────────────────────
        with tab3:
            st.markdown("<div class='section-header'>Multivariate Analysis</div>",
                        unsafe_allow_html=True)
            analysis = st.selectbox("Analysis type",
                ["Correlation Heatmap", "Pairplot (Scatter Matrix)", "Parallel Coordinates",
                 "3D Scatter", "Bubble Chart", "Radar Chart (by station)"], key="multi_type")
            sel_cols = st.multiselect("Select variables", num_cols,
                default=num_cols[:min(6, len(num_cols))], key="multi_cols")

            if len(sel_cols) < 2:
                st.warning("Select at least 2 variables.")
            else:
                if analysis == "Correlation Heatmap":
                    corr_m = df[sel_cols].corr()
                    fig    = px.imshow(corr_m, text_auto=".2f",
                                       color_continuous_scale="RdBu_r",
                                       aspect="auto", zmin=-1, zmax=1)
                    fig    = style_fig(fig, "Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)

                elif analysis == "Pairplot (Scatter Matrix)":
                    sample    = df.sample(min(2000, len(df)))
                    color_dim = "station" if "station" in df.columns else None
                    fig       = px.scatter_matrix(sample, dimensions=sel_cols[:6],
                                                   color=color_dim,
                                                   color_discrete_sequence=px.colors.sequential.Blues_r)
                    fig.update_traces(diagonal_visible=True, showupperhalf=False)
                    fig       = style_fig(fig, "Scatter Matrix")
                    st.plotly_chart(fig, use_container_width=True)

                elif analysis == "Parallel Coordinates":
                    sample = df.sample(min(3000, len(df)))
                    fig    = px.parallel_coordinates(sample, dimensions=sel_cols,
                                                     color=sel_cols[0],
                                                     color_continuous_scale="Blues")
                    fig    = style_fig(fig, "Parallel Coordinates")
                    st.plotly_chart(fig, use_container_width=True)

                elif analysis == "3D Scatter":
                    if len(sel_cols) >= 3:
                        ca, cb, cc = st.columns(3)
                        ax = ca.selectbox("X", sel_cols, key="3dx")
                        ay = cb.selectbox("Y", sel_cols, index=1, key="3dy")
                        az = cc.selectbox("Z", sel_cols, index=2, key="3dz")
                        sample    = df.sample(min(3000, len(df)))
                        color_dim = "station" if "station" in df.columns else None
                        fig       = px.scatter_3d(sample, x=ax, y=ay, z=az, color=color_dim,
                                                  opacity=0.6,
                                                  color_discrete_sequence=px.colors.sequential.Blues_r)
                        fig       = style_fig(fig, f"3D: {ax} / {ay} / {az}")
                        st.plotly_chart(fig, use_container_width=True)

                elif analysis == "Bubble Chart":
                    ca, cb, cc = st.columns(3)
                    ax = ca.selectbox("X",    sel_cols, key="bx")
                    ay = cb.selectbox("Y",    sel_cols, index=min(1, len(sel_cols) - 1), key="by")
                    az = cc.selectbox("Size", sel_cols, index=min(2, len(sel_cols) - 1), key="bz")
                    sample    = df.sample(min(2000, len(df)))
                    color_dim = "station" if "station" in df.columns else None
                    fig       = px.scatter(sample, x=ax, y=ay, size=az, color=color_dim,
                                           size_max=20, opacity=0.7,
                                           color_discrete_sequence=px.colors.sequential.Blues_r)
                    fig       = style_fig(fig, "Bubble Chart")
                    st.plotly_chart(fig, use_container_width=True)

                elif analysis == "Radar Chart (by station)":
                    if "station" in df.columns:
                        avg  = df.groupby("station")[sel_cols].mean()
                        norm = (avg - avg.min()) / (avg.max() - avg.min() + 1e-9)
                        fig  = go.Figure()
                        for s in norm.index:
                            fig.add_trace(go.Scatterpolar(
                                r=norm.loc[s].tolist() + [norm.loc[s].tolist()[0]],
                                theta=sel_cols + [sel_cols[0]],
                                fill="toself", name=s, opacity=0.7,
                            ))
                        fig.update_layout(**PLOT_THEME,
                            polar=dict(bgcolor="rgba(13,24,34,0.8)",
                                       radialaxis=dict(color="#8ab4c9"),
                                       angularaxis=dict(color="#8ab4c9")),
                            title=dict(text="Radar – Normalised means by station",
                                       font=dict(family="Space Mono", size=13, color="#63caff")))
                        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Visualisations":
    st.markdown("<h2 style='color:#63caff;font-family:Space Mono'>Visualisation Gallery</h2>",
                unsafe_allow_html=True)

    if df is None:
        st.markdown("<div class='info-box'>⚠️ No data loaded.</div>", unsafe_allow_html=True)
    else:
        viz = st.selectbox("Chart Type", [
            "📈 Time Series", "🌡️ Monthly Heatmap", "🌬️ Wind Rose",
            "📦 Pollutant Box Plots", "🍂 PM2.5 by Season & Station",
            "📉 Rolling Average", "🗺️ Station Comparison Bar",
            "🏷️ AQI Category Distribution", "⏰ Hourly Pattern",
            "🔗 Correlation Network",
        ])
        num_cols = get_numeric(df)

        if viz == "📈 Time Series":
            if "datetime" in df.columns:
                cols_sel = st.multiselect("Variables", num_cols,
                    default=num_cols[:min(3, len(num_cols))])
                station_sel = (
                    st.multiselect("Stations",
                        df["station"].unique().tolist(),
                        default=df["station"].unique().tolist()[:2])
                    if "station" in df.columns else []
                )
                mask = (df["station"].isin(station_sel)
                        if station_sel and "station" in df.columns
                        else [True] * len(df))
                sub    = df[mask].sort_values("datetime")
                fig    = go.Figure()
                colors = px.colors.sequential.Blues_r
                for i, c in enumerate(cols_sel):
                    if "station" in sub.columns and station_sel:
                        for j, s in enumerate(station_sel):
                            d = sub[sub["station"] == s].groupby("datetime")[c].mean().reset_index()
                            fig.add_trace(go.Scatter(
                                x=d["datetime"], y=d[c], name=f"{c}|{s}", mode="lines",
                                line=dict(width=1.5, color=colors[min(i * 2, len(colors) - 1)]),
                            ))
                    else:
                        d = sub.groupby("datetime")[c].mean().reset_index()
                        fig.add_trace(go.Scatter(x=d["datetime"], y=d[c], name=c, mode="lines"))
                fig = style_fig(fig, "Time Series")
                fig.update_layout(hovermode="x unified", legend=dict(font=dict(size=10)))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No datetime column found — preprocessing may not have run.")

        elif viz == "🌡️ Monthly Heatmap":
            col_sel = st.selectbox("Variable", num_cols)
            if "month" in df.columns and "year" in df.columns:
                pivot = df.pivot_table(values=col_sel, index="year", columns="month", aggfunc="mean")
                pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]
                fig = px.imshow(pivot, text_auto=".1f", color_continuous_scale="YlOrRd",
                                labels=dict(color=col_sel))
                fig = style_fig(fig, f"Monthly Average {col_sel}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need 'month' and 'year' columns.")

        elif viz == "🌬️ Wind Rose":
            if "wd" in df.columns and "WSPM" in df.columns:
                wr  = df.groupby("wd")["WSPM"].mean().reset_index()
                fig = px.bar_polar(wr, r="WSPM", theta="wd",
                                   color="WSPM", color_continuous_scale="Blues",
                                   template="plotly_dark")
                fig.update_layout(**PLOT_THEME,
                    title=dict(text="Wind Rose (Avg Speed by Direction)",
                               font=dict(family="Space Mono", size=13, color="#63caff")))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need 'wd' (wind direction) and 'WSPM' columns.")

        elif viz == "📦 Pollutant Box Plots":
            pollutants = [c for c in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"] if c in df.columns]
            if pollutants:
                sel = st.multiselect("Pollutants", pollutants, default=pollutants)
                fig = go.Figure()
                for p in sel:
                    fig.add_trace(go.Box(
                        y=df[p].dropna(), name=p, boxpoints="outliers",
                        jitter=0.3,
                        marker=dict(color="#63caff", size=2),
                        line=dict(color="#63caff"),
                    ))
                fig = style_fig(fig, "Pollutant Distribution (Box Plots)")
                st.plotly_chart(fig, use_container_width=True)

        elif viz == "🍂 PM2.5 by Season & Station":
            if "PM2.5" in df.columns and "season" in df.columns and "station" in df.columns:
                grp = df.groupby(["station", "season"])["PM2.5"].mean().reset_index()
                fig = px.bar(grp, x="station", y="PM2.5", color="season", barmode="group",
                             color_discrete_sequence=px.colors.sequential.Blues_r)
                fig = style_fig(fig, "Mean PM2.5 by Station and Season")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need PM2.5, season and station columns.")

        elif viz == "📉 Rolling Average":
            col_sel = st.selectbox("Variable", num_cols, key="roll_col")
            window  = st.slider("Rolling window (hours)", 6, 720, 24)
            if "datetime" in df.columns:
                sub = (df[["datetime", col_sel] + (["station"] if "station" in df.columns else [])]
                       .sort_values("datetime"))
                fig = go.Figure()
                if "station" in sub.columns:
                    for s in sub["station"].unique():
                        d      = sub[sub["station"] == s].set_index("datetime")[col_sel].dropna()
                        rolled = d.rolling(window).mean()
                        fig.add_trace(go.Scatter(x=rolled.index, y=rolled.values,
                                                 name=f"{s} ({window}h MA)", mode="lines"))
                fig = style_fig(fig, f"{window}h Rolling Average – {col_sel}")
                st.plotly_chart(fig, use_container_width=True)

        elif viz == "🗺️ Station Comparison Bar":
            col_sel = st.selectbox("Variable", num_cols, key="stcomp")
            agg_fn  = st.radio("Aggregation", ["mean", "median", "max", "min"], horizontal=True)
            if "station" in df.columns:
                agg = (df.groupby("station")[col_sel]
                       .agg(agg_fn)
                       .sort_values(ascending=False)
                       .reset_index())
                fig = px.bar(agg, x="station", y=col_sel,
                             color=col_sel, color_continuous_scale="Blues", text_auto=".1f")
                fig = style_fig(fig, f"{agg_fn.title()} {col_sel} by Station")
                st.plotly_chart(fig, use_container_width=True)

        elif viz == "🏷️ AQI Category Distribution":
            if "AQI_Cat" in df.columns:
                vc = df["AQI_Cat"].value_counts().reset_index()
                vc.columns = ["Category", "Count"]
                fig = px.pie(vc, names="Category", values="Count",
                             color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.45)
                fig = style_fig(fig, "AQI Category Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("AQI_Cat column not found. Need PM2.5 data.")

        elif viz == "⏰ Hourly Pattern":
            col_sel = st.selectbox("Variable", num_cols, key="hourly")
            if "hour" in df.columns:
                grp_cols = ["hour"] + (["season"] if "season" in df.columns else [])
                agg      = df.groupby(grp_cols)[col_sel].mean().reset_index()
                if "season" in agg.columns:
                    fig = px.line(agg, x="hour", y=col_sel, color="season",
                                  markers=True,
                                  color_discrete_sequence=px.colors.sequential.Blues_r)
                else:
                    fig = px.line(agg, x="hour", y=col_sel, markers=True)
                fig = style_fig(fig, f"Hourly Pattern: {col_sel}")
                st.plotly_chart(fig, use_container_width=True)

        elif viz == "🔗 Correlation Network":
            sel_cols = st.multiselect("Variables", num_cols,
                default=num_cols[:min(8, len(num_cols))], key="netcols")
            if len(sel_cols) >= 3:
                corr_m    = df[sel_cols].corr()
                threshold = st.slider("Correlation threshold", 0.0, 1.0, 0.5, 0.05)
                fig       = go.Figure()
                n         = len(sel_cols)
                angles    = [2 * np.pi * i / n for i in range(n)]
                xs        = [np.cos(a) for a in angles]
                ys        = [np.sin(a) for a in angles]
                for i in range(n):
                    for j in range(i + 1, n):
                        r = abs(corr_m.iloc[i, j])
                        if r >= threshold:
                            fig.add_trace(go.Scatter(
                                x=[xs[i], xs[j], None], y=[ys[i], ys[j], None],
                                mode="lines",
                                line=dict(width=r * 4, color=f"rgba(99,202,255,{r:.2f})"),
                                showlegend=False,
                            ))
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers+text", text=sel_cols,
                    textposition="top center",
                    marker=dict(size=14, color="#63caff"), showlegend=False,
                ))
                fig.update_layout(**PLOT_THEME,
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    title=dict(text="Correlation Network",
                               font=dict(family="Space Mono", size=13, color="#63caff")))
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODELLING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Modelling":
    st.markdown("<h2 style='color:#63caff;font-family:Space Mono'>Machine Learning Modelling</h2>",
                unsafe_allow_html=True)

    if df is None:
        st.markdown("<div class='info-box'>⚠️ No data loaded.</div>", unsafe_allow_html=True)
    else:
        num_cols = get_numeric(df)
        task     = st.radio("Task Type", ["Regression", "Classification"], horizontal=True)

        st.markdown("<div class='section-header'>Feature & Target Selection</div>",
                    unsafe_allow_html=True)
        target   = st.selectbox("Target variable", num_cols)
        features = st.multiselect("Feature variables",
            [c for c in num_cols if c != target],
            default=[c for c in num_cols if c != target][:6])

        if task == "Regression":
            model_options = {
                "Linear Regression":   LinearRegression(),
                "Ridge Regression":    Ridge(),
                "Lasso Regression":    Lasso(),
                "Random Forest":       RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                "Gradient Boosting":   GradientBoostingRegressor(n_estimators=50, random_state=42),
                "Decision Tree":       DecisionTreeRegressor(max_depth=8, random_state=42),
                "SVR (slow on large)": SVR(),
            }
        else:
            model_options = {
                "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
                "Random Forest":        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                "Gradient Boosting":    GradientBoostingClassifier(n_estimators=50, random_state=42),
                "Decision Tree":        DecisionTreeClassifier(max_depth=8, random_state=42),
                "K-Nearest Neighbours": KNeighborsClassifier(n_jobs=-1),
                "Naive Bayes":          GaussianNB(),
                "SVM (slow on large)":  SVC(probability=True, random_state=42),
            }

        selected_models = st.multiselect(
            "Select model(s) – run multiple simultaneously",
            list(model_options.keys()),
            default=list(model_options.keys())[:2],
        )

        st.markdown("<div class='section-header'>Preprocessing Options</div>",
                    unsafe_allow_html=True)
        c1, c2, c3  = st.columns(3)
        scaler_type = c1.selectbox("Scaler", ["None", "StandardScaler", "MinMaxScaler"])
        test_size   = c2.slider("Test split %", 10, 40, 20)
        cv_folds    = c3.slider("CV folds", 2, 10, 5)
        max_rows    = st.slider("Max training rows (reduce for speed)", 5_000, 100_000, 20_000, 5_000)

        st.markdown(
            f"<div class='info-box'>💡 <b>Speed tip:</b> SVR/SVM are slow on >5k rows. "
            f"Random Forest & Gradient Boosting use <code>n_jobs=-1</code>. "
            f"Current cap: <b>{max_rows:,} rows</b>.</div>",
            unsafe_allow_html=True,
        )

        if task == "Classification" and "AQI_Cat" not in df.columns:
            st.markdown("""
            <div class='info-box'>
            ℹ️ For classification the target will be binarised at its median.
            If you have a categorical column like <b>AQI_Cat</b>, select it as target.
            </div>
            """, unsafe_allow_html=True)

        run_btn = st.button("🚀 Run Selected Models")

        if run_btn and len(selected_models) > 0 and len(features) > 0:
            sub = df[features + [target]].dropna()
            if len(sub) > max_rows:
                sub = sub.sample(max_rows, random_state=42)
                st.info(f"ℹ️ Data capped to {max_rows:,} rows for speed.")

            X     = sub[features]
            y_raw = sub[target]

            if task == "Classification":
                if y_raw.dtype == object or str(y_raw.dtype) == "category":
                    le = LabelEncoder()
                    y  = le.fit_transform(y_raw.astype(str))
                else:
                    y = (y_raw >= y_raw.median()).astype(int)
            else:
                y = y_raw.values

            if scaler_type == "StandardScaler":
                sc = StandardScaler()
                X  = pd.DataFrame(sc.fit_transform(X), columns=features)
            elif scaler_type == "MinMaxScaler":
                sc = MinMaxScaler()
                X  = pd.DataFrame(sc.fit_transform(X), columns=features)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42)

            results, models_fitted = [], {}
            prog = st.progress(0)

            for i, mname in enumerate(selected_models):
                m = model_options[mname]
                with st.spinner(f"Training {mname}..."):
                    m.fit(X_train, y_train)
                    y_pred = m.predict(X_test)
                    cv     = cross_val_score(m, X, y, cv=cv_folds,
                                             scoring="r2" if task == "Regression" else "accuracy",
                                             n_jobs=-1)
                    models_fitted[mname] = (m, y_pred)

                    if task == "Regression":
                        results.append({
                            "Model": mname,
                            "R²":    round(r2_score(y_test, y_pred), 4),
                            "MAE":   round(mean_absolute_error(y_test, y_pred), 4),
                            "RMSE":  round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                            f"CV R² ({cv_folds}-fold)": round(cv.mean(), 4),
                            "CV Std": round(cv.std(), 4),
                        })
                    else:
                        results.append({
                            "Model":         mname,
                            "Accuracy":      round(accuracy_score(y_test, y_pred), 4),
                            "F1 (weighted)": round(f1_score(y_test, y_pred, average="weighted"), 4),
                            f"CV Acc ({cv_folds}-fold)": round(cv.mean(), 4),
                            "CV Std":        round(cv.std(), 4),
                        })
                prog.progress((i + 1) / len(selected_models))

            prog.empty()
            st.markdown("<div class='section-header'>📊 Model Comparison</div>",
                        unsafe_allow_html=True)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df.style.highlight_max(axis=0, color="#1a4a6b"),
                         use_container_width=True)

            metric = "R²" if task == "Regression" else "Accuracy"
            if metric in res_df.columns:
                fig = px.bar(res_df, x="Model", y=metric,
                             color=metric, color_continuous_scale="Blues", text_auto=".3f")
                fig = style_fig(fig, f"Model Comparison – {metric}")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='section-header'>🔍 Detailed Results</div>",
                        unsafe_allow_html=True)
            for mname, (m, y_pred) in models_fitted.items():
                with st.expander(f"🔎 {mname}"):
                    if task == "Regression":
                        c1, c2 = st.columns(2)
                        with c1:
                            fig = px.scatter(x=y_test, y=y_pred, opacity=0.5,
                                             labels={"x": "Actual", "y": "Predicted"},
                                             color_discrete_sequence=["#63caff"])
                            fig.add_shape(type="line",
                                x0=float(y_test.min()), y0=float(y_test.min()),
                                x1=float(y_test.max()), y1=float(y_test.max()),
                                line=dict(color="#ff6b6b", dash="dash"))
                            fig = style_fig(fig, "Actual vs Predicted")
                            st.plotly_chart(fig, use_container_width=True)
                        with c2:
                            residuals = y_test - y_pred
                            fig = px.histogram(x=residuals, nbins=40,
                                               labels={"x": "Residual"},
                                               color_discrete_sequence=["#63caff"])
                            fig = style_fig(fig, "Residuals Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        cm = confusion_matrix(y_test, y_pred)
                        c1, c2 = st.columns(2)
                        with c1:
                            fig = px.imshow(cm, text_auto=True,
                                            color_continuous_scale="Blues",
                                            labels=dict(x="Predicted", y="Actual"))
                            fig = style_fig(fig, "Confusion Matrix")
                            st.plotly_chart(fig, use_container_width=True)
                        with c2:
                            st.text(classification_report(y_test, y_pred))

                    if hasattr(m, "feature_importances_"):
                        imp = pd.DataFrame({
                            "Feature":    features,
                            "Importance": m.feature_importances_,
                        }).sort_values("Importance", ascending=False)
                        fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                                     color="Importance", color_continuous_scale="Blues")
                        fig = style_fig(fig, "Feature Importance")
                        st.plotly_chart(fig, use_container_width=True)
                    elif hasattr(m, "coef_"):
                        coefs = m.coef_.flatten()[:len(features)]
                        imp   = pd.DataFrame({
                            "Feature":     features[:len(coefs)],
                            "Coefficient": np.abs(coefs),
                        }).sort_values("Coefficient", ascending=False)
                        fig = px.bar(imp, x="Coefficient", y="Feature", orientation="h",
                                     color="Coefficient", color_continuous_scale="Blues")
                        fig = style_fig(fig, "Feature Coefficients (abs)")
                        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Report":
    st.markdown("<h2 style='color:#63caff;font-family:Space Mono'>Analysis Report</h2>",
                unsafe_allow_html=True)

    if df is None:
        st.markdown("<div class='info-box'>⚠️ No data loaded.</div>", unsafe_allow_html=True)
    else:
        num_cols = get_numeric(df)

        st.markdown("<div class='section-header'>Dataset Summary</div>", unsafe_allow_html=True)
        stations_str = (", ".join(df["station"].unique().tolist())
                        if "station" in df.columns else "N/A")
        st.markdown(f"""
        <div class='info-box'>
        <b>Total Records:</b> {len(df):,}<br>
        <b>Columns:</b> {df.shape[1]}<br>
        <b>Numeric Variables:</b> {len(num_cols)}<br>
        <b>Missing Values:</b> {df.isnull().sum().sum():,} ({df.isnull().mean().mean() * 100:.1f}%)<br>
        <b>Duplicate Rows:</b> {df.duplicated().sum():,}<br>
        <b>Stations:</b> {stations_str}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Statistical Summary</div>", unsafe_allow_html=True)
        st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)

        if "PM2.5" in df.columns:
            st.markdown("<div class='section-header'>PM2.5 Key Insights</div>",
                        unsafe_allow_html=True)
            pm = df["PM2.5"].dropna()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean PM2.5",           f"{pm.mean():.1f} μg/m³")
            c2.metric("Max PM2.5",            f"{pm.max():.1f} μg/m³")
            c3.metric("Days > 75 μg/m³",      f"{(pm > 75).sum():,}")
            c4.metric("WHO Exceedances (>15)", f"{(pm > 15).mean() * 100:.1f}%")

            st.markdown("<div class='section-header'>Correlation with PM2.5</div>",
                        unsafe_allow_html=True)
            corr_pm = df[num_cols].corr()["PM2.5"].drop("PM2.5").sort_values()
            fig     = px.bar(corr_pm, orientation="h",
                             color=corr_pm.values, color_continuous_scale="RdBu_r",
                             labels={"value": "Correlation", "index": "Variable"})
            fig     = style_fig(fig, "Pearson Correlation with PM2.5")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>References</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        Brauer, M. et al. (2021). Ambient particulate matter air pollution exposure and mortality.
        <i>Environmental Health Perspectives.</i><br>
        Li, Z. et al. (2019). Air pollution and health in China.
        <i>Environmental Research Letters.</i><br>
        Li, Z. et al. (2024). Recent trends in Beijing air quality.
        <i>Atmospheric Environment.</i><br>
        Lim, S.S. et al. (2020). Air quality and health burden. <i>The Lancet.</i><br>
        Sokhi, R.S. et al. (2022). Global air quality challenges.
        <i>npj Climate and Atmospheric Science.</i><br>
        Xu, J. & Zhang, Y. (2020). Emission controls in Beijing.
        <i>Science of the Total Environment.</i>
        </div>
        """, unsafe_allow_html=True)

