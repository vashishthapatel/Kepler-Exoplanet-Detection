from pathlib import Path
from html import escape
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
from pandas.errors import EmptyDataError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import extract_features, preprocess_flux


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_DATA_CANDIDATES = [
    ROOT_DIR / "exo_all_combined.csv",
    ROOT_DIR / "exoTrain.csv",
    ROOT_DIR / "Data" / "exo_all_combined.csv",
    ROOT_DIR / "Data" / "exoTrain.csv",
]


st.set_page_config(
    page_title="Kepler Light Curve Detector",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #080d15;
        --panel: #101827;
        --panel-2: #152235;
        --ink: #e8eef7;
        --muted: #96a6bb;
        --line: #26364c;
        --accent: #58d3e6;
        --accent-2: #79f2a6;
        --danger: #ff6b6b;
        --warn: #f7c948;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(88, 211, 230, .13), transparent 34rem),
            linear-gradient(180deg, #080d15 0%, #0b1320 52%, #070b12 100%);
        color: var(--ink);
    }
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: var(--ink);
        letter-spacing: 0;
    }
    [data-testid="stSidebar"] {
        background: #0b121f;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * {
        color: var(--ink);
    }
    .block-container {
        max-width: 1440px;
        padding-top: 1.7rem;
        padding-left: clamp(1rem, 2vw, 2.25rem);
        padding-right: clamp(1rem, 2vw, 2.25rem);
    }
    .hero {
        padding: clamp(1.25rem, 2.2vw, 2rem);
        background:
            linear-gradient(135deg, rgba(6, 12, 22, .92), rgba(14, 77, 95, .8)),
            url("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&w=1600&q=80");
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(88, 211, 230, .25);
        border-radius: 8px;
        box-shadow: 0 22px 54px rgba(0, 0, 0, .32);
        margin-bottom: 18px;
    }
    .hero h1 {
        font-size: clamp(1.8rem, 3vw, 3.25rem);
        line-height: 1.04;
        margin: 0 0 10px 0;
    }
    .hero p {
        max-width: 920px;
        color: rgba(232, 238, 247, .82);
        margin: 0;
        font-size: 1.02rem;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(var(--metric-columns, 4), minmax(0, 1fr));
        gap: 1rem;
        margin: .25rem 0 1.15rem 0;
        width: 100%;
    }
    .metric-card {
        background: linear-gradient(180deg, var(--panel-2), var(--panel));
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 16px;
        min-height: 118px;
        box-shadow: 0 16px 34px rgba(0, 0, 0, .24);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-width: 0;
        margin-bottom: 1rem;
    }
    .metric-label {
        color: var(--muted);
        font-size: .82rem;
        margin-bottom: 7px;
        overflow-wrap: anywhere;
    }
    .metric-value {
        color: var(--ink);
        font-size: clamp(1.25rem, 1.8vw, 1.75rem);
        font-weight: 760;
        line-height: 1.1;
        overflow-wrap: anywhere;
    }
    .metric-note {
        color: var(--muted);
        font-size: .8rem;
        margin-top: 7px;
        overflow-wrap: anywhere;
    }
    div[data-testid="column"] {
        min-width: 0;
    }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stImage"]),
    div[data-testid="stPlotlyChart"],
    div[data-testid="stPyplot"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .4rem;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        background: var(--panel);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        color: var(--ink);
        padding: 8px 14px;
        min-width: 112px;
        justify-content: center;
    }
    .stTabs [aria-selected="true"] {
        background: #102d39;
        border-color: rgba(88, 211, 230, .55);
    }
    .stButton button,
    .stDownloadButton button {
        background: #123849;
        color: var(--ink);
        border: 1px solid rgba(88, 211, 230, .42);
        border-radius: 8px;
    }
    .stButton button:hover,
    .stDownloadButton button:hover {
        background: #165166;
        border-color: var(--accent);
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: var(--panel);
        color: var(--ink);
    }
    @media (max-width: 1180px) {
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 720px) {
        .block-container {
            padding-left: .85rem;
            padding-right: .85rem;
        }
        .metric-grid {
            grid-template-columns: 1fr;
            gap: .75rem;
        }
        .metric-card {
            min-height: 104px;
        }
        .stTabs [data-baseweb="tab"] {
            min-width: calc(50% - 8px);
        }
        .hero {
            margin-bottom: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(str(label))}</div>
            <div class="metric-value">{escape(str(value))}</div>
            <div class="metric-note">{escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_grid(cards, columns=4):
    columns = max(1, min(columns, len(cards)))
    for start in range(0, len(cards), columns):
        row_cards = cards[start:start + columns]
        cols = st.columns(len(row_cards), gap="medium")
        for col, (label, value, note) in zip(cols, row_cards):
            with col:
                metric_card(label, value, note)


def format_number(value, decimals=2):
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)) or abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def dark_figure(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#101827")
    ax.set_facecolor("#101827")
    ax.tick_params(colors="#c8d3e1")
    ax.xaxis.label.set_color("#d9e3f0")
    ax.yaxis.label.set_color("#d9e3f0")
    ax.title.set_color("#e8eef7")
    for spine in ax.spines.values():
        spine.set_color("#32455f")
    return fig, ax


@st.cache_data(show_spinner=False)
def generate_demo_data(rows=180, points=320, planet_ratio=0.16, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1], size=rows, p=[1 - planet_ratio, planet_ratio])
    curves = []

    for label in labels:
        x = np.linspace(0, 1, points)
        baseline = rng.normal(0, 0.7, points)
        wave = 0.35 * np.sin(2 * np.pi * rng.uniform(1.5, 6) * x + rng.uniform(0, np.pi))
        curve = baseline + wave

        if label == 1:
            period = rng.integers(55, 115)
            width = rng.integers(4, 10)
            depth = rng.uniform(2.5, 5.8)
            start = rng.integers(0, period)
            for center in range(start, points, period):
                lo = max(0, center - width // 2)
                hi = min(points, center + width // 2)
                curve[lo:hi] -= depth
        else:
            if rng.random() < 0.35:
                center = rng.integers(0, points)
                width = rng.integers(2, 6)
                curve[max(0, center - width):min(points, center + width)] -= rng.uniform(1, 2)

        curves.append(curve)

    flux_columns = [f"FLUX_{i + 1}" for i in range(points)]
    df = pd.DataFrame(curves, columns=flux_columns)
    df.insert(0, "LABEL", labels)
    return df


def find_local_dataset():
    for path in LOCAL_DATA_CANDIDATES:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def read_csv_source(source, uploaded=False):
    if uploaded:
        source.seek(0)
    try:
        return pd.read_csv(source)
    except EmptyDataError:
        st.error("The selected CSV file is empty. Upload a valid light-curve CSV with a LABEL column.")
        st.stop()


def normalize_light_curve_data(raw_df: pd.DataFrame):
    if "LABEL" not in raw_df.columns:
        return None, "This is not the notebook dataset format. The frontend needs a `LABEL` column plus numeric flux columns."

    df = raw_df.copy()
    df["LABEL"] = pd.to_numeric(df["LABEL"], errors="coerce")
    unique_labels = set(df["LABEL"].dropna().astype(int).unique())
    if 2 in unique_labels:
        df["LABEL"] = df["LABEL"].map({2: 1, 1: 0})
    df = df[df["LABEL"].isin([0, 1])]

    flux = df.drop(columns=["LABEL"]).apply(pd.to_numeric, errors="coerce")
    flux = flux.dropna(axis=1, how="all")
    good_rows = flux.notna().sum(axis=1) > 0
    flux = flux.loc[good_rows].fillna(flux.median(numeric_only=True))
    labels = df.loc[good_rows, "LABEL"].astype(int)

    if flux.empty or labels.empty:
        return None, "No usable numeric flux columns were found after cleaning the CSV."

    cleaned = pd.concat([labels.rename("LABEL"), flux], axis=1).reset_index(drop=True)
    return cleaned, None


def load_data_from_ui():
    st.sidebar.header("Dataset")
    uploaded = st.sidebar.file_uploader("Upload notebook CSV", type=["csv"])
    local_path = find_local_dataset()
    warning_message = None

    if uploaded is not None:
        raw_df = read_csv_source(uploaded, uploaded=True)
        source_label = uploaded.name
    elif local_path is not None:
        raw_df = read_csv_source(local_path)
        source_label = local_path.name
    else:
        raw_df = generate_demo_data()
        source_label = "Demo synthetic light curves"
        st.sidebar.info("No `exo_all_combined.csv` found, so demo mode is active.")

    cleaned, error = normalize_light_curve_data(raw_df)
    if error:
        warning_message = (
            f"`{source_label}` was loaded, but it is not the light-curve dataset used by `main.ipynb`. "
            "The app is running with synthetic demo light curves for now. Upload `exo_all_combined.csv`, "
            "`exoTrain.csv`, or any CSV with `LABEL` and numeric flux columns to train on real notebook data."
        )
        raw_df = generate_demo_data()
        source_label = "Demo synthetic light curves"
        cleaned, error = normalize_light_curve_data(raw_df)
        if error:
            st.error(error)
            st.stop()

    if warning_message:
        st.warning(warning_message)
    return cleaned, source_label


def prepare_sample(df: pd.DataFrame):
    st.sidebar.header("Run Settings")
    min_rows = min(20, len(df))
    max_available = min(2000, len(df))
    default_rows = min(700, max_available)
    step = 20 if max_available >= 100 else 1
    max_rows = st.sidebar.slider("Rows used for interactive training", min_rows, max_available, default_rows, step)
    sigma = st.sidebar.slider("Gaussian smoothing sigma", 1, 20, 10)
    test_size = st.sidebar.slider("Test split", 0.10, 0.35, 0.20, 0.05)
    model_name = st.sidebar.selectbox("Model", ["Random Forest", "SVM (RBF)", "XGBoost"])

    if len(df) > max_rows:
        sampled_parts = []
        for _, part in df.groupby("LABEL"):
            part_size = max(1, round(max_rows * len(part) / len(df)))
            sampled_parts.append(part.sample(min(len(part), part_size), random_state=42))
        sampled = pd.concat(sampled_parts)
        if len(sampled) > max_rows:
            sampled = sampled.sample(max_rows, random_state=42)
        elif len(sampled) < max_rows:
            remainder = df.drop(sampled.index).sample(min(max_rows - len(sampled), len(df) - len(sampled)), random_state=42)
            sampled = pd.concat([sampled, remainder])
        sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        sampled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return sampled, sigma, test_size, model_name


@st.cache_data(show_spinner=False)
def compute_features(flux_df: pd.DataFrame, sigma: int):
    processed = preprocess_flux(flux_df, sigma=sigma)
    features = extract_features(processed)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return processed, features


def build_model(model_name, y_train):
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=260,
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "SVM (RBF)":
        return SVC(C=2.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42)

    n_pos = max(1, int((y_train == 1).sum()))
    n_neg = max(1, int((y_train == 0).sum()))
    return XGBClassifier(
        n_estimators=220,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=n_neg / n_pos,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="aucpr",
        n_jobs=-1,
    )


@st.cache_resource(show_spinner=False)
def train_pipeline(features: pd.DataFrame, labels: pd.Series, test_size: float, model_name: str):
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=40,
        stratify=labels,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    minority_count = int(y_train.value_counts().min())
    if minority_count > 1:
        smote = SMOTE(random_state=42, k_neighbors=min(5, minority_count - 1))
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
        smote_note = "SMOTE applied to training split"
    else:
        X_train_final, y_train_final = X_train_scaled, y_train
        smote_note = "SMOTE skipped: minority class too small"

    model = build_model(model_name, y_train)
    model.fit(X_train_final, y_train_final)

    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_test.nunique() == 2 else np.nan,
        "avg_precision": average_precision_score(y_test, y_prob) if y_test.nunique() == 2 else np.nan,
        "confusion": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No Planet", "Exoplanet"],
            zero_division=0,
        ),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "smote_note": smote_note,
        "resampled_counts": pd.Series(y_train_final).value_counts().sort_index().to_dict(),
    }
    return model, scaler, metrics


def plot_class_balance(labels: pd.Series):
    counts = labels.value_counts().sort_index()
    fig, ax = dark_figure((6.8, 4))
    bars = ax.bar(["No Planet", "Exoplanet"], [counts.get(0, 0), counts.get(1, 0)], color=["#58d3e6", "#79f2a6"])
    ax.bar_label(bars, color="#e8eef7", padding=3)
    ax.set_title("Class Balance")
    ax.set_ylabel("Rows")
    st.pyplot(fig, width="stretch")


def plot_light_curve(raw, processed, title):
    fig, ax = dark_figure((9, 3.5))
    ax.plot(raw, color="#73879f", linewidth=0.9, alpha=0.8, label="Raw")
    ax.plot(processed, color="#58d3e6", linewidth=1.5, label="Preprocessed")
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Flux")
    ax.legend(frameon=False, labelcolor="#d9e3f0")
    st.pyplot(fig, width="stretch")


def plot_variability(flux_df, labels):
    stats = pd.DataFrame({"Flux std": flux_df.std(axis=1), "Label": labels.map({0: "No Planet", 1: "Exoplanet"})})
    fig, ax = dark_figure((6.8, 4))
    sns.boxplot(data=stats, x="Label", y="Flux std", ax=ax, palette=["#58d3e6", "#79f2a6"], hue="Label", legend=False)
    ax.set_title("Flux Variability by Class")
    st.pyplot(fig, width="stretch")


def plot_confusion(cm):
    fig, ax = dark_figure((5.5, 4.4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="mako",
        xticklabels=["No Planet", "Exoplanet"],
        yticklabels=["No Planet", "Exoplanet"],
        ax=ax,
        cbar=False,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, width="stretch")


def plot_feature_importance(model, features):
    if not hasattr(model, "feature_importances_"):
        st.info("Feature importance is available for Random Forest and XGBoost.")
        return
    top = (
        pd.DataFrame({"Feature": features.columns, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(12)
        .sort_values("Importance")
    )
    fig, ax = dark_figure((7, 4.8))
    ax.barh(top["Feature"], top["Importance"], color="#58d3e6")
    ax.set_title("Top Engineered Features")
    ax.set_xlabel("Importance")
    st.pyplot(fig, width="stretch")


data, source_name = load_data_from_ui()
sampled_data, sigma, test_size, model_name = prepare_sample(data)

labels = sampled_data["LABEL"].astype(int)
flux_df = sampled_data.drop(columns=["LABEL"])

if labels.nunique() < 2:
    st.error("The selected data contains only one class. Training needs both no-planet and exoplanet rows.")
    st.stop()
if labels.value_counts().min() < 2:
    st.error("Each class needs at least two rows for a stratified train-test split. Add more rows or increase the row limit.")
    st.stop()

st.markdown(
    f"""
    <section class="hero">
        <h1>Kepler Light Curve Exoplanet Detector</h1>
        <p>Frontend built from <strong>main.ipynb</strong>: load labeled flux time-series, normalize and smooth each light curve, engineer statistical and frequency features, balance training data with SMOTE, and evaluate a binary exoplanet classifier.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Preprocessing light curves and extracting notebook features..."):
    processed_flux, features = compute_features(flux_df, sigma)

with st.spinner(f"Training {model_name}..."):
    model, scaler, metrics = train_pipeline(features, labels, test_size, model_name)

metric_grid(
    [
        ("Dataset", f"{len(sampled_data):,}", source_name),
        ("Flux Points", f"{flux_df.shape[1]:,}", "Per light curve"),
        ("Exoplanets", f"{int(labels.sum()):,}", f"{labels.mean():.1%} positive class"),
        ("Recall", format_number(metrics["recall"]), "Primary notebook metric"),
        ("ROC-AUC", format_number(metrics["roc_auc"]), model_name),
    ],
    columns=5,
)

overview_tab, training_tab, predict_tab, data_tab = st.tabs(["Overview", "Training", "Predict", "Data"])

with overview_tab:
    left, right = st.columns(2)
    with left:
        plot_class_balance(labels)
    with right:
        plot_variability(flux_df, labels)

    example_label = st.radio("Sample curve", ["No Planet", "Exoplanet"], horizontal=True)
    label_value = 0 if example_label == "No Planet" else 1
    matching = np.flatnonzero(labels.to_numpy() == label_value)
    idx = int(matching[0]) if len(matching) else 0
    plot_light_curve(
        flux_df.iloc[idx].to_numpy(dtype=float),
        processed_flux[idx],
        f"Sample Light Curve - {example_label}",
    )

with training_tab:
    metric_grid(
        [
            ("Precision", format_number(metrics["precision"]), "Exoplanet class"),
            ("Recall", format_number(metrics["recall"]), "Exoplanet class"),
            ("Avg Precision", format_number(metrics["avg_precision"]), "PR-AUC proxy"),
            ("Test Rows", f"{metrics['test_size']:,}", metrics["smote_note"]),
        ],
        columns=4,
    )

    left, right = st.columns([1, 1])
    with left:
        plot_confusion(metrics["confusion"])
    with right:
        plot_feature_importance(model, features)

    with st.expander("Classification report"):
        st.code(metrics["report"])
    with st.expander("Feature table preview"):
        st.dataframe(features.head(20), width="stretch", hide_index=True)

with predict_tab:
    st.subheader("Inspect One Light Curve")
    selected_row = st.slider("Row from current sampled dataset", 0, len(sampled_data) - 1, 0)
    one_flux = flux_df.iloc[[selected_row]]
    one_processed = preprocess_flux(one_flux, sigma=sigma)
    one_features = extract_features(one_processed).replace([np.inf, -np.inf], np.nan).fillna(0)
    one_scaled = scaler.transform(one_features)
    prediction = int(model.predict(one_scaled)[0])
    probability = float(model.predict_proba(one_scaled)[0, 1])

    metric_grid(
        [
            ("Actual Label", "Exoplanet" if labels.iloc[selected_row] == 1 else "No Planet", "Ground truth"),
            ("Prediction", "Exoplanet" if prediction == 1 else "No Planet", model_name),
            ("Exoplanet Probability", f"{probability:.1%}", "Model confidence"),
        ],
        columns=3,
    )

    plot_light_curve(
        one_flux.iloc[0].to_numpy(dtype=float),
        one_processed[0],
        "Selected Light Curve",
    )

with data_tab:
    st.subheader("Loaded Light-Curve Data")
    preview_cols = ["LABEL"] + list(flux_df.columns[:12])
    st.dataframe(sampled_data[preview_cols].head(50), width="stretch", hide_index=True)

    summary = pd.DataFrame(
        {
            "Item": ["Source", "Rows available", "Rows used", "Flux columns", "Engineered features", "Train rows", "Test rows"],
            "Value": [
                source_name,
                f"{len(data):,}",
                f"{len(sampled_data):,}",
                f"{flux_df.shape[1]:,}",
                f"{features.shape[1]:,}",
                f"{metrics['train_size']:,}",
                f"{metrics['test_size']:,}",
            ],
        }
    )
    st.dataframe(summary, width="stretch", hide_index=True)
