
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Nexus | Breast Cancer Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS: NAVY, CYAN AND CLEAR SIDEBAR
# =========================================================
st.markdown("""
<style>
/* Main application */
.stApp {
    background: linear-gradient(135deg, #08111f 0%, #101d33 100%);
    color: #f1f5f9;
}

h1, h2, h3 {
    color: #f1f5f9;
}

[data-testid="stHeader"] {
    background: rgba(8, 17, 31, 0.85);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #334155;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.3rem 1rem;
}

/* Sidebar brand */
[data-testid="stSidebar"] h1 {
    color: #38bdf8;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: 1px;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f1f5f9;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #e2e8f0;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #a8b6ca;
    font-size: 0.78rem;
}

/* Sidebar navigation cards */
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: #f1f5f9;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 8px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 7px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    background: #142136;
    border: 1px solid #2b405c;
    border-radius: 10px;
    padding: 11px 12px;
    margin: 0;
    transition: background 0.2s ease, border-color 0.2s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background: #20334d;
    border-color: #38bdf8;
}

/* Radio controls */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
    color: #e2e8f0;
}

/* Sidebar slider labels and help text */
[data-testid="stSidebar"] [data-testid="stSlider"] label {
    color: #e2e8f0;
    font-weight: 500;
}

/* Sidebar separators */
[data-testid="stSidebar"] hr {
    border-color: #334155;
    margin: 1.1rem 0;
}

/* Main metric cards */
[data-testid="stMetric"] {
    background: #14243a;
    border: 1px solid #2b405c;
    padding: 18px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #a8b6ca;
}

[data-testid="stMetricValue"] {
    color: #38bdf8;
}

/* Buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    background: #0284c7;
    color: white;
    border: 1px solid #0284c7;
    border-radius: 9px;
    padding: 9px 18px;
    font-weight: 600;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #0369a1;
    color: white;
    border-color: #38bdf8;
}

/* Input fields */
.stTextInput input,
.stNumberInput input {
    background: #142136;
    color: #f1f5f9;
    border-color: #334155;
    border-radius: 8px;
}

/* Plot backgrounds */
[data-testid="stPlotlyChart"] {
    background: #101d33;
    border: 1px solid #263b57;
    border-radius: 12px;
    padding: 5px;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #263b57;
    border-radius: 10px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Footer */
.footer {
    color: #94a3b8;
    text-align: center;
    font-size: 0.85rem;
    padding: 15px 0;
}

hr {
    border-color: #263b57;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    dataset = load_breast_cancer()

    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    y = pd.Series(dataset.target, name="target")

    df = X.copy()
    df["diagnosis"] = y.map({
        0: "Malignant",
        1: "Benign"
    })

    return X, y, df


X, y, df = load_data()


# =========================================================
# MODEL TRAINING AND CACHING
# =========================================================
@st.cache_resource
def train_model(n_estimators, test_fraction):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Class 0 = malignant, class 1 = benign.
    malignant_index = list(model.classes_).index(0)
    malignant_scores = probabilities[:, malignant_index]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "auc": roc_auc_score(
            (y_test == 0).astype(int), malignant_scores
        ),
        "confusion": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ),
        "report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Malignant", "Benign"],
            output_dict=True,
            zero_division=0
        ),
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "malignant_scores": malignant_scores
    }

    return model, metrics


# =========================================================
# SIDEBAR: CLEAR NAVIGATION AND SETTINGS
# =========================================================
with st.sidebar:

    st.markdown("# ◈ NEXUS")
    st.caption("BREAST CANCER ANALYTICS")
    st.divider()

    st.markdown("### 🧭 Navigation")

    page = st.radio(
        "Choose a page",
        [
            "🏠 Overview",
            "📊 Data Analysis",
            "🤖 Cancer Prediction",
            "🎯 Model Evaluation",
            "🗃️ Dataset Explorer",
            "ℹ️ About"
        ],
        label_visibility="collapsed",
        key="dashboard_navigation"
    )

    st.divider()

    st.markdown("### ⚙️ Model Settings")

    trees = st.select_slider(
        "Number of trees",
        options=[50, 100, 150, 200, 300],
        value=100,
        help="Number of decision trees in the Random Forest."
    )

    test_percent = st.slider(
        "Test dataset size",
        min_value=10,
        max_value=35,
        value=20,
        step=5,
        format="%d%%",
        help="Percentage of records reserved for testing."
    )

    test_fraction = test_percent / 100

    st.divider()

    st.markdown("### 🟢 System Status")
    st.success("Dataset loaded successfully")

    st.caption(f"Records: {len(df):,}")
    st.caption(f"Features: {X.shape[1]}")
    st.caption("Algorithm: Random Forest")

    st.divider()
    st.caption("NEXUS · VERSION 2.1")
    st.caption("Educational use only")


# Train once per unique configuration and reuse across reruns.
model, metrics = train_model(trees, test_fraction)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    "<h1 style='margin-bottom:0'>Breast Cancer Intelligence</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color:#a8b6ca'>"
    "Tumour data exploration, classification and model evaluation"
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# OVERVIEW
# =========================================================
if page == "🏠 Overview":

    st.subheader("Dataset Overview")
    st.caption("Wisconsin Diagnostic Breast Cancer dataset")

    malignant = int((y == 0).sum())
    benign = int((y == 1).sum())

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Samples", f"{len(df):,}")
    c2.metric("Benign Samples", f"{benign:,}")
    c3.metric("Malignant Samples", f"{malignant:,}")
    c4.metric("Input Features", X.shape[1])

    left, right = st.columns(2)

    with left:
        diagnosis_counts = (
            df["diagnosis"]
            .value_counts()
            .rename_axis("Diagnosis")
            .reset_index(name="Count")
        )

        fig = px.pie(
            diagnosis_counts,
            names="Diagnosis",
            values="Count",
            hole=0.55,
            color="Diagnosis",
            color_discrete_map={
                "Benign": "#34d399",
                "Malignant": "#fb7185"
            },
            title="Diagnosis Distribution",
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="#101d33",
            plot_bgcolor="#101d33",
            font_color="#e2e8f0"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            df,
            x="mean radius",
            y="mean texture",
            color="diagnosis",
            color_discrete_map={
                "Benign": "#34d399",
                "Malignant": "#fb7185"
            },
            opacity=0.75,
            title="Mean Radius vs Mean Texture",
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="#101d33",
            plot_bgcolor="#101d33",
            font_color="#e2e8f0"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature Correlations")

    selected = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness"
    ]

    fig = px.imshow(
        X[selected].corr(),
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Between Selected Measurements",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="#101d33",
        font_color="#e2e8f0"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The dataset contains 569 samples and 30 numerical features. "
        "The labels are the recorded benign and malignant classes."
    )


# =========================================================
# DATA ANALYSIS
# =========================================================
elif page == "📊 Data Analysis":

    st.subheader("Exploratory Data Analysis")

    diagnosis_filter = st.multiselect(
        "Filter diagnoses",
        ["Benign", "Malignant"],
        default=["Benign", "Malignant"]
    )

    feature = st.selectbox(
        "Choose a measurement",
        list(X.columns),
        index=list(X.columns).index("mean radius")
    )

    filtered = df[df["diagnosis"].isin(diagnosis_filter)]

    if filtered.empty:
        st.warning("Select at least one diagnosis.")
    else:
        left, right = st.columns(2)

        with left:
            fig = px.histogram(
                filtered,
                x=feature,
                color="diagnosis",
                marginal="box",
                barmode="overlay",
                opacity=0.75,
                color_discrete_map={
                    "Benign": "#34d399",
                    "Malignant": "#fb7185"
                },
                title=f"Distribution of {feature}",
                template="plotly_dark"
            )

            fig.update_layout(
                paper_bgcolor="#101d33",
                plot_bgcolor="#101d33",
                font_color="#e2e8f0"
            )

            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.box(
                filtered,
                x="diagnosis",
                y=feature,
                color="diagnosis",
                color_discrete_map={
                    "Benign": "#34d399",
                    "Malignant": "#fb7185"
                },
                title=f"{feature} by Diagnosis",
                template="plotly_dark"
            )

            fig.update_layout(
                paper_bgcolor="#101d33",
                plot_bgcolor="#101d33",
                font_color="#e2e8f0"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Compare Two Features")

        c1, c2 = st.columns(2)

        with c1:
            x_feature = st.selectbox(
                "X-axis feature",
                list(X.columns),
                index=list(X.columns).index("mean radius")
            )

        with c2:
            y_feature = st.selectbox(
                "Y-axis feature",
                list(X.columns),
                index=list(X.columns).index("mean texture")
            )

        fig = px.scatter(
            filtered,
            x=x_feature,
            y=y_feature,
            color="diagnosis",
            color_discrete_map={
                "Benign": "#34d399",
                "Malignant": "#fb7185"
            },
            title="Relationship Between Measurements",
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="#101d33",
            plot_bgcolor="#101d33",
            font_color="#e2e8f0"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Statistical Summary")
        st.dataframe(
            filtered.groupby("diagnosis")[X.columns].mean().T,
            use_container_width=True
        )


# =========================================================
# CANCER PREDICTION
# =========================================================
elif page == "🤖 Cancer Prediction":

    st.subheader("Tumour Classification")
    st.write(
        "Enter the 30 numerical measurements required by the model."
    )

    st.warning(
        "This is an educational model, not a medical diagnostic tool. "
        "Do not use its output for clinical decisions."
    )

    st.caption(
        "The default values are dataset averages, not measurements from a patient."
    )

    user_values = {}

    with st.form("prediction_form"):

        input_columns = st.columns(3)

        for i, feature_name in enumerate(X.columns):
            with input_columns[i % 3]:

                mean_value = float(X[feature_name].mean())
                min_value = float(X[feature_name].min())
                max_value = float(X[feature_name].max())
                std_value = float(X[feature_name].std())

                user_values[feature_name] = st.number_input(
                    feature_name,
                    min_value=min_value,
                    max_value=max_value,
                    value=mean_value,
                    step=max(std_value / 10, 0.01),
                    format="%.4f",
                    help=(
                        f"Observed dataset range: "
                        f"{min_value:.4f} to {max_value:.4f}"
                    )
                )

        submitted = st.form_submit_button(
            "Run Classification",
            type="primary",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [user_values],
            columns=X.columns
        )

        predicted_class = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        class_to_probability = {
            int(class_label): float(probability)
            for class_label, probability
            in zip(model.classes_, probabilities)
        }

        malignant_probability = class_to_probability[0]
        benign_probability = class_to_probability[1]

        st.divider()
        st.markdown("### Model Output")

        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                "Predicted dataset class",
                "Malignant" if predicted_class == 0 else "Benign"
            )

        with r2:
            st.metric(
                "Malignant-class model score",
                f"{malignant_probability:.2%}"
            )

        prob_df = pd.DataFrame({
            "Class": ["Malignant", "Benign"],
            "Model Probability": [
                malignant_probability,
                benign_probability
            ]
        })

        fig = px.bar(
            prob_df,
            x="Class",
            y="Model Probability",
            color="Class",
            color_discrete_map={
                "Benign": "#34d399",
                "Malignant": "#fb7185"
            },
            range_y=[0, 1],
            title="Model Class Probabilities",
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="#101d33",
            plot_bgcolor="#101d33",
            font_color="#e2e8f0"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "These values reflect the model's output, not calibrated "
            "patient-level cancer risk."
        )

        st.info(
            "A qualified healthcare professional must interpret actual "
            "medical findings. This application cannot diagnose cancer."
        )


# =========================================================
# MODEL EVALUATION
# =========================================================
elif page == "🎯 Model Evaluation":

    st.subheader("Model Performance")

    st.caption(
        f"Random Forest · {trees} trees · "
        f"{100 - test_percent}% training · {test_percent}% testing"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    m2.metric("Malignant Precision", f"{metrics['precision']:.2%}")
    m3.metric("Malignant Recall", f"{metrics['recall']:.2%}")
    m4.metric("Malignant F1 Score", f"{metrics['f1']:.2%}")

    st.caption(
        "Malignant precision, recall and F1 use the malignant class as positive."
    )

    st.markdown("### Confusion Matrix")

    cm_df = pd.DataFrame(
        metrics["confusion"],
        index=["Actual Malignant", "Actual Benign"],
        columns=["Predicted Malignant", "Predicted Benign"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Actual vs Predicted Classes",
        aspect="auto"
    )

    fig.update_layout(
        paper_bgcolor="#101d33",
        font_color="#e2e8f0"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ROC Curve")

    fpr, tpr, _ = roc_curve(
        (metrics["y_test"] == 0).astype(int),
        metrics["malignant_scores"]
    )

    roc_fig = go.Figure()

    roc_fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"Malignant class (AUC = {metrics['auc']:.3f})",
        line=dict(width=3, color="#38bdf8")
    ))

    roc_fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random baseline",
        line=dict(dash="dash", color="#94a3b8")
    ))

    roc_fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        paper_bgcolor="#101d33",
        plot_bgcolor="#101d33",
        font_color="#e2e8f0"
    )

    st.plotly_chart(roc_fig, use_container_width=True)

    st.markdown("### Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#164e63", "#38bdf8"],
        title="Top 15 Model Features",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="#101d33",
        plot_bgcolor="#101d33",
        font_color="#e2e8f0",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Classification Report")

    report_df = pd.DataFrame(metrics["report"]).T
    st.dataframe(report_df.round(3), use_container_width=True)

    st.caption(
        "These results come from a single held-out split of a benchmark "
        "dataset and are not evidence of clinical readiness."
    )


# =========================================================
# DATASET EXPLORER
# =========================================================
elif page == "🗃️ Dataset Explorer":

    st.subheader("Breast Cancer Dataset Explorer")
    st.write("Search, inspect and download dataset records.")

    search = st.text_input(
        "Search feature names",
        placeholder="e.g. radius, texture, area"
    )

    display_df = df.copy()

    if search.strip():
        selected_features = [
            col for col in X.columns
            if search.strip().lower() in col.lower()
        ]

        display_df = display_df[
            selected_features + ["diagnosis"]
        ]

    st.dataframe(
        display_df.head(100),
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric(
        "Missing Values",
        f"{int(df.isna().sum().sum()):,}"
    )

    with st.expander("Feature descriptions"):

        descriptions = pd.DataFrame({
            "Feature": X.columns,
            "Mean": X.mean().values,
            "Standard Deviation": X.std().values,
            "Minimum": X.min().values,
            "Maximum": X.max().values
        })

        st.dataframe(
            descriptions.round(4),
            use_container_width=True
        )

    st.download_button(
        "Download Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="breast_cancer_dataset.csv",
        mime="text/csv"
    )


# =========================================================
# ABOUT
# =========================================================
elif page == "ℹ️ About":

    st.subheader("About Nexus Breast Cancer Analytics")

    st.markdown("""
    **Nexus Breast Cancer Analytics** is an educational application for
    exploring a labelled breast tumour measurement dataset.

    ### Technology Stack

    - Python
    - Streamlit
    - Pandas and NumPy
    - Plotly
    - Scikit-learn

    ### Dataset

    The Wisconsin Diagnostic Breast Cancer dataset contains 569 samples
    and 30 numerical features derived from digitized images of fine-needle
    aspirates of breast masses.

    ### Machine Learning

    A Random Forest classifier learns from a stratified training split.
    The application reports performance on a held-out test split.

    ### Limitations

    - This is not a clinically validated diagnostic model.
    - A held-out split from the same dataset is not external validation.
    - Model probabilities are not calibrated patient-level risk estimates.
    - Results may differ on new patients, hospitals and equipment.
    - Medical decisions must be made with qualified healthcare professionals.
    """)


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown(
    "<div class='footer'>"
    "NEXUS · BREAST CANCER ANALYTICS · BUILT WITH STREAMLIT"
    "</div>",
    unsafe_allow_html=True
)
