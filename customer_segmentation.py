import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide"
)


# =========================================================
# PROFESSIONAL CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #9ca3af;
    margin-bottom: 25px;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
}

.stButton button {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">👥 Customer Segmentation Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Understand customer behavior, demographics and purchasing patterns '
    'using K-Means clustering.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("customer_segmentation.csv")


# =========================================================
# DATA CLEANING
# =========================================================

df.columns = df.columns.str.strip()

df = df.drop_duplicates()

numeric_columns = [
    "Age",
    "AnnualIncome",
    "SpendingScore",
    "PurchaseFrequency",
    "TotalPurchaseAmount"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(
    subset=numeric_columns
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Segmentation Settings")

    clustering_features = st.multiselect(
        "Select clustering features",
        numeric_columns,
        default=[
            "AnnualIncome",
            "SpendingScore",
            "PurchaseFrequency",
            "TotalPurchaseAmount"
        ]
    )

    number_of_clusters = st.slider(
        "Number of Customer Segments",
        2,
        8,
        4
    )


# =========================================================
# VALIDATE FEATURES
# =========================================================

if len(clustering_features) < 2:

    st.warning(
        "Please select at least two clustering features."
    )

    st.stop()


# =========================================================
# K-MEANS
# =========================================================

X = df[clustering_features].copy()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=number_of_clusters,
    random_state=42,
    n_init=10
)

cluster_numbers = model.fit_predict(X_scaled)


# =========================================================
# SEGMENT LABELS
# =========================================================

df["Segment"] = [
    f"Segment {int(cluster) + 1}"
    for cluster in cluster_numbers
]


# =========================================================
# SILHOUETTE SCORE
# =========================================================

silhouette = silhouette_score(
    X_scaled,
    cluster_numbers
)


# =========================================================
# KPI SECTION
# =========================================================

st.subheader("📌 Customer Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "👥 Total Customers",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "💰 Average Purchase",
        f"₹{df['TotalPurchaseAmount'].mean():,.0f}"
    )

with col3:

    st.metric(
        "🛒 Avg Purchase Frequency",
        f"{df['PurchaseFrequency'].mean():.1f}"
    )

with col4:

    st.metric(
        "🎯 Customer Segments",
        number_of_clusters
    )


st.divider()


# =========================================================
# TABS
# =========================================================

overview_tab, segment_tab, behavior_tab, demographic_tab, insights_tab = st.tabs(
    [
        "📊 Overview",
        "👥 Customer Segments",
        "🛒 Purchase Behavior",
        "👤 Demographics",
        "💡 Business Insights"
    ]
)


# =========================================================
# OVERVIEW
# =========================================================

with overview_tab:

    st.subheader("📊 Customer Distribution")

    segment_counts = (
        df["Segment"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        segment_counts,
        height=400
    )

    st.subheader("🎯 Clustering Quality")

    st.metric(
        "Silhouette Score",
        f"{silhouette:.3f}"
    )

    st.info(
        "The silhouette score measures how well-separated "
        "the customer segments are. Higher values generally "
        "indicate better-defined clusters."
    )


# =========================================================
# CUSTOMER SEGMENTS
# =========================================================

with segment_tab:

    st.subheader("👥 Customer Segments")

    col1, col2 = st.columns(2)

    with col1:

        x_axis = st.selectbox(
            "Select X-axis",
            clustering_features,
            key="x_axis"
        )

    with col2:

        y_axis = st.selectbox(
            "Select Y-axis",
            clustering_features,
            index=1,
            key="y_axis"
        )


    # -----------------------------------------------------
    # PREPARE CHART DATA
    # -----------------------------------------------------

    chart_data = df[
        [
            x_axis,
            y_axis,
            "Segment"
        ]
    ].copy()

    chart_data[x_axis] = pd.to_numeric(
        chart_data[x_axis],
        errors="coerce"
    )

    chart_data[y_axis] = pd.to_numeric(
        chart_data[y_axis],
        errors="coerce"
    )

    chart_data = chart_data.dropna()


    # -----------------------------------------------------
    # NATIVE STREAMLIT SCATTER CHART
    # -----------------------------------------------------

    st.write(
        f"**{x_axis} vs {y_axis}**"
    )

    st.scatter_chart(
        chart_data,
        x=x_axis,
        y=y_axis,
        height=500
    )


    # -----------------------------------------------------
    # SEGMENT SUMMARY
    # -----------------------------------------------------

    st.subheader("📋 Segment Characteristics")

    segment_summary = (
        df.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Age=("Age", "mean"),
            Avg_Income=("AnnualIncome", "mean"),
            Avg_Spending_Score=("SpendingScore", "mean"),
            Avg_Purchase_Frequency=("PurchaseFrequency", "mean"),
            Avg_Purchase_Amount=("TotalPurchaseAmount", "mean")
        )
        .round(2)
        .reset_index()
    )

    st.dataframe(
        segment_summary,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PURCHASE BEHAVIOR
# =========================================================

with behavior_tab:

    st.subheader("🛒 Purchase Behavior")

    st.write("### Purchase Frequency by Segment")

    frequency_data = (
        df.groupby("Segment")["PurchaseFrequency"]
        .mean()
        .sort_index()
    )

    st.bar_chart(
        frequency_data,
        height=400
    )


    st.write("### Average Purchase Amount")

    purchase_data = (
        df.groupby("Segment")["TotalPurchaseAmount"]
        .mean()
        .sort_index()
    )

    st.bar_chart(
        purchase_data,
        height=400
    )


    st.write("### Spending Score")

    spending_data = (
        df.groupby("Segment")["SpendingScore"]
        .mean()
        .sort_index()
    )

    st.bar_chart(
        spending_data,
        height=400
    )


# =========================================================
# DEMOGRAPHICS
# =========================================================

with demographic_tab:

    st.subheader("👤 Customer Demographics")


    st.write("### Average Age by Segment")

    age_data = (
        df.groupby("Segment")["Age"]
        .mean()
        .sort_index()
    )

    st.bar_chart(
        age_data,
        height=400
    )


    st.write("### Gender Distribution")

    gender_data = pd.crosstab(
        df["Segment"],
        df["Gender"]
    )

    st.bar_chart(
        gender_data,
        height=400
    )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

with insights_tab:

    st.subheader("💡 Business Insights")

    insight_summary = (
        df.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Income=("AnnualIncome", "mean"),
            Avg_Spending=("SpendingScore", "mean"),
            Avg_Frequency=("PurchaseFrequency", "mean"),
            Avg_Purchase=("TotalPurchaseAmount", "mean")
        )
        .round(2)
        .reset_index()
    )


    # Highest spending

    highest_spending = insight_summary.loc[
        insight_summary["Avg_Spending"].idxmax()
    ]


    # Highest frequency

    highest_frequency = insight_summary.loc[
        insight_summary["Avg_Frequency"].idxmax()
    ]


    # Highest purchase value

    highest_value = insight_summary.loc[
        insight_summary["Avg_Purchase"].idxmax()
    ]


    col1, col2, col3 = st.columns(3)

    with col1:

        st.success(
            f"💰 Highest Spending\n\n"
            f"**{highest_spending['Segment']}**\n\n"
            f"Score: {highest_spending['Avg_Spending']:.1f}"
        )

    with col2:

        st.info(
            f"🛒 Most Frequent\n\n"
            f"**{highest_frequency['Segment']}**\n\n"
            f"Frequency: {highest_frequency['Avg_Frequency']:.1f}"
        )

    with col3:

        st.warning(
            f"💎 Highest Value\n\n"
            f"**{highest_value['Segment']}**\n\n"
            f"Purchase: ₹{highest_value['Avg_Purchase']:,.0f}"
        )


    st.subheader("📋 Customer Segmentation Results")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader("🎯 Business Recommendation")

    st.write(
        "High-value and high-spending customers can be targeted "
        "with loyalty programs, personalized offers and exclusive "
        "promotions. Lower-frequency customers can be encouraged "
        "through discounts and targeted campaigns."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Customer Segmentation Dashboard | "
    "Python • Pandas • Scikit-learn • Streamlit"
)