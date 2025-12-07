import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.patches import Patch


# -------------------------------
# Load CSV into DataFrame
# -------------------------------
df = pd.read_csv("telecom_customer_churn.csv")

df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('%', 'pct', regex=False)
    )

df = df.drop_duplicates(subset='customer_id')

# Drop rows/columns with high missingness
thresh = int(len(df) * 0.1)
df = df.dropna(axis=1, thresh=thresh)
df = df.dropna(thresh=len(df.columns) * 0.5)

# Clean object/string columns
obj_cols = df.select_dtypes(include=['object', 'string']).columns
for c in obj_cols:
    df[c] = df[c].astype('string').str.strip()
    df[c] = df[c].str.lower()
    df[c] = df[c].fillna("unknown")

# Clean numeric columns
numeric_auto = df.select_dtypes(include=['int64', 'float64']).columns
for c in numeric_auto:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(0)

# Add churn flag
if 'customer_status' in df.columns:
    df['churn_flag'] = (df['customer_status'].astype('string').str.lower() == 'churned').astype(int)
df = df.reset_index(drop=True)

# Age tiers
bins = [0, 17, 24, 34, 44, 54, 64, 150]
labels = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age_tier'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True)
df['age_tier'] = df['age_tier'].cat.as_ordered()

# Helper for top categories
def _prepare_top(df, col, top_n=None):
    if top_n is None:
        return df
    top_vals = df[col].value_counts().nlargest(top_n).index
    return df[df[col].isin(top_vals)].copy()




# Plotly stacked count
def stacked_count_plot_plotly(df_plot, col):
    pivot = df_plot.groupby([col, 'churn_flag']).size().reset_index(name='count')
    pivot['status'] = pivot['churn_flag'].map({0: "Stayed", 1: "Churned"})

    fig = px.bar(
        pivot,
        x=col,
        y="count",
        color="status",
        barmode="stack",
        color_discrete_map={"Stayed": "#4c78a8", "Churned": "#f58523"},
        labels={"status": "Status", "count": "Count"}
    )
    fig.update_layout(
        title=f"Stacked counts by {col} (Status)",
        xaxis_title=col,
        yaxis_title="Count",
        legend_title="Status",
        xaxis_tickangle=-45,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=25),  # Increase global font size
        title_font=dict(size=25, color="white"),  # Bigger title
        xaxis=dict(title_font=dict(size=25), tickfont=dict(size=23)),
        yaxis=dict(title_font=dict(size=25), tickfont=dict(size=23)),
        legend=dict(font=dict(size=23))
    )
    return fig


def stacked_percent_plot_plotly(df_plot, col):
    pivot = df_plot.groupby([col, 'churn_flag']).size().unstack(fill_value=0)
    pct = pivot.div(pivot.sum(axis=1), axis=0).reset_index()
    pct = pct.melt(id_vars=col, value_vars=[0, 1], var_name="churn_flag", value_name="percent")
    pct['status'] = pct['churn_flag'].map({0: "Stayed", 1: "Churned"})

    fig = px.bar(
        pct,
        x=col,
        y="percent",
        color="status",
        barmode="stack",
        color_discrete_map={"Stayed": "#4c78a8", "Churned": "#f58523"},
        labels={"status": "Status", "percent": "Percent"}
    )
    fig.update_layout(
        title=f"Stacked percent by {col} (Status)",
        xaxis_title=col,
        yaxis_title="Percent",
        legend_title="Status",
        xaxis_tickangle=-45,
        yaxis=dict(tickformat=".0%", title_font=dict(size=25), tickfont=dict(size=23)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=25),
        title_font=dict(size=25, color="white"),
        xaxis=dict(title_font=dict(size=25), tickfont=dict(size=23)),
        legend=dict(font=dict(size=23))
    )
    return fig


# -------------------------------
# Streamlit UI
# -------------------------------
# Modern UI Styling
# -------------------------------
st.set_page_config(
    page_title="Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for background + modern look

st.markdown(
    """
    <style>
    /* Background gradient - brighter */
    .stApp {
        background: linear-gradient(135deg, #a1c4fd, #c2e9fb); /* light blue gradient */
        color: #000000;
        font-weight: bold;
    }

    /* Titles and headers */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: bold !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* Tables */
    .stTable {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* Divider line */
    hr {
        border: 1px solid #00000033;
    }

    /* General text (markdown, labels, selectbox, slider, etc.) */
    .stMarkdown, .stText, .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label {
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* Plotly charts text */
    .js-plotly-plot .plotly .main-svg {
        color: #000000 !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# -------------------------------
st.title("📊 Churn Dashboard")

# Metrics
churn_counts = df['churn_flag'].value_counts().reindex([0,1], fill_value=0)
col1, col2 = st.columns(2)
col1.metric("Stayed (0)", int(churn_counts.loc[0]))
col2.metric("Churned (1)", int(churn_counts.loc[1]))

avg_rev = df.groupby('churn_flag')['total_revenue'].mean().reindex([0, 1])
rev_1, rev_2 = st.columns(2)
rev_1.metric("Avg total_revenue (Stayed)", f"${avg_rev.loc[0]:,.2f}")
rev_2.metric("Avg total_revenue (Churned)", f"${avg_rev.loc[1]:,.2f}")

df2=df
########################################################################################################################################
########################################################################################################################################
# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("Filters")

# Tenure slider
tenure_slider = st.sidebar.slider(
    "Select maximum tenure (months)",
    min_value=int(df['tenure_in_months'].min()),
    max_value=int(df['tenure_in_months'].max()),
    value=int(df['tenure_in_months'].max())
)

# Churn filter
churn_filter = st.sidebar.multiselect(
    "Select churn flag(s)",
    options=df['churn_flag'].dropna().unique().tolist()
)

# Gender filter
gender_filter = st.sidebar.multiselect(
    "Select gender(s)",
    options=df['gender'].dropna().unique().tolist()
)

# Payment method filter
payment_filter = st.sidebar.multiselect(
    "Select payment method(s)",
    options=df['payment_method'].dropna().unique().tolist()
)

# Age tier filter
age_filter = st.sidebar.multiselect(
    "Select age tier(s)",
    options=df['age_tier'].dropna().unique().tolist()
)

# city tier filter
city_filter = st.sidebar.multiselect(
    "Select city",
    options=df['city'].dropna().unique().tolist()
)

# -------------------------------
# Apply filters
# -------------------------------
df_filtered = df[df['tenure_in_months'] <= tenure_slider]

if payment_filter:
    df_filtered = df_filtered[df_filtered['churn_flag'].isin(churn_filter)]

if gender_filter:
    df_filtered = df_filtered[df_filtered['gender'].isin(gender_filter)]

if payment_filter:
    df_filtered = df_filtered[df_filtered['payment_method'].isin(payment_filter)]

if age_filter:
    df_filtered = df_filtered[df_filtered['age_tier'].isin(age_filter)]

if city_filter:
    df_filtered = df_filtered[df_filtered['city'].isin(city_filter)]


df=df_filtered
########################################################################################################################################
########################################################################################################################################

# Churn category counts
st.subheader("Churn category counts")
if 'churn_category' in df.columns:
    counts = df.loc[df['churn_category'] != 'unknown', 'churn_category'].value_counts().reset_index()
    counts.columns = ['churn_category', 'count']
    fig_cat = px.bar(
        counts, 
        x='churn_category', 
        y='count', 
        text='count',
        color='count', 
        color_continuous_scale="Blues"
    )
    fig_cat.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)", 
        font=dict(color="white", size=23),   # Increase overall font size
        xaxis=dict(title_font=dict(size=25), tickfont=dict(size=25)),  # Axis labels & ticks
        yaxis=dict(title_font=dict(size=25), tickfont=dict(size=25)),
        legend=dict(font=dict(size=25))      # Legend font size
    )
    st.plotly_chart(fig_cat, use_container_width=True)



# Top 10 churn reasons
top5 = (
    df.loc[df['churn_reason'] != 'unknown', 'churn_reason']
    .value_counts()
    .nlargest(10)
)
top5_df = top5.rename_axis('churn_reason').reset_index(name='count')
top5_df = top5_df.sort_values('count', ascending=False)
st.markdown("---")
st.subheader("Top 10 Churn Reasons")
st.table(top5_df)


##########################################################################################
##########################################################################################
# Load US states map from Natural Earth data via URL

us_states = gpd.read_file("ne_110m_admin_1_states_provinces.zip")
california = us_states[us_states['name'] == 'California']


# Create figure
fig, ax = plt.subplots(figsize=(15, 10))

# Plot California
california.plot(ax=ax, color='lightgray', edgecolor='black')

# Plot points colored by churn flag
colors = ['red' if x == 1 else 'green' for x in df['churn_flag']]
ax.scatter(df['longitude'], df['latitude'], c=colors, s=10, alpha=0.7, edgecolors='black')

# Add legend
legend_elements = [Patch(facecolor='green', label='Retained'),
                   Patch(facecolor='red', label='Churned')]
ax.legend(handles=legend_elements, loc='upper left')

# Set labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('California Cities with Customer Churn Status')
ax.set_xlim([-124.5, -114.0])
ax.set_ylim([32.5, 42.0])

plt.tight_layout()
st.pyplot(fig)
##########################################################################################
##########################################################################################
######################################################################################################################################
########################################################################################################################################

# Selection and controls
col_option = st.selectbox("Select column to analyze:", ['tenure_in_months', 'gender',  'married', 'number_of_dependents',
       'city', 'zip_code', 'number_of_referrals',
        'offer', 'phone_service',
       'avg_monthly_long_distance_charges', 'multiple_lines',
       'internet_service', 'internet_type', 'avg_monthly_gb_download',
       'online_security', 'online_backup', 'device_protection_plan',
       'premium_tech_support', 'streaming_tv', 'streaming_movies',
       'streaming_music', 'unlimited_data', 'contract', 'paperless_billing',
       'payment_method', 'monthly_charge', 'total_charges', 'total_refunds',
       'total_extra_data_charges', 'total_long_distance_charges',
       'total_revenue', 'customer_status', 'churn_category', 'churn_reason',
       'churn_flag', 'age_tier'])

top_n = None
if col_option == 'city':
    top_n = st.slider("Show top N cities", min_value=5, max_value=30, value=15)

df_sel = _prepare_top(df, col_option, top_n=top_n)

fig1 = stacked_count_plot_plotly(df_sel, col_option)
st.plotly_chart(fig1, use_container_width=True)

fig2 = stacked_percent_plot_plotly(df_sel, col_option)
st.plotly_chart(fig2, use_container_width=True)