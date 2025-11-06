import streamlit as st
import os
import requests
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import plotly.express as px


# Load SQLite data
@st.cache_data
def load_data():
    conn = sqlite3.connect("database.db")
    df = pd.read_sql("SELECT * FROM unemployment", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    return df

df = load_data()
st.success("✅ Data loaded from SQLite")

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
selected_state = st.sidebar.selectbox("Select State", df['State'].unique())
filtered_df = df[df['State'] == selected_state]

# Display data
st.subheader(f"📊 Unemployment Data for {selected_state}")
st.dataframe(filtered_df)

# Bar chart: Estimated Unemployment Rate
st.subheader("📈 Estimated Unemployment Rate Over Time")
fig1, ax1 = plt.subplots()
sns.lineplot(data=filtered_df, x="Date", y="Estimated Unemployment Rate (%)", ax=ax1)
st.pyplot(fig1)

# Heatmap: Correlation
st.subheader("📊 Heatmap of Correlation Between Columns")
fig2, ax2 = plt.subplots()
sns.heatmap(filtered_df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# Prophet Forecast for 2025
# Prophet Forecast for 2025
st.subheader("🔮 Forecast: Estimated Unemployment Rate in 2025")

# Prepare data for Prophet
prophet_df = filtered_df[["Date", "Estimated Unemployment Rate (%)"]].rename(columns={
    "Date": "ds",
    "Estimated Unemployment Rate (%)": "y"
})

# Initialize and train model
model = Prophet()
model.fit(prophet_df)

# Forecast until 2025 (add enough months ahead)
future = model.make_future_dataframe(periods=60, freq='M')  # 60 months = 5 years approx
forecast = model.predict(future)

# Ensure datetime
forecast['ds'] = pd.to_datetime(forecast['ds'])

# Filter only forecast for 2025
forecast_2025 = forecast[forecast['ds'].dt.year == 2025].copy()

# Rename columns for display
forecast_2025.rename(columns={
    'ds': 'Date',
    'yhat': 'Estimated Unemployment Rate (%)',
    'yhat_lower': 'Lower Estimate (%)',
    'yhat_upper': 'Upper Estimate (%)'
}, inplace=True)

# Format Date for table display
forecast_2025['Date'] = forecast_2025['Date'].dt.strftime('%b %Y')

# Show Forecast Table
st.subheader("📈 Forecasted Unemployment Rate for 2025")
if not forecast_2025.empty:
    st.dataframe(forecast_2025[['Date', 'Estimated Unemployment Rate (%)', 'Lower Estimate (%)', 'Upper Estimate (%)']])
else:
    st.warning("❌ No forecast data available for 2025. Try increasing the forecast period.")

# Plot full forecast with confidence interval
st.subheader("📊 Forecast Visualization")

# Rename for plotting
plot_df = forecast.rename(columns={
    'ds': 'Date',
    'yhat': 'Estimated Unemployment Rate (%)',
    'yhat_lower': 'Lower Estimate (%)',
    'yhat_upper': 'Upper Estimate (%)'
})

fig = px.line(
    plot_df,
    x='Date',
    y='Estimated Unemployment Rate (%)',
    title="📉 Forecasted Unemployment Rate (with Confidence Interval)",
    labels={"Estimated Unemployment Rate (%)": "Unemployment Rate (%)"},
)

# Add confidence bands
fig.add_scatter(
    x=plot_df['Date'],
    y=plot_df['Upper Estimate (%)'],
    mode='lines',
    line=dict(width=0),
    name='Upper Estimate (%)',
    showlegend=True
)
fig.add_scatter(
    x=plot_df['Date'],
    y=plot_df['Lower Estimate (%)'],
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    name='Lower Estimate (%)',
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# Smart Summary using Hugging Face summarizer

# ===== Compact, robust AI summary (replace previous block) =====
import streamlit as st
import requests

HF_TOKEN = st.secrets.get("HF_API_TOKEN")  # make sure this exists in Streamlit Secrets
ROUTER = "https://router.huggingface.co/hf-inference"

def make_readable_summary_text(df, cols):
    """Convert numeric stats into plain-language sentences."""
    parts = []
    for c in cols:
        ser = df[c].dropna()
        if ser.empty:
            continue
        mean = ser.mean()
        median = ser.median()
        mn = ser.min()
        mx = ser.max()
        parts.append(f"{c}: mean {mean:.2f}, median {median:.2f}, min {mn:.2f}, max {mx:.2f}.")
    if not parts:
        return ""
    # Optionally include state/date header
    header = ""
    if "State" in df.columns:
        states = ", ".join(map(str, df["State"].unique()[:6]))
        header += f"State(s): {states}. "
    if "Date" in df.columns:
        header += f"Data from {df['Date'].min().date()} to {df['Date'].max().date()}. "
    return header + " ".join(parts)

def local_quick_summary(text, n_sentences=2):
    # Very short fallback: return the first n sentences (text is already short)
    sents = [p.strip() for p in text.split('.') if p.strip()]
    return '. '.join(sents[:n_sentences]) + ('.' if len(sents) >= n_sentences else '')









