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

# Replace previous summarize_text_remote with this (uses new HF router endpoint)
# --- Minimal AI summary (paste after filtered_df is defined) ---
import requests
import streamlit as st

HF_TOKEN = st.secrets.get("HF_API_TOKEN")
ROUTER = "https://router.huggingface.co/hf-inference"

@st.cache_resource
def hf_summary_remote(text, model="facebook/bart-large-cnn", max_length=140, min_length=30):
    if not HF_TOKEN:
        raise RuntimeError("HF_API_TOKEN missing in Streamlit Secrets.")
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": model, "inputs": text, "parameters": {"max_length": max_length, "min_length": min_length}}
    resp = requests.post(ROUTER, headers=headers, json=payload, timeout=45)
    resp.raise_for_status()
    out = resp.json()
    # handle common response shapes
    if isinstance(out, list) and out and isinstance(out[0], dict):
        return out[0].get("summary_text") or out[0].get("generated_text") or str(out[0])
    if isinstance(out, dict):
        return out.get("summary_text") or out.get("generated_text") or str(out)
    return str(out)

def local_fallback(text, n=2):
    parts = [p.strip() for p in text.split('.') if p.strip()]
    return ('. '.join(parts[:n]) + ('.' if parts else '')) or "No text to summarize."

# choose dataframe to summarize
df_for_summary = filtered_df.copy() if "filtered_df" in globals() and filtered_df is not None else df.copy()

st.subheader("AI-Generated Summary")
if df_for_summary.empty:
    st.info("No data to summarize for current selection.")
else:
    cols = [c for c in ['Estimated Unemployment Rate (%)','Estimated Employed',
                        'Estimated Labour Participation Rate (%)','Literacy Rate (%)','GDP per Capita'] if c in df_for_summary.columns]
    if not cols:
        st.info("No matching columns found for summary.")
    else:
        text = df_for_summary[cols].describe().to_string()
        if "State" in df_for_summary.columns:
            text = f"State(s): {', '.join(map(str, df_for_summary['State'].unique()[:8]))}\n\n" + text
        try:
            with st.spinner("Generating AI summary..."):
                result = hf_summary_remote(text)
            st.info(result)
        except Exception:
            st.warning("Remote summarizer failed — showing quick local summary.")
            st.info(local_fallback(text))




