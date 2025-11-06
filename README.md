# Labour.Mind
📊 Unemployment &amp; Economic Insights Dashboard — A Streamlit-powered data analysis and forecasting tool that visualizes state-wise unemployment, literacy, GDP per capita, and labour data, with AI-generated summaries and Prophet-based future predictions.

The dashboard integrates Prophet for time-series forecasting and Hugging Face Transformers for AI-generated summaries, providing deep insights into India’s employment landscape — both historically and for the future.

🔧 How It Works

Data Loading

Reads data from database.db using sqlite3 and pandas.

Uses @st.cache_data to avoid reloading during refresh.

Visualization

Generates dynamic plots and heatmaps using Seaborn, Matplotlib, and Plotly.

Displays trends in unemployment, employment, and labor participation rates.

Forecasting

Uses Prophet to predict future unemployment rates.

Handles trends, seasonal components, and time-based features automatically.

AI-Generated Summary

Integrates Hugging Face’s summarization pipeline (sshleifer/distilbart-cnn-12-6).

Summarizes statistical data into a short natural language insight block.
