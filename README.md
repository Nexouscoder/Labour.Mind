# Labour.Mind
📊 Unemployment &amp; Economic Insights Dashboard — A Streamlit-powered data analysis and forecasting tool that visualizes state-wise unemployment, literacy, GDP per capita, and labour data, with AI-generated summaries and Prophet-based future predictions.

The dashboard integrates Prophet for time-series forecasting and Hugging Face Transformers for AI-generated summaries, providing deep insights into India’s employment landscape — both historically and for the future.

🚀 Key Features:
📈 State-wise Analytics: Explore detailed metrics for each Indian state, including
                         Estimated Unemployment Rate (%)
                         Estimated Employed Population
                         Estimated Labour Participation Rate (%)
                         Literacy Rate (%)
                         GDP per Capita
                         Total Population
                         
🔮 Forecasting: Predicts future unemployment trends using Facebook Prophet based on past data.
🗺️ Heatmaps: Visualizes state-wise unemployment and literacy levels using Seaborn and Plotly Express.
🧠 AI-Generated Summaries: Uses Hugging Face Transformers (DistilBART) to summarize key insights and trends automatically.
📊 Historical vs. Future View: Compare historical data with future projections using line graphs and forecast charts.
🧮 Interactive Filtering: Select specific states, metrics, or years using Streamlit widgets for personalized insights.
⚡ Optimized Performance: Uses @st.cache_data and @st.cache_resource for efficient data loading and model caching.

📉 Insights Provided by Dashboard:
                                  Unemployment Trend Analysis – state-wise and national patterns.
                                  Literacy & Labour Link – correlation between literacy and employment.
                                  GDP vs Employment Growth – economic strength’s impact on jobs.
                                  Population Pressure – population growth impact on unemployment.
                                  AI Summaries – automatic insight generation for each state.
