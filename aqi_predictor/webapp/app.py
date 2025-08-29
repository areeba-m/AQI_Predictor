"""
Streamlit Web Application for AQI Prediction System
Interactive dashboard for real-time AQI predictions and forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.fetch_data import OpenMeteoDataFetcher
from features.feature_engineering import FeatureEngineer
from models.model_utils import AQIPredictor
from pipelines.pipeline import AQIPipeline

# Configure Streamlit page
st.set_page_config(
    page_title="AQI Predictor - Lahore",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .alert-success {
        background-color: #d1edff;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_predictor():
    """Load the AQI predictor with caching"""
    try:
        predictor = AQIPredictor()
        success = predictor.load_models()
        if success:
            return predictor, None
        else:
            return None, "No trained models found. Please train models first."
    except ImportError as e:
        if "tensorflow" in str(e).lower():
            return None, f"TensorFlow not available: {str(e)}. Sklearn models may still work."
        else:
            return None, f"Import error: {str(e)}"
    except Exception as e:
        error_msg = str(e)
        if "could not deserialize" in error_msg.lower() or "keras" in error_msg.lower():
            return None, f"TensorFlow model compatibility issue: {error_msg}. Try using sklearn models only."
        else:
            return None, f"Error loading models: {error_msg}"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_selected_features():
    """Load the selected features list"""
    try:
        with open('data/selected_feature_names.txt', 'r') as f:
            selected_features = [line.strip() for line in f.readlines() if line.strip()]
        return selected_features
    except Exception as e:
        st.warning(f"Could not load selected features: {e}")
        return None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_latest_processed_data():
    """Fetch latest processed data from Hopsworks feature store"""
    try:
        from pipelines.fetch_data import HopsworksIntegration
        
        # Use Hopsworks integration to get processed data
        hops = HopsworksIntegration()
        
        if not hops.enabled:
            return None, None, "Hopsworks not available - using fallback data fetch"
        
        # IMPORTANT: Get selected features data (optimized for predictions), not raw engineered features
        selected_data = hops.load_from_feature_store(stage="selected")
        
        if selected_data is None or len(selected_data) == 0:
            # Fallback to engineered features if selected not available
            st.warning("Selected features not found, using engineered features")
            selected_data = hops.load_from_feature_store(stage="engineered")
            
            if selected_data is None or len(selected_data) == 0:
                return None, None, "No processed data available in feature store"
        
        # Sort by datetime and get latest records
        if 'datetime' in selected_data.columns:
            selected_data = selected_data.sort_values('datetime').tail(48)  # Last 48 hours
        
        # Also get raw data for weather correlations (if needed for UI)
        raw_data = hops.load_from_feature_store(stage="raw")
        if raw_data is not None and 'datetime' in raw_data.columns:
            raw_data = raw_data.sort_values('datetime').tail(48)
        
        return raw_data, selected_data, None
        
    except Exception as e:
        # Fallback to old method if Hopsworks fails
        return fetch_latest_data_fallback()

def fetch_latest_data_fallback():
    """Fallback method - fetch and process latest data (original method)"""
    try:
        # Initialize components
        fetcher = OpenMeteoDataFetcher()
        engineer = FeatureEngineer()
        
        # Fetch latest 48 hours of data
        latest_data = fetcher.fetch_latest_data(hours_back=48)
        
        if latest_data is None:
            return None, None, "Failed to fetch latest data"
        
        # Engineer features
        engineered_data = engineer.engineer_features(latest_data)
        engineered_data = engineer.handle_missing_values(engineered_data)
        
        # Apply feature selection to match trained models
        selected_features = load_selected_features()
        if selected_features:
            # Keep datetime and target columns, plus selected features
            available_features = [f for f in selected_features if f in engineered_data.columns]
            keep_columns = ['datetime'] + available_features
            if 'us_aqi' in engineered_data.columns:
                keep_columns.append('us_aqi')
            
            selected_data = engineered_data[keep_columns].copy()
            st.info(f"Using {len(available_features)}/{len(selected_features)} selected features")
        else:
            selected_data = engineered_data
            
        return latest_data, selected_data, "Fallback data processing used"
        
    except Exception as e:
        return None, None, f"Error fetching data: {str(e)}"

def get_aqi_color(aqi_value):
    """Get color based on AQI value"""
    if aqi_value <= 50:
        return "#00E400"  # Green
    elif aqi_value <= 100:
        return "#FFFF00"  # Yellow
    elif aqi_value <= 150:
        return "#FF7E00"  # Orange
    elif aqi_value <= 200:
        return "#FF0000"  # Red
    elif aqi_value <= 300:
        return "#8F3F97"  # Purple
    else:
        return "#7E0023"  # Maroon

def create_aqi_gauge(aqi_value, title="Current AQI"):
    """Create an AQI gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={'axis': {'range': [None, 300]},
               'bar': {'color': get_aqi_color(aqi_value)},
               'steps': [
                   {'range': [0, 50], 'color': "#d4edda"},
                   {'range': [50, 100], 'color': "#fff3cd"},
                   {'range': [100, 150], 'color': "#f8d7da"},
                   {'range': [150, 200], 'color': "#f5c6cb"},
                   {'range': [200, 300], 'color': "#e2d4f7"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 150}}))
    
    fig.update_layout(height=300)
    return fig

def create_forecast_chart(forecast_data):
    """Create forecast visualization"""
    timestamps = pd.to_datetime(forecast_data['timestamps'])
    aqi_values = forecast_data['aqi_values']
    
    # Create color mapping for AQI categories
    colors = [get_aqi_color(val) for val in aqi_values]
    
    fig = go.Figure()
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=aqi_values,
        mode='lines+markers',
        name='AQI Forecast',
        line=dict(width=3),
        marker=dict(size=6, color=colors)
    ))
    
    # Add AQI category zones
    fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
    fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
    fig.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy for Sensitive")
    fig.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Unhealthy")
    
    fig.update_layout(
        title="72-Hour AQI Forecast",
        xaxis_title="Time",
        yaxis_title="AQI Value",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_weather_correlation_chart(raw_data):
    """Create weather correlation visualization"""
    if raw_data is None or len(raw_data) == 0:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature vs AQI', 'Humidity vs AQI', 'Wind Speed vs AQI', 'Pressure vs AQI'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Temperature vs AQI
    fig.add_trace(
        go.Scatter(x=raw_data['temperature'], y=raw_data['us_aqi'], 
                  mode='markers', name='Temp vs AQI', marker=dict(color='red', size=4)),
        row=1, col=1
    )
    
    # Humidity vs AQI
    fig.add_trace(
        go.Scatter(x=raw_data['humidity'], y=raw_data['us_aqi'], 
                  mode='markers', name='Humidity vs AQI', marker=dict(color='blue', size=4)),
        row=1, col=2
    )
    
    # Wind Speed vs AQI
    fig.add_trace(
        go.Scatter(x=raw_data['wind_speed'], y=raw_data['us_aqi'], 
                  mode='markers', name='Wind vs AQI', marker=dict(color='green', size=4)),
        row=2, col=1
    )
    
    # Pressure vs AQI
    fig.add_trace(
        go.Scatter(x=raw_data['pressure_msl'], y=raw_data['us_aqi'], 
                  mode='markers', name='Pressure vs AQI', marker=dict(color='purple', size=4)),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Weather Parameters vs AQI Correlation")
    
    return fig

def create_time_series_chart(raw_data):
    """Create time series chart of recent AQI and weather data"""
    if raw_data is None or len(raw_data) == 0:
        return None
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('AQI Over Time', 'Temperature & Humidity', 'Wind Speed & Pressure'),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": True}]]
    )
    
    # AQI over time
    fig.add_trace(
        go.Scatter(x=raw_data['datetime'], y=raw_data['us_aqi'], 
                  mode='lines', name='AQI', line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Temperature and humidity
    fig.add_trace(
        go.Scatter(x=raw_data['datetime'], y=raw_data['temperature'], 
                  mode='lines', name='Temperature (°C)', line=dict(color='orange')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=raw_data['datetime'], y=raw_data['humidity'], 
                  mode='lines', name='Humidity (%)', line=dict(color='blue')),
        row=2, col=1, secondary_y=True
    )
    
    # Wind speed and pressure
    fig.add_trace(
        go.Scatter(x=raw_data['datetime'], y=raw_data['wind_speed'], 
                  mode='lines', name='Wind Speed (m/s)', line=dict(color='green')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=raw_data['datetime'], y=raw_data['pressure_msl'], 
                  mode='lines', name='Pressure (hPa)', line=dict(color='purple')),
        row=3, col=1, secondary_y=True
    )
    
    fig.update_layout(height=800, title_text="Recent Weather and AQI Trends")
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌍 AQI Predictor - Lahore, Pakistan</h1>', unsafe_allow_html=True)
    st.markdown("Real-time Air Quality Index prediction using machine learning and weather data")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Load predictor
    predictor, error = load_predictor()
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 To train models, run: `python pipelines/pipeline.py --mode train`")
        return
    
    # Model Selection
    st.sidebar.subheader("🤖 Model Selection")
    
    # Get available models info
    sklearn_available = predictor.model_manager.sklearn_model is not None
    dl_available = predictor.model_manager.dl_model is not None
    
    model_options = []
    model_labels = []
    
    if sklearn_available and dl_available:
        model_options = ['best', 'sklearn', 'deep_learning', 'ensemble']
        model_labels = [
            "🏆 Best Model (Auto-select)",
            f"🌳 Random Forest (R²: {predictor.model_manager.sklearn_metadata.get('test_r2', 0):.4f})",
            f"🧠 Deep Learning (R²: {predictor.model_manager.dl_metadata.get('test_r2', 0):.4f})",
            "🤝 Ensemble (Both models)"
        ]
    elif sklearn_available:
        model_options = ['sklearn']
        model_labels = [f"🌳 Random Forest (R²: {predictor.model_manager.sklearn_metadata.get('test_r2', 0):.4f})"]
    elif dl_available:
        model_options = ['deep_learning']
        model_labels = [f"🧠 Deep Learning (R²: {predictor.model_manager.dl_metadata.get('test_r2', 0):.4f})"]
    else:
        st.sidebar.error("No models available")
        return
    
    selected_model = st.sidebar.selectbox(
        "Choose prediction model:",
        options=model_options,
        format_func=lambda x: model_labels[model_options.index(x)],
        index=0
    )
    
    # Display model info
    if selected_model == 'sklearn' or (selected_model == 'best' and sklearn_available):
        st.sidebar.info(f"📊 Sklearn Model: {predictor.model_manager.sklearn_metadata.get('model_name', 'Random Forest')}")
    elif selected_model == 'deep_learning' or (selected_model == 'best' and not sklearn_available):
        st.sidebar.info(f"🧠 DL Model: {predictor.model_manager.dl_metadata.get('model_name', 'Deep Feedforward')}")
    elif selected_model == 'ensemble':
        st.sidebar.info("🤝 Using both models in ensemble")
    
    # Sidebar options
    show_forecast = st.sidebar.checkbox("📈 Show 72-hour Forecast", value=True)
    show_correlations = st.sidebar.checkbox("📊 Show Weather Correlations", value=True)
    show_trends = st.sidebar.checkbox("📉 Show Recent Trends", value=True)
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (5 min)", value=False)
    
    if auto_refresh:
        st.sidebar.info("Page will refresh every 5 minutes")
        # Auto refresh every 5 minutes
        import time
        time.sleep(300)
        st.experimental_rerun()
    
    # Fetch latest data
    with st.spinner("📡 Fetching latest processed data from Hopsworks..."):
        raw_data, selected_data, fetch_error = fetch_latest_processed_data()
    
    if fetch_error:
        if "Hopsworks not available" in str(fetch_error):
            st.warning("⚠️ Using fallback data fetching (slower)")
        else:
            st.error(f"❌ {fetch_error}")
            return
    
    # Current AQI prediction
    st.header("🔮 Current AQI Prediction")
    
    try:
        with st.spinner("🧠 Making prediction..."):
            current_prediction = predictor.predict_aqi(selected_data.tail(1), model_type=selected_model)
        
        if 'error' in current_prediction:
            st.error(f"❌ Prediction error: {current_prediction['error']}")
            return
        
        # Display current AQI
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            aqi_value = current_prediction['aqi_value']
            gauge_fig = create_aqi_gauge(aqi_value)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            st.metric(
                label="AQI Level",
                value=f"{aqi_value:.0f}",
                delta=f"{current_prediction['category']}"
            )
            
            # AQI category with emoji
            category_color = "green" if aqi_value <= 50 else "orange" if aqi_value <= 100 else "red"
            st.markdown(f"**Status:** <span style='color:{category_color}'>{current_prediction['emoji']} {current_prediction['category']}</span>", 
                       unsafe_allow_html=True)
        
        with col3:
            st.metric(
                label="Data Timestamp",
                value=raw_data['datetime'].iloc[-1].strftime('%H:%M'),
                delta=raw_data['datetime'].iloc[-1].strftime('%Y-%m-%d')
            )
            
            # Display model used
            model_used = current_prediction.get('model_type', selected_model)
            model_icon = "🏆" if model_used == 'best' else "🌳" if model_used == 'sklearn' else "🧠" if model_used == 'deep_learning' else "🤝"
            st.markdown(f"**Model:** {model_icon} {model_used.replace('_', ' ').title()}")
            
            # Health advisory
            if current_prediction['is_hazardous']:
                st.markdown('<div class="alert-box alert-danger">🚨 <strong>Health Alert:</strong> Hazardous air quality! Avoid outdoor activities.</div>', 
                           unsafe_allow_html=True)
            elif aqi_value > 100:
                st.markdown('<div class="alert-box alert-warning">⚠️ <strong>Caution:</strong> Unhealthy for sensitive groups.</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-box alert-success">✅ <strong>Good:</strong> Air quality is acceptable.</div>', 
                           unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return
    
    # Forecast section
    if show_forecast:
        st.header("📈 72-Hour AQI Forecast")
        
        try:
            with st.spinner("🔮 Generating forecast..."):
                forecast = predictor.predict_forecast(selected_data.tail(1), hours_ahead=72, model_type=selected_model)
            
            if 'error' not in forecast:
                # Forecast metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average AQI", f"{forecast['mean_aqi']:.0f}")
                
                with col2:
                    st.metric("Maximum AQI", f"{forecast['max_aqi']:.0f}")
                
                with col3:
                    st.metric("Unhealthy Hours", f"{forecast['unhealthy_hours']}")
                
                with col4:
                    st.metric("Hazardous Hours", f"{forecast['hazardous_hours']}")
                
                # Forecast chart
                forecast_fig = create_forecast_chart(forecast)
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Alerts
                if forecast['alerts']:
                    st.subheader("🚨 Forecast Alerts")
                    for alert in forecast['alerts']:
                        if "HAZARDOUS" in alert:
                            st.markdown(f'<div class="alert-box alert-danger">{alert}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="alert-box alert-warning">{alert}</div>', unsafe_allow_html=True)
                
                # Forecast summary
                with st.expander("📊 Detailed Forecast Summary"):
                    summary_df = pd.DataFrame({
                        'Period': ['Next 24 hours', 'Next 48 hours', 'Next 72 hours'],
                        'Average AQI': [
                            f"{forecast['forecast_summary']['next_24h_avg']:.1f}",
                            f"{forecast['forecast_summary']['next_48h_avg']:.1f}",
                            f"{forecast['forecast_summary']['next_72h_avg']:.1f}" if forecast['forecast_summary']['next_72h_avg'] else "N/A"
                        ]
                    })
                    st.dataframe(summary_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error generating forecast: {str(e)}")
    
    # Weather correlations
    if show_correlations and raw_data is not None:
        st.header("📊 Weather Parameters vs AQI")
        
        correlation_fig = create_weather_correlation_chart(raw_data)
        if correlation_fig:
            st.plotly_chart(correlation_fig, use_container_width=True)
        
        # Current weather conditions
        st.subheader("🌤️ Current Weather Conditions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        latest_weather = raw_data.iloc[-1]
        
        with col1:
            st.metric("Temperature", f"{latest_weather['temperature']:.1f}°C")
            st.metric("Humidity", f"{latest_weather['humidity']:.0f}%")
        
        with col2:
            st.metric("Wind Speed", f"{latest_weather['wind_speed']:.1f} m/s")
            st.metric("Wind Direction", f"{latest_weather['wind_direction']:.0f}°")
        
        with col3:
            st.metric("Pressure (MSL)", f"{latest_weather['pressure_msl']:.1f} hPa")
            st.metric("Cloud Cover", f"{latest_weather['cloud_cover']:.0f}%")
        
        with col4:
            st.metric("Visibility", f"{latest_weather['visibility']:.0f} m")
            st.metric("UV Index", f"{latest_weather['uv_index']:.1f}")
    
    # Recent trends
    if show_trends and raw_data is not None:
        st.header("📉 Recent Trends (48 Hours)")
        
        trends_fig = create_time_series_chart(raw_data)
        if trends_fig:
            st.plotly_chart(trends_fig, use_container_width=True)
    
    # Model information
    with st.expander("🤖 Model Information"):
        if predictor.model_manager.sklearn_metadata:
            st.subheader("Scikit-learn Model")
            sklearn_info = predictor.model_manager.sklearn_metadata
            st.write(f"**Model:** {sklearn_info.get('model_name', 'Unknown')}")
            st.write(f"**Test R²:** {sklearn_info.get('test_r2', 'N/A'):.4f}")
            st.write(f"**Test RMSE:** {sklearn_info.get('test_rmse', 'N/A'):.2f}")
        
        if predictor.model_manager.dl_metadata:
            st.subheader("Deep Learning Model")
            dl_info = predictor.model_manager.dl_metadata
            st.write(f"**Model:** {dl_info.get('model_name', 'Unknown')}")
            st.write(f"**Test R²:** {dl_info.get('test_r2', 'N/A'):.4f}")
            st.write(f"**Test RMSE:** {dl_info.get('test_rmse', 'N/A'):.2f}")
    
    # Footer
    st.markdown("---")
    st.markdown("🌍 **AQI Predictor** | Data source: Open-Meteo API | Location: Lahore, Pakistan")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
