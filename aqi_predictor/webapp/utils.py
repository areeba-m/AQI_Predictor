"""
Utility functions for the Streamlit web application
Includes SHAP explanations, alerts, and dashboard helpers
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP not available. Install with: pip install shap")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQIAlertSystem:
    """AQI alert and notification system"""
    
    def __init__(self):
        self.alert_thresholds = {
            'moderate': 100,
            'unhealthy_sensitive': 150,
            'unhealthy': 200,
            'very_unhealthy': 300,
            'hazardous': 301
        }
        
        self.health_recommendations = {
            'good': {
                'title': "✅ Good Air Quality",
                'message': "Air quality is satisfactory. Enjoy outdoor activities!",
                'color': "green",
                'recommendations': [
                    "Great day for outdoor exercise",
                    "Perfect for children to play outside",
                    "Windows can be opened for ventilation"
                ]
            },
            'moderate': {
                'title': "🟡 Moderate Air Quality",
                'message': "Air quality is acceptable for most people.",
                'color': "yellow",
                'recommendations': [
                    "Sensitive individuals should limit prolonged outdoor exertion",
                    "Still generally safe for most outdoor activities",
                    "Consider reducing intense outdoor exercise"
                ]
            },
            'unhealthy_sensitive': {
                'title': "🟠 Unhealthy for Sensitive Groups",
                'message': "Sensitive groups should reduce outdoor exposure.",
                'color': "orange",
                'recommendations': [
                    "People with heart or lung disease should reduce outdoor activities",
                    "Children and older adults should limit outdoor exposure",
                    "Consider wearing masks when outdoors"
                ]
            },
            'unhealthy': {
                'title': "🔴 Unhealthy Air Quality",
                'message': "Everyone should reduce outdoor exposure.",
                'color': "red",
                'recommendations': [
                    "Avoid prolonged outdoor activities",
                    "Keep windows closed",
                    "Use air purifiers indoors",
                    "Wear N95 masks when going outside"
                ]
            },
            'very_unhealthy': {
                'title': "🟣 Very Unhealthy Air Quality",
                'message': "Health alert: everyone may experience health effects.",
                'color': "purple",
                'recommendations': [
                    "Avoid all outdoor activities",
                    "Stay indoors with air purification",
                    "Seek medical attention if experiencing symptoms",
                    "Postpone outdoor events"
                ]
            },
            'hazardous': {
                'title': "🟤 Hazardous Air Quality",
                'message': "Emergency conditions: everyone is at risk.",
                'color': "maroon",
                'recommendations': [
                    "EMERGENCY: Stay indoors",
                    "Do not go outside unless absolutely necessary",
                    "Use high-efficiency air purifiers",
                    "Seek immediate medical attention for any symptoms"
                ]
            }
        }
    
    def get_aqi_category(self, aqi_value: float) -> str:
        """Get AQI category based on value"""
        if aqi_value <= 50:
            return 'good'
        elif aqi_value <= 100:
            return 'moderate'
        elif aqi_value <= 150:
            return 'unhealthy_sensitive'
        elif aqi_value <= 200:
            return 'unhealthy'
        elif aqi_value <= 300:
            return 'very_unhealthy'
        else:
            return 'hazardous'
    
    def generate_alert(self, aqi_value: float) -> Dict[str, Any]:
        """Generate appropriate alert for AQI value"""
        category = self.get_aqi_category(aqi_value)
        alert_info = self.health_recommendations[category].copy()
        alert_info['aqi_value'] = aqi_value
        alert_info['category'] = category
        alert_info['timestamp'] = datetime.now().isoformat()
        
        return alert_info
    
    def should_send_notification(self, aqi_value: float, last_notification_aqi: float = None) -> bool:
        """Determine if a notification should be sent"""
        current_category = self.get_aqi_category(aqi_value)
        
        # Always notify for hazardous conditions
        if current_category == 'hazardous':
            return True
        
        # Notify if crossing thresholds
        if last_notification_aqi is None:
            return aqi_value > self.alert_thresholds['moderate']
        
        last_category = self.get_aqi_category(last_notification_aqi)
        
        # Notify if category changed and getting worse
        threshold_order = ['good', 'moderate', 'unhealthy_sensitive', 'unhealthy', 'very_unhealthy', 'hazardous']
        current_idx = threshold_order.index(current_category)
        last_idx = threshold_order.index(last_category)
        
        return current_idx > last_idx
    
    def display_alert_dashboard(self, aqi_value: float) -> None:
        """Display alert information in Streamlit dashboard"""
        alert = self.generate_alert(aqi_value)
        
        # Create alert box
        alert_html = f"""
        <div style="
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid {alert['color']};
            background-color: {'#f8d7da' if alert['color'] == 'red' else '#fff3cd' if alert['color'] == 'orange' else '#d1edff'};
            margin: 1rem 0;
        ">
            <h4 style="margin: 0 0 0.5rem 0; color: {alert['color']};">{alert['title']}</h4>
            <p style="margin: 0 0 1rem 0; font-weight: bold;">{alert['message']}</p>
            <ul style="margin: 0; padding-left: 1.5rem;">
        """
        
        for rec in alert['recommendations']:
            alert_html += f"<li>{rec}</li>"
        
        alert_html += "</ul></div>"
        
        st.markdown(alert_html, unsafe_allow_html=True)

class SHAPExplainer:
    """SHAP model explanation functionality"""
    
    def __init__(self, model, feature_names: List[str], model_type: str = 'sklearn'):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        
        if not SHAP_AVAILABLE:
            st.warning("⚠️ SHAP explanations not available. Install SHAP to enable feature importance explanations.")
            return
        
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type"""
        if not SHAP_AVAILABLE:
            return
        
        try:
            if self.model_type == 'sklearn':
                if hasattr(self.model, 'feature_importances_'):
                    # Tree-based models
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    # Linear models
                    # For linear models, we need background data
                    # This is a simplified approach
                    self.explainer = shap.LinearExplainer(self.model, np.zeros((1, len(self.feature_names))))
            
            elif self.model_type == 'deep_learning':
                # For neural networks, use deep explainer
                # This would need background data in practice
                background = np.zeros((100, len(self.feature_names)))
                self.explainer = shap.DeepExplainer(self.model, background)
            
            logger.info(f"SHAP explainer initialized for {self.model_type} model")
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            st.warning(f"⚠️ Could not initialize SHAP explainer: {e}")
    
    def explain_prediction(self, features: np.ndarray, top_k: int = 10) -> Optional[Dict[str, Any]]:
        """Generate SHAP explanation for a prediction"""
        if not SHAP_AVAILABLE or self.explainer is None:
            return None
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features)
            
            # For regression models, shap_values might be 2D or 3D
            if len(shap_values.shape) > 2:
                shap_values = shap_values[:, :, 0]  # Take first output for regression
            
            # Get the explanation for the first (or only) sample
            if len(shap_values.shape) == 2:
                explanation = shap_values[0]
            else:
                explanation = shap_values
            
            # Create feature importance summary
            feature_importance = list(zip(self.feature_names, explanation))
            feature_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
            
            # Top positive and negative contributors
            positive_contributors = [(name, val) for name, val in feature_importance if val > 0][:top_k]
            negative_contributors = [(name, val) for name, val in feature_importance if val < 0][:top_k]
            
            result = {
                'shap_values': explanation,
                'feature_names': self.feature_names,
                'feature_importance': feature_importance[:top_k],
                'positive_contributors': positive_contributors,
                'negative_contributors': negative_contributors,
                'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
            }
            
            logger.info("SHAP explanation generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            st.error(f"❌ Error generating explanation: {e}")
            return None
    
    def display_explanation(self, explanation: Dict[str, Any], prediction_value: float) -> None:
        """Display SHAP explanation in Streamlit"""
        if explanation is None:
            st.info("💡 SHAP explanations are not available for this model type.")
            return
        
        st.subheader("🔍 Prediction Explanation (SHAP)")
        
        # Feature importance chart
        feature_names = [item[0] for item in explanation['feature_importance']]
        shap_values = [item[1] for item in explanation['feature_importance']]
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        colors = ['red' if val < 0 else 'green' for val in shap_values]
        
        fig.add_trace(go.Bar(
            y=feature_names,
            x=shap_values,
            orientation='h',
            marker_color=colors,
            name='SHAP Values'
        ))
        
        fig.update_layout(
            title=f"Feature Contributions to Prediction (AQI: {prediction_value:.1f})",
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="Features",
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔺 Top Factors Increasing AQI:**")
            if explanation['positive_contributors']:
                for name, value in explanation['positive_contributors'][:5]:
                    st.write(f"• {name}: +{value:.3f}")
            else:
                st.write("No significant positive contributors")
        
        with col2:
            st.markdown("**🔻 Top Factors Decreasing AQI:**")
            if explanation['negative_contributors']:
                for name, value in explanation['negative_contributors'][:5]:
                    st.write(f"• {name}: {value:.3f}")
            else:
                st.write("No significant negative contributors")

class DashboardUtils:
    """Utility functions for dashboard creation and data visualization"""
    
    @staticmethod
    def create_metric_card(title: str, value: str, delta: str = None, color: str = "blue") -> str:
        """Create a metric card HTML"""
        delta_html = f"<p style='color: gray; margin: 0;'>{delta}</p>" if delta else ""
        
        return f"""
        <div style="
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid {color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        ">
            <h4 style="margin: 0; color: {color};">{title}</h4>
            <h2 style="margin: 0.5rem 0; color: #333;">{value}</h2>
            {delta_html}
        </div>
        """
    
    @staticmethod
    def create_progress_bar(value: float, max_value: float = 300, label: str = "AQI") -> None:
        """Create a progress bar for AQI"""
        progress = min(value / max_value, 1.0)
        
        # Color based on AQI value
        if value <= 50:
            color = "#00E400"
        elif value <= 100:
            color = "#FFFF00"
        elif value <= 150:
            color = "#FF7E00"
        elif value <= 200:
            color = "#FF0000"
        elif value <= 300:
            color = "#8F3F97"
        else:
            color = "#7E0023"
        
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <p style="margin-bottom: 0.5rem; font-weight: bold;">{label}: {value:.0f}</p>
            <div style="
                width: 100%;
                height: 20px;
                background-color: #f0f0f0;
                border-radius: 10px;
                overflow: hidden;
            ">
                <div style="
                    width: {progress*100}%;
                    height: 100%;
                    background-color: {color};
                    transition: width 0.3s ease;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_forecast_summary_table(forecast_data: Dict[str, Any]) -> pd.DataFrame:
        """Create a summary table for forecast data"""
        summary_data = []
        
        # Parse forecast data
        timestamps = pd.to_datetime(forecast_data['timestamps'])
        aqi_values = forecast_data['aqi_values']
        
        # Group by time periods
        now = datetime.now()
        
        for hours, label in [(6, "Next 6h"), (12, "Next 12h"), (24, "Next 24h"), (48, "Next 48h"), (72, "Next 72h")]:
            if len(aqi_values) >= hours:
                period_values = aqi_values[:hours]
                avg_aqi = np.mean(period_values)
                max_aqi = np.max(period_values)
                min_aqi = np.min(period_values)
                
                # Determine category
                if avg_aqi <= 50:
                    category = "Good 🟢"
                elif avg_aqi <= 100:
                    category = "Moderate 🟡"
                elif avg_aqi <= 150:
                    category = "Unhealthy for Sensitive 🟠"
                elif avg_aqi <= 200:
                    category = "Unhealthy 🔴"
                elif avg_aqi <= 300:
                    category = "Very Unhealthy 🟣"
                else:
                    category = "Hazardous 🟤"
                
                summary_data.append({
                    'Period': label,
                    'Avg AQI': f"{avg_aqi:.0f}",
                    'Min AQI': f"{min_aqi:.0f}",
                    'Max AQI': f"{max_aqi:.0f}",
                    'Category': category
                })
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def display_model_performance_metrics(model_metadata: Dict[str, Any]) -> None:
        """Display model performance metrics"""
        if not model_metadata:
            return
        
        st.markdown("### 📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            r2_score = model_metadata.get('test_r2', 0)
            DashboardUtils.create_progress_bar(r2_score * 100, 100, "R² Score (%)")
        
        with col2:
            rmse = model_metadata.get('test_rmse', 0)
            st.metric("RMSE", f"{rmse:.2f}", "AQI units")
        
        with col3:
            mae = model_metadata.get('test_mae', 0)
            st.metric("MAE", f"{mae:.2f}", "AQI units")
    
    @staticmethod
    def create_weather_radar_chart(weather_data: Dict[str, float]) -> go.Figure:
        """Create a radar chart for weather conditions"""
        # Normalize weather parameters to 0-100 scale for radar chart
        categories = []
        values = []
        
        if 'temperature' in weather_data:
            categories.append('Temperature')
            values.append(min(weather_data['temperature'] / 40 * 100, 100))  # Normalize to 40°C max
        
        if 'humidity' in weather_data:
            categories.append('Humidity')
            values.append(weather_data['humidity'])  # Already 0-100
        
        if 'wind_speed' in weather_data:
            categories.append('Wind Speed')
            values.append(min(weather_data['wind_speed'] / 20 * 100, 100))  # Normalize to 20 m/s max
        
        if 'pressure_msl' in weather_data:
            categories.append('Pressure')
            values.append((weather_data['pressure_msl'] - 980) / 60 * 100)  # Normalize 980-1040 hPa
        
        if 'cloud_cover' in weather_data:
            categories.append('Cloud Cover')
            values.append(weather_data['cloud_cover'])  # Already 0-100
        
        if 'uv_index' in weather_data:
            categories.append('UV Index')
            values.append(min(weather_data['uv_index'] / 11 * 100, 100))  # Normalize to 11 max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Conditions',
            line_color='blue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Current Weather Conditions"
        )
        
        return fig

def main():
    """Test utility functions"""
    st.title("🧪 Testing Dashboard Utilities")
    
    # Test alert system
    alert_system = AQIAlertSystem()
    
    st.subheader("Alert System Test")
    test_aqi = st.slider("Test AQI Value", 0, 400, 75)
    alert_system.display_alert_dashboard(test_aqi)
    
    # Test metric cards
    st.subheader("Metric Cards Test")
    metric_html = DashboardUtils.create_metric_card("Test Metric", "42", "Last updated: now", "green")
    st.markdown(metric_html, unsafe_allow_html=True)
    
    # Test progress bar
    st.subheader("Progress Bar Test")
    DashboardUtils.create_progress_bar(test_aqi, 300, "Test AQI")
    
    # Test weather radar
    st.subheader("Weather Radar Test")
    sample_weather = {
        'temperature': 25,
        'humidity': 65,
        'wind_speed': 5,
        'pressure_msl': 1013,
        'cloud_cover': 40,
        'uv_index': 6
    }
    
    radar_fig = DashboardUtils.create_weather_radar_chart(sample_weather)
    st.plotly_chart(radar_fig, use_container_width=True)

if __name__ == "__main__":
    main()
