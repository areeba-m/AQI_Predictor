#!/usr/bin/env python3
"""
OpenMeteo AQI Forecast Comparison Script
Fetches AQI forecasts from OpenMeteo API and compares with our model predictions
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_utils import AQIPredictor
from pipelines.fetch_data import OpenMeteoDataFetcher
from features.feature_engineering import FeatureEngineer
import warnings
warnings.filterwarnings('ignore')

class OpenMeteoForecastComparison:
    """Compare our model predictions with OpenMeteo's AQI forecasts"""
    
    def __init__(self, latitude=31.5204, longitude=74.3587):  # Lahore coordinates
        self.latitude = latitude
        self.longitude = longitude
        self.predictor = None
        self.fetcher = None
        self.engineer = None
        
    def initialize(self):
        """Initialize components"""
        print("🚀 Initializing AQI Forecast Comparison with OpenMeteo...")
        
        try:
            # Load our predictor
            self.predictor = AQIPredictor()
            models_loaded = self.predictor.load_models()
            
            if not models_loaded:
                print("⚠️ No trained models found - will only show OpenMeteo forecasts")
                self.predictor = None
            else:
                print("✅ Our models loaded successfully")
            
            # Initialize data components
            self.fetcher = OpenMeteoDataFetcher()
            self.engineer = FeatureEngineer()
            
            print("✅ Components initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def fetch_openmeteo_forecast(self, days=3):
        """Fetch AQI forecast from OpenMeteo API"""
        print(f"🌤️ Fetching {days}-day AQI forecast from OpenMeteo...")
        
        try:
            # OpenMeteo Air Quality API endpoint
            url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            
            params = {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'hourly': [
                    'us_aqi',
                    'carbon_monoxide',
                    'nitrogen_dioxide',
                    'sulphur_dioxide',
                    'ozone'
                ],
                'forecast_days': days,
                'timezone': 'auto'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            hourly_data = data.get('hourly', {})
            
            if not hourly_data:
                print("❌ No hourly forecast data received")
                return None
            
            # Create DataFrame
            df = pd.DataFrame()
            df['datetime'] = pd.to_datetime(hourly_data['time'])
            df['openmeteo_aqi'] = hourly_data.get('us_aqi', [])
            df['carbon_monoxide'] = hourly_data.get('carbon_monoxide', [])
            df['nitrogen_dioxide'] = hourly_data.get('nitrogen_dioxide', [])
            df['sulphur_dioxide'] = hourly_data.get('sulphur_dioxide', [])
            df['ozone'] = hourly_data.get('ozone', [])
            
            print(f"✅ OpenMeteo forecast fetched: {len(df)} hours")
            print(f"📅 Forecast period: {df['datetime'].min()} to {df['datetime'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching OpenMeteo forecast: {e}")
            return None
    
    def fetch_our_forecast(self, days=3):
        """Generate forecast using our trained models"""
        print(f"🤖 Generating {days}-day forecast with our models...")
        
        if not self.predictor:
            print("⚠️ No trained models available")
            return None
        
        try:
            # Fetch recent data for prediction base
            recent_data = self.fetcher.fetch_latest_data(hours_back=48)
            if recent_data is None:
                print("❌ Could not fetch recent data for prediction")
                return None
            
            # Engineer features
            engineered_data = self.engineer.engineer_features(recent_data)
            engineered_data = self.engineer.handle_missing_values(engineered_data)
            
            # Generate forecast for the requested days
            forecast_hours = days * 24
            latest_features = engineered_data.tail(1)
            
            # Generate forecasts with different models
            forecasts = {}
            
            # Check available models
            models_available = {
                'sklearn': self.predictor.model_manager.sklearn_model is not None,
                'deep_learning': self.predictor.model_manager.dl_model is not None
            }
            
            model_types = ['sklearn', 'deep_learning', 'ensemble'] if all(models_available.values()) else \
                         ['sklearn'] if models_available['sklearn'] else \
                         ['deep_learning'] if models_available['deep_learning'] else []
            
            for model_type in model_types:
                try:
                    forecast = self.predictor.predict_forecast(
                        latest_features, 
                        hours_ahead=forecast_hours, 
                        model_type=model_type
                    )
                    
                    if 'error' not in forecast:
                        forecasts[model_type] = forecast
                        print(f"   ✅ {model_type} forecast generated")
                    else:
                        print(f"   ❌ {model_type} forecast failed: {forecast['error']}")
                        
                except Exception as e:
                    print(f"   ❌ {model_type} forecast error: {e}")
            
            return forecasts
            
        except Exception as e:
            print(f"❌ Error generating our forecast: {e}")
            return None
    
    def display_comparison(self, openmeteo_forecast, our_forecasts, days=3):
        """Display comparison between OpenMeteo and our forecasts"""
        print(f"\n📊 AQI FORECAST COMPARISON ({days} DAYS)")
        print("=" * 80)
        
        if openmeteo_forecast is not None:
            print("\n🌤️ OPENMETEO FORECAST:")
            print("-" * 40)
            
            # Group by day and show daily summaries
            openmeteo_forecast['date'] = openmeteo_forecast['datetime'].dt.date
            daily_summary = openmeteo_forecast.groupby('date').agg({
                'openmeteo_aqi': ['min', 'max', 'mean'],
                'datetime': 'count'
            }).round(1)
            
            for date, row in daily_summary.iterrows():
                min_aqi = row[('openmeteo_aqi', 'min')]
                max_aqi = row[('openmeteo_aqi', 'max')]
                avg_aqi = row[('openmeteo_aqi', 'mean')]
                hours = row[('datetime', 'count')]
                
                # Get AQI category for average
                category, emoji = self.get_aqi_category(avg_aqi)
                
                print(f"📅 {date}: {avg_aqi:.0f} AQI {emoji} ({min_aqi:.0f}-{max_aqi:.0f}) - {category}")
        
        if our_forecasts:
            print(f"\n🤖 OUR MODEL FORECASTS:")
            print("-" * 40)
            
            for model_type, forecast in our_forecasts.items():
                print(f"\n{self.get_model_icon(model_type)} {model_type.upper()} MODEL:")
                
                # Create hourly forecast data
                forecast_data = []
                current_time = datetime.now()
                
                for hour in range(len(forecast.get('predictions', []))):
                    forecast_time = current_time + timedelta(hours=hour)
                    aqi_value = forecast['predictions'][hour]
                    forecast_data.append({
                        'datetime': forecast_time,
                        'aqi': aqi_value,
                        'date': forecast_time.date()
                    })
                
                # Convert to DataFrame and group by day
                df = pd.DataFrame(forecast_data)
                if not df.empty:
                    daily_forecast = df.groupby('date').agg({
                        'aqi': ['min', 'max', 'mean']
                    }).round(1)
                    
                    for date, row in daily_forecast.iterrows():
                        min_aqi = row[('aqi', 'min')]
                        max_aqi = row[('aqi', 'max')]
                        avg_aqi = row[('aqi', 'mean')]
                        
                        category, emoji = self.get_aqi_category(avg_aqi)
                        print(f"   📅 {date}: {avg_aqi:.0f} AQI {emoji} ({min_aqi:.0f}-{max_aqi:.0f}) - {category}")
        
        # Comparison summary
        if openmeteo_forecast is not None and our_forecasts:
            print(f"\n🔍 COMPARISON SUMMARY:")
            print("-" * 40)
            
            # Calculate overall averages
            openmeteo_avg = openmeteo_forecast['openmeteo_aqi'].mean()
            print(f"🌤️ OpenMeteo Average: {openmeteo_avg:.1f} AQI")
            
            for model_type, forecast in our_forecasts.items():
                our_avg = np.mean(forecast.get('predictions', [0]))
                difference = our_avg - openmeteo_avg
                
                print(f"🤖 {model_type.title()} Average: {our_avg:.1f} AQI (Δ {difference:+.1f})")
        
        # Health recommendations
        print(f"\n💡 HEALTH RECOMMENDATIONS:")
        print("-" * 40)
        
        if openmeteo_forecast is not None:
            max_aqi = openmeteo_forecast['openmeteo_aqi'].max()
            avg_aqi = openmeteo_forecast['openmeteo_aqi'].mean()
            
            if max_aqi > 150:
                print("🚨 HIGH AQI EXPECTED - Hazardous conditions predicted")
                print("   • Avoid outdoor activities")
                print("   • Keep windows closed")
                print("   • Use air purifiers indoors")
            elif avg_aqi > 100:
                print("⚠️ MODERATE-HIGH AQI - Unhealthy for sensitive groups")
                print("   • Sensitive individuals should limit outdoor exposure")
                print("   • Consider wearing N95 masks outdoors")
            else:
                print("✅ GOOD AQI CONDITIONS - Safe for outdoor activities")
    
    def get_aqi_category(self, aqi_value):
        """Get AQI category and emoji"""
        if aqi_value <= 50:
            return "Good", "🟢"
        elif aqi_value <= 100:
            return "Moderate", "🟡"
        elif aqi_value <= 150:
            return "Unhealthy for Sensitive", "🟠"
        elif aqi_value <= 200:
            return "Unhealthy", "🔴"
        elif aqi_value <= 300:
            return "Very Unhealthy", "🟣"
        else:
            return "Hazardous", "🟤"
    
    def get_model_icon(self, model_type):
        """Get icon for model type"""
        icons = {
            'sklearn': '🌳',
            'deep_learning': '🧠', 
            'ensemble': '🤝'
        }
        return icons.get(model_type, '🤖')
    
    def run_comparison(self, days=3):
        """Main comparison workflow"""
        print("🚀 STARTING OPENMETEO AQI FORECAST COMPARISON")
        print("=" * 80)
        print(f"📍 Location: Lahore, Pakistan ({self.latitude}, {self.longitude})")
        print(f"📅 Forecast Period: {days} days")
        print()
        
        # Initialize
        if not self.initialize():
            return False
        
        # Fetch OpenMeteo forecast
        openmeteo_forecast = self.fetch_openmeteo_forecast(days=days)
        
        # Fetch our forecast (if models available)
        our_forecasts = self.fetch_our_forecast(days=days)
        
        # Display comparison
        if openmeteo_forecast is not None or our_forecasts:
            self.display_comparison(openmeteo_forecast, our_forecasts, days=days)
        else:
            print("❌ Could not fetch any forecasts")
            return False
        
        print(f"\n✅ Forecast comparison completed!")
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Compare AQI forecasts with OpenMeteo")
    parser.add_argument(
        '--days', 
        type=int, 
        default=3,
        help="Number of forecast days (default: 3, max: 5)"
    )
    parser.add_argument(
        '--lat', 
        type=float, 
        default=31.5204,
        help="Latitude for forecast location (default: Lahore)"
    )
    parser.add_argument(
        '--lon', 
        type=float, 
        default=74.3587,
        help="Longitude for forecast location (default: Lahore)"
    )
    
    args = parser.parse_args()
    
    # Validate days
    if args.days < 1 or args.days > 5:
        print("❌ Days must be between 1 and 5")
        return 1
    
    # Initialize comparison
    comparison = OpenMeteoForecastComparison(latitude=args.lat, longitude=args.lon)
    
    # Run comparison
    success = comparison.run_comparison(days=args.days)
    
    if success:
        print("✅ OpenMeteo forecast comparison completed successfully!")
        return 0
    else:
        print("❌ OpenMeteo forecast comparison failed!")
        return 1

if __name__ == "__main__":
    exit(main())
