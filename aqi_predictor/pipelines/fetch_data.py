"""
Enhanced AQI Data Fetching Pipeline with Hopsworks Integration
Fetches historical and recent weather/air quality data for AQI prediction
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple
import yaml

# Lahore coordinates
LAHORE_LAT = 31.5204
LAHORE_LON = 74.3587

class OpenMeteoDataFetcher:
    """Enhanced data fetcher with historical + recent data capability"""
    
    def __init__(self, lat: float = LAHORE_LAT, lon: float = LAHORE_LON):
        self.lat = lat
        self.lon = lon
        self.archive_weather_url = 'https://archive-api.open-meteo.com/v1/archive'
        self.archive_air_quality_url = 'https://air-quality-api.open-meteo.com/v1/air-quality'
        self.forecast_weather_url = 'https://api.open-meteo.com/v1/forecast'
        self.forecast_air_quality_url = 'https://air-quality-api.open-meteo.com/v1/air-quality'
        
    def fetch_historical_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data using archive API
        
        Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
        Returns:
        pandas.DataFrame: Historical weather and air quality data
        """
        print(f"📊 Fetching historical data from {start_date} to {end_date}...")
        
        # Weather API parameters
        weather_params = {
            'latitude': self.lat,
            'longitude': self.lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': [
                'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
                'apparent_temperature', 'precipitation', 'rain', 'snowfall',
                'weather_code', 'pressure_msl', 'surface_pressure', 'cloud_cover',
                'visibility', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'
            ],
            'timezone': 'Asia/Karachi'
        }
        
        # Air Quality API parameters (excluding PM2.5, PM10 to prevent data leakage)
        air_quality_params = {
            'latitude': self.lat,
            'longitude': self.lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ['carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'uv_index', 'us_aqi'],
            'timezone': 'Asia/Karachi'
        }
        
        try:
            # Fetch weather data
            weather_response = requests.get(self.archive_weather_url, params=weather_params, timeout=30)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            # Fetch air quality data
            air_quality_response = requests.get(self.archive_air_quality_url, params=air_quality_params, timeout=30)
            air_quality_response.raise_for_status()
            air_quality_data = air_quality_response.json()
            
            # Create combined dataframe
            combined_df = self._combine_weather_air_quality(weather_data, air_quality_data)
            
            print(f"✅ Historical data fetched: {len(combined_df)} records")
            return combined_df
            
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return None
    
    def fetch_recent_data(self, days_back: int = 7, days_forward: int = 0) -> Optional[pd.DataFrame]:
        """
        Fetch recent data using forecast API (includes past_days feature)
        
        Parameters:
        days_back (int): Number of days back from today
        days_forward (int): Number of days forward from today
        
        Returns:
        pandas.DataFrame: Recent weather and air quality data
        """
        today = datetime.now().date()
        start_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = (today + timedelta(days=days_forward)).strftime('%Y-%m-%d')
        
        print(f"🔄 Fetching recent data from {start_date} to {end_date}...")
        
        # Weather forecast parameters with past_days
        weather_params = {
            'latitude': self.lat,
            'longitude': self.lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': [
                'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
                'apparent_temperature', 'precipitation', 'rain', 'snowfall',
                'weather_code', 'pressure_msl', 'surface_pressure', 'cloud_cover',
                'visibility', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'
            ],
            'timezone': 'Asia/Karachi',
            'past_days': days_back
        }
        
        # Air quality forecast parameters with past_days
        air_quality_params = {
            'latitude': self.lat,
            'longitude': self.lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ['carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'uv_index', 'us_aqi'],
            'timezone': 'Asia/Karachi',
            'past_days': days_back
        }
        
        try:
            # Fetch weather forecast (includes recent past data)
            weather_response = requests.get(self.forecast_weather_url, params=weather_params, timeout=30)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            # Fetch air quality forecast (includes recent past data)
            air_quality_response = requests.get(self.forecast_air_quality_url, params=air_quality_params, timeout=30)
            air_quality_response.raise_for_status()
            air_quality_data = air_quality_response.json()
            
            # Create combined dataframe
            combined_df = self._combine_weather_air_quality(weather_data, air_quality_data)
            
            print(f"✅ Recent data fetched: {len(combined_df)} records")
            return combined_df
            
        except Exception as e:
            print(f"❌ Error fetching recent data: {e}")
            return None
    
    def _combine_weather_air_quality(self, weather_data: dict, air_quality_data: dict) -> pd.DataFrame:
        """Combine weather and air quality data into single DataFrame"""
        
        # Create weather DataFrame
        weather_df = pd.DataFrame({
            'datetime': pd.to_datetime(weather_data['hourly']['time']),
            'temperature': weather_data['hourly']['temperature_2m'],
            'humidity': weather_data['hourly']['relative_humidity_2m'],
            'dew_point': weather_data['hourly']['dew_point_2m'],
            'apparent_temperature': weather_data['hourly']['apparent_temperature'],
            'precipitation': weather_data['hourly']['precipitation'],
            'rain': weather_data['hourly']['rain'],
            'snowfall': weather_data['hourly']['snowfall'],
            'weather_code': weather_data['hourly']['weather_code'],
            'pressure_msl': weather_data['hourly']['pressure_msl'],
            'surface_pressure': weather_data['hourly']['surface_pressure'],
            'cloud_cover': weather_data['hourly']['cloud_cover'],
            'visibility': weather_data['hourly']['visibility'],
            'wind_speed': weather_data['hourly']['wind_speed_10m'],
            'wind_direction': weather_data['hourly']['wind_direction_10m'],
            'wind_gusts': weather_data['hourly']['wind_gusts_10m']
        })
        
        # Create air quality DataFrame
        air_quality_df = pd.DataFrame({
            'datetime': pd.to_datetime(air_quality_data['hourly']['time']),
            'carbon_monoxide': air_quality_data['hourly']['carbon_monoxide'],
            'nitrogen_dioxide': air_quality_data['hourly']['nitrogen_dioxide'],
            'sulphur_dioxide': air_quality_data['hourly']['sulphur_dioxide'],
            'ozone': air_quality_data['hourly']['ozone'],
            'uv_index': air_quality_data['hourly']['uv_index'],
            'us_aqi': air_quality_data['hourly']['us_aqi']
        })
        
        # Merge datasets
        combined_df = pd.merge(weather_df, air_quality_df, on='datetime', how='inner')
        return combined_df
    
    def create_complete_dataset(self, years_back: int = 1, include_recent: bool = True) -> Optional[pd.DataFrame]:
        """
        Create complete dataset with both historical and recent data
        
        Parameters:
        years_back (int): Years of historical data to fetch
        include_recent (bool): Whether to include recent data via forecast API
        
        Returns:
        pandas.DataFrame: Complete dataset ready for feature engineering
        """
        print(f"\n🚀 CREATING COMPLETE DATASET")
        print("="*60)
        
        today = datetime.now().date()
        archive_end = today - timedelta(days=3)  # Archive API usually has 2-3 day delay
        historical_start = today - timedelta(days=365 * years_back)
        
        all_data = []
        
        # Step 1: Fetch historical data in chunks
        print(f"\n📚 STEP 1: FETCHING HISTORICAL DATA")
        print(f"📅 Range: {historical_start} to {archive_end}")
        
        current_date = historical_start
        chunk_size = 30  # 30 days per chunk
        chunk_count = 0
        
        while current_date <= archive_end:
            chunk_end = min(current_date + timedelta(days=chunk_size), archive_end)
            chunk_count += 1
            
            print(f"📦 Chunk {chunk_count}: {current_date} to {chunk_end}")
            
            chunk_data = self.fetch_historical_data(
                current_date.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d')
            )
            
            if chunk_data is not None and len(chunk_data) > 0:
                all_data.append(chunk_data)
                print(f"✅ Chunk {chunk_count}: {len(chunk_data)} records")
            else:
                print(f"⚠️ Chunk {chunk_count}: Failed to fetch data")
            
            current_date = chunk_end + timedelta(days=1)
        
        # Step 2: Fetch recent data if requested
        if include_recent:
            print(f"\n🔄 STEP 2: FETCHING RECENT DATA (LAST 7 DAYS)")
            recent_data = self.fetch_recent_data(days_back=7, days_forward=0)
            
            if recent_data is not None and len(recent_data) > 0:
                all_data.append(recent_data)
                print(f"✅ Recent data: {len(recent_data)} records")
            else:
                print(f"⚠️ Failed to fetch recent data")
        
        # Step 3: Combine all data
        if not all_data:
            print("❌ No data collected!")
            return None
        
        print(f"\n🔧 STEP 3: COMBINING AND CLEANING DATA")
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates and sort
        print(f"📊 Before deduplication: {len(combined_data)} records")
        combined_data = combined_data.drop_duplicates(subset=['datetime']).sort_values('datetime')
        print(f"📊 After deduplication: {len(combined_data)} records")
        
        # Data quality check
        missing_aqi = combined_data['us_aqi'].isnull().sum()
        if missing_aqi > len(combined_data) * 0.1:  # More than 10% missing
            print(f"⚠️ Warning: {missing_aqi} missing AQI values ({missing_aqi/len(combined_data)*100:.1f}%)")
        
        print(f"\n🎉 COMPLETE DATASET CREATED!")
        print(f"📊 Total records: {len(combined_data):,}")
        print(f"📅 Date range: {combined_data['datetime'].min()} to {combined_data['datetime'].max()}")
        print(f"🎯 Target (AQI) range: {combined_data['us_aqi'].min():.1f} - {combined_data['us_aqi'].max():.1f}")
        
        return combined_data

class HopsworksIntegration:
    """Hopsworks feature store and model registry integration"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.project = None
        self.fs = None
        self.mr = None
        self.enabled = self.config.get('hopsworks', {}).get('enable', False)
        
        if self.enabled:
            self._connect()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"⚠️ Config loading failed: {e}")
            return {}
    
    def _connect(self):
        """Connect to Hopsworks"""
        try:
            import hopsworks
            
            # Get credentials from environment or config
            api_key = os.getenv('HOPSWORKS_API_KEY')
            project_name = os.getenv('HOPSWORKS_PROJECT_NAME') or self.config.get('hopsworks', {}).get('project_name')
            
            if not api_key:
                print("⚠️ HOPSWORKS_API_KEY not found in environment variables")
                self.enabled = False
                return
            
            print(f"🔗 Connecting to Hopsworks project: {project_name}")
            self.project = hopsworks.login(
                api_key_value=api_key,
                project=project_name
            )
            
            self.fs = self.project.get_feature_store()
            self.mr = self.project.get_model_registry()
            
            print(f"✅ Connected to Hopsworks: {self.project.name}")
            
        except ImportError:
            print("❌ Hopsworks package not installed. Run: pip install hopsworks")
            self.enabled = False
        except Exception as e:
            print(f"❌ Hopsworks connection failed: {e}")
            self.enabled = False
    
    def save_to_feature_store(self, df: pd.DataFrame, stage: str = "raw") -> bool:
        """
        Save data to Hopsworks feature store
        
        Parameters:
        df (pd.DataFrame): Data to save
        stage (str): Data stage ('raw', 'engineered', 'selected')
        
        Returns:
        bool: Success status
        """
        if not self.enabled:
            print("⚠️ Hopsworks not enabled, skipping feature store save")
            return False
        
        try:
            # Configure feature group name based on stage
            fg_name = f"aqi_{stage}_features"
            fg_version = 1
            
            print(f"💾 Saving to feature store: {fg_name} v{fg_version}")
            
            # Get or create feature group
            fg = self.fs.get_or_create_feature_group(
                name=fg_name,
                version=fg_version,
                description=f"AQI {stage} features with weather and air quality data",
                primary_key=["datetime"],
                event_time="datetime",
                online_enabled=True  # Enable for real-time predictions
            )
            
            # Insert data
            fg.insert(df, write_options={"wait_for_job": False})
            
            print(f"✅ Data saved to feature store: {len(df)} records")
            return True
            
        except Exception as e:
            print(f"❌ Feature store save failed: {e}")
            return False
    
    def load_from_feature_store(self, stage: str = "selected", version: int = 1) -> Optional[pd.DataFrame]:
        """
        Load data from Hopsworks feature store
        
        Parameters:
        stage (str): Data stage to load ('raw', 'engineered', 'selected')
        version (int): Feature group version
        
        Returns:
        pandas.DataFrame: Loaded data
        """
        if not self.enabled:
            print("⚠️ Hopsworks not enabled, cannot load from feature store")
            return None
        
        try:
            fg_name = f"aqi_{stage}_features"
            print(f"📥 Loading from feature store: {fg_name} v{version}")
            
            fg = self.fs.get_feature_group(name=fg_name, version=version)
            df = fg.read()
            
            print(f"✅ Data loaded from feature store: {len(df)} records")
            return df
            
        except Exception as e:
            print(f"❌ Feature store load failed: {e}")
            return None
    
    def save_model(self, model_path: str, model_name: str, model_type: str = "sklearn") -> bool:
        """
        Save model to Hopsworks model registry
        
        Parameters:
        model_path (str): Local path to model file
        model_type (str): Type of model ('sklearn', 'tensorflow', 'pytorch')
        
        Returns:
        bool: Success status
        """
        if not self.enabled:
            print("⚠️ Hopsworks not enabled, skipping model registry save")
            return False
        
        try:
            print(f"🤖 Saving model to registry: {model_name}")
            
            # Create model in registry
            model = self.mr.create_model(
                name=model_name,
                version=1,
                description=f"AQI prediction model - {model_type}",
                metrics={"framework": model_type}
            )
            
            # Save model files
            model.save(model_path)
            
            print(f"✅ Model saved to registry: {model.name} v{model.version}")
            return True
            
        except Exception as e:
            print(f"❌ Model registry save failed: {e}")
            return False

def save_data_locally(df: pd.DataFrame, filename: str, data_dir: str = "data") -> str:
    """Save DataFrame locally with timestamp"""
    os.makedirs(data_dir, exist_ok=True)
    
    # Add timestamp to filename if it's the main dataset
    if filename in ['historical_raw_data.csv', 'complete_dataset.csv']:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = filename.split('.')
        filename = f"{name}_{timestamp}.{ext}"
    
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"💾 Data saved locally: {filepath}")
    return filepath

def main():
    """Test the enhanced data fetching pipeline"""
    print("🚀 Testing Enhanced AQI Data Fetching Pipeline")
    print("="*60)
    
    # Initialize fetcher
    fetcher = OpenMeteoDataFetcher()
    
    # Test complete dataset creation
    print("🧪 Testing complete dataset creation...")
    complete_data = fetcher.create_complete_dataset(years_back=1, include_recent=True)
    
    if complete_data is not None:
        # Save locally
        save_data_locally(complete_data, "complete_dataset.csv")
        
        # Test Hopsworks integration
        hops = HopsworksIntegration()
        if hops.enabled:
            hops.save_to_feature_store(complete_data, stage="raw")
        
        print(f"\n✅ Pipeline test completed successfully!")
        print(f"📊 Final dataset shape: {complete_data.shape}")
        print(f"📅 Date coverage: {complete_data['datetime'].min()} to {complete_data['datetime'].max()}")
        
    else:
        print("❌ Pipeline test failed!")

if __name__ == "__main__":
    main()
