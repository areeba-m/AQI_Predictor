"""
Enhanced AQI Data Fetching Pipeline with Hopsworks Integration
Fetches historical and recent weather/air quality data for AQI prediction
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import glob
import tempfile
import shutil
import traceback
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import yaml

# Lahore coordinates
LAHORE_LAT = 31.5204
LAHORE_LON = 74.3587

# Helper functions
def log_error(msg: str, e: Exception) -> None:
    """Centralized error logging helper"""
    print(f"❌ {msg}: {e}")
    print(f"   Error type: {type(e).__name__}")
    traceback.print_exc()

def safe_save(fetcher, df: pd.DataFrame, stage: str) -> bool:
    """Helper to safely save data to Hopsworks if enabled"""
    if not fetcher.hops_integration or not fetcher.hops_integration.enabled:
        print(f"⚠️ Hopsworks not enabled, skipping {stage} save")
        return False
    
    print(f"☁️ Attempting to save {len(df)} {stage} records to Hopsworks...")
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df['datetime'].min() if 'datetime' in df.columns else 'N/A'} to {df['datetime'].max() if 'datetime' in df.columns else 'N/A'}")
    
    try:
        success = fetcher.hops_integration.save_to_feature_store(df, stage=stage)
        if success:
            print(f"✅ Successfully saved {stage} features to Hopsworks: {len(df)} records")
            print(f"   💡 Hopsworks feature group should now be updated")
        else:
            print(f"❌ Failed to save {stage} features to Hopsworks")
            print(f"   🔍 Check Hopsworks logs for details")
        return success
    except Exception as e:
        log_error(f"Hopsworks {stage} save failed", e)
        print(f"   💥 Exception details: {str(e)}")
        return False

def load_latest_data(fetcher) -> Optional[pd.DataFrame]:
    """Helper to load latest data from Hopsworks first, then local files"""
    raw_data = None
    
    # Try loading from Hopsworks first
    if fetcher.hops_integration and fetcher.hops_integration.enabled:
        print("📥 Loading data from Hopsworks feature store...")
        try:
            raw_data = fetcher.hops_integration.load_from_feature_store(stage="raw")
            if raw_data is not None and len(raw_data) > 0:
                print(f"✅ Loaded {len(raw_data)} records from Hopsworks")
                return raw_data
        except Exception as e:
            log_error("Failed to load from Hopsworks", e)
    
    # Try local files as fallback
    print("⚠️ No data in Hopsworks, trying local files...")
    csv_files = glob.glob("data/historical_*.csv") + glob.glob("data/complete_dataset_*.csv")
    if csv_files:
        try:
            latest_file = max(csv_files, key=os.path.getctime)
            print(f"📁 Loading from local file: {latest_file}")
            raw_data = pd.read_csv(latest_file)
            raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
            print(f"✅ Loaded {len(raw_data)} records from local file")
            return raw_data
        except Exception as e:
            log_error("Failed to load from local files", e)
    
    print("❌ No local data files found")
    return None

def save_with_timestamp(df: pd.DataFrame, filename: str, data_dir: str = "data") -> str:
    """Helper to save DataFrame locally with timestamp"""
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

def fetch_data_for_period(fetcher, start_date: str, end_date: str, description: str = "") -> Optional[pd.DataFrame]:
    """Helper to fetch data for a specific period with consistent error handling"""
    print(f"📊 Fetching {description} data from {start_date} to {end_date}...")
    
    try:
        data = fetcher.fetch_historical_data(start_date, end_date)
        if data is not None and len(data) > 0:
            print(f"✅ {description} data fetched: {len(data)} records")
            return data
        else:
            print(f"⚠️ No {description} data available for this period")
            return None
    except Exception as e:
        log_error(f"{description} data fetch failed", e)
        return None

class OpenMeteoDataFetcher:
    """Enhanced data fetcher with historical + recent data capability"""
    
    def __init__(self, lat: float = LAHORE_LAT, lon: float = LAHORE_LON):
        self.lat = lat
        self.lon = lon
        self.archive_weather_url = 'https://archive-api.open-meteo.com/v1/archive'
        self.archive_air_quality_url = 'https://air-quality-api.open-meteo.com/v1/air-quality'
        self.forecast_weather_url = 'https://api.open-meteo.com/v1/forecast'
        self.forecast_air_quality_url = 'https://air-quality-api.open-meteo.com/v1/air-quality'
        
        # Initialize Hopsworks integration
        self.hops_integration = HopsworksIntegration()
        
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
            'hourly': ','.join([
                'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
                'apparent_temperature', 'precipitation', 'rain', 'snowfall',
                'weather_code', 'pressure_msl', 'surface_pressure', 'cloud_cover',
                'visibility', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'
            ]),
            'timezone': 'Asia/Karachi'
        }
        
        # Air Quality API parameters (excluding PM2.5, PM10 to prevent data leakage)
        air_quality_params = {
            'latitude': self.lat,
            'longitude': self.lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ','.join(['carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'uv_index', 'us_aqi']),
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
        Fetch recent data using archive API for recent past data
        
        Parameters:
        days_back (int): Number of days back from today
        days_forward (int): Number of days forward from today (should be 0 for historical data)
        
        Returns:
        pandas.DataFrame: Recent weather and air quality data
        """
        print(f"🔄 Fetching recent data for last {days_back} days...")
        
        # Use archive API for recent past data instead of forecast API
        today = datetime.now().date()
        start_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')  # Yesterday
        
        print(f"📅 Using archive API for date range: {start_date} to {end_date}")
        
        # Use the historical data fetcher for recent data
        return self.fetch_historical_data(start_date, end_date)
    
    def _combine_weather_air_quality(self, weather_data: dict, air_quality_data: dict) -> pd.DataFrame:
        """Combine weather and air quality data into single DataFrame"""
        
        # Create weather DataFrame with null handling
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
        
        # Handle visibility nulls specifically (common issue with Open-Meteo)
        if weather_df['visibility'].isnull().all():
            weather_df['visibility'] = 10000.0  # Default 10km visibility
        elif weather_df['visibility'].isnull().any():
            weather_df['visibility'] = weather_df['visibility'].fillna(10000.0)
        
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
        
        # Final cleanup - ensure no null dtypes
        for col in combined_df.columns:
            if col != 'datetime' and combined_df[col].isnull().any():
                combined_df[col] = combined_df[col].fillna(combined_df[col].median() if combined_df[col].dtype in ['float64', 'int64'] else 0)
        
        return combined_df
    
    def create_complete_dataset(self, years_back: int = 1, include_recent: bool = True) -> Optional[pd.DataFrame]:
        """
        Create complete dataset by fetching fresh data
        
        Parameters:
        years_back (int): Years of historical data to fetch
        include_recent (bool): Whether to include recent data updates
        
        Returns:
        pandas.DataFrame: Complete dataset ready for feature engineering
        """
        print(f"\n🚀 CREATING COMPLETE DATASET")
        print("="*60)
        print(f"📅 Historical: {years_back} year(s) back")
        print(f"🔄 Include recent: {include_recent}")
        
        all_data = []
        today = datetime.now()
        
        try:
            # Fetch historical data in chunks
            archive_end = today - timedelta(days=3)  # Archive API usually has 2-3 day delay
            historical_start = today - timedelta(days=365 * years_back)
            
            print(f"\n📈 Fetching historical data from {historical_start.date()} to {archive_end.date()}")
            
            current_date = historical_start
            chunk_size = 30  # 30 days per chunk
            chunk_count = 0
            
            while current_date <= archive_end:
                chunk_end = min(current_date + timedelta(days=chunk_size), archive_end)
                chunk_count += 1
                
                chunk_data = fetch_data_for_period(
                    self,
                    current_date.strftime('%Y-%m-%d'),
                    chunk_end.strftime('%Y-%m-%d'),
                    f"Chunk {chunk_count}"
                )
                
                if chunk_data is not None:
                    all_data.append(chunk_data)
                
                current_date = chunk_end + timedelta(days=1)
            
            # Fetch recent data if requested
            if include_recent:
                print(f"\n🔄 Fetching latest data (last 2 days)")
                recent_data = fetch_data_for_period(
                    self,
                    (today - timedelta(days=2)).strftime('%Y-%m-%d'),
                    (today - timedelta(days=1)).strftime('%Y-%m-%d'),
                    "Recent"
                )
                if recent_data is not None:
                    all_data.append(recent_data)
            
            # Combine all data
            if not all_data:
                print("❌ No data collected!")
                return None
            
            print(f"\n🔧 Combining and cleaning data")
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates and sort
            print(f"📊 Before deduplication: {len(combined_data)} records")
            combined_data = combined_data.drop_duplicates(subset=['datetime']).sort_values('datetime')
            print(f"📊 After deduplication: {len(combined_data)} records")
            
            print(f"\n🎉 Dataset ready!")
            print(f"📊 Total records: {len(combined_data):,}")
            print(f"📅 Date range: {combined_data['datetime'].min()} to {combined_data['datetime'].max()}")
            
            return combined_data
            
        except Exception as e:
            log_error("Dataset creation failed", e)
            return None
        
    def get_latest_data_timestamp(self) -> Optional[datetime]:
        """
        Get the latest timestamp from existing data sources
        
        Returns:
        datetime: Latest timestamp or None if no data exists
        """
        latest_timestamps = []
        
        # Check Hopsworks feature store first
        if self.hops_integration and self.hops_integration.enabled:
            try:
                # Check if feature group exists first
                fg_name = "aqi_raw_features"
                print(f"🔍 Checking if feature group {fg_name} exists...")
                
                try:
                    fg = self.hops_integration.fs.get_feature_group(name=fg_name, version=1)
                    if fg is None:
                        print(f"📭 Feature group {fg_name} does not exist yet")
                    else:
                        print(f"✅ Feature group {fg_name} exists, loading data...")
                        existing_data = self.hops_integration.load_from_feature_store(stage="raw")
                        if existing_data is not None and len(existing_data) > 0:
                            # Ensure datetime column is timezone-naive for consistency
                            existing_data['datetime'] = pd.to_datetime(existing_data['datetime'])
                            if hasattr(existing_data['datetime'].iloc[0], 'tz') and existing_data['datetime'].iloc[0].tz is not None:
                                existing_data['datetime'] = existing_data['datetime'].dt.tz_localize(None)
                            latest_hops = existing_data['datetime'].max()
                            latest_timestamps.append(latest_hops)
                            print(f"📊 Latest in Hopsworks: {latest_hops}")
                        else:
                            print(f"📭 Feature group exists but contains no data")
                except Exception as fg_error:
                    print(f"📭 Feature group {fg_name} does not exist: {fg_error}")
                    
            except Exception as e:
                print(f"⚠️ Could not check Hopsworks data: {e}")
        
        # Check local CSV files
        import glob
        csv_files = glob.glob("data/historical_*.csv") + glob.glob("data/complete_dataset_*.csv")
        if csv_files:
            try:
                latest_file = max(csv_files, key=os.path.getctime)
                print(f"📁 Checking local file: {latest_file}")
                local_data = pd.read_csv(latest_file)
                # Ensure datetime is timezone-naive for consistency
                local_data['datetime'] = pd.to_datetime(local_data['datetime'])
                if hasattr(local_data['datetime'].iloc[0], 'tz') and local_data['datetime'].iloc[0].tz is not None:
                    local_data['datetime'] = local_data['datetime'].dt.tz_localize(None)
                latest_local = local_data['datetime'].max()
                latest_timestamps.append(latest_local)
                print(f"📁 Latest in local files: {latest_local}")
            except Exception as e:
                print(f"⚠️ Could not check local files: {e}")
        
        if latest_timestamps:
            latest = max(latest_timestamps)
            print(f"🕐 Latest data timestamp: {latest}")
            return latest
        else:
            print(f"📭 No existing data found in Hopsworks or local files")
            return None

    def fetch_incremental_data(self, since_timestamp: datetime, max_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch data incrementally since the last timestamp
        
        Parameters:
        since_timestamp (datetime): Last data timestamp
        max_days (int): Maximum days to fetch in one go
        
        Returns:
        pandas.DataFrame: New incremental data
        """
        try:
            now = datetime.now()
            
            # Handle timezone differences between now and since_timestamp
            if hasattr(since_timestamp, 'tz') and since_timestamp.tz is not None:
                # since_timestamp is timezone-aware, make it timezone-naive for comparison
                if hasattr(since_timestamp, 'tz_localize'):
                    since_timestamp = since_timestamp.tz_localize(None)
                else:
                    # Convert pandas timestamp to naive datetime
                    since_timestamp = since_timestamp.replace(tzinfo=None)
            
            start_date = since_timestamp + timedelta(hours=1)  # Start from next hour
            
            # For archive API, we can get data up to about 3 hours ago
            # Use the earlier of: (3 hours ago) or (requested end date)
            max_available_date = now - timedelta(hours=3)
            requested_end_date = start_date + timedelta(days=max_days)
            end_date = min(max_available_date, requested_end_date)
            
            # Ensure we don't try to fetch data from the future or too far back
            if start_date >= end_date:
                print(f"⏰ No new data to fetch (start: {start_date}, end: {end_date})")
                print(f"   Data is too recent - archive API has ~3 hour delay")
                return None
            
            return fetch_data_for_period(
                self,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                "Incremental"
            )
            
        except Exception as e:
            log_error("Incremental data fetch failed", e)
            return None

    def setup_initial_historical_data(self, years_back: int = 1) -> bool:
        """
        Set up initial historical data if none exists
        
        Parameters:
        years_back (int): Years of historical data to fetch
        
        Returns:
        bool: Success status
        """
        print(f"🏗️ Setting up initial historical data ({years_back} year(s))")
        
        try:
            # Use create_complete_dataset to get full historical data
            print(f"📊 Fetching historical data...")
            historical_data = self.create_complete_dataset(years_back=years_back, include_recent=False)
            
            if historical_data is None or len(historical_data) == 0:
                print("❌ Failed to fetch initial historical data")
                return False
            
            print(f"✅ Successfully fetched {len(historical_data)} historical records")
            print(f"📅 Date range: {historical_data['datetime'].min()} to {historical_data['datetime'].max()}")
            
            # Save locally first (always works)
            print(f"💾 Saving data locally...")
            local_path = save_with_timestamp(historical_data, "historical_raw_data.csv")
            print(f"✅ Saved {len(historical_data)} historical records locally: {local_path}")
            
            # Save to Hopsworks using helper function
            hopsworks_success = safe_save(self, historical_data, "raw")
            
            if not hopsworks_success:
                print("⚠️ Failed to save historical data to Hopsworks, but local save succeeded")
                print("   Pipeline can continue using local files")
            
            print(f"✅ Initial historical data setup complete")
            return True
            
        except Exception as e:
            log_error("Initial historical data setup failed", e)
            return False
    
    def fetch_latest_data(self, hours_back: int = 48) -> Optional[pd.DataFrame]:
        """
        Fetch latest data for real-time predictions using archive API
        
        Parameters:
        hours_back (int): Number of hours back to fetch
        
        Returns:
        pandas.DataFrame: Latest weather and air quality data
        """
        print(f"🔄 Fetching latest {hours_back} hours of data...")
        
        # Calculate date range for latest data
        now = datetime.now()
        start_datetime = now - timedelta(hours=hours_back)
        end_datetime = now - timedelta(hours=2)  # 2 hours ago to ensure data availability
        
        start_date = start_datetime.strftime('%Y-%m-%d')
        end_date = end_datetime.strftime('%Y-%m-%d')
        
        print(f"� Fetching from {start_date} to {end_date}")
        
        # Use the archive API for reliable data
        latest_data = self.fetch_historical_data(start_date, end_date)
        
        if latest_data is not None:
            # Filter to get only the requested hours if we got more data
            latest_data['datetime'] = pd.to_datetime(latest_data['datetime'])
            cutoff_time = now - timedelta(hours=hours_back)
            latest_data = latest_data[latest_data['datetime'] >= cutoff_time]
            
            print(f"✅ Latest data fetched: {len(latest_data)} records")
            if len(latest_data) > 0:
                print(f"📅 Time range: {latest_data['datetime'].min()} to {latest_data['datetime'].max()}")
        
        return latest_data
    
    def check_existing_data(self) -> Optional[Tuple[datetime, int]]:
        """
        Check for existing data in both Hopsworks and local files
        
        Returns:
        Tuple[datetime, int]: (latest_timestamp, record_count) or None if no data
        """
        latest_timestamp = self.get_latest_data_timestamp()
        if latest_timestamp is None:
            return None
        
        # Get record count
        record_count = 0
        if self.hops_integration and self.hops_integration.enabled:
            try:
                existing_data = self.hops_integration.load_from_feature_store(stage="raw")
                if existing_data is not None:
                    record_count = len(existing_data)
            except:
                pass
        
        if record_count == 0:
            # Try local files
            import glob
            csv_files = glob.glob("data/historical_*.csv")
            if csv_files:
                latest_file = max(csv_files, key=os.path.getctime)
                local_data = pd.read_csv(latest_file)
                record_count = len(local_data)
        
        return (latest_timestamp, record_count) if record_count > 0 else None

class HopsworksIntegration:
    """Hopsworks feature store and model registry integration"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
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
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"⚠️ Config loading failed: {e}")
            return {}
    
    def _connect(self):
        """Connect to Hopsworks"""
        try:
            print("📦 Importing Hopsworks...")
            import hopsworks
            print("✅ Hopsworks imported successfully")
            
            # Get credentials from .env file first, then environment variables
            api_key = os.getenv('HOPSWORKS_API_KEY')
            project_name = os.getenv('HOPSWORKS_PROJECT_NAME') or self.config.get('hopsworks', {}).get('project_name')
            
            print(f"🔑 API Key: {'Found' if api_key else 'Not found'}")
            print(f"📁 Project Name: {project_name}")
            
            if not api_key:
                print("⚠️ HOPSWORKS_API_KEY not found in .env file or environment variables")
                print("   Please add HOPSWORKS_API_KEY=your_key to your .env file")
                self.enabled = False
                return
            
            if not project_name:
                print("⚠️ HOPSWORKS_PROJECT_NAME not found in .env file or config")
                print("   Please add HOPSWORKS_PROJECT_NAME=your_project to your .env file")
                self.enabled = False
                return
            
            print(f"🔗 Connecting to Hopsworks project: {project_name}")
            print(f"🔑 Using API key from .env file...")
            
            self.project = hopsworks.login(
                api_key_value=api_key,
                project=project_name
            )
            
            if self.project is None:
                print("❌ Failed to connect to Hopsworks project")
                self.enabled = False
                return
            
            print(f"✅ Connected to project: {self.project.name}")
            
            self.fs = self.project.get_feature_store()
            print(f"🏪 Feature store object: {self.fs}")
            if self.fs is None:
                print("❌ Failed to get feature store")
                self.enabled = False
                return
            
            print(f"✅ Feature store obtained: {self.fs.name}")
            
            self.mr = self.project.get_model_registry()
            print(f"✅ Model registry obtained")
            
            print(f"✅ Connected to Hopsworks: {self.project.name}")
            
        except ImportError as ie:
            print(f"❌ Hopsworks package not installed: {ie}")
            print("   Run: pip install hopsworks")
            self.enabled = False
        except Exception as e:
            log_error("Hopsworks connection failed", e)
            print("   Please check your API key and project name in .env file")
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
            # Prepare dataframe for Hopsworks
            df_copy = df.copy()
            
            # Simple data cleaning for Hopsworks compatibility
            print(f"🧹 Cleaning data for Hopsworks...")
            
            # Fill null values with appropriate defaults
            for col in df_copy.columns:
                if col == 'datetime':
                    continue
                    
                if df_copy[col].isnull().any():
                    if col == 'visibility':
                        df_copy[col] = df_copy[col].fillna(10000.0)
                    elif df_copy[col].dtype in ['float64', 'int64']:
                        df_copy[col] = df_copy[col].fillna(0.0)
                    else:
                        df_copy[col] = df_copy[col].fillna(0.0)
            
            # Define columns that should be integers in Hopsworks
            # Note: Keep this list conservative - only include columns we're sure should be integers
            integer_columns = [
                'humidity',           # 0-100 percentage  
                'weather_code',       # API weather codes
                'cloud_cover',        # 0-100 percentage
                'wind_direction',     # 0-360 degrees
                'us_aqi',            # AQI values are typically whole numbers
                # Note: uv_index removed - Hopsworks expects it as double
                # Note: precipitation, rain, snowfall can be fractional so keeping as float
            ]
            
            # Convert data types appropriately for Hopsworks compatibility
            for col in df_copy.columns:
                if col == 'datetime':
                    continue
                elif col in integer_columns:
                    # Convert to integers (fill NaN with 0 first)
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0.0).astype('int64')
                else:
                    # Keep as float for other numeric columns
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0.0)
            
            # Add integer primary key (required for feature groups)
            df_copy['id'] = range(len(df_copy))
            
            # Ensure datetime column is properly formatted
            if 'datetime' in df_copy.columns:
                df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
            
            # Debug: Print data types to help identify schema issues
            print(f"🔍 Data types for Hopsworks compatibility:")
            for col in df_copy.columns:
                dtype = df_copy[col].dtype
                print(f"   {col}: {dtype}")
            
            print(f"   ✅ Shape: {df_copy.shape}")
            print(f"   ✅ Null values: {df_copy.isnull().sum().sum()}")
            
            # Configure feature group name based on stage
            fg_name = f"aqi_{stage}_features"
            fg_version = 1
            
            print(f"💾 Saving to feature store: {fg_name} v{fg_version}")
            
            # Get or create feature group with schema compatibility checking
            fg = None
            
            # Check for existing feature group and schema compatibility
            print(f"🔍 Checking for existing feature group: {fg_name}")
            try:
                fg = self.fs.get_feature_group(name=fg_name, version=fg_version)
                if fg is not None:
                    print(f"📋 Found existing feature group: {fg_name}")
                    
                    # Check schema compatibility
                    print(f"🔍 Checking schema compatibility...")
                    existing_schema = {f.name: f.type for f in fg.features}
                    new_columns = set(df_copy.columns)
                    existing_columns = set(existing_schema.keys())
                    
                    missing_in_new = existing_columns - new_columns
                    extra_in_new = new_columns - existing_columns
                    
                    schema_compatible = len(missing_in_new) == 0 and len(extra_in_new) == 0
                    
                    if schema_compatible:
                        print(f"✅ Schema is compatible, will update existing data")
                    else:
                        print(f"⚠️ Schema mismatch detected:")
                        if missing_in_new:
                            print(f"   Missing columns in new data: {list(missing_in_new)}")
                        if extra_in_new:
                            print(f"   Extra columns in new data: {list(extra_in_new)}")
                        
                        print(f"🔄 Since full pipeline is running, will recreate feature group with new schema")
                        print(f"💡 This will overwrite all previous data as requested")
                        
                        # Delete existing feature group
                        print(f"🗑️ Deleting existing feature group: {fg_name}")
                        try:
                            fg.delete()
                            print(f"✅ Successfully deleted existing feature group")
                            fg = None  # Set to None to trigger recreation
                        except Exception as delete_error:
                            print(f"⚠️ Failed to delete feature group: {delete_error}")
                            print(f"   Will try creating a new version instead")
                            fg_version += 1
                            fg = None
                else:
                    print(f"📭 Feature group {fg_name} not found, will create new one")
                    raise Exception("Feature group not found, creating new")
            except Exception as get_error:
                print(f"🆕 Will create feature group: {fg_name}")
                if "not found" not in str(get_error).lower():
                    print(f"   Reason: {get_error}")
            
            # Create feature group if needed
            if fg is None:
                print(f"🔧 Creating new feature group: {fg_name} v{fg_version}")
                
                # Create new feature group with robust configuration
                creation_attempts = [
                    # Attempt 1: With event_time
                    {
                        "name": fg_name,
                        "version": fg_version,
                        "description": f"AQI {stage} features with weather and air quality data for Lahore, Pakistan (Schema updated: {datetime.now().strftime('%Y-%m-%d')})",
                        "primary_key": ["id"],
                        "event_time": "datetime" if "datetime" in df_copy.columns else None,
                        "online_enabled": False
                    },
                    # Attempt 2: Without event_time
                    {
                        "name": fg_name,
                        "version": fg_version,
                        "description": f"AQI {stage} features with weather and air quality data",
                        "primary_key": ["id"],
                        "online_enabled": False
                    },
                    # Attempt 3: Minimal configuration
                    {
                        "name": fg_name,
                        "description": f"AQI {stage} features",
                        "primary_key": ["id"]
                    }
                ]
                
                fg_created = False
                for i, config in enumerate(creation_attempts, 1):
                    try:
                        print(f"   Attempt {i}: Creating with config: {list(config.keys())}")
                        fg = self.fs.create_feature_group(**config)
                        if fg is not None:
                            print(f"✅ Successfully created feature group with attempt {i}")
                            fg_created = True
                            break
                        else:
                            print(f"⚠️ Attempt {i} returned None")
                    except Exception as create_error:
                        print(f"⚠️ Attempt {i} failed: {create_error}")
                        continue
                
                if not fg_created or fg is None:
                    print(f"❌ All feature group creation attempts failed")
                    return False
            
            # Verify feature group exists before inserting
            if fg is None:
                print("❌ Feature group is None, cannot insert data")
                return False
            
            print(f"✅ Feature group ready: {fg_name}")
            
            # Insert data using direct insert
            print(f"📤 Inserting {len(df_copy)} records into {fg_name}...")
            try:
                # Simple direct insert without complex options
                fg.insert(df_copy)
                print(f"✅ Data inserted to feature store: {len(df_copy)} records")
                print(f"   💡 Data should appear in Hopsworks UI shortly")
                
            except Exception as insert_error:
                # Check if it's a schema mismatch error
                error_msg = str(insert_error)
                
                if "Features are not compatible" in error_msg:
                    print(f"🔧 Schema compatibility issue detected...")
                    
                    # Check if it's a column mismatch (missing/extra columns) vs type mismatch
                    if "is missing from input dataframe" in error_msg or "does not exist in feature group" in error_msg:
                        print(f"   ❌ Column schema mismatch - feature group has different columns than current data")
                        print(f"   💡 This happens when feature engineering changes between pipeline runs")
                        print(f"   🔄 The feature group schema was already updated during this run")
                        print(f"   ⚠️ If you're still seeing this error, there may be a race condition")
                        print(f"   🔧 Try running the pipeline again, or check Hopsworks UI for schema updates")
                        
                        # Show details about the mismatch
                        lines = error_msg.split('\n')
                        for line in lines:
                            if "is missing from input dataframe" in line or "does not exist in feature group" in line:
                                print(f"       {line.strip()}")
                        
                        log_error("Schema column mismatch - feature group may need manual cleanup", insert_error)
                        return False
                        
                    elif "has the wrong type" in error_msg:
                        print(f"   🔧 Data type mismatch detected, attempting to fix...")
                        
                        # Try to automatically fix type issues
                        try:
                            # Parse the error message to find problematic columns and their expected types
                            problem_columns = {}  # col_name -> expected_type
                            
                            if "has the wrong type" in error_msg:
                                lines = error_msg.split('\n')
                                for line in lines:
                                    if "has the wrong type" in line and "expected type:" in line:
                                        # Extract column name (before the first parenthesis, remove "- " prefix)
                                        col_part = line.split('(')[0].strip()
                                        if col_part.startswith('- '):
                                            col_name = col_part[2:].strip()  # Remove "- " prefix
                                        else:
                                            col_name = col_part.strip()
                                    
                                    # Extract expected type
                                    if "expected type: 'bigint'" in line:
                                        expected_type = 'bigint'
                                    elif "expected type: 'double'" in line:
                                        expected_type = 'double'
                                    else:
                                        expected_type = 'unknown'
                                    
                                        if col_name and col_name in df_copy.columns:
                                            problem_columns[col_name] = expected_type
                            
                            # Convert problematic columns to the expected types
                            for col, expected_type in problem_columns.items():
                                if expected_type == 'bigint':
                                    print(f"   🔧 Converting {col} to integer (bigint)...")
                                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0.0).astype('int64')
                                elif expected_type == 'double':
                                    print(f"   🔧 Converting {col} to float (double)...")
                                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0.0).astype('float64')
                                else:
                                    print(f"   ⚠️ Unknown expected type for {col}: {expected_type}")
                            
                            if problem_columns:
                                print(f"   🔄 Retrying data insertion with fixed types...")
                                fg.insert(df_copy)
                                print(f"✅ Data inserted successfully after type fixes: {len(df_copy)} records")
                                return True
                            else:
                                print(f"   ⚠️ Could not automatically identify problematic columns from error message:")
                                print(f"       {error_msg[:200]}...")  # Show first 200 chars of error
                                log_error("Data insertion failed", insert_error)
                                return False
                                
                        except Exception as fix_error:
                            print(f"   ⚠️ Auto-fix attempt failed: {fix_error}")
                            log_error("Data insertion failed", insert_error)
                            return False
                else:
                    log_error("Data insertion failed", insert_error)
                    return False
            
            return True
            
        except Exception as e:
            log_error("Feature store save failed", e)
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
            
            # Get feature group with error handling
            try:
                fg = self.fs.get_feature_group(name=fg_name, version=version)
                if fg is None:
                    print(f"⚠️ Feature group {fg_name} not found")
                    return None
            except Exception as fg_error:
                print(f"⚠️ Failed to get feature group {fg_name}: {fg_error}")
                return None
            
            # Try multiple read methods with robust error handling
            df = None
            read_methods = [
                # Method 1: Direct read (fastest)
                lambda: fg.read(),
                # Method 2: Select all then read
                lambda: fg.select_all().read(),
                # Method 3: Read with limit (if data is too large)
                lambda: fg.select_all().limit(50000).read(),
                # Method 4: Read without query service (offline only)
                lambda: fg.read(online=False) if hasattr(fg, 'read') else None
            ]
            
            for i, method in enumerate(read_methods, 1):
                try:
                    print(f"🔄 Trying read method {i}...")
                    df = method()
                    if df is not None and len(df) > 0:
                        print(f"✅ Read successful with method {i}")
                        break
                except Exception as read_error:
                    print(f"⚠️ Read method {i} failed: {read_error}")
                    continue
            
            if df is None or len(df) == 0:
                print(f"❌ All read methods failed or returned empty data")
                return None
            
            # Remove the artificial 'id' column if it exists
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
            
            print(f"✅ Data loaded from feature store: {len(df)} records")
            print(f"📅 Date range: {df['datetime'].min() if 'datetime' in df.columns else 'N/A'} to {df['datetime'].max() if 'datetime' in df.columns else 'N/A'}")
            
            return df
            
        except Exception as e:
            print(f"❌ Feature store load failed: {e}")
            print("   Data might not be ready yet or Query Service unavailable")
            print("   Try running the data fetching pipeline first")
            return None
    
    def save_model(self, model_dir: str, model_name: str, model_type: str = "sklearn") -> bool:
        """
        Save model to Hopsworks model registry
        
        Parameters:
        model_dir (str): Local directory containing model files
        model_name (str): Name for the model in registry
        model_type (str): Type of model ('sklearn', 'tensorflow', 'pytorch')
        
        Returns:
        bool: Success status
        """
        if not self.enabled:
            print("⚠️ Hopsworks not enabled, skipping model registry save")
            return False
        
        try:
            print(f"🤖 Saving model to registry: {model_name}")
            print(f"📁 Model directory: {model_dir}")
            
            # Verify the directory exists and list contents
            if os.path.isdir(model_dir):
                files_in_dir = os.listdir(model_dir)
                print(f"📂 Files in model directory: {files_in_dir}")
            else:
                print(f"❌ Model directory does not exist: {model_dir}")
                return False
            
            # Prepare metrics dictionary with only numeric values
            metrics = {}
            
            # Look for appropriate metadata file based on model type
            if model_type == "sklearn":
                metadata_file = os.path.join(model_dir, "sklearn_model_metadata.json")
            elif model_type in ["tensorflow", "dl", "deep_learning"]:
                metadata_file = os.path.join(model_dir, "dl_model_metadata.json")
            else:
                metadata_file = os.path.join(model_dir, f"{model_type}_model_metadata.json")
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r') as f:
                    model_metadata = json.load(f)
                    print(f"📊 Loading metrics from: {metadata_file}")
                    # Only add numeric metrics, skip framework string
                    for key, value in model_metadata.items():
                        if isinstance(value, (int, float)) and key in ["test_r2", "test_rmse", "test_mae", "train_r2", "cv_score"]:
                            metrics[key] = float(value)
                            print(f"   {key}: {value}")
            else:
                print(f"⚠️ Metadata file not found: {metadata_file}")
            
            # Add default metrics if none found
            if not metrics:
                metrics = {"test_rmse": 0.0, "test_r2": 0.0}
                print("⚠️ Using default metrics")
            
            # Create new model (always create new version to avoid conflicts)
            print(f"🆕 Creating new model: {model_name}")
            
            # Use python client interface for model creation
            try:
                model = self.mr.python.create_model(
                    name=model_name,
                    description=f"AQI prediction model - {model_type}",
                    metrics=metrics
                )
                
                if model is None:
                    raise Exception("Model creation returned None")
                    
                print(f"✅ Created model: {model_name} v{model.version}")
                
            except Exception as create_error:
                print(f"⚠️ Failed with python client, trying standard client: {create_error}")
                # Fallback to standard client
                model = self.mr.create_model(
                    name=model_name,
                    description=f"AQI prediction model - {model_type}",
                    metrics=metrics
                )
                
                if model is None:
                    raise Exception("Standard model creation also returned None")
            
            # Verify model object before saving
            if not hasattr(model, 'save'):
                raise Exception(f"Model object does not have 'save' method. Type: {type(model)}")
            
            # Save model files to registry
            print(f"📤 Uploading model files from: {model_dir}")
            
            if os.path.isdir(model_dir):
                model.save(model_dir)
            else:
                # If single file, create temp directory
                import tempfile
                import shutil
                with tempfile.TemporaryDirectory() as temp_dir:
                    shutil.copy2(model_dir, temp_dir)
                    model.save(temp_dir)
            
            print(f"✅ Model saved to registry: {model.name} v{model.version}")
            return True
            
        except Exception as e:
            log_error("Model registry save failed", e)
            return False
    
    def clean_test_data(self):
        """Clean test data from Hopsworks feature store"""
        if not self.enabled:
            print("⚠️ Hopsworks not enabled")
            return False
        
        try:
            print("🧹 Cleaning test data from Hopsworks...")
            
            # List of test feature groups to clean
            test_stages = ["test", "raw"]  # Clean both test and raw stages
            
            for stage in test_stages:
                fg_name = f"aqi_{stage}_features"
                try:
                    fg = self.fs.get_feature_group(name=fg_name, version=1)
                    if fg is not None:
                        print(f"🗑️ Deleting feature group: {fg_name}")
                        fg.delete()
                        print(f"✅ Deleted {fg_name}")
                except Exception as e:
                    print(f"⚠️ Could not delete {fg_name}: {e}")
            
            # Clean test models from model registry
            try:
                test_models = ["test_aqi_model", "test_aqi_pipeline_model"]
                for model_name in test_models:
                    try:
                        models = self.mr.get_models(name=model_name)
                        for model in models:
                            print(f"🗑️ Deleting model: {model_name} v{model.version}")
                            model.delete()
                            print(f"✅ Deleted {model_name} v{model.version}")
                    except Exception as e:
                        print(f"⚠️ Could not delete model {model_name}: {e}")
            except Exception as e:
                print(f"⚠️ Model cleanup error: {e}")
            
            print("✅ Test data cleanup completed")
            return True
            
        except Exception as e:
            log_error("Test data cleanup failed", e)
            return False
    
    def clean_all_feature_groups(self, confirm: bool = False):
        """
        Clean all AQI feature groups for a fresh start
        
        Parameters:
        confirm (bool): Must be True to actually delete data
        """
        if not self.enabled:
            print("⚠️ Hopsworks not enabled")
            return False
        
        if not confirm:
            print("⚠️ This will delete ALL AQI feature group data from Hopsworks!")
            print("   To confirm, call clean_all_feature_groups(confirm=True)")
            return False
        
        try:
            print("🧹 Cleaning ALL AQI feature groups from Hopsworks...")
            print("   ⚠️ This will delete all historical data!")
            
            # List of all AQI feature group stages
            stages = ["raw", "engineered", "selected"]
            deleted_count = 0
            
            for stage in stages:
                fg_name = f"aqi_{stage}_features"
                try:
                    # Try multiple versions
                    for version in range(1, 5):  # Check versions 1-4
                        try:
                            fg = self.fs.get_feature_group(name=fg_name, version=version)
                            if fg is not None:
                                print(f"🗑️ Deleting feature group: {fg_name} v{version}")
                                fg.delete()
                                print(f"✅ Deleted {fg_name} v{version}")
                                deleted_count += 1
                        except Exception:
                            # Version doesn't exist, continue
                            pass
                            
                except Exception as e:
                    print(f"⚠️ Error checking {fg_name}: {e}")
            
            print(f"✅ Feature group cleanup completed - deleted {deleted_count} feature groups")
            return True
            
        except Exception as e:
            log_error("Feature group cleanup failed", e)
            return False

def run_hourly_data_pipeline():
    """
    Hourly pipeline: Fetch incremental data, handle duplicates, and update feature store
    """
    import pandas as pd
    
    print("🔄 HOURLY DATA PIPELINE")
    print("="*50)
    
    try:
        fetcher = OpenMeteoDataFetcher()
        
        # Check for existing data
        latest_timestamp = fetcher.get_latest_data_timestamp()
        
        if latest_timestamp is None:
            print("📭 No existing data found, setting up initial historical data...")
            success = fetcher.setup_initial_historical_data(years_back=1)
            if success:
                print("✅ Initial historical data setup complete")
                return True
            else:
                print("❌ Failed to set up initial data")
                return False
        
        # Check if we need to fetch incremental data
        now = datetime.now()
        
        # Handle timezone differences between now and latest_timestamp
        if hasattr(latest_timestamp, 'tz') and latest_timestamp.tz is not None:
            # latest_timestamp is timezone-aware, make now timezone-naive for comparison
            if hasattr(latest_timestamp, 'tz_localize'):
                latest_timestamp = latest_timestamp.tz_localize(None)
            else:
                # Convert pandas timestamp to naive datetime
                latest_timestamp = latest_timestamp.replace(tzinfo=None)
        
        hours_since_last = (now - latest_timestamp).total_seconds() / 3600
        
        if hours_since_last < 0.5:  # Changed from 1 hour to 30 minutes
            print(f"⏰ Data is very recent ({hours_since_last:.1f} hours old), no update needed")
            return True
        
        print(f"📊 Last data: {latest_timestamp} ({hours_since_last:.1f} hours ago)")
        print(f"💡 Attempting to fetch newer data...")
        
        # Fetch incremental data
        new_data = fetcher.fetch_incremental_data(latest_timestamp)
        
        if new_data is None or len(new_data) == 0:
            # Try a different approach - fetch the latest available data directly
            print("📭 No incremental data available, trying to fetch latest available data...")
            latest_data = fetcher.fetch_latest_data(hours_back=24)  # Last 24 hours
            
            if latest_data is not None and len(latest_data) > 0:
                # Filter to get only data newer than our latest timestamp
                latest_data['datetime'] = pd.to_datetime(latest_data['datetime'])
                newer_data = latest_data[latest_data['datetime'] > latest_timestamp]
                
                if len(newer_data) > 0:
                    print(f"📈 Found {len(newer_data)} newer records via latest data fetch")
                    new_data = newer_data
                else:
                    print("📭 No newer data found in latest fetch")
                    return True
            else:
                print("📭 No data available from latest fetch either")
                return True
        else:
            print(f"📈 Fetched {len(new_data)} incremental records")
        
        # IMPORTANT: Handle duplicates before saving
        print("🔍 Checking for duplicates...")
        existing_data = load_latest_data(fetcher)
        
        if existing_data is not None and len(existing_data) > 0:
            # Merge and remove duplicates based on datetime
            print("🔧 Merging with existing data and removing duplicates...")
            
            # Debug: Check column structure before merging
            print(f"🔍 Existing data columns: {len(existing_data.columns)} columns")
            print(f"🔍 New data columns: {len(new_data.columns)} columns")
            
            # Ensure both datasets have the same column structure (raw data format)
            # Get the expected raw data columns from the new_data (which is fresh from OpenMeteo)
            expected_columns = set(new_data.columns)
            existing_columns = set(existing_data.columns)
            
            # Check for column mismatch
            if expected_columns != existing_columns:
                print("⚠️ Column mismatch detected between existing and new data")
                print(f"   Expected (new): {sorted(expected_columns)}")
                print(f"   Existing: {sorted(existing_columns)}")
                
                # If existing data has many more columns, it might be processed data
                if len(existing_columns) > len(expected_columns) * 2:
                    print("🔧 Existing data appears to be processed/engineered data")
                    print("   Loading only raw data for comparison...")
                    
                    # Try to get only raw data from Hopsworks
                    raw_only_data = None
                    if fetcher.hops_integration and fetcher.hops_integration.enabled:
                        try:
                            raw_only_data = fetcher.hops_integration.load_from_feature_store(stage="raw")
                            if raw_only_data is not None:
                                # Keep only columns that match new_data structure
                                common_columns = list(expected_columns.intersection(set(raw_only_data.columns)))
                                if 'datetime' in common_columns and len(common_columns) > 5:  # Ensure we have basic columns
                                    existing_data = raw_only_data[common_columns]
                                    print(f"✅ Using raw data with {len(common_columns)} matching columns")
                                else:
                                    print("⚠️ Insufficient matching columns, using new data only")
                                    existing_data = None
                        except Exception as e:
                            print(f"⚠️ Could not load raw-only data: {e}")
                            existing_data = None
                
                # If we still have column mismatch, align the columns
                if existing_data is not None:
                    # Keep only common columns
                    common_columns = list(expected_columns.intersection(set(existing_data.columns)))
                    if 'datetime' in common_columns and len(common_columns) > 5:
                        existing_data = existing_data[common_columns]
                        new_data = new_data[common_columns]
                        print(f"✅ Aligned data to {len(common_columns)} common columns")
                    else:
                        print("❌ Too few common columns, skipping merge")
                        existing_data = None
            
            if existing_data is not None:
                # Ensure both datasets have consistent timezone handling
                
                # Normalize timezone for existing_data
                if 'datetime' in existing_data.columns:
                    existing_data['datetime'] = pd.to_datetime(existing_data['datetime'])
                    if len(existing_data) > 0 and hasattr(existing_data['datetime'].iloc[0], 'tz') and existing_data['datetime'].iloc[0].tz is not None:
                        existing_data['datetime'] = existing_data['datetime'].dt.tz_localize(None)
                
                # Normalize timezone for new_data
                if 'datetime' in new_data.columns:
                    new_data['datetime'] = pd.to_datetime(new_data['datetime'])
                    if len(new_data) > 0 and hasattr(new_data['datetime'].iloc[0], 'tz') and new_data['datetime'].iloc[0].tz is not None:
                        new_data['datetime'] = new_data['datetime'].dt.tz_localize(None)
                
                # Extra safety: ensure column order and data types match
                common_columns = [col for col in new_data.columns if col in existing_data.columns]
                existing_data = existing_data[common_columns].copy()
                new_data = new_data[common_columns].copy()
                
                # Align data types
                for col in common_columns:
                    if col != 'datetime':
                        try:
                            # Convert both to same dtype (prefer float64 for numeric)
                            if existing_data[col].dtype != new_data[col].dtype:
                                target_dtype = 'float64' if pd.api.types.is_numeric_dtype(existing_data[col]) or pd.api.types.is_numeric_dtype(new_data[col]) else 'object'
                                existing_data[col] = existing_data[col].astype(target_dtype)
                                new_data[col] = new_data[col].astype(target_dtype)
                        except Exception as dtype_error:
                            print(f"⚠️ Could not align dtype for column {col}: {dtype_error}")
                
                print(f"✅ Data alignment complete - both datasets have {len(common_columns)} columns")
                
                # Convert to dict and back to DataFrame to ensure clean structure
                # This fixes internal array dimension mismatches
                print("🔧 Rebuilding DataFrames for compatibility...")
                existing_dict = existing_data.to_dict('records')
                new_dict = new_data.to_dict('records')
                
                # Create fresh DataFrames
                existing_clean = pd.DataFrame(existing_dict)
                new_clean = pd.DataFrame(new_dict)
                
                # Ensure column order is consistent
                column_order = sorted(common_columns)
                existing_clean = existing_clean[column_order]
                new_clean = new_clean[column_order]
                
                # Now try the concat
                combined_data = pd.concat([existing_clean, new_clean], ignore_index=True)
                
                # Remove duplicates based on datetime (keep the last occurrence)
                before_count = len(combined_data)
                combined_data = combined_data.drop_duplicates(subset=['datetime'], keep='last')
                after_count = len(combined_data)
                
                # Sort by datetime (now both are timezone-naive)
                combined_data = combined_data.sort_values('datetime')
            else:
                # No existing data to merge with, just use new data
                print("🔧 Using new data only (no existing data to merge)")
                combined_data = new_data
                before_count = len(combined_data)
                after_count = len(combined_data)
            
            print(f"📊 Before deduplication: {before_count} records")
            print(f"📊 After deduplication: {after_count} records")
            print(f"📊 Duplicates removed: {before_count - after_count}")
            
            # Only save if there are actually new records
            # Calculate new records based on what we added vs deduplicated result
            original_existing_count = len(existing_data) if existing_data is not None else 0
            new_data_count = len(new_data) if new_data is not None else 0
            
            print(f"🔢 Original existing data: {original_existing_count} records")
            print(f"🔢 New data fetched: {new_data_count} records") 
            print(f"🔢 Combined after deduplication: {after_count} records")
            
            # If we have new data and the combined dataset is larger than just existing data
            if new_data_count > 0 and after_count > original_existing_count:
                net_new_records = after_count - original_existing_count
                print(f"✅ Found {net_new_records} net new records to save")
                
                # Save the complete updated dataset (Hopsworks will replace, not append)
                print("💾 Saving updated dataset...")
                
                # Save locally first (always works)
                local_path = save_with_timestamp(combined_data, "complete_raw_data.csv")
                print(f"💾 Saved updated data locally: {local_path}")
                
                # For Hopsworks, save the complete updated dataset
                # Note: Hopsworks feature groups with primary keys handle duplicates automatically
                hopsworks_success = safe_save(fetcher, combined_data, "raw")
                
                if hopsworks_success:
                    print("✅ Updated data saved to Hopsworks (duplicates handled automatically)")
                    
            elif new_data_count > 0:
                print("⚠️ New data was fetched but appears to be duplicates")
                print("💾 Saving anyway to ensure data consistency...")
                
                # Save locally and to Hopsworks anyway
                local_path = save_with_timestamp(combined_data, "complete_raw_data.csv")
                hopsworks_success = safe_save(fetcher, combined_data, "raw")
                
            else:
                print("ℹ️ No new data to save")
                return True
        else:
            # No existing data, just save the new data
            print("📊 No existing data to merge with, saving new data directly")
            
            # Save locally first (always works)
            local_path = save_with_timestamp(new_data, "incremental_data.csv")
            print(f"💾 Saved incremental data locally: {local_path}")
            
            # Save to Hopsworks using helper function
            hopsworks_success = safe_save(fetcher, new_data, "raw")
        
        print("✅ Hourly pipeline completed successfully")
        print("💡 Note: Hopsworks Feature Groups with primary keys automatically handle duplicates")
        print("          Local files are timestamped to preserve history")
        
        return True
        
    except Exception as e:
        log_error("Hourly pipeline failed", e)
        return False

def run_daily_model_pipeline():
    """
    Daily pipeline: Load latest data, perform COMPLETE feature engineering, and train ALL models
    This mirrors the full pipeline training process exactly
    """
    print("🤖 DAILY MODEL TRAINING PIPELINE")
    print("="*50)
    
    try:
        # Load latest data using helper function
        fetcher = OpenMeteoDataFetcher()
        raw_data = load_latest_data(fetcher)
        
        if raw_data is None or len(raw_data) == 0:
            print("❌ No data available for training")
            print("   Please run the data fetching pipeline first:")
            print("   python scripts/fetch_hourly_data.py")
            return False
        
        print(f"📊 Using {len(raw_data)} records for training")
        print(f"📅 Date range: {raw_data['datetime'].min()} to {raw_data['datetime'].max()}")
        
        # STEP 1: COMPREHENSIVE FEATURE ENGINEERING (same as full pipeline)
        print("\n🔧 STEP 1: COMPREHENSIVE FEATURE ENGINEERING")
        print("="*50)
        from features.feature_engineering import FeatureEngineer, AdvancedFeatureSelector
        
        engineer = FeatureEngineer()
        print("🔧 Engineering features...")
        engineered_data = engineer.engineer_features(raw_data)
        print("🔧 Handling missing values...")
        engineered_data = engineer.handle_missing_values(engineered_data)
        
        print(f"✅ Feature engineering completed!")
        print(f"📊 Features shape: {engineered_data.shape}")
        
        # Save engineered features using helper function
        safe_save(fetcher, engineered_data, "engineered")
        
        # STEP 2: ADVANCED FEATURE SELECTION (same as full pipeline)
        print("\n🎯 STEP 2: ADVANCED FEATURE SELECTION")
        print("="*50)
        selector = AdvancedFeatureSelector()
        selected_data, selected_features = selector.select_features(engineered_data, max_features=25)
        
        print(f"✅ Feature selection completed!")
        print(f"📊 Selected features: {len(selected_features)}")
        
        # Save selected features using helper function
        safe_save(fetcher, selected_data, "selected")
        
        # Save feature names locally as well
        try:
            import os
            os.makedirs("data", exist_ok=True)
            feature_names_path = os.path.join("data", "selected_feature_names.txt")
            with open(feature_names_path, 'w') as f:
                f.write("\n".join(selected_features))
            print(f"📝 Feature names saved: {feature_names_path}")
        except Exception as e:
            print(f"⚠️ Could not save feature names: {e}")
        
        # STEP 3A: TRAIN SKLEARN MODELS (enhanced training)
        print("\n🚀 STEP 3A: TRAINING SKLEARN MODELS")
        print("="*50)
        from models.train_sklearn import SklearnModelTrainer
        
        sklearn_trainer = SklearnModelTrainer()
        sklearn_results = sklearn_trainer.train_all_models(selected_data, test_size=0.2)
        
        # Save best sklearn model
        sklearn_model_path = None
        if sklearn_results:
            sklearn_model_path = sklearn_trainer.save_best_model()
            print(f"💾 Best sklearn model saved: {sklearn_model_path}")
            print(f"🏆 Best sklearn model: {sklearn_trainer.best_model_name}")
            if sklearn_trainer.best_model_name in sklearn_trainer.results:
                best_r2 = sklearn_trainer.results[sklearn_trainer.best_model_name]['test_r2']
                print(f"📊 Best sklearn R²: {best_r2:.4f}")
        
        # STEP 3B: TRAIN DEEP LEARNING MODELS (NEW - now included!)
        print("\n🧠 STEP 3B: TRAINING DEEP LEARNING MODELS")
        print("="*50)
        from models.train_dl import DeepLearningModelTrainer
        
        dl_trainer = DeepLearningModelTrainer()
        dl_results = dl_trainer.train_all_models(selected_data, test_size=0.2, epochs=100)
        
        # Save best deep learning model
        dl_model_path = None
        if dl_results:
            dl_model_path = dl_trainer.save_best_model()
            print(f"💾 Best deep learning model saved: {dl_model_path}")
            print(f"🏆 Best deep learning model: {dl_trainer.best_model_name}")
            if dl_trainer.best_model_name in dl_trainer.results:
                best_r2 = dl_trainer.results[dl_trainer.best_model_name]['test_r2']
                print(f"📊 Best deep learning R²: {best_r2:.4f}")
        
        # FINAL SUMMARY
        print("\n🎉 DAILY MODEL PIPELINE COMPLETED!")
        print("="*50)
        print(f"📊 Total features engineered: {engineered_data.shape[1]}")
        print(f"🎯 Selected features: {len(selected_features)}")
        print(f"📈 Training data shape: {selected_data.shape}")
        
        if sklearn_results:
            print(f"🏆 Best sklearn model: {sklearn_trainer.best_model_name}")
        if dl_results:
            print(f"🏆 Best deep learning model: {dl_trainer.best_model_name}")
        
        # Upload models to Hopsworks if enabled (after BOTH models are trained)
        if fetcher.hops_integration and fetcher.hops_integration.enabled:
            print("\n☁️ Uploading models to Hopsworks Model Registry...")
            try:
                # Use the models directory with absolute path (both sklearn and dl models are saved there)
                models_dir = os.path.abspath("models")
                
                # Upload sklearn model if trained successfully
                if sklearn_results and sklearn_model_path and os.path.exists(sklearn_model_path):
                    print(f"📤 Uploading sklearn model from: {models_dir}")
                    sklearn_success = fetcher.hops_integration.save_model(
                        models_dir, "aqi_sklearn_model", "sklearn"
                    )
                    if sklearn_success:
                        print("✅ Sklearn model uploaded to Hopsworks")
                    else:
                        print("❌ Sklearn model upload failed")
                
                # Upload deep learning model if trained successfully
                if dl_results and dl_model_path and os.path.exists(dl_model_path):
                    print(f"📤 Uploading deep learning model from: {models_dir}")
                    dl_success = fetcher.hops_integration.save_model(
                        models_dir, "aqi_dl_model", "tensorflow"
                    )
                    if dl_success:
                        print("✅ Deep learning model uploaded to Hopsworks")
                    else:
                        print("❌ Deep learning model upload failed")
                else:
                    print("[] Couldnt upload deep learning model to hopswork")
                        
            except Exception as upload_error:
                print(f"⚠️ Model upload failed: {upload_error}")
        else:
            print("[] Fetcher hopsworks integration error\n")
        
        print("✅ Daily model pipeline completed successfully!")
        return True
        
    except Exception as e:
        log_error("Daily pipeline failed", e)
        return False

def main():
    """Main function for testing pipelines"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "hourly":
            run_hourly_data_pipeline()
        elif sys.argv[1] == "daily":
            run_daily_model_pipeline()
        else:
            print("Usage: python fetch_data.py [hourly|daily]")
    else:
        print("🧪 Running test mode...")
        print("Use 'python fetch_data.py hourly' for hourly data pipeline")
        print("Use 'python fetch_data.py daily' for daily model pipeline")

if __name__ == "__main__":
    main()
