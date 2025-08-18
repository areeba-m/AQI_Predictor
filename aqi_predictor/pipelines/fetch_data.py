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
            
            print(f"📦 Chunk {chunk_count}: {current_date.date()} to {chunk_end.date()}")
            
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
        
        # Fetch recent data if requested
        if include_recent:
            print(f"\n🔄 Fetching latest data (last 2 days)")
            try:
                recent_data = self.fetch_recent_data(days_back=2, days_forward=0)
                if recent_data is not None and len(recent_data) > 0:
                    all_data.append(recent_data)
                    print(f"✅ Recent data: {len(recent_data)} records")
            except Exception as e:
                print(f"⚠️ Recent data fetch failed: {e}")
        
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
                            latest_hops = pd.to_datetime(existing_data['datetime']).max()
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
                local_data['datetime'] = pd.to_datetime(local_data['datetime'])
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
        now = datetime.now()
        start_date = since_timestamp + timedelta(hours=1)  # Start from next hour
        
        # For archive API, use up to 3 days ago to ensure data availability
        end_date = min(now - timedelta(days=3), start_date + timedelta(days=max_days))
        
        if start_date >= end_date:
            print(f"⏰ No new data to fetch (start: {start_date}, end: {end_date})")
            return None
        
        print(f"📈 Fetching incremental data from {start_date} to {end_date}")
        
        return self.fetch_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

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
            local_path = save_data_locally(historical_data, "historical_raw_data.csv")
            print(f"✅ Saved {len(historical_data)} historical records locally: {local_path}")
            
            # Save to Hopsworks (might fail, but that's ok)
            if self.hops_integration and self.hops_integration.enabled:
                print(f"☁️ Saving data to Hopsworks feature store...")
                success = self.hops_integration.save_to_feature_store(historical_data, stage="raw")
                if success:
                    print(f"✅ Saved {len(historical_data)} historical records to Hopsworks")
                else:
                    print("⚠️ Failed to save historical data to Hopsworks, but local save succeeded")
                    print("   Pipeline can continue using local files")
            else:
                print("⚠️ Hopsworks not enabled, using local files only")
            
            print(f"✅ Initial historical data setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"❌ Hopsworks connection failed: {e}")
            print(f"   Error type: {type(e)}")
            print("   Please check your API key and project name in .env file")
            import traceback
            traceback.print_exc()
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
            
            # Ensure all numeric columns are float64
            for col in df_copy.columns:
                if col != 'datetime':
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0.0)
            
            # Add integer primary key (required for feature groups)
            df_copy['id'] = range(len(df_copy))
            
            # Ensure datetime column is properly formatted
            if 'datetime' in df_copy.columns:
                df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
            
            print(f"   ✅ Shape: {df_copy.shape}")
            print(f"   ✅ Null values: {df_copy.isnull().sum().sum()}")
            
            # Configure feature group name based on stage
            fg_name = f"aqi_{stage}_features"
            fg_version = 1
            
            print(f"💾 Saving to feature store: {fg_name} v{fg_version}")
            
            # Get or create feature group with integer primary key
            fg = None
            
            # Always try to create feature group if it doesn't exist
            print(f"🔍 Checking for existing feature group: {fg_name}")
            try:
                fg = self.fs.get_feature_group(name=fg_name, version=fg_version)
                if fg is not None:
                    print(f"📋 Using existing feature group: {fg_name}")
                else:
                    print(f"📭 Feature group {fg_name} not found, will create new one")
                    raise Exception("Feature group not found, creating new")
            except Exception as get_error:
                print(f"🆕 Creating new feature group: {fg_name}")
                print(f"   Reason: {get_error}")
                
                # Create new feature group with robust configuration
                creation_attempts = [
                    # Attempt 1: With event_time
                    {
                        "name": fg_name,
                        "version": fg_version,
                        "description": f"AQI {stage} features with weather and air quality data for Lahore, Pakistan",
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
                print(f"❌ Data insertion failed: {insert_error}")
                print(f"   Error type: {type(insert_error)}")
                import traceback
                traceback.print_exc()
                return False
                return False
            
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
            
            # Prepare metrics dictionary with only numeric values
            metrics = {}
            metadata_file = os.path.join(model_dir, "sklearn_model_metadata.json")
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r') as f:
                    model_metadata = json.load(f)
                    # Only add numeric metrics, skip framework string
                    for key, value in model_metadata.items():
                        if isinstance(value, (int, float)) and key in ["test_r2", "test_rmse", "test_mae", "train_r2", "cv_score"]:
                            metrics[key] = float(value)
            
            # Add default metrics if none found
            if not metrics:
                metrics = {"test_rmse": 0.0, "test_r2": 0.0}
            
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
            print(f"❌ Model registry save failed: {e}")
            print(f"   Error details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
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
            print(f"❌ Cleanup failed: {e}")
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

def run_hourly_data_pipeline():
    """
    Hourly pipeline: Fetch incremental data and update feature store
    """
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
        hours_since_last = (now - latest_timestamp).total_seconds() / 3600
        
        if hours_since_last < 1:
            print(f"⏰ Data is recent ({hours_since_last:.1f} hours old), no update needed")
            return True
        
        print(f"📊 Last data: {latest_timestamp} ({hours_since_last:.1f} hours ago)")
        
        # Fetch incremental data
        new_data = fetcher.fetch_incremental_data(latest_timestamp)
        
        if new_data is None or len(new_data) == 0:
            print("📭 No new data available")
            return True
        
        print(f"📈 Fetched {len(new_data)} new records")
        
        # Save locally (append to existing) - this always works
        local_path = save_data_locally(new_data, "incremental_data.csv")
        print(f"💾 Saved incremental data locally: {local_path}")
        
        # Save to Hopsworks - optional, pipeline continues if this fails
        hopsworks_success = False
        if fetcher.hops_integration and fetcher.hops_integration.enabled:
            try:
                success = fetcher.hops_integration.save_to_feature_store(new_data, stage="raw")
                if success:
                    print(f"☁️ Appended {len(new_data)} records to Hopsworks feature store")
                    hopsworks_success = True
                else:
                    print("⚠️ Failed to save incremental data to Hopsworks")
            except Exception as e:
                print(f"⚠️ Hopsworks save failed: {e}")
        
        if hopsworks_success:
            print("✅ Hourly pipeline completed successfully (Hopsworks + Local)")
        else:
            print("✅ Hourly pipeline completed successfully (Local only)")
            print("   Data saved locally, can be used for training")
        
        return True
        
    except Exception as e:
        print(f"❌ Hourly pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_daily_model_pipeline():
    """
    Daily pipeline: Load latest data, perform feature engineering, and train models
    """
    print("🤖 DAILY MODEL TRAINING PIPELINE")
    print("="*50)
    
    try:
        # Load latest data from Hopsworks or local files
        fetcher = OpenMeteoDataFetcher()
        raw_data = None
        
        # Try loading from Hopsworks first
        if fetcher.hops_integration and fetcher.hops_integration.enabled:
            print("📥 Loading data from Hopsworks feature store...")
            raw_data = fetcher.hops_integration.load_from_feature_store(stage="raw")
        
        # If Hopsworks fails, try local files as fallback
        if raw_data is None or len(raw_data) == 0:
            print("⚠️ No data in Hopsworks, trying local files...")
            import glob
            import pandas as pd
            
            # Look for local data files
            csv_files = glob.glob("data/historical_*.csv") + glob.glob("data/complete_dataset_*.csv")
            if csv_files:
                latest_file = max(csv_files, key=os.path.getctime)
                print(f"📁 Loading from local file: {latest_file}")
                raw_data = pd.read_csv(latest_file)
                raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
                print(f"✅ Loaded {len(raw_data)} records from local file")
            else:
                print("❌ No local data files found either")
                print("   Please run the data fetching pipeline first:")
                print("   python scripts/fetch_hourly_data.py")
                return False
        
        if raw_data is None or len(raw_data) == 0:
            print("❌ No data available for training")
            return False
        
        print(f"📊 Using {len(raw_data)} records for training")
        print(f"📅 Date range: {raw_data['datetime'].min()} to {raw_data['datetime'].max()}")
        
        # Feature engineering
        print("🔧 Performing feature engineering...")
        from features.feature_engineering import FeatureEngineer, AdvancedFeatureSelector
        
        engineer = FeatureEngineer()
        engineered_data = engineer.engineer_features(raw_data)
        engineered_data = engineer.handle_missing_values(engineered_data)
        
        # Save engineered features to Hopsworks (if available)
        if fetcher.hops_integration and fetcher.hops_integration.enabled:
            success = fetcher.hops_integration.save_to_feature_store(engineered_data, stage="engineered")
            if success:
                print(f"☁️ Saved engineered features to Hopsworks")
            else:
                print(f"⚠️ Failed to save engineered features to Hopsworks")
        
        # Feature selection
        print("🎯 Performing feature selection...")
        selector = AdvancedFeatureSelector()
        selected_data, selected_features = selector.select_features(engineered_data, max_features=20)
        
        # Save selected features to Hopsworks (if available)
        if fetcher.hops_integration and fetcher.hops_integration.enabled:
            success = fetcher.hops_integration.save_to_feature_store(selected_data, stage="selected")
            if success:
                print(f"☁️ Saved selected features to Hopsworks")
            else:
                print(f"⚠️ Failed to save selected features to Hopsworks")
        
        print(f"🎯 Selected {len(selected_features)} features for training")
        
        # Train models
        print("🚀 Training models...")
        from models.train_sklearn import SklearnModelTrainer
        
        trainer = SklearnModelTrainer()
        results = trainer.train_all_models(selected_data, test_size=0.2)
        
        # Save best model (automatically uploads to Hopsworks if available)
        model_path = trainer.save_best_model()
        
        print(f"✅ Daily model pipeline completed successfully")
        print(f"🏆 Best model: {trainer.best_model_name}")
        print(f"📁 Model saved: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Daily pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
        # Save engineered features to Hopsworks
        success = fetcher.hops_integration.save_to_feature_store(engineered_data, stage="engineered")
        if success:
            print(f"☁️ Saved engineered features to Hopsworks")
        
        # Feature selection
        selector = AdvancedFeatureSelector()
        selected_data, selected_features = selector.select_features(engineered_data, max_features=20)
        
        # Save selected features to Hopsworks
        success = fetcher.hops_integration.save_to_feature_store(selected_data, stage="selected")
        if success:
            print(f"☁️ Saved selected features to Hopsworks")
        
        print(f"🎯 Selected {len(selected_features)} features for training")
        
        # Train models
        print("🚀 Training models...")
        from models.train_sklearn import SklearnModelTrainer
        
        trainer = SklearnModelTrainer()
        results = trainer.train_all_models(selected_data, test_size=0.2)
        
        # Save best model (automatically uploads to Hopsworks)
        model_path = trainer.save_best_model()
        
        print(f"✅ Daily model pipeline completed successfully")
        print(f"🏆 Best model: {trainer.best_model_name}")
        print(f"📁 Model saved: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Daily pipeline failed: {e}")
        import traceback
        traceback.print_exc()
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
