# AQI Prediction Pipeline - Production Setup

## Overview

This project now includes a production-ready pipeline with:

- **Hourly data fetching**: Incremental data updates every hour
- **Daily model training**: Feature engineering and model training daily
- **Hopsworks integration**: Cloud-based feature store and model registry
- **GitHub Actions**: Automated scheduling

## Pipeline Architecture

### 1. Hourly Data Pipeline (`scripts/fetch_hourly_data.py`)

- Checks for existing data in Hopsworks and local files
- Fetches only new incremental data since last update
- Avoids duplicate API calls by checking timestamps
- Saves data to both local CSV and Hopsworks feature store

### 2. Daily Model Pipeline (`scripts/train_daily_models.py`)

- Loads latest data from Hopsworks feature store
- Performs feature engineering and selection
- Trains multiple ML models (Random Forest, XGBoost, etc.)
- Uploads best model to Hopsworks model registry

## Setup Instructions

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Hopsworks credentials in .env file
HOPSWORKS_API_KEY=your_api_key
HOPSWORKS_PROJECT_NAME=your_project_name
HOPSWORKS_HOST=your_host_url
```

### 2. GitHub Secrets Setup

In your GitHub repository settings, add these secrets:

- `HOPSWORKS_API_KEY`
- `HOPSWORKS_PROJECT_NAME`
- `HOPSWORKS_HOST`

### 3. Initial Data Setup

```bash
# Clean any test data first
python scripts/cleanup_test_data.py

# Run initial data setup (manual first run)
python pipelines/fetch_data.py hourly
```

## Usage

### Manual Execution

```bash
# Run hourly data fetching
python scripts/fetch_hourly_data.py

# Run daily model training
python scripts/train_daily_models.py

# Clean test data from Hopsworks
python scripts/cleanup_test_data.py
```

### Automated Execution (GitHub Actions)

- **Hourly**: Runs automatically every hour at minute 0
- **Daily**: Runs automatically every day at 2 AM UTC
- **Manual**: Can be triggered manually from GitHub Actions tab

## Data Flow

```
1. Hourly Pipeline:
   API → Local CSV → Hopsworks Feature Store (Raw)

2. Daily Pipeline:
   Hopsworks (Raw) → Feature Engineering → Hopsworks (Engineered)
   → Feature Selection → Hopsworks (Selected) → Model Training
   → Hopsworks Model Registry
```

## File Structure

```
aqi_predictor/
├── scripts/
│   ├── fetch_hourly_data.py      # Hourly data fetching
│   ├── train_daily_models.py     # Daily model training
│   └── cleanup_test_data.py      # Test data cleanup
├── .github/workflows/
│   ├── hourly-data-fetch.yml     # Hourly GitHub Action
│   └── daily-model-training.yml  # Daily GitHub Action
├── pipelines/
│   └── fetch_data.py             # Core pipeline functions
├── models/
│   └── train_sklearn.py          # Model training code
└── features/
    └── feature_engineering.py    # Feature engineering
```

## Key Features

### Incremental Data Fetching

- **Smart Updates**: Only fetches new data since last timestamp
- **Duplicate Prevention**: Checks both local and cloud storage
- **Error Recovery**: Falls back to full fetch if incremental fails

### Cloud-First Architecture

- **Feature Store**: All data stored in Hopsworks for consistency
- **Model Registry**: Automated model versioning and storage
- **Scalability**: Ready for production deployment

### GitHub Actions Integration

- **Scheduled Execution**: Fully automated pipeline
- **Error Handling**: Artifact upload on failures
- **Manual Triggers**: Can run pipelines on-demand

## Monitoring

### Success Indicators

- Hourly: New data appears in Hopsworks feature store
- Daily: New model version in Hopsworks model registry
- Both: GitHub Actions show green status

### Troubleshooting

- Check GitHub Actions logs for errors
- Verify Hopsworks credentials in .env file
- Run scripts manually to debug issues
- Use cleanup script to reset test data

## Data Sources

- **Weather Data**: Open-Meteo Archive API
- **Air Quality**: Open-Meteo Air Quality API
- **Location**: Lahore, Pakistan (31.5204°N, 74.3587°E)
- **Frequency**: Hourly data updates
- **Retention**: 1+ years of historical data
