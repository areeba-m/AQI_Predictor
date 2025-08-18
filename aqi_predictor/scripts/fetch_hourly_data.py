#!/usr/bin/env python3
"""
Hourly Data Fetching Script for GitHub Actions
Fetches incremental data and updates Hopsworks feature store
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pipelines.fetch_data import run_hourly_data_pipeline

def main():
    """Run hourly data pipeline"""
    print("🕐 HOURLY DATA FETCHING JOB STARTED")
    print("="*60)
    
    try:
        success = run_hourly_data_pipeline()
        
        if success:
            print("\n✅ HOURLY JOB COMPLETED SUCCESSFULLY")
            sys.exit(0)
        else:
            print("\n❌ HOURLY JOB FAILED")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 HOURLY JOB CRASHED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
