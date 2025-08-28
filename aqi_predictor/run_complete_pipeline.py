#!/usr/bin/env python3
"""
Complete AQI Pipeline Runner
Comprehensive script to run the full AQI prediction pipeline with proper error handling
This script ensures all steps are executed in the correct order for maximum accuracy
"""

import sys
import os
from datetime import datetime, timedelta
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def run_full_pipeline(clean_first: bool = False):
    """
    Run the complete pipeline using the main pipeline orchestrator
    This ensures maximum accuracy by using the full feature engineering process
    
    Parameters:
    clean_first (bool): If True, clean all feature groups first for a fresh start
    """
    print("🚀 RUNNING COMPLETE AQI PIPELINE (MAXIMUM ACCURACY)")
    print("="*80)
    
    try:
        from pipelines.pipeline import AQIPipeline
        
        # Initialize pipeline
        pipeline = AQIPipeline()
        
        # Optional: Clean feature groups first for fresh start
        if clean_first:
            print("\n🧹 CLEANING EXISTING FEATURE GROUPS FOR FRESH START")
            print("="*60)
            if pipeline.hopsworks.enabled:
                success = pipeline.hopsworks.clean_all_feature_groups(confirm=True)
                if success:
                    print("✅ Feature groups cleaned successfully")
                else:
                    print("⚠️ Feature group cleanup had issues, continuing anyway")
            else:
                print("⚠️ Hopsworks not enabled, skipping cleanup")
        
        # Run full training pipeline with comprehensive settings
        results = pipeline.run_full_training_pipeline(
            years_back=1,           # 1 year of historical data
            max_features=25,        # Use more features for better accuracy
            train_sklearn=True,     # Train sklearn models
            train_dl=True,          # Train deep learning models
            dl_epochs=100           # Sufficient epochs for convergence
        )
        
        if results['final_status'] == 'success':
            print("\n🎉 COMPLETE PIPELINE SUCCESSFUL!")
            print("="*50)
            
            if 'sklearn_best_model' in results:
                print(f"🏆 Best Sklearn Model: {results['sklearn_best_model']}")
                print(f"📊 Sklearn R²: {results['sklearn_best_r2']:.4f}")
                
            if 'dl_best_model' in results:
                print(f"🏆 Best Deep Learning Model: {results['dl_best_model']}")
                print(f"📊 Deep Learning R²: {results['dl_best_r2']:.4f}")
            
            print(f"\n✅ All models trained and saved successfully!")
            print(f"✅ Feature engineering completed with {results.get('data_shape', ['N/A', 'N/A'])[1]} features")
            
            return True
        else:
            print(f"\n❌ Pipeline completed with status: {results['final_status']}")
            if results.get('errors'):
                print("Errors encountered:")
                for error in results['errors']:
                    print(f"  - {error}")
            return False
            
    except ImportError as ie:
        print(f"❌ Import error: {ie}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Complete pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_incremental_pipeline():
    """
    Run the incremental pipeline (hourly + daily training)
    This uses the updated pipeline functions with complete feature engineering
    """
    print("🔄 RUNNING INCREMENTAL PIPELINE (HOURLY + DAILY)")
    print("="*70)
    
    try:
        from pipelines.fetch_data import run_hourly_data_pipeline, run_daily_model_pipeline
        
        # Step 1: Fetch/update data
        print("🔄 STEP 1: FETCHING LATEST DATA")
        print("-" * 40)
        
        hourly_success = run_hourly_data_pipeline()
        if not hourly_success:
            print("❌ Hourly data pipeline failed")
            return False
        
        print("\n✅ Data fetching completed successfully!")
        
        # Step 2: Train models with complete feature engineering
        print("\n🤖 STEP 2: TRAINING MODELS WITH COMPLETE FEATURE ENGINEERING")
        print("-" * 60)
        
        daily_success = run_daily_model_pipeline()
        if not daily_success:
            print("❌ Daily model training pipeline failed")
            return False
        
        print("\n🎉 INCREMENTAL PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅ Data updated and models trained with complete feature engineering")
        print("✅ Both Sklearn and Deep Learning models trained")
        
        return True
        
    except Exception as e:
        print(f"❌ Incremental pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 CHECKING ENVIRONMENT...")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Package name mapping: requirement name -> import name
    package_map = {
        "scikit-learn": "sklearn",
        "python-dotenv": "dotenv",
        "pyyaml": "yaml"
    }
    
    required_packages = [
        "pandas", "numpy", "scikit-learn", "tensorflow", 
        "requests", "joblib", "pyyaml", "python-dotenv"
    ]
    
    for package in required_packages:
        import_name = package_map.get(package, package)
        try:
            __import__(import_name.replace("-", "_"))
        except ImportError:
            issues.append(f"Missing package: {package}")
    
    # Check for config files
    if not os.path.exists('configs/config.yaml'):
        issues.append("Missing config file: configs/config.yaml")
    
    # Check for .env file (optional but recommended)
    if not os.path.exists('.env'):
        issues.append("Missing .env file (optional but recommended for Hopsworks)")
    
    if issues:
        print("❌ Environment issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nTo fix:")
        print("  1. pip install -r requirements.txt")
        print("  2. Ensure config files are present")
        print("  3. Create .env file with Hopsworks credentials (if using cloud)")
        return False
    else:
        print("✅ Environment looks good!")
        return True

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='AQI Pipeline Runner')
    parser.add_argument(
        'mode', 
        choices=['full', 'incremental', 'check', 'clean-full'],
        help='Pipeline mode: full (complete), incremental (hourly+daily), clean-full (clean & full), or check (environment)'
    )
    parser.add_argument(
        '--skip-check', 
        action='store_true',
        help='Skip environment check'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean existing feature groups before running (use with caution!)'
    )
    
    args = parser.parse_args()
    
    print("🌟 AQI PREDICTION PIPELINE RUNNER")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Environment check
    if not args.skip_check and args.mode != 'check':
        if not check_environment():
            print("\n❌ Environment check failed. Fix issues above and try again.")
            sys.exit(1)
        print()
    
    # Run selected mode
    if args.mode == 'check':
        success = check_environment()
    elif args.mode == 'full':
        print("📋 FULL PIPELINE MODE")
        print("This will:")
        print("  - Fetch 1 year of historical data")
        print("  - Perform comprehensive feature engineering")
        print("  - Train both Sklearn and Deep Learning models")
        print("  - Achieve maximum accuracy")
        if args.clean:
            print("  - ⚠️ CLEAN existing feature groups first (will delete data!)")
        print()
        success = run_full_pipeline(clean_first=args.clean)
    elif args.mode == 'clean-full':
        print("📋 CLEAN + FULL PIPELINE MODE")
        print("This will:")
        print("  - ⚠️ DELETE all existing feature group data")
        print("  - Fetch 1 year of historical data")
        print("  - Perform comprehensive feature engineering")
        print("  - Train both Sklearn and Deep Learning models")
        print("  - Ensure no schema conflicts")
        print()
        print("⚠️ WARNING: This will delete all previous data!")
        confirmation = input("Type 'YES' to confirm: ")
        if confirmation == 'YES':
            success = run_full_pipeline(clean_first=True)
        else:
            print("❌ Cancelled by user")
            success = False
    elif args.mode == 'incremental':
        print("📋 INCREMENTAL PIPELINE MODE")
        print("This will:")
        print("  - Update data incrementally")
        print("  - Perform complete feature engineering (same as full mode)")
        print("  - Train both Sklearn and Deep Learning models")
        print("  - Handle duplicates automatically")
        if args.clean:
            print("  - ⚠️ CLEAN existing feature groups first (not recommended for incremental)")
        print()
        success = run_incremental_pipeline()
    
    # Final status
    if success:
        print("\n🎉 PIPELINE EXECUTION SUCCESSFUL!")
        if args.mode in ['full', 'incremental', 'clean-full']:
            print("\n📱 Next steps:")
            print("  - Run the web app: python webapp/app.py")
            print("  - Check model files in the data/ directory")
            print("  - View feature store data in Hopsworks (if enabled)")
        sys.exit(0)
    else:
        print("\n❌ PIPELINE EXECUTION FAILED!")
        print("Check the error messages above for troubleshooting")
        sys.exit(1)

if __name__ == "__main__":
    main()
