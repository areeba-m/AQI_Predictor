#!/usr/bin/env python3
"""
Clean Engineered Features Script
Deletes only the aqi_engineered_features feature group from Hopsworks
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def clean_engineered_features():
    """Delete the aqi_engineered_features feature group from Hopsworks"""
    
    print("🧹 CLEANING ENGINEERED FEATURES FEATURE GROUP")
    print("="*60)
    
    try:
        from pipelines.fetch_data import HopsworksIntegration
        
        # Initialize Hopsworks connection
        hops = HopsworksIntegration()
        
        if not hops.enabled:
            print("❌ Hopsworks not enabled - check your .env file")
            return False
        
        print(f"🔗 Connected to Hopsworks: {hops.project.name}")
        
        # Delete the engineered features feature group
        fg_name = "aqi_engineered_features"
        deleted_count = 0
        
        print(f"\n🗑️ Deleting feature group: {fg_name}")
        
        # Try multiple versions (just in case)
        for version in range(1, 10):  # Check versions 1-9
            try:
                print(f"   Checking version {version}...")
                fg = hops.fs.get_feature_group(name=fg_name, version=version)
                
                if fg is not None:
                    print(f"   🗑️ Deleting {fg_name} v{version}...")
                    fg.delete()
                    print(f"   ✅ Deleted {fg_name} v{version}")
                    deleted_count += 1
                else:
                    print(f"   📭 Version {version} does not exist")
                    
            except Exception as version_error:
                error_msg = str(version_error).lower()
                if "does not exist" in error_msg or "not found" in error_msg or "feature group could not be found" in error_msg:
                    print(f"   📭 Version {version} does not exist")
                    if version > 3:  # Stop checking after a few non-existent versions
                        break
                else:
                    print(f"   ⚠️ Error checking version {version}: {version_error}")
        
        if deleted_count > 0:
            print(f"\n✅ Successfully deleted {deleted_count} feature group version(s)")
            print(f"💡 The {fg_name} feature group has been completely removed")
            print(f"🔄 Next full pipeline run will recreate engineered features with fresh data")
            print(f"📊 Raw features will remain untouched")
            return True
        else:
            print(f"\n📭 No {fg_name} feature groups found to delete")
            print(f"💡 Feature group may have already been cleaned")
            return True
            
    except Exception as e:
        print(f"❌ Error cleaning engineered features: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    success = clean_engineered_features()
    
    if success:
        print("\n🎉 ENGINEERED FEATURES CLEANUP COMPLETED!")
        print("="*50)
        print("✅ Engineered features feature group has been deleted")
        print("📊 Raw features remain intact")
        print("🔄 You can now run the full pipeline to recreate engineered features")
        print("💡 This will regenerate feature engineering from existing raw data")
        sys.exit(0)
    else:
        print("\n💥 CLEANUP FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
