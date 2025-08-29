#!/usr/bin/env python3
"""
Clean Raw Features Script
Deletes the aqi_raw_features feature group from Hopsworks to remove duplicates
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def clean_raw_features():
    """Delete the aqi_raw_features feature group from Hopsworks"""
    
    print("🧹 CLEANING RAW FEATURES FEATURE GROUP")
    print("="*60)
    
    try:
        from pipelines.fetch_data import HopsworksIntegration
        
        # Initialize Hopsworks connection
        hops = HopsworksIntegration()
        
        if not hops.enabled:
            print("❌ Hopsworks not enabled - check your .env file")
            return False
        
        print(f"🔗 Connected to Hopsworks: {hops.project.name}")
        
        # Delete the raw features feature group
        fg_name = "aqi_raw_features"
        deleted_count = 0
        
        print(f"\n🗑️ Deleting feature group: {fg_name}")
        
        # Try multiple versions (just in case)
        for version in range(1, 5):  # Check versions 1-4
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
                if "does not exist" in str(version_error).lower() or "not found" in str(version_error).lower():
                    print(f"   📭 Version {version} does not exist")
                else:
                    print(f"   ⚠️ Error checking version {version}: {version_error}")
        
        if deleted_count > 0:
            print(f"\n✅ Successfully deleted {deleted_count} feature group version(s)")
            print(f"💡 The {fg_name} feature group has been completely removed")
            print(f"🔄 Next full pipeline run will recreate it with fresh data")
            return True
        else:
            print(f"\n📭 No {fg_name} feature groups found to delete")
            print(f"💡 Feature group may have already been cleaned")
            return True
            
    except Exception as e:
        print(f"❌ Error cleaning raw features: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    success = clean_raw_features()
    
    if success:
        print("\n🎉 RAW FEATURES CLEANUP COMPLETED!")
        print("="*50)
        print("✅ Raw features feature group has been deleted")
        print("🔄 You can now run the full pipeline to recreate it with fresh data")
        print("💡 After that, hourly scripts will only add incremental data")
        sys.exit(0)
    else:
        print("\n💥 CLEANUP FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
