#!/usr/bin/env python3
"""
Ultra AI Termux API Integration Test
Tests available Termux APIs and their functionality
"""

import subprocess
import json
import time
import os

def run_termux_command(command):
    """Run a Termux API command and return result"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        return {
            "success": result.returncode == 0,
            "output": result.stdout.strip(),
            "error": result.stderr.strip(),
            "command": command
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "command": command
        }

def test_termux_apis():
    """Test available Termux APIs"""
    print("🔧 Ultra AI Termux API Integration Test")
    print("=" * 50)
    
    # Check Termux environment
    print("🏠 Environment Check:")
    prefix = os.environ.get('PREFIX', '')
    if 'termux' in prefix:
        print(f"✅ Running in Termux environment: {prefix}")
    else:
        print(f"⚠️  Not in Termux environment: {prefix}")
    
    # Test basic APIs
    api_tests = [
        {
            "name": "Battery Status",
            "command": "termux-battery-status",
            "description": "Get device battery information"
        },
        {
            "name": "Device Info", 
            "command": "termux-telephony-deviceinfo",
            "description": "Get device telephony information"
        },
        {
            "name": "WiFi Connection",
            "command": "termux-wifi-connectioninfo",
            "description": "Get WiFi connection details"
        },
        {
            "name": "Notification Test",
            "command": "termux-notification --title 'Ultra AI Test' --content 'Termux API working!'",
            "description": "Send test notification"
        },
        {
            "name": "Sensor List",
            "command": "termux-sensor -l",
            "description": "List available device sensors"
        }
    ]
    
    working_apis = []
    failed_apis = []
    
    print(f"\n🧪 Testing {len(api_tests)} Termux APIs...")
    
    for api_test in api_tests:
        print(f"\n🔍 Testing {api_test['name']}...")
        result = run_termux_command(api_test['command'])
        
        if result['success']:
            print(f"✅ {api_test['name']}: WORKING")
            if result['output']:
                # Parse JSON output if possible
                try:
                    data = json.loads(result['output'])
                    if isinstance(data, dict) and len(data) <= 5:  # Show small objects
                        for key, value in list(data.items())[:3]:  # First 3 items
                            print(f"   {key}: {value}")
                        if len(data) > 3:
                            print(f"   ... and {len(data) - 3} more fields")
                    else:
                        print(f"   Data: {str(result['output'])[:50]}...")
                except:
                    # Not JSON, show raw output
                    output = result['output'][:100]
                    if output:
                        print(f"   Output: {output}...")
            working_apis.append(api_test['name'])
        else:
            print(f"❌ {api_test['name']}: FAILED")
            if result['error']:
                print(f"   Error: {result['error'][:100]}...")
            failed_apis.append(api_test['name'])
    
    # Test location (needs permission)
    print(f"\n🌍 Testing Location Services (may require permission)...")
    location_result = run_termux_command("timeout 10 termux-location -p network")
    if location_result['success']:
        print("✅ Location Services: WORKING")
        try:
            location_data = json.loads(location_result['output'])
            if 'latitude' in location_data and 'longitude' in location_data:
                print(f"   Location: {location_data['latitude']:.3f}, {location_data['longitude']:.3f}")
                print(f"   Provider: {location_data.get('provider', 'unknown')}")
                working_apis.append('Location Services')
        except:
            print(f"   Raw output: {location_result['output'][:50]}...")
            working_apis.append('Location Services')
    else:
        print("⚠️  Location Services: Permission required or unavailable")
        failed_apis.append('Location Services')
    
    # Summary
    print(f"\n📊 Termux API Test Results:")
    print(f"✅ Working APIs: {len(working_apis)}")
    for api in working_apis:
        print(f"   - {api}")
    
    if failed_apis:
        print(f"\n❌ Failed APIs: {len(failed_apis)}")
        for api in failed_apis:
            print(f"   - {api}")
    
    # Integration status
    working_ratio = len(working_apis) / len(api_tests + ['Location Services'])
    if working_ratio >= 0.7:
        print(f"\n🎉 TERMUX API INTEGRATION: GOOD ({working_ratio:.0%} functional)")
        return True
    elif working_ratio >= 0.4:
        print(f"\n⚠️  TERMUX API INTEGRATION: PARTIAL ({working_ratio:.0%} functional)")
        return True
    else:
        print(f"\n❌ TERMUX API INTEGRATION: LIMITED ({working_ratio:.0%} functional)")
        return False

if __name__ == "__main__":
    success = test_termux_apis()
    if success:
        print("\n✅ Ultra AI Termux Integration: FUNCTIONAL")
        exit(0)
    else:
        print("\n⚠️  Ultra AI Termux Integration: NEEDS ATTENTION")
        exit(1)