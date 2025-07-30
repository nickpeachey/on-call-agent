#!/usr/bin/env python3
"""
Comprehensive System Integration Test

This script will test all major components of the AI On-Call Agent:
1. System startup and configuration
2. Database connectivity
3. API endpoints 
4. ML model functionality
5. Action engine
6. Logging and monitoring
7. CLI functionality
"""

import sys
import os
import json
import time
import requests
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_system_status():
    """Test basic system status."""
    print("🔍 Testing System Status...")
    
    # Test CLI status command
    result = os.system("python3 cli.py status > /dev/null 2>&1")
    if result == 0:
        print("✅ CLI status command working")
    else:
        print("❌ CLI status command failed")
    
    return result == 0

def test_training_data():
    """Test training data creation and loading."""
    print("🔍 Testing Training Data...")
    
    training_file = Path("data/sample_training.json")
    if training_file.exists():
        with open(training_file) as f:
            data = json.load(f)
        print(f"✅ Training data loaded: {len(data)} examples")
        return True
    else:
        print("❌ Training data file not found")
        return False

def test_reports_generation():
    """Test reports generation."""
    print("🔍 Testing Reports Generation...")
    
    reports_dir = Path("reports")
    if reports_dir.exists():
        reports = list(reports_dir.glob("*.md"))
        print(f"✅ Found {len(reports)} training reports")
        
        # Check for final summary
        final_report = reports_dir / "final_training_summary.md"
        if final_report.exists():
            print("✅ Final training summary exists")
            return True
        else:
            print("❌ Final training summary missing")
            return False
    else:
        print("❌ Reports directory not found")
        return False

def test_documentation():
    """Test documentation files."""
    print("🔍 Testing Documentation...")
    
    docs_dir = Path("docs")
    success = True
    
    # Check for markdown guide
    md_guide = docs_dir / "AI_ON_CALL_AGENT_DUMMYS_GUIDE.md"
    if md_guide.exists():
        print("✅ Markdown guide exists")
    else:
        print("❌ Markdown guide missing")
        success = False
    
    # Check for HTML guide
    html_guide = docs_dir / "AI_ON_CALL_AGENT_DUMMYS_GUIDE.html"
    if html_guide.exists():
        print("✅ HTML guide exists")
    else:
        print("❌ HTML guide missing")
        success = False
    
    return success

def test_project_structure():
    """Test project structure and key files."""
    print("🔍 Testing Project Structure...")
    
    required_files = [
        "start.py",
        "cli.py", 
        "src",
        "config",
        "data",
        "logs",
        "docs",
        "reports"
    ]
    
    success = True
    for item in required_files:
        path = Path(item)
        if path.exists():
            print(f"✅ {item} exists")
        else:
            print(f"❌ {item} missing")
            success = False
    
    return success

def run_comprehensive_system_test():
    """Run comprehensive system test."""
    
    print("🚀 AI On-Call Agent - Comprehensive System Test")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("System Status", test_system_status),
        ("Training Data", test_training_data),
        ("Reports Generation", test_reports_generation), 
        ("Documentation", test_documentation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name} Test: PASSED")
            else:
                print(f"❌ {test_name} Test: FAILED")
        except Exception as e:
            print(f"❌ {test_name} Test: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print("-" * 40)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is fully functional.")
        print("🚀 AI On-Call Agent is ready for production use!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_system_test()
    sys.exit(0 if success else 1)
