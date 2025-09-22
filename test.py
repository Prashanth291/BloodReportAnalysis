#!/usr/bin/env python3
"""
Comprehensive Gemini API Test Script
This script tests various aspects of the Gemini API to diagnose issues
"""

import os
import sys
import json
from pathlib import Path

def print_separator(title):
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50)

def test_1_environment_variables():
    """Test if environment variables are loading correctly"""
    print_separator("TEST 1: ENVIRONMENT VARIABLES")
    
    # Method 1: Direct os.environ
    api_key_direct = os.environ.get("GEMINI_API_KEY")
    print(f"Direct os.environ: {bool(api_key_direct)}")
    if api_key_direct:
        print(f"  Length: {len(api_key_direct)}")
        print(f"  First 10: {api_key_direct[:10]}...")
        print(f"  Last 4: ...{api_key_direct[-4:]}")
    
    # Method 2: Using python-dotenv
    try:
        from dotenv import load_dotenv
        print("python-dotenv available: ✓")
        
        # Check if .env file exists
        env_file = Path(".env")
        print(f".env file exists: {env_file.exists()}")
        
        if env_file.exists():
            print(f".env file size: {env_file.stat().st_size} bytes")
            # Show contents (hide sensitive data)
            with open(".env", "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    if "GEMINI_API_KEY" in line:
                        parts = line.strip().split("=", 1)
                        if len(parts) == 2:
                            key_value = parts[1]
                            print(f"  Line {i}: GEMINI_API_KEY={key_value[:10]}...{key_value[-4:] if len(key_value) > 14 else ''}")
                        else:
                            print(f"  Line {i}: {line.strip()} (INVALID FORMAT)")
        
        # Load and test
        load_dotenv()
        api_key_dotenv = os.environ.get("GEMINI_API_KEY")
        print(f"After load_dotenv(): {bool(api_key_dotenv)}")
        
        return api_key_dotenv
        
    except ImportError:
        print("python-dotenv not available: ✗")
        print("Install with: pip install python-dotenv")
        return api_key_direct

def test_2_google_generativeai():
    """Test if google-generativeai library is working"""
    print_separator("TEST 2: GOOGLE GENERATIVEAI LIBRARY")
    
    try:
        import google.generativeai as genai
        print("google.generativeai import: ✓")
        
        # Check version
        try:
            version = genai.__version__
            print(f"Version: {version}")
        except:
            print("Version: Unknown")
        
        return genai
    except ImportError as e:
        print(f"google.generativeai import: ✗")
        print(f"Error: {e}")
        print("Install with: pip install google-generativeai")
        return None

def test_3_api_configuration(genai, api_key):
    """Test API configuration"""
    print_separator("TEST 3: API CONFIGURATION")
    
    if not genai:
        print("Skipping - library not available")
        return False
        
    if not api_key:
        print("Skipping - no API key")
        return False
    
    try:
        genai.configure(api_key=api_key)
        print("API configuration: ✓")
        return True
    except Exception as e:
        print(f"API configuration: ✗")
        print(f"Error: {e}")
        return False

def test_4_model_access(genai):
    """Test different model access"""
    print_separator("TEST 4: MODEL ACCESS")
    
    if not genai:
        print("Skipping - library not available")
        return
    
    models_to_test = [
        'gemini-1.5-flash',
        'gemini-1.5-pro', 
        'gemini-2.0-flash',
        'gemini-2.5-flash',
        'gemini-2.5-pro'
    ]
    
    working_models = []
    
    for model_name in models_to_test:
        try:
            model = genai.GenerativeModel(model_name)
            print(f"{model_name}: ✓")
            working_models.append(model_name)
        except Exception as e:
            print(f"{model_name}: ✗ ({str(e)[:50]}...)")
    
    return working_models

def test_5_simple_generation(genai, working_models):
    """Test simple text generation"""
    print_separator("TEST 5: SIMPLE TEXT GENERATION")
    
    if not genai or not working_models:
        print("Skipping - no working models")
        return
    
    test_prompt = "Say exactly: 'API_TEST_SUCCESS'"
    
    for model_name in working_models[:2]:  # Test first 2 working models
        try:
            print(f"Testing {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(test_prompt)
            print(f"  Response: {response.text.strip()}")
            print(f"  Status: ✓")
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Status: ✗")

def test_6_json_generation(genai, working_models):
    """Test JSON generation (for your use case)"""
    print_separator("TEST 6: JSON GENERATION")
    
    if not genai or not working_models:
        print("Skipping - no working models")
        return
    
    json_prompt = """
    Return ONLY a valid JSON object with this exact structure:
    {"test": "success", "number": 42, "working": true}
    """
    
    for model_name in working_models[:1]:  # Test first working model
        try:
            print(f"Testing JSON with {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(json_prompt)
            response_text = response.text.strip()
            print(f"  Raw response: {response_text}")
            
            # Try to parse JSON
            try:
                parsed = json.loads(response_text)
                print(f"  Parsed JSON: ✓")
                print(f"  Content: {parsed}")
            except json.JSONDecodeError as je:
                print(f"  JSON parsing: ✗ ({je})")
                
        except Exception as e:
            print(f"  Error: {e}")

def test_7_file_upload_capability(genai, working_models):
    """Test file upload capability"""
    print_separator("TEST 7: FILE UPLOAD CAPABILITY")
    
    if not genai or not working_models:
        print("Skipping - no working models")
        return
    
    # Create a test text file
    test_file_path = "test_upload.txt"
    try:
        with open(test_file_path, "w") as f:
            f.write("This is a test file for Gemini API upload capability.")
        
        print(f"Created test file: {test_file_path}")
        
        # Try to upload
        try:
            uploaded_file = genai.upload_file(test_file_path)
            print(f"File upload: ✓")
            print(f"  File name: {uploaded_file.name}")
            print(f"  State: {uploaded_file.state.name}")
            
            # Test with model
            if working_models:
                model = genai.GenerativeModel(working_models[0])
                response = model.generate_content([
                    "What does this file contain?", 
                    uploaded_file
                ])
                print(f"  Model response: {response.text.strip()}")
            
            # Cleanup
            genai.delete_file(uploaded_file.name)
            print(f"  Cleanup: ✓")
            
        except Exception as e:
            print(f"File upload: ✗")
            print(f"  Error: {e}")
        
        # Remove test file
        os.remove(test_file_path)
        
    except Exception as e:
        print(f"Test file creation failed: {e}")

def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE GEMINI API TEST")
    print("This script will test your Gemini API setup step by step")
    
    # Test 1: Environment Variables
    api_key = test_1_environment_variables()
    
    # Test 2: Library Import
    genai = test_2_google_generativeai()
    
    # Test 3: API Configuration
    config_success = test_3_api_configuration(genai, api_key)
    
    # Test 4: Model Access
    working_models = test_4_model_access(genai) if config_success else []
    
    # Test 5: Simple Generation
    test_5_simple_generation(genai, working_models)
    
    # Test 6: JSON Generation
    test_6_json_generation(genai, working_models)
    
    # Test 7: File Upload
    test_7_file_upload_capability(genai, working_models)
    
    # Summary
    print_separator("SUMMARY")
    print(f"API Key Found: {'✓' if api_key else '✗'}")
    print(f"Library Working: {'✓' if genai else '✗'}")
    print(f"Configuration OK: {'✓' if config_success else '✗'}")
    print(f"Working Models: {len(working_models) if working_models else 0}")
    
    if working_models:
        print("✅ Your Gemini API setup appears to be working!")
        print(f"   Recommended model: {working_models[0]}")
    else:
        print("❌ There are issues with your Gemini API setup")
        print("   Check the errors above for details")

if __name__ == "__main__":
    main()