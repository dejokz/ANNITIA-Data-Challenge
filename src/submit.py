#!/usr/bin/env python3
"""
Trustii.io Submission Script

Submits notebook and CSV predictions to the ANNITIA Data Challenge.
Uses .env file for API credentials.

Usage:
    python src/submit.py
    
Environment Variables (in .env file):
    TOKEN: Trustii API token
    CHALLENGE_ID: Challenge ID (1551 for ANNITIA)
"""

import os
import sys
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
def load_env_file():
    """Load .env file from project root."""
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    
    if env_path.exists():
        # Try standard dotenv first
        load_dotenv(env_path)
        
        # If that doesn't work, parse manually (handles both = and : formats)
        if not os.getenv('TOKEN'):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # Handle both KEY=VALUE and KEY:VALUE formats
                    if '=' in line:
                        key, value = line.split('=', 1)
                    elif ':' in line:
                        key, value = line.split(':', 1)
                    else:
                        continue
                    os.environ[key.strip()] = value.strip()
        
        print(f"✅ Loaded environment from: {env_path}")
    else:
        print(f"⚠️  No .env file found at: {env_path}")
        print("   Expected format:")
        print("   TOKEN=your_api_token_here")
        print("   CHALLENGE_ID=1551")
    
    return project_root


def validate_files(project_root: Path) -> tuple:
    """Validate submission files exist."""
    # Default paths
    csv_path = project_root / 'submissions' / 'notebook_submission.csv'
    ipynb_path = project_root / 'submissions' / 'annitia_submission.ipynb'
    
    # Check if files exist
    files_found = []
    
    if csv_path.exists():
        files_found.append(('CSV', csv_path))
    else:
        # Try alternative CSV files
        alternatives = [
            project_root / 'submissions' / 'optimized_submission.csv',
            project_root / 'submissions' / 'final_submission.csv',
        ]
        for alt in alternatives:
            if alt.exists():
                files_found.append(('CSV', alt))
                break
    
    if ipynb_path.exists():
        files_found.append(('IPYNB', ipynb_path))
    else:
        alternatives = [
            project_root / 'notebooks' / 'annitia_submission.ipynb',
        ]
        for alt in alternatives:
            if alt.exists():
                files_found.append(('IPYNB', alt))
                break
    
    if len(files_found) < 2:
        print("\n❌ Error: Required files not found!")
        print("\nExpected files:")
        print(f"  - CSV: {csv_path} (or alternative)")
        print(f"  - IPYNB: {ipynb_path} (or alternative)")
        print("\nAvailable files in submissions/:")
        submissions_dir = project_root / 'submissions'
        if submissions_dir.exists():
            for f in submissions_dir.iterdir():
                print(f"  - {f.name}")
        sys.exit(1)
    
    csv_file = next(f[1] for f in files_found if f[0] == 'CSV')
    ipynb_file = next(f[1] for f in files_found if f[0] == 'IPYNB')
    
    print(f"✅ CSV file: {csv_file}")
    print(f"✅ Notebook: {ipynb_file}")
    
    return csv_file, ipynb_file


def submit_to_trustii(csv_path: Path, ipynb_path: Path, token: str, challenge_id: str):
    """Submit files to Trustii API."""
    
    endpoint_url = f'https://api.trustii.io/api/ds/notebook/datasets/{challenge_id}/prediction'
    
    print(f"\n🚀 Submitting to challenge {challenge_id}...")
    print(f"📡 API Endpoint: {endpoint_url}")
    
    # Read files
    with open(csv_path, 'rb') as f:
        csv_data = f.read()
    with open(ipynb_path, 'rb') as f:
        ipynb_data = f.read()
    
    # Prepare request
    headers = {'Trustii-Api-User-Token': token}
    files = {
        'csv_file': (csv_path.name, csv_data, 'text/csv'),
        'ipynb_file': (ipynb_path.name, ipynb_data, 'application/x-ipynb+json'),
    }
    
    # Send request
    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            files=files,
            timeout=60
        )
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*70)
            print("✅ SUBMISSION SUCCESSFUL!")
            print("="*70)
            print(f"\nResponse:")
            print(json.dumps(result, indent=2))
            return result
        else:
            print("\n" + "="*70)
            print("❌ SUBMISSION FAILED")
            print("="*70)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            # Provide helpful error messages
            if response.status_code == 401:
                print("\n💡 Tip: Check your TOKEN in .env file")
            elif response.status_code == 404:
                print("\n💡 Tip: Check your CHALLENGE_ID in .env file")
            elif response.status_code == 413:
                print("\n💡 Tip: File size too large")
            
            sys.exit(1)
            
    except requests.exceptions.Timeout:
        print("\n❌ Error: Request timed out (60s)")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Connection failed. Check internet connection.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def main():
    """Main submission workflow."""
    print("="*70)
    print("TRUSTII.IO SUBMISSION SCRIPT")
    print("="*70)
    
    # Load environment
    project_root = load_env_file()
    
    # Get credentials
    token = os.getenv('TOKEN')
    challenge_id = os.getenv('CHALLENGE_ID', '1551')
    
    if not token:
        print("\n❌ Error: TOKEN not found in .env file")
        print("   Please add: TOKEN=your_api_token_here")
        sys.exit(1)
    
    if not challenge_id:
        print("\n❌ Error: CHALLENGE_ID not found in .env file")
        print("   Please add: CHALLENGE_ID=1551")
        sys.exit(1)
    
    # Mask token for display
    masked_token = token[:10] + "..." + token[-10:] if len(token) > 20 else "***"
    print(f"🔑 Token: {masked_token}")
    print(f"🏆 Challenge ID: {challenge_id}")
    
    # Validate files
    print("\n" + "-"*70)
    print("VALIDATING FILES")
    print("-"*70)
    csv_path, ipynb_path = validate_files(project_root)
    
    # Confirm submission (auto-yes for automation)
    print("\n" + "-"*70)
    # response = input("Proceed with submission? (yes/no): ")
    # if response.lower() not in ['yes', 'y']:
    #     print("❌ Submission cancelled")
    #     sys.exit(0)
    print("🚀 Auto-submitting...")
    
    # Submit
    result = submit_to_trustii(csv_path, ipynb_path, token, challenge_id)
    
    print("\n" + "="*70)
    print("SUBMISSION COMPLETE")
    print("="*70)
    print("\nCheck your submission at: https://app.trustii.io/datasets/1551")


if __name__ == '__main__':
    main()
