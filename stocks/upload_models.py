"""
upload_models.py — One-time upload of stock model files to Supabase Storage.

Usage:
    SUPABASE_URL=... SUPABASE_KEY=... python3 upload_models.py

Creates bucket 'stock-models' if it doesn't exist, then uploads:
  - model_{ticker}.joblib for all tickers
  - features_{ticker}.csv for all tickers
  - ticker_summary.csv
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import db

def main():
    client = db._get_client()
    if not client:
        print("ERROR: No Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY.")
        sys.exit(1)

    # Create bucket if needed (ignore error if already exists)
    try:
        client.storage.create_bucket("stock-models", options={"public": False})
        print("Created bucket 'stock-models'")
    except Exception as e:
        if "already exists" in str(e).lower() or "Duplicate" in str(e):
            print("Bucket 'stock-models' already exists")
        else:
            print(f"Bucket creation note: {e}")

    files_to_upload = []

    # ticker_summary.csv
    if (ROOT / "ticker_summary.csv").exists():
        files_to_upload.append(("ticker_summary.csv", ROOT / "ticker_summary.csv"))

    # model and feature files
    for path in sorted(ROOT.glob("model_*.joblib")):
        if "crypto" in path.name:
            continue  # skip crypto models
        files_to_upload.append((path.name, path))

    for path in sorted(ROOT.glob("features_*.csv")):
        if "crypto" in path.name:
            continue
        files_to_upload.append((path.name, path))

    print(f"\nUploading {len(files_to_upload)} files...")
    ok = 0
    fail = 0
    for filename, local_path in files_to_upload:
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"  {filename} ({size_mb:.1f} MB)...", end=" ", flush=True)
        if db.upload_stock_file(filename, local_path):
            print("OK")
            ok += 1
        else:
            print("FAILED")
            fail += 1

    print(f"\nDone: {ok} uploaded, {fail} failed.")

if __name__ == "__main__":
    main()
