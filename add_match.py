"""
Add Match Helper Script
Downloads a match from GRID and processes it into the matches folder.

Usage:
    python add_match.py <SERIES_ID> <MATCH_NAME>

Example:
    python add_match.py 2648634 VCT_Finals_Game1
    python add_match.py 2700000 Masters_Madrid
"""

import sys
import os
from grid_client import download_match
from process_kills import process_kills


def add_match(series_id, match_name):
    """
    Download and process a match, saving it to the matches folder.

    Args:
        series_id: GRID series ID
        match_name: Name for the match (used for file naming)
    """
    print(f"\n🎮 Adding match: {match_name} (Series ID: {series_id})")
    print("=" * 60)

    # Step 1: Download the match
    print("\n📥 Step 1: Downloading match data from GRID...")
    jsonl_path = download_match(series_id, match_name)

    if not jsonl_path or not os.path.exists(jsonl_path):
        print("❌ Failed to download match data.")
        return False

    # Step 2: Process the match
    print("\n⚙️ Step 2: Processing kill data...")
    os.makedirs("matches", exist_ok=True)
    output_csv = f"matches/{match_name}.csv"

    process_kills(jsonl_path, output_csv)

    if os.path.exists(output_csv):
        print(f"\n✅ SUCCESS! Match added: {match_name}")
        print(f"📁 CSV saved to: {output_csv}")
        print("\n👉 Refresh your Streamlit app to see the new match in the dropdown!")
        return True
    else:
        print("❌ Failed to process match data.")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("\n❌ Error: Please provide both SERIES_ID and MATCH_NAME")
        print("\nTo find Series IDs, use find_match_id.py or check the GRID portal.")
        sys.exit(1)

    series_id = sys.argv[1]
    match_name = sys.argv[2]

    # Clean match name (remove spaces and special chars)
    match_name = match_name.replace(" ", "_").replace("-", "_")

    success = add_match(series_id, match_name)
    sys.exit(0 if success else 1)

