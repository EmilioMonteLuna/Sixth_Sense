"""
Optimized GRID API Client for Sixth Sense.

Improvements:
- Streaming downloads (memory efficient for large files)
- Connection pooling via requests.Session
- Better error handling and retry logic
- Progress indication for large downloads
"""

import os
import sys
import requests
import zipfile
import io
from dotenv import load_dotenv
from typing import Optional

# Load environment
load_dotenv()

# Configuration
API_KEY = os.getenv("GRID_API_KEY")
BASE_URL = "https://api.grid.gg/file-download"
DEFAULT_SERIES_ID = "2648634"

# Connection pooling - reuse TCP connections
_session: Optional[requests.Session] = None


def get_session() -> requests.Session:
    """Get or create a reusable session with connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"x-api-key": API_KEY})
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=5,
            pool_maxsize=10,
            max_retries=3
        )
        _session.mount("https://", adapter)
    return _session


def check_api_key() -> bool:
    """Verify API key is configured."""
    if not API_KEY:
        print("❌ Error: GRID_API_KEY not found in .env")
        return False
    return True


def get_download_url(series_id: str) -> Optional[str]:
    """
    Get download URL for a match's event file.

    Args:
        series_id: GRID series ID

    Returns:
        Download URL or None if not found
    """
    if not check_api_key():
        return None

    print(f"🔍 Checking files for Match {series_id}...")

    session = get_session()
    url = f"{BASE_URL}/list/{series_id}"

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Error: {e}")
        return None

    files = response.json().get('files', [])

    # Find event file
    for f in files:
        name = f['fileName'].lower()
        if 'events' in name and 'grid' in name and '.zip' in name:
            print(f"✅ Found data file: {f['fileName']}")
            return f['fullURL']

    print("❌ No valid event file found.")
    if files:
        print("Available files:", [f['fileName'] for f in files])
    return None


def download_with_progress(url: str, output_path: str, chunk_size: int = 8192) -> bool:
    """
    Stream download with progress indication.
    Memory efficient - doesn't load entire file into memory.

    Args:
        url: Download URL
        output_path: Path to save file
        chunk_size: Download chunk size (default 8KB)

    Returns:
        True if successful
    """
    session = get_session()

    try:
        response = session.get(url, stream=True, timeout=120)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        print("⬇️ Downloading", end="", flush=True)

        # Stream to temporary file first
        temp_path = output_path + ".tmp"
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Progress dots
                    if total_size > 0 and downloaded % (1024 * 1024) < chunk_size:
                        print(".", end="", flush=True)

        print(" Done!")

        # Rename temp to final
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)

        return True

    except requests.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        return False


def extract_jsonl_from_zip(zip_path: str, output_path: str) -> bool:
    """
    Extract JSONL file from downloaded ZIP.

    Args:
        zip_path: Path to ZIP file
        output_path: Path for extracted JSONL

    Returns:
        True if successful
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file_name in z.namelist():
                if file_name.endswith('.jsonl') or file_name.endswith('.json'):
                    print(f"📂 Extracting {file_name}...")

                    # Extract with streaming for large files
                    with z.open(file_name) as src, open(output_path, 'wb') as dst:
                        # Copy in chunks
                        chunk_size = 1024 * 1024  # 1MB
                        while True:
                            chunk = src.read(chunk_size)
                            if not chunk:
                                break
                            dst.write(chunk)

                    print(f"✅ Extracted to: {output_path}")
                    return True

        print("❌ No JSONL file found in ZIP")
        return False

    except zipfile.BadZipFile:
        print("❌ Error: Invalid ZIP file")
        return False


def download_and_save(url: str, output_path: str = "data/real_match.jsonl") -> Optional[str]:
    """
    Download ZIP, extract JSONL, and clean up.

    Args:
        url: Download URL
        output_path: Path for output JSONL

    Returns:
        Output path if successful, None otherwise
    """
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Download ZIP
    zip_path = output_path + ".zip"
    if not download_with_progress(url, zip_path):
        return None

    # Extract JSONL
    if not extract_jsonl_from_zip(zip_path, output_path):
        return None

    # Clean up ZIP
    try:
        os.remove(zip_path)
    except OSError:
        pass

    print(f"\n🎉 SUCCESS! Data saved to: {output_path}")
    print("👉 NOW RUN: python process_kills.py")

    return output_path


def download_match(series_id: str, match_name: Optional[str] = None) -> Optional[str]:
    """
    Download a match from GRID API.

    Args:
        series_id: GRID series ID
        match_name: Optional name for output file

    Returns:
        Path to downloaded JSONL, or None if failed
    """
    if match_name:
        output_path = f"data/{match_name}.jsonl"
        os.makedirs("data", exist_ok=True)
    else:
        output_path = "data/real_match.jsonl"

    download_url = get_download_url(series_id)
    if download_url:
        return download_and_save(download_url, output_path)
    return None


if __name__ == "__main__":
    if not check_api_key():
        sys.exit(1)

    # Parse arguments
    series_id = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_SERIES_ID
    match_name = sys.argv[2] if len(sys.argv) >= 3 else None

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           GRID Match Downloader - Sixth Sense                ║
╠══════════════════════════════════════════════════════════════╣
║ Usage: python grid_client.py [SERIES_ID] [MATCH_NAME]        ║
║                                                              ║
║ Examples:                                                    ║
║   python grid_client.py                    (default match)   ║
║   python grid_client.py 2648634            (specific match)  ║
║   python grid_client.py 2648634 VCT_Final  (with name)       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    result = download_match(series_id, match_name)

    if result:
        print(f"\n📊 To process the match data, run:")
        if match_name:
            print(f"   python process_kills.py {result} matches/{match_name}.csv")
        else:
            print(f"   python process_kills.py")

    sys.exit(0 if result else 1)
