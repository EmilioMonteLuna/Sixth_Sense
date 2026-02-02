"""
Optimized Kill Data Processor for Sixth Sense.

Improvements:
- Buffered CSV writing (reduces I/O operations)
- Generator-based JSONL reading (memory efficient)
- Batch processing for large files
- Progress tracking
- Map detection from GRID data
- Enhanced agent/weapon extraction
"""

import csv
import os
import sys
import json
from typing import Generator, Tuple, Optional

# Import from utils for shared functionality
from utils import iter_jsonl, extract_player_info, extract_weapon_info, ensure_dirs

# Default values
DEFAULT_INPUT_FILE = 'data/real_match.jsonl'
DEFAULT_OUTPUT_FILE = 'data/kills_data.csv'
MATCHES_DIR = 'matches'
DATA_DIR = 'data'

# CSV Headers
HEADERS = [
    'timestamp', 'round_number', 'map_name',
    'killer_name', 'killer_x', 'killer_y', 'killer_agent',
    'victim_name', 'victim_x', 'victim_y', 'victim_agent',
    'weapon', 'headshot', 'is_first_blood'
]


def detect_map_name(filepath: str) -> str:
    """
    Detect the map name from the JSONL file.

    Returns:
        Map name or 'Unknown'
    """
    map_name = 'Unknown'

    for transaction in iter_jsonl(filepath):
        events = transaction.get('events', [])
        for event in events:
            event_type = event.get('type', '')

            # Look for map in series-started-game or game-started events
            if event_type in ['series-started-game', 'game-started', 'match-started']:
                actor = event.get('actor', {})
                state = actor.get('state', {})

                # Try games array
                games = state.get('games', [])
                if games and len(games) > 0:
                    game = games[0]
                    map_data = game.get('map', {})
                    if isinstance(map_data, dict):
                        map_name = map_data.get('name') or map_data.get('displayName') or map_data.get('id', 'Unknown')
                    elif isinstance(map_data, str):
                        map_name = map_data
                    if map_name != 'Unknown':
                        return map_name

                # Try direct map field
                map_data = state.get('map', {})
                if isinstance(map_data, dict):
                    map_name = map_data.get('name') or map_data.get('displayName') or 'Unknown'
                elif isinstance(map_data, str):
                    map_name = map_data
                if map_name != 'Unknown':
                    return map_name

            # Also check target for map info
            if 'target' in event:
                target = event.get('target', {})
                state = target.get('state', {})
                map_data = state.get('map', {})
                if isinstance(map_data, dict):
                    map_name = map_data.get('name') or map_data.get('displayName') or 'Unknown'
                elif isinstance(map_data, str) and map_data:
                    map_name = map_data
                if map_name != 'Unknown':
                    return map_name

    return map_name


def extract_kills(filepath: str, map_name: str = 'Unknown') -> Generator[Tuple, None, None]:
    """
    Generator that extracts kill events from JSONL file.
    Memory-efficient: processes one line at a time.

    Yields:
        Tuple of kill data ready for CSV writing
    """
    current_round = 0
    round_first_blood = {}  # Track first blood per round

    for transaction in iter_jsonl(filepath):
        timestamp = transaction.get('occurredAt', 0)

        events = transaction.get('events', [])
        for event in events:
            event_type = event.get('type')

            # Track round changes
            if event_type == 'round-started':
                try:
                    current_round = event.get('target', {}).get('state', {}).get('roundNumber', current_round + 1)
                except (KeyError, TypeError):
                    current_round += 1
                round_first_blood[current_round] = False

            # Process kills
            elif event_type == 'player-killed-player':
                # Extract killer info - try multiple paths
                k_name, k_x, k_y, k_agent = "Unknown", None, None, "Unknown"
                try:
                    actor = event.get('actor', {})
                    state = actor.get('state', {})
                    game = state.get('game', state)  # Some formats have 'game' nested
                    k_name, k_x, k_y, k_agent = extract_player_info(game)
                except (KeyError, TypeError):
                    pass

                # Extract victim info - try multiple paths
                v_name, v_x, v_y, v_agent = "Unknown", None, None, "Unknown"
                try:
                    target = event.get('target', {})
                    state = target.get('state', {})
                    game = state.get('game', state)  # Some formats have 'game' nested
                    v_name, v_x, v_y, v_agent = extract_player_info(game)
                except (KeyError, TypeError):
                    pass

                # Skip if no victim location
                if v_x is None or v_y is None:
                    continue

                # Extract weapon info
                weapon, headshot = extract_weapon_info(event)

                # Determine first blood
                is_first_blood = not round_first_blood.get(current_round, False)
                if is_first_blood:
                    round_first_blood[current_round] = True

                yield (
                    timestamp, current_round, map_name,
                    k_name, k_x, k_y, k_agent,
                    v_name, v_x, v_y, v_agent,
                    weapon, headshot, is_first_blood
                )


def process_kills(input_file: str = None, output_file: str = None,
                  batch_size: int = 1000, show_progress: bool = True,
                  save_to_matches: bool = False) -> int:
    """
    Process kill/death data from JSONL to CSV.

    Optimizations:
    - Buffered writing (writes in batches)
    - Generator-based reading (low memory)
    - Progress indication
    - Auto-detects map name from data

    Args:
        input_file: Path to input JSONL
        output_file: Path to output CSV
        batch_size: Number of rows to buffer before writing
        show_progress: Whether to show progress dots
        save_to_matches: If True, also save to matches/ folder

    Returns:
        Number of kills processed
    """
    input_file = input_file or DEFAULT_INPUT_FILE
    output_file = output_file or DEFAULT_OUTPUT_FILE

    print(f"💀 Extracting Kill/Death data from {input_file}...")

    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        return 0

    # Ensure directories exist
    ensure_dirs()

    # Detect map name
    print("🗺️ Detecting map...")
    map_name = detect_map_name(input_file)
    print(f"   Map detected: {map_name}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    batch = []

    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(HEADERS)

        # Process kills using generator with map name
        for kill_data in extract_kills(input_file, map_name):
            batch.append(kill_data)
            count += 1

            # Write batch when full
            if len(batch) >= batch_size:
                writer.writerows(batch)
                batch.clear()
                if show_progress:
                    print(f".", end="", flush=True)

        # Write remaining batch
        if batch:
            writer.writerows(batch)

    if show_progress and count > 0:
        print()  # Newline after dots

    if count > 0:
        print(f"\n🎉 SUCCESS! Extracted {count} kills.")
        print(f"🗺️ Map: {map_name}")
        print(f"💾 Saved to: {output_file}")
        print(f"📊 Data includes: round numbers, map, agents, weapons, headshots, first bloods")

        # Also save to matches folder if requested or by default
        if save_to_matches or output_file == DEFAULT_OUTPUT_FILE:
            match_filename = os.path.basename(input_file).replace('.jsonl', '.csv')
            match_path = os.path.join(MATCHES_DIR, match_filename)

            # Copy to matches folder
            import shutil
            shutil.copy2(output_file, match_path)
            print(f"📂 Also saved to: {match_path}")

        print("👉 NOW RUN: streamlit run app.py")
    else:
        print("\n❌ Found 0 kills. Check the file path and format.")

    return count


def process_kills_fast(input_file: str = None, output_file: str = None) -> int:
    """
    Ultra-fast version that skips progress tracking.
    Use for automated pipelines.
    """
    return process_kills(input_file, output_file, batch_size=5000, show_progress=False)


if __name__ == "__main__":
    # Parse command line arguments
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE
    output_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_FILE

    process_kills(input_file, output_file)
