"""
Shared utilities for Sixth Sense application.
Centralized functions to avoid code duplication and improve performance.
"""

import os
import json
import numpy as np
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Generator
import pandas as pd


# --- Constants ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MATCHES_DIR = os.path.join(PROJECT_ROOT, "matches")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
DEFAULT_CSV = os.path.join(DATA_DIR, "kills_data.csv")


def ensure_dirs():
    """Ensure all required directories exist."""
    for d in [MATCHES_DIR, DATA_DIR]:
        os.makedirs(d, exist_ok=True)


# --- Fast JSON Line Reader ---
def iter_jsonl(filepath: str, chunk_size: int = 8192) -> Generator[dict, None, None]:
    """
    Memory-efficient JSONL reader using generator pattern.
    Reads file in chunks for better I/O performance.

    Args:
        filepath: Path to JSONL file
        chunk_size: Read buffer size (default 8KB)

    Yields:
        Parsed JSON objects
    """
    with open(filepath, 'r', encoding='utf-8', buffering=chunk_size) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def count_lines_fast(filepath: str) -> int:
    """Fast line count using buffer reading."""
    count = 0
    with open(filepath, 'rb') as f:
        buf_size = 1024 * 1024  # 1MB buffer
        buf = f.raw.read(buf_size)
        while buf:
            count += buf.count(b'\n')
            buf = f.raw.read(buf_size)
    return count


# --- Cached Data Loading ---
@lru_cache(maxsize=16)
def load_match_data(match_name: str) -> Optional[pd.DataFrame]:
    """
    Load match data with LRU caching.
    Caches up to 16 most recently used matches.

    Args:
        match_name: Name of the match or "Default Match"

    Returns:
        DataFrame or None if not found
    """
    if match_name == "Default Match":
        csv_file = DEFAULT_CSV
    else:
        csv_file = os.path.join(MATCHES_DIR, f"{match_name}.csv")

    if not os.path.exists(csv_file):
        return None

    # Use optimized pandas read with specified dtypes
    dtypes = {
        'killer_name': 'category',
        'victim_name': 'category',
        'killer_agent': 'category',
        'victim_agent': 'category',
        'weapon': 'category',
        'headshot': 'bool',
        'is_first_blood': 'bool'
    }

    try:
        df = pd.read_csv(csv_file, dtype=dtypes)
        return df
    except Exception:
        # Fallback without dtype optimization
        return pd.read_csv(csv_file)


def get_available_matches() -> List[str]:
    """Get list of available match names."""
    ensure_dirs()

    matches = []

    # Check matches directory
    if os.path.exists(MATCHES_DIR):
        for file in os.listdir(MATCHES_DIR):
            if file.endswith(".csv"):
                matches.append(os.path.splitext(file)[0])

    # Check default match
    if os.path.exists(DEFAULT_CSV):
        matches.append("Default Match")

    return matches if matches else ["Default Match"]


# --- Optimized Statistics ---
def calculate_stats_vectorized(df: pd.DataFrame) -> Dict:
    """
    Calculate common statistics using vectorized operations.
    Much faster than iterating row by row.

    Args:
        df: DataFrame with kill data

    Returns:
        Dictionary of statistics
    """
    stats = {}

    # Basic counts
    stats['total_kills'] = len(df)

    # Kills per player (vectorized)
    stats['kills_by_player'] = df['killer_name'].value_counts().to_dict()
    stats['deaths_by_player'] = df['victim_name'].value_counts().to_dict()

    # First bloods
    if 'is_first_blood' in df.columns:
        fb_df = df[df['is_first_blood'] == True]
        stats['first_bloods_by_player'] = fb_df['killer_name'].value_counts().to_dict()
        stats['first_deaths_by_player'] = fb_df['victim_name'].value_counts().to_dict()

    # Headshots
    if 'headshot' in df.columns:
        hs_df = df[df['headshot'] == True]
        stats['headshots_by_player'] = hs_df['killer_name'].value_counts().to_dict()
        stats['headshot_rate'] = len(hs_df) / len(df) if len(df) > 0 else 0

    # Weapons
    if 'weapon' in df.columns:
        stats['kills_by_weapon'] = df['weapon'].value_counts().to_dict()

    return stats


def calculate_bounds_simple(x_coords: np.ndarray, y_coords: np.ndarray,
                            padding: float = 0.2) -> Dict:
    """
    Fast bounds calculation without DBSCAN (for when speed > outlier handling).

    Args:
        x_coords: Array of X coordinates
        y_coords: Array of Y coordinates
        padding: Padding percentage

    Returns:
        Dict with scale, off_x, off_y
    """
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    x_range = (max_x - min_x) * (1 + padding)
    y_range = (max_y - min_y) * (1 + padding)

    return {
        "scale": max(x_range, y_range),
        "off_x": center_x,
        "off_y": center_y
    }


# --- Event Extraction Helpers ---
def extract_player_info(game_state: dict) -> Tuple[str, Optional[float], Optional[float], str]:
    """
    Extract player info from game state efficiently.
    Updated to handle multiple GRID API JSON structures.

    Returns:
        (name, x, y, agent)
    """
    name = game_state.get('name', 'Unknown')

    # Try multiple paths for position data
    pos = game_state.get('position', {})
    if not pos:
        pos = game_state.get('location', {})

    x = pos.get('x')
    y = pos.get('y')

    # Try multiple paths for agent data
    agent = 'Unknown'
    agent_data = game_state.get('agent', {})

    if isinstance(agent_data, dict):
        agent = agent_data.get('name') or agent_data.get('displayName') or agent_data.get('id', 'Unknown')
    elif isinstance(agent_data, str) and agent_data:
        agent = agent_data

    # Also check character field (alternative naming in some GRID data)
    if agent == 'Unknown':
        char_data = game_state.get('character', {})
        if isinstance(char_data, dict):
            agent = char_data.get('name') or char_data.get('displayName') or char_data.get('id', 'Unknown')
        elif isinstance(char_data, str) and char_data:
            agent = char_data

    # Check selectedAgent field
    if agent == 'Unknown':
        sel_agent = game_state.get('selectedAgent', {})
        if isinstance(sel_agent, dict):
            agent = sel_agent.get('name') or sel_agent.get('displayName') or sel_agent.get('id', 'Unknown')
        elif isinstance(sel_agent, str) and sel_agent:
            agent = sel_agent

    return name, x, y, agent


def extract_weapon_info(event: dict) -> Tuple[str, bool]:
    """
    Extract weapon and headshot info from event.
    Updated to handle multiple GRID API JSON structures.

    Returns:
        (weapon_name, is_headshot)
    """
    weapon = "Unknown"
    headshot = False

    # Try action field first
    action = event.get('action', {})
    if isinstance(action, dict):
        # Try different weapon field names
        weapon_data = action.get('weapon') or action.get('damageSource') or action.get('ability')
        if isinstance(weapon_data, dict):
            weapon = weapon_data.get('name') or weapon_data.get('displayName') or weapon_data.get('id', 'Unknown')
        elif isinstance(weapon_data, str) and weapon_data:
            weapon = weapon_data

        # Check for headshot
        headshot = bool(action.get('isHeadshot') or action.get('headshot') or action.get('isHeadShot', False))

        # Check damageType for headshot
        damage_type = action.get('damageType', '')
        if isinstance(damage_type, str) and 'head' in damage_type.lower():
            headshot = True

    # Try weapon at event level
    if weapon == "Unknown":
        w = event.get('weapon', {})
        if isinstance(w, dict):
            weapon = w.get('name') or w.get('displayName') or w.get('id', 'Unknown')
        elif isinstance(w, str) and w:
            weapon = w

    # Try ability field (for ability kills)
    if weapon == "Unknown":
        ability = event.get('ability', {})
        if isinstance(ability, dict):
            weapon = ability.get('name') or ability.get('displayName') or ability.get('id', 'Unknown')
        elif isinstance(ability, str) and ability:
            weapon = ability

    # Try damageSource at event level
    if weapon == "Unknown":
        ds = event.get('damageSource', {})
        if isinstance(ds, dict):
            weapon = ds.get('name') or ds.get('displayName') or ds.get('id', 'Unknown')
        elif isinstance(ds, str) and ds:
            weapon = ds

    return weapon, headshot

