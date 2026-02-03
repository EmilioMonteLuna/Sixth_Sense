# Sixth Sense: Comprehensive Assistant Coach

## 🏆 Cloud9 x JetBrains Hackathon Submission

**Category 1: Comprehensive Assistant Coach**

Sixth Sense is a data-driven assistant coach for VALORANT that provides **personalized player insights**, **automated macro game review**, and **predictive "what-if" analysis** - exactly as specified in the hackathon requirements.

---

## ⚡ Key Features

### 1. Personalized Player/Team Improvement Insights

**Example Output:**

> **DATA:** Team loses approximately **78%** of rounds when **f0rsakeN** dies first (without KAST)
>
> **INSIGHT:** f0rsakeN's opening duel success rate heavily impacts the team. They died first in 8 rounds, potentially costing ~6 rounds.
>
> **RECOMMENDATION:** Review f0rsakeN's opening pathing and ensure trade support is always available.

### 2.  Automated Macro Game Review

- First Blood Impact Analysis (KAST-style)
- Round-by-round breakdown with kill distribution  
- Death Trap identification using ML clustering
- Weapon effectiveness analysis
- Player matchup patterns (Nemesis detection)

### 3. Predictive "What-If" Analysis

- "What if we avoided the main Death Trap areas?"
- "What if our most vulnerable player improved opening duels?"
- "What if we traded effectively after first blood deaths?"

Each prediction includes:
- Current state analysis with actual data
- Predicted improvement percentages
- Confidence level
- Implementation strategies

---

##  Unique Differentiator: Visual Map Analysis

Unlike typical analytics tools, Sixth Sense provides **spatial visualization** on actual map layouts:

-  **Auto-calibrated Map Images** aligned with GRID coordinates
-  **ML-Powered Death Trap Detection** using DBSCAN clustering
-  **Kill Zone Mapping** to identify successful positions
-  **Interactive Filtering** by player, map, and sensitivity

*Coaches can literally SEE where problems occur on the map.*

---

##  Quick Start

### Step 1: Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install required packages
pip install -r requirements.txt
```

### Step 2: Get Match Data (Optional - sample data included)
```bash
# Set your GRID API key
$env:GRID_API_KEY="<your_key>"

# Download a match
python grid_client.py <SERIES_ID>
```
*Skip this step if using the included `real_match.jsonl` sample data.*

### Step 3: Process Kill Data ⚠️ REQUIRED
```bash
python process_kills.py
```
This parses `real_match.jsonl` → `data/kills_data.csv`

**You must run this before launching the app!**

### Step 4: Launch the Dashboard
```bash
streamlit run app.py
```

### Adding More Matches
```bash
python process_kills.py <input.jsonl> matches/<match_name>.csv
```
Example:
```bash
python process_kills.py game2.jsonl matches/VCT_Game2.csv
```

---

## Sample Insights Generated

### Critical Impact Analysis
```
 KAST Impact Analysis

DATA: The team loses approximately 78% of rounds when f0rsakeN dies first

FINDING: f0rsakeN died first in 8 rounds (33.3% of all rounds)

IMPACT: This potentially cost the team ~6 rounds

STRATEGIC INSIGHT: f0rsakeN's opening duel success rate heavily impacts 
team performance. When they die "for free" (without Kill/Assist/Survive/Trade), 
the team has only a ~22% chance of winning that round.
```

### Death Pattern Analysis
```
 DATA: 35.2% of all deaths (127/361) occur in 3 identifiable death traps

 INSIGHT: These concentrated death zones indicate predictable positioning 
that opponents exploit.

📈 RECOMMENDATION:
1. Review the Death Map for exact locations
2. Develop alternative approach paths
3. Use utility to clear or block these areas
```

---

## ️ Project Structure

```
Sixth_Sense/
├── app.py                 # Main Streamlit application
├── grid_client.py         # GRID API client (streaming downloads)
├── process_kills.py       # Kill data extraction (generator-based)
├── add_match.py           # Helper to add new matches
├── utils.py               # Shared utilities & caching
├── requirements.txt       # Python dependencies
├── assets/                # Map images
│   ├── Ascent.png
│   ├── Bind.png
│   ├── Split.png
│   └── Abyss.png
└── matches/               # Additional match data
```

---

##  Technical Highlights

- **Memory Efficient**: Generator-based JSONL processing
- **Optimized I/O**: Batched CSV writing, streaming downloads
- **Smart Caching**: LRU cache for data, TTL cache for Streamlit
- **ML Analytics**: DBSCAN clustering for death pattern detection
- **Real Data**: All insights calculated from actual match events

---

##  Supported Data
- VALORANT match data from GRID API
- Kill/death events with coordinates
- Round information
- Weapon and headshot data
- First blood tracking

---

##  Hackathon Alignment Checklist

| Requirement | Implementation |
|-------------|---------------|
| Personalized Player Insights | KAST-style analysis, Data→Insight format |
| Team Improvement Insights | Death patterns, matchup analysis | 
| Automated Macro Game Review | Round analysis, first blood impact |  
| Predict "What-If" Scenarios | 5 data-backed prediction scenarios |  
| GRID Data Integration | Download, process, visualize pipeline |  
| Provide Data/Reasoning | Every insight shows supporting data |  

---

*Developed for the Cloud9 x JetBrains Hackathon 2026*
