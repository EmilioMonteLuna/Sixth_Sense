import json
import csv
import os

# CONFIGURATION
INPUT_FILE = 'data/real_match.jsonl'
OUTPUT_FILE = 'data/match_data.csv'


def extract_player_positions():
    print(f" Processing {INPUT_FILE}...")

    if not os.path.exists(INPUT_FILE):
        print("❌ Error: File not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
            open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f_out:

        # Define our simple CSV columns
        writer = csv.writer(f_out)
        writer.writerow(['timestamp', 'player_id', 'team_id', 'x', 'y'])

        count = 0
        found_positions = False

        for line_num, line in enumerate(f_in):
            try:
                # Parse the transaction
                transaction = json.loads(line)
                timestamp = transaction.get('occurredAt', 0)

                # Loop through events in this transaction
                if 'events' in transaction:
                    for event in transaction['events']:

                        # We are looking for the "Sampled" event (The Snapshot)
                        # It usually contains the full state of the game
                        if event.get('type') == 'grid-sampled-series':

                            # DIVE DEEP into the nested JSON
                            # Path: target -> state -> games -> [current] -> teams -> [all] -> players -> [all]
                            try:
                                state = event['target']['state']
                                if 'games' in state:
                                    # Usually the last game is the active one
                                    game = state['games'][-1]

                                    if 'teams' in game:
                                        for team in game['teams']:
                                            team_id = team.get('id', 'unknown')

                                            if 'players' in team:
                                                for player in team['players']:
                                                    player_id = player.get('id', 'unknown')

                                                    # CHECK FOR COORDINATES
                                                    # They might be in 'location' or directly in the player object
                                                    x, y = None, None

                                                    if 'location' in player:
                                                        x = player['location'].get('x')
                                                        y = player['location'].get('y')
                                                    elif 'x' in player and 'y' in player:
                                                        x = player['x']
                                                        y = player['y']

                                                    # If we found valid coordinates, save them!
                                                    if x is not None and y is not None:
                                                        writer.writerow([timestamp, player_id, team_id, x, y])
                                                        found_positions = True
                                                        count += 1

                            except (KeyError, IndexError, TypeError):
                                continue  # Skip malformed events

            except json.JSONDecodeError:
                continue

            # Progress Indicator
            if line_num % 1000 == 0:
                print(f"   Processed {line_num} lines... Found {count} positions so far.")

        if found_positions:
            print(f"\n SUCCESS! Extracted {count} player positions.")
            print(f" Saved to: {OUTPUT_FILE}")
        else:
            print("\n WARNING: Processed file but found NO player coordinates.")
            print("The structure might be different. We may need to look for 'payload' or other keys.")


if __name__ == "__main__":
    extract_player_positions()