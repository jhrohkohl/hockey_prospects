from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# League to league group mapping
LEAGUE_GROUPS = {
    'BELARUS': 'EURO-II',
    'CZECHIA': 'EURO-I',
    'DEL': 'EURO-II',
    'KHL': 'EURO-I',
    'LIGUE MAGNUS': 'EURO-II',
    'LIIGA': 'EURO-I',
    'NL': 'EURO-I',
    'NORWAY': 'EURO-II',
    'SHL': 'EURO-I',
    'SLOVAKIA': 'EURO-II',
    'BELARUS VYSSHAYA': 'EURO-III',
    'CZECHIA2': 'EURO-II',
    'HOCKEYALLSVENSKAN': 'EURO-II',
    'MESTIS': 'EURO-II',
    'MHL': 'EURO-II',
    'SL': 'EURO-II',
    'VHL': 'EURO-II',
    'J20 NATIONELL': 'EURO-III',
    'CZECH U20': 'EURO-III',
    'HOCKEYETTAN': 'EURO-III',
    'J18 NATIONELL': 'EURO-III',
    'J18 REGION': 'EURO-III',
    'RUSSIA U18': 'EURO-III',
    'U20 SM-SARJA': 'EURO-III',
    'U18 SM-SARJA': 'EURO-III',
    'OHL': 'CAN-MJ',
    'NCAA': 'NCAA',
    'QMJHL': 'CAN-MJ',
    'WHL': 'CAN-MJ',
    'NTDP': 'USCAN-I',
    'USHL': 'USCAN-I',
    'OJHL': 'USCAN-II',
    'AJHL': 'USCAN-II',
    'BCHL': 'USCAN-II',
    'CAHS': 'PREP',
    'CISAA': 'PREP',
    'MPHL': 'PREP',
    'PHC': 'PREP',
    'USHS-MN': 'PREP',
    'USHS-MI': 'PREP',
    'USHS-PREP': 'PREP',
    'USPHL PREMIER': 'USCAN-II',
    'U18 AAA': 'USCAN-II'
}

# Improved file loading logic
def find_data_file():
    """
    Find the data file using multiple search strategies.
    
    Returns:
    --------
    str
        Path to the data file, or None if not found
    """
    # Get various possible directories
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    working_dir = os.getcwd()
    
    # Possible file names
    possible_filenames = ['d0.csv', 'd0.xlsx', 'd0', 'd0.xls']
    
    # Possible locations to search
    search_dirs = [
        current_dir,                          # Directory of this script
        working_dir,                          # Current working directory
        os.path.join(current_dir, 'data'),    # data subdirectory of script dir
        os.path.join(working_dir, 'data'),    # data subdirectory of working dir
        os.path.join(os.path.dirname(current_dir), 'data')  # Parent dir's data folder
    ]
    
    # Also check if path is provided via environment variable
    env_path = os.environ.get('CSV_PATH')
    if env_path and os.path.exists(env_path):
        print(f"Found data file from environment variable: {env_path}")
        return env_path
    
    # Try all combinations of directories and filenames
    for directory in search_dirs:
        if os.path.exists(directory):
            print(f"Searching in directory: {directory}")
            for filename in possible_filenames:
                potential_path = os.path.join(directory, filename)
                if os.path.exists(potential_path):
                    print(f"Found data file: {potential_path}")
                    return potential_path
        else:
            print(f"Directory does not exist: {directory}")
    
    # If we can't find the file, also try hardcoded paths for common locations
    hardcoded_paths = [
        r'C:\Users\jhroh\Desktop\player-similarity-api\d0.csv',
        r'/app/d0.csv',  # For Docker/container environments
    ]
    
    for path in hardcoded_paths:
        if os.path.exists(path):
            print(f"Found data file at hardcoded path: {path}")
            return path
    
    print("Data file not found after trying all possible locations")
    return None

# Find the data file
CSV_FILE_PATH = find_data_file()
print(f"Selected data file path: {CSV_FILE_PATH}")

# Global variable to store the loaded dataframe
player_data = None

def get_league_group(league):
    """
    Get the league group for a given league.
    
    Parameters:
    -----------
    league : str
        The league name
    
    Returns:
    --------
    str
        The corresponding league group
    """
    if not isinstance(league, str):
        return "UNKNOWN"
    return LEAGUE_GROUPS.get(league.upper(), "UNKNOWN")

def find_similar_players(df, league, position, gp, g, a, points, ppg, ht, wt, n_neighbors=6):
    """
    Find the most similar player seasons based on provided stats,
    using Goals per Game and Assists per Game instead of raw totals.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing player season data
    league : str
        The league the player belongs to
    position : str
        The position of the player
    gp : int
        Games played
    g : int
        Goals scored (will be converted to Goals per Game)
    a : int
        Assists (will be converted to Assists per Game)
    points : int
        Total points (will not be used since we're using PPG)
    ppg : float
        Points per game
    ht : float
        Height in inches
    wt : int
        Weight in pounds
    n_neighbors : int, default=3
        Number of similar players to return
    
    Returns:
    --------
    pandas DataFrame
        The n_neighbors most similar player seasons with their stats
    """
    # NHLe factors mapping
    NHLE_FACTORS = {
        'NHL': 1,
        'KHL': 0.771732,
        'CZECHIA': 0.582944,
        'SHL': 0.565769,
        'NL': 0.458792,
        'LIIGA': 0.441241,
        'AHL': 0.389427,
        'DEL': 0.351879,
        'HOCKEYALLSVENSKAN': 0.351322,
        'HOCKEYETTAN': 0.351322,
        'VHL': 0.328261,
        'SLOVAKIA': 0.295397,
        'CZECHIA2': 0.240439,
        'NCAA': 0.193751,
        'NORWAY': 0.172818,
        'MESTIS': 0.177838,
        'SL': 0.176281,
        'OHL': 0.144065,
        'MHL': 0.143426,
        'USHL': 0.143111,
        'WHL': 0.141272,
        'RUSSIA U18': 0.032422,
        'J20 NATIONELL': 0.091359,
        'J18 NATIONELL': 0.037962,
        'QMJHL': 0.112517,
        'NTDP': 0.121427,
        'U18 SM-SARJA': 0.03986,
        'CZECH U20': 0.07382,
        'J18 REGION': 0.029468,
        'USHS-PREP': 0.028378,
        'AJHL': 0.062462,
        'U20 SM-SARJA': 0.083147,
        'CAHS': 0.020197,
        'CISAA': 0.02742,
        'MPHL': 0.034798,
        'OJHL': 0.034296,
        'USPHL PREMIER': 0.04564,
        'BCHL': 0.080136,
        'USHS-MA': 0.028378,
        'USHS-MN': 0.024125,
        'USHS-MI': 0.028378,
        'BELARUS': 0.242295,
        'BELARUS VYSSHAYA': 0.052128,
        'LIGUE MAGNUS': 0.250187,
        'PHC': 0.124583,
        'U18 AAA': 0.031348
    }
    
    # 1. Filter dataframe to only include players from the same position (removing league group filtering)
    filtered_df = df[df['Position'] == position].copy()
    
    if len(filtered_df) < n_neighbors:
        print(f"Warning: Only {len(filtered_df)} players found with position={position}")
        if len(filtered_df) == 0:
            return pd.DataFrame()  # Return empty DataFrame if no matches
    
    # Calculate Goals per Game and Assists per Game for filtered dataframe
    # Handle division by zero or NaN values
    filtered_df['GPG'] = filtered_df.apply(
        lambda row: row['G'] / row['GP'] if pd.notna(row['G']) and pd.notna(row['GP']) and row['GP'] > 0 else 0, 
        axis=1
    )
    
    filtered_df['APG'] = filtered_df.apply(
        lambda row: row['A'] / row['GP'] if pd.notna(row['A']) and pd.notna(row['GP']) and row['GP'] > 0 else 0, 
        axis=1
    )
    
    # Calculate input player's GPG and APG
    input_gpg = g / gp if gp > 0 else 0
    input_apg = a / gp if gp > 0 else 0
    
    # 2. Add NHLe factor to the filtered dataframe
    filtered_df['NHLe'] = filtered_df['League'].apply(
        lambda x: NHLE_FACTORS.get(x, 0.1) if isinstance(x, str) else 0.1  # Default to 0.1 if not found
    )
    
    # Get NHLe factor for input player
    input_nhle = NHLE_FACTORS.get(league, 0.1) if isinstance(league, str) else 0.1
    
    # 3. Add League Match feature (1 if same league, 0 if different)
    filtered_df['League_Match'] = filtered_df['League'].apply(
        lambda x: 1 if x == league else 0
    )
    
    # Select numerical features for comparison, including NHLe and League_Match
    features = ['GPG', 'APG', 'PPG', 'NHLe', 'League_Match', 'Ht', 'Wt']
    X = filtered_df[features].copy()
    
    # Handle missing values - fill with median which is more robust than mean
    for feature in features:
        X[feature] = X[feature].fillna(X[feature].median())
    
    # Check if we still have NaN values after imputation
    if X.isna().any().any():
        print("Warning: After imputation, some NaN values remain. Removing these rows...")
        X = X.dropna()
        filtered_df = filtered_df.loc[X.index]
        
        if len(X) == 0:
            print(f"Error: No valid data points remain for position={position}")
            return pd.DataFrame()
        
        if len(X) < n_neighbors:
            print(f"Warning: Only {len(X)} valid players remain after NaN removal.")
            n_neighbors = min(n_neighbors, len(X))
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create input player vector (including NHLe and League_Match) and scale it
    # League_Match is always 1 for self-comparison
    input_player = np.array([[input_gpg, input_apg, ppg, input_nhle, 1, ht, wt]])
    input_player_scaled = scaler.transform(input_player)
    
    # Find nearest neighbors
    n_neighbors_to_use = min(n_neighbors+1, len(X))
    if n_neighbors_to_use <= 0:
        print("Error: Not enough valid data points for KNN.")
        return pd.DataFrame()
        
    nbrs = NearestNeighbors(n_neighbors=n_neighbors_to_use, algorithm='auto').fit(X_scaled)
    distances, indices = nbrs.kneighbors(input_player_scaled)
    
    # Get the similar players (skip the first one if it's an exact match)
    similar_indices = indices[0][1:] if distances[0][0] < 1e-6 and len(distances[0]) > 1 else indices[0][:min(n_neighbors, len(indices[0]))]
    
    # Return the similar players with their stats
    result = filtered_df.iloc[similar_indices].copy()
    
    # Add distance score to the result
    similarity_distances = distances[0][1:] if distances[0][0] < 1e-6 and len(distances[0]) > 1 else distances[0][:min(n_neighbors, len(distances[0]))]
    result['Similarity_Score'] = 1 / (1 + similarity_distances)
    
    # Add input values for comparison
    result['Input_GPG'] = input_gpg
    result['Input_APG'] = input_apg
    result['Input_PPG'] = ppg
    result['Input_Ht'] = ht
    result['Input_Wt'] = wt
    result['Input_League'] = league
    
    # Select relevant columns for display
    display_cols = ['Player Name', 'Draft Year', 'Draft Round', 'Draft Overall Pick', 'League', 
                   'Position', 'GP', 'G', 'A', 'Points', 'GPG', 'APG', 'PPG', 'Ht', 'Wt', 
                   'Similarity_Score']
    
    # Make sure all requested columns exist in the result
    existing_cols = [col for col in display_cols if col in result.columns]
    
    return result[existing_cols].sort_values('Similarity_Score', ascending=False)

def load_player_data():
    """Load the player data from the CSV file."""
    global player_data
    
    try:
        print(f"Attempting to load data from: {CSV_FILE_PATH}")
        
        if CSV_FILE_PATH is None:
            print("Error: CSV_FILE_PATH is None. No data file found.")
            return None
            
        # Check file extension to determine how to load it
        if CSV_FILE_PATH.endswith('.xlsx') or CSV_FILE_PATH.endswith('.xls'):
            # Load Excel file
            df = pd.read_excel(CSV_FILE_PATH)
        else:
            # Try loading as CSV
            try:
                # First try with default settings
                df = pd.read_csv(CSV_FILE_PATH)
            except Exception as e:
                print(f"Error with default CSV load: {e}")
                # Try with different encodings and separators
                encodings = ['utf-8', 'latin1', 'ISO-8859-1']
                separators = [',', ';', '\t']
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            print(f"Trying with encoding {encoding} and separator '{sep}'")
                            df = pd.read_csv(CSV_FILE_PATH, encoding=encoding, sep=sep)
                            print("Success!")
                            break
                        except Exception as e:
                            print(f"Failed with encoding {encoding} and separator '{sep}': {e}")
                    else:
                        continue
                    break
                else:
                    raise Exception("Could not load CSV with any combination of encoding and separator")
        
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {df.columns.tolist()}")
        
        # Make sure 'League Group' column exists
        if 'League Group' not in df.columns and 'League' in df.columns:
            df['League Group'] = df['League'].apply(get_league_group)
        
        # Return the loaded DataFrame
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

@app.route('/')
def index():
    """Simple test endpoint"""
    return jsonify({
        "status": "ok",
        "message": "Player Similarity API is running",
        "file_path": CSV_FILE_PATH,
        "file_exists": CSV_FILE_PATH is not None and os.path.exists(CSV_FILE_PATH),
        "working_dir": os.getcwd(),
        "endpoints": {
            "/api/league-groups": "GET - Get league to league group mappings",
            "/api/similar-players": "POST - Find similar players",
            "/api/data-info": "GET - Get information about the loaded data"
        }
    })

@app.route('/api/league-groups', methods=['GET'])
def get_league_groups():
    """
    Get the mapping of leagues to league groups.
    
    Returns:
    --------
    JSON response with mapping
    """
    return jsonify(LEAGUE_GROUPS)

@app.route('/api/similar-players', methods=['POST'])
def similar_players():
    """
    Find similar players based on input parameters.
    
    Request body (JSON):
    - league: str
    - position: str
    - gp: int
    - g: int
    - a: int
    - points: int
    - ppg: float
    - ht: float
    - wt: int
    - n_neighbors: int (optional, default=3)
    
    Returns:
    --------
    JSON response with similar players
    """
    global player_data
    
    # Lazy-load the data if it's not already loaded
    if player_data is None:
        player_data = load_player_data()
        
    if player_data is None:
        return jsonify({"error": "No data loaded. Please check if the CSV file exists."}), 500
    
    try:
        # Get parameters from request
        data = request.json
        
        # Validate required parameters
        required_params = ['league', 'position', 'gp', 'g', 'a', 'points', 'ppg', 'ht', 'wt']
        missing_params = [param for param in required_params if param not in data]
        
        if missing_params:
            return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400
        
        # Extract and convert parameters
        league = data['league']
        position = data['position']
        gp = int(data['gp'])
        g = int(data['g'])
        a = int(data['a'])
        points = int(data['points'])
        ppg = float(data['ppg'])
        ht = float(data['ht'])
        wt = int(data['wt'])
        n_neighbors = int(data.get('n_neighbors', 3))
        
        # Find similar players
        result = find_similar_players(
            player_data, league, position, gp, g, a, points, ppg, ht, wt, n_neighbors
        )
        
        # Convert to JSON-serializable format
        if result.empty:
            return jsonify([])
        
        # Convert the result to a list of dictionaries
        result_dict = result.to_dict(orient='records')
        
        # Convert NaN values to None for JSON serialization
        for i in range(len(result_dict)):
            for key, value in result_dict[i].items():
                if pd.isna(value):
                    result_dict[i][key] = None
        
        return jsonify(result_dict)
        
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route('/api/data-info', methods=['GET'])
def get_data_info():
    """
    Get information about the loaded data.
    
    Returns:
    --------
    JSON response with data information
    """
    global player_data
    
    # Lazy-load the data if it's not already loaded
    if player_data is None:
        player_data = load_player_data()
        
    if player_data is None:
        return jsonify({"error": "No data loaded. Please check if the CSV file exists."}), 500
    
    try:
        # Get basic info about the dataframe
        leagues = player_data['League'].unique().tolist() if 'League' in player_data.columns else []
        positions = player_data['Position'].unique().tolist() if 'Position' in player_data.columns else []
        league_groups = player_data['League Group'].unique().tolist() if 'League Group' in player_data.columns else []
        
        info = {
            "rows": len(player_data),
            "columns": player_data.columns.tolist(),
            "leagues": leagues,
            "positions": positions,
            "league_groups": league_groups
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({"error": f"Error retrieving data info: {str(e)}"}), 500

# Load the player data when the application starts (for local development)
if __name__ == '__main__':
    # Load the player data
    player_data = load_player_data()
    
    if player_data is None:
        print(f"Error: Could not load player data.")
        print("Make sure the CSV file exists and has the required columns.")
    else:
        print(f"Successfully loaded {len(player_data)} player records.")
    
    # Start the Flask application
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
else:
    # For Vercel - load data when module is imported
    player_data = load_player_data()
