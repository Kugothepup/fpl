#!/usr/bin/env python3
"""
FPL Manager v3 - Historical Data Collection Service
Manages historical FPL data for improved ML predictions
"""

import sqlite3
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import csv
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PlayerPerformance:
    """Player performance in a specific match"""
    player_id: int
    element_id: int
    team_id: int
    opponent_id: int
    home_away: str  # 'home' or 'away'
    minutes: int
    goals: int
    assists: int
    saves: int
    yellow_cards: int
    red_cards: int
    bonus: int
    bps: int  # Bonus point system score
    cost: float
    selected_by_percent: float
    final_points: int

@dataclass
class MatchResult:
    """Complete match result with all player performances"""
    match_id: int
    gameweek: int
    season: str
    kickoff_time: datetime
    home_team_id: int
    away_team_id: int
    home_score: int
    away_score: int
    finished: bool
    player_performances: List[PlayerPerformance]

class HistoricalDataService:
    """Service for collecting and managing historical FPL data"""
    
    def __init__(self, db_path: str = "fpl_historical_data.db"):
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Initialize database schema for historical data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Seasons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS seasons (
                season_id TEXT PRIMARY KEY,
                start_date DATE,
                end_date DATE,
                total_gameweeks INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Teams table (maps team IDs across seasons)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER,
                season_id TEXT,
                name TEXT,
                short_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, season_id),
                FOREIGN KEY (season_id) REFERENCES seasons(season_id)
            )
        ''')
        
        # Players table (maps player IDs across seasons)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                element_id INTEGER,
                season_id TEXT,
                web_name TEXT,
                first_name TEXT,
                second_name TEXT,
                team_id INTEGER,
                position INTEGER,
                cost_start REAL,
                cost_end REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (element_id, season_id),
                FOREIGN KEY (season_id) REFERENCES seasons(season_id)
            )
        ''')
        
        # Matches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER,
                season_id TEXT,
                gameweek INTEGER,
                kickoff_time TIMESTAMP,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_score INTEGER,
                away_score INTEGER,
                finished BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (match_id, season_id),
                FOREIGN KEY (season_id) REFERENCES seasons(season_id)
            )
        ''')
        
        # Player performances table (detailed match-by-match data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_performances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                season_id TEXT,
                gameweek INTEGER,
                element_id INTEGER,
                team_id INTEGER,
                opponent_id INTEGER,
                home_away TEXT,
                minutes INTEGER DEFAULT 0,
                goals INTEGER DEFAULT 0,
                assists INTEGER DEFAULT 0,
                saves INTEGER DEFAULT 0,
                yellow_cards INTEGER DEFAULT 0,
                red_cards INTEGER DEFAULT 0,
                bonus INTEGER DEFAULT 0,
                bps INTEGER DEFAULT 0,
                cost REAL,
                selected_by_percent REAL,
                final_points INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id, season_id) REFERENCES matches(match_id, season_id),
                FOREIGN KEY (element_id, season_id) REFERENCES players(element_id, season_id)
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_player_performances_element ON player_performances(element_id, season_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_player_performances_gameweek ON player_performances(gameweek, season_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_player_performances_team ON player_performances(team_id, season_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_gameweek ON matches(gameweek, season_id)')
        
        conn.commit()
        conn.close()
        logger.info("Historical data database schema initialized")
    
    def parse_stats_json(self, stats_json: str) -> Dict[str, Dict[int, int]]:
        """Parse the complex stats JSON from CSV"""
        try:
            stats = json.loads(stats_json.replace("'", '"'))
            parsed = {}
            
            for stat in stats:
                stat_name = stat['identifier']
                parsed[stat_name] = {}
                
                # Home team stats
                for entry in stat.get('h', []):
                    element_id = entry['element']
                    value = entry['value']
                    if element_id not in parsed[stat_name]:
                        parsed[stat_name][element_id] = 0
                    parsed[stat_name][element_id] += value
                
                # Away team stats  
                for entry in stat.get('a', []):
                    element_id = entry['element']
                    value = entry['value']
                    if element_id not in parsed[stat_name]:
                        parsed[stat_name][element_id] = 0
                    parsed[stat_name][element_id] += value
                        
            return parsed
        except Exception as e:
            logger.error(f"Error parsing stats JSON: {e}")
            return {}
    
    def import_fixtures_csv(self, csv_path: str, season_id: str = "2024-25"):
        """Import historical fixture data from CSV"""
        logger.info(f"Importing fixtures from {csv_path} for season {season_id}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add season if not exists
        cursor.execute('''
            INSERT OR IGNORE INTO seasons (season_id, start_date, end_date, total_gameweeks)
            VALUES (?, ?, ?, ?)
        ''', (season_id, "2024-08-15", "2025-05-25", 38))
        
        matches_imported = 0
        performances_imported = 0
        
        try:
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                match_id = int(row['id'])
                gameweek = int(row['event'])
                home_team = int(row['team_h'])
                away_team = int(row['team_a'])
                home_score = int(row['team_h_score']) if pd.notna(row['team_h_score']) else 0
                away_score = int(row['team_a_score']) if pd.notna(row['team_a_score']) else 0
                finished = row['finished'] == True
                kickoff_time = row['kickoff_time']
                stats_json = row['stats']
                
                # Insert match
                cursor.execute('''
                    INSERT OR REPLACE INTO matches 
                    (match_id, season_id, gameweek, kickoff_time, home_team_id, away_team_id, 
                     home_score, away_score, finished)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (match_id, season_id, gameweek, kickoff_time, home_team, away_team,
                      home_score, away_score, finished))
                matches_imported += 1
                
                # Parse and insert player performances
                if finished and stats_json:
                    stats = self.parse_stats_json(stats_json)
                    
                    # Get all players who participated
                    all_players = set()
                    for stat_name, players in stats.items():
                        all_players.update(players.keys())
                    
                    for element_id in all_players:
                        # Determine team and opponent
                        team_id = home_team  # Default assumption
                        opponent_id = away_team
                        home_away = 'home'
                        
                        # Extract stats for this player
                        goals = stats.get('goals_scored', {}).get(element_id, 0)
                        assists = stats.get('assists', {}).get(element_id, 0)
                        saves = stats.get('saves', {}).get(element_id, 0)
                        yellow_cards = stats.get('yellow_cards', {}).get(element_id, 0)
                        red_cards = stats.get('red_cards', {}).get(element_id, 0)
                        bonus = stats.get('bonus', {}).get(element_id, 0)
                        bps = stats.get('bps', {}).get(element_id, 0)
                        
                        # Calculate basic FPL points
                        points = self.calculate_fpl_points(
                            goals, assists, saves, yellow_cards, red_cards, bonus,
                            is_goalkeeper=saves > 0  # Simple heuristic
                        )
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO player_performances
                            (match_id, season_id, gameweek, element_id, team_id, opponent_id,
                             home_away, goals, assists, saves, yellow_cards, red_cards,
                             bonus, bps, final_points)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (match_id, season_id, gameweek, element_id, team_id, opponent_id,
                              home_away, goals, assists, saves, yellow_cards, red_cards,
                              bonus, bps, points))
                        performances_imported += 1
            
            conn.commit()
            logger.info(f"Imported {matches_imported} matches and {performances_imported} player performances")
            
        except Exception as e:
            logger.error(f"Error importing CSV: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def calculate_fpl_points(self, goals: int, assists: int, saves: int, 
                           yellow_cards: int, red_cards: int, bonus: int,
                           is_goalkeeper: bool = False) -> int:
        """Calculate FPL points based on performance"""
        points = 0
        
        # Goals (position dependent)
        if is_goalkeeper or saves > 0:  # Goalkeeper
            points += goals * 6
        elif goals > 0:  # Assume defender if saves=0 and goals>0 (simplified)
            points += goals * 6  # Conservative assumption
        else:  # Midfielder/Forward
            points += goals * 5  # Conservative assumption
            
        # Assists
        points += assists * 3
        
        # Saves (goalkeepers only)
        points += (saves // 3) * 1  # 1 point per 3 saves
        
        # Cards
        points -= yellow_cards * 1
        points -= red_cards * 3
        
        # Bonus
        points += bonus
        
        # Appearance (assume 2 points for playing)
        if goals > 0 or assists > 0 or saves > 0 or yellow_cards > 0 or red_cards > 0:
            points += 2
        
        return max(0, points)
    
    def get_player_historical_stats(self, element_id: int, season_id: str = None, 
                                  last_n_games: int = None) -> Dict[str, Any]:
        """Get historical performance stats for a player"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM player_performances 
            WHERE element_id = ?
        '''
        params = [element_id]
        
        if season_id:
            query += ' AND season_id = ?'
            params.append(season_id)
            
        query += ' ORDER BY gameweek DESC'
        
        if last_n_games:
            query += f' LIMIT {last_n_games}'
        
        cursor.execute(query, params)
        performances = cursor.fetchall()
        
        if not performances:
            conn.close()
            return {}
        
        # Calculate aggregated stats
        total_games = len(performances)
        total_points = sum(p[18] for p in performances)  # final_points column
        total_goals = sum(p[9] for p in performances)
        total_assists = sum(p[10] for p in performances)
        avg_points = total_points / total_games if total_games > 0 else 0
        
        conn.close()
        
        return {
            'element_id': element_id,
            'games_played': total_games,
            'total_points': total_points,
            'average_points': round(avg_points, 2),
            'total_goals': total_goals,
            'total_assists': total_assists,
            'recent_form': [p[18] for p in performances[:5]],  # Last 5 games
            'last_updated': datetime.now().isoformat()
        }
    
    def get_team_vs_team_history(self, team1_id: int, team2_id: int, 
                                last_n_meetings: int = 6) -> List[Dict]:
        """Get head-to-head history between two teams"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM matches 
            WHERE ((home_team_id = ? AND away_team_id = ?) 
                   OR (home_team_id = ? AND away_team_id = ?))
            AND finished = 1
            ORDER BY kickoff_time DESC
            LIMIT ?
        ''', (team1_id, team2_id, team2_id, team1_id, last_n_meetings))
        
        matches = cursor.fetchall()
        conn.close()
        
        history = []
        for match in matches:
            history.append({
                'gameweek': match[2],
                'kickoff_time': match[3],
                'home_team': match[4],
                'away_team': match[5], 
                'home_score': match[6],
                'away_score': match[7],
                'winner': 'home' if match[6] > match[7] else 'away' if match[7] > match[6] else 'draw'
            })
        
        return history
    
    def update_current_season_data(self, fpl_manager, season_id: str = "2025-26"):
        """Update database with current season's live data from FPL API"""
        logger.info(f"Updating current season data for {season_id}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Add current season if not exists
            cursor.execute('''
                INSERT OR IGNORE INTO seasons (season_id, start_date, end_date, total_gameweeks)
                VALUES (?, ?, ?, ?)
            ''', (season_id, "2025-08-15", "2026-05-25", 38))
            
            # Get current FPL data
            bootstrap_data = fpl_manager.fetch_bootstrap_data()
            if not bootstrap_data:
                logger.error("Failed to fetch current FPL data")
                return
            
            # Update players data for current season
            players_data = bootstrap_data.get('elements', [])
            teams_data = bootstrap_data.get('teams', [])
            
            # Update teams table
            for team in teams_data:
                cursor.execute('''
                    INSERT OR REPLACE INTO teams (team_id, season_id, name, short_name)
                    VALUES (?, ?, ?, ?)
                ''', (team['id'], season_id, team['name'], team['short_name']))
            
            # Update players table  
            for player in players_data:
                cursor.execute('''
                    INSERT OR REPLACE INTO players 
                    (element_id, season_id, web_name, first_name, second_name, 
                     team_id, position, cost_start, cost_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (player['id'], season_id, player['web_name'], 
                      player['first_name'], player['second_name'],
                      player['team'], player['element_type'],
                      player['now_cost'] / 10.0, player['now_cost'] / 10.0))
            
            # Get and update fixtures/results
            fixtures_data = fpl_manager.fetch_fixtures()
            if fixtures_data:
                matches_updated = 0
                performances_updated = 0
                
                for fixture in fixtures_data:
                    if fixture.get('finished', False):
                        # Add finished match
                        cursor.execute('''
                            INSERT OR REPLACE INTO matches
                            (match_id, season_id, gameweek, kickoff_time, home_team_id, 
                             away_team_id, home_score, away_score, finished)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (fixture['id'], season_id, fixture['event'],
                              fixture['kickoff_time'], fixture['team_h'], fixture['team_a'],
                              fixture.get('team_h_score', 0), fixture.get('team_a_score', 0), True))
                        matches_updated += 1
                        
                        # Process player performances from stats if available
                        if fixture.get('stats'):
                            stats = self.parse_stats_json(str(fixture['stats']))
                            
                            # Get all players who participated
                            all_players = set()
                            for stat_name, players in stats.items():
                                all_players.update(players.keys())
                            
                            for element_id in all_players:
                                # Extract performance stats
                                goals = stats.get('goals_scored', {}).get(element_id, 0)
                                assists = stats.get('assists', {}).get(element_id, 0)
                                saves = stats.get('saves', {}).get(element_id, 0)
                                yellow_cards = stats.get('yellow_cards', {}).get(element_id, 0)
                                red_cards = stats.get('red_cards', {}).get(element_id, 0)
                                bonus = stats.get('bonus', {}).get(element_id, 0)
                                bps = stats.get('bps', {}).get(element_id, 0)
                                
                                # Determine team (simplified)
                                team_id = fixture['team_h']  # Default assumption
                                opponent_id = fixture['team_a']
                                home_away = 'home'
                                
                                # Calculate points
                                points = self.calculate_fpl_points(
                                    goals, assists, saves, yellow_cards, red_cards, bonus,
                                    is_goalkeeper=saves > 0
                                )
                                
                                cursor.execute('''
                                    INSERT OR REPLACE INTO player_performances
                                    (match_id, season_id, gameweek, element_id, team_id, opponent_id,
                                     home_away, goals, assists, saves, yellow_cards, red_cards,
                                     bonus, bps, final_points)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (fixture['id'], season_id, fixture['event'], element_id,
                                      team_id, opponent_id, home_away, goals, assists, saves,
                                      yellow_cards, red_cards, bonus, bps, points))
                                performances_updated += 1
                
                logger.info(f"Updated {matches_updated} matches and {performances_updated} performances for current season")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating current season data: {e}")
            if conn:
                conn.rollback()
                conn.close()
    
    def setup_automated_collection(self, fpl_manager, collection_interval_hours: int = 6):
        """Setup automated collection of current season data"""
        import threading
        import time
        
        def collection_worker():
            while True:
                try:
                    logger.info("Starting automated data collection")
                    self.update_current_season_data(fpl_manager)
                    logger.info(f"Automated collection complete. Next run in {collection_interval_hours} hours")
                    time.sleep(collection_interval_hours * 3600)  # Convert hours to seconds
                except Exception as e:
                    logger.error(f"Automated collection failed: {e}")
                    time.sleep(1800)  # Wait 30 minutes before retry
        
        # Start background thread
        collection_thread = threading.Thread(target=collection_worker, daemon=True)
        collection_thread.start()
        logger.info(f"Automated data collection started (every {collection_interval_hours} hours)")
    
    def get_enhanced_player_features(self, element_id: int) -> Dict[str, Any]:
        """Get enhanced features for ML model training"""
        historical_stats = self.get_player_historical_stats(element_id)
        
        if not historical_stats:
            return {}
        
        # Add advanced metrics
        features = historical_stats.copy()
        
        # Form metrics
        recent_form = features.get('recent_form', [])
        if len(recent_form) >= 3:
            features['form_last_3'] = sum(recent_form[:3]) / 3
            features['form_trend'] = recent_form[0] - recent_form[-1] if len(recent_form) > 1 else 0
        
        # Consistency metrics
        if len(recent_form) >= 5:
            import statistics
            features['form_variance'] = statistics.variance(recent_form)
        
        return features
    
    def import_vaastav_data(self, season_id: str = "2024-25"):
        """Import comprehensive data from vaastav FPL repository"""
        logger.info(f"Importing vaastav FPL data for season {season_id}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Add enhanced player performances table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS enhanced_player_performances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    season_id TEXT,
                    gameweek INTEGER,
                    element_id INTEGER,
                    name TEXT,
                    position TEXT,
                    team TEXT,
                    minutes INTEGER DEFAULT 0,
                    goals_scored INTEGER DEFAULT 0,
                    assists INTEGER DEFAULT 0,
                    expected_goals REAL DEFAULT 0.0,
                    expected_assists REAL DEFAULT 0.0,
                    expected_goal_involvements REAL DEFAULT 0.0,
                    expected_goals_conceded REAL DEFAULT 0.0,
                    goals_conceded INTEGER DEFAULT 0,
                    clean_sheets INTEGER DEFAULT 0,
                    saves INTEGER DEFAULT 0,
                    penalties_missed INTEGER DEFAULT 0,
                    penalties_saved INTEGER DEFAULT 0,
                    yellow_cards INTEGER DEFAULT 0,
                    red_cards INTEGER DEFAULT 0,
                    bonus INTEGER DEFAULT 0,
                    bps INTEGER DEFAULT 0,
                    creativity REAL DEFAULT 0.0,
                    influence REAL DEFAULT 0.0,
                    threat REAL DEFAULT 0.0,
                    ict_index REAL DEFAULT 0.0,
                    total_points INTEGER DEFAULT 0,
                    selected_by_percent REAL DEFAULT 0.0,
                    value REAL DEFAULT 0.0,
                    transfers_in INTEGER DEFAULT 0,
                    transfers_out INTEGER DEFAULT 0,
                    transfers_balance INTEGER DEFAULT 0,
                    was_home BOOLEAN DEFAULT FALSE,
                    opponent_team TEXT,
                    fixture_id INTEGER,
                    kickoff_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (season_id) REFERENCES seasons(season_id)
                )
            ''')
            
            # Create indexes for enhanced data
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_enhanced_performances_element ON enhanced_player_performances(element_id, season_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_enhanced_performances_gameweek ON enhanced_player_performances(gameweek, season_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_enhanced_performances_team ON enhanced_player_performances(team, season_id)')
            
            conn.commit()
            logger.info("Enhanced player performances table created")
            
        except Exception as e:
            logger.error(f"Error setting up enhanced tables: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def import_vaastav_gameweek_data(self, gameweek: int, season_id: str = "2024-25"):
        """Import specific gameweek data from vaastav repository"""
        import requests
        
        logger.info(f"Importing gameweek {gameweek} data for season {season_id}")
        
        # Download gameweek data
        url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season_id}/gws/gw{gameweek}.csv"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            temp_file = f"temp_gw{gameweek}_{season_id}.csv"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Import the data
            df = pd.read_csv(temp_file)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            performances_imported = 0
            
            for _, row in df.iterrows():
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO enhanced_player_performances
                        (season_id, gameweek, element_id, name, position, team, minutes,
                         goals_scored, assists, expected_goals, expected_assists, 
                         expected_goal_involvements, expected_goals_conceded, goals_conceded,
                         clean_sheets, saves, penalties_missed, penalties_saved,
                         yellow_cards, red_cards, bonus, bps, creativity, influence, threat,
                         ict_index, total_points, selected_by_percent, value, transfers_in,
                         transfers_out, transfers_balance, was_home, opponent_team, 
                         fixture_id, kickoff_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        season_id, gameweek, row.get('element', 0), row.get('name', ''),
                        row.get('position', ''), row.get('team', ''), row.get('minutes', 0),
                        row.get('goals_scored', 0), row.get('assists', 0), 
                        row.get('expected_goals', 0.0), row.get('expected_assists', 0.0),
                        row.get('expected_goal_involvements', 0.0), row.get('expected_goals_conceded', 0.0),
                        row.get('goals_conceded', 0), row.get('clean_sheets', 0), row.get('saves', 0),
                        row.get('penalties_missed', 0), row.get('penalties_saved', 0),
                        row.get('yellow_cards', 0), row.get('red_cards', 0), row.get('bonus', 0),
                        row.get('bps', 0), row.get('creativity', 0.0), row.get('influence', 0.0),
                        row.get('threat', 0.0), row.get('ict_index', 0.0), row.get('total_points', 0),
                        row.get('selected_by_percent', 0.0), row.get('value', 0.0),
                        row.get('transfers_in', 0), row.get('transfers_out', 0),
                        row.get('transfers_balance', 0), row.get('was_home', False),
                        row.get('opponent_team', ''), row.get('fixture', 0), row.get('kickoff_time', '')
                    ))
                    performances_imported += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to import row for gameweek {gameweek}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)
            
            logger.info(f"Imported {performances_imported} enhanced performances for gameweek {gameweek}")
            return performances_imported
            
        except Exception as e:
            logger.error(f"Error importing gameweek {gameweek} data: {e}")
            return 0
    
    def import_all_vaastav_gameweeks(self, season_id: str = "2024-25", max_gameweek: int = 38):
        """Import all available gameweek data from vaastav repository"""
        logger.info(f"Importing all gameweeks for season {season_id}")
        
        # First setup the enhanced tables
        self.import_vaastav_data(season_id)
        
        total_imported = 0
        
        for gw in range(1, max_gameweek + 1):
            try:
                imported = self.import_vaastav_gameweek_data(gw, season_id)
                total_imported += imported
                logger.info(f"Gameweek {gw}: {imported} performances imported")
                
                # Small delay to be respectful to GitHub
                import time
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Failed to import gameweek {gw}: {e}")
                continue
        
        logger.info(f"Total enhanced performances imported: {total_imported}")
        return total_imported
    
    def get_enhanced_player_features(self, element_id: int, season_id: str = None, last_n_games: int = None) -> Dict[str, Any]:
        """Get enhanced features from vaastav data for ML model training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM enhanced_player_performances 
            WHERE element_id = ?
        '''
        params = [element_id]
        
        if season_id:
            query += ' AND season_id = ?'
            params.append(season_id)
            
        query += ' ORDER BY gameweek DESC'
        
        if last_n_games:
            query += f' LIMIT {last_n_games}'
        
        cursor.execute(query, params)
        performances = cursor.fetchall()
        
        if not performances:
            conn.close()
            return {}
        
        # Calculate advanced features
        total_games = len(performances)
        
        # Helper function to safely convert to number
        def safe_num(val, default=0):
            try:
                return float(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        def safe_int(val, default=0):
            try:
                return int(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        # Basic aggregations
        total_points = sum(safe_int(p[27]) for p in performances)  # total_points
        total_goals = sum(safe_int(p[7]) for p in performances)     # goals_scored
        total_assists = sum(safe_int(p[8]) for p in performances)   # assists
        total_minutes = sum(safe_int(p[6]) for p in performances)   # minutes
        
        # Advanced metrics
        avg_expected_goals = sum(safe_num(p[9]) for p in performances) / total_games if total_games > 0 else 0  # expected_goals
        avg_expected_assists = sum(safe_num(p[10]) for p in performances) / total_games if total_games > 0 else 0  # expected_assists
        avg_creativity = sum(safe_num(p[23]) for p in performances) / total_games if total_games > 0 else 0  # creativity
        avg_influence = sum(safe_num(p[24]) for p in performances) / total_games if total_games > 0 else 0  # influence
        avg_threat = sum(safe_num(p[25]) for p in performances) / total_games if total_games > 0 else 0  # threat
        avg_ict_index = sum(safe_num(p[26]) for p in performances) / total_games if total_games > 0 else 0  # ict_index
        
        # Form metrics
        recent_points = [safe_int(p[27]) for p in performances[:5]]  # Last 5 games total_points
        recent_form = sum(recent_points) / len(recent_points) if recent_points else 0
        
        # Consistency
        if len(recent_points) >= 3:
            import statistics
            points_variance = statistics.variance(recent_points) if len(recent_points) > 1 else 0
        else:
            points_variance = 0
        
        # Home/Away performance
        home_games = [p for p in performances if p[33]]  # was_home
        away_games = [p for p in performances if not p[33]]
        
        home_avg_points = sum(safe_int(p[27]) for p in home_games) / len(home_games) if home_games else 0
        away_avg_points = sum(safe_int(p[27]) for p in away_games) / len(away_games) if away_games else 0
        home_away_diff = home_avg_points - away_avg_points
        
        conn.close()
        
        return {
            'element_id': element_id,
            'games_played': total_games,
            'total_points': total_points,
            'average_points': round(total_points / total_games if total_games > 0 else 0, 2),
            'total_goals': total_goals,
            'total_assists': total_assists,
            'total_minutes': total_minutes,
            'avg_expected_goals': round(avg_expected_goals, 3),
            'avg_expected_assists': round(avg_expected_assists, 3),
            'avg_creativity': round(avg_creativity, 2),
            'avg_influence': round(avg_influence, 2),
            'avg_threat': round(avg_threat, 2),
            'avg_ict_index': round(avg_ict_index, 2),
            'recent_form': round(recent_form, 2),
            'recent_points': recent_points,
            'points_variance': round(points_variance, 2),
            'home_avg_points': round(home_avg_points, 2),
            'away_avg_points': round(away_avg_points, 2),
            'home_away_diff': round(home_away_diff, 2),
            'last_updated': datetime.now().isoformat()
        }

    def export_training_data(self, season_id: str = None, output_path: str = None) -> str:
        """Export historical data for ML model training"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                pp.*,
                m.home_score,
                m.away_score,
                (CASE WHEN pp.team_id = m.home_team_id THEN m.home_score ELSE m.away_score END) as team_score,
                (CASE WHEN pp.team_id = m.home_team_id THEN m.away_score ELSE m.home_score END) as opponent_score
            FROM player_performances pp
            JOIN matches m ON pp.match_id = m.match_id AND pp.season_id = m.season_id
            WHERE m.finished = 1
        '''
        
        if season_id:
            query += f" AND pp.season_id = '{season_id}'"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if output_path is None:
            output_path = f"fpl_training_data_{datetime.now().strftime('%Y%m%d')}.csv"
        
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} records to {output_path}")
        
        return output_path
    
    def export_enhanced_training_data(self, season_id: str = None, output_path: str = None) -> str:
        """Export enhanced vaastav data for ML model training"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM enhanced_player_performances
            WHERE total_points IS NOT NULL
        '''
        
        if season_id:
            query += f" AND season_id = '{season_id}'"
        
        query += ' ORDER BY gameweek, element_id'
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if output_path is None:
            output_path = f"fpl_enhanced_training_data_{datetime.now().strftime('%Y%m%d')}.csv"
        
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} enhanced records to {output_path}")
        
        return output_path

if __name__ == "__main__":
    # Example usage
    service = HistoricalDataService()
    
    # Import the historical fixtures
    csv_path = "/mnt/c/Users/steev/Downloads/fixtures2024-2025.csv"
    service.import_fixtures_csv(csv_path, "2024-25")
    
    # Example queries
    player_stats = service.get_player_historical_stats(381, "2024-25")  # Salah
    print(f"Salah historical stats: {player_stats}")