#!/usr/bin/env python3
"""
FPL Manager v3 - Real Fixture Service
Handles 2025-26 season fixtures from Vaastav's repository
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import requests
import time

logger = logging.getLogger(__name__)

@dataclass
class RealFixture:
    """Real fixture data model from Vaastav's dataset"""
    code: int
    event: int  # Gameweek number
    finished: bool
    finished_provisional: bool
    id: int
    kickoff_time: str
    minutes: int
    provisional_start_time: bool
    started: bool
    team_a: int  # Away team ID
    team_a_score: Optional[int]
    team_h: int  # Home team ID
    team_h_score: Optional[int]
    stats: List
    team_h_difficulty: int
    team_a_difficulty: int
    pulse_id: int

class RealFixtureService:
    """Service to handle real 2025-26 season fixtures"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or "fpl_fixtures_2025_26.db"
        self.fixtures_csv_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/fixtures.csv"
        self.teams_csv_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/teams.csv"
        self.init_database()
        
        # Team ID mapping (from Vaastav's data structure)
        self.team_names = {}
        self.load_team_names()
        
    def init_database(self):
        """Initialize database for fixtures"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS fixtures_2025_26 (
                    id INTEGER PRIMARY KEY,
                    code INTEGER,
                    event INTEGER,
                    finished BOOLEAN,
                    finished_provisional BOOLEAN,
                    kickoff_time TEXT,
                    minutes INTEGER,
                    provisional_start_time BOOLEAN,
                    started BOOLEAN,
                    team_a INTEGER,
                    team_a_score INTEGER,
                    team_h INTEGER,
                    team_h_score INTEGER,
                    stats TEXT,
                    team_h_difficulty INTEGER,
                    team_a_difficulty INTEGER,
                    pulse_id INTEGER,
                    last_updated REAL
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS teams_2025_26 (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    short_name TEXT,
                    code INTEGER,
                    strength INTEGER,
                    last_updated REAL
                )
                ''')
                
                conn.commit()
                logger.info("Real fixtures database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize fixtures database: {e}")
    
    def load_team_names(self):
        """Load team names from database or fetch fresh data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, name, short_name FROM teams_2025_26')
                teams = cursor.fetchall()
                
                if teams:
                    self.team_names = {team[0]: {'name': team[1], 'short_name': team[2]} for team in teams}
                    logger.info(f"Loaded {len(self.team_names)} team names from cache")
                else:
                    # Fetch fresh team data
                    self.fetch_team_data()
                    
        except Exception as e:
            logger.error(f"Error loading team names: {e}")
            # Use default team mapping if fetch fails
            self.set_default_teams()
    
    def fetch_team_data(self):
        """Fetch fresh team data from Vaastav's repository"""
        try:
            response = requests.get(self.teams_csv_url, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file and load with pandas
            with open('temp_teams.csv', 'w') as f:
                f.write(response.text)
            
            df = pd.read_csv('temp_teams.csv')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for _, row in df.iterrows():
                    cursor.execute('''
                    INSERT OR REPLACE INTO teams_2025_26 
                    (id, name, short_name, code, strength, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        row['id'], row['name'], row['short_name'], 
                        row['code'], row.get('strength', 1000), time.time()
                    ))
                    
                    self.team_names[row['id']] = {
                        'name': row['name'],
                        'short_name': row['short_name']
                    }
                
                conn.commit()
                logger.info(f"Fetched and cached {len(df)} teams")
            
            # Clean up
            Path('temp_teams.csv').unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to fetch team data: {e}")
            self.set_default_teams()
    
    def set_default_teams(self):
        """Set default team names if fetch fails"""
        self.team_names = {
            1: {'name': 'Arsenal', 'short_name': 'ARS'},
            2: {'name': 'Aston Villa', 'short_name': 'AVL'},
            3: {'name': 'Bournemouth', 'short_name': 'BOU'},
            4: {'name': 'Brentford', 'short_name': 'BRE'},
            5: {'name': 'Brighton', 'short_name': 'BHA'},
            6: {'name': 'Chelsea', 'short_name': 'CHE'},
            7: {'name': 'Crystal Palace', 'short_name': 'CRY'},
            8: {'name': 'Everton', 'short_name': 'EVE'},
            9: {'name': 'Fulham', 'short_name': 'FUL'},
            10: {'name': 'Ipswich', 'short_name': 'IPS'},
            11: {'name': 'Leicester', 'short_name': 'LEI'},
            12: {'name': 'Liverpool', 'short_name': 'LIV'},
            13: {'name': 'Man City', 'short_name': 'MCI'},
            14: {'name': 'Man Utd', 'short_name': 'MUN'},
            15: {'name': 'Newcastle', 'short_name': 'NEW'},
            16: {'name': 'Nott\'m Forest', 'short_name': 'NFO'},
            17: {'name': 'Southampton', 'short_name': 'SOU'},
            18: {'name': 'Spurs', 'short_name': 'TOT'},
            19: {'name': 'West Ham', 'short_name': 'WHU'},
            20: {'name': 'Wolves', 'short_name': 'WOL'}
        }
    
    def fetch_fixtures_data(self) -> bool:
        """Fetch and cache the latest fixtures data"""
        try:
            response = requests.get(self.fixtures_csv_url, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file and load with pandas
            with open('temp_fixtures.csv', 'w') as f:
                f.write(response.text)
            
            df = pd.read_csv('temp_fixtures.csv')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear old data
                cursor.execute('DELETE FROM fixtures_2025_26')
                
                for _, row in df.iterrows():
                    # Convert boolean strings to actual booleans
                    finished = row['finished'] if isinstance(row['finished'], bool) else str(row['finished']).lower() == 'true'
                    finished_provisional = row['finished_provisional'] if isinstance(row['finished_provisional'], bool) else str(row['finished_provisional']).lower() == 'true'
                    provisional_start_time = row['provisional_start_time'] if isinstance(row['provisional_start_time'], bool) else str(row['provisional_start_time']).lower() == 'true'
                    started = row['started'] if isinstance(row['started'], bool) else str(row['started']).lower() == 'true'
                    
                    cursor.execute('''
                    INSERT INTO fixtures_2025_26 
                    (id, code, event, finished, finished_provisional, kickoff_time, 
                     minutes, provisional_start_time, started, team_a, team_a_score, 
                     team_h, team_h_score, stats, team_h_difficulty, team_a_difficulty, 
                     pulse_id, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['id'], row['code'], row['event'], 
                        finished, finished_provisional, row['kickoff_time'],
                        row['minutes'], provisional_start_time, started,
                        row['team_a'], row.get('team_a_score'), row['team_h'], row.get('team_h_score'),
                        json.dumps(row.get('stats', [])), row['team_h_difficulty'], 
                        row['team_a_difficulty'], row['pulse_id'], time.time()
                    ))
                
                conn.commit()
                logger.info(f"Cached {len(df)} fixtures from 2025-26 season")
            
            # Clean up
            Path('temp_fixtures.csv').unlink(missing_ok=True)
            return True
            
        except Exception as e:
            logger.error(f"Failed to fetch fixtures: {e}")
            return False
    
    def get_current_gameweek(self) -> int:
        """Determine current gameweek based on dates and fixtures"""
        try:
            now = datetime.now(timezone.utc)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all gameweeks with their earliest kickoff times
                cursor.execute('''
                SELECT event, MIN(kickoff_time) as earliest_kickoff 
                FROM fixtures_2025_26 
                WHERE kickoff_time IS NOT NULL 
                GROUP BY event 
                ORDER BY event
                ''')
                
                gameweeks = cursor.fetchall()
                
                if not gameweeks:
                    logger.warning("No gameweeks found, defaulting to GW1")
                    return 1
                
                current_gw = 1
                for gw, kickoff_str in gameweeks:
                    try:
                        kickoff = datetime.fromisoformat(kickoff_str.replace('Z', '+00:00'))
                        if now >= kickoff:
                            current_gw = gw
                        else:
                            break
                    except:
                        continue
                
                logger.info(f"Current gameweek determined as: {current_gw}")
                return current_gw
                
        except Exception as e:
            logger.error(f"Error determining current gameweek: {e}")
            return 1
    
    def get_fixtures(self, gameweek: int = None, team_id: int = None) -> List[RealFixture]:
        """Get fixtures for specific gameweek or team"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = 'SELECT * FROM fixtures_2025_26 WHERE 1=1'
                params = []
                
                if gameweek:
                    query += ' AND event = ?'
                    params.append(gameweek)
                
                if team_id:
                    query += ' AND (team_h = ? OR team_a = ?)'
                    params.extend([team_id, team_id])
                
                query += ' ORDER BY kickoff_time'
                
                cursor.execute(query, params)
                fixtures = cursor.fetchall()
                
                return [RealFixture(
                    code=f[1], event=f[2], finished=bool(f[3]), finished_provisional=bool(f[4]),
                    id=f[0], kickoff_time=f[5], minutes=f[6], provisional_start_time=bool(f[7]),
                    started=bool(f[8]), team_a=f[9], team_a_score=f[10], team_h=f[11],
                    team_h_score=f[12], stats=json.loads(f[13] or '[]'),
                    team_h_difficulty=f[14], team_a_difficulty=f[15], pulse_id=f[16]
                ) for f in fixtures]
                
        except Exception as e:
            logger.error(f"Error getting fixtures: {e}")
            return []
    
    def get_next_gameweeks(self, current_gw: int, num_gameweeks: int = 2) -> List[int]:
        """Get list of next gameweeks starting from current if not finished, or next if current is finished"""
        # Check if current gameweek has any unfinished fixtures
        current_fixtures = self.get_fixtures(gameweek=current_gw)
        current_has_unfinished = any(not f.finished for f in current_fixtures)
        
        if current_has_unfinished:
            # Include current gameweek if it has unfinished fixtures
            return list(range(current_gw, current_gw + num_gameweeks))
        else:
            # Start from next gameweek if current is finished
            return list(range(current_gw + 1, current_gw + 1 + num_gameweeks))
    
    def get_team_name(self, team_id: int) -> str:
        """Get team name by ID"""
        return self.team_names.get(team_id, {}).get('name', f'Team {team_id}')
    
    def get_team_short_name(self, team_id: int) -> str:
        """Get team short name by ID"""
        return self.team_names.get(team_id, {}).get('short_name', f'T{team_id}')
    
    def refresh_data(self):
        """Refresh fixtures and team data"""
        logger.info("Refreshing fixtures data for 2025-26 season")
        self.fetch_team_data()
        return self.fetch_fixtures_data()
    
    def get_fixtures_summary(self) -> Dict:
        """Get summary of fixtures data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM fixtures_2025_26')
                total_fixtures = cursor.fetchone()[0]
                
                cursor.execute('SELECT MIN(event), MAX(event) FROM fixtures_2025_26')
                gw_range = cursor.fetchone()
                
                cursor.execute('SELECT COUNT(*) FROM fixtures_2025_26 WHERE finished = 1')
                finished_fixtures = cursor.fetchone()[0]
                
                return {
                    'total_fixtures': total_fixtures,
                    'gameweek_range': f"GW{gw_range[0]} - GW{gw_range[1]}" if gw_range[0] else "No data",
                    'finished_fixtures': finished_fixtures,
                    'teams_loaded': len(self.team_names),
                    'current_gameweek': self.get_current_gameweek()
                }
                
        except Exception as e:
            logger.error(f"Error getting fixtures summary: {e}")
            return {}

# Global instance
real_fixture_service = RealFixtureService()