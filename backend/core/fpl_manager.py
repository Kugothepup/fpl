#!/usr/bin/env python3
"""
FPL Manager v3 - Core FPL Data Management and Analysis
Handles real FPL data integration with no fake/mock data
"""

import requests
import pandas as pd
import numpy as np
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import Config

logger = logging.getLogger(__name__)

@dataclass
class Player:
    """Player data model with real FPL attributes"""
    id: int
    name: str
    position: str
    team: str
    cost: float
    total_points: int
    points_per_game: float
    form: float
    goals: int
    assists: int
    clean_sheets: int
    bonus: int
    minutes: int
    saves: int
    yellow_cards: int
    red_cards: int
    fixture_difficulty: float
    selected_by_percent: float
    transfers_in: int
    transfers_out: int
    news: str
    injuries: bool
    
@dataclass
class WeatherData:
    """Weather data model for match impact analysis"""
    location: str
    temperature: float
    conditions: str
    humidity: int
    wind_speed: float
    precipitation: float
    weather_code: int

@dataclass
class Fixture:
    """Fixture data model"""
    id: int
    gameweek: int
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    kickoff_time: str
    difficulty_home: int
    difficulty_away: int
    finished: bool
    home_score: Optional[int] = None
    away_score: Optional[int] = None

class FPLDataCache:
    """SQLite-based caching system for FPL data"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.CACHE_DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize cache database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Players cache table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS players_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp REAL,
                    expires_at REAL
                )
                ''')
                
                # Fixtures cache table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS fixtures_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp REAL,
                    expires_at REAL
                )
                ''')
                
                # General cache table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS general_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp REAL,
                    expires_at REAL
                )
                ''')
                
                conn.commit()
                logger.info("Cache database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize cache database: {e}")
    
    def get(self, key: str, table: str = 'general_cache') -> Optional[dict]:
        """Get cached data if still valid"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                SELECT data, expires_at FROM {table} 
                WHERE cache_key = ? AND expires_at > ?
                ''', (key, time.time()))
                
                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])
                return None
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, data: dict, table: str = 'general_cache', ttl_minutes: int = None):
        """Cache data with TTL"""
        try:
            ttl_minutes = ttl_minutes or Config.CACHE_DURATION_MINUTES
            expires_at = time.time() + (ttl_minutes * 60)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                INSERT OR REPLACE INTO {table} 
                (cache_key, data, timestamp, expires_at)
                VALUES (?, ?, ?, ?)
                ''', (key, json.dumps(data), time.time(), expires_at))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def clear_expired(self):
        """Remove expired cache entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                current_time = time.time()
                
                for table in ['players_cache', 'fixtures_cache', 'general_cache']:
                    cursor.execute(f'DELETE FROM {table} WHERE expires_at < ?', (current_time,))
                
                conn.commit()
                logger.info("Expired cache entries cleared")
                
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

class FPLManager:
    """Main FPL Manager class - handles all real FPL data integration"""
    
    def __init__(self):
        self.base_url = Config.FPL_BASE_URL
        self.team_id = Config.FPL_TEAM_ID
        self.cache = FPLDataCache()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FPL-Manager-v3/1.0'
        })
        
        # Initialize state
        self.current_gameweek = None
        self.season_started = False
        self.teams_data = {}
        
        # Import real fixture service for accurate gameweek detection
        try:
            from services.fixture_service import real_fixture_service
            self.real_fixture_service = real_fixture_service
        except ImportError:
            self.real_fixture_service = None
        
        # Weather integration
        self.weather_api_key = Config.OPENWEATHER_API_KEY
        self.weather_base_url = Config.OPENWEATHER_BASE_URL
        
        logger.info("FPL Manager v3 initialized with real data integration")
    
    def fetch_bootstrap_data(self) -> Optional[Dict]:
        """Fetch main FPL bootstrap data (players, teams, gameweeks)"""
        cache_key = "bootstrap_static"
        
        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data:
            logger.info("Using cached bootstrap data")
            return cached_data
        
        try:
            logger.info("Fetching fresh bootstrap data from FPL API")
            response = self.session.get(f"{self.base_url}/bootstrap-static/", timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Update state from bootstrap data - prefer real fixture service if available
            if self.real_fixture_service:
                self.current_gameweek = self.real_fixture_service.get_current_gameweek()
                self.season_started = True  # 2025-26 season is active
                logger.info(f"Using real fixture service - Current GW: {self.current_gameweek}")
            elif 'events' in data:
                current_event = next((event for event in data['events'] if event['is_current']), None)
                if current_event:
                    self.current_gameweek = current_event['id']
                    self.season_started = current_event['finished'] or any(
                        event['finished'] for event in data['events']
                    )
            
            # Cache teams data for quick lookup
            if 'teams' in data:
                self.teams_data = {team['id']: team for team in data['teams']}
            
            # Cache the data
            self.cache.set(cache_key, data, ttl_minutes=15)
            
            logger.info(f"Bootstrap data fetched successfully. Current GW: {self.current_gameweek}, Season started: {self.season_started}")
            return data
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch bootstrap data: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing bootstrap data: {e}")
            return None
    
    def fetch_fixtures(self) -> List[Fixture]:
        """Fetch fixture data"""
        cache_key = "fixtures_data"
        
        # Check cache first
        cached_data = self.cache.get(cache_key, 'fixtures_cache')
        if cached_data:
            return [Fixture(**fixture) for fixture in cached_data]
        
        try:
            logger.info("Fetching fixtures from FPL API")
            response = self.session.get(f"{self.base_url}/fixtures/", timeout=30)
            response.raise_for_status()
            
            fixtures_data = response.json()
            fixtures = []
            
            for fixture in fixtures_data:
                try:
                    home_team = self.teams_data.get(fixture['team_h'], {}).get('name', f"Team {fixture['team_h']}")
                    away_team = self.teams_data.get(fixture['team_a'], {}).get('name', f"Team {fixture['team_a']}")
                    
                    fixture_obj = Fixture(
                        id=fixture['id'],
                        gameweek=fixture.get('event', 0),
                        home_team=home_team,
                        away_team=away_team,
                        home_team_id=fixture['team_h'],
                        away_team_id=fixture['team_a'],
                        kickoff_time=fixture.get('kickoff_time', ''),
                        difficulty_home=fixture.get('team_h_difficulty', 3),
                        difficulty_away=fixture.get('team_a_difficulty', 3),
                        finished=fixture.get('finished', False),
                        home_score=fixture.get('team_h_score'),
                        away_score=fixture.get('team_a_score')
                    )
                    fixtures.append(fixture_obj)
                    
                except Exception as e:
                    logger.warning(f"Error processing fixture {fixture.get('id', 'unknown')}: {e}")
                    continue
            
            # Cache fixtures data
            self.cache.set(cache_key, [fixture.__dict__ for fixture in fixtures], 'fixtures_cache', ttl_minutes=60)
            
            logger.info(f"Fetched {len(fixtures)} fixtures successfully")
            return fixtures
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch fixtures: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing fixtures: {e}")
            return []
    
    def fetch_user_team(self, team_id: str = None) -> Optional[Dict]:
        """Fetch user's current FPL team"""
        team_id = team_id or self.team_id
        cache_key = f"user_team_{team_id}"
        
        # Check cache first (shorter TTL for team data)
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            logger.info(f"Fetching user team data for team ID: {team_id}")
            
            # Fetch current gameweek picks
            current_gw = self.current_gameweek or 1
            response = self.session.get(f"{self.base_url}/entry/{team_id}/event/{current_gw}/picks/", timeout=30)
            response.raise_for_status()
            
            team_data = response.json()
            
            # Cache team data (shorter TTL since it changes more frequently)
            self.cache.set(cache_key, team_data, ttl_minutes=5)
            
            logger.info(f"Successfully fetched team data with {len(team_data.get('picks', []))} players")
            return team_data
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch user team: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing user team data: {e}")
            return None
    
    def analyze_player_performance(self, player_data: Dict, fixtures: List[Fixture]) -> Player:
        """Analyze individual player performance with real FPL data"""
        try:
            # Map position types
            position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            
            # Calculate fixture difficulty for upcoming games
            player_team_id = player_data.get('team')
            upcoming_fixtures = [
                f for f in fixtures 
                if not f.finished and (f.home_team_id == player_team_id or f.away_team_id == player_team_id)
            ][:5]  # Next 5 fixtures
            
            if upcoming_fixtures:
                fixture_difficulty = sum([
                    f.difficulty_away if f.home_team_id == player_team_id else f.difficulty_home
                    for f in upcoming_fixtures
                ]) / len(upcoming_fixtures)
            else:
                fixture_difficulty = 3.0  # Neutral if no fixtures
            
            # Create Player object with real data
            player = Player(
                id=player_data['id'],
                name=player_data.get('web_name', 'Unknown'),
                position=position_map.get(player_data.get('element_type'), 'UNK'),
                team=self.teams_data.get(player_team_id, {}).get('name', 'Unknown'),
                cost=float(player_data.get('now_cost', 0)) / 10,  # Convert from tenths
                total_points=int(player_data.get('total_points', 0)),
                points_per_game=float(player_data.get('points_per_game', 0)),
                form=float(player_data.get('form', 0)),
                goals=int(player_data.get('goals_scored', 0)),
                assists=int(player_data.get('assists', 0)),
                clean_sheets=int(player_data.get('clean_sheets', 0)),
                bonus=int(player_data.get('bonus', 0)),
                minutes=int(player_data.get('minutes', 0)),
                saves=int(player_data.get('saves', 0)),
                yellow_cards=int(player_data.get('yellow_cards', 0)),
                red_cards=int(player_data.get('red_cards', 0)),
                fixture_difficulty=fixture_difficulty,
                selected_by_percent=float(player_data.get('selected_by_percent', 0)),
                transfers_in=int(player_data.get('transfers_in_event', 0)),
                transfers_out=int(player_data.get('transfers_out_event', 0)),
                news=player_data.get('news', ''),
                injuries=bool(player_data.get('news', '').lower().find('injury') != -1)
            )
            
            return player
            
        except Exception as e:
            logger.error(f"Error analyzing player {player_data.get('id', 'unknown')}: {e}")
            # Return minimal player data on error
            return Player(
                id=player_data.get('id', 0),
                name=player_data.get('web_name', 'Unknown'),
                position='UNK',
                team='Unknown',
                cost=0.0,
                total_points=0,
                points_per_game=0.0,
                form=0.0,
                goals=0,
                assists=0,
                clean_sheets=0,
                bonus=0,
                minutes=0,
                saves=0,
                yellow_cards=0,
                red_cards=0,
                fixture_difficulty=3.0,
                selected_by_percent=0.0,
                transfers_in=0,
                transfers_out=0,
                news='',
                injuries=False
            )
    
    def get_weather_data(self, city: str) -> WeatherData:
        """Fetch real weather data for match impact analysis"""
        if not self.weather_api_key:
            logger.warning("Weather API key not configured")
            return self._get_default_weather(city)
        
        cache_key = f"weather_{city}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return WeatherData(**cached_data)
        
        try:
            url = f"{self.weather_base_url}/weather"
            params = {
                'q': f"{city},GB",
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            weather = WeatherData(
                location=city,
                temperature=float(data['main']['temp']),
                conditions=data['weather'][0]['description'],
                humidity=int(data['main']['humidity']),
                wind_speed=float(data['wind'].get('speed', 0)),
                precipitation=float(data.get('rain', {}).get('1h', 0)),
                weather_code=int(data['weather'][0]['id'])
            )
            
            # Cache weather data for 30 minutes
            self.cache.set(cache_key, weather.__dict__, ttl_minutes=30)
            
            return weather
            
        except Exception as e:
            logger.error(f"Failed to fetch weather for {city}: {e}")
            return self._get_default_weather(city)
    
    def _get_default_weather(self, city: str) -> WeatherData:
        """Return default weather data when API is unavailable"""
        return WeatherData(
            location=city,
            temperature=15.0,
            conditions="Data unavailable",
            humidity=60,
            wind_speed=5.0,
            precipitation=0.0,
            weather_code=800
        )
    
    def fetch_fpl_data(self) -> Optional[Dict]:
        """Main method to fetch all FPL data"""
        return self.fetch_bootstrap_data()
    
    def get_teams_data(self) -> Dict:
        """Get teams data"""
        if not self.teams_data:
            bootstrap_data = self.fetch_bootstrap_data()
            if bootstrap_data and 'teams' in bootstrap_data:
                self.teams_data = {team['id']: team for team in bootstrap_data['teams']}
        
        return self.teams_data
    
    def cleanup_cache(self):
        """Clean expired cache entries"""
        self.cache.clear_expired()
    
    def get_season_status(self) -> Dict:
        """Get current season status"""
        return {
            'current_gameweek': self.current_gameweek,
            'season_started': self.season_started,
            'teams_loaded': len(self.teams_data) > 0
        }