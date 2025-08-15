#!/usr/bin/env python3
"""
FPL Manager v3 Configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 5000))
    
    # FPL API Configuration
    FPL_BASE_URL = 'https://fantasy.premierleague.com/api'
    FPL_TEAM_ID = os.getenv('FPL_TEAM_ID', '3387577')  # Your FPL team ID
    
    # External API Keys (Real Data Sources)
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
    PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    
    # API URLs
    OPENWEATHER_BASE_URL = 'https://api.openweathermap.org/data/2.5'
    PERPLEXITY_API_URL = 'https://api.perplexity.ai/chat/completions'
    MISTRAL_API_URL = 'https://api.mistral.ai/v1/chat/completions'
    
    # Model Configuration
    MISTRAL_MODEL = os.getenv('MISTRAL_MODEL', 'mistral-large-latest')
    
    # Database Configuration
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'fpl_manager_v3.db')
    CACHE_DATABASE_PATH = os.getenv('CACHE_DATABASE_PATH', 'fpl_cache_v3.db')
    PREDICTIONS_DATABASE_PATH = os.getenv('PREDICTIONS_DATABASE_PATH', 'fpl_predictions_v3.db')
    
    # ML Model Configuration
    ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', 'models/')
    ENABLE_ENHANCED_ML = os.getenv('ENABLE_ENHANCED_ML', 'True').lower() == 'true'
    
    # Cache Configuration
    CACHE_DURATION_MINUTES = int(os.getenv('CACHE_DURATION_MINUTES', 30))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'fpl_manager_v3.log')
    
    # Feature Flags
    ENABLE_NEWS_INTEGRATION = os.getenv('ENABLE_NEWS_INTEGRATION', 'True').lower() == 'true'
    ENABLE_WEATHER_INTEGRATION = os.getenv('ENABLE_WEATHER_INTEGRATION', 'True').lower() == 'true'
    ENABLE_MANAGER_AI = os.getenv('ENABLE_MANAGER_AI', 'True').lower() == 'true'
    ENABLE_ACCURACY_TRACKING = os.getenv('ENABLE_ACCURACY_TRACKING', 'True').lower() == 'true'
    
    # Rate Limiting
    API_RATE_LIMIT_PER_MINUTE = int(os.getenv('API_RATE_LIMIT_PER_MINUTE', 60))
    
    # Premier League Teams (for reference)
    PREMIER_LEAGUE_TEAMS = {
        1: 'Arsenal', 2: 'Aston Villa', 3: 'Bournemouth', 4: 'Brentford',
        5: 'Brighton', 6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton',
        9: 'Fulham', 10: 'Ipswich', 11: 'Leicester', 12: 'Liverpool',
        13: 'Man City', 14: 'Man Utd', 15: 'Newcastle', 16: 'Nottingham Forest',
        17: 'Southampton', 18: 'Spurs', 19: 'West Ham', 20: 'Wolves'
    }
    
    # Major UK Cities for Weather Data
    WEATHER_CITIES = [
        'London', 'Manchester', 'Liverpool', 'Birmingham', 
        'Newcastle', 'Leeds', 'Leicester', 'Brighton'
    ]

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set in production

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DATABASE_PATH = ':memory:'  # In-memory database for tests

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}