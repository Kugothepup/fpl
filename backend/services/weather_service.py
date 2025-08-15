#!/usr/bin/env python3
"""
FPL Manager v3 - Weather Service
Real weather data integration for match impact analysis
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import Config

logger = logging.getLogger(__name__)

@dataclass
class WeatherImpact:
    """Weather impact analysis for FPL"""
    temperature_impact: str
    wind_impact: str
    precipitation_impact: str
    overall_impact: str
    impact_score: float  # -1 to 1, negative is bad for performance

@dataclass
class MatchWeather:
    """Weather data for a specific match"""
    location: str
    temperature: float
    conditions: str
    humidity: int
    wind_speed: float
    precipitation: float
    weather_code: int
    impact_analysis: WeatherImpact

class WeatherService:
    """Service for fetching and analyzing weather data impact on FPL"""
    
    def __init__(self):
        self.api_key = Config.OPENWEATHER_API_KEY
        self.base_url = Config.OPENWEATHER_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FPL-Manager-v3-Weather/1.0'
        })
        
        # Weather impact thresholds
        self.impact_thresholds = {
            'temperature': {
                'very_cold': 5,    # Below 5°C
                'cold': 10,        # 5-10°C
                'optimal': 20,     # 10-20°C
                'warm': 25,        # 20-25°C
                'hot': 30          # Above 30°C
            },
            'wind': {
                'calm': 10,        # Below 10 km/h
                'moderate': 20,    # 10-20 km/h
                'strong': 35,      # 20-35 km/h
                'very_strong': 50  # Above 50 km/h
            },
            'precipitation': {
                'none': 0,
                'light': 2.5,      # 0-2.5mm
                'moderate': 10,    # 2.5-10mm
                'heavy': 50        # Above 50mm
            }
        }
        
        # Stadium locations for Premier League teams
        self.stadium_locations = {
            'Arsenal': 'London',
            'Aston Villa': 'Birmingham',
            'Bournemouth': 'Bournemouth',
            'Brentford': 'London',
            'Brighton': 'Brighton',
            'Chelsea': 'London',
            'Crystal Palace': 'London',
            'Everton': 'Liverpool',
            'Fulham': 'London',
            'Ipswich': 'Ipswich',
            'Leicester': 'Leicester',
            'Liverpool': 'Liverpool',
            'Man City': 'Manchester',
            'Man Utd': 'Manchester',
            'Newcastle': 'Newcastle',
            'Nottingham Forest': 'Nottingham',
            'Southampton': 'Southampton',
            'Spurs': 'London',
            'West Ham': 'London',
            'Wolves': 'Wolverhampton'
        }
        
        logger.info(f"Weather Service initialized. API available: {bool(self.api_key)}")
    
    def get_current_weather(self, city: str) -> Optional[MatchWeather]:
        """Get current weather for a city"""
        if not self.api_key:
            logger.warning("Weather API key not configured")
            return self._get_default_weather(city)
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                'q': f"{city},GB",
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract weather data
            weather_data = {
                'location': city,
                'temperature': float(data['main']['temp']),
                'conditions': data['weather'][0]['description'],
                'humidity': int(data['main']['humidity']),
                'wind_speed': float(data['wind'].get('speed', 0)) * 3.6,  # Convert m/s to km/h
                'precipitation': float(data.get('rain', {}).get('1h', 0) or data.get('snow', {}).get('1h', 0)),
                'weather_code': int(data['weather'][0]['id'])
            }
            
            # Analyze impact
            impact_analysis = self._analyze_weather_impact(
                weather_data['temperature'],
                weather_data['wind_speed'],
                weather_data['precipitation'],
                weather_data['conditions']
            )
            
            return MatchWeather(
                **weather_data,
                impact_analysis=impact_analysis
            )
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch weather for {city}: {e}")
            return self._get_default_weather(city)
        except Exception as e:
            logger.error(f"Error processing weather data for {city}: {e}")
            return self._get_default_weather(city)
    
    def get_forecast(self, city: str, days: int = 5) -> List[MatchWeather]:
        """Get weather forecast for upcoming days"""
        if not self.api_key:
            return [self._get_default_weather(city) for _ in range(days)]
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'q': f"{city},GB",
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            forecasts = []
            
            # Group by day and take midday forecast (closest to match time)
            daily_forecasts = {}
            for item in data['list']:
                date = datetime.fromtimestamp(item['dt']).date()
                hour = datetime.fromtimestamp(item['dt']).hour
                
                # Prefer forecasts around 15:00 (typical match time)
                if date not in daily_forecasts or abs(hour - 15) < abs(daily_forecasts[date]['hour'] - 15):
                    daily_forecasts[date] = {
                        'data': item,
                        'hour': hour
                    }
            
            # Convert to MatchWeather objects
            for date, forecast_info in sorted(daily_forecasts.items()):
                item = forecast_info['data']
                
                weather_data = {
                    'location': city,
                    'temperature': float(item['main']['temp']),
                    'conditions': item['weather'][0]['description'],
                    'humidity': int(item['main']['humidity']),
                    'wind_speed': float(item['wind'].get('speed', 0)) * 3.6,
                    'precipitation': float(item.get('rain', {}).get('3h', 0) or item.get('snow', {}).get('3h', 0)),
                    'weather_code': int(item['weather'][0]['id'])
                }
                
                impact_analysis = self._analyze_weather_impact(
                    weather_data['temperature'],
                    weather_data['wind_speed'],
                    weather_data['precipitation'],
                    weather_data['conditions']
                )
                
                forecasts.append(MatchWeather(
                    **weather_data,
                    impact_analysis=impact_analysis
                ))
            
            logger.info(f"Retrieved {len(forecasts)} weather forecasts for {city}")
            return forecasts
            
        except Exception as e:
            logger.error(f"Failed to fetch forecast for {city}: {e}")
            return [self._get_default_weather(city) for _ in range(days)]
    
    def get_match_weather(self, home_team: str, away_team: str) -> Optional[MatchWeather]:
        """Get weather for a specific match based on home team's stadium"""
        try:
            # Get stadium location
            stadium_city = self.stadium_locations.get(home_team)
            if not stadium_city:
                logger.warning(f"No stadium location found for {home_team}")
                stadium_city = 'London'  # Default to London
            
            weather = self.get_current_weather(stadium_city)
            
            if weather:
                logger.info(f"Retrieved weather for {home_team} vs {away_team} at {stadium_city}")
            
            return weather
            
        except Exception as e:
            logger.error(f"Failed to get match weather for {home_team} vs {away_team}: {e}")
            return self._get_default_weather('London')
    
    def get_gameweek_weather_summary(self, fixtures: List[Dict]) -> Dict[str, MatchWeather]:
        """Get weather summary for all matches in upcoming gameweek"""
        try:
            weather_summary = {}
            
            for fixture in fixtures:
                home_team = fixture.get('home_team', 'Unknown')
                match_key = f"{home_team}_vs_{fixture.get('away_team', 'Unknown')}"
                
                weather = self.get_match_weather(home_team, fixture.get('away_team'))
                if weather:
                    weather_summary[match_key] = weather
            
            logger.info(f"Generated weather summary for {len(weather_summary)} matches")
            return weather_summary
            
        except Exception as e:
            logger.error(f"Failed to generate gameweek weather summary: {e}")
            return {}
    
    def _analyze_weather_impact(self, temperature: float, wind_speed: float, 
                               precipitation: float, conditions: str) -> WeatherImpact:
        """Analyze how weather conditions impact player performance"""
        try:
            impact_factors = []
            impact_score = 0.0
            
            # Temperature impact
            if temperature < self.impact_thresholds['temperature']['very_cold']:
                temp_impact = "Very cold conditions may reduce player performance and increase injury risk"
                impact_score -= 0.3
            elif temperature < self.impact_thresholds['temperature']['cold']:
                temp_impact = "Cold conditions may slightly affect player performance"
                impact_score -= 0.1
            elif temperature <= self.impact_thresholds['temperature']['optimal']:
                temp_impact = "Optimal temperature for football performance"
                impact_score += 0.1
            elif temperature <= self.impact_thresholds['temperature']['warm']:
                temp_impact = "Warm conditions - generally good for performance"
                impact_score += 0.05
            elif temperature <= self.impact_thresholds['temperature']['hot']:
                temp_impact = "Hot conditions may cause fatigue in later stages"
                impact_score -= 0.1
            else:
                temp_impact = "Very hot conditions likely to significantly impact performance"
                impact_score -= 0.4
            
            # Wind impact
            if wind_speed < self.impact_thresholds['wind']['calm']:
                wind_impact = "Calm conditions ideal for passing and set pieces"
                impact_score += 0.05
            elif wind_speed < self.impact_thresholds['wind']['moderate']:
                wind_impact = "Light wind - minimal impact on play"
                impact_score += 0.02
            elif wind_speed < self.impact_thresholds['wind']['strong']:
                wind_impact = "Moderate wind may affect long passes and crosses"
                impact_score -= 0.1
            elif wind_speed < self.impact_thresholds['wind']['very_strong']:
                wind_impact = "Strong winds will significantly impact ball flight"
                impact_score -= 0.2
            else:
                wind_impact = "Very strong winds may disrupt normal play patterns"
                impact_score -= 0.3
            
            # Precipitation impact
            if precipitation == 0:
                precip_impact = "Dry conditions optimal for technical play"
                impact_score += 0.1
            elif precipitation < self.impact_thresholds['precipitation']['light']:
                precip_impact = "Light rain may make the ball and pitch slightly slippery"
                impact_score -= 0.05
            elif precipitation < self.impact_thresholds['precipitation']['moderate']:
                precip_impact = "Moderate rain will affect ball control and passing accuracy"
                impact_score -= 0.15
            elif precipitation < self.impact_thresholds['precipitation']['heavy']:
                precip_impact = "Heavy rain significantly impacts technical play"
                impact_score -= 0.25
            else:
                precip_impact = "Very heavy rain/snow creates challenging playing conditions"
                impact_score -= 0.4
            
            # Overall impact assessment
            if impact_score > 0.1:
                overall_impact = "Weather conditions favor good football and higher scores"
            elif impact_score > -0.1:
                overall_impact = "Weather conditions are neutral for match quality"
            elif impact_score > -0.3:
                overall_impact = "Weather conditions may negatively impact player performance"
            else:
                overall_impact = "Adverse weather conditions likely to significantly affect the match"
            
            # Clamp impact score
            impact_score = max(-1.0, min(1.0, impact_score))
            
            return WeatherImpact(
                temperature_impact=temp_impact,
                wind_impact=wind_impact,
                precipitation_impact=precip_impact,
                overall_impact=overall_impact,
                impact_score=impact_score
            )
            
        except Exception as e:
            logger.error(f"Weather impact analysis failed: {e}")
            return WeatherImpact(
                temperature_impact="Analysis unavailable",
                wind_impact="Analysis unavailable", 
                precipitation_impact="Analysis unavailable",
                overall_impact="Weather impact analysis unavailable",
                impact_score=0.0
            )
    
    def _get_default_weather(self, city: str) -> MatchWeather:
        """Return default weather data when API is unavailable"""
        default_impact = WeatherImpact(
            temperature_impact="Weather data unavailable",
            wind_impact="Weather data unavailable",
            precipitation_impact="Weather data unavailable", 
            overall_impact="Weather impact analysis unavailable - using default neutral conditions",
            impact_score=0.0
        )
        
        return MatchWeather(
            location=city,
            temperature=15.0,
            conditions="Data unavailable",
            humidity=60,
            wind_speed=10.0,
            precipitation=0.0,
            weather_code=800,
            impact_analysis=default_impact
        )
    
    def get_weather_alerts(self, cities: List[str]) -> List[Dict]:
        """Get weather alerts that might impact matches"""
        try:
            alerts = []
            
            for city in cities:
                weather = self.get_current_weather(city)
                if weather and weather.impact_analysis.impact_score < -0.2:
                    alerts.append({
                        'city': city,
                        'alert_type': 'performance_impact',
                        'severity': 'high' if weather.impact_analysis.impact_score < -0.3 else 'medium',
                        'message': weather.impact_analysis.overall_impact,
                        'temperature': weather.temperature,
                        'conditions': weather.conditions,
                        'wind_speed': weather.wind_speed,
                        'precipitation': weather.precipitation
                    })
            
            logger.info(f"Generated {len(alerts)} weather alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to generate weather alerts: {e}")
            return []