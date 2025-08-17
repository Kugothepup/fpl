"""
Age Performance Analysis Service
Analyzes relationship between player age and performance metrics using regression models
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
import json
import sqlite3
import os
from mistralai import Mistral
from scipy import stats
import requests

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PlayerAgeData:
    """Player data with age information"""
    player_id: int
    name: str
    position: str
    team: str
    age: float
    birth_date: Optional[str]
    points: float
    goals: int
    assists: int
    minutes: int
    games_played: int
    form: float
    points_per_game: float
    xg: float = 0.0
    xa: float = 0.0
    clean_sheets: int = 0
    saves: int = 0
    season: str = "2024-25"

@dataclass
class RegressionResult:
    """Result of regression analysis"""
    model_type: str
    r2_score: float
    mae: float
    coefficients: List[float]
    intercept: float
    feature_names: List[str]
    predictions: List[float]
    ages: List[float]
    actual_values: List[float]
    confidence_interval: Optional[Dict[str, List[float]]] = None
    p_value: Optional[float] = None
    correlation: Optional[float] = None
    is_significant: Optional[bool] = None

@dataclass
class EnrichedPlayerData:
    """Player data enriched with online information"""
    player_id: int
    name: str
    enriched_age: Optional[float]
    birth_date: Optional[str]
    nationality: Optional[str]
    injury_status: Optional[str]
    last_updated: str
    data_source: str
    confidence: float

@dataclass
class AIAnalysisSummary:
    """AI-generated analysis summary"""
    overall_summary: str
    key_findings: List[str]
    age_insights: List[str]
    performance_trends: str
    recommendations: List[str]
    statistical_interpretation: str
    confidence_level: str
    generated_at: str

@dataclass
class AgeAnalysisResult:
    """Complete age analysis for a metric"""
    metric: str
    position_filter: Optional[str]
    linear_model: RegressionResult
    polynomial_model: RegressionResult
    best_model: str
    peak_age: float
    age_range_analysis: Dict[str, Any]
    age_groups_analysis: Dict[str, Any]
    player_comparisons: List[Dict[str, Any]]
    insights: List[str]
    ai_summary: Optional[AIAnalysisSummary] = None
    enriched_data_count: int = 0

class AgePerformanceService:
    """Service for analyzing age vs performance relationships with AI enrichment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.age_data_cache = {}
        self.analysis_cache = {}
        
        # Initialize Mistral AI client
        self.mistral_api_key = os.getenv('MISTRAL_API_KEY')
        self.mistral_model = os.getenv('MISTRAL_MODEL', 'mistral-large-latest')
        self.mistral_client = None
        if self.mistral_api_key:
            try:
                self.mistral_client = Mistral(api_key=self.mistral_api_key)
                self.logger.info(f"Mistral AI client initialized successfully with model: {self.mistral_model}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Mistral AI client: {e}")
        else:
            self.logger.warning("MISTRAL_API_KEY not found. AI enrichment features will be disabled.")
        
        # Initialize player data database
        self.db_path = "player_enrichment_data.db"
        self._init_enrichment_database()
    
    def _init_enrichment_database(self):
        """Initialize SQLite database for storing enriched player data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS player_enrichment (
                    player_id INTEGER PRIMARY KEY,
                    name TEXT,
                    enriched_age REAL,
                    birth_date TEXT,
                    nationality TEXT,
                    injury_status TEXT,
                    last_updated TEXT,
                    data_source TEXT,
                    confidence REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Player enrichment database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enrichment database: {e}")
    
    def _get_enriched_player_data(self, player_id: int) -> Optional[EnrichedPlayerData]:
        """Get enriched data for a player from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM player_enrichment WHERE player_id = ?
            ''', (player_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return EnrichedPlayerData(
                    player_id=row[0],
                    name=row[1],
                    enriched_age=row[2],
                    birth_date=row[3],
                    nationality=row[4],
                    injury_status=row[5],
                    last_updated=row[6],
                    data_source=row[7],
                    confidence=row[8]
                )
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving enriched data for player {player_id}: {e}")
            return None
    
    def _save_enriched_player_data(self, enriched_data: EnrichedPlayerData):
        """Save enriched player data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO player_enrichment 
                (player_id, name, enriched_age, birth_date, nationality, injury_status, 
                 last_updated, data_source, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                enriched_data.player_id,
                enriched_data.name,
                enriched_data.enriched_age,
                enriched_data.birth_date,
                enriched_data.nationality,
                enriched_data.injury_status,
                enriched_data.last_updated,
                enriched_data.data_source,
                enriched_data.confidence
            ))
            
            conn.commit()
            conn.close()
            self.logger.info(f"Saved enriched data for player {enriched_data.name}")
            
        except Exception as e:
            self.logger.error(f"Error saving enriched data: {e}")
    
    async def enrich_player_data_with_ai(self, player_name: str, team_name: str, player_id: int) -> Optional[EnrichedPlayerData]:
        """Use Mistral AI to search for and enrich player data"""
        if not self.mistral_client:
            return None
        
        try:
            # Check if we have recent data
            existing_data = self._get_enriched_player_data(player_id)
            if existing_data:
                # Check if data is less than 7 days old
                last_updated = datetime.fromisoformat(existing_data.last_updated)
                if (datetime.now() - last_updated).days < 7:
                    return existing_data
            
            # Create search prompt for Mistral AI with current date and web search instructions
            current_date = datetime.now().strftime("%Y-%m-%d")
            search_prompt = f"""
            You are a football data research assistant with web search capabilities. 
            
            CURRENT DATE: {current_date}
            
            Please use your web search capabilities to find current information about this football player:
            
            Player Name: {player_name}
            Team: {team_name}
            
            Search multiple reliable sources including:
            - Official club websites
            - Transfermarkt.com
            - Premier League official site
            - BBC Sport, Sky Sports
            - FBref.com
            - ESPN Football
            
            I need you to find and return the following information in a structured JSON format:
            1. Date of birth (format: YYYY-MM-DD)
            2. Current age (calculated from birth date to {current_date})
            3. Nationality
            4. Current injury status as of {current_date}
            5. Confidence level of the information (0-1 scale)
            
            Please return the information in this exact JSON format:
            {{
                "birth_date": "YYYY-MM-DD or null if unknown",
                "age": number or null,
                "nationality": "country name or null",
                "injury_status": "current injury status or 'Fit' or null",
                "confidence": 0.95,
                "data_source": "specific sources used for verification"
            }}
            
            IMPORTANT:
            - Use web search to find the most current and accurate information
            - Cross-reference multiple sources for accuracy
            - For injury status, search for the most recent team news and medical updates
            - If you cannot find reliable information, set values to null and explain in data_source field
            - Only provide information you are confident about after web verification
            """
            
            # Make API call to Mistral
            messages = [{"role": "user", "content": search_prompt}]
            
            chat_response = self.mistral_client.chat.complete(
                model=self.mistral_model,
                messages=messages,
                max_tokens=500,
                temperature=0.1
            )
            
            response_content = chat_response.choices[0].message.content
            
            # Parse the JSON response
            try:
                # Extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    ai_data = json.loads(json_match.group())
                    
                    # Calculate age from birth date if provided
                    enriched_age = ai_data.get('age')
                    birth_date = ai_data.get('birth_date')
                    
                    if birth_date and birth_date != 'null' and not enriched_age:
                        enriched_age = self.calculate_age_from_birth_date(birth_date)
                    
                    # Create enriched data object
                    enriched_data = EnrichedPlayerData(
                        player_id=player_id,
                        name=player_name,
                        enriched_age=enriched_age,
                        birth_date=birth_date if birth_date != 'null' else None,
                        nationality=ai_data.get('nationality') if ai_data.get('nationality') != 'null' else None,
                        injury_status=ai_data.get('injury_status') if ai_data.get('injury_status') != 'null' else None,
                        last_updated=datetime.now().isoformat(),
                        data_source=f"Mistral AI: {ai_data.get('data_source', 'Online search')}",
                        confidence=float(ai_data.get('confidence', 0.5))
                    )
                    
                    # Save to database
                    self._save_enriched_player_data(enriched_data)
                    
                    return enriched_data
                    
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse AI response as JSON: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error enriching player data with AI: {e}")
            return None
    
    async def generate_ai_analysis_summary(self, analysis_result: 'AgeAnalysisResult') -> Optional[AIAnalysisSummary]:
        """Generate AI-powered analysis summary using Mistral"""
        if not self.mistral_client:
            return None
        
        try:
            # Prepare analysis data for AI
            analysis_data = {
                'metric': analysis_result.metric,
                'position_filter': analysis_result.position_filter,
                'sample_size': len(analysis_result.linear_model.ages),
                'linear_r2': analysis_result.linear_model.r2_score,
                'polynomial_r2': analysis_result.polynomial_model.r2_score,
                'best_model': analysis_result.best_model,
                'peak_age': analysis_result.peak_age,
                'age_ranges': analysis_result.age_range_analysis,
                'insights': analysis_result.insights
            }
            
            summary_prompt = f"""
            You are a Premier League football analytics expert with current knowledge of the {datetime.now().strftime("%Y-%m-%d")} season. Analyze the following age vs performance data and provide insights:
            
            Analysis Data:
            - Metric analyzed: {analysis_data['metric']}
            - Position filter: {analysis_data['position_filter'] or 'All positions'}
            - Sample size: {analysis_data['sample_size']} players
            - Linear model R²: {analysis_data['linear_r2']:.4f}
            - Polynomial model R²: {analysis_data['polynomial_r2']:.4f}
            - Best fitting model: {analysis_data['best_model']}
            - Peak performance age: {analysis_data['peak_age']} years (if applicable)
            
            Age Range Performance:
            {json.dumps(analysis_data['age_ranges'], indent=2)}
            
            Statistical Insights:
            {json.dumps(analysis_data['insights'], indent=2)}
            
            Please provide a comprehensive analysis in the following JSON format:
            {{
                "overall_summary": "A 2-3 sentence overview of the key findings",
                "key_findings": ["finding 1", "finding 2", "finding 3"],
                "age_insights": ["insight about age patterns", "insight about peak performance"],
                "performance_trends": "Description of how performance changes with age",
                "recommendations": ["recommendation 1", "recommendation 2"],
                "statistical_interpretation": "Explanation of the statistical significance and model quality",
                "confidence_level": "High/Medium/Low based on R² and sample size"
            }}
            
            Focus on:
            1. Practical implications for FPL managers
            2. Age-related performance patterns
            3. Statistical significance and reliability
            4. Actionable insights for player selection
            """
            
            messages = [{"role": "user", "content": summary_prompt}]
            
            chat_response = self.mistral_client.chat.complete(
                model=self.mistral_model,
                messages=messages,
                max_tokens=800,
                temperature=0.3
            )
            
            response_content = chat_response.choices[0].message.content
            
            # Parse JSON response
            try:
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    ai_summary_data = json.loads(json_match.group())
                    
                    return AIAnalysisSummary(
                        overall_summary=ai_summary_data.get('overall_summary', ''),
                        key_findings=ai_summary_data.get('key_findings', []),
                        age_insights=ai_summary_data.get('age_insights', []),
                        performance_trends=ai_summary_data.get('performance_trends', ''),
                        recommendations=ai_summary_data.get('recommendations', []),
                        statistical_interpretation=ai_summary_data.get('statistical_interpretation', ''),
                        confidence_level=ai_summary_data.get('confidence_level', 'Medium'),
                        generated_at=datetime.now().isoformat()
                    )
                    
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse AI summary as JSON: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating AI analysis summary: {e}")
            return None
        
    def calculate_age_from_birth_date(self, birth_date_str: str) -> float:
        """Calculate current age from birth date string"""
        try:
            if not birth_date_str:
                return None
                
            # Parse birth date (assuming format: YYYY-MM-DD)
            birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d').date()
            today = date.today()
            
            # Calculate age with decimal precision
            age = today.year - birth_date.year
            if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
                age -= 1
                
            # Add fractional part based on days
            days_since_birthday = (today - date(today.year, birth_date.month, birth_date.day)).days
            if days_since_birthday < 0:
                days_since_birthday += 365
                
            fractional_age = age + (days_since_birthday / 365.0)
            return round(fractional_age, 2)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate age from birth date {birth_date_str}: {e}")
            return None
    
    def calculate_age_at_season_start(self, birth_date_str: str, season: str = "2024-25") -> int:
        """Calculate player's age at the start of the season (whole years)"""
        try:
            if not birth_date_str:
                return None
                
            # Parse birth date
            birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d').date()
            
            # Determine season start date (FPL season typically starts in August)
            season_year = int(season.split('-')[0])
            season_start = date(season_year, 8, 1)  # August 1st as season start approximation
            
            # Calculate age at season start (whole years only)
            age_at_season_start = season_start.year - birth_date.year
            if season_start.month < birth_date.month or (season_start.month == birth_date.month and season_start.day < birth_date.day):
                age_at_season_start -= 1
                
            return age_at_season_start
            
        except Exception as e:
            self.logger.warning(f"Could not calculate season start age from birth date {birth_date_str}: {e}")
            return None
    
    async def extract_player_age_data(self, fpl_data: Dict, historical_data: Optional[Dict] = None, use_ai_enrichment: bool = True, force_new_enrichment: bool = False) -> List[PlayerAgeData]:
        """Extract player data with age information from FPL data"""
        try:
            players_data = fpl_data.get('elements', [])
            teams_data = fpl_data.get('teams', [])
            
            # Create team lookup
            team_lookup = {team['id']: team['name'] for team in teams_data}
            
            age_data = []
            
            enriched_count = 0
            
            for player in players_data:
                try:
                    # Calculate age from birth date if available
                    birth_date = player.get('date_of_birth')
                    age = self.calculate_age_from_birth_date(birth_date) if birth_date else None
                    
                    # Try to get enriched data if no age available and AI enrichment is enabled
                    if age is None and use_ai_enrichment and self.mistral_client:
                        player_name = player.get('web_name', '')
                        team_name = team_lookup.get(player.get('team'), 'Unknown')
                        
                        if player_name and team_name != 'Unknown':
                            # Try to get cached enriched data first
                            enriched_data = self._get_enriched_player_data(player['id'])
                            
                            # If no cached data and force_new_enrichment is enabled, try AI enrichment
                            if not enriched_data and force_new_enrichment:
                                try:
                                    enriched_data = await self.enrich_player_data_with_ai(
                                        player_name, team_name, player['id']
                                    )
                                    if enriched_data:
                                        enriched_count += 1
                                except Exception as ai_error:
                                    self.logger.warning(f"AI enrichment failed for {player_name}: {ai_error}")
                            
                            # Use enriched age if available
                            if enriched_data and enriched_data.enriched_age:
                                age = enriched_data.enriched_age
                                birth_date = enriched_data.birth_date
                    
                    # Skip players without age data
                    if age is None:
                        continue
                    
                    # Map position
                    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                    position = pos_map.get(player.get('element_type'), 'UNK')
                    
                    # Extract performance metrics
                    player_data = PlayerAgeData(
                        player_id=player['id'],
                        name=player.get('web_name', ''),
                        position=position,
                        team=team_lookup.get(player.get('team'), 'Unknown'),
                        age=age,
                        birth_date=birth_date,
                        points=float(player.get('total_points', 0)),
                        goals=int(player.get('goals_scored', 0)),
                        assists=int(player.get('assists', 0)),
                        minutes=int(player.get('minutes', 0)),
                        games_played=int(player.get('games_played', 0)),
                        form=float(player.get('form', 0)) if player.get('form') else 0,
                        points_per_game=float(player.get('points_per_game', 0)) if player.get('points_per_game') else 0,
                        xg=float(player.get('expected_goals', 0)) if player.get('expected_goals') else 0,
                        xa=float(player.get('expected_assists', 0)) if player.get('expected_assists') else 0,
                        clean_sheets=int(player.get('clean_sheets', 0)),
                        saves=int(player.get('saves', 0))
                    )
                    
                    age_data.append(player_data)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing player {player.get('id')}: {e}")
                    continue
            
            self.logger.info(f"Extracted age data for {len(age_data)} players ({enriched_count} AI-enriched)")
            
            # Store enriched count for later use
            self.last_enriched_count = enriched_count
            
            return age_data
            
        except Exception as e:
            self.logger.error(f"Error extracting player age data: {e}")
            return []
    
    def fit_regression_models(self, ages: List[float], values: List[float], metric: str) -> Tuple[RegressionResult, RegressionResult]:
        """Fit both linear and polynomial regression models"""
        try:
            ages_array = np.array(ages).reshape(-1, 1)
            values_array = np.array(values)
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                ages_array, values_array, test_size=0.2, random_state=42
            )
            
            # Linear regression
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            
            linear_predictions = linear_model.predict(ages_array)
            linear_r2 = r2_score(values_array, linear_predictions)
            linear_mae = mean_absolute_error(values_array, linear_predictions)
            
            # Calculate statistical significance for linear model
            linear_correlation, linear_p_value = stats.pearsonr(ages, values)
            linear_is_significant = linear_p_value < 0.05
            
            linear_result = RegressionResult(
                model_type='linear',
                r2_score=linear_r2,
                mae=linear_mae,
                coefficients=[linear_model.coef_[0]],
                intercept=linear_model.intercept_,
                feature_names=['age'],
                predictions=linear_predictions.tolist(),
                ages=ages,
                actual_values=values,
                p_value=linear_p_value,
                correlation=linear_correlation,
                is_significant=linear_is_significant
            )
            
            # Polynomial regression (degree 2)
            poly_pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
            
            poly_pipeline.fit(X_train, y_train)
            
            poly_predictions = poly_pipeline.predict(ages_array)
            poly_r2 = r2_score(values_array, poly_predictions)
            poly_mae = mean_absolute_error(values_array, poly_predictions)
            
            # Get polynomial coefficients
            poly_coeffs = poly_pipeline.named_steps['linear'].coef_
            poly_intercept = poly_pipeline.named_steps['linear'].intercept_
            
            # For polynomial, use correlation between predictions and actual values as significance measure
            poly_correlation, poly_p_value = stats.pearsonr(poly_predictions, values_array)
            poly_is_significant = poly_p_value < 0.05
            
            poly_result = RegressionResult(
                model_type='polynomial',
                r2_score=poly_r2,
                mae=poly_mae,
                coefficients=poly_coeffs.tolist(),
                intercept=poly_intercept,
                feature_names=['constant', 'age', 'age^2'],
                predictions=poly_predictions.tolist(),
                ages=ages,
                actual_values=values,
                p_value=poly_p_value,
                correlation=poly_correlation,
                is_significant=poly_is_significant
            )
            
            return linear_result, poly_result
            
        except Exception as e:
            self.logger.error(f"Error fitting regression models for {metric}: {e}")
            # Return empty results
            empty_result = RegressionResult(
                model_type='error',
                r2_score=0.0,
                mae=float('inf'),
                coefficients=[],
                intercept=0.0,
                feature_names=[],
                predictions=[],
                ages=ages,
                actual_values=values
            )
            return empty_result, empty_result
    
    def find_peak_age(self, polynomial_result: RegressionResult) -> float:
        """Find peak age from polynomial regression"""
        try:
            if len(polynomial_result.coefficients) >= 3:
                # For quadratic: ax^2 + bx + c, peak is at x = -b/(2a)
                a = polynomial_result.coefficients[2]  # age^2 coefficient
                b = polynomial_result.coefficients[1]  # age coefficient
                
                if a < 0:  # Downward parabola has a maximum
                    peak_age = -b / (2 * a)
                    return round(peak_age, 1)
                    
            return None
            
        except Exception as e:
            self.logger.warning(f"Error finding peak age: {e}")
            return None
    
    def analyze_age_ranges(self, age_data: List[PlayerAgeData], metric: str) -> Dict[str, Any]:
        """Analyze performance across different age ranges"""
        try:
            if not age_data:
                return {}
                
            # Define age ranges
            age_ranges = [
                (16, 20, 'Youth (16-20)'),
                (21, 24, 'Young (21-24)'),
                (25, 28, 'Prime (25-28)'),
                (29, 32, 'Experienced (29-32)'),
                (33, 40, 'Veteran (33+)')
            ]
            
            range_analysis = {}
            
            for min_age, max_age, label in age_ranges:
                range_players = [p for p in age_data if min_age <= p.age <= max_age]
                
                if range_players:
                    values = [getattr(p, metric) for p in range_players]
                    
                    range_analysis[label] = {
                        'count': len(range_players),
                        'avg_value': round(np.mean(values), 2),
                        'median_value': round(np.median(values), 2),
                        'std_value': round(np.std(values), 2),
                        'min_value': round(min(values), 2),
                        'max_value': round(max(values), 2),
                        'age_range': f"{min_age}-{max_age}",
                        'top_players': sorted(
                            [(p.name, p.age, getattr(p, metric)) for p in range_players],
                            key=lambda x: x[2],
                            reverse=True
                        )[:5]
                    }
            
            return range_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing age ranges: {e}")
            return {}
    
    def analyze_age_groups_by_year(self, age_data: List[PlayerAgeData], metric: str, season: str = "2024-25") -> Dict[str, Any]:
        """Analyze performance grouped by age at season start (whole years)"""
        try:
            if not age_data:
                return {}
            
            # Group players by their age at season start
            age_groups = {}
            
            for player in age_data:
                # Calculate age at season start
                birth_date = player.birth_date
                season_start_age = None
                
                if birth_date:
                    season_start_age = self.calculate_age_at_season_start(birth_date, season)
                
                # Fall back to current age rounded down if no birth date
                if season_start_age is None and player.age:
                    season_start_age = int(player.age)
                
                if season_start_age is not None:
                    age_key = f"Age {season_start_age}"
                    
                    if age_key not in age_groups:
                        age_groups[age_key] = []
                    
                    age_groups[age_key].append(player)
            
            # Calculate statistics for each age group
            age_analysis = {}
            
            for age_key, players in age_groups.items():
                if len(players) >= 3:  # Only include ages with at least 3 players
                    values = [getattr(p, metric) for p in players]
                    
                    age_analysis[age_key] = {
                        'count': len(players),
                        'avg_value': round(np.mean(values), 2),
                        'median_value': round(np.median(values), 2),
                        'std_value': round(np.std(values), 2),
                        'min_value': round(min(values), 2),
                        'max_value': round(max(values), 2),
                        'age': int(age_key.split()[-1]),  # Extract age number
                        'top_players': sorted(
                            [(p.name, p.age, getattr(p, metric), p.position, p.team) for p in players],
                            key=lambda x: x[2],
                            reverse=True
                        )[:5]
                    }
            
            # Sort by age
            sorted_ages = dict(sorted(age_analysis.items(), key=lambda x: x[1]['age']))
            
            return {
                'age_groups': sorted_ages,
                'total_players': len(age_data),
                'ages_analyzed': len(sorted_ages),
                'age_range': {
                    'min': min([data['age'] for data in sorted_ages.values()]) if sorted_ages else 0,
                    'max': max([data['age'] for data in sorted_ages.values()]) if sorted_ages else 0
                },
                'season': season
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing age groups by year: {e}")
            return {}
    
    def generate_insights(self, analysis_result: AgeAnalysisResult) -> List[str]:
        """Generate insights from age analysis"""
        insights = []
        
        try:
            # Model performance insight
            if analysis_result.best_model == 'polynomial':
                insights.append(f"Performance follows a curved pattern with age, suggesting an optimal age range for {analysis_result.metric}")
            else:
                insights.append(f"Performance shows a linear relationship with age for {analysis_result.metric}")
            
            # Peak age insight
            if analysis_result.peak_age:
                insights.append(f"Peak performance age is around {analysis_result.peak_age} years")
            
            # Age range insights
            if analysis_result.age_range_analysis:
                best_range = max(
                    analysis_result.age_range_analysis.items(),
                    key=lambda x: x[1].get('avg_value', 0)
                )
                insights.append(f"Best performing age group: {best_range[0]} with average {analysis_result.metric} of {best_range[1]['avg_value']}")
            
            # R² insight
            best_r2 = max(analysis_result.linear_model.r2_score, analysis_result.polynomial_model.r2_score)
            if best_r2 > 0.3:
                insights.append(f"Strong age-performance correlation (R² = {best_r2:.3f})")
            elif best_r2 > 0.1:
                insights.append(f"Moderate age-performance correlation (R² = {best_r2:.3f})")
            else:
                insights.append(f"Weak age-performance correlation (R² = {best_r2:.3f})")
                
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            insights.append("Analysis completed with limited insights due to data constraints")
        
        return insights
    
    async def analyze_age_performance(self, fpl_data: Dict, metric: str = 'points', position_filter: Optional[str] = None, use_ai_enrichment: bool = True, generate_ai_summary: bool = True, force_new_enrichment: bool = False, player_ids: Optional[List[int]] = None) -> AgeAnalysisResult:
        """Main method to analyze age vs performance relationship"""
        try:
            # Extract age data
            age_data = await self.extract_player_age_data(fpl_data, use_ai_enrichment=use_ai_enrichment, force_new_enrichment=force_new_enrichment)
            
            if not age_data:
                raise ValueError("No age data available for analysis")
            
            # Apply position filter
            if position_filter:
                age_data = [p for p in age_data if p.position == position_filter]
            
            # Apply player IDs filter
            if player_ids:
                age_data = [p for p in age_data if p.player_id in player_ids]
            
            # Filter for players with meaningful data (played some minutes)
            age_data = [p for p in age_data if p.minutes > 0 and hasattr(p, metric)]
            
            if len(age_data) < 10:
                raise ValueError(f"Insufficient data for analysis (only {len(age_data)} players)")
            
            # Extract ages and metric values
            # Round ages down to whole numbers for consistent grouping (24.1, 24.6, 24.9 all become 24)
            ages = [int(p.age) for p in age_data if p.age is not None]
            values = [getattr(p, metric) for p in age_data if p.age is not None]
            
            # Fit regression models
            linear_result, poly_result = self.fit_regression_models(ages, values, metric)
            
            # Determine best model
            best_model = 'polynomial' if poly_result.r2_score > linear_result.r2_score else 'linear'
            
            # Find peak age
            peak_age = self.find_peak_age(poly_result)
            
            # Analyze age ranges
            age_range_analysis = self.analyze_age_ranges(age_data, metric)
            
            # Analyze age groups by whole years (season start age)
            age_groups_analysis = self.analyze_age_groups_by_year(age_data, metric, "2024-25")
            
            # Player comparisons (top performers by age group)
            player_comparisons = []
            for player in sorted(age_data, key=lambda p: getattr(p, metric), reverse=True)[:20]:
                if player.age is not None:
                    player_comparisons.append({
                        'name': player.name,
                        'age': int(player.age),  # Round down to whole number for consistency
                        'position': player.position,
                        'team': player.team,
                        'value': getattr(player, metric),
                        'minutes': player.minutes
                    })
            
            # Create analysis result
            analysis_result = AgeAnalysisResult(
                metric=metric,
                position_filter=position_filter,
                linear_model=linear_result,
                polynomial_model=poly_result,
                best_model=best_model,
                peak_age=peak_age,
                age_range_analysis=age_range_analysis,
                age_groups_analysis=age_groups_analysis,
                player_comparisons=player_comparisons,
                insights=[],
                ai_summary=None,
                enriched_data_count=getattr(self, 'last_enriched_count', 0)
            )
            
            # Generate insights
            analysis_result.insights = self.generate_insights(analysis_result)
            
            # Generate AI summary if requested
            if generate_ai_summary and self.mistral_client:
                try:
                    ai_summary = await self.generate_ai_analysis_summary(analysis_result)
                    analysis_result.ai_summary = ai_summary
                except Exception as ai_error:
                    self.logger.warning(f"Failed to generate AI summary: {ai_error}")
            
            self.logger.info(f"Age analysis completed for {metric} with {len(age_data)} players")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in age performance analysis: {e}")
            raise
    
    async def analyze_season_correlation(self, current_season_data: Dict, previous_season: str = "2023-24") -> Dict[str, Any]:
        """Analyze correlation between current season and previous season performance"""
        try:
            self.logger.info(f"Starting season correlation analysis: current vs {previous_season}")
            
            # Fetch previous season data
            previous_data = await self.fetch_historical_season_data(previous_season)
            if not previous_data:
                raise ValueError(f"Failed to fetch {previous_season} season data")
            
            # Extract current season players
            current_players = current_season_data.get('elements', [])
            previous_players = previous_data.get('elements', [])
            
            # Create lookup for previous season by player name and team
            previous_lookup = {}
            teams_2023 = {team['id']: team['name'] for team in previous_data.get('teams', [])}
            
            for player in previous_players:
                player_name = player.get('web_name', '').strip()
                team_name = teams_2023.get(player.get('team'), '').strip()
                if player_name and team_name:
                    key = f"{player_name}-{team_name}"
                    previous_lookup[key] = player
            
            # Match players between seasons
            matched_players = []
            teams_2024 = {team['id']: team['name'] for team in current_season_data.get('teams', [])}
            
            for current_player in current_players:
                current_name = current_player.get('web_name', '').strip()
                current_team = teams_2024.get(current_player.get('team'), '').strip()
                
                if not current_name or not current_team:
                    continue
                
                # Try to find matching player in previous season
                key = f"{current_name}-{current_team}"
                previous_player = previous_lookup.get(key)
                
                # If exact match not found, try just by player name for transfers
                if not previous_player:
                    for prev_key, prev_player in previous_lookup.items():
                        if prev_player.get('web_name', '').strip() == current_name:
                            previous_player = prev_player
                            break
                
                if previous_player:
                    current_points = float(current_player.get('total_points', 0))
                    previous_points = float(previous_player.get('total_points', 0))
                    
                    # Only include players with meaningful minutes in previous season
                    # (current season may not have started yet)
                    if previous_player.get('minutes', 0) > 180:
                        
                        matched_players.append({
                            'player_name': current_name,
                            'team': current_team,
                            'current_points': current_points,
                            'previous_points': previous_points,
                            'current_minutes': int(current_player.get('minutes', 0)),
                            'previous_minutes': int(previous_player.get('minutes', 0)),
                            'position': {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(current_player.get('element_type'), 'UNK')
                        })
            
            if len(matched_players) < 10:
                raise ValueError(f"Insufficient matched players: {len(matched_players)}")
            
            # Perform correlation analysis
            current_points = [p['current_points'] for p in matched_players]
            previous_points = [p['previous_points'] for p in matched_players]
            
            # Calculate statistics
            correlation, p_value = stats.pearsonr(previous_points, current_points)
            r2_score_val = correlation ** 2
            
            # Linear regression for trend line
            X = np.array(previous_points).reshape(-1, 1)
            y = np.array(current_points)
            
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)
            
            # Identify top consistent and inconsistent players
            matched_players_with_diff = []
            for player in matched_players:
                expected = model.predict([[player['previous_points']]])[0]
                actual = player['current_points']
                difference = actual - expected
                player['expected_points'] = round(expected, 1)
                player['difference'] = round(difference, 1)
                matched_players_with_diff.append(player)
            
            # Sort by consistency (smallest absolute difference = most consistent)
            most_consistent = sorted(matched_players_with_diff, key=lambda x: abs(x['difference']))[:10]
            biggest_improvers = sorted(matched_players_with_diff, key=lambda x: x['difference'], reverse=True)[:10]
            biggest_decliners = sorted(matched_players_with_diff, key=lambda x: x['difference'])[:10]
            
            result = {
                'analysis_type': 'season_correlation',
                'seasons_compared': f"{previous_season} vs 2024-25",
                'sample_size': len(matched_players),
                'correlation': round(correlation, 4),
                'r2_score': round(r2_score_val, 4),
                'p_value': round(p_value, 4),
                'is_significant': bool(p_value < 0.05),
                'regression': {
                    'slope': round(model.coef_[0], 4),
                    'intercept': round(model.intercept_, 4),
                    'previous_points': previous_points,
                    'current_points': current_points,
                    'predictions': [round(p, 1) for p in predictions.tolist()]
                },
                'player_analysis': {
                    'most_consistent': most_consistent,
                    'biggest_improvers': biggest_improvers,
                    'biggest_decliners': biggest_decliners,
                    'all_players': matched_players_with_diff
                },
                'matched_players_count': len(matched_players),
                'insights': self.generate_season_correlation_insights(correlation, r2_score_val, p_value, len(matched_players))
            }
            
            self.logger.info(f"Season correlation analysis completed: r={correlation:.3f}, p={p_value:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in season correlation analysis: {e}")
            raise
    
    async def fetch_historical_season_data(self, season: str) -> Optional[Dict]:
        """Fetch historical season data from Vaastav's repository"""
        try:
            # Construct URL for the season's players_raw.csv (has team data)
            base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
            players_url = f"{base_url}/{season}/players_raw.csv"
            teams_url = f"{base_url}/{season}/teams.csv"
            
            self.logger.info(f"Fetching {season} data from Vaastav's repository")
            
            # Fetch players data
            players_response = requests.get(players_url, timeout=30)
            players_response.raise_for_status()
            
            # Fetch teams data
            teams_response = requests.get(teams_url, timeout=30)
            teams_response.raise_for_status()
            
            # Save to temporary files and load with pandas
            with open(f'temp_players_{season.replace("-", "_")}.csv', 'w', encoding='utf-8') as f:
                f.write(players_response.text)
            
            with open(f'temp_teams_{season.replace("-", "_")}.csv', 'w', encoding='utf-8') as f:
                f.write(teams_response.text)
            
            players_df = pd.read_csv(f'temp_players_{season.replace("-", "_")}.csv')
            teams_df = pd.read_csv(f'temp_teams_{season.replace("-", "_")}.csv')
            
            # Convert to FPL API format
            elements = []
            for idx, player in players_df.iterrows():
                # Use web_name directly or create from first_name and second_name
                web_name = str(player.get('web_name', '')).strip()
                if not web_name:
                    first_name = str(player.get('first_name', '')).strip()
                    second_name = str(player.get('second_name', '')).strip()
                    web_name = second_name if second_name else first_name
                
                # Use element_type directly (it's already numeric in raw data)
                element_type = int(player.get('element_type', 0)) if pd.notna(player.get('element_type', 0)) else 0
                
                elements.append({
                    'id': int(player.get('id', idx + 1)) if pd.notna(player.get('id', idx + 1)) else idx + 1,
                    'web_name': web_name,
                    'team': int(player.get('team', 0)) if pd.notna(player.get('team', 0)) else 0,
                    'element_type': element_type,
                    'total_points': int(float(player.get('total_points', 0))) if pd.notna(player.get('total_points', 0)) else 0,
                    'minutes': int(float(player.get('minutes', 0))) if pd.notna(player.get('minutes', 0)) else 0,
                    'goals_scored': int(float(player.get('goals_scored', 0))) if pd.notna(player.get('goals_scored', 0)) else 0,
                    'assists': int(float(player.get('assists', 0))) if pd.notna(player.get('assists', 0)) else 0,
                    'clean_sheets': int(float(player.get('clean_sheets', 0))) if pd.notna(player.get('clean_sheets', 0)) else 0
                })
            
            teams = []
            for idx, team in teams_df.iterrows():
                teams.append({
                    'id': int(team.get('id', idx + 1)) if pd.notna(team.get('id', idx + 1)) else idx + 1,
                    'name': str(team.get('name', '')).strip(),
                    'short_name': str(team.get('short_name', '')).strip()
                })
            
            # Clean up temp files
            try:
                os.remove(f'temp_players_{season.replace("-", "_")}.csv')
                os.remove(f'temp_teams_{season.replace("-", "_")}.csv')
            except:
                pass
            
            result = {
                'elements': elements,
                'teams': teams
            }
            
            self.logger.info(f"Successfully fetched {season} data: {len(elements)} players, {len(teams)} teams")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to fetch {season} data: {e}")
            return None
    
    def generate_season_correlation_insights(self, correlation: float, r2: float, p_value: float, sample_size: int) -> List[str]:
        """Generate insights for season correlation analysis"""
        insights = []
        
        # Correlation strength
        if abs(correlation) > 0.7:
            insights.append(f"Strong correlation (r={correlation:.3f}) between seasons - player performance is highly predictable")
        elif abs(correlation) > 0.5:
            insights.append(f"Moderate correlation (r={correlation:.3f}) between seasons - some predictability in performance")
        elif abs(correlation) > 0.3:
            insights.append(f"Weak correlation (r={correlation:.3f}) between seasons - limited predictability")
        else:
            insights.append(f"Very weak correlation (r={correlation:.3f}) between seasons - performance is largely unpredictable")
        
        # Statistical significance
        if p_value < 0.001:
            insights.append("Correlation is highly statistically significant (p < 0.001)")
        elif p_value < 0.05:
            insights.append("Correlation is statistically significant (p < 0.05)")
        else:
            insights.append("Correlation is not statistically significant - may be due to chance")
        
        # Variance explained
        variance_explained = r2 * 100
        insights.append(f"Previous season performance explains {variance_explained:.1f}% of current season variance")
        
        # Sample size assessment
        if sample_size > 100:
            insights.append(f"Large sample size ({sample_size} players) provides reliable results")
        elif sample_size > 50:
            insights.append(f"Good sample size ({sample_size} players) for analysis")
        else:
            insights.append(f"Small sample size ({sample_size} players) - results should be interpreted cautiously")
        
        return insights
    
    def get_player_age_prediction(self, player_age: float, analysis_result: AgeAnalysisResult) -> Dict[str, Any]:
        """Predict performance for a specific age using the analysis model"""
        try:
            # Use the best model for prediction
            if analysis_result.best_model == 'polynomial':
                model_result = analysis_result.polynomial_model
                # Polynomial prediction: ax^2 + bx + c
                if len(model_result.coefficients) >= 3:
                    prediction = (model_result.coefficients[2] * player_age**2 + 
                                model_result.coefficients[1] * player_age + 
                                model_result.intercept)
                else:
                    prediction = model_result.intercept
            else:
                model_result = analysis_result.linear_model
                # Linear prediction: ax + b
                prediction = (model_result.coefficients[0] * player_age + 
                            model_result.intercept)
            
            return {
                'predicted_value': round(prediction, 2),
                'model_used': analysis_result.best_model,
                'confidence': model_result.r2_score,
                'age': player_age
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting for age {player_age}: {e}")
            return {
                'predicted_value': 0,
                'model_used': 'error',
                'confidence': 0,
                'age': player_age
            }
    
    def compare_players_by_age(self, player_ids: List[int], fpl_data: Dict, metric: str = 'points') -> Dict[str, Any]:
        """Compare specific players in the context of age-performance analysis"""
        try:
            # Get age data
            age_data = self.extract_player_age_data(fpl_data)
            
            # Find requested players
            target_players = [p for p in age_data if p.player_id in player_ids]
            
            if not target_players:
                return {'error': 'No matching players found'}
            
            # Get analysis for the metric
            analysis_result = self.analyze_age_performance(fpl_data, metric)
            
            # Analyze each player
            player_analysis = []
            for player in target_players:
                prediction = self.get_player_age_prediction(player.age, analysis_result)
                actual_value = getattr(player, metric)
                
                # Performance vs expected
                performance_ratio = actual_value / prediction['predicted_value'] if prediction['predicted_value'] > 0 else 1
                
                player_analysis.append({
                    'player_id': player.player_id,
                    'name': player.name,
                    'age': player.age,
                    'position': player.position,
                    'team': player.team,
                    'actual_value': actual_value,
                    'predicted_value': prediction['predicted_value'],
                    'performance_ratio': round(performance_ratio, 2),
                    'performance_category': 'Above Expected' if performance_ratio > 1.1 else 'Below Expected' if performance_ratio < 0.9 else 'As Expected',
                    'minutes': player.minutes
                })
            
            return {
                'player_comparisons': player_analysis,
                'analysis_metadata': {
                    'metric': metric,
                    'total_players_analyzed': len(age_data),
                    'model_r2': max(analysis_result.linear_model.r2_score, analysis_result.polynomial_model.r2_score),
                    'peak_age': analysis_result.peak_age
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing players by age: {e}")
            return {'error': str(e)}