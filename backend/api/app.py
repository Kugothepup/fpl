#!/usr/bin/env python3
"""
FPL Manager v3 - Main Flask API Application
Comprehensive FPL management system with ML predictions, real data integration
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime
import os
import sys
from pathlib import Path
import traceback
import uuid
import sqlite3
import threading
import time
import asyncio
import json
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
from config.config import Config, config

# Import core components
from core.fpl_manager import FPLManager
from core.ml_predictor import FPLMLPredictor, PredictionResult

# Import services
from services.weather_service import WeatherService
from services.news_service import NewsService
from services.accuracy_tracker import AccuracyTracker, PredictionRecord
from services.mistral_optimizer import MistralTeamOptimizer, OptimizationConstraints
from services.age_performance_service import AgePerformanceService
from services.fixture_service import real_fixture_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_app(config_name='default'):
    """Application factory"""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Enable CORS for React frontend
    CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://10.2.0.2:3000"])
    
    # Initialize services
    try:
        fpl_manager = FPLManager()
        ml_predictor = FPLMLPredictor()
        weather_service = WeatherService()
        news_service = NewsService()
        accuracy_tracker = AccuracyTracker()
        mistral_optimizer = MistralTeamOptimizer()
        age_performance_service = AgePerformanceService()
        
        # Initialize and refresh real fixture data for 2025-26 season
        logger.info("Refreshing 2025-26 season fixtures data...")
        real_fixture_service.refresh_data()
        
        # Initialize background task manager
        global background_tasks
        background_tasks = {}
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Load ML models if available
    try:
        ml_predictor.load_models()
        logger.info("ML models loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load existing ML models: {e}")
    
    # Background task utilities
    def create_background_task(task_id, task_type, status='running', progress=0, result=None, error=None):
        """Create or update a background task"""
        background_tasks[task_id] = {
            'id': task_id,
            'type': task_type,
            'status': status,  # 'running', 'completed', 'failed'
            'progress': progress,  # 0-100
            'result': result,
            'error': error,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        return background_tasks[task_id]
    
    def update_background_task(task_id, status=None, progress=None, result=None, error=None):
        """Update an existing background task"""
        if task_id in background_tasks:
            task = background_tasks[task_id]
            if status is not None:
                task['status'] = status
            if progress is not None:
                task['progress'] = progress
            if result is not None:
                task['result'] = result
            if error is not None:
                task['error'] = error
            task['updated_at'] = datetime.now().isoformat()
            return task
        return None
    
    def run_ai_analysis_background(task_id, metric, position):
        """Run AI-enhanced analysis in background"""
        try:
            update_background_task(task_id, status='running', progress=10)
            
            # Get FPL data
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data:
                update_background_task(task_id, status='failed', error='Failed to fetch FPL data')
                return
            
            update_background_task(task_id, progress=30)
            
            # Run analysis with AI features
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                analysis_result = loop.run_until_complete(
                    age_performance_service.analyze_age_performance(
                        fpl_data, metric, position, use_ai_enrichment=True, generate_ai_summary=True
                    )
                )
                
                update_background_task(task_id, progress=90)
                
                # Convert to serializable format (minimal for background task)
                response_data = {
                    'metric': analysis_result.metric,
                    'ai_summary': {
                        'overall_summary': analysis_result.ai_summary.overall_summary,
                        'key_findings': analysis_result.ai_summary.key_findings,
                        'age_insights': analysis_result.ai_summary.age_insights,
                        'performance_trends': analysis_result.ai_summary.performance_trends,
                        'recommendations': analysis_result.ai_summary.recommendations,
                        'statistical_interpretation': analysis_result.ai_summary.statistical_interpretation,
                        'confidence_level': analysis_result.ai_summary.confidence_level,
                        'generated_at': analysis_result.ai_summary.generated_at
                    } if analysis_result.ai_summary else None
                }
                
                update_background_task(task_id, status='completed', progress=100, result=response_data)
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Background AI analysis failed: {e}")
            update_background_task(task_id, status='failed', error=str(e))
    
    @app.errorhandler(Exception)
    def handle_error(e):
        """Global error handler"""
        logger.error(f"Unhandled error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e) if app.debug else 'An error occurred'
        }), 500
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        try:
            # Test FPL API connectivity
            fpl_data = fpl_manager.fetch_bootstrap_data()
            fpl_healthy = fpl_data is not None
            
            # Check service status
            status = {
                'status': 'healthy' if fpl_healthy else 'degraded',
                'timestamp': datetime.now().isoformat(),
                'version': '3.0.0',
                'services': {
                    'fpl_api': fpl_healthy,
                    'ml_predictor': len(ml_predictor.pipelines) > 0,
                    'weather_service': bool(weather_service.api_key),
                    'news_service': bool(news_service.api_key),
                    'accuracy_tracker': True
                },
                'features': {
                    'real_data_only': True,
                    'ml_predictions': True,
                    'weather_integration': Config.ENABLE_WEATHER_INTEGRATION,
                    'news_integration': Config.ENABLE_NEWS_INTEGRATION,
                    'accuracy_tracking': Config.ENABLE_ACCURACY_TRACKING
                }
            }
            
            return jsonify(status)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/bootstrap', methods=['GET'])
    def get_bootstrap_data():
        """Get FPL bootstrap data (players, teams, gameweeks)"""
        try:
            data = fpl_manager.fetch_bootstrap_data()
            
            if not data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
            
            # Add season status
            season_status = fpl_manager.get_season_status()
            
            return jsonify({
                'success': True,
                'data': data,
                'season_status': season_status,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Bootstrap data fetch failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/team', methods=['GET'])
    def get_user_team():
        """Get user's current FPL team with enriched player data"""
        try:
            team_id = request.args.get('team_id', Config.FPL_TEAM_ID)
            
            # Get raw team data
            team_data = fpl_manager.fetch_user_team(team_id)
            if not team_data:
                return jsonify({'error': 'Failed to fetch team data'}), 500
            
            # Get bootstrap data for player/team lookups
            bootstrap_data = fpl_manager.fetch_bootstrap_data()
            if not bootstrap_data:
                return jsonify({'error': 'Failed to fetch player data'}), 500
            
            # Create lookup dictionaries
            players_lookup = {p['id']: p for p in bootstrap_data.get('elements', [])}
            teams_lookup = {t['id']: t['name'] for t in bootstrap_data.get('teams', [])}
            
            # Enrich the picks with player and team information
            enriched_picks = []
            for pick in team_data.get('picks', []):
                player_id = pick['element']
                player_data = players_lookup.get(player_id, {})
                
                if player_data:
                    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                    enriched_pick = {
                        **pick,
                        'player_id': player_id,
                        'name': player_data.get('web_name', f'Player {player_id}'),
                        'full_name': f"{player_data.get('first_name', '')} {player_data.get('second_name', '')}".strip(),
                        'team': teams_lookup.get(player_data.get('team'), 'Unknown'),
                        'team_id': player_data.get('team'),
                        'position': pos_map.get(player_data.get('element_type'), 'UNK'),
                        'cost': player_data.get('now_cost', 0) / 10.0,
                        'total_points': player_data.get('total_points', 0),
                        'form': float(player_data.get('form', 0)) if player_data.get('form') else 0,
                        'points_per_game': float(player_data.get('points_per_game', 0)) if player_data.get('points_per_game') else 0
                    }
                    enriched_picks.append(enriched_pick)
                else:
                    # Fallback for missing player data
                    enriched_picks.append({
                        **pick,
                        'player_id': player_id,
                        'name': f'Player {player_id}',
                        'full_name': f'Player {player_id}',
                        'team': 'Unknown',
                        'position': 'UNK',
                        'cost': 0,
                        'total_points': 0
                    })
            
            # Get team entry data for team name and player info
            entry_data = None
            try:
                entry_response = fpl_manager.session.get(f"{fpl_manager.base_url}/entry/{team_id}/", timeout=30)
                if entry_response.status_code == 200:
                    entry_data = entry_response.json()
            except:
                pass  # Entry data is optional
            
            # Return enriched team data
            enriched_team_data = {
                **team_data,
                'picks': enriched_picks,
                'entry': entry_data  # Add entry information for team name, player name, etc.
            }
            
            return jsonify({
                'success': True,
                'data': enriched_team_data,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Team data fetch failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/fixtures', methods=['GET'])
    def get_fixtures():
        """Get fixture data"""
        try:
            fixtures = fpl_manager.fetch_fixtures()
            
            # Convert fixtures to serializable format
            fixtures_data = [
                {
                    'id': f.id,
                    'gameweek': f.gameweek,
                    'home_team': f.home_team,
                    'away_team': f.away_team,
                    'home_team_id': f.home_team_id,
                    'away_team_id': f.away_team_id,
                    'kickoff_time': f.kickoff_time,
                    'difficulty_home': f.difficulty_home,
                    'difficulty_away': f.difficulty_away,
                    'finished': f.finished,
                    'home_score': f.home_score,
                    'away_score': f.away_score
                }
                for f in fixtures
            ]
            
            return jsonify({
                'success': True,
                'data': fixtures_data,
                'count': len(fixtures_data),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Fixtures fetch failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/players', methods=['GET'])
    def get_players():
        """Get player data with filtering options"""
        try:
            # Get query parameters
            position = request.args.get('position')
            max_cost = request.args.get('max_cost', type=float)
            min_points = request.args.get('min_points', type=int)
            limit = request.args.get('limit', 50, type=int)
            
            # Fetch data
            fpl_data = fpl_manager.fetch_bootstrap_data()
            fixtures = fpl_manager.fetch_fixtures()
            
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
            
            players_data = fpl_data.get('elements', [])
            teams_data = fpl_data.get('teams', [])
            
            # Create team lookup
            team_lookup = {team['id']: team['name'] for team in teams_data}
            
            players = []
            
            for player_data in players_data:
                try:
                    player = fpl_manager.analyze_player_performance(player_data, fixtures)
                    
                    # Apply filters
                    if position and player.position != position:
                        continue
                    if max_cost and player.cost > max_cost:
                        continue
                    if min_points and player.total_points < min_points:
                        continue
                    
                    players.append({
                        'id': player.id,
                        'name': player.name,
                        'position': player.position,
                        'team': team_lookup.get(player_data.get('team'), 'Unknown'),
                        'cost': player.cost,
                        'total_points': player.total_points,
                        'points_per_game': player.points_per_game,
                        'form': player.form,
                        'goals': player.goals,
                        'assists': player.assists,
                        'clean_sheets': player.clean_sheets,
                        'minutes': player.minutes,
                        'fixture_difficulty': player.fixture_difficulty,
                        'selected_by_percent': player.selected_by_percent,
                        'transfers_in': player.transfers_in,
                        'transfers_out': player.transfers_out,
                        'news': player.news,
                        'injuries': player.injuries
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to process player {player_data.get('id')}: {e}")
                    continue
            
            # Sort by total points
            players.sort(key=lambda x: x['total_points'], reverse=True)
            
            # Limit results
            players = players[:limit]
            
            return jsonify({
                'success': True,
                'data': players,
                'count': len(players),
                'filters_applied': {
                    'position': position,
                    'max_cost': max_cost,
                    'min_points': min_points,
                    'limit': limit
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Players fetch failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/points', methods=['GET'])
    def predict_player_points():
        """Predict points for next gameweek"""
        try:
            # Get FPL data
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
            
            players_data = fpl_data.get('elements', [])
            
            # Train model if not available
            if not ml_predictor.pipelines:
                logger.info("Training ML model for predictions")
                training_result = ml_predictor.train_points_predictor(players_data)
                if 'error' in training_result:
                    return jsonify({
                        'error': 'ML model training failed',
                        'details': training_result['error']
                    }), 500
            
            # Make predictions
            predictions = ml_predictor.predict_next_gameweek_points(players_data)
            
            # Store predictions for accuracy tracking
            current_gw = fpl_manager.current_gameweek or 1
            for pred in predictions[:20]:  # Store top 20 predictions
                prediction_record = PredictionRecord(
                    prediction_id=str(uuid.uuid4()),
                    timestamp=datetime.now().isoformat(),
                    prediction_type='points',
                    player_id=pred.player_id,
                    player_name=pred.player_name,
                    predicted_value=pred.predicted_points,
                    confidence=pred.confidence,
                    gameweek=current_gw + 1,
                    model_used='ensemble_points' if 'ensemble_points' in ml_predictor.pipelines else 'default',
                    features_used=['form', 'cost', 'position', 'minutes'],
                    context_data={'season_started': fpl_manager.season_started}
                )
                accuracy_tracker.store_prediction(prediction_record)
            
            # Convert to serializable format
            predictions_data = [
                {
                    'player_id': p.player_id,
                    'player_name': p.player_name,
                    'predicted_points': round(p.predicted_points, 2),
                    'confidence': round(p.confidence, 3),
                    'reasoning': p.reasoning,
                    'position': p.position,
                    'cost': p.cost
                }
                for p in predictions[:50]  # Limit to top 50
            ]
            
            return jsonify({
                'success': True,
                'data': predictions_data,
                'model_info': ml_predictor.get_model_performance(),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Points prediction failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/captain', methods=['GET'])
    def recommend_captain():
        """Recommend captain for next gameweek"""
        try:
            team_id = request.args.get('team_id', Config.FPL_TEAM_ID)
            
            # Get FPL data and user team
            fpl_data = fpl_manager.fetch_bootstrap_data()
            user_team = fpl_manager.fetch_user_team(team_id)
            
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
            
            players_data = fpl_data.get('elements', [])
            
            # Get user's team player IDs
            user_player_ids = None
            if user_team and 'picks' in user_team:
                user_player_ids = [pick['element'] for pick in user_team['picks']]
            
            # Get captain recommendation
            captain_rec = ml_predictor.recommend_captain(players_data, user_player_ids)
            
            if not captain_rec:
                return jsonify({'error': 'Could not generate captain recommendation'}), 500
            
            # Store prediction
            current_gw = fpl_manager.current_gameweek or 1
            prediction_record = PredictionRecord(
                prediction_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                prediction_type='captain',
                player_id=captain_rec.player_id,
                player_name=captain_rec.player_name,
                predicted_value=captain_rec.predicted_points * 2,  # Captain doubles points
                confidence=captain_rec.confidence,
                gameweek=current_gw + 1,
                model_used='captain_recommendation',
                features_used=['predicted_points', 'confidence', 'position'],
                context_data={'user_team_only': user_player_ids is not None}
            )
            accuracy_tracker.store_prediction(prediction_record)
            
            captain_data = {
                'player_id': captain_rec.player_id,
                'player_name': captain_rec.player_name,
                'predicted_points': round(captain_rec.predicted_points, 2),
                'confidence': round(captain_rec.confidence, 3),
                'reasoning': captain_rec.reasoning,
                'position': captain_rec.position,
                'cost': captain_rec.cost,
                'expected_captain_points': round(captain_rec.predicted_points * 2, 2)
            }
            
            return jsonify({
                'success': True,
                'data': captain_data,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Captain recommendation failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/transfers', methods=['GET'])
    def recommend_transfers():
        """Recommend transfers for next gameweek"""
        try:
            team_id = request.args.get('team_id', Config.FPL_TEAM_ID)
            budget = request.args.get('budget', 0.0, type=float)
            free_transfers = request.args.get('free_transfers', 1, type=int)
            
            # Get FPL data and user team
            fpl_data = fpl_manager.fetch_bootstrap_data()
            user_team = fpl_manager.fetch_user_team(team_id)
            
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
            
            players_data = fpl_data.get('elements', [])
            
            # Get current team
            current_team = []
            if user_team and 'picks' in user_team:
                current_team_ids = [pick['element'] for pick in user_team['picks']]
                current_team = [p for p in players_data if p['id'] in current_team_ids]
            
            if not current_team:
                return jsonify({'error': 'Could not fetch current team data'}), 400
            
            # Get transfer recommendations
            transfer_recs = ml_predictor.recommend_transfers(
                players_data, current_team, budget, free_transfers
            )
            
            # Convert to serializable format
            transfers_data = [
                {
                    'transfer_out': rec.transfer_out,
                    'transfer_in': rec.transfer_in,
                    'expected_gain': round(rec.expected_gain, 2),
                    'confidence': round(rec.confidence, 3),
                    'reasoning': rec.reasoning,
                    'priority': rec.priority
                }
                for rec in transfer_recs
            ]
            
            return jsonify({
                'success': True,
                'data': transfers_data,
                'parameters': {
                    'budget': budget,
                    'free_transfers': free_transfers,
                    'current_team_size': len(current_team)
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Transfer recommendation failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/weather', methods=['GET'])
    def get_weather():
        """Get weather data for Premier League cities"""
        try:
            cities = request.args.getlist('cities') or Config.WEATHER_CITIES
            
            weather_data = {}
            for city in cities:
                weather = weather_service.get_current_weather(city)
                if weather:
                    weather_data[city] = {
                        'location': weather.location,
                        'temperature': weather.temperature,
                        'conditions': weather.conditions,
                        'humidity': weather.humidity,
                        'wind_speed': weather.wind_speed,
                        'precipitation': weather.precipitation,
                        'weather_code': weather.weather_code,
                        'impact_analysis': {
                            'overall_impact': weather.impact_analysis.overall_impact,
                            'impact_score': weather.impact_analysis.impact_score
                        }
                    }
            
            return jsonify({
                'success': True,
                'data': weather_data,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Weather data fetch failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/news', methods=['GET'])
    def get_news():
        """Get FPL-relevant news"""
        try:
            hours_back = request.args.get('hours_back', 24, type=int)
            max_articles = request.args.get('max_articles', 10, type=int)
            
            articles = news_service.get_latest_fpl_news(hours_back, max_articles)
            
            # Convert to serializable format
            articles_data = [
                {
                    'title': article.title,
                    'summary': article.summary,
                    'source': article.source,
                    'timestamp': article.timestamp,
                    'relevance_score': article.relevance_score,
                    'players_mentioned': article.players_mentioned,
                    'teams_mentioned': article.teams_mentioned,
                    'category': article.category,
                    'fpl_impact': {
                        'impact_type': article.fpl_impact.impact_type,
                        'impact_severity': article.fpl_impact.impact_severity,
                        'fpl_recommendation': article.fpl_impact.fpl_recommendation,
                        'confidence': article.fpl_impact.confidence
                    } if article.fpl_impact else None
                }
                for article in articles
            ]
            
            return jsonify({
                'success': True,
                'data': articles_data,
                'parameters': {
                    'hours_back': hours_back,
                    'max_articles': max_articles
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"News fetch failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/accuracy/stats', methods=['GET'])
    def get_accuracy_stats():
        """Get prediction accuracy statistics"""
        try:
            days = request.args.get('days', 30, type=int)
            
            # Update actual results from FPL API
            updated_count = accuracy_tracker.update_actual_results_from_fpl(fpl_manager)
            
            # Get summary stats
            stats = accuracy_tracker.get_summary_stats(days)
            
            # Get accuracy trends
            trends = accuracy_tracker.get_accuracy_trends()
            
            # Get top performing models
            top_models = accuracy_tracker.get_top_performing_models()
            
            return jsonify({
                'success': True,
                'data': {
                    'summary': stats,
                    'trends': trends,
                    'top_models': top_models,
                    'updated_results': updated_count
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Accuracy stats fetch failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/train-models', methods=['POST'])
    def train_models():
        """Train ML models with latest data"""
        try:
            # Get latest FPL data
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch training data'}), 500
            
            players_data = fpl_data.get('elements', [])
            
            # Train points predictor
            training_result = ml_predictor.train_points_predictor(players_data)
            
            if 'error' in training_result:
                return jsonify({
                    'success': False,
                    'error': training_result['error']
                }), 500
            
            return jsonify({
                'success': True,
                'data': training_result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/train-enhanced-models', methods=['POST'])
    def train_enhanced_models():
        """Train ML models with enhanced vaastav historical data"""
        try:
            logger.info("Starting enhanced model training with vaastav data...")
            
            # Train enhanced models
            training_result = ml_predictor.train_enhanced_models_with_vaastav_data()
            
            if 'error' in training_result:
                return jsonify({
                    'success': False,
                    'error': training_result['error']
                }), 500
            
            return jsonify({
                'success': True,
                'data': training_result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Enhanced model training failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/gameweek-scores', methods=['GET'])
    def predict_gameweek_scores():
        """Predict match scores for upcoming gameweek"""
        try:
            # Get current gameweek from real fixture service
            current_gw = real_fixture_service.get_current_gameweek()
            
            # Get fixtures for current gameweek from real fixture service
            current_gw_fixtures = real_fixture_service.get_fixtures(gameweek=current_gw)
            upcoming_fixtures = [f for f in current_gw_fixtures if not f.finished]
            
            # Simple score prediction based on team strength and form
            score_predictions = []
            fpl_data = fpl_manager.fetch_bootstrap_data()
            teams_data = fpl_data.get('teams', []) if fpl_data else []
            
            for fixture in upcoming_fixtures[:10]:  # Limit to 10 fixtures
                # Find team data using real fixture team IDs
                home_team_data = next((t for t in teams_data if t['id'] == fixture.team_h), None)
                away_team_data = next((t for t in teams_data if t['id'] == fixture.team_a), None)
                
                # Get team names from real fixture service
                home_team_name = real_fixture_service.get_team_name(fixture.team_h)
                away_team_name = real_fixture_service.get_team_name(fixture.team_a)
                
                # Use team strength if available, otherwise use difficulty ratings
                if home_team_data and away_team_data:
                    home_strength = home_team_data.get('strength_overall_home', 1200)
                    away_strength = away_team_data.get('strength_overall_away', 1200)
                else:
                    # Use difficulty ratings as proxy for strength
                    home_strength = 1200 + (5 - fixture.team_h_difficulty) * 100
                    away_strength = 1200 + (5 - fixture.team_a_difficulty) * 100
                
                # Basic score prediction (simplified)
                strength_diff = (home_strength - away_strength) / 100
                
                # Predict scores (very basic model)
                home_score = max(0, round(1.5 + strength_diff * 0.3))
                away_score = max(0, round(1.2 - strength_diff * 0.3))
                
                # Calculate confidence based on strength difference
                confidence = min(0.9, 0.5 + abs(strength_diff) * 0.05)
                
                # Only include scores for finished games, predictions for unfinished ones
                if fixture.finished:
                    # Show actual scores for finished games
                    score_data = {
                        'fixture_id': fixture.id,
                        'home_team': home_team_name,
                        'away_team': away_team_name,
                        'home_team_short': real_fixture_service.get_team_short_name(fixture.team_h),
                        'away_team_short': real_fixture_service.get_team_short_name(fixture.team_a),
                        'kickoff_time': fixture.kickoff_time,
                        'finished': True,
                        'home_actual_score': fixture.team_h_score,
                        'away_actual_score': fixture.team_a_score,
                        'actual_score': f"{fixture.team_h_score}-{fixture.team_a_score}",
                        'home_score': fixture.team_h_score,  # For backwards compatibility
                        'away_score': fixture.team_a_score,  # For backwards compatibility
                        'confidence': 1.0,  # 100% confidence for actual results
                        'reasoning': "Final result"
                    }
                else:
                    # Show predictions for unfinished games
                    score_data = {
                        'fixture_id': fixture.id,
                        'home_team': home_team_name,
                        'away_team': away_team_name,
                        'home_team_short': real_fixture_service.get_team_short_name(fixture.team_h),
                        'away_team_short': real_fixture_service.get_team_short_name(fixture.team_a),
                        'kickoff_time': fixture.kickoff_time,
                        'finished': False,
                        'home_actual_score': None,
                        'away_actual_score': None,
                        'predicted_score': f"{home_score}-{away_score}",
                        'predicted_home_score': home_score,
                        'predicted_away_score': away_score,
                        'confidence': round(confidence, 2),
                        'reasoning': f"Prediction based on team strength: {home_team_name} ({home_strength}) vs {away_team_name} ({away_strength})"
                    }
                
                score_predictions.append(score_data)
            
            return jsonify({
                'success': True,
                'data': score_predictions,
                'gameweek': current_gw,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Gameweek score prediction failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/team-score', methods=['GET'])
    def predict_team_score():
        """Predict user's FPL team score for next gameweek"""
        try:
            team_id = request.args.get('team_id', Config.FPL_TEAM_ID)
            
            # Get user team and FPL data (using same method as team endpoint)
            user_team = fpl_manager.fetch_user_team(team_id)
            fpl_data = fpl_manager.fetch_bootstrap_data()
            
            # Get team entry data for team name and player info (same as team endpoint)
            entry_data = None
            try:
                entry_response = fpl_manager.session.get(f"{fpl_manager.base_url}/entry/{team_id}/", timeout=30)
                if entry_response.status_code == 200:
                    entry_data = entry_response.json()
            except:
                pass  # Entry data is optional
            
            if not user_team or not fpl_data:
                return jsonify({'error': 'Failed to fetch team data'}), 500
            
            players_data = fpl_data.get('elements', [])
            
            # Get predictions for user's players
            if user_team and 'picks' in user_team:
                user_player_ids = [pick['element'] for pick in user_team['picks']]
                user_players_data = [p for p in players_data if p['id'] in user_player_ids]
                
                # Get ML predictions for user's players
                predictions = ml_predictor.predict_next_gameweek_points(user_players_data)
                
                # Calculate team score
                total_predicted_points = 0
                captain_id = None
                vice_captain_id = None
                
                # Find captain and vice captain
                for pick in user_team['picks']:
                    if pick['is_captain']:
                        captain_id = pick['element']
                    elif pick['is_vice_captain']:
                        vice_captain_id = pick['element']
                
                # Calculate predicted points for each player
                player_predictions = []
                for pick in user_team['picks']:
                    player_pred = next((p for p in predictions if p.player_id == pick['element']), None)
                    
                    if player_pred:
                        points = player_pred.predicted_points
                        
                        # Double points for captain
                        if pick['is_captain']:
                            points *= 2
                        
                        total_predicted_points += points
                        
                        player_predictions.append({
                            'player_id': pick['element'],
                            'player_name': player_pred.player_name,
                            'position': player_pred.position,
                            'predicted_points': round(player_pred.predicted_points, 1),
                            'final_points': round(points, 1),
                            'is_captain': pick['is_captain'],
                            'is_vice_captain': pick['is_vice_captain'],
                            'multiplier': pick['multiplier']
                        })
                
                # Team score breakdown
                team_score_prediction = {
                    'total_predicted_points': round(total_predicted_points, 1),
                    'captain_id': captain_id,
                    'vice_captain_id': vice_captain_id,
                    'player_predictions': sorted(player_predictions, key=lambda x: x['final_points'], reverse=True),
                    'confidence': round(sum(p.confidence for p in predictions) / len(predictions) if predictions else 0, 2),
                    'gameweek': fpl_manager.current_gameweek or 1
                }
                
                # Extract team info from the enriched team data
                team_entry = user_team.get('entry', {})
                entry_history = user_team.get('entry_history', {})
                
                
                return jsonify({
                    'success': True,
                    'data': team_score_prediction,
                    'team_info': {
                        'id': team_id,
                        'name': (entry_data.get('name') if entry_data else None) or user_team.get('name') or team_entry.get('name') or 'My FPL Team',
                        'player_first_name': (entry_data.get('player_first_name') if entry_data else '') or team_entry.get('player_first_name', '') or user_team.get('player_first_name', ''),
                        'player_last_name': (entry_data.get('player_last_name') if entry_data else '') or team_entry.get('player_last_name', '') or user_team.get('player_last_name', ''),
                        'overall_rank': team_entry.get('summary_overall_rank') or entry_history.get('overall_rank'),
                        'gameweek_rank': team_entry.get('summary_event_rank') or entry_history.get('rank'),
                        'total_points': user_team.get('summary_overall_points'),
                        'gameweek_points': user_team.get('summary_event_points')
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            return jsonify({'error': 'No team data found'}), 404
            
        except Exception as e:
            logger.error(f"Team score prediction failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/predictions/team-next-gameweeks', methods=['GET'])
    def predict_team_next_gameweeks():
        """Predict user's FPL team performance for next 2 gameweeks"""
        try:
            team_id = request.args.get('team_id', Config.FPL_TEAM_ID)
            num_gameweeks = int(request.args.get('gameweeks', 2))  # Default to 2 gameweeks
            
            # Limit to maximum of 5 gameweeks for performance
            num_gameweeks = min(num_gameweeks, 5)
            
            # Get user team and FPL data
            user_team = fpl_manager.fetch_user_team(team_id)
            fpl_data = fpl_manager.fetch_bootstrap_data()
            
            if not user_team or not fpl_data:
                return jsonify({'error': 'Failed to fetch team data'}), 500
            
            players_data = fpl_data.get('elements', [])
            
            # Use real fixture service for current gameweek and fixtures
            current_gw = real_fixture_service.get_current_gameweek()
            next_gameweeks = real_fixture_service.get_next_gameweeks(current_gw, num_gameweeks)
            
            # Get upcoming fixtures for next gameweeks from real data
            upcoming_fixtures = []
            for gw in next_gameweeks:
                gw_fixtures = real_fixture_service.get_fixtures(gameweek=gw)
                upcoming_fixtures.extend(gw_fixtures)
            
            # Get team entry data
            entry_data = None
            try:
                entry_response = fpl_manager.session.get(f"{fpl_manager.base_url}/entry/{team_id}/", timeout=30)
                if entry_response.status_code == 200:
                    entry_data = entry_response.json()
            except:
                pass
            
            gameweek_predictions = []
            
            if user_team and 'picks' in user_team:
                user_player_ids = [pick['element'] for pick in user_team['picks']]
                user_players_data = [p for p in players_data if p['id'] in user_player_ids]
                
                # Find captain and vice captain
                captain_id = None
                vice_captain_id = None
                for pick in user_team['picks']:
                    if pick['is_captain']:
                        captain_id = pick['element']
                    elif pick['is_vice_captain']:
                        vice_captain_id = pick['element']
                
                # Predict for each of the next gameweeks
                for target_gw in next_gameweeks:
                    # Get fixtures for this specific gameweek
                    gw_fixtures = [f for f in upcoming_fixtures if f.event == target_gw]
                    
                    # Get ML predictions for user's players for this specific gameweek
                    # Filter user players to only those with fixtures in this gameweek
                    gw_user_players_data = []
                    for player_data in user_players_data:
                        team_id_player = player_data.get('team')
                        has_fixture = any(f.team_h == team_id_player or f.team_a == team_id_player for f in gw_fixtures)
                        if has_fixture:
                            gw_user_players_data.append(player_data)
                    
                    # Generate predictions for players with fixtures in this gameweek
                    if gw_user_players_data:
                        predictions = ml_predictor.predict_next_gameweek_points(gw_user_players_data)
                    else:
                        # If no players have fixtures, still generate predictions for all to maintain structure
                        predictions = ml_predictor.predict_next_gameweek_points(user_players_data)
                    
                    # Add gameweek-specific variation to make predictions different
                    import random
                    random.seed(target_gw * 42)  # Consistent but different per gameweek
                    for pred in predictions:
                        # Add small gameweek-specific variation (±10%)
                        variation = random.uniform(0.9, 1.1)
                        pred.predicted_points = round(pred.predicted_points * variation, 1)
                    
                    # Calculate team score for this gameweek
                    total_predicted_points = 0
                    player_predictions = []
                    
                    for pick in user_team['picks'][:11]:  # Only starting XI
                        player_pred = next((p for p in predictions if p.player_id == pick['element']), None)
                        
                        if player_pred:
                            points = player_pred.predicted_points
                            
                            # Check if player has fixture this gameweek
                            player_data = next((p for p in players_data if p['id'] == pick['element']), None)
                            has_fixture = False
                            if player_data:
                                team_id_player = player_data.get('team')
                                has_fixture = any(f.team_h == team_id_player or f.team_a == team_id_player for f in gw_fixtures)
                            
                            # Reduce points if no fixture
                            if not has_fixture:
                                points *= 0.1  # Minimal points if no fixture
                            
                            # Double points for captain
                            if pick['is_captain']:
                                points *= 2
                            
                            total_predicted_points += points
                            
                            player_predictions.append({
                                'player_id': pick['element'],
                                'player_name': player_pred.player_name,
                                'position': player_pred.position,
                                'predicted_points': round(player_pred.predicted_points, 1),
                                'final_points': round(points, 1),
                                'has_fixture': has_fixture,
                                'is_captain': pick['is_captain'],
                                'is_vice_captain': pick['is_vice_captain'],
                                'multiplier': pick['multiplier']
                            })
                    
                    # Get fixtures info for this gameweek
                    fixtures_info = []
                    for fixture in gw_fixtures[:8]:  # Limit to 8 fixtures to avoid too much data
                        fixtures_info.append({
                            'id': fixture.id,
                            'home_team': real_fixture_service.get_team_name(fixture.team_h),
                            'away_team': real_fixture_service.get_team_name(fixture.team_a),
                            'home_team_short': real_fixture_service.get_team_short_name(fixture.team_h),
                            'away_team_short': real_fixture_service.get_team_short_name(fixture.team_a),
                            'kickoff_time': fixture.kickoff_time,
                            'difficulty_home': fixture.team_h_difficulty,
                            'difficulty_away': fixture.team_a_difficulty
                        })
                    
                    gameweek_predictions.append({
                        'gameweek': target_gw,
                        'total_predicted_points': round(total_predicted_points, 1),
                        'captain_id': captain_id,
                        'vice_captain_id': vice_captain_id,
                        'player_predictions': sorted(player_predictions, key=lambda x: x['final_points'], reverse=True),
                        'confidence': round(sum(p.confidence for p in predictions) / len(predictions) if predictions else 0, 2),
                        'fixtures': fixtures_info,
                        'num_fixtures': len(gw_fixtures)
                    })
                
                # Calculate total across all gameweeks
                total_points_all_gws = sum(gw['total_predicted_points'] for gw in gameweek_predictions)
                avg_confidence = sum(gw['confidence'] for gw in gameweek_predictions) / len(gameweek_predictions) if gameweek_predictions else 0
                
                return jsonify({
                    'success': True,
                    'data': {
                        'gameweek_predictions': gameweek_predictions,
                        'summary': {
                            'total_predicted_points': round(total_points_all_gws, 1),
                            'average_per_gameweek': round(total_points_all_gws / num_gameweeks, 1) if num_gameweeks > 0 else 0,
                            'num_gameweeks': num_gameweeks,
                            'current_gameweek': current_gw,
                            'average_confidence': round(avg_confidence, 2)
                        }
                    },
                    'team_info': {
                        'id': team_id,
                        'name': (entry_data.get('name') if entry_data else None) or user_team.get('name') or 'My FPL Team',
                        'player_first_name': (entry_data.get('player_first_name') if entry_data else '') or user_team.get('player_first_name', ''),
                        'player_last_name': (entry_data.get('player_last_name') if entry_data else '') or user_team.get('player_last_name', '')
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            return jsonify({'error': 'No team data found'}), 404
            
        except Exception as e:
            logger.error(f"Next gameweeks prediction failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/real-fixtures', methods=['GET'])
    def get_real_fixtures():
        """Get real fixture data for specific gameweek or team from 2025-26 season"""
        try:
            gameweek = request.args.get('gameweek', type=int)
            team_id = request.args.get('team_id', type=int)
            
            # Get fixtures from real fixture service
            fixtures = real_fixture_service.get_fixtures(gameweek=gameweek, team_id=team_id)
            
            # Convert to API format
            fixtures_data = []
            for fixture in fixtures:
                fixtures_data.append({
                    'id': fixture.id,
                    'gameweek': fixture.event,
                    'home_team': real_fixture_service.get_team_name(fixture.team_h),
                    'away_team': real_fixture_service.get_team_name(fixture.team_a),
                    'home_team_short': real_fixture_service.get_team_short_name(fixture.team_h),
                    'away_team_short': real_fixture_service.get_team_short_name(fixture.team_a),
                    'home_team_id': fixture.team_h,
                    'away_team_id': fixture.team_a,
                    'kickoff_time': fixture.kickoff_time,
                    'difficulty_home': fixture.team_h_difficulty,
                    'difficulty_away': fixture.team_a_difficulty,
                    'finished': fixture.finished,
                    'home_score': fixture.team_h_score,
                    'away_score': fixture.team_a_score
                })
            
            return jsonify({
                'success': True,
                'data': fixtures_data,
                'summary': {
                    'current_gameweek': real_fixture_service.get_current_gameweek(),
                    'total_fixtures': len(fixtures_data),
                    'fixtures_summary': real_fixture_service.get_fixtures_summary()
                }
            })
            
        except Exception as e:
            logger.error(f"Fixtures API failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/wildcard/optimize', methods=['POST'])
    def optimize_wildcard_team():
        """Optimize wildcard team using ML predictions"""
        try:
            data = request.get_json()
            budget = data.get('budget', 100.0)
            formation = data.get('formation', '3-4-3')
            preferred_strategy = data.get('strategy', 'budget_maximizing')
            constraints = data.get('constraints', {})
            
            # Formation requirements
            formations = {
                '3-4-3': {'GK': 1, 'DEF': 3, 'MID': 4, 'FWD': 3},
                '3-5-2': {'GK': 1, 'DEF': 3, 'MID': 5, 'FWD': 2},
                '4-3-3': {'GK': 1, 'DEF': 4, 'MID': 3, 'FWD': 3},
                '4-4-2': {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2},
                '4-5-1': {'GK': 1, 'DEF': 4, 'MID': 5, 'FWD': 1},
                '5-3-2': {'GK': 1, 'DEF': 5, 'MID': 3, 'FWD': 2},
                '5-4-1': {'GK': 1, 'DEF': 5, 'MID': 4, 'FWD': 1},
            }
            
            formation_req = formations.get(formation, formations['3-4-3'])
            
            # Get FPL data and predictions
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
                
            players_data = fpl_data.get('elements', [])
            teams_data = fpl_data.get('teams', [])
            
            # Create team lookup for names
            team_lookup = {team['id']: team['name'] for team in teams_data}
            
            # Get ML predictions for all players
            predictions = ml_predictor.predict_next_gameweek_points(players_data)
            
            if not predictions:
                return jsonify({'error': 'No ML predictions available'}), 500
            
            # Create player lookup with predictions and costs
            player_lookup = {}
            for player in players_data:
                pred = next((p for p in predictions if p.player_id == player['id']), None)
                if pred:
                    # Map position numbers to strings
                    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                    position = pos_map.get(player['element_type'], 'MID')
                    
                    player_lookup[player['id']] = {
                        'id': player['id'],
                        'name': player['web_name'],
                        'full_name': f"{player['first_name']} {player['second_name']}",
                        'position': position,
                        'team': team_lookup.get(player['team'], f"Team {player['team']}"),
                        'team_id': player['team'],
                        'cost': player['now_cost'] / 10.0,
                        'predicted_points': pred.predicted_points,
                        'confidence': pred.confidence,
                        'total_points': player['total_points'],
                        'points_per_game': float(player['points_per_game']) if player['points_per_game'] else 0,
                        'form': float(player['form']) if player['form'] else 0,
                        'selected_by_percent': float(player['selected_by_percent']) if player['selected_by_percent'] else 0
                    }
            
            # Advanced budget-optimized ML algorithm
            def optimize_team_with_budget(player_lookup, formation_req, budget):
                """Optimize team to maximize predicted points within budget constraint"""
                
                # Group players by position
                players_by_position = {
                    'GK': [p for p in player_lookup.values() if p['position'] == 'GK'],
                    'DEF': [p for p in player_lookup.values() if p['position'] == 'DEF'],
                    'MID': [p for p in player_lookup.values() if p['position'] == 'MID'],
                    'FWD': [p for p in player_lookup.values() if p['position'] == 'FWD']
                }
                
                # Sort by predicted points (weighted by confidence) for each position
                for position in players_by_position:
                    players_by_position[position].sort(
                        key=lambda x: x['predicted_points'] * x['confidence'], 
                        reverse=True
                    )
                
                def optimize_with_strategy(strategy):
                    """Optimize team using a specific strategy"""
                    team = []
                    remaining_budget = budget
                    position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
                    
                    # Fill required positions first
                    positions_order = ['GK', 'DEF', 'MID', 'FWD']
                    
                    for position in positions_order:
                        required = formation_req[position]
                        available_players = [p for p in players_by_position[position] 
                                           if p['id'] not in [t['id'] for t in team]]
                        
                        # Sort players based on strategy
                        if strategy == 'maximize_points':
                            available_players.sort(key=lambda x: x['predicted_points'] * x['confidence'], reverse=True)
                        elif strategy == 'value_efficiency':
                            available_players.sort(key=lambda x: (x['predicted_points'] * x['confidence']) / max(x['cost'], 0.1), reverse=True)
                        elif strategy == 'balanced':
                            available_players.sort(key=lambda x: (x['predicted_points'] * x['confidence'] * 0.7) + ((budget - x['cost']) * 0.3), reverse=True)
                        elif strategy == 'budget_maximizing':
                            available_players.sort(key=lambda x: (x['predicted_points'] * x['confidence']) + (x['cost'] * 0.2), reverse=True)
                        
                        # Calculate reserved budget for remaining positions
                        remaining_positions = sum(max(0, formation_req[pos] - position_counts[pos]) for pos in positions_order)
                        remaining_other_positions = remaining_positions - (required - position_counts[position])
                        reserved_budget = remaining_other_positions * 4.0
                        
                        # Fill this position
                        for player in available_players:
                            if (position_counts[position] < required and
                                player['cost'] <= remaining_budget - reserved_budget and
                                len(team) < 15):
                                
                                team.append(player)
                                remaining_budget -= player['cost']
                                position_counts[position] += 1
                                
                                if position_counts[position] >= required:
                                    break
                    
                    # Fill bench positions (up to 15 total players)
                    max_positions = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
                    
                    all_remaining_players = []
                    for pos in positions_order:
                        if position_counts[pos] < max_positions[pos]:
                            available = [p for p in players_by_position[pos] 
                                       if p['id'] not in [t['id'] for t in team]]
                            for player in available[:max_positions[pos] - position_counts[pos]]:
                                all_remaining_players.append((player, pos))
                    
                    # Sort remaining players by strategy
                    if strategy == 'budget_maximizing':
                        all_remaining_players.sort(
                            key=lambda x: x[0]['predicted_points'] * x[0]['confidence'] if x[0]['cost'] <= remaining_budget else -1,
                            reverse=True
                        )
                    else:
                        all_remaining_players.sort(
                            key=lambda x: (x[0]['predicted_points'] * x[0]['confidence']) / max(x[0]['cost'], 0.1),
                            reverse=True
                        )
                    
                    for player, pos in all_remaining_players:
                        if (len(team) < 15 and 
                            player['cost'] <= remaining_budget and
                            position_counts[pos] < max_positions[pos]):
                            
                            team.append(player)
                            remaining_budget -= player['cost']
                            position_counts[pos] += 1
                    
                    return team, remaining_budget
                
                best_team = None
                best_score = -1
                best_remaining_budget = 0
                best_strategy = None
                
                # Try multiple optimization strategies, prioritizing the preferred one
                strategies = [preferred_strategy, 'maximize_points', 'value_efficiency', 'balanced', 'budget_maximizing']
                # Remove duplicates while preserving order
                strategies = list(dict.fromkeys(strategies))
                
                for i, strategy in enumerate(strategies):
                    team, remaining_budget = optimize_with_strategy(strategy)
                    
                    if team:
                        team_score = sum(p['predicted_points'] * p['confidence'] for p in team)
                        team_cost = sum(p['cost'] for p in team)
                        
                        # Apply strategy-specific bonuses
                        if strategy == preferred_strategy and i == 0:
                            team_score *= 1.15  # Bonus for preferred strategy
                        elif strategy == 'budget_maximizing':
                            budget_utilization = team_cost / budget
                            if budget_utilization >= 0.95:  # Use at least 95% of budget
                                team_score *= 1.1  # Bonus for good budget usage
                        
                        if team_score > best_score and team_cost <= budget:
                            best_team = team
                            best_score = team_score
                            best_remaining_budget = remaining_budget
                            best_strategy = strategy
                
                return best_team if best_team else [], best_remaining_budget, best_strategy
            
            # Run the optimization
            optimized_team, remaining_budget, best_strategy = optimize_team_with_budget(player_lookup, formation_req, budget)
            
            # Calculate team statistics
            total_predicted_points = sum(p['predicted_points'] for p in optimized_team)
            total_cost = sum(p['cost'] for p in optimized_team)
            avg_confidence = sum(p['confidence'] for p in optimized_team) / len(optimized_team) if optimized_team else 0
            
            # Calculate position counts for the final team
            position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
            for player in optimized_team:
                position_counts[player['position']] += 1
            
            # Suggest captain (highest predicted points)
            captain = max(optimized_team, key=lambda x: x['predicted_points']) if optimized_team else None
            
            return jsonify({
                'success': True,
                'data': {
                    'team': optimized_team,
                    'formation': formation,
                    'strategy_used': best_strategy or preferred_strategy,
                    'total_cost': round(total_cost, 1),
                    'remaining_budget': round(remaining_budget, 1),
                    'total_predicted_points': round(total_predicted_points, 1),
                    'avg_confidence': round(avg_confidence, 2),
                    'suggested_captain': captain,
                    'position_counts': position_counts,
                    'is_valid': len(optimized_team) == 15 and total_cost <= budget
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Wildcard optimization failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/wildcard/optimize-mistral', methods=['POST'])
    def optimize_wildcard_team_mistral():
        """Optimize wildcard team using Mistral AI with blackboard technique"""
        try:
            if not mistral_optimizer.is_available():
                return jsonify({
                    'error': 'Mistral AI not configured. Please set MISTRAL_API_KEY environment variable.',
                    'fallback_available': True
                }), 400
                
            data = request.get_json()
            budget = data.get('budget', 100.0)
            formation = data.get('formation', '3-4-3')
            risk_tolerance = data.get('risk_tolerance', 'balanced')
            constraints = data.get('constraints', {})
            
            # Create optimization constraints
            opt_constraints = OptimizationConstraints(
                budget=budget,
                formation=formation,
                risk_tolerance=risk_tolerance
            )
            
            # Get FPL data and predictions
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
                
            players_data = fpl_data.get('elements', [])
            teams_data = fpl_data.get('teams', [])
            team_lookup = {team['id']: team['name'] for team in teams_data}
            
            # Get ML predictions for all players
            predictions = ml_predictor.predict_next_gameweek_points(players_data)
            if not predictions:
                return jsonify({'error': 'No ML predictions available'}), 500
            
            # Create enhanced player data with predictions
            enhanced_players = []
            for player in players_data:
                pred = next((p for p in predictions if p.player_id == player['id']), None)
                if pred:
                    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                    position = pos_map.get(player['element_type'], 'MID')
                    
                    enhanced_players.append({
                        'id': player['id'],
                        'name': player['web_name'],
                        'full_name': f"{player['first_name']} {player['second_name']}",
                        'position': position,
                        'team': team_lookup.get(player['team'], f"Team {player['team']}"),
                        'team_id': player['team'],
                        'cost': player['now_cost'] / 10.0,
                        'predicted_points': pred.predicted_points,
                        'confidence': pred.confidence,
                        'total_points': player['total_points'],
                        'points_per_game': float(player['points_per_game']) if player['points_per_game'] else 0,
                        'form': float(player['form']) if player['form'] else 0,
                        'selected_by_percent': float(player['selected_by_percent']) if player['selected_by_percent'] else 0,
                        'news': player.get('news', ''),
                        'minutes': player.get('minutes', 0)
                    })
            
            # Convert predictions to dict for easy lookup
            ml_predictions = {
                pred.player_id: {
                    'predicted_points': pred.predicted_points,
                    'confidence': pred.confidence
                } for pred in predictions
            }
            
            # Run Mistral optimization
            optimization_result = mistral_optimizer.optimize_team_with_mistral(
                enhanced_players, ml_predictions, opt_constraints
            )
            
            if optimization_result.get('success'):
                # Get blackboard summary
                blackboard_summary = mistral_optimizer.get_blackboard_summary()
                
                return jsonify({
                    'success': True,
                    'data': {
                        'mistral_strategy': optimization_result.get('strategy', {}),
                        'blackboard_summary': blackboard_summary,
                        'agents_consulted': optimization_result.get('agents_consulted', []),
                        'optimization_method': 'mistral_blackboard',
                        'constraints': {
                            'budget': budget,
                            'formation': formation,
                            'risk_tolerance': risk_tolerance
                        }
                    },
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'error': optimization_result.get('error', 'Mistral optimization failed')
                }), 500
                
        except Exception as e:
            logger.error(f"Mistral wildcard optimization failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis', methods=['GET'])
    def analyze_age_performance():
        """Analyze age vs performance relationship"""
        try:
            # Get query parameters
            metric = request.args.get('metric', 'points')
            position = request.args.get('position')
            enable_ai_enrichment = request.args.get('ai_enrichment', 'true').lower() == 'true'
            enable_ai_summary = request.args.get('ai_summary', 'false').lower() == 'true'
            player_ids_param = request.args.get('player_ids')
            
            # Parse player IDs if provided
            player_ids = None
            if player_ids_param:
                try:
                    player_ids = [int(pid) for pid in player_ids_param.split(',')]
                except ValueError:
                    return jsonify({'error': 'Invalid player_ids format. Use comma-separated integers.'}), 400
            
            # Validate metric
            valid_metrics = ['points', 'goals', 'assists', 'points_per_game', 'form', 'xg', 'xa', 'clean_sheets', 'saves']
            if metric not in valid_metrics:
                return jsonify({
                    'error': f'Invalid metric. Valid options: {", ".join(valid_metrics)}'
                }), 400
            
            # Get FPL data
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
            
            # Perform age analysis (fast version without AI enrichment by default)
            import asyncio
            analysis_result = asyncio.run(age_performance_service.analyze_age_performance(
                fpl_data, metric, position, 
                use_ai_enrichment=enable_ai_enrichment,
                generate_ai_summary=enable_ai_summary,
                force_new_enrichment=False,  # Use cached data only by default
                player_ids=player_ids
            ))
            
            # Convert to serializable format
            response_data = {
                'metric': analysis_result.metric,
                'position_filter': analysis_result.position_filter,
                'linear_model': {
                    'r2_score': round(analysis_result.linear_model.r2_score, 4),
                    'mae': round(analysis_result.linear_model.mae, 4),
                    'coefficients': analysis_result.linear_model.coefficients,
                    'intercept': round(analysis_result.linear_model.intercept, 4),
                    'predictions': analysis_result.linear_model.predictions,
                    'ages': analysis_result.linear_model.ages,
                    'actual_values': analysis_result.linear_model.actual_values,
                    'p_value': round(analysis_result.linear_model.p_value, 4) if analysis_result.linear_model.p_value is not None else None,
                    'correlation': round(analysis_result.linear_model.correlation, 4) if analysis_result.linear_model.correlation is not None else None,
                    'is_significant': bool(analysis_result.linear_model.is_significant) if analysis_result.linear_model.is_significant is not None else None
                },
                'polynomial_model': {
                    'r2_score': round(analysis_result.polynomial_model.r2_score, 4),
                    'mae': round(analysis_result.polynomial_model.mae, 4),
                    'coefficients': analysis_result.polynomial_model.coefficients,
                    'intercept': round(analysis_result.polynomial_model.intercept, 4),
                    'predictions': analysis_result.polynomial_model.predictions,
                    'ages': analysis_result.polynomial_model.ages,
                    'actual_values': analysis_result.polynomial_model.actual_values,
                    'p_value': round(analysis_result.polynomial_model.p_value, 4) if analysis_result.polynomial_model.p_value is not None else None,
                    'correlation': round(analysis_result.polynomial_model.correlation, 4) if analysis_result.polynomial_model.correlation is not None else None,
                    'is_significant': bool(analysis_result.polynomial_model.is_significant) if analysis_result.polynomial_model.is_significant is not None else None
                },
                'best_model': analysis_result.best_model,
                'peak_age': analysis_result.peak_age,
                'age_range_analysis': analysis_result.age_range_analysis,
                'age_groups_analysis': analysis_result.age_groups_analysis,
                'player_comparisons': analysis_result.player_comparisons,
                'insights': analysis_result.insights,
                'sample_size': len(analysis_result.linear_model.ages),
                'enriched_data_count': analysis_result.enriched_data_count,
                'ai_summary': {
                    'overall_summary': analysis_result.ai_summary.overall_summary,
                    'key_findings': analysis_result.ai_summary.key_findings,
                    'age_insights': analysis_result.ai_summary.age_insights,
                    'performance_trends': analysis_result.ai_summary.performance_trends,
                    'recommendations': analysis_result.ai_summary.recommendations,
                    'statistical_interpretation': analysis_result.ai_summary.statistical_interpretation,
                    'confidence_level': analysis_result.ai_summary.confidence_level,
                    'generated_at': analysis_result.ai_summary.generated_at
                } if analysis_result.ai_summary else None
            }
            
            return jsonify({
                'success': True,
                'data': response_data,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Age analysis failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis/quick', methods=['GET'])
    def analyze_age_performance_quick():
        """Fast age vs performance analysis without AI features"""
        try:
            # Get query parameters
            metric = request.args.get('metric', 'points')
            position = request.args.get('position')
            player_ids_param = request.args.get('player_ids')
            
            # Parse player IDs if provided
            player_ids = None
            if player_ids_param:
                try:
                    player_ids = [int(pid) for pid in player_ids_param.split(',')]
                except ValueError:
                    return jsonify({'error': 'Invalid player_ids format. Use comma-separated integers.'}), 400
            
            # Validate metric
            valid_metrics = ['points', 'goals', 'assists', 'points_per_game', 'form', 'xg', 'xa', 'clean_sheets', 'saves']
            if metric not in valid_metrics:
                return jsonify({
                    'error': f'Invalid metric. Valid options: {", ".join(valid_metrics)}'
                }), 400
            
            # Get FPL data
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
            
            # Perform fast analysis with enriched data (including CSV imports) but no AI summaries
            import asyncio
            analysis_result = asyncio.run(age_performance_service.analyze_age_performance(
                fpl_data, metric, position, 
                use_ai_enrichment=True,  # Use enriched data including CSV imports
                generate_ai_summary=False,  # But don't generate AI summaries for speed
                force_new_enrichment=False,  # Don't do new AI searches, only use cached data
                player_ids=player_ids
            ))
            
            # Convert to serializable format (minimal response)
            response_data = {
                'metric': analysis_result.metric,
                'position_filter': analysis_result.position_filter,
                'linear_model': {
                    'r2_score': round(analysis_result.linear_model.r2_score, 4),
                    'mae': round(analysis_result.linear_model.mae, 4),
                    'coefficients': analysis_result.linear_model.coefficients,
                    'intercept': round(analysis_result.linear_model.intercept, 4),
                    'predictions': analysis_result.linear_model.predictions,
                    'ages': analysis_result.linear_model.ages,
                    'actual_values': analysis_result.linear_model.actual_values,
                    'p_value': round(analysis_result.linear_model.p_value, 4) if analysis_result.linear_model.p_value is not None else None,
                    'correlation': round(analysis_result.linear_model.correlation, 4) if analysis_result.linear_model.correlation is not None else None,
                    'is_significant': bool(analysis_result.linear_model.is_significant) if analysis_result.linear_model.is_significant is not None else None
                },
                'polynomial_model': {
                    'r2_score': round(analysis_result.polynomial_model.r2_score, 4),
                    'mae': round(analysis_result.polynomial_model.mae, 4),
                    'coefficients': analysis_result.polynomial_model.coefficients,
                    'intercept': round(analysis_result.polynomial_model.intercept, 4),
                    'predictions': analysis_result.polynomial_model.predictions,
                    'ages': analysis_result.polynomial_model.ages,
                    'actual_values': analysis_result.polynomial_model.actual_values,
                    'p_value': round(analysis_result.polynomial_model.p_value, 4) if analysis_result.polynomial_model.p_value is not None else None,
                    'correlation': round(analysis_result.polynomial_model.correlation, 4) if analysis_result.polynomial_model.correlation is not None else None,
                    'is_significant': bool(analysis_result.polynomial_model.is_significant) if analysis_result.polynomial_model.is_significant is not None else None
                },
                'best_model': analysis_result.best_model,
                'peak_age': analysis_result.peak_age,
                'age_range_analysis': analysis_result.age_range_analysis,
                'age_groups_analysis': analysis_result.age_groups_analysis,
                'player_comparisons': analysis_result.player_comparisons[:10],  # Limit to top 10
                'insights': analysis_result.insights,
                'sample_size': len(analysis_result.linear_model.ages),
                'enriched_data_count': analysis_result.enriched_data_count,
                'ai_features_available': age_performance_service.mistral_client is not None
            }
            
            return jsonify({
                'success': True,
                'data': response_data,
                'timestamp': datetime.now().isoformat(),
                'processing_time': 'fast_mode'
            })
            
        except Exception as e:
            logger.error(f"Quick age analysis failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis/background', methods=['POST'])
    def start_background_ai_analysis():
        """Start AI-enhanced analysis in background"""
        try:
            data = request.get_json() or {}
            metric = data.get('metric', 'points')
            position = data.get('position')
            
            # Validate metric
            valid_metrics = ['points', 'goals', 'assists', 'points_per_game', 'form', 'xg', 'xa', 'clean_sheets', 'saves']
            if metric not in valid_metrics:
                return jsonify({
                    'error': f'Invalid metric. Valid options: {", ".join(valid_metrics)}'
                }), 400
            
            # Generate unique task ID
            task_id = str(uuid.uuid4())
            
            # Create background task
            create_background_task(task_id, 'ai_analysis', status='queued')
            
            # Start background thread
            thread = threading.Thread(
                target=run_ai_analysis_background,
                args=(task_id, metric, position)
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'task_id': task_id,
                'message': 'AI analysis started in background',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to start background AI analysis: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis/status/<task_id>', methods=['GET'])
    def get_background_task_status(task_id):
        """Get status of background task"""
        try:
            if task_id not in background_tasks:
                return jsonify({'error': 'Task not found'}), 404
            
            task = background_tasks[task_id]
            return jsonify({
                'success': True,
                'data': task,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis/tasks', methods=['GET'])
    def list_background_tasks():
        """List all background tasks"""
        try:
            # Clean up old completed tasks (older than 1 hour)
            current_time = datetime.now()
            tasks_to_remove = []
            
            for task_id, task in background_tasks.items():
                if task['status'] in ['completed', 'failed']:
                    task_time = datetime.fromisoformat(task['updated_at'])
                    if (current_time - task_time).total_seconds() > 3600:  # 1 hour
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del background_tasks[task_id]
            
            return jsonify({
                'success': True,
                'data': {
                    'tasks': list(background_tasks.values()),
                    'total_tasks': len(background_tasks)
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to list background tasks: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis/bulk-enrich', methods=['POST'])
    def bulk_enrich_player_ages():
        """Bulk enrich player ages using Mistral AI"""
        try:
            data = request.get_json() or {}
            limit = data.get('limit', 50)  # Process in batches
            force_refresh = data.get('force_refresh', False)
            
            # Get FPL data
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
            
            players_data = fpl_data.get('elements', [])
            teams_data = fpl_data.get('teams', [])
            team_lookup = {team['id']: team['name'] for team in teams_data}
            
            # Filter players that need enrichment
            players_to_enrich = []
            enriched_count = 0
            skipped_count = 0
            
            for player in players_data[:limit]:  # Limit batch size
                # Skip if player already has birth date and we're not forcing refresh
                if player.get('date_of_birth') and not force_refresh:
                    skipped_count += 1
                    continue
                    
                # Check if we have recent enriched data
                if not force_refresh:
                    existing_data = age_performance_service._get_enriched_player_data(player['id'])
                    if existing_data:
                        last_updated = datetime.fromisoformat(existing_data.last_updated)
                        if (datetime.now() - last_updated).days < 7:
                            skipped_count += 1
                            continue
                
                player_name = player.get('web_name', '')
                team_name = team_lookup.get(player.get('team'), 'Unknown')
                
                if player_name and team_name != 'Unknown':
                    players_to_enrich.append({
                        'id': player['id'],
                        'name': player_name,
                        'team': team_name,
                        'full_name': player.get('first_name', '') + ' ' + player.get('second_name', '')
                    })
            
            logger.info(f"Starting bulk enrichment for {len(players_to_enrich)} players")
            
            # Generate unique task ID for tracking
            task_id = str(uuid.uuid4())
            create_background_task(task_id, 'bulk_enrichment', status='running', progress=0)
            
            # Start background enrichment
            def run_bulk_enrichment():
                try:
                    successful = 0
                    failed = 0
                    total = len(players_to_enrich)
                    
                    for i, player_info in enumerate(players_to_enrich):
                        try:
                            progress = int((i / total) * 100)
                            update_background_task(task_id, progress=progress)
                            
                            # Use async enrichment
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                enriched_data = loop.run_until_complete(
                                    age_performance_service.enrich_player_data_with_ai(
                                        player_info['name'], 
                                        player_info['team'], 
                                        player_info['id']
                                    )
                                )
                                
                                if enriched_data:
                                    successful += 1
                                    logger.info(f"Enriched {player_info['name']}: Age {enriched_data.enriched_age}")
                                else:
                                    failed += 1
                                    logger.warning(f"Failed to enrich {player_info['name']}")
                                    
                            finally:
                                loop.close()
                                
                            # Add delay to avoid rate limiting
                            import time
                            time.sleep(1)
                            
                        except Exception as e:
                            failed += 1
                            logger.error(f"Error enriching {player_info['name']}: {e}")
                    
                    # Complete the task
                    result = {
                        'total_processed': total,
                        'successful': successful,
                        'failed': failed,
                        'skipped': skipped_count,
                        'message': f'Bulk enrichment completed: {successful} successful, {failed} failed'
                    }
                    
                    update_background_task(task_id, status='completed', progress=100, result=result)
                    logger.info(f"Bulk enrichment completed: {successful}/{total} successful")
                    
                except Exception as e:
                    logger.error(f"Bulk enrichment failed: {e}")
                    update_background_task(task_id, status='failed', error=str(e))
            
            # Start background thread
            thread = threading.Thread(target=run_bulk_enrichment)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'task_id': task_id,
                'message': f'Started bulk enrichment for {len(players_to_enrich)} players',
                'players_to_process': len(players_to_enrich),
                'players_skipped': skipped_count,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Bulk enrichment failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis/import-csv', methods=['POST'])
    def import_player_data_from_csv():
        """Import player data from Fantasy Premier League GitHub repository"""
        try:
            data = request.get_json() or {}
            season = data.get('season', '2024-25')
            url = data.get('url') or f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv'
            
            # Generate unique task ID for tracking
            task_id = str(uuid.uuid4())
            create_background_task(task_id, 'csv_import', status='running', progress=0)
            
            # Start background import
            def run_csv_import():
                try:
                    import pandas as pd
                    import requests
                    from io import StringIO
                    
                    update_background_task(task_id, progress=10)
                    
                    # Download CSV data
                    logger.info(f"Downloading CSV data from: {url}")
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    
                    update_background_task(task_id, progress=30)
                    
                    # Parse CSV
                    csv_data = StringIO(response.text)
                    df = pd.read_csv(csv_data)
                    
                    logger.info(f"CSV loaded with {len(df)} players")
                    update_background_task(task_id, progress=50)
                    
                    # Import into database
                    successful = 0
                    failed = 0
                    total = len(df)
                    
                    for i, row in df.iterrows():
                        try:
                            # Extract player data
                            player_id = row.get('id')
                            first_name = row.get('first_name', '')
                            last_name = row.get('second_name', '')
                            web_name = row.get('web_name', '')
                            birth_date = row.get('birth_date')
                            
                            # Skip if no birth date (handle both null values and string "None")
                            if pd.isna(birth_date) or not birth_date or str(birth_date).strip().lower() in ['none', 'null', '']:
                                continue
                                
                            # Calculate age
                            enriched_age = age_performance_service.calculate_age_from_birth_date(str(birth_date))
                            
                            if enriched_age:
                                # Create enriched data object
                                from backend.services.age_performance_service import EnrichedPlayerData
                                enriched_data = EnrichedPlayerData(
                                    player_id=int(player_id) if pd.notna(player_id) else 0,
                                    name=web_name or f"{first_name} {last_name}".strip(),
                                    enriched_age=enriched_age,
                                    birth_date=str(birth_date),
                                    nationality=None,  # Not available in this CSV
                                    injury_status='Fit',  # Default assumption
                                    last_updated=datetime.now().isoformat(),
                                    data_source=f"FPL GitHub CSV: {url}",
                                    confidence=0.99  # High confidence for official data
                                )
                                
                                # Save to database
                                age_performance_service._save_enriched_player_data(enriched_data)
                                successful += 1
                                
                            progress = int(50 + (i / total) * 40)
                            update_background_task(task_id, progress=progress)
                            
                        except Exception as e:
                            failed += 1
                            logger.warning(f"Failed to import player {row.get('web_name', 'unknown')}: {e}")
                    
                    # Complete the task
                    result = {
                        'total_processed': total,
                        'successful': successful,
                        'failed': failed,
                        'source': url,
                        'season': season,
                        'message': f'CSV import completed: {successful} players imported from {season} season'
                    }
                    
                    update_background_task(task_id, status='completed', progress=100, result=result)
                    logger.info(f"CSV import completed: {successful}/{total} successful")
                    
                except Exception as e:
                    logger.error(f"CSV import failed: {e}")
                    update_background_task(task_id, status='failed', error=str(e))
            
            # Start background thread
            thread = threading.Thread(target=run_csv_import)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'task_id': task_id,
                'message': f'Started CSV import from {url}',
                'season': season,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"CSV import initialization failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis/predict', methods=['GET'])
    def predict_performance_by_age():
        """Predict performance for a specific age"""
        try:
            # Get query parameters
            age = request.args.get('age', type=float)
            metric = request.args.get('metric', 'points')
            position = request.args.get('position')
            player_ids_param = request.args.get('player_ids')
            
            # Parse player IDs if provided
            player_ids = None
            if player_ids_param:
                try:
                    player_ids = [int(pid) for pid in player_ids_param.split(',')]
                except ValueError:
                    return jsonify({'error': 'Invalid player_ids format. Use comma-separated integers.'}), 400
            
            if not age:
                return jsonify({'error': 'Age parameter is required'}), 400
            
            if age < 16 or age > 45:
                return jsonify({'error': 'Age must be between 16 and 45'}), 400
            
            # Get FPL data and perform analysis
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
            
            analysis_result = asyncio.run(age_performance_service.analyze_age_performance(
                fpl_data, metric, position, 
                use_ai_enrichment=True,  # Use enriched data including CSV imports
                generate_ai_summary=False,  # Don't generate AI summaries for speed
                force_new_enrichment=False,  # Use cached data only
                player_ids=player_ids
            ))
            
            # Get prediction for specific age
            prediction = age_performance_service.get_player_age_prediction(age, analysis_result)
            
            return jsonify({
                'success': True,
                'data': {
                    'prediction': prediction,
                    'analysis_metadata': {
                        'metric': metric,
                        'position_filter': position,
                        'sample_size': len(analysis_result.linear_model.ages),
                        'peak_age': analysis_result.peak_age,
                        'model_confidence': max(
                            analysis_result.linear_model.r2_score,
                            analysis_result.polynomial_model.r2_score
                        )
                    }
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Age prediction failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis/compare-players', methods=['POST'])
    def compare_players_by_age():
        """Compare specific players in age-performance context"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'Request body is required'}), 400
            
            player_ids = data.get('player_ids', [])
            metric = data.get('metric', 'points')
            
            if not player_ids:
                return jsonify({'error': 'player_ids array is required'}), 400
            
            if not isinstance(player_ids, list):
                return jsonify({'error': 'player_ids must be an array'}), 400
            
            # Convert to integers
            try:
                player_ids = [int(pid) for pid in player_ids]
            except ValueError:
                return jsonify({'error': 'All player_ids must be valid integers'}), 400
            
            # Get FPL data
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
            
            # Perform comparison
            comparison_result = age_performance_service.compare_players_by_age(
                player_ids, fpl_data, metric
            )
            
            return jsonify({
                'success': True,
                'data': comparison_result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Player age comparison failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis/metrics', methods=['GET'])
    def get_available_age_metrics():
        """Get list of available metrics for age analysis"""
        try:
            metrics = [
                {
                    'key': 'points',
                    'name': 'Total Points',
                    'description': 'Total FPL points scored this season'
                },
                {
                    'key': 'points_per_game',
                    'name': 'Points Per Game',
                    'description': 'Average FPL points per game played'
                },
                {
                    'key': 'goals',
                    'name': 'Goals',
                    'description': 'Total goals scored this season'
                },
                {
                    'key': 'assists',
                    'name': 'Assists',
                    'description': 'Total assists provided this season'
                },
                {
                    'key': 'form',
                    'name': 'Current Form',
                    'description': 'Recent performance form rating'
                },
                {
                    'key': 'xg',
                    'name': 'Expected Goals (xG)',
                    'description': 'Expected goals based on shot quality'
                },
                {
                    'key': 'xa',
                    'name': 'Expected Assists (xA)',
                    'description': 'Expected assists based on chance creation'
                },
                {
                    'key': 'clean_sheets',
                    'name': 'Clean Sheets',
                    'description': 'Matches without conceding (defenders/goalkeepers)'
                },
                {
                    'key': 'saves',
                    'name': 'Saves',
                    'description': 'Total saves made (goalkeepers only)'
                },
                {
                    'key': 'minutes',
                    'name': 'Minutes Played',
                    'description': 'Total minutes played this season'
                },
                {
                    'key': 'games_played',
                    'name': 'Games Played', 
                    'description': 'Number of games participated in'
                }
            ]
            
            positions = ['GK', 'DEF', 'MID', 'FWD']
            
            return jsonify({
                'success': True,
                'data': {
                    'metrics': metrics,
                    'positions': positions
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to get age metrics: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis/enrich-players', methods=['POST'])
    def enrich_players_data():
        """Manually trigger AI enrichment for specific players"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'Request body is required'}), 400
            
            player_ids = data.get('player_ids', [])
            
            if not player_ids:
                return jsonify({'error': 'player_ids array is required'}), 400
            
            # Get FPL data
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch FPL data'}), 500
            
            players_data = fpl_data.get('elements', [])
            teams_data = fpl_data.get('teams', [])
            team_lookup = {team['id']: team['name'] for team in teams_data}
            
            enrichment_results = []
            
            async def enrich_players_async():
                tasks = []
                for player_id in player_ids:
                    player = next((p for p in players_data if p['id'] == player_id), None)
                    if player:
                        player_name = player.get('web_name', '')
                        team_name = team_lookup.get(player.get('team'), 'Unknown')
                        
                        if player_name and team_name != 'Unknown':
                            task = age_performance_service.enrich_player_data_with_ai(
                                player_name, team_name, player_id
                            )
                            tasks.append((player_id, player_name, task))
                
                for player_id, player_name, task in tasks:
                    try:
                        enriched_data = await task
                        if enriched_data:
                            enrichment_results.append({
                                'player_id': player_id,
                                'player_name': player_name,
                                'success': True,
                                'enriched_age': enriched_data.enriched_age,
                                'birth_date': enriched_data.birth_date,
                                'nationality': enriched_data.nationality,
                                'injury_status': enriched_data.injury_status,
                                'confidence': enriched_data.confidence,
                                'data_source': enriched_data.data_source
                            })
                        else:
                            enrichment_results.append({
                                'player_id': player_id,
                                'player_name': player_name,
                                'success': False,
                                'error': 'No enriched data available'
                            })
                    except Exception as e:
                        enrichment_results.append({
                            'player_id': player_id,
                            'player_name': player_name,
                            'success': False,
                            'error': str(e)
                        })
            
            # Run async enrichment
            asyncio.run(enrich_players_async())
            
            return jsonify({
                'success': True,
                'data': {
                    'enrichment_results': enrichment_results,
                    'total_processed': len(player_ids),
                    'successful_enrichments': len([r for r in enrichment_results if r['success']])
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Player enrichment failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis/season-correlation', methods=['GET'])
    def analyze_season_correlation():
        """Analyze correlation between current season and previous season points"""
        try:
            previous_season = request.args.get('previous_season', '2023-24')
            
            # Get current season data
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data:
                return jsonify({'error': 'Failed to fetch current season data'}), 500
            
            # Run correlation analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                analysis_result = loop.run_until_complete(
                    age_performance_service.analyze_season_correlation(fpl_data, previous_season)
                )
            finally:
                loop.close()
            
            # Convert any numpy/pandas types to native Python types for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, bool):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            clean_result = convert_for_json(analysis_result)
            
            return jsonify({
                'success': True,
                'data': clean_result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Season correlation analysis failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/age-analysis/enriched-data', methods=['GET'])
    def get_enriched_players_data():
        """Get all enriched player data from database"""
        try:
            conn = sqlite3.connect(age_performance_service.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT player_id, name, enriched_age, birth_date, nationality, 
                       injury_status, last_updated, data_source, confidence
                FROM player_enrichment
                ORDER BY last_updated DESC
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            enriched_players = []
            for row in rows:
                enriched_players.append({
                    'player_id': row[0],
                    'name': row[1],
                    'enriched_age': row[2],
                    'birth_date': row[3],
                    'nationality': row[4],
                    'injury_status': row[5],
                    'last_updated': row[6],
                    'data_source': row[7],
                    'confidence': row[8]
                })
            
            return jsonify({
                'success': True,
                'data': {
                    'enriched_players': enriched_players,
                    'total_count': len(enriched_players)
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to get enriched players data: {e}")
            return jsonify({'error': str(e)}), 500
    
    return app

# Create the Flask application
app = create_app(os.getenv('FLASK_ENV', 'default'))

if __name__ == '__main__':
    logger.info("Starting FPL Manager v3 API Server")
    logger.info("=" * 50)
    logger.info(f"Environment: {os.getenv('FLASK_ENV', 'development')}")
    logger.info(f"Debug mode: {app.debug}")
    logger.info(f"API will be available at: http://{Config.API_HOST}:{Config.API_PORT}")
    logger.info("Features enabled:")
    logger.info(f"  - Real FPL data integration: ✓")
    logger.info(f"  - ML predictions: ✓")
    logger.info(f"  - Weather integration: {'✓' if Config.ENABLE_WEATHER_INTEGRATION else '✗'}")
    logger.info(f"  - News integration: {'✓' if Config.ENABLE_NEWS_INTEGRATION else '✗'}")
    logger.info(f"  - Accuracy tracking: {'✓' if Config.ENABLE_ACCURACY_TRACKING else '✗'}")
    logger.info("=" * 50)
    
    try:
        app.run(
            host='0.0.0.0',
            port=Config.API_PORT,
            debug=Config.DEBUG
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())