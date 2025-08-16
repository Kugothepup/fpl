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
    
    @app.route('/api/predictions/gameweek-scores', methods=['GET'])
    def predict_gameweek_scores():
        """Predict match scores for upcoming gameweek"""
        try:
            # Get fixtures for next gameweek
            fixtures = fpl_manager.fetch_fixtures()
            upcoming_fixtures = [f for f in fixtures if not f.finished and f.gameweek == (fpl_manager.current_gameweek or 1)]
            
            # Simple score prediction based on team strength and form
            score_predictions = []
            fpl_data = fpl_manager.fetch_bootstrap_data()
            teams_data = fpl_data.get('teams', []) if fpl_data else []
            
            for fixture in upcoming_fixtures[:10]:  # Limit to 10 fixtures
                # Find team data
                home_team_data = next((t for t in teams_data if t['id'] == fixture.home_team_id), None)
                away_team_data = next((t for t in teams_data if t['id'] == fixture.away_team_id), None)
                
                if home_team_data and away_team_data:
                    # Simple prediction based on team strength
                    home_strength = home_team_data.get('strength_overall_home', 1200)
                    away_strength = away_team_data.get('strength_overall_away', 1200)
                    
                    # Basic score prediction (simplified)
                    strength_diff = (home_strength - away_strength) / 100
                    
                    # Predict scores (very basic model)
                    home_score = max(0, round(1.5 + strength_diff * 0.3))
                    away_score = max(0, round(1.2 - strength_diff * 0.3))
                    
                    # Calculate confidence based on strength difference
                    confidence = min(0.9, 0.5 + abs(strength_diff) * 0.05)
                    
                    score_predictions.append({
                        'fixture_id': fixture.id,
                        'home_team': fixture.home_team,
                        'away_team': fixture.away_team,
                        'kickoff_time': fixture.kickoff_time,
                        'predicted_score': f"{home_score}-{away_score}",
                        'home_score': home_score,
                        'away_score': away_score,
                        'confidence': round(confidence, 2),
                        'reasoning': f"Based on team strength: {fixture.home_team} ({home_strength}) vs {fixture.away_team} ({away_strength})"
                    })
            
            return jsonify({
                'success': True,
                'data': score_predictions,
                'gameweek': fpl_manager.current_gameweek or 1,
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
            
            # Get user team and FPL data
            user_team = fpl_manager.fetch_user_team(team_id)
            fpl_data = fpl_manager.fetch_bootstrap_data()
            
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
                
                return jsonify({
                    'success': True,
                    'data': team_score_prediction,
                    'team_info': {
                        'id': team_id,
                        'name': user_team.get('name', 'Unknown Team'),
                        'player_first_name': user_team.get('player_first_name', ''),
                        'player_last_name': user_team.get('player_last_name', ''),
                        'overall_rank': user_team.get('summary_overall_rank'),
                        'gameweek_rank': user_team.get('summary_event_rank'),
                        'total_points': user_team.get('summary_overall_points'),
                        'gameweek_points': user_team.get('summary_event_points')
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            return jsonify({'error': 'No team data found'}), 404
            
        except Exception as e:
            logger.error(f"Team score prediction failed: {e}")
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