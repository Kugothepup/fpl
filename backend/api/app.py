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
    CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
    
    # Initialize services
    try:
        fpl_manager = FPLManager()
        ml_predictor = FPLMLPredictor()
        weather_service = WeatherService()
        news_service = NewsService()
        accuracy_tracker = AccuracyTracker()
        
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
        """Get user's current FPL team"""
        try:
            team_id = request.args.get('team_id', Config.FPL_TEAM_ID)
            
            team_data = fpl_manager.fetch_user_team(team_id)
            
            if not team_data:
                return jsonify({'error': 'Failed to fetch team data'}), 500
            
            return jsonify({
                'success': True,
                'data': team_data,
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
                        'team': player.team,
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
            host=Config.API_HOST,
            port=Config.API_PORT,
            debug=Config.DEBUG
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())