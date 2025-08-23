#!/usr/bin/env python3
"""
Simple FPL Backend Server - Minimal version to get real data working
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
from pathlib import Path
import logging

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'backend'))

# Import core components
from backend.core.fpl_manager import FPLManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Initialize FPL Manager
try:
    logger.info("Initializing FPL Manager...")
    fpl_manager = FPLManager()
    logger.info("FPL Manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize FPL Manager: {e}")
    fpl_manager = None

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'FPL Manager API is running',
        'fpl_manager_status': 'ok' if fpl_manager else 'error'
    })

@app.route('/api/bootstrap')
def get_bootstrap():
    """Get FPL bootstrap data"""
    try:
        if not fpl_manager:
            return jsonify({'error': 'FPL Manager not initialized'}), 500
            
        logger.info("Fetching FPL bootstrap data...")
        data = fpl_manager.fetch_bootstrap_data()
        
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f"Bootstrap data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/team')
def get_team():
    """Get user team data"""
    try:
        if not fpl_manager:
            return jsonify({'error': 'FPL Manager not initialized'}), 500
            
        team_id = request.args.get('team_id')
        logger.info(f"Fetching team data for team_id: {team_id}")
        
        data = fpl_manager.fetch_user_team(team_id)
        
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f"Team data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/players')
def get_players():
    """Get players data"""
    try:
        if not fpl_manager:
            return jsonify({'error': 'FPL Manager not initialized'}), 500
            
        logger.info("Fetching players data...")
        # For now, get players from bootstrap data
        bootstrap_data = fpl_manager.fetch_bootstrap_data()
        data = bootstrap_data.get('elements', []) if bootstrap_data else []
        
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f"Players data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/points')
def get_predictions():
    """Get player predictions - simplified"""
    try:
        return jsonify({
            'success': True,
            'data': []  # Empty for now since ML predictor is complex
        })
    except Exception as e:
        logger.error(f"Predictions error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Simple FPL Manager API Server")
    logger.info("=" * 50)
    logger.info("Server will be available at: http://127.0.0.1:5001")
    logger.info("=" * 50)
    
    try:
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")