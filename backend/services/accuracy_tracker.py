#!/usr/bin/env python3
"""
FPL Manager v3 - Accuracy Tracking System
Measures and improves ML prediction accuracy over time
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import Config

logger = logging.getLogger(__name__)

@dataclass
class PredictionRecord:
    """Record of a prediction made by the system"""
    prediction_id: str
    timestamp: str
    prediction_type: str  # 'points', 'captain', 'transfer', 'match'
    player_id: Optional[int]
    player_name: str
    predicted_value: float
    confidence: float
    gameweek: int
    model_used: str
    features_used: List[str]
    context_data: Dict  # Weather, news, etc.

@dataclass
class ActualResult:
    """Actual outcome to compare against prediction"""
    prediction_id: str
    actual_value: float
    gameweek: int
    timestamp_actual: str
    data_source: str  # 'fpl_api', 'manual', etc.

@dataclass
class AccuracyMetrics:
    """Accuracy metrics for model evaluation"""
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    accuracy_percentage: float
    r2_score: float
    prediction_count: int
    confidence_correlation: float  # How well confidence predicts accuracy

@dataclass
class PerformanceInsight:
    """Insight about model performance"""
    insight_type: str
    description: str
    impact_level: str  # 'low', 'medium', 'high'
    recommendation: str
    confidence: float

class AccuracyTracker:
    """System for tracking and analyzing prediction accuracy"""
    
    def __init__(self):
        self.db_path = Config.PREDICTIONS_DATABASE_PATH
        self.init_database()
        
        # Performance thresholds
        self.accuracy_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'fair': 0.7,
            'poor': 0.6
        }
        
        # Model performance history
        self.performance_history = defaultdict(list)
        
        logger.info(f"Accuracy Tracker initialized with database: {self.db_path}")
    
    def init_database(self):
        """Initialize the accuracy tracking database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Predictions table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    player_id INTEGER,
                    player_name TEXT NOT NULL,
                    predicted_value REAL NOT NULL,
                    confidence REAL NOT NULL,
                    gameweek INTEGER NOT NULL,
                    model_used TEXT NOT NULL,
                    features_used TEXT,  -- JSON
                    context_data TEXT   -- JSON
                )
                ''')
                
                # Actual results table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS actual_results (
                    prediction_id TEXT PRIMARY KEY,
                    actual_value REAL NOT NULL,
                    gameweek INTEGER NOT NULL,
                    timestamp_actual TEXT NOT NULL,
                    data_source TEXT NOT NULL,
                    FOREIGN KEY (prediction_id) REFERENCES predictions (prediction_id)
                )
                ''')
                
                # Accuracy metrics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS accuracy_metrics (
                    metric_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    gameweek_start INTEGER,
                    gameweek_end INTEGER,
                    mae REAL,
                    mse REAL,
                    rmse REAL,
                    mape REAL,
                    accuracy_percentage REAL,
                    r2_score REAL,
                    prediction_count INTEGER,
                    confidence_correlation REAL
                )
                ''')
                
                # Performance insights table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_insights (
                    insight_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    impact_level TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    confidence REAL NOT NULL
                )
                ''')
                
                conn.commit()
                logger.info("Accuracy tracking database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize accuracy database: {e}")
            raise
    
    def store_prediction(self, prediction: PredictionRecord) -> bool:
        """Store a prediction for future accuracy evaluation"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT OR REPLACE INTO predictions 
                (prediction_id, timestamp, prediction_type, player_id, player_name,
                 predicted_value, confidence, gameweek, model_used, features_used, context_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction.prediction_id,
                    prediction.timestamp,
                    prediction.prediction_type,
                    prediction.player_id,
                    prediction.player_name,
                    prediction.predicted_value,
                    prediction.confidence,
                    prediction.gameweek,
                    prediction.model_used,
                    json.dumps(prediction.features_used),
                    json.dumps(prediction.context_data)
                ))
                
                conn.commit()
                logger.debug(f"Stored prediction {prediction.prediction_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store prediction {prediction.prediction_id}: {e}")
            return False
    
    def store_actual_result(self, result: ActualResult) -> bool:
        """Store actual result for a prediction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT OR REPLACE INTO actual_results
                (prediction_id, actual_value, gameweek, timestamp_actual, data_source)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    result.prediction_id,
                    result.actual_value,
                    result.gameweek,
                    result.timestamp_actual,
                    result.data_source
                ))
                
                conn.commit()
                logger.debug(f"Stored actual result for prediction {result.prediction_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store actual result for {result.prediction_id}: {e}")
            return False
    
    def calculate_accuracy_metrics(self, prediction_type: str = None, 
                                 gameweek_start: int = None, gameweek_end: int = None,
                                 model_name: str = None) -> Optional[AccuracyMetrics]:
        """Calculate accuracy metrics for predictions with actual results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Build query with filters
                query = '''
                SELECT p.predicted_value, p.confidence, a.actual_value
                FROM predictions p
                JOIN actual_results a ON p.prediction_id = a.prediction_id
                WHERE 1=1
                '''
                params = []
                
                if prediction_type:
                    query += ' AND p.prediction_type = ?'
                    params.append(prediction_type)
                
                if gameweek_start:
                    query += ' AND p.gameweek >= ?'
                    params.append(gameweek_start)
                
                if gameweek_end:
                    query += ' AND p.gameweek <= ?'
                    params.append(gameweek_end)
                
                if model_name:
                    query += ' AND p.model_used = ?'
                    params.append(model_name)
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    logger.warning("No predictions with actual results found for accuracy calculation")
                    return None
                
                # Calculate metrics
                predicted = df['predicted_value'].values
                actual = df['actual_value'].values
                confidence = df['confidence'].values
                
                # Basic accuracy metrics
                mae = np.mean(np.abs(predicted - actual))
                mse = np.mean((predicted - actual) ** 2)
                rmse = np.sqrt(mse)
                
                # Avoid division by zero in MAPE
                actual_nonzero = actual + 1e-8  # Add small epsilon
                mape = np.mean(np.abs((actual - predicted) / actual_nonzero)) * 100
                
                # Accuracy percentage (within reasonable threshold)
                threshold = np.std(actual) * 0.5  # 50% of standard deviation
                accurate_predictions = np.abs(predicted - actual) <= threshold
                accuracy_percentage = np.mean(accurate_predictions) * 100
                
                # R-squared score
                ss_res = np.sum((actual - predicted) ** 2)
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                r2_score = 1 - (ss_res / (ss_tot + 1e-8))  # Avoid division by zero
                
                # Confidence correlation
                absolute_errors = np.abs(predicted - actual)
                confidence_correlation = np.corrcoef(confidence, 1 / (absolute_errors + 1e-8))[0, 1]
                if np.isnan(confidence_correlation):
                    confidence_correlation = 0.0
                
                metrics = AccuracyMetrics(
                    mae=float(mae),
                    mse=float(mse),
                    rmse=float(rmse),
                    mape=float(mape),
                    accuracy_percentage=float(accuracy_percentage),
                    r2_score=float(r2_score),
                    prediction_count=len(df),
                    confidence_correlation=float(confidence_correlation)
                )
                
                logger.info(f"Calculated accuracy metrics: MAE={mae:.3f}, Accuracy={accuracy_percentage:.1f}%")
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to calculate accuracy metrics: {e}")
            return None
    
    def update_actual_results_from_fpl(self, fpl_manager) -> int:
        """Update actual results by fetching latest FPL data"""
        try:
            updated_count = 0
            
            # Get predictions without actual results
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT p.prediction_id, p.player_id, p.gameweek, p.prediction_type
                FROM predictions p
                LEFT JOIN actual_results a ON p.prediction_id = a.prediction_id
                WHERE a.prediction_id IS NULL
                AND p.gameweek <= (SELECT MAX(gameweek) FROM predictions) - 1
                ORDER BY p.gameweek DESC
                ''')
                
                pending_predictions = cursor.fetchall()
            
            if not pending_predictions:
                logger.info("No pending predictions to update")
                return 0
            
            # Fetch current FPL data
            fpl_data = fpl_manager.fetch_bootstrap_data()
            if not fpl_data or 'elements' not in fpl_data:
                logger.warning("Could not fetch FPL data for accuracy updates")
                return 0
            
            # Create player lookup
            players_map = {player['id']: player for player in fpl_data['elements']}
            
            # Update results
            for pred_id, player_id, gameweek, pred_type in pending_predictions:
                try:
                    if player_id not in players_map:
                        continue
                    
                    player_data = players_map[player_id]
                    
                    # Get actual value based on prediction type
                    actual_value = None
                    if pred_type == 'points':
                        # Use points per game as approximation
                        actual_value = float(player_data.get('points_per_game', 0))
                    elif pred_type == 'captain':
                        # This would require more specific data about captaincy performance
                        actual_value = float(player_data.get('total_points', 0)) * 2  # Captain doubles points
                    
                    if actual_value is not None:
                        result = ActualResult(
                            prediction_id=pred_id,
                            actual_value=actual_value,
                            gameweek=gameweek,
                            timestamp_actual=datetime.now().isoformat(),
                            data_source='fpl_api'
                        )
                        
                        if self.store_actual_result(result):
                            updated_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to update prediction {pred_id}: {e}")
                    continue
            
            logger.info(f"Updated {updated_count} actual results from FPL data")
            return updated_count
            
        except Exception as e:
            logger.error(f"Failed to update actual results: {e}")
            return 0
    
    def generate_performance_insights(self, lookback_gameweeks: int = 5) -> List[PerformanceInsight]:
        """Generate insights about model performance"""
        try:
            insights = []
            
            # Calculate recent metrics
            current_gw = self._get_current_gameweek()
            if not current_gw:
                return insights
            
            start_gw = max(1, current_gw - lookback_gameweeks)
            
            # Overall performance insight
            overall_metrics = self.calculate_accuracy_metrics(
                gameweek_start=start_gw,
                gameweek_end=current_gw
            )
            
            if overall_metrics:
                performance_level = self._classify_performance(overall_metrics.accuracy_percentage)
                
                insights.append(PerformanceInsight(
                    insight_type='overall_performance',
                    description=f"Overall prediction accuracy is {overall_metrics.accuracy_percentage:.1f}% ({performance_level})",
                    impact_level='high' if performance_level in ['poor', 'excellent'] else 'medium',
                    recommendation=self._get_performance_recommendation(performance_level),
                    confidence=0.9
                ))
            
            # Model comparison insights
            model_insights = self._analyze_model_performance(start_gw, current_gw)
            insights.extend(model_insights)
            
            # Confidence calibration insights
            confidence_insights = self._analyze_confidence_calibration(start_gw, current_gw)
            insights.extend(confidence_insights)
            
            # Feature importance insights
            feature_insights = self._analyze_feature_performance(start_gw, current_gw)
            insights.extend(feature_insights)
            
            # Store insights
            for insight in insights:
                self._store_insight(insight)
            
            logger.info(f"Generated {len(insights)} performance insights")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate performance insights: {e}")
            return []
    
    def get_accuracy_trends(self, gameweeks: int = 10) -> Dict:
        """Get accuracy trends over recent gameweeks"""
        try:
            current_gw = self._get_current_gameweek()
            if not current_gw:
                return {}
            
            trends = {
                'gameweeks': [],
                'accuracy_percentages': [],
                'mae_values': [],
                'prediction_counts': []
            }
            
            for gw in range(max(1, current_gw - gameweeks), current_gw + 1):
                metrics = self.calculate_accuracy_metrics(
                    gameweek_start=gw,
                    gameweek_end=gw
                )
                
                if metrics:
                    trends['gameweeks'].append(gw)
                    trends['accuracy_percentages'].append(metrics.accuracy_percentage)
                    trends['mae_values'].append(metrics.mae)
                    trends['prediction_counts'].append(metrics.prediction_count)
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to get accuracy trends: {e}")
            return {}
    
    def get_top_performing_models(self, limit: int = 5) -> List[Dict]:
        """Get top performing models by accuracy"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                SELECT p.model_used, 
                       AVG(ABS(p.predicted_value - a.actual_value)) as avg_error,
                       COUNT(*) as prediction_count,
                       AVG(p.confidence) as avg_confidence
                FROM predictions p
                JOIN actual_results a ON p.prediction_id = a.prediction_id
                GROUP BY p.model_used
                HAVING prediction_count >= 5
                ORDER BY avg_error ASC
                LIMIT ?
                '''
                
                cursor = conn.cursor()
                cursor.execute(query, (limit,))
                results = cursor.fetchall()
                
                models = []
                for row in results:
                    models.append({
                        'model_name': row[0],
                        'average_error': round(row[1], 3),
                        'prediction_count': row[2],
                        'average_confidence': round(row[3], 3),
                        'performance_score': round(1 / (1 + row[1]), 3)  # Inverse error score
                    })
                
                return models
                
        except Exception as e:
            logger.error(f"Failed to get top performing models: {e}")
            return []
    
    def _get_current_gameweek(self) -> Optional[int]:
        """Get current gameweek from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT MAX(gameweek) FROM predictions')
                result = cursor.fetchone()
                return result[0] if result and result[0] else None
        except:
            return None
    
    def _classify_performance(self, accuracy_percentage: float) -> str:
        """Classify performance level"""
        if accuracy_percentage >= self.accuracy_thresholds['excellent'] * 100:
            return 'excellent'
        elif accuracy_percentage >= self.accuracy_thresholds['good'] * 100:
            return 'good'
        elif accuracy_percentage >= self.accuracy_thresholds['fair'] * 100:
            return 'fair'
        else:
            return 'poor'
    
    def _get_performance_recommendation(self, performance_level: str) -> str:
        """Get recommendation based on performance level"""
        recommendations = {
            'excellent': 'Maintain current model configuration and continue monitoring',
            'good': 'Consider fine-tuning hyperparameters for further improvement',
            'fair': 'Review feature engineering and consider ensemble methods',
            'poor': 'Investigate model architecture and retrain with more data'
        }
        return recommendations.get(performance_level, 'Monitor performance closely')
    
    def _analyze_model_performance(self, start_gw: int, end_gw: int) -> List[PerformanceInsight]:
        """Analyze individual model performance"""
        insights = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                SELECT p.model_used, 
                       AVG(ABS(p.predicted_value - a.actual_value)) as avg_error,
                       COUNT(*) as count
                FROM predictions p
                JOIN actual_results a ON p.prediction_id = a.prediction_id
                WHERE p.gameweek BETWEEN ? AND ?
                GROUP BY p.model_used
                HAVING count >= 3
                ORDER BY avg_error ASC
                '''
                
                cursor = conn.cursor()
                cursor.execute(query, (start_gw, end_gw))
                results = cursor.fetchall()
                
                if len(results) > 1:
                    best_model = results[0]
                    worst_model = results[-1]
                    
                    insights.append(PerformanceInsight(
                        insight_type='model_comparison',
                        description=f"Best performing model: {best_model[0]} (avg error: {best_model[1]:.3f})",
                        impact_level='medium',
                        recommendation=f"Consider using {best_model[0]} for future predictions",
                        confidence=0.8
                    ))
                    
                    if best_model[1] * 1.5 < worst_model[1]:  # Significant difference
                        insights.append(PerformanceInsight(
                            insight_type='model_underperformance',
                            description=f"Model {worst_model[0]} showing poor performance (avg error: {worst_model[1]:.3f})",
                            impact_level='high',
                            recommendation=f"Review or retrain {worst_model[0]} model",
                            confidence=0.9
                        ))
        
        except Exception as e:
            logger.error(f"Model performance analysis failed: {e}")
        
        return insights
    
    def _analyze_confidence_calibration(self, start_gw: int, end_gw: int) -> List[PerformanceInsight]:
        """Analyze how well confidence scores predict accuracy"""
        insights = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                SELECT p.confidence, ABS(p.predicted_value - a.actual_value) as error
                FROM predictions p
                JOIN actual_results a ON p.prediction_id = a.prediction_id
                WHERE p.gameweek BETWEEN ? AND ?
                '''
                
                df = pd.read_sql_query(query, conn, params=(start_gw, end_gw))
                
                if not df.empty:
                    correlation = df['confidence'].corr(-df['error'])  # Higher confidence should mean lower error
                    
                    if correlation < 0.3:
                        insights.append(PerformanceInsight(
                            insight_type='confidence_calibration',
                            description=f"Confidence scores poorly calibrated (correlation: {correlation:.3f})",
                            impact_level='medium',
                            recommendation="Review confidence calculation methodology",
                            confidence=0.7
                        ))
                    elif correlation > 0.7:
                        insights.append(PerformanceInsight(
                            insight_type='confidence_calibration',
                            description=f"Confidence scores well calibrated (correlation: {correlation:.3f})",
                            impact_level='low',
                            recommendation="Maintain current confidence calculation approach",
                            confidence=0.8
                        ))
        
        except Exception as e:
            logger.error(f"Confidence calibration analysis failed: {e}")
        
        return insights
    
    def _analyze_feature_performance(self, start_gw: int, end_gw: int) -> List[PerformanceInsight]:
        """Analyze which features contribute to better predictions"""
        insights = []
        
        # This would require more detailed feature tracking
        # For now, return a general insight about feature importance
        insights.append(PerformanceInsight(
            insight_type='feature_analysis',
            description="Feature importance analysis requires enhanced feature tracking",
            impact_level='low',
            recommendation="Implement detailed feature importance logging for better insights",
            confidence=0.6
        ))
        
        return insights
    
    def _store_insight(self, insight: PerformanceInsight):
        """Store performance insight in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                insight_id = f"{insight.insight_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                cursor.execute('''
                INSERT INTO performance_insights
                (insight_id, timestamp, insight_type, description, impact_level, recommendation, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    insight_id,
                    datetime.now().isoformat(),
                    insight.insight_type,
                    insight.description,
                    insight.impact_level,
                    insight.recommendation,
                    insight.confidence
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store insight: {e}")
    
    def get_summary_stats(self, days: int = 30) -> Dict:
        """Get summary statistics for the dashboard"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total predictions
                cursor.execute('SELECT COUNT(*) FROM predictions WHERE timestamp > ?', (cutoff_date,))
                total_predictions = cursor.fetchone()[0]
                
                # Predictions with results
                cursor.execute('''
                SELECT COUNT(*) FROM predictions p
                JOIN actual_results a ON p.prediction_id = a.prediction_id
                WHERE p.timestamp > ?
                ''', (cutoff_date,))
                evaluated_predictions = cursor.fetchone()[0]
                
                # Overall accuracy
                overall_metrics = self.calculate_accuracy_metrics()
                
                stats = {
                    'total_predictions': total_predictions,
                    'evaluated_predictions': evaluated_predictions,
                    'evaluation_rate': (evaluated_predictions / total_predictions * 100) if total_predictions > 0 else 0,
                    'overall_accuracy': overall_metrics.accuracy_percentage if overall_metrics else 0,
                    'overall_mae': overall_metrics.mae if overall_metrics else 0,
                    'last_updated': datetime.now().isoformat()
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get summary stats: {e}")
            return {
                'total_predictions': 0,
                'evaluated_predictions': 0,
                'evaluation_rate': 0,
                'overall_accuracy': 0,
                'overall_mae': 0,
                'last_updated': datetime.now().isoformat()
            }