/**
 * FPL Manager v3 - Accuracy Widget
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Alert,
  CircularProgress,
  LinearProgress,
  Grid,
  Chip,
} from '@mui/material';
import { Analytics, TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';
import { fetchAccuracyStats } from '../../services/api';

const AccuracyWidget = () => {
  const [accuracy, setAccuracy] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadAccuracyStats();
  }, []);

  const loadAccuracyStats = async () => {
    try {
      setLoading(true);
      const response = await fetchAccuracyStats(30); // Last 30 days
      setAccuracy(response.data || {});
      setError(null);
    } catch (err) {
      setError('Accuracy data unavailable');
      // Set some mock data for display
      setAccuracy({
        total_predictions: 156,
        accurate_predictions: 98,
        accuracy_percentage: 62.8,
        avg_error: 1.2,
        recent_trend: 'improving',
        model_performance: {
          'ensemble_points': 0.68,
          'random_forest_points': 0.65,
          'gradient_boost_points': 0.64
        }
      });
    } finally {
      setLoading(false);
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'improving': return <TrendingUp color="success" />;
      case 'declining': return <TrendingDown color="error" />;
      default: return <TrendingFlat color="warning" />;
    }
  };

  const getAccuracyColor = (percentage) => {
    if (percentage >= 70) return 'success';
    if (percentage >= 60) return 'warning';
    return 'error';
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Analytics sx={{ mr: 1 }} />
          <Typography variant="h6" component="div">
            Prediction Accuracy
          </Typography>
        </Box>

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
            <CircularProgress size={24} />
          </Box>
        ) : accuracy ? (
          <>
            {/* Overall Accuracy */}
            <Box sx={{ textAlign: 'center', mb: 2 }}>
              <Typography variant="h4" color="primary" fontWeight="bold">
                {accuracy.accuracy_percentage?.toFixed(1) || 'N/A'}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Overall Accuracy
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={accuracy.accuracy_percentage || 0} 
                color={getAccuracyColor(accuracy.accuracy_percentage)}
                sx={{ mt: 1, height: 6 }}
              />
            </Box>

            {/* Stats Grid */}
            <Grid container spacing={1} sx={{ mb: 2 }}>
              <Grid item xs={6}>
                <Typography variant="caption" color="textSecondary">
                  Predictions
                </Typography>
                <Typography variant="body2" fontWeight="bold">
                  {accuracy.total_predictions || 0}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="textSecondary">
                  Accurate
                </Typography>
                <Typography variant="body2" fontWeight="bold">
                  {accuracy.accurate_predictions || 0}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="textSecondary">
                  Avg Error
                </Typography>
                <Typography variant="body2" fontWeight="bold">
                  {accuracy.avg_error?.toFixed(1) || 'N/A'} pts
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="textSecondary">
                  Trend
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  {getTrendIcon(accuracy.recent_trend)}
                  <Typography variant="body2" fontWeight="bold">
                    {accuracy.recent_trend || 'stable'}
                  </Typography>
                </Box>
              </Grid>
            </Grid>

            {/* Model Performance */}
            {accuracy.model_performance && (
              <Box>
                <Typography variant="caption" color="textSecondary" sx={{ mb: 1, display: 'block' }}>
                  Model Performance:
                </Typography>
                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                  {Object.entries(accuracy.model_performance).map(([model, score]) => (
                    <Chip
                      key={model}
                      label={`${model.replace('_points', '')}: ${(score * 100).toFixed(0)}%`}
                      size="small"
                      variant="outlined"
                      color={score > 0.7 ? 'success' : score > 0.6 ? 'warning' : 'default'}
                      sx={{ fontSize: '0.65rem', height: 20 }}
                    />
                  ))}
                </Box>
              </Box>
            )}

            {error && (
              <Alert severity="info" sx={{ mt: 1, fontSize: '0.75rem' }}>
                Showing sample accuracy data - predictions need time to accumulate
              </Alert>
            )}
          </>
        ) : (
          <Alert severity="info">
            Make some predictions first to see accuracy tracking
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default AccuracyWidget;