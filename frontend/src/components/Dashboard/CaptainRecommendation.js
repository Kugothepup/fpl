/**
 * FPL Manager v3 - Captain Recommendation Widget
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  Avatar,
  Chip,
  LinearProgress,
  Alert,
  Skeleton,
  Tooltip,
} from '@mui/material';
import {
  Star as CaptainIcon,
  TrendingUp,
  Refresh as RefreshIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

import { apiService } from '../../services/api';

const CaptainRecommendation = () => {
  const [loading, setLoading] = useState(true);
  const [recommendation, setRecommendation] = useState(null);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadRecommendation();
  }, []);

  const loadRecommendation = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await apiService.getCaptainRecommendation();
      
      if (response.success) {
        setRecommendation(response.data);
      } else {
        throw new Error(response.error || 'Failed to get captain recommendation');
      }

      setLoading(false);
    } catch (err) {
      console.error('Captain recommendation error:', err);
      setError(apiService.handleError(err, 'Failed to load captain recommendation'));
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadRecommendation();
    setRefreshing(false);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 0.8) return 'High Confidence';
    if (confidence >= 0.6) return 'Medium Confidence';
    return 'Low Confidence';
  };

  const formatPosition = (position) => {
    const positionMap = {
      'GK': 'Goalkeeper',
      'DEF': 'Defender',
      'MID': 'Midfielder',
      'FWD': 'Forward'
    };
    return positionMap[position] || position;
  };

  if (error) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" component="div">
              Captain Recommendation
            </Typography>
            <Button
              size="small"
              startIcon={<RefreshIcon />}
              onClick={handleRefresh}
              disabled={refreshing}
            >
              Retry
            </Button>
          </Box>
          <Alert severity="error">
            {error}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" component="div">
            Captain Recommendation
          </Typography>
          <Button
            size="small"
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
            disabled={refreshing || loading}
          >
            Refresh
          </Button>
        </Box>

        {loading ? (
          <Box>
            <Skeleton variant="circular" width={60} height={60} sx={{ mb: 2 }} />
            <Skeleton variant="text" height={30} sx={{ mb: 1 }} />
            <Skeleton variant="text" height={20} sx={{ mb: 2 }} />
            <Skeleton variant="rectangular" height={60} />
          </Box>
        ) : recommendation ? (
          <Box>
            {/* Player Info */}
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Avatar
                sx={{
                  width: 60,
                  height: 60,
                  bgcolor: 'primary.main',
                  mr: 2,
                  fontSize: '1.5rem',
                  fontWeight: 'bold',
                }}
              >
                <CaptainIcon />
              </Avatar>
              <Box sx={{ flex: 1 }}>
                <Typography variant="h6" component="div" fontWeight="bold">
                  {recommendation.player_name}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                  <Chip
                    label={formatPosition(recommendation.position)}
                    size="small"
                    color="primary"
                    variant="outlined"
                  />
                  <Chip
                    label={`£${recommendation.cost}M`}
                    size="small"
                    variant="outlined"
                  />
                </Box>
              </Box>
            </Box>

            {/* Prediction Details */}
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Predicted Points
                </Typography>
                <Typography variant="h6" fontWeight="bold" color="primary.main">
                  {recommendation.predicted_points}
                </Typography>
              </Box>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Captain Points (2x)
                </Typography>
                <Typography variant="h6" fontWeight="bold" color="success.main">
                  {recommendation.expected_captain_points}
                </Typography>
              </Box>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Confidence
                </Typography>
                <Chip
                  label={getConfidenceLabel(recommendation.confidence)}
                  color={getConfidenceColor(recommendation.confidence)}
                  size="small"
                />
              </Box>

              {/* Confidence Bar */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  Confidence: {Math.round(recommendation.confidence * 100)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={recommendation.confidence * 100}
                  color={getConfidenceColor(recommendation.confidence)}
                  sx={{ mt: 0.5, height: 6, borderRadius: 3 }}
                />
              </Box>
            </Box>

            {/* Reasoning */}
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <InfoIcon fontSize="small" color="primary" sx={{ mr: 0.5 }} />
                <Typography variant="subtitle2" color="text.secondary">
                  AI Analysis
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ fontStyle: 'italic' }}>
                {recommendation.reasoning}
              </Typography>
            </Box>

            {/* Action Button */}
            <Button
              variant="contained"
              fullWidth
              startIcon={<CaptainIcon />}
              sx={{
                background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
                fontWeight: 'bold',
              }}
            >
              Set as Captain
            </Button>

            {/* Additional Info */}
            <Box sx={{ mt: 2, p: 1, bgcolor: 'background.default', borderRadius: 1 }}>
              <Typography variant="caption" color="text.secondary">
                💡 Tip: Captain choice doubles the points scored by your selected player. 
                Choose wisely based on fixture difficulty and recent form.
              </Typography>
            </Box>
          </Box>
        ) : (
          <Alert severity="info">
            No captain recommendation available. Train ML models first.
          </Alert>
        )}

        {(refreshing || loading) && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
              {loading ? 'Loading captain recommendation...' : 'Refreshing recommendation...'}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default CaptainRecommendation;