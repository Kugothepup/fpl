/**
 * FPL Manager v3 - Top Predictions Widget
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Chip,
  LinearProgress,
  Alert,
  Skeleton,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  VisibilityOff,
} from '@mui/icons-material';

import { apiService } from '../../services/api';

const TopPredictions = () => {
  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadPredictions();
  }, []);

  const loadPredictions = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await apiService.predictPlayerPoints();
      
      if (response.success) {
        // Take top 5 predictions
        setPredictions(response.data.slice(0, 5));
      } else {
        throw new Error(response.error || 'Failed to get predictions');
      }

      setLoading(false);
    } catch (err) {
      console.error('Top predictions error:', err);
      setError(apiService.handleError(err, 'Failed to load predictions'));
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadPredictions();
    setRefreshing(false);
  };

  const getPositionColor = (position) => {
    const colors = {
      'GK': 'warning',
      'DEF': 'info',
      'MID': 'success',
      'FWD': 'error'
    };
    return colors[position] || 'default';
  };

  const getConfidenceWidth = (confidence) => {
    return Math.round(confidence * 100);
  };

  if (error) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" component="div">
              Top Predictions
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
      <CardContent sx={{ pb: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" component="div">
            Top Predictions
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
            {[...Array(5)].map((_, index) => (
              <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Skeleton variant="circular" width={40} height={40} sx={{ mr: 2 }} />
                <Box sx={{ flex: 1 }}>
                  <Skeleton variant="text" height={20} sx={{ mb: 0.5 }} />
                  <Skeleton variant="text" height={16} width="60%" />
                </Box>
                <Skeleton variant="text" width={40} height={20} />
              </Box>
            ))}
          </Box>
        ) : predictions.length > 0 ? (
          <List dense sx={{ pt: 0 }}>
            {predictions.map((prediction, index) => (
              <ListItem
                key={prediction.player_id}
                sx={{
                  border: '2px solid',
                  borderColor: index === 0 ? '#d32f2f' : '#e0e0e0',
                  borderRadius: 2,
                  mb: 1,
                  bgcolor: index === 0 ? '#ffebee' : '#f9f9f9',
                }}
              >
                <ListItemAvatar>
                  <Avatar
                    sx={{
                      bgcolor: index === 0 ? '#d32f2f' : '#666666',
                      color: 'white',
                      width: 32,
                      height: 32,
                      fontSize: '0.875rem',
                      fontWeight: 'bold'
                    }}
                  >
                    {index + 1}
                  </Avatar>
                </ListItemAvatar>
                
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2" fontWeight="bold" color={index === 0 ? '#d32f2f' : '#333333'}>
                        {prediction.player_name}
                      </Typography>
                      <Chip
                        label={prediction.position}
                        size="small"
                        color={getPositionColor(prediction.position)}
                        variant="outlined"
                        sx={{ height: 20 }}
                      />
                    </Box>
                  }
                  secondary={
                    <Box sx={{ mt: 0.5 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                        <Typography variant="caption" color={index === 0 ? '#d32f2f' : '#666666'} sx={{ fontWeight: 'medium' }}>
                          Confidence: {Math.round(prediction.confidence * 100)}%
                        </Typography>
                        <Typography variant="caption" color="primary.main" fontWeight="bold">
                          £{prediction.cost}M
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={getConfidenceWidth(prediction.confidence)}
                        color={prediction.confidence > 0.7 ? 'success' : prediction.confidence > 0.5 ? 'warning' : 'error'}
                        sx={{ height: 3, borderRadius: 1.5 }}
                      />
                    </Box>
                  }
                />
                
                <Box sx={{ textAlign: 'right', ml: 1 }}>
                  <Typography variant="h6" color="primary.main" fontWeight="bold">
                    {prediction.predicted_points}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    pts
                  </Typography>
                </Box>
              </ListItem>
            ))}
          </List>
        ) : (
          <Alert severity="info" icon={<VisibilityOff />}>
            No predictions available. Train ML models to see player predictions.
          </Alert>
        )}

        {predictions.length > 0 && (
          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Button
              variant="outlined"
              size="small"
              onClick={() => window.location.href = '/predictions'}
            >
              View All Predictions
            </Button>
          </Box>
        )}

        {(refreshing || loading) && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
              {loading ? 'Loading ML predictions...' : 'Refreshing predictions...'}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default TopPredictions;