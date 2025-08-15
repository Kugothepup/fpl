/**
 * FPL Manager v3 - ML Predictions Component
 */

import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Button,
  Box,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  Chip,
  Avatar,
  LinearProgress,
} from '@mui/material';
import { 
  ArrowBack, 
  TrendingUp, 
  EmojiEvents,
  Psychology,
  Star
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { fetchPredictions, fetchCaptainRecommendation, trainModels } from '../../services/api';

const Predictions = () => {
  const navigate = useNavigate();
  const [predictions, setPredictions] = useState([]);
  const [captainRec, setCaptainRec] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [training, setTraining] = useState(false);

  useEffect(() => {
    loadPredictions();
    loadCaptainRecommendation();
  }, []);

  const loadPredictions = async () => {
    try {
      setLoading(true);
      const response = await fetchPredictions();
      setPredictions(response.data || []);
      setModelInfo(response.model_info);
      setError(null);
    } catch (err) {
      setError('Failed to load predictions: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadCaptainRecommendation = async () => {
    try {
      const response = await fetchCaptainRecommendation();
      setCaptainRec(response.data);
    } catch (err) {
      console.error('Captain recommendation failed:', err);
    }
  };

  const handleTrainModels = async () => {
    try {
      setTraining(true);
      await trainModels();
      await loadPredictions();
      await loadCaptainRecommendation();
    } catch (err) {
      setError('Model training failed: ' + err.message);
    } finally {
      setTraining(false);
    }
  };

  const getPositionColor = (position) => {
    const colors = {
      'GK': '#e3f2fd',
      'DEF': '#e8f5e8', 
      'MID': '#fff3e0',
      'FWD': '#ffebee'
    };
    return colors[position] || '#f5f5f5';
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4 }}>
      <Box sx={{ mb: 3 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/')}
          sx={{ mb: 2 }}
        >
          Back to Dashboard
        </Button>
        <Typography variant="h4" component="h1" gutterBottom>
          ML Predictions
        </Typography>
      </Box>

      {/* Model Status */}
      {modelInfo && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              <Psychology sx={{ mr: 1 }} />
              Model Status
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={3}>
                <Typography variant="body2" color="textSecondary">Best Model</Typography>
                <Typography variant="body1">{modelInfo.latest_performance?.best_model}</Typography>
              </Grid>
              <Grid item xs={12} md={3}>
                <Typography variant="body2" color="textSecondary">Accuracy</Typography>
                <Typography variant="body1">{(modelInfo.latest_performance?.best_score * 100).toFixed(1)}%</Typography>
              </Grid>
              <Grid item xs={12} md={3}>
                <Typography variant="body2" color="textSecondary">Models Available</Typography>
                <Typography variant="body1">{modelInfo.models_available?.length || 0}</Typography>
              </Grid>
              <Grid item xs={12} md={3}>
                <Button
                  variant="contained"
                  onClick={handleTrainModels}
                  disabled={training}
                  startIcon={training ? <CircularProgress size={20} /> : <TrendingUp />}
                >
                  {training ? 'Training...' : 'Retrain Models'}
                </Button>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Captain Recommendation */}
      {captainRec && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              <EmojiEvents sx={{ mr: 1 }} />
              Captain Recommendation
            </Typography>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={8}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Avatar sx={{ bgcolor: getPositionColor(captainRec.position) }}>
                    <Star />
                  </Avatar>
                  <Box>
                    <Typography variant="h6">{captainRec.player_name}</Typography>
                    <Typography variant="body2" color="textSecondary">
                      {captainRec.position} • £{captainRec.cost}m
                    </Typography>
                  </Box>
                </Box>
              </Grid>
              <Grid item xs={12} md={4}>
                <Box sx={{ textAlign: 'right' }}>
                  <Typography variant="h5" color="primary">
                    {captainRec.expected_captain_points} pts
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={captainRec.confidence * 100} 
                    sx={{ mt: 1 }}
                  />
                  <Typography variant="caption">
                    {(captainRec.confidence * 100).toFixed(0)}% confidence
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="body2">{captainRec.reasoning}</Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
          <Button onClick={loadPredictions} sx={{ ml: 2 }}>
            Retry
          </Button>
        </Alert>
      )}

      {/* Predictions Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Next Gameweek Predictions
          </Typography>
          
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress />
            </Box>
          ) : predictions.length === 0 ? (
            <Alert severity="info">
              No predictions available. Try training the models first.
            </Alert>
          ) : (
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Player</TableCell>
                    <TableCell>Position</TableCell>
                    <TableCell>Cost</TableCell>
                    <TableCell>Predicted Points</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Reasoning</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {predictions.slice(0, 20).map((player, index) => (
                    <TableRow key={player.player_id}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Avatar 
                            sx={{ 
                              width: 32, 
                              height: 32, 
                              bgcolor: getPositionColor(player.position),
                              fontSize: '0.8rem'
                            }}
                          >
                            {index + 1}
                          </Avatar>
                          {player.player_name}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={player.position} 
                          size="small"
                          sx={{ bgcolor: getPositionColor(player.position) }}
                        />
                      </TableCell>
                      <TableCell>£{player.cost}m</TableCell>
                      <TableCell>
                        <Typography variant="body1" fontWeight="bold" color="primary">
                          {player.predicted_points}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <LinearProgress 
                            variant="determinate" 
                            value={player.confidence * 100} 
                            sx={{ width: 60, height: 6 }}
                          />
                          <Typography variant="caption">
                            {(player.confidence * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="textSecondary">
                          {player.reasoning}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>
    </Container>
  );
};

export default Predictions;