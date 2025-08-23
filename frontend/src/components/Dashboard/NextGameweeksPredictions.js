/**
 * FPL Manager v3 - Next Gameweeks Predictions Component
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  ExpandMore,
  Timeline,
  Star,
  Person,
  TrendingUp,
  Schedule,
  Refresh,
} from '@mui/icons-material';
import { apiService } from '../../services/api';

const NextGameweeksPredictions = ({ teamId = null, gameweeks = 2 }) => {
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    loadPredictions();
  }, [teamId, gameweeks]);

  const loadPredictions = async () => {
    try {
      setLoading(true);
      const response = await apiService.getTeamNextGameweeksPredictions(teamId, gameweeks);
      setPredictions(response.data);
      setError(null);
    } catch (err) {
      setError(apiService.handleError(err, 'Failed to load gameweek predictions'));
    } finally {
      setLoading(false);
    }
  };

  const getPositionColor = (position) => {
    const colors = {
      'GK': '#1976d2',
      'DEF': '#2e7d32', 
      'MID': '#f57c00',
      'FWD': '#d32f2f'
    };
    return colors[position] || '#666666';
  };

  const formatDate = (dateString) => {
    if (!dateString) return '';
    return new Date(dateString).toLocaleDateString('en-GB', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleAccordionChange = (panel) => (event, isExpanded) => {
    setExpanded(isExpanded ? panel : false);
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
            <CircularProgress size={24} />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">{error}</Alert>
        </CardContent>
      </Card>
    );
  }

  if (!predictions) {
    return (
      <Card>
        <CardContent>
          <Alert severity="info">No predictions available</Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Timeline sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6" component="div">
              Next {gameweeks} Gameweeks Predictions
            </Typography>
          </Box>
          <Tooltip title="Refresh predictions">
            <IconButton size="small" onClick={loadPredictions}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Summary */}
        <Box sx={{ mb: 2, p: 2, backgroundColor: '#e8f5e8', borderRadius: 1, border: '1px solid #4caf50' }}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="caption" color="textSecondary">
                Total Predicted Points
              </Typography>
              <Typography variant="h6" fontWeight="bold" color="#2e7d32">
                {predictions.summary?.total_predicted_points || 0}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="caption" color="textSecondary">
                Average per GW
              </Typography>
              <Typography variant="h6" fontWeight="bold" color="#2e7d32">
                {predictions.summary?.average_per_gameweek || 0}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="caption" color="textSecondary">
                Confidence
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                {Math.round((predictions.summary?.average_confidence || 0) * 100)}%
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="caption" color="textSecondary">
                Current GW
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                {predictions.summary?.current_gameweek || 'N/A'}
              </Typography>
            </Grid>
          </Grid>
        </Box>

        {/* Gameweek Predictions */}
        {predictions.gameweek_predictions?.map((gwPred, index) => (
          <Accordion 
            key={gwPred.gameweek}
            expanded={expanded === `gw${gwPred.gameweek}`}
            onChange={handleAccordionChange(`gw${gwPred.gameweek}`)}
            sx={{ mb: 1 }}
          >
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', mr: 2 }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  Gameweek {gwPred.gameweek}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Chip 
                    label={`${gwPred.total_predicted_points} pts`}
                    color="primary"
                    size="small"
                    sx={{ fontWeight: 'bold' }}
                  />
                  <Chip 
                    label={`${gwPred.num_fixtures} fixtures`}
                    variant="outlined"
                    size="small"
                  />
                </Box>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              {/* Fixtures */}
              {gwPred.fixtures && gwPred.fixtures.length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                    <Schedule sx={{ fontSize: 16, mr: 0.5 }} />
                    Key Fixtures
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {gwPred.fixtures.slice(0, 6).map((fixture) => (
                      <Chip
                        key={fixture.id}
                        label={`${fixture.home_team} vs ${fixture.away_team}`}
                        variant="outlined"
                        size="small"
                        sx={{ fontSize: '0.7rem' }}
                      />
                    ))}
                  </Box>
                </Box>
              )}

              {/* Player Predictions */}
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Starting XI Predictions:
              </Typography>
              <TableContainer component={Paper} sx={{ boxShadow: 'none', border: '1px solid #e0e0e0' }}>
                <Table size="small">
                  <TableHead>
                    <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                      <TableCell sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold' }}>Player</TableCell>
                      <TableCell align="center" sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold' }}>Predicted</TableCell>
                      <TableCell align="center" sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold' }}>Final</TableCell>
                      <TableCell align="center" sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold' }}>Fixture</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {gwPred.player_predictions?.slice(0, 11).map((player) => (
                      <TableRow 
                        key={player.player_id}
                        sx={{ 
                          backgroundColor: player.is_captain 
                            ? 'rgba(255, 215, 0, 0.1)' 
                            : player.is_vice_captain 
                            ? 'rgba(192, 192, 192, 0.1)' 
                            : 'transparent',
                          border: player.is_captain ? '1px solid gold' : 'none',
                          '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.04)' }
                        }}
                      >
                        <TableCell sx={{ padding: '6px 4px' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Typography variant="body2" sx={{ fontSize: '0.75rem', fontWeight: '500' }}>
                              {player.player_name}
                            </Typography>
                            {player.is_captain && <Star sx={{ fontSize: 10, color: 'gold' }} />}
                            {player.is_vice_captain && <Person sx={{ fontSize: 10, color: 'silver' }} />}
                            <Chip 
                              label={player.position} 
                              size="small"
                              sx={{ 
                                height: 14, 
                                fontSize: '0.55rem',
                                bgcolor: getPositionColor(player.position),
                                color: 'white',
                                fontWeight: 'bold'
                              }}
                            />
                          </Box>
                        </TableCell>
                        <TableCell align="center" sx={{ padding: '6px 4px' }}>
                          <Typography 
                            variant="body2" 
                            sx={{ 
                              fontSize: '0.75rem',
                              fontWeight: '500',
                              color: player.predicted_points > 0 ? '#2e7d32' : 'text.secondary'
                            }}
                          >
                            {player.predicted_points}
                          </Typography>
                        </TableCell>
                        <TableCell align="center" sx={{ padding: '6px 4px' }}>
                          <Typography 
                            variant="body2" 
                            sx={{ 
                              fontSize: '0.75rem',
                              fontWeight: '500',
                              color: player.final_points > 0 ? '#1976d2' : 'text.secondary'
                            }}
                          >
                            {player.final_points}
                          </Typography>
                        </TableCell>
                        <TableCell align="center" sx={{ padding: '6px 4px' }}>
                          <Chip
                            label={player.has_fixture ? 'Yes' : 'No'}
                            color={player.has_fixture ? 'success' : 'error'}
                            size="small"
                            sx={{ height: 16, fontSize: '0.6rem' }}
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>

              <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="caption" color="textSecondary">
                  Confidence: {Math.round((gwPred.confidence || 0) * 100)}%
                </Typography>
                <Typography variant="caption" color="primary" sx={{ fontWeight: 'bold' }}>
                  Total: {gwPred.total_predicted_points} points
                </Typography>
              </Box>
            </AccordionDetails>
          </Accordion>
        ))}

        {/* Footer */}
        <Typography variant="caption" color="textSecondary" sx={{ mt: 2, display: 'block', textAlign: 'center' }}>
          Predictions based on ML models and fixture difficulty. Captain points are doubled.
        </Typography>
      </CardContent>
    </Card>
  );
};

export default NextGameweeksPredictions;