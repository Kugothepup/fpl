/**
 * FPL Manager v3 - My Team Widget
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  CircularProgress,
  Alert,
  Avatar,
  Chip,
  Divider,
  IconButton,
  Collapse,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import { 
  Groups, 
  Star, 
  ExpandMore, 
  ExpandLess, 
  Person,
  Schedule
} from '@mui/icons-material';
import { apiService } from '../../services/api';

const MyTeamWidget = () => {
  const [teamData, setTeamData] = useState(null);
  const [teamPredictions, setTeamPredictions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    loadTeamData();
  }, []);

  const loadTeamData = async () => {
    try {
      setLoading(true);
      
      // Load both team data and predictions
      const [teamResponse, predictionsResponse] = await Promise.allSettled([
        apiService.getUserTeam(),
        apiService.getTeamScorePrediction()
      ]);

      if (teamResponse.status === 'fulfilled') {
        setTeamData(teamResponse.value.data);
      }

      if (predictionsResponse.status === 'fulfilled') {
        setTeamPredictions(predictionsResponse.value.data);
      }

      setError(null);
    } catch (err) {
      setError('Failed to load team data');
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

  const mapElementTypeToPosition = (elementType) => {
    const positionMap = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'};
    return positionMap[elementType] || 'Unknown';
  };

  const formatLastUpdated = (dateString) => {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleString();
  };

  const getPlayerPrediction = (playerId) => {
    if (!teamPredictions?.player_predictions) return null;
    return teamPredictions.player_predictions.find(p => p.player_id === playerId);
  };

  const getEnhancedPlayerData = () => {
    if (!teamData?.picks) return [];
    
    return teamData.picks.slice(0, 11).map((pick, index) => {
      const prediction = getPlayerPrediction(pick.element);
      const actualPoints = pick.total_points || 0;
      const predictedPoints = prediction?.predicted_points || 0;
      const difference = predictedPoints - actualPoints;
      
      return {
        ...pick,
        index: index + 1,
        prediction,
        actualPoints,
        predictedPoints,
        difference
      };
    });
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Groups sx={{ mr: 1 }} />
            <Typography variant="h6" component="div">
              My FPL Team 
            </Typography>
          </Box>
          <IconButton
            onClick={() => setExpanded(!expanded)}
            size="small"
          >
            {expanded ? <ExpandLess /> : <ExpandMore />}
          </IconButton>
        </Box>

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
            <CircularProgress size={24} />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ fontSize: '0.8rem' }}>
            {error}
          </Alert>
        ) : teamData ? (
          <>
            {/* Team Header */}
            <Box sx={{ mb: 2, p: 2, backgroundColor: '#e3f2fd', borderRadius: 1, border: '1px solid #1976d2' }}>
              <Typography variant="subtitle1" fontWeight="bold" color="#1976d2">
                {teamData.entry?.name || 'My Team'}
              </Typography>
              <Typography variant="caption" color="#1976d2" sx={{ fontWeight: 'medium' }}>
                {teamData.entry?.player_first_name} {teamData.entry?.player_last_name}
              </Typography>
              {teamData.entry_history?.bank && (
                <Typography variant="caption" sx={{ ml: 1, color: '#1976d2', fontWeight: 'medium' }}>
                  • £{(teamData.entry_history.bank / 10).toFixed(1)}M in bank
                </Typography>
              )}
            </Box>

            {/* Team Stats */}
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={6}>
                <Typography variant="caption" color="textSecondary">
                  Total Points
                </Typography>
                <Typography variant="body1" fontWeight="bold">
                  {teamData.entry_history?.total_points || 0}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="textSecondary">
                  Overall Rank
                </Typography>
                <Typography variant="body1" fontWeight="bold">
                  {teamData.entry_history?.overall_rank?.toLocaleString() || 'N/A'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="textSecondary">
                  GW Points
                </Typography>
                <Typography variant="body1" fontWeight="bold">
                  {teamData.entry_history?.points || 0}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="textSecondary">
                  GW Rank
                </Typography>
                <Typography variant="body1" fontWeight="bold">
                  {teamData.entry_history?.rank?.toLocaleString() || 'N/A'}
                </Typography>
              </Grid>
            </Grid>

            {/* Last Updated */}
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, fontSize: '0.75rem', color: 'text.secondary' }}>
              <Schedule sx={{ fontSize: 14, mr: 0.5 }} />
              <Typography variant="caption">
                Updated: {formatLastUpdated(teamData.last_deadline_total_transfers)}
              </Typography>
            </Box>

            {/* Team Players (Expandable) */}
            <Collapse in={expanded}>
              <Divider sx={{ mb: 1 }} />
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Current Squad:
              </Typography>
              
              {teamData.picks && teamData.picks.length > 0 ? (
                <TableContainer component={Paper} sx={{ boxShadow: 'none', border: '1px solid #e0e0e0' }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                        <TableCell sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold', color: '#333333' }}>#</TableCell>
                        <TableCell sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold', color: '#333333' }}>Player</TableCell>
                        <TableCell align="center" sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold', color: '#333333' }}>Pred</TableCell>
                        <TableCell align="center" sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold', color: '#333333' }}>Actual</TableCell>
                        <TableCell align="center" sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold', color: '#333333' }}>Diff</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {getEnhancedPlayerData().map((player) => (
                        <TableRow 
                          key={player.element}
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
                            <Avatar 
                              sx={{ 
                                width: 24, 
                                height: 24, 
                                fontSize: '0.65rem',
                                bgcolor: getPositionColor(player.position || mapElementTypeToPosition(player.element_type)),
                                color: 'white',
                                fontWeight: 'bold'
                              }}
                            >
                              {player.index}
                            </Avatar>
                          </TableCell>
                          <TableCell sx={{ padding: '6px 4px' }}>
                            <Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                                <Typography variant="body2" sx={{ fontSize: '0.75rem', fontWeight: '500' }}>
                                  {player.name || `Player ${player.element}`}
                                </Typography>
                                {player.is_captain && <Star sx={{ fontSize: 10, color: 'gold' }} />}
                                {player.is_vice_captain && <Person sx={{ fontSize: 10, color: 'silver' }} />}
                                <Chip 
                                  label={player.position || mapElementTypeToPosition(player.element_type)} 
                                  size="small"
                                  sx={{ 
                                    height: 14, 
                                    fontSize: '0.55rem',
                                    bgcolor: getPositionColor(player.position || mapElementTypeToPosition(player.element_type)),
                                    color: 'white',
                                    fontWeight: 'bold'
                                  }}
                                />
                              </Box>
                              <Typography variant="caption" color="textSecondary" sx={{ fontSize: '0.65rem' }}>
                                {player.team || 'Unknown'} • £{player.cost}M
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="center" sx={{ padding: '6px 4px' }}>
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                fontSize: '0.75rem',
                                fontWeight: '500',
                                color: player.predictedPoints > 0 ? '#2e7d32' : 'text.secondary'
                              }}
                            >
                              {player.predictedPoints.toFixed(1)}
                            </Typography>
                          </TableCell>
                          <TableCell align="center" sx={{ padding: '6px 4px' }}>
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                fontSize: '0.75rem',
                                fontWeight: '500',
                                color: player.actualPoints > 0 ? '#1976d2' : 'text.secondary'
                              }}
                            >
                              {player.actualPoints}
                            </Typography>
                          </TableCell>
                          <TableCell align="center" sx={{ padding: '6px 4px' }}>
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                fontSize: '0.75rem',
                                fontWeight: '500',
                                color: player.difference > 0 
                                  ? '#2e7d32' 
                                  : player.difference < 0 
                                  ? '#d32f2f' 
                                  : 'text.secondary'
                              }}
                            >
                              {player.difference > 0 ? '+' : ''}{player.difference.toFixed(1)}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Alert severity="info" sx={{ fontSize: '0.75rem' }}>
                  No squad data available
                </Alert>
              )}

              {teamData.picks && teamData.picks.length > 11 && (
                <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
                  + {teamData.picks.length - 11} bench players
                </Typography>
              )}
            </Collapse>

            {!expanded && teamData.picks && (
              <Typography variant="caption" color="primary" sx={{ cursor: 'pointer' }}>
                Click to view {teamData.picks.length} players
              </Typography>
            )}
          </>
        ) : (
          <Alert severity="info">
            Team data not available
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default MyTeamWidget;