/**
 * FPL Manager v3 - Team Score Prediction Widget
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Grid,
  Avatar,
  Chip,
  LinearProgress,
} from '@mui/material';
import { EmojiEvents, Person, Star } from '@mui/icons-material';
import { fetchTeamScore } from '../../services/api';

const TeamScoreWidget = () => {
  const [teamScore, setTeamScore] = useState(null);
  const [teamInfo, setTeamInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadTeamScore();
  }, []);

  const loadTeamScore = async () => {
    try {
      setLoading(true);
      const response = await fetchTeamScore();
      setTeamScore(response.data);
      setTeamInfo(response.team_info);
      setError(null);
    } catch (err) {
      setError('Failed to load team score prediction');
    } finally {
      setLoading(false);
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
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <EmojiEvents sx={{ mr: 1 }} />
          <Typography variant="h6" component="div">
            My Team Score Prediction
          </Typography>
        </Box>

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
            <CircularProgress size={24} />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ fontSize: '0.8rem' }}>
            {error}
          </Alert>
        ) : teamScore && teamInfo ? (
          <>
            {/* Team Info */}
            <Box sx={{ mb: 2, p: 1, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
              <Typography variant="subtitle2" fontWeight="bold">
                {teamInfo.name}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                {teamInfo.player_first_name} {teamInfo.player_last_name}
              </Typography>
              {teamInfo.overall_rank && (
                <Typography variant="caption" sx={{ ml: 1 }}>
                  • Rank: {teamInfo.overall_rank.toLocaleString()}
                </Typography>
              )}
            </Box>

            {/* Predicted Score */}
            <Box sx={{ textAlign: 'center', mb: 2 }}>
              <Typography variant="h3" color="primary" fontWeight="bold">
                {teamScore.total_predicted_points}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Predicted Points (GW {teamScore.gameweek})
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={teamScore.confidence * 100} 
                sx={{ mt: 1, height: 6 }}
              />
              <Typography variant="caption">
                {(teamScore.confidence * 100).toFixed(0)}% confidence
              </Typography>
            </Box>

            {/* Top Players */}
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Top Predicted Performers:
            </Typography>
            
            <Grid container spacing={1}>
              {teamScore.player_predictions.slice(0, 3).map((player, index) => (
                <Grid item xs={12} key={player.player_id}>
                  <Box sx={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: 1, 
                    p: 0.5,
                    backgroundColor: getPositionColor(player.position),
                    borderRadius: 0.5,
                    border: player.is_captain ? '2px solid gold' : 'none'
                  }}>
                    <Avatar sx={{ width: 24, height: 24, fontSize: '0.7rem' }}>
                      {index + 1}
                    </Avatar>
                    
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Typography variant="caption" sx={{ fontWeight: 500 }}>
                        {player.player_name}
                        {player.is_captain && <Star sx={{ fontSize: 14, color: 'gold', ml: 0.5 }} />}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                        <Chip 
                          label={player.position} 
                          size="small" 
                          sx={{ height: 16, fontSize: '0.6rem' }}
                        />
                      </Box>
                    </Box>
                    
                    <Typography variant="caption" fontWeight="bold" color="primary">
                      {player.final_points} pts
                      {player.is_captain && (
                        <Typography variant="caption" color="textSecondary" sx={{ ml: 0.5 }}>
                          (C)
                        </Typography>
                      )}
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>

            {/* Current Stats */}
            {teamInfo.total_points && (
              <Box sx={{ mt: 2, pt: 1, borderTop: '1px solid #e0e0e0' }}>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="textSecondary">
                      Total Points
                    </Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {teamInfo.total_points}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="textSecondary">
                      Last GW
                    </Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {teamInfo.gameweek_points || 0}
                    </Typography>
                  </Grid>
                </Grid>
              </Box>
            )}
          </>
        ) : (
          <Alert severity="info" sx={{ fontSize: '0.8rem' }}>
            No team data available
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default TeamScoreWidget;