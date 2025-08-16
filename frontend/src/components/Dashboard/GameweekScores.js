/**
 * FPL Manager v3 - Gameweek Scores Widget
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
  Chip,
  LinearProgress,
} from '@mui/material';
import { SportsScore, Schedule } from '@mui/icons-material';
import { fetchGameweekScores } from '../../services/api';

const GameweekScores = () => {
  const [scores, setScores] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadGameweekScores();
  }, []);

  const loadGameweekScores = async () => {
    try {
      setLoading(true);
      const response = await fetchGameweekScores();
      setScores(response.data || []);
      setError(null);
    } catch (err) {
      setError('Failed to load score predictions');
      setScores([]);
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <SportsScore sx={{ mr: 1 }} />
          <Typography variant="h6" component="div">
            Gameweek Score Predictions
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
        ) : scores.length === 0 ? (
          <Alert severity="info" sx={{ fontSize: '0.8rem' }}>
            No upcoming fixtures found
          </Alert>
        ) : (
          <Grid container spacing={1}>
            {scores.slice(0, 6).map((match, index) => (
              <Grid item xs={12} key={index}>
                <Box 
                  sx={{ 
                    p: 1.5, 
                    border: '2px solid #f57c00', 
                    borderRadius: 1, 
                    mb: 1,
                    backgroundColor: '#fff3e0',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="body2" fontWeight="bold" color="#f57c00">
                      {match.home_team} vs {match.away_team}
                    </Typography>
                    <Chip 
                      label={match.predicted_score} 
                      size="small" 
                      color="primary"
                      sx={{ fontWeight: 'bold' }}
                    />
                  </Box>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="caption" color="#f57c00" sx={{ fontWeight: 'medium' }}>
                      <Schedule sx={{ fontSize: 12, mr: 0.5, color: '#f57c00' }} />
                      {match.kickoff_time ? new Date(match.kickoff_time).toLocaleString() : 'TBD'}
                    </Typography>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="caption" color="#f57c00" sx={{ fontWeight: 'bold' }}>
                        {(match.confidence * 100).toFixed(0)}%
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={match.confidence * 100} 
                        color={getConfidenceColor(match.confidence)}
                        sx={{ width: 40, height: 4 }}
                      />
                    </Box>
                  </Box>
                </Box>
              </Grid>
            ))}
          </Grid>
        )}
      </CardContent>
    </Card>
  );
};

export default GameweekScores;