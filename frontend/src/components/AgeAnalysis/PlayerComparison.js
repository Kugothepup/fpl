/**
 * Player Comparison Component for Age Analysis
 * Allows users to select and compare specific players in age-performance context
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Alert,
  Grid,
  Autocomplete,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  CircularProgress,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Compare as CompareIcon,
} from '@mui/icons-material';

// Import API service
import { apiService } from '../../services/api';

const PlayerComparison = ({ selectedMetric = 'points' }) => {
  // State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [players, setPlayers] = useState([]);
  const [selectedPlayers, setSelectedPlayers] = useState([]);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  // Load players data
  useEffect(() => {
    loadPlayers();
  }, []);

  const loadPlayers = async () => {
    try {
      const response = await apiService.getPlayers({ limit: 200 });
      setPlayers(response.data || []);
    } catch (err) {
      console.error('Error loading players:', err);
      setError('Failed to load players data');
    }
  };

  const handleAddPlayer = (player) => {
    if (player && !selectedPlayers.find(p => p.id === player.id)) {
      setSelectedPlayers([...selectedPlayers, player]);
      setSearchTerm('');
    }
  };

  const handleRemovePlayer = (playerId) => {
    setSelectedPlayers(selectedPlayers.filter(p => p.id !== playerId));
  };

  const handleCompare = async () => {
    if (selectedPlayers.length < 2) {
      setError('Please select at least 2 players to compare');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const playerIds = selectedPlayers.map(p => p.id);
      const response = await apiService.comparePlayersByAge(playerIds, selectedMetric);
      setComparisonResult(response.data);
      
    } catch (err) {
      console.error('Comparison error:', err);
      setError('Failed to perform player comparison');
    } finally {
      setLoading(false);
    }
  };

  const getPerformanceCategoryColor = (category) => {
    switch (category) {
      case 'Above Expected': return 'success';
      case 'Below Expected': return 'error';
      case 'As Expected': return 'info';
      default: return 'default';
    }
  };

  // Filter players for autocomplete
  const filteredPlayers = players.filter(player =>
    player.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    player.team?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Player Age-Performance Comparison
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Player Selection */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Select Players to Compare
          </Typography>
          
          <Autocomplete
            options={filteredPlayers}
            getOptionLabel={(option) => `${option.name} (${option.team || 'Unknown'}) - ${option.position}`}
            renderInput={(params) => (
              <TextField
                {...params}
                label="Search and select players"
                variant="outlined"
                fullWidth
              />
            )}
            onChange={(event, value) => {
              if (value) {
                handleAddPlayer(value);
              }
            }}
            value={null}
            inputValue={searchTerm}
            onInputChange={(event, newValue) => setSearchTerm(newValue)}
            renderOption={(props, option) => (
              <Box component="li" {...props}>
                <Box>
                  <Typography variant="body1">{option.name}</Typography>
                  <Typography variant="caption" color="textSecondary">
                    {option.team} • {option.position} • £{option.cost}m
                  </Typography>
                </Box>
              </Box>
            )}
            noOptionsText="No players found"
          />
        </Box>

        {/* Selected Players */}
        {selectedPlayers.length > 0 && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Selected Players ({selectedPlayers.length})
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {selectedPlayers.map((player) => (
                <Chip
                  key={player.id}
                  label={`${player.name} (${player.position})`}
                  onDelete={() => handleRemovePlayer(player.id)}
                  color="primary"
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>
        )}

        {/* Compare Button */}
        <Button
          variant="contained"
          onClick={handleCompare}
          disabled={loading || selectedPlayers.length < 2}
          startIcon={loading ? <CircularProgress size={20} /> : <CompareIcon />}
          fullWidth
          sx={{ mb: 3 }}
        >
          {loading ? 'Comparing...' : `Compare Players (${selectedMetric})`}
        </Button>

        {/* Comparison Results */}
        {comparisonResult && comparisonResult.player_comparisons && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Comparison Results
            </Typography>
            
            {/* Analysis Metadata */}
            <Alert severity="info" sx={{ mb: 2 }}>
              Analysis based on {comparisonResult.analysis_metadata.total_players_analyzed} players • 
              Model confidence: {(comparisonResult.analysis_metadata.model_r2 * 100).toFixed(1)}% • 
              {comparisonResult.analysis_metadata.peak_age && 
                ` Peak age: ${comparisonResult.analysis_metadata.peak_age} years`
              }
            </Alert>

            {/* Comparison Table */}
            <TableContainer component={Paper} sx={{ backgroundColor: 'rgba(255, 255, 255, 0.05)' }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Player</TableCell>
                    <TableCell>Age</TableCell>
                    <TableCell>Position</TableCell>
                    <TableCell>Team</TableCell>
                    <TableCell align="right">Actual {selectedMetric}</TableCell>
                    <TableCell align="right">Expected {selectedMetric}</TableCell>
                    <TableCell align="center">Performance</TableCell>
                    <TableCell align="right">Minutes</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {comparisonResult.player_comparisons
                    .sort((a, b) => b.performance_ratio - a.performance_ratio)
                    .map((player) => (
                    <TableRow key={player.player_id}>
                      <TableCell>
                        <Typography variant="body2" fontWeight="bold">
                          {player.name}
                        </Typography>
                      </TableCell>
                      <TableCell>{player.age}</TableCell>
                      <TableCell>{player.position}</TableCell>
                      <TableCell>{player.team}</TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" fontWeight="bold">
                          {player.actual_value}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        {player.predicted_value.toFixed(1)}
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={`${(player.performance_ratio * 100).toFixed(0)}%`}
                          color={getPerformanceCategoryColor(player.performance_category)}
                          size="small"
                        />
                        <Typography variant="caption" display="block">
                          {player.performance_category}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">{player.minutes}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            {/* Performance Insights */}
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Performance Insights
              </Typography>
              <Grid container spacing={2}>
                {comparisonResult.player_comparisons.map((player) => (
                  <Grid item xs={12} sm={6} md={4} key={player.player_id}>
                    <Alert 
                      severity={
                        player.performance_category === 'Above Expected' ? 'success' :
                        player.performance_category === 'Below Expected' ? 'warning' : 'info'
                      }
                      variant="outlined"
                    >
                      <Typography variant="body2">
                        <strong>{player.name}</strong> ({player.age}y) is performing{' '}
                        <strong>{player.performance_category.toLowerCase()}</strong> for their age,
                        with {player.actual_value} vs expected {player.predicted_value.toFixed(1)} {selectedMetric}.
                      </Typography>
                    </Alert>
                  </Grid>
                ))}
              </Grid>
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default PlayerComparison;