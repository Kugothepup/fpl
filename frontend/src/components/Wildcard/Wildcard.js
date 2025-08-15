/**
 * FPL Manager v3 - Wildcard Helper Component
 */

import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Alert,
  Button,
  Box,
  Grid,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Divider,
} from '@mui/material';
import {
  ArrowBack,
  AutoAwesome,
  Calculate,
  TrendingUp,
  Warning,
  CheckCircle,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

import { apiService } from '../../services/api';

const Wildcard = () => {
  const navigate = useNavigate();
  
  // State
  const [loading, setLoading] = useState(false);
  const [players, setPlayers] = useState([]);
  const [selectedPlayers, setSelectedPlayers] = useState([]);
  const [budget, setBudget] = useState(100.0);
  const [formation, setFormation] = useState('3-4-3');
  const [error, setError] = useState(null);
  const [calculations, setCalculations] = useState(null);

  // Formation requirements
  const formations = {
    '3-4-3': { GK: 1, DEF: 3, MID: 4, FWD: 3 },
    '3-5-2': { GK: 1, DEF: 3, MID: 5, FWD: 2 },
    '4-3-3': { GK: 1, DEF: 4, MID: 3, FWD: 3 },
    '4-4-2': { GK: 1, DEF: 4, MID: 4, FWD: 2 },
    '4-5-1': { GK: 1, DEF: 4, MID: 5, FWD: 1 },
    '5-3-2': { GK: 1, DEF: 5, MID: 3, FWD: 2 },
    '5-4-1': { GK: 1, DEF: 5, MID: 4, FWD: 1 },
  };

  useEffect(() => {
    loadTopPlayers();
  }, []);

  const loadTopPlayers = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await apiService.getPlayers({ limit: 100 });
      
      if (response.success) {
        setPlayers(response.data);
      } else {
        throw new Error(response.error || 'Failed to load players');
      }

      setLoading(false);
    } catch (err) {
      console.error('Players load error:', err);
      setError(apiService.handleError(err, 'Failed to load player data'));
      setLoading(false);
    }
  };

  const calculateTeamStats = () => {
    const totalCost = selectedPlayers.reduce((sum, player) => sum + player.cost, 0);
    const totalPoints = selectedPlayers.reduce((sum, player) => sum + player.total_points, 0);
    const avgPointsPerGame = selectedPlayers.reduce((sum, player) => sum + player.points_per_game, 0) / Math.max(selectedPlayers.length, 1);
    
    const positionCounts = { GK: 0, DEF: 0, MID: 0, FWD: 0 };
    selectedPlayers.forEach(player => {
      positionCounts[player.position] = (positionCounts[player.position] || 0) + 1;
    });

    const formationReq = formations[formation];
    const isValidFormation = Object.keys(formationReq).every(
      pos => positionCounts[pos] >= formationReq[pos]
    );

    const remaining = budget - totalCost;
    const isWithinBudget = remaining >= 0;
    const isValidTeamSize = selectedPlayers.length <= 15;

    return {
      totalCost: totalCost.toFixed(1),
      totalPoints,
      avgPointsPerGame: avgPointsPerGame.toFixed(1),
      remaining: remaining.toFixed(1),
      positionCounts,
      isValidFormation,
      isWithinBudget,
      isValidTeamSize,
      isComplete: selectedPlayers.length === 15 && isValidFormation && isWithinBudget
    };
  };

  const addPlayer = (player) => {
    if (selectedPlayers.length < 15 && !selectedPlayers.find(p => p.id === player.id)) {
      setSelectedPlayers([...selectedPlayers, player]);
    }
  };

  const removePlayer = (playerId) => {
    setSelectedPlayers(selectedPlayers.filter(p => p.id !== playerId));
  };

  const clearTeam = () => {
    setSelectedPlayers([]);
  };

  const optimizeTeam = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/wildcard/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          budget: budget,
          formation: formation,
          constraints: {}
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Map the optimized team to our player format
        const optimizedPlayers = data.data.team.map(player => ({
          id: player.id,
          name: player.name,
          full_name: player.full_name,
          position: player.position,
          cost: player.cost,
          total_points: player.total_points,
          points_per_game: player.points_per_game,
          predicted_points: player.predicted_points,
          confidence: player.confidence,
          form: player.form,
          selected_by_percent: player.selected_by_percent
        }));
        
        setSelectedPlayers(optimizedPlayers);
        setCalculations({
          totalCost: data.data.total_cost,
          totalPredictedPoints: data.data.total_predicted_points,
          avgConfidence: data.data.avg_confidence,
          suggestedCaptain: data.data.suggested_captain,
          isValid: data.data.is_valid
        });
        
        setError(null);
      } else {
        throw new Error(data.error || 'Optimization failed');
      }
      
      setLoading(false);
    } catch (err) {
      setError('Failed to optimize team: ' + err.message);
      setLoading(false);
    }
  };

  const stats = calculateTeamStats();

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 3 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/')}
          sx={{ mb: 2 }}
        >
          Back to Dashboard
        </Button>
        <Typography variant="h4" component="h1" gutterBottom>
          <AutoAwesome sx={{ mr: 1, verticalAlign: 'middle' }} />
          Wildcard Helper
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Build your optimal 15-player squad within FPL rules
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Configuration Panel */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Configuration
              </Typography>
              
              <Box sx={{ mb: 3 }}>
                <TextField
                  fullWidth
                  label="Budget (£M)"
                  type="number"
                  value={budget}
                  onChange={(e) => setBudget(parseFloat(e.target.value) || 100)}
                  inputProps={{ min: 80, max: 120, step: 0.1 }}
                  sx={{ mb: 2 }}
                />
                
                <FormControl fullWidth>
                  <InputLabel>Formation</InputLabel>
                  <Select
                    value={formation}
                    label="Formation"
                    onChange={(e) => setFormation(e.target.value)}
                  >
                    {Object.keys(formations).map(form => (
                      <MenuItem key={form} value={form}>{form}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>

              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Formation Requirements ({formation})
                </Typography>
                {Object.entries(formations[formation]).map(([pos, count]) => (
                  <Chip
                    key={pos}
                    label={`${pos}: ${count}`}
                    variant="outlined"
                    size="small"
                    sx={{ mr: 0.5, mb: 0.5 }}
                  />
                ))}
              </Box>

              <Box sx={{ mb: 2 }}>
                <Button
                  variant="contained"
                  fullWidth
                  startIcon={<Calculate />}
                  onClick={optimizeTeam}
                  disabled={loading}
                  sx={{ mb: 1 }}
                >
                  Optimize with ML
                </Button>
                <Button
                  variant="outlined"
                  fullWidth
                  onClick={clearTeam}
                  disabled={selectedPlayers.length === 0}
                >
                  Clear Team
                </Button>
              </Box>
            </CardContent>
          </Card>

          {/* Team Stats */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Team Statistics
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Players Selected</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {selectedPlayers.length}/15
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={(selectedPlayers.length / 15) * 100}
                  color={selectedPlayers.length === 15 ? 'success' : 'primary'}
                />
              </Box>

              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Budget Used</Typography>
                  <Typography 
                    variant="body2" 
                    fontWeight="bold"
                    color={stats.isWithinBudget ? 'success.main' : 'error.main'}
                  >
                    £{stats.totalCost}M / £{budget}M
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={(parseFloat(stats.totalCost) / budget) * 100}
                  color={stats.isWithinBudget ? 'success' : 'error'}
                />
              </Box>

              <Divider sx={{ my: 2 }} />

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">Total Points</Typography>
                <Typography variant="body2" fontWeight="bold">
                  {stats.totalPoints}
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">Avg PPG</Typography>
                <Typography variant="body2" fontWeight="bold">
                  {stats.avgPointsPerGame}
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2">Remaining</Typography>
                <Typography 
                  variant="body2" 
                  fontWeight="bold"
                  color={stats.isWithinBudget ? 'success.main' : 'error.main'}
                >
                  £{stats.remaining}M
                </Typography>
              </Box>

              {/* Validation Status */}
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Validation
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                  {stats.isWithinBudget ? <CheckCircle color="success" fontSize="small" /> : <Warning color="error" fontSize="small" />}
                  <Typography variant="caption" sx={{ ml: 0.5 }}>
                    Within Budget
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                  {stats.isValidFormation ? <CheckCircle color="success" fontSize="small" /> : <Warning color="error" fontSize="small" />}
                  <Typography variant="caption" sx={{ ml: 0.5 }}>
                    Valid Formation
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {stats.isValidTeamSize ? <CheckCircle color="success" fontSize="small" /> : <Warning color="error" fontSize="small" />}
                  <Typography variant="caption" sx={{ ml: 0.5 }}>
                    Team Size (≤15)
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Player Selection */}
        <Grid item xs={12} md={8}>
          {error && (
            <Alert severity="info" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {/* Selected Team */}
          {selectedPlayers.length > 0 && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Selected Team ({selectedPlayers.length}/15)
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Player</TableCell>
                        <TableCell>Position</TableCell>
                        <TableCell>Team</TableCell>
                        <TableCell align="right">Cost</TableCell>
                        <TableCell align="right">Points</TableCell>
                        <TableCell align="center">Action</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {selectedPlayers.map((player) => (
                        <TableRow key={player.id}>
                          <TableCell>{player.name}</TableCell>
                          <TableCell>
                            <Chip
                              label={player.position}
                              size="small"
                              color="primary"
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>{player.team}</TableCell>
                          <TableCell align="right">£{player.cost}M</TableCell>
                          <TableCell align="right">{player.total_points}</TableCell>
                          <TableCell align="center">
                            <Button
                              size="small"
                              color="error"
                              onClick={() => removePlayer(player.id)}
                            >
                              Remove
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}

          {/* Available Players */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Available Players
              </Typography>
              
              {loading ? (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Loading player data...
                  </Typography>
                </Box>
              ) : (
                <TableContainer component={Paper} sx={{ maxHeight: 600 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>Player</TableCell>
                        <TableCell>Position</TableCell>
                        <TableCell>Team</TableCell>
                        <TableCell align="right">Cost</TableCell>
                        <TableCell align="right">Points</TableCell>
                        <TableCell align="right">PPG</TableCell>
                        <TableCell align="center">Action</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {players.slice(0, 50).map((player) => {
                        const isSelected = selectedPlayers.find(p => p.id === player.id);
                        const canAdd = selectedPlayers.length < 15 && !isSelected;
                        
                        return (
                          <TableRow key={player.id} sx={{ opacity: isSelected ? 0.5 : 1 }}>
                            <TableCell>{player.name}</TableCell>
                            <TableCell>
                              <Chip
                                label={player.position}
                                size="small"
                                color="primary"
                                variant="outlined"
                              />
                            </TableCell>
                            <TableCell>{player.team}</TableCell>
                            <TableCell align="right">£{player.cost}M</TableCell>
                            <TableCell align="right">{player.total_points}</TableCell>
                            <TableCell align="right">{player.points_per_game}</TableCell>
                            <TableCell align="center">
                              <Button
                                size="small"
                                variant={isSelected ? "outlined" : "contained"}
                                disabled={!canAdd && !isSelected}
                                onClick={() => isSelected ? removePlayer(player.id) : addPlayer(player)}
                              >
                                {isSelected ? 'Selected' : 'Add'}
                              </Button>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Wildcard;