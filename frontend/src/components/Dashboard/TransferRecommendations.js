/**
 * FPL Manager v3 - Transfer Recommendations Component
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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Button,
  TextField,
  InputAdornment,
} from '@mui/material';
import {
  SwapHoriz,
  TrendingUp,
  TrendingDown,
  AttachMoney,
  Refresh,
  ArrowForward,
  Star,
  CompareArrows,
} from '@mui/icons-material';
import { apiService } from '../../services/api';

const TransferRecommendations = ({ teamId = null, teamData = null }) => {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [budget, setBudget] = useState(0);
  const [freeTransfers, setFreeTransfers] = useState(1);

  useEffect(() => {
    // Extract budget and free transfers from team data if available
    if (teamData?.entry_history) {
      setBudget(teamData.entry_history.bank / 10 || 0); // Convert from £0.1M to £M
    }
    if (teamData?.entry) {
      setFreeTransfers(teamData.entry.current_event_transfers || 1);
    }
  }, [teamData]);

  useEffect(() => {
    if (budget >= 0) {
      loadRecommendations();
    }
  }, [teamId, budget, freeTransfers]);

  const loadRecommendations = async () => {
    try {
      setLoading(true);
      const response = await apiService.getTransferRecommendations(teamId, budget, freeTransfers);
      setRecommendations(response.data);
      setError(null);
    } catch (err) {
      setError(apiService.handleError(err, 'Failed to load transfer recommendations'));
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

  const formatCurrency = (value) => {
    return `£${value?.toFixed(1)}M`;
  };

  const getPriorityColor = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'high': return '#d32f2f';
      case 'medium': return '#f57c00';
      case 'low': return '#2e7d32';
      default: return '#666666';
    }
  };

  const handleParameterChange = () => {
    loadRecommendations();
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

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <SwapHoriz sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6" component="div">
              Transfer Recommendations
            </Typography>
          </Box>
          <Tooltip title="Refresh recommendations">
            <IconButton size="small" onClick={loadRecommendations}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Parameters */}
        <Box sx={{ mb: 3, p: 2, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={4}>
              <TextField
                label="Available Budget"
                value={budget}
                onChange={(e) => setBudget(parseFloat(e.target.value) || 0)}
                onBlur={handleParameterChange}
                type="number"
                size="small"
                fullWidth
                InputProps={{
                  startAdornment: <InputAdornment position="start">£</InputAdornment>,
                  endAdornment: <InputAdornment position="end">M</InputAdornment>,
                }}
                inputProps={{ step: 0.1, min: 0 }}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                label="Free Transfers"
                value={freeTransfers}
                onChange={(e) => setFreeTransfers(parseInt(e.target.value) || 1)}
                onBlur={handleParameterChange}
                type="number"
                size="small"
                fullWidth
                inputProps={{ min: 1, max: 5 }}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <Button
                variant="outlined"
                onClick={handleParameterChange}
                size="small"
                startIcon={<Refresh />}
                fullWidth
              >
                Update
              </Button>
            </Grid>
          </Grid>
        </Box>

        {recommendations ? (
          <>
            {/* Summary */}
            {recommendations.summary && (
              <Box sx={{ mb: 2, p: 2, backgroundColor: '#e3f2fd', borderRadius: 1, border: '1px solid #1976d2' }}>
                <Typography variant="subtitle2" sx={{ mb: 1, color: '#1976d2', fontWeight: 'bold' }}>
                  Transfer Strategy Summary
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="textSecondary">
                      Expected Points Gain
                    </Typography>
                    <Typography variant="body1" fontWeight="bold" color="#2e7d32">
                      +{recommendations.summary.expected_points_gain?.toFixed(1) || 0}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="textSecondary">
                      Transfer Cost
                    </Typography>
                    <Typography variant="body1" fontWeight="bold" color={recommendations.summary.transfer_cost > 0 ? '#d32f2f' : '#2e7d32'}>
                      {recommendations.summary.transfer_cost > 0 ? `-${recommendations.summary.transfer_cost}` : '0'} pts
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="textSecondary">
                      Budget Used
                    </Typography>
                    <Typography variant="body1" fontWeight="bold">
                      {formatCurrency(recommendations.summary.budget_used || 0)}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="textSecondary">
                      Transfers Used
                    </Typography>
                    <Typography variant="body1" fontWeight="bold">
                      {recommendations.summary.transfers_used || 0}/{freeTransfers}
                    </Typography>
                  </Grid>
                </Grid>
              </Box>
            )}

            {/* Transfer Recommendations */}
            {recommendations.transfers && recommendations.transfers.length > 0 ? (
              <TableContainer component={Paper} sx={{ boxShadow: 'none', border: '1px solid #e0e0e0' }}>
                <Table size="small">
                  <TableHead>
                    <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                      <TableCell sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold' }}>Transfer</TableCell>
                      <TableCell align="center" sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold' }}>Priority</TableCell>
                      <TableCell align="center" sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold' }}>Points Δ</TableCell>
                      <TableCell align="center" sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold' }}>Cost</TableCell>
                      <TableCell align="center" sx={{ padding: '8px 4px', fontSize: '0.7rem', fontWeight: 'bold' }}>Reason</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {recommendations.transfers.slice(0, 5).map((transfer, index) => (
                      <TableRow 
                        key={index}
                        sx={{ 
                          '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.04)' },
                          backgroundColor: transfer.priority === 'High' ? 'rgba(255, 0, 0, 0.05)' : 'transparent'
                        }}
                      >
                        <TableCell sx={{ padding: '8px 4px' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {/* Player Out */}
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <Avatar 
                                sx={{ 
                                  width: 20, 
                                  height: 20, 
                                  fontSize: '0.6rem',
                                  bgcolor: getPositionColor(transfer.player_out?.position),
                                  color: 'white'
                                }}
                              >
                                {transfer.player_out?.position?.charAt(0) || 'P'}
                              </Avatar>
                              <Typography variant="caption" sx={{ fontSize: '0.7rem', maxWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                {transfer.player_out?.name || 'Unknown'}
                              </Typography>
                            </Box>
                            
                            <ArrowForward sx={{ fontSize: 12, color: 'primary.main' }} />
                            
                            {/* Player In */}
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <Avatar 
                                sx={{ 
                                  width: 20, 
                                  height: 20, 
                                  fontSize: '0.6rem',
                                  bgcolor: getPositionColor(transfer.player_in?.position),
                                  color: 'white'
                                }}
                              >
                                {transfer.player_in?.position?.charAt(0) || 'P'}
                              </Avatar>
                              <Typography variant="caption" sx={{ fontSize: '0.7rem', maxWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis', fontWeight: 'bold' }}>
                                {transfer.player_in?.name || 'Unknown'}
                              </Typography>
                            </Box>
                          </Box>
                        </TableCell>
                        
                        <TableCell align="center" sx={{ padding: '6px 4px' }}>
                          <Chip
                            label={transfer.priority || 'Medium'}
                            size="small"
                            sx={{ 
                              height: 16, 
                              fontSize: '0.6rem',
                              bgcolor: getPriorityColor(transfer.priority),
                              color: 'white',
                              fontWeight: 'bold'
                            }}
                          />
                        </TableCell>
                        
                        <TableCell align="center" sx={{ padding: '6px 4px' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            {(transfer.expected_points_gain || 0) > 0 ? (
                              <TrendingUp sx={{ fontSize: 12, color: '#2e7d32' }} />
                            ) : (
                              <TrendingDown sx={{ fontSize: 12, color: '#d32f2f' }} />
                            )}
                            <Typography 
                              variant="caption" 
                              sx={{ 
                                fontSize: '0.7rem',
                                fontWeight: 'bold',
                                color: (transfer.expected_points_gain || 0) > 0 ? '#2e7d32' : '#d32f2f'
                              }}
                            >
                              {(transfer.expected_points_gain || 0) > 0 ? '+' : ''}{(transfer.expected_points_gain || 0).toFixed(1)}
                            </Typography>
                          </Box>
                        </TableCell>
                        
                        <TableCell align="center" sx={{ padding: '6px 4px' }}>
                          <Typography variant="caption" sx={{ fontSize: '0.7rem', fontWeight: '500' }}>
                            {formatCurrency(transfer.cost_difference || 0)}
                          </Typography>
                        </TableCell>
                        
                        <TableCell sx={{ padding: '6px 4px' }}>
                          <Typography 
                            variant="caption" 
                            sx={{ 
                              fontSize: '0.65rem',
                              maxWidth: 120,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap'
                            }}
                          >
                            {transfer.reason || 'Better predicted performance'}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Alert severity="info" sx={{ fontSize: '0.8rem' }}>
                No beneficial transfers found for your current budget and constraints.
              </Alert>
            )}

            {/* Additional Info */}
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="caption" color="textSecondary">
                Recommendations based on ML predictions and fixture difficulty
              </Typography>
              {recommendations.confidence && (
                <Typography variant="caption" color="primary" sx={{ fontWeight: 'bold' }}>
                  Confidence: {Math.round(recommendations.confidence * 100)}%
                </Typography>
              )}
            </Box>
          </>
        ) : (
          <Alert severity="info">
            No transfer recommendations available
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default TransferRecommendations;