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
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Divider,
  IconButton,
  Collapse,
} from '@mui/material';
import { 
  Groups, 
  Star, 
  ExpandMore, 
  ExpandLess, 
  EmojiEvents,
  Person,
  Schedule
} from '@mui/icons-material';
import { apiService } from '../../services/api';

const MyTeamWidget = () => {
  const [teamData, setTeamData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    loadTeamData();
  }, []);

  const loadTeamData = async () => {
    try {
      setLoading(true);
      const response = await apiService.getUserTeam();
      setTeamData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to load team data');
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

  const mapElementTypeToPosition = (elementType) => {
    const positionMap = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'};
    return positionMap[elementType] || 'Unknown';
  };

  const formatLastUpdated = (dateString) => {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleString();
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
            <Box sx={{ mb: 2, p: 1, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
              <Typography variant="subtitle1" fontWeight="bold">
                {teamData.name || 'My Team'}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                {teamData.player_first_name} {teamData.player_last_name}
              </Typography>
              {teamData.last_deadline_bank && (
                <Typography variant="caption" sx={{ ml: 1 }}>
                  • £{(teamData.last_deadline_bank / 10).toFixed(1)}M in bank
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
                  {teamData.summary_overall_points || 0}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="textSecondary">
                  Overall Rank
                </Typography>
                <Typography variant="body1" fontWeight="bold">
                  {teamData.summary_overall_rank?.toLocaleString() || 'N/A'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="textSecondary">
                  GW Points
                </Typography>
                <Typography variant="body1" fontWeight="bold">
                  {teamData.summary_event_points || 0}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="textSecondary">
                  GW Rank
                </Typography>
                <Typography variant="body1" fontWeight="bold">
                  {teamData.summary_event_rank?.toLocaleString() || 'N/A'}
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
                <List dense sx={{ p: 0 }}>
                  {teamData.picks.slice(0, 11).map((pick, index) => (
                    <ListItem 
                      key={pick.element} 
                      sx={{ 
                        px: 0, 
                        py: 0.5,
                        backgroundColor: pick.is_captain 
                          ? 'rgba(255, 215, 0, 0.1)' 
                          : pick.is_vice_captain 
                          ? 'rgba(192, 192, 192, 0.1)' 
                          : 'transparent',
                        borderRadius: 0.5,
                        border: pick.is_captain ? '1px solid gold' : 'none'
                      }}
                    >
                      <ListItemAvatar>
                        <Avatar 
                          sx={{ 
                            width: 28, 
                            height: 28, 
                            fontSize: '0.7rem',
                            bgcolor: getPositionColor(mapElementTypeToPosition(pick.element_type))
                          }}
                        >
                          {index + 1}
                        </Avatar>
                      </ListItemAvatar>
                      
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                              Player {pick.element}
                            </Typography>
                            {pick.is_captain && <Star sx={{ fontSize: 12, color: 'gold' }} />}
                            {pick.is_vice_captain && <Person sx={{ fontSize: 12, color: 'silver' }} />}
                            <Chip 
                              label={mapElementTypeToPosition(pick.element_type)} 
                              size="small"
                              sx={{ 
                                height: 16, 
                                fontSize: '0.6rem',
                                bgcolor: getPositionColor(mapElementTypeToPosition(pick.element_type))
                              }}
                            />
                          </Box>
                        }
                        secondary={
                          <Typography variant="caption" color="textSecondary">
                            {pick.is_captain ? 'Captain' : pick.is_vice_captain ? 'Vice Captain' : `Multiplier: ${pick.multiplier}`}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
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