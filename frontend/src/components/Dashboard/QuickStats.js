/**
 * FPL Manager v3 - Quick Stats Widget
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  Chip,
  LinearProgress,
  Skeleton,
} from '@mui/material';
import {
  TrendingUp,
  Person,
  Stadium,
  Psychology,
} from '@mui/icons-material';

import { apiService } from '../../services/api';

const QuickStats = ({ systemStatus }) => {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load multiple data sources
      const [bootstrapResponse, teamResponse] = await Promise.allSettled([
        apiService.getBootstrapData(),
        apiService.getUserTeam(),
      ]);

      let computedStats = {
        totalPlayers: 0,
        activeGameweek: 0,
        seasonStarted: false,
        teamValue: 0,
        teamPoints: 0,
        freeTransfers: 1,
        bank: 0,
      };

      // Process bootstrap data
      if (bootstrapResponse.status === 'fulfilled' && bootstrapResponse.value.success) {
        const data = bootstrapResponse.value.data;
        computedStats.totalPlayers = data.elements?.length || 0;
        
        if (data.events) {
          const currentEvent = data.events.find(event => event.is_current) || 
                              data.events.find(event => event.is_next);
          computedStats.activeGameweek = currentEvent?.id || 1;
        }

        computedStats.seasonStarted = bootstrapResponse.value.season_status?.season_started || false;
      }

      // Process team data
      if (teamResponse.status === 'fulfilled' && teamResponse.value.success) {
        const teamData = teamResponse.value.data;
        computedStats.teamValue = teamData.entry?.value ? teamData.entry.value / 10 : 100.0;
        computedStats.teamPoints = teamData.entry?.summary_overall_points || 0;
        computedStats.freeTransfers = teamData.entry?.extra_free_transfers || 1;
        computedStats.bank = teamData.entry?.bank ? teamData.entry.bank / 10 : 0;
      }

      setStats(computedStats);
      setLoading(false);
    } catch (err) {
      console.error('Quick stats load error:', err);
      setError(err.message);
      setLoading(false);
    }
  };

  const formatValue = (value, prefix = '', suffix = '') => {
    if (value === null || value === undefined) return 'N/A';
    return `${prefix}${typeof value === 'number' ? value.toLocaleString() : value}${suffix}`;
  };

  const StatCard = ({ title, value, icon, color = 'primary', loading: itemLoading = false }) => (
    <Card sx={{ height: '100%', background: `linear-gradient(135deg, ${color === 'primary' ? '#1976d2' : color === 'success' ? '#388e3c' : color === 'warning' ? '#f57c00' : '#d32f2f'}22, transparent)` }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          {icon}
          <Typography variant="subtitle2" sx={{ ml: 1, color: 'text.secondary' }}>
            {title}
          </Typography>
        </Box>
        {itemLoading ? (
          <Skeleton variant="text" height={40} />
        ) : (
          <Typography variant="h4" component="div" fontWeight="bold">
            {value}
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  if (error) {
    return (
      <Card>
        <CardContent>
          <Typography color="error">
            Failed to load stats: {error}
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h6" component="div">
            Quick Stats
          </Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              label={`GW ${stats?.activeGameweek || '?'}`}
              color="primary"
              variant="outlined"
              size="small"
            />
            <Chip
              label={stats?.seasonStarted ? 'Season Active' : 'Pre-Season'}
              color={stats?.seasonStarted ? 'success' : 'warning'}
              variant="outlined"
              size="small"
            />
            {systemStatus?.features?.real_data_only && (
              <Chip
                label="Real Data"
                color="success"
                variant="outlined"
                size="small"
              />
            )}
          </Box>
        </Box>

        <Grid container spacing={2}>
          <Grid item xs={6} sm={3}>
            <StatCard
              title="Team Value"
              value={formatValue(stats?.teamValue, '£', 'M')}
              icon={<TrendingUp color="primary" />}
              color="primary"
              loading={loading}
            />
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <StatCard
              title="Total Points"
              value={formatValue(stats?.teamPoints)}
              icon={<Person color="success" />}
              color="success"
              loading={loading}
            />
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <StatCard
              title="Free Transfers"
              value={formatValue(stats?.freeTransfers)}
              icon={<Stadium color="warning" />}
              color="warning"
              loading={loading}
            />
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <StatCard
              title="Bank"
              value={formatValue(stats?.bank, '£', 'M')}
              icon={<Psychology color="info" />}
              color="info"
              loading={loading}
            />
          </Grid>
        </Grid>

        {/* System Status Indicators */}
        {systemStatus && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle2" gutterBottom color="text.secondary">
              System Status
            </Typography>
            <Grid container spacing={1}>
              {Object.entries(systemStatus.services || {}).map(([service, status]) => (
                <Grid item key={service}>
                  <Chip
                    label={service.replace('_', ' ')}
                    color={status ? 'success' : 'error'}
                    variant="outlined"
                    size="small"
                  />
                </Grid>
              ))}
            </Grid>
            
            {systemStatus.features && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom color="text.secondary">
                  Active Features
                </Typography>
                <Grid container spacing={1}>
                  {Object.entries(systemStatus.features)
                    .filter(([_, enabled]) => enabled)
                    .map(([feature, _]) => (
                      <Grid item key={feature}>
                        <Chip
                          label={feature.replace(/_/g, ' ')}
                          color="primary"
                          variant="outlined"
                          size="small"
                        />
                      </Grid>
                    ))}
                </Grid>
              </Box>
            )}
          </Box>
        )}

        {/* Loading Progress */}
        {loading && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
              Loading real FPL data...
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default QuickStats;