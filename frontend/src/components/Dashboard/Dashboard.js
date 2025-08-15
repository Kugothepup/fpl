/**
 * FPL Manager v3 - Main Dashboard Component
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  Button,
  Alert,
  CircularProgress,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Fab,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  People as TeamIcon,
  TrendingUp as PredictionsIcon,
  Assessment as ReportsIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  AutoAwesome as WildcardIcon,
  Analytics as AccuracyIcon,
  Menu as MenuIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

// Import child components
import QuickStats from './QuickStats';
import CaptainRecommendation from './CaptainRecommendation';
import TopPredictions from './TopPredictions';
import WeatherWidget from './WeatherWidget';
import NewsWidget from './NewsWidget';
import AccuracyWidget from './AccuracyWidget';
import GameweekScores from './GameweekScores';
import TeamScoreWidget from './TeamScoreWidget';
import MyTeamWidget from './MyTeamWidget';

// Import API service
import { apiService } from '../../services/api';

const Dashboard = () => {
  const navigate = useNavigate();
  
  // State
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState(null);

  // Navigation menu items
  const menuItems = [
    { label: 'Dashboard', icon: <DashboardIcon />, path: '/', active: true },
    { label: 'Team Management', icon: <TeamIcon />, path: '/team' },
    { label: 'Predictions', icon: <PredictionsIcon />, path: '/predictions' },
    { label: 'Reports', icon: <ReportsIcon />, path: '/reports' },
    { label: 'Wildcard Helper', icon: <WildcardIcon />, path: '/wildcard' },
    { label: 'Accuracy Tracking', icon: <AccuracyIcon />, path: '/accuracy' },
  ];

  // Load initial data
  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Check system health
      const healthResponse = await apiService.getHealth();
      setSystemStatus(healthResponse);

      setLoading(false);
    } catch (err) {
      console.error('Dashboard load error:', err);
      setError(apiService.handleError(err, 'Failed to load dashboard data'));
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
  };

  const handleMenuOpen = (event) => {
    setMenuAnchor(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  const handleNavigate = (path) => {
    navigate(path);
    handleMenuClose();
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'error';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
        flexDirection="column"
        gap={2}
      >
        <CircularProgress size={60} />
        <Typography variant="h6" color="textSecondary">
          Loading FPL Manager v3...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* App Bar */}
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={handleMenuOpen}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            FPL Manager v3 - Intelligence Dashboard
          </Typography>

          {/* System Status */}
          {systemStatus && (
            <Chip
              label={`System: ${systemStatus.status}`}
              color={getStatusColor(systemStatus.status)}
              variant="outlined"
              size="small"
              sx={{ mr: 2 }}
            />
          )}

          <IconButton color="inherit" onClick={handleRefresh} disabled={refreshing}>
            <RefreshIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Navigation Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
        PaperProps={{
          sx: { minWidth: 200 }
        }}
      >
        {menuItems.map((item) => (
          <MenuItem
            key={item.path}
            onClick={() => handleNavigate(item.path)}
            selected={item.active}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {item.icon}
              {item.label}
            </Box>
          </MenuItem>
        ))}
      </Menu>

      {/* Main Content */}
      <Container
        component="main"
        maxWidth="xl"
        sx={{
          mt: 10,
          mb: 4,
          flexGrow: 1,
        }}
      >
        {/* Error Alert */}
        {error && (
          <Alert
            severity="error"
            sx={{ mb: 3 }}
            action={
              <Button color="inherit" size="small" onClick={loadDashboardData}>
                Retry
              </Button>
            }
          >
            {error}
          </Alert>
        )}

        {/* System Status Banner */}
        {systemStatus && systemStatus.status !== 'healthy' && (
          <Alert
            severity={systemStatus.status === 'degraded' ? 'warning' : 'error'}
            sx={{ mb: 3 }}
          >
            System Status: {systemStatus.status}. Some features may be limited.
          </Alert>
        )}

        {/* Dashboard Content */}
        <Grid container spacing={3}>
          {/* Quick Stats */}
          <Grid item xs={12}>
            <QuickStats systemStatus={systemStatus} />
          </Grid>

          {/* My Team Details */}
          <Grid item xs={12} md={6}>
            <MyTeamWidget />
          </Grid>

          {/* Team Score Prediction */}
          <Grid item xs={12} md={6}>
            <TeamScoreWidget />
          </Grid>

          {/* Captain Recommendation */}
          <Grid item xs={12} md={6}>
            <CaptainRecommendation />
          </Grid>

          {/* Gameweek Score Predictions */}
          <Grid item xs={12} md={6}>
            <GameweekScores />
          </Grid>

          {/* Top Predictions */}
          <Grid item xs={12} md={6}>
            <TopPredictions />
          </Grid>

          {/* Weather Widget */}
          <Grid item xs={12} md={4}>
            <WeatherWidget />
          </Grid>

          {/* News Widget */}
          <Grid item xs={12} md={4}>
            <NewsWidget />
          </Grid>

          {/* Accuracy Widget */}
          <Grid item xs={12} md={4}>
            <AccuracyWidget />
          </Grid>

          {/* Quick Actions */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Quick Actions
                </Typography>
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <Button
                    variant="contained"
                    startIcon={<TeamIcon />}
                    onClick={() => navigate('/team')}
                  >
                    Manage Team
                  </Button>
                  <Button
                    variant="contained"
                    startIcon={<PredictionsIcon />}
                    onClick={() => navigate('/predictions')}
                  >
                    View Predictions
                  </Button>
                  <Button
                    variant="contained"
                    startIcon={<WildcardIcon />}
                    onClick={() => navigate('/wildcard')}
                  >
                    Wildcard Helper
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<ReportsIcon />}
                    onClick={() => navigate('/reports')}
                  >
                    Team Reports
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<AccuracyIcon />}
                    onClick={() => navigate('/accuracy')}
                  >
                    View Accuracy
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>

      {/* Floating Action Button for Wildcard */}
      <Fab
        color="primary"
        aria-label="wildcard"
        sx={{
          position: 'fixed',
          bottom: 16,
          right: 16,
        }}
        onClick={() => navigate('/wildcard')}
      >
        <WildcardIcon />
      </Fab>

      {/* Footer */}
      <Box
        component="footer"
        sx={{
          py: 2,
          px: 2,
          mt: 'auto',
          backgroundColor: 'background.paper',
          borderTop: 1,
          borderColor: 'divider',
        }}
      >
        <Container maxWidth="xl">
          <Typography variant="body2" color="text.secondary" align="center">
            FPL Manager v3 - Advanced Fantasy Premier League Management System
            {systemStatus && (
              <Box component="span" sx={{ ml: 2 }}>
                • Version {systemStatus.version} • 
                Features: Real Data, ML Predictions, Weather Analysis, News Integration
              </Box>
            )}
          </Typography>
        </Container>
      </Box>
    </Box>
  );
};

export default Dashboard;