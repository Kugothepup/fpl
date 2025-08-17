/**
 * Age Performance Analysis Component
 * Analyzes relationship between player age and performance using regression models
 * Enhanced with Mistral AI data enrichment and analysis summaries
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
  LinearProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Menu,
  MenuItem as MenuDropdownItem,
  Fab,
  TextField,
  Tabs,
  Tab,
  Switch,
  FormControlLabel,
  Divider,
  Autocomplete,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  People as TeamIcon,
  TrendingUp as PredictionsIcon,
  Assessment as ReportsIcon,
  AutoAwesome as WildcardIcon,
  Analytics as AccuracyIcon,
  TrendingUp as AgeAnalysisIcon,
  Refresh as RefreshIcon,
  Menu as MenuIcon,
  ShowChart as ChartIcon,
  CompareArrows as CompareIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Import API service
import { apiService } from '../../services/api';

// Import child components
import PlayerComparison from './PlayerComparison';
import PlayerEnrichment from './PlayerEnrichment';
import SeasonCorrelationChart from './SeasonCorrelationChart';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const AgeAnalysis = () => {
  const navigate = useNavigate();
  
  // State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [availableMetrics, setAvailableMetrics] = useState([]);
  const [selectedMetric, setSelectedMetric] = useState('points');
  const [selectedPosition, setSelectedPosition] = useState('');
  const [availablePlayers, setAvailablePlayers] = useState([]);
  const [selectedPlayers, setSelectedPlayers] = useState([]);
  const [menuAnchor, setMenuAnchor] = useState(null);
  const [predictionAge, setPredictionAge] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [useAIFeatures, setUseAIFeatures] = useState(false);
  const [loadingAI, setLoadingAI] = useState(false);
  const [backgroundTask, setBackgroundTask] = useState(null);
  const [backgroundTaskStatus, setBackgroundTaskStatus] = useState(null);

  // Navigation menu items
  const menuItems = [
    { label: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { label: 'Team Management', icon: <TeamIcon />, path: '/team' },
    { label: 'Predictions', icon: <PredictionsIcon />, path: '/predictions' },
    { label: 'Reports', icon: <ReportsIcon />, path: '/reports' },
    { label: 'Wildcard Helper', icon: <WildcardIcon />, path: '/wildcard' },
    { label: 'Accuracy Tracking', icon: <AccuracyIcon />, path: '/accuracy' },
    { label: 'Age Analysis', icon: <AgeAnalysisIcon />, path: '/age-analysis', active: true },
  ];

  // Load initial data
  useEffect(() => {
    loadMetrics();
    loadPlayers();
  }, []);

  useEffect(() => {
    if (selectedMetric) {
      performAnalysis();
    }
  }, [selectedMetric, selectedPosition, selectedPlayers]);

  const loadMetrics = async () => {
    try {
      const response = await apiService.getAgeAnalysisMetrics();
      setAvailableMetrics(response.data.metrics);
    } catch (err) {
      console.error('Error loading metrics:', err);
      setError('Failed to load available metrics');
    }
  };

  const loadPlayers = async () => {
    try {
      const response = await apiService.getPlayers({ limit: 200 });
      setAvailablePlayers(response.data || []);
    } catch (err) {
      console.error('Error loading players:', err);
      // Don't set error for players as it's not critical
    }
  };

  const performAnalysis = async (enableAI = false) => {
    try {
      setLoading(true);
      setError(null);
      
      let response;
      if (enableAI) {
        // Start background AI analysis
        setLoadingAI(true);
        const backgroundResponse = await apiService.startBackgroundAIAnalysis(selectedMetric, selectedPosition);
        setBackgroundTask(backgroundResponse.task_id);
        
        // Start polling for status
        pollBackgroundTask(backgroundResponse.task_id);
        
        // Also get quick analysis for immediate display
        const playerIds = selectedPlayers.map(p => p.id);
        response = await apiService.getAgeAnalysis(selectedMetric, selectedPosition, true, playerIds);
      } else {
        const playerIds = selectedPlayers.map(p => p.id);
        response = await apiService.getAgeAnalysis(selectedMetric, selectedPosition, true, playerIds);
      }
      
      setAnalysisData(response.data);
      
    } catch (err) {
      console.error('Analysis error:', err);
      setError(apiService.handleError(err, 'Failed to perform age analysis'));
      setLoadingAI(false);
    } finally {
      setLoading(false);
    }
  };

  const pollBackgroundTask = async (taskId) => {
    try {
      const statusResponse = await apiService.getBackgroundTaskStatus(taskId);
      setBackgroundTaskStatus(statusResponse.data);
      
      if (statusResponse.data.status === 'completed') {
        setLoadingAI(false);
        
        // Merge AI summary into existing analysis data
        if (statusResponse.data.result && statusResponse.data.result.ai_summary) {
          setAnalysisData(prev => ({
            ...prev,
            ai_summary: statusResponse.data.result.ai_summary
          }));
        }
        
        setBackgroundTask(null);
        setBackgroundTaskStatus(null);
        
      } else if (statusResponse.data.status === 'failed') {
        setLoadingAI(false);
        setError(`AI analysis failed: ${statusResponse.data.error}`);
        setBackgroundTask(null);
        setBackgroundTaskStatus(null);
        
      } else {
        // Continue polling
        setTimeout(() => pollBackgroundTask(taskId), 2000);
      }
      
    } catch (err) {
      console.error('Background task polling error:', err);
      setLoadingAI(false);
      setBackgroundTask(null);
      setBackgroundTaskStatus(null);
    }
  };

  const handlePredictAge = async () => {
    if (!predictionAge || predictionAge < 16 || predictionAge > 45) {
      setError('Please enter a valid age between 16 and 45');
      return;
    }

    try {
      const playerIds = selectedPlayers.map(p => p.id);
      const response = await apiService.predictPerformanceByAge(predictionAge, selectedMetric, selectedPosition, playerIds);
      setPredictionResult(response.data);
      
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Failed to predict performance for the specified age');
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await performAnalysis();
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

  const generateChartData = () => {
    if (!analysisData) return null;

    const { linear_model, polynomial_model, best_model } = analysisData;
    
    // Create scatter plot data points
    const actualDataPoints = linear_model.ages.map((age, i) => ({
      x: age,
      y: linear_model.actual_values[i]
    }));

    // Create regression line points (sorted by age for smooth lines)
    const ageRange = Math.max(...linear_model.ages) - Math.min(...linear_model.ages);
    const linePoints = 50; // Number of points for smooth regression lines
    const minAge = Math.min(...linear_model.ages);
    const maxAge = Math.max(...linear_model.ages);
    
    const regressionAges = Array.from({length: linePoints}, (_, i) => 
      minAge + (i / (linePoints - 1)) * ageRange
    );

    // Generate linear regression line points
    const linearLinePoints = regressionAges.map(age => {
      const prediction = linear_model.coefficients[0] * age + linear_model.intercept;
      return { x: age, y: prediction };
    });

    // Generate polynomial regression line points
    const polyLinePoints = regressionAges.map(age => {
      let prediction = polynomial_model.intercept;
      if (polynomial_model.coefficients.length >= 2) {
        prediction += polynomial_model.coefficients[1] * age;
      }
      if (polynomial_model.coefficients.length >= 3) {
        prediction += polynomial_model.coefficients[2] * Math.pow(age, 2);
      }
      return { x: age, y: prediction };
    });

    return {
      datasets: [
        {
          label: 'Player Data Points',
          data: actualDataPoints,
          backgroundColor: 'rgba(0, 255, 135, 0.7)',
          borderColor: 'rgb(0, 255, 135)',
          borderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
          showLine: false
        },
        {
          label: `Linear Regression${best_model === 'linear' ? ' (Best)' : ''}`,
          data: linearLinePoints,
          borderColor: 'rgb(255, 152, 0)',
          backgroundColor: 'rgba(255, 152, 0, 0.1)',
          borderWidth: best_model === 'linear' ? 3 : 2,
          pointRadius: 0,
          showLine: true,
          tension: 0,
          borderDash: best_model === 'linear' ? [] : [5, 5]
        },
        {
          label: `Polynomial Regression${best_model === 'polynomial' ? ' (Best)' : ''}`,
          data: polyLinePoints,
          borderColor: 'rgb(156, 39, 176)',
          backgroundColor: 'rgba(156, 39, 176, 0.1)',
          borderWidth: best_model === 'polynomial' ? 3 : 2,
          pointRadius: 0,
          showLine: true,
          tension: 0.1,
          borderDash: best_model === 'polynomial' ? [] : [5, 5]
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: { color: '#ffffff' }
      },
      title: {
        display: true,
        text: `Age vs ${selectedMetric.toUpperCase()} Scatter Analysis`,
        color: '#ffffff',
        font: { size: 16 }
      },
      tooltip: {
        mode: 'point',
        intersect: false,
        callbacks: {
          label: function(context) {
            const datasetLabel = context.dataset.label || '';
            if (datasetLabel.includes('Data Points')) {
              return `Player: Age ${context.parsed.x.toFixed(1)}, ${selectedMetric}: ${context.parsed.y.toFixed(1)}`;
            }
            return `${datasetLabel}: Age ${context.parsed.x.toFixed(1)}, Predicted: ${context.parsed.y.toFixed(1)}`;
          }
        }
      }
    },
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        display: true,
        title: {
          display: true,
          text: 'Age (years)',
          color: '#ffffff'
        },
        ticks: { color: '#ffffff' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1),
          color: '#ffffff'
        },
        ticks: { color: '#ffffff' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      }
    },
    interaction: {
      mode: 'point',
      intersect: false
    },
    elements: {
      point: {
        hoverRadius: 8
      }
    }
  };

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
          
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
            <AgeAnalysisIcon />
            Age Performance Analysis
          </Typography>

          <IconButton
            color="inherit"
            onClick={handleRefresh}
            disabled={loading || refreshing}
          >
            <RefreshIcon className={refreshing ? 'spinning' : ''} />
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Navigation Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        transformOrigin={{ vertical: 'top', horizontal: 'left' }}
      >
        {menuItems.map((item) => (
          <MenuDropdownItem
            key={item.path}
            onClick={() => handleNavigate(item.path)}
            selected={item.active}
            sx={{ gap: 2 }}
          >
            {item.icon}
            {item.label}
          </MenuDropdownItem>
        ))}
      </Menu>

      {/* Main Content */}
      <Container maxWidth="xl" sx={{ mt: 10, mb: 4, flexGrow: 1 }}>
        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Tab Navigation */}
        <Card sx={{ mb: 3 }}>
          <Tabs 
            value={activeTab} 
            onChange={(e, newValue) => setActiveTab(newValue)}
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab label="Age Analysis" />
            <Tab label="Player Comparison" />
            <Tab label="Season Correlation" />
            <Tab label="Data Enrichment" />
          </Tabs>
        </Card>

        {/* Tab Content */}
        {activeTab === 0 && (
          <React.Fragment>
            {/* Controls */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Analysis Controls
                </Typography>
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} sm={6} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Performance Metric</InputLabel>
                  <Select
                    value={selectedMetric}
                    label="Performance Metric"
                    onChange={(e) => setSelectedMetric(e.target.value)}
                  >
                    {availableMetrics.map((metric) => (
                      <MenuItem key={metric.key} value={metric.key}>
                        {metric.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Position Filter (Optional)</InputLabel>
                  <Select
                    value={selectedPosition}
                    label="Position Filter (Optional)"
                    onChange={(e) => setSelectedPosition(e.target.value)}
                  >
                    <MenuItem value="">All Positions</MenuItem>
                    <MenuItem value="GK">Goalkeepers</MenuItem>
                    <MenuItem value="DEF">Defenders</MenuItem>
                    <MenuItem value="MID">Midfielders</MenuItem>
                    <MenuItem value="FWD">Forwards</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Autocomplete
                  multiple
                  options={availablePlayers}
                  getOptionLabel={(option) => `${option.name} (${option.team || 'Unknown'}) - £${option.cost}m`}
                  value={selectedPlayers}
                  onChange={(event, newValue) => setSelectedPlayers(newValue)}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      label="Player Filter (Optional)"
                      variant="outlined"
                      fullWidth
                    />
                  )}
                  renderTags={(value, getTagProps) =>
                    value.slice(0, 2).map((option, index) => (
                      <Chip
                        key={option.id}
                        label={option.name}
                        {...getTagProps({ index })}
                        color="primary"
                        variant="outlined"
                        size="small"
                      />
                    )).concat(
                      value.length > 2 ? [
                        <Chip
                          key="more"
                          label={`+${value.length - 2} more`}
                          color="primary"
                          variant="outlined"
                          size="small"
                        />
                      ] : []
                    )
                  }
                  limitTags={2}
                  size="small"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Button
                  variant="contained"
                  onClick={() => performAnalysis(useAIFeatures)}
                  disabled={loading || loadingAI}
                  startIcon={loading || loadingAI ? <CircularProgress size={20} /> : <ChartIcon />}
                  fullWidth
                >
                  {loadingAI ? 'AI Processing...' : loading ? 'Analyzing...' : 'Run Analysis'}
                </Button>
              </Grid>
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={useAIFeatures}
                        onChange={(e) => setUseAIFeatures(e.target.checked)}
                        disabled={loading || loadingAI}
                      />
                    }
                    label="Enable AI Features (slower)"
                  />
                  {analysisData?.ai_features_available === false && (
                    <Chip 
                      label="AI Features Unavailable" 
                      color="warning" 
                      size="small"
                    />
                  )}
                </Box>
                <Typography variant="caption" color="textSecondary" display="block">
                  AI features include online player data enrichment and intelligent analysis summaries, but may take longer to process.
                </Typography>
                
                {loadingAI && backgroundTaskStatus && (
                  <Box sx={{ mt: 2 }}>
                    <Alert severity="info" sx={{ mb: 1 }}>
                      AI Analysis in Progress: {backgroundTaskStatus.status}
                    </Alert>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box sx={{ width: '100%' }}>
                        <LinearProgress 
                          variant="determinate" 
                          value={backgroundTaskStatus.progress || 0}
                        />
                      </Box>
                      <Typography variant="caption">
                        {backgroundTaskStatus.progress || 0}%
                      </Typography>
                    </Box>
                  </Box>
                )}
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {analysisData && (
          <React.Fragment>
            {/* Chart Visualization */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Age vs Performance Regression Analysis
                </Typography>
                <Box sx={{ height: 400, mt: 2 }}>
                  <Scatter data={generateChartData()} options={chartOptions} />
                </Box>
              </CardContent>
            </Card>

            {/* Analysis Summary */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Model Performance
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">Linear R²</Typography>
                        <Typography variant="h6">
                          {analysisData.linear_model.r2_score.toFixed(4)}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">Polynomial R²</Typography>
                        <Typography variant="h6">
                          {analysisData.polynomial_model.r2_score.toFixed(4)}
                        </Typography>
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="body2" color="textSecondary">Best Model</Typography>
                        <Chip 
                          label={analysisData.best_model.charAt(0).toUpperCase() + analysisData.best_model.slice(1)}
                          color="primary"
                          variant="outlined"
                        />
                      </Grid>
                      {analysisData.peak_age && (
                        <Grid item xs={12}>
                          <Typography variant="body2" color="textSecondary">Peak Age</Typography>
                          <Typography variant="h6">
                            {analysisData.peak_age} years
                          </Typography>
                        </Grid>
                      )}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Age Prediction Tool
                    </Typography>
                    <Grid container spacing={2} alignItems="center">
                      <Grid item xs={8}>
                        <TextField
                          fullWidth
                          label="Enter Age"
                          type="number"
                          value={predictionAge}
                          onChange={(e) => setPredictionAge(e.target.value)}
                          inputProps={{ min: 16, max: 45, step: 0.1 }}
                        />
                      </Grid>
                      <Grid item xs={4}>
                        <Button
                          variant="contained"
                          onClick={handlePredictAge}
                          disabled={!predictionAge}
                          fullWidth
                        >
                          Predict
                        </Button>
                      </Grid>
                      {predictionResult && (
                        <Grid item xs={12}>
                          <Alert severity="info">
                            Predicted {selectedMetric}: <strong>{predictionResult.prediction.predicted_value}</strong>
                            <br />
                            Confidence: {(predictionResult.prediction.confidence * 100).toFixed(1)}%
                          </Alert>
                        </Grid>
                      )}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              {/* AI Summary */}
              {analysisData.ai_summary && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <AgeAnalysisIcon />
                        AI Analysis Summary
                        <Chip 
                          label={analysisData.ai_summary.confidence_level} 
                          color="primary" 
                          size="small"
                        />
                      </Typography>
                      
                      <Typography variant="body1" paragraph sx={{ fontStyle: 'italic', mb: 3 }}>
                        {analysisData.ai_summary.overall_summary}
                      </Typography>

                      <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                          <Typography variant="h6" gutterBottom>Key Findings</Typography>
                          {analysisData.ai_summary.key_findings.map((finding, index) => (
                            <Alert key={index} severity="success" sx={{ mb: 1 }}>
                              {finding}
                            </Alert>
                          ))}
                        </Grid>

                        <Grid item xs={12} md={6}>
                          <Typography variant="h6" gutterBottom>Age Insights</Typography>
                          {analysisData.ai_summary.age_insights.map((insight, index) => (
                            <Alert key={index} severity="info" sx={{ mb: 1 }}>
                              {insight}
                            </Alert>
                          ))}
                        </Grid>

                        <Grid item xs={12}>
                          <Typography variant="h6" gutterBottom>Performance Trends</Typography>
                          <Alert severity="warning" sx={{ mb: 2 }}>
                            {analysisData.ai_summary.performance_trends}
                          </Alert>
                        </Grid>

                        <Grid item xs={12} md={6}>
                          <Typography variant="h6" gutterBottom>Recommendations</Typography>
                          {analysisData.ai_summary.recommendations.map((rec, index) => (
                            <Alert key={index} severity="success" variant="outlined" sx={{ mb: 1 }}>
                              {rec}
                            </Alert>
                          ))}
                        </Grid>

                        <Grid item xs={12} md={6}>
                          <Typography variant="h6" gutterBottom>Statistical Interpretation</Typography>
                          <Alert severity="info" variant="outlined">
                            {analysisData.ai_summary.statistical_interpretation}
                          </Alert>
                        </Grid>
                      </Grid>
                      
                      <Typography variant="caption" color="textSecondary" display="block" sx={{ mt: 2 }}>
                        Generated by Mistral AI on {new Date(analysisData.ai_summary.generated_at).toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {/* Traditional Insights */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Statistical Insights
                    </Typography>
                    {analysisData.insights.map((insight, index) => (
                      <Alert key={index} severity="info" sx={{ mb: 1 }}>
                        {insight}
                      </Alert>
                    ))}
                    
                    {analysisData.enriched_data_count > 0 && (
                      <Alert severity="success" sx={{ mt: 2 }}>
                        🤖 AI Enhanced: {analysisData.enriched_data_count} players enriched with additional data from online sources
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Age Range Analysis */}
              {Object.keys(analysisData.age_range_analysis).length > 0 && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Performance by Age Group
                      </Typography>
                      <TableContainer component={Paper} sx={{ backgroundColor: 'rgba(255, 255, 255, 0.05)' }}>
                        <Table>
                          <TableHead>
                            <TableRow>
                              <TableCell>Age Group</TableCell>
                              <TableCell align="right">Players</TableCell>
                              <TableCell align="right">Average</TableCell>
                              <TableCell align="right">Median</TableCell>
                              <TableCell align="right">Range</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Object.entries(analysisData.age_range_analysis).map(([group, data]) => (
                              <TableRow key={group}>
                                <TableCell>{group}</TableCell>
                                <TableCell align="right">{data.count}</TableCell>
                                <TableCell align="right">{data.avg_value}</TableCell>
                                <TableCell align="right">{data.median_value}</TableCell>
                                <TableCell align="right">{data.min_value} - {data.max_value}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {/* Age Groups Analysis (by Year) */}
              {analysisData.age_groups_analysis && Object.keys(analysisData.age_groups_analysis.age_groups || {}).length > 0 && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Performance by Age (Season Start)
                      </Typography>
                      <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                        Showing players grouped by their age at the start of the {analysisData.age_groups_analysis.season} season
                      </Typography>
                      <TableContainer component={Paper} sx={{ backgroundColor: 'rgba(255, 255, 255, 0.05)' }}>
                        <Table>
                          <TableHead>
                            <TableRow>
                              <TableCell>Age</TableCell>
                              <TableCell align="right">Players</TableCell>
                              <TableCell align="right">Average</TableCell>
                              <TableCell align="right">Median</TableCell>
                              <TableCell align="right">Range</TableCell>
                              <TableCell>Top Performer</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Object.entries(analysisData.age_groups_analysis.age_groups).map(([ageGroup, data]) => (
                              <TableRow key={ageGroup}>
                                <TableCell>
                                  <Chip 
                                    label={ageGroup} 
                                    size="small" 
                                    color="primary" 
                                    variant="outlined"
                                  />
                                </TableCell>
                                <TableCell align="right">{data.count}</TableCell>
                                <TableCell align="right">
                                  <Typography variant="body2" fontWeight="bold">
                                    {data.avg_value}
                                  </Typography>
                                </TableCell>
                                <TableCell align="right">{data.median_value}</TableCell>
                                <TableCell align="right">
                                  <Typography variant="caption" color="textSecondary">
                                    {data.min_value} - {data.max_value}
                                  </Typography>
                                </TableCell>
                                <TableCell>
                                  {data.top_players && data.top_players[0] && (
                                    <Box>
                                      <Typography variant="body2" fontWeight="bold">
                                        {data.top_players[0][0]}
                                      </Typography>
                                      <Typography variant="caption" color="textSecondary">
                                        {data.top_players[0][4]} • {data.top_players[0][2]} pts
                                      </Typography>
                                    </Box>
                                  )}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                      
                      <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                        <Typography variant="caption" color="textSecondary">
                          Total Players: {analysisData.age_groups_analysis.total_players}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          Ages Analyzed: {analysisData.age_groups_analysis.ages_analyzed}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          Age Range: {analysisData.age_groups_analysis.age_range?.min || 0} - {analysisData.age_groups_analysis.age_range?.max || 0} years
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              )}

            </Grid>
          </React.Fragment>
        )}
          </React.Fragment>
        )}

        {/* Player Comparison Tab */}
        {activeTab === 1 && (
          <PlayerComparison selectedMetric={selectedMetric} />
        )}

        {/* Season Correlation Tab */}
        {activeTab === 2 && (
          <SeasonCorrelationChart />
        )}

        {/* Data Enrichment Tab */}
        {activeTab === 3 && (
          <PlayerEnrichment />
        )}
      </Container>

      {/* Refresh FAB */}
      <Fab
        color="primary"
        aria-label="refresh"
        onClick={handleRefresh}
        disabled={loading || refreshing}
        sx={{
          position: 'fixed',
          bottom: 16,
          right: 16,
        }}
      >
        <RefreshIcon className={refreshing ? 'spinning' : ''} />
      </Fab>
    </Box>
  );
};

export default AgeAnalysis;