/**
 * Season Correlation Chart Component
 * Shows scatter plot of 2024 vs 2023 season points with age analysis
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts';

// Import API service
import { apiService } from '../../services/api';

const SeasonCorrelationChart = () => {
  // State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [correlationData, setCorrelationData] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [sortBy, setSortBy] = useState('current_points');
  const [showAgeGroups, setShowAgeGroups] = useState(true);
  const [showRegression, setShowRegression] = useState(false);

  // Load correlation data on component mount
  useEffect(() => {
    loadCorrelationData();
  }, []);

  // Process chart data when correlation data changes
  useEffect(() => {
    if (correlationData) {
      processChartData();
    }
  }, [correlationData, sortBy, showAgeGroups]);

  const loadCorrelationData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiService.getSeasonCorrelationAnalysis();
      setCorrelationData(response.data);
      
    } catch (err) {
      console.error('Error loading correlation data:', err);
      setError('Failed to load season correlation data');
    } finally {
      setLoading(false);
    }
  };

  const processChartData = () => {
    if (!correlationData?.player_analysis?.all_players) return;

    const players = correlationData.player_analysis.all_players;
    
    // Sort players based on selected criteria
    let sortedPlayers = [...players];
    switch (sortBy) {
      case 'current_points':
        sortedPlayers.sort((a, b) => b.current_points - a.current_points);
        break;
      case 'previous_points':
        sortedPlayers.sort((a, b) => b.previous_points - a.previous_points);
        break;
      case 'difference':
        sortedPlayers.sort((a, b) => Math.abs(b.difference) - Math.abs(a.difference));
        break;
      case 'player_name':
        sortedPlayers.sort((a, b) => a.player_name.localeCompare(b.player_name));
        break;
      default:
        break;
    }

    // Create chart data with age groups if enabled
    const processedData = sortedPlayers.map((player, index) => {
      // Estimate age group based on typical career patterns
      let ageGroup = 'Unknown';
      const points2023 = player.previous_points;
      const points2024 = player.current_points;
      
      // Rough age estimation based on performance patterns
      if (points2023 > 150 && points2024 >= 0) {
        ageGroup = 'Prime (25-28)';
      } else if (points2023 > 100 && points2024 === 0) {
        ageGroup = 'Veteran (33+)';
      } else if (points2023 < 50 && points2024 >= 0) {
        ageGroup = 'Youth (16-20)';
      } else if (points2023 > 50 && points2023 < 150) {
        ageGroup = 'Young (21-24)';
      } else {
        ageGroup = 'Experienced (29-32)';
      }

      return {
        id: player.player_name,
        x: player.current_points, // 2024 points on X-axis
        y: player.previous_points, // 2023 points on Y-axis
        name: player.player_name,
        team: player.team,
        position: player.position,
        difference: player.difference,
        expected: player.expected_points,
        ageGroup: showAgeGroups ? ageGroup : 'All Players',
        index,
        // Color based on position
        fill: getPositionColor(player.position),
        // Size based on absolute difference from expected
        size: Math.max(30, Math.min(120, Math.abs(player.difference || 0) * 15 + 30))
      };
    });

    setChartData(processedData);
  };

  // Generate trend line data points
  const generateTrendLineData = () => {
    if (!correlationData?.regression || !chartData.length) return [];
    
    const { slope, intercept } = correlationData.regression;
    const xValues = chartData.map(d => d.x);
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);
    
    // Create trend line points
    return [
      { x: minX, y: slope * minX + intercept, name: 'Trend Start' },
      { x: maxX, y: slope * maxX + intercept, name: 'Trend End' }
    ];
  };

  const getPositionColor = (position) => {
    switch (position) {
      case 'GK': return '#8884d8';
      case 'DEF': return '#82ca9d';
      case 'MID': return '#ffc658';
      case 'FWD': return '#ff7c7c';
      default: return '#8dd1e1';
    }
  };

  const getAgeGroupColor = (ageGroup) => {
    switch (ageGroup) {
      case 'Youth (16-20)': return '#4caf50';
      case 'Young (21-24)': return '#2196f3';
      case 'Prime (25-28)': return '#ff9800';
      case 'Experienced (29-32)': return '#f44336';
      case 'Veteran (33+)': return '#9c27b0';
      default: return '#666666';
    }
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Box
          sx={{
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: 1.5,
            borderRadius: 1,
            fontSize: '0.875rem'
          }}
        >
          <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 0.5 }}>
            {data.name}
          </Typography>
          <Typography variant="body2">Team: {data.team}</Typography>
          <Typography variant="body2">Position: {data.position}</Typography>
          <Typography variant="body2">2024 Points: {data.x}</Typography>
          <Typography variant="body2">2023 Points: {data.y}</Typography>
          <Typography variant="body2">Expected: {data.expected?.toFixed(1)}</Typography>
          <Typography variant="body2">
            Difference: {data.difference > 0 ? '+' : ''}{data.difference?.toFixed(1)}
          </Typography>
          {showAgeGroups && (
            <Typography variant="body2">Age Group: {data.ageGroup}</Typography>
          )}
        </Box>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
        <Typography variant="body2" sx={{ ml: 2 }}>
          Loading correlation data...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 3 }}>
        {error}
      </Alert>
    );
  }

  if (!correlationData) {
    return (
      <Alert severity="info" sx={{ mb: 3 }}>
        No correlation data available. Please try refreshing the page.
      </Alert>
    );
  }

  // Group data by age group or position for different scatter series
  const groupedData = showAgeGroups 
    ? chartData.reduce((acc, item) => {
        const group = item.ageGroup;
        if (!acc[group]) acc[group] = [];
        acc[group].push(item);
        return acc;
      }, {})
    : chartData.reduce((acc, item) => {
        const group = item.position;
        if (!acc[group]) acc[group] = [];
        acc[group].push(item);
        return acc;
      }, {});

  return (
    <Box>
      {/* Header */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            📊 Season Performance Correlation: 2024 vs 2023
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            Scatter plot showing 2024 season points (X-axis) vs 2023 season points (Y-axis). 
            Each point represents a player, with size indicating deviation from expected performance.
          </Typography>
          
          {/* Correlation Stats */}
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
            <Chip 
              label={`Correlation: ${correlationData.correlation?.toFixed(3) || 'N/A'}`}
              color={correlationData.correlation > 0.3 ? 'success' : 'warning'}
            />
            <Chip 
              label={`R²: ${correlationData.r2_score?.toFixed(3) || 'N/A'}`}
              color="info"
            />
            <Chip 
              label={`P-value: ${correlationData.p_value?.toFixed(4) || 'N/A'}`}
              color={correlationData.is_significant ? 'success' : 'error'}
            />
            <Chip 
              label={`${correlationData.matched_players_count || 0} players`}
              color="default"
            />
          </Box>
        </CardContent>
      </Card>

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 3, alignItems: 'center', flexWrap: 'wrap' }}>
            <FormControl sx={{ minWidth: 150 }}>
              <InputLabel>Sort By</InputLabel>
              <Select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                label="Sort By"
              >
                <MenuItem value="current_points">2024 Points</MenuItem>
                <MenuItem value="previous_points">2023 Points</MenuItem>
                <MenuItem value="difference">Performance Gap</MenuItem>
                <MenuItem value="player_name">Player Name</MenuItem>
              </Select>
            </FormControl>
            
            <FormControlLabel
              control={
                <Switch
                  checked={showAgeGroups}
                  onChange={(e) => setShowAgeGroups(e.target.checked)}
                />
              }
              label="Color by Age Groups"
            />
            
            <FormControlLabel
              control={
                <Switch
                  checked={showRegression}
                  onChange={(e) => setShowRegression(e.target.checked)}
                />
              }
              label="Show Trend Line"
            />
          </Box>
        </CardContent>
      </Card>

      {/* Chart */}
      <Card>
        <CardContent>
          <Box sx={{ width: '100%', height: 600 }}>
            <ResponsiveContainer>
              <ScatterChart
                data={chartData}
                margin={{ top: 20, right: 30, bottom: 60, left: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  type="number" 
                  dataKey="x" 
                  name="2024 Points"
                  label={{ value: '2024 Season Points', position: 'bottom', offset: -5 }}
                />
                <YAxis 
                  type="number" 
                  dataKey="y" 
                  name="2023 Points"
                  label={{ value: '2023 Season Points', angle: -90, position: 'insideLeft' }}
                />
                <RechartsTooltip content={<CustomTooltip />} />
                <Legend />
                
                {/* Render scatter plots for each group */}
                {Object.entries(groupedData).map(([groupName, groupData]) => (
                  <Scatter
                    key={groupName}
                    name={groupName}
                    data={groupData}
                    fill={showAgeGroups ? getAgeGroupColor(groupName) : getPositionColor(groupName)}
                  />
                ))}
                
                {/* Trend Line using ReferenceLine */}
                {showRegression && correlationData?.regression && chartData.length > 0 && (
                  <ReferenceLine
                    segment={[
                      { 
                        x: Math.min(...chartData.map(d => d.x)), 
                        y: correlationData.regression.slope * Math.min(...chartData.map(d => d.x)) + correlationData.regression.intercept 
                      },
                      { 
                        x: Math.max(...chartData.map(d => d.x)), 
                        y: correlationData.regression.slope * Math.max(...chartData.map(d => d.x)) + correlationData.regression.intercept 
                      }
                    ]}
                    stroke="#ff4444"
                    strokeWidth={3}
                    strokeDasharray="8 8"
                    label={{
                      value: `Trend Line (r=${correlationData.correlation?.toFixed(3)})`,
                      position: "top"
                    }}
                  />
                )}
              </ScatterChart>
            </ResponsiveContainer>
          </Box>
          
          {/* Insights */}
          {correlationData.insights && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                📈 Key Insights
              </Typography>
              {correlationData.insights.map((insight, index) => (
                <Alert key={index} severity="info" sx={{ mb: 1 }}>
                  {insight}
                </Alert>
              ))}
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default SeasonCorrelationChart;