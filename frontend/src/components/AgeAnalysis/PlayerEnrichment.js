/**
 * Player Data Enrichment Component
 * Allows users to manage AI-powered player data enrichment
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Alert,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Autocomplete,
  TextField,
  CircularProgress,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  CloudSync as EnrichIcon,
  Person as PlayerIcon,
  Update as UpdateIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';

// Import API service
import { apiService } from '../../services/api';

const PlayerEnrichment = () => {
  // State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [players, setPlayers] = useState([]);
  const [enrichedPlayers, setEnrichedPlayers] = useState([]);
  const [selectedPlayers, setSelectedPlayers] = useState([]);
  const [enrichmentProgress, setEnrichmentProgress] = useState(null);
  const [enrichmentResults, setEnrichmentResults] = useState(null);
  const [showResultsDialog, setShowResultsDialog] = useState(false);
  const [bulkEnrichmentTask, setBulkEnrichmentTask] = useState(null);
  const [bulkEnrichmentStatus, setBulkEnrichmentStatus] = useState(null);
  const [csvImportTask, setCsvImportTask] = useState(null);
  const [csvImportStatus, setCsvImportStatus] = useState(null);

  // Load data on component mount
  useEffect(() => {
    loadPlayers();
    loadEnrichedData();
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

  const loadEnrichedData = async () => {
    try {
      const response = await apiService.getEnrichedPlayersData();
      setEnrichedPlayers(response.data.enriched_players || []);
    } catch (err) {
      console.error('Error loading enriched data:', err);
      // This is not critical, so we don't set an error
    }
  };

  const handleEnrichPlayers = async () => {
    if ((selectedPlayers || []).length === 0) {
      setError('Please select at least one player to enrich');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setEnrichmentProgress({ current: 0, total: (selectedPlayers || []).length });

      const playerIds = (selectedPlayers || []).map(p => p.id);
      const response = await apiService.enrichPlayersData(playerIds);
      
      setEnrichmentResults(response.data);
      setShowResultsDialog(true);
      
      // Reload enriched data
      await loadEnrichedData();
      
      // Clear selection
      setSelectedPlayers([]);
      
    } catch (err) {
      console.error('Enrichment error:', err);
      setError(apiService.handleError(err, 'Failed to enrich player data'));
    } finally {
      setLoading(false);
      setEnrichmentProgress(null);
    }
  };

  const getEnrichmentStatus = (playerId) => {
    const enriched = enrichedPlayers.find(p => p.player_id === playerId);
    if (!enriched) return { status: 'none', data: null };
    
    const lastUpdated = new Date(enriched.last_updated);
    const daysSinceUpdate = (new Date() - lastUpdated) / (1000 * 60 * 60 * 24);
    
    return {
      status: daysSinceUpdate > 7 ? 'stale' : 'fresh',
      data: enriched
    };
  };

  const formatLastUpdated = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };

  const handleBulkEnrichment = async (limit = 50) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiService.bulkEnrichPlayerAges(limit, false);
      setBulkEnrichmentTask(response.task_id);
      
      // Start polling for status
      pollBulkEnrichmentTask(response.task_id);
      
    } catch (err) {
      console.error('Bulk enrichment error:', err);
      setError(apiService.handleError(err, 'Failed to start bulk enrichment'));
      setLoading(false);
    }
  };

  const pollBulkEnrichmentTask = async (taskId) => {
    try {
      const statusResponse = await apiService.getBackgroundTaskStatus(taskId);
      setBulkEnrichmentStatus(statusResponse.data);
      
      if (statusResponse.data.status === 'completed') {
        setLoading(false);
        setBulkEnrichmentTask(null);
        setBulkEnrichmentStatus(null);
        
        // Show results
        if (statusResponse.data.result) {
          setEnrichmentResults(statusResponse.data.result);
          setShowResultsDialog(true);
        }
        
        // Reload enriched data
        await loadEnrichedData();
        
      } else if (statusResponse.data.status === 'failed') {
        setLoading(false);
        setError(`Bulk enrichment failed: ${statusResponse.data.error}`);
        setBulkEnrichmentTask(null);
        setBulkEnrichmentStatus(null);
        
      } else {
        // Continue polling
        setTimeout(() => pollBulkEnrichmentTask(taskId), 3000);
      }
      
    } catch (err) {
      console.error('Bulk enrichment polling error:', err);
      setLoading(false);
      setBulkEnrichmentTask(null);
      setBulkEnrichmentStatus(null);
    }
  };

  const handleCSVImport = async (season = '2024-25') => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiService.importPlayerDataFromCSV(season);
      setCsvImportTask(response.task_id);
      
      // Start polling for status
      pollCSVImportTask(response.task_id);
      
    } catch (err) {
      console.error('CSV import error:', err);
      setError(apiService.handleError(err, 'Failed to start CSV import'));
      setLoading(false);
    }
  };

  const pollCSVImportTask = async (taskId) => {
    try {
      const statusResponse = await apiService.getBackgroundTaskStatus(taskId);
      setCsvImportStatus(statusResponse.data);
      
      if (statusResponse.data.status === 'completed') {
        setLoading(false);
        setCsvImportTask(null);
        setCsvImportStatus(null);
        
        // Show results
        if (statusResponse.data.result) {
          setEnrichmentResults(statusResponse.data.result);
          setShowResultsDialog(true);
        }
        
        // Reload enriched data
        await loadEnrichedData();
        
      } else if (statusResponse.data.status === 'failed') {
        setLoading(false);
        setError(`CSV import failed: ${statusResponse.data.error}`);
        setCsvImportTask(null);
        setCsvImportStatus(null);
        
      } else {
        // Continue polling
        setTimeout(() => pollCSVImportTask(taskId), 2000);
      }
      
    } catch (err) {
      console.error('CSV import polling error:', err);
      setLoading(false);
      setCsvImportTask(null);
      setCsvImportStatus(null);
    }
  };

  return (
    <Box>
      {/* Header */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <EnrichIcon />
            AI Player Data Enrichment
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Use Mistral AI to search for and enrich player data with ages, birth dates, nationalities, and injury status.
          </Typography>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* CSV Import */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            📊 CSV Import (Recommended)
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            Import player data from the official Fantasy Premier League GitHub repository. 
            This is faster and more reliable than AI searches, providing birth dates for most players.
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={() => handleCSVImport('2024-25')}
              disabled={loading || csvImportTask}
              startIcon={loading ? <CircularProgress size={20} /> : <EnrichIcon />}
            >
              {loading ? 'Importing...' : 'Import 2024-25 Season'}
            </Button>
            
            <Button
              variant="outlined"
              onClick={() => handleCSVImport('2023-24')}
              disabled={loading || csvImportTask}
            >
              Import 2023-24 Season
            </Button>
            
            <Chip 
              label="Official FPL Data" 
              color="success" 
              size="small"
            />
          </Box>

          {csvImportStatus && (
            <Box sx={{ mt: 2 }}>
              <Alert severity="info" sx={{ mb: 1 }}>
                CSV Import: {csvImportStatus.status} 
                {csvImportStatus.type === 'csv_import' && ' (downloading from GitHub)'}
              </Alert>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: '100%' }}>
                  <LinearProgress 
                    variant="determinate" 
                    value={csvImportStatus.progress || 0}
                  />
                </Box>
                <Typography variant="caption">
                  {csvImportStatus.progress || 0}%
                </Typography>
              </Box>
              <Typography variant="caption" color="textSecondary" display="block" sx={{ mt: 1 }}>
                Downloading and importing official Fantasy Premier League player data...
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Bulk Enrichment */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Bulk Age Data Enrichment
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            Use Mistral AI to search online and populate age data for all players automatically. 
            This will solve the "No age data available" issue by enriching player information in batches.
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={() => handleBulkEnrichment(50)}
              disabled={loading || bulkEnrichmentTask}
              startIcon={loading ? <CircularProgress size={20} /> : <EnrichIcon />}
            >
              {loading ? 'Processing...' : 'Enrich 50 Players'}
            </Button>
            
            <Button
              variant="outlined"
              onClick={() => handleBulkEnrichment(20)}
              disabled={loading || bulkEnrichmentTask}
            >
              Enrich 20 Players
            </Button>
            
            <Chip 
              label={`${(enrichedPlayers || []).length} players enriched`} 
              color="info" 
              size="small"
            />
          </Box>

          {bulkEnrichmentStatus && (
            <Box sx={{ mt: 2 }}>
              <Alert severity="info" sx={{ mb: 1 }}>
                Bulk Enrichment: {bulkEnrichmentStatus.status} 
                {bulkEnrichmentStatus.type === 'bulk_enrichment' && ' (searching online for player ages)'}
              </Alert>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: '100%' }}>
                  <LinearProgress 
                    variant="determinate" 
                    value={bulkEnrichmentStatus.progress || 0}
                  />
                </Box>
                <Typography variant="caption">
                  {bulkEnrichmentStatus.progress || 0}%
                </Typography>
              </Box>
              <Typography variant="caption" color="textSecondary" display="block" sx={{ mt: 1 }}>
                This may take several minutes as we search for each player's age data online...
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Player Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Select Players for Enrichment
          </Typography>
          
          <Autocomplete
            multiple
            options={players || []}
            getOptionLabel={(option) => `${option.name} (${option.team || 'Unknown'}) - ${option.position}`}
            value={selectedPlayers || []}
            onChange={(event, newValue) => setSelectedPlayers(newValue || [])}
            renderInput={(params) => (
              <TextField
                {...params}
                label="Search and select players"
                variant="outlined"
                fullWidth
              />
            )}
            renderOption={(props, option) => {
              const enrichmentStatus = getEnrichmentStatus(option.id);
              return (
                <Box component="li" {...props}>
                  <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="body1">{option.name}</Typography>
                      <Typography variant="caption" color="textSecondary">
                        {option.team} • {option.position} • £{option.cost}m
                      </Typography>
                    </Box>
                    <Box>
                      {enrichmentStatus.status === 'fresh' && (
                        <Chip label="Enriched" color="success" size="small" />
                      )}
                      {enrichmentStatus.status === 'stale' && (
                        <Chip label="Needs Update" color="warning" size="small" />
                      )}
                    </Box>
                  </Box>
                </Box>
              );
            }}
            renderTags={(value, getTagProps) =>
              (value || []).map((option, index) => (
                <Chip
                  key={option.id}
                  label={option.name}
                  {...getTagProps({ index })}
                  color="primary"
                  variant="outlined"
                />
              ))
            }
          />

          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="body2" color="textSecondary">
              {(selectedPlayers || []).length} player(s) selected
            </Typography>
            <Button
              variant="contained"
              onClick={handleEnrichPlayers}
              disabled={loading || (selectedPlayers || []).length === 0}
              startIcon={loading ? <CircularProgress size={20} /> : <EnrichIcon />}
            >
              {loading ? 'Enriching...' : 'Enrich Selected Players'}
            </Button>
          </Box>

          {enrichmentProgress && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" gutterBottom>
                Enriching players... ({enrichmentProgress.current}/{enrichmentProgress.total})
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={(enrichmentProgress.current / enrichmentProgress.total) * 100} 
              />
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Enriched Players Data */}
      {(enrichedPlayers || []).length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Enriched Player Database ({(enrichedPlayers || []).length} players)
            </Typography>
            
            <TableContainer component={Paper} sx={{ backgroundColor: 'rgba(255, 255, 255, 0.05)' }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Player</TableCell>
                    <TableCell>Age</TableCell>
                    <TableCell>Birth Date</TableCell>
                    <TableCell>Nationality</TableCell>
                    <TableCell>Injury Status</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Last Updated</TableCell>
                    <TableCell>Source</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {(enrichedPlayers || []).map((player) => (
                    <TableRow key={player.player_id}>
                      <TableCell>
                        <Typography variant="body2" fontWeight="bold">
                          {player.name}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {player.enriched_age ? player.enriched_age.toFixed(1) : 'N/A'}
                      </TableCell>
                      <TableCell>{player.birth_date || 'N/A'}</TableCell>
                      <TableCell>{player.nationality || 'N/A'}</TableCell>
                      <TableCell>
                        {player.injury_status ? (
                          <Chip 
                            label={player.injury_status}
                            color={player.injury_status === 'Fit' ? 'success' : 'error'}
                            size="small"
                          />
                        ) : 'N/A'}
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={`${(player.confidence * 100).toFixed(0)}%`}
                          color={player.confidence > 0.8 ? 'success' : player.confidence > 0.5 ? 'warning' : 'error'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{formatLastUpdated(player.last_updated)}</TableCell>
                      <TableCell>
                        <Typography variant="caption">
                          {player.data_source}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* Results Dialog */}
      <Dialog 
        open={showResultsDialog} 
        onClose={() => setShowResultsDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Enrichment Results</DialogTitle>
        <DialogContent>
          {enrichmentResults && (
            <Box>
              <Alert severity="info" sx={{ mb: 2 }}>
                {enrichmentResults.total_processed ? (
                  <>
                    Processed {enrichmentResults.total_processed} players • 
                    Successfully enriched {enrichmentResults.successful_enrichments || enrichmentResults.successful} players
                    {enrichmentResults.failed > 0 && ` • ${enrichmentResults.failed} failed`}
                    {enrichmentResults.skipped > 0 && ` • ${enrichmentResults.skipped} skipped`}
                  </>
                ) : (
                  enrichmentResults.message
                )}
              </Alert>
              
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Player</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Details</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {enrichmentResults.enrichment_results && enrichmentResults.enrichment_results.length > 0 ? (
                      enrichmentResults.enrichment_results.map((result) => (
                        <TableRow key={result.player_id}>
                          <TableCell>{result.player_name}</TableCell>
                          <TableCell>
                            {result.success ? (
                              <Chip icon={<SuccessIcon />} label="Success" color="success" size="small" />
                            ) : (
                              <Chip icon={<ErrorIcon />} label="Failed" color="error" size="small" />
                            )}
                          </TableCell>
                          <TableCell>
                            {result.success ? (
                              <Typography variant="caption">
                                Age: {result.enriched_age?.toFixed(1) || 'N/A'} • 
                                Nationality: {result.nationality || 'N/A'} • 
                                Status: {result.injury_status || 'N/A'}
                              </Typography>
                            ) : (
                              <Typography variant="caption" color="error">
                                {result.error}
                              </Typography>
                            )}
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={3} align="center">
                          <Typography variant="body2" color="textSecondary">
                            No detailed results available
                          </Typography>
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowResultsDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default PlayerEnrichment;