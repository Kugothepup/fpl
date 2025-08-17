/**
 * FPL Manager v3 - API Service
 * Handles all communication with the Flask backend
 */

import axios from 'axios';

// Configure axios defaults
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://10.2.0.2:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    
    // Handle common error scenarios
    if (error.response?.status === 404) {
      console.warn('API endpoint not found');
    } else if (error.response?.status >= 500) {
      console.error('Server error occurred');
    } else if (error.code === 'ECONNABORTED') {
      console.error('Request timeout');
    }
    
    return Promise.reject(error);
  }
);

// API service methods
export const apiService = {
  // Health and status
  async getHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  // FPL Data
  async getBootstrapData() {
    const response = await api.get('/bootstrap');
    return response.data;
  },

  async getUserTeam(teamId = null) {
    const params = teamId ? { team_id: teamId } : {};
    const response = await api.get('/team', { params });
    return response.data;
  },

  async getFixtures() {
    const response = await api.get('/fixtures');
    return response.data;
  },

  async getPlayers(filters = {}) {
    const response = await api.get('/players', { params: filters });
    return response.data;
  },

  // ML Predictions
  async predictPlayerPoints() {
    const response = await api.get('/predictions/points');
    return response.data;
  },

  async getCaptainRecommendation(teamId = null) {
    const params = teamId ? { team_id: teamId } : {};
    const response = await api.get('/predictions/captain', { params });
    return response.data;
  },

  async getTransferRecommendations(teamId = null, budget = 0, freeTransfers = 1) {
    const params = {
      ...(teamId && { team_id: teamId }),
      budget,
      free_transfers: freeTransfers,
    };
    const response = await api.get('/predictions/transfers', { params });
    return response.data;
  },

  async getGameweekScorePredictions() {
    const response = await api.get('/predictions/gameweek-scores');
    return response.data;
  },

  async getTeamScorePrediction(teamId = null) {
    const params = teamId ? { team_id: teamId } : {};
    const response = await api.get('/predictions/team-score', { params });
    return response.data;
  },

  async getTeamNextGameweeksPredictions(teamId = null, gameweeks = 2) {
    const params = {
      ...(teamId && { team_id: teamId }),
      gameweeks,
    };
    const response = await api.get('/predictions/team-next-gameweeks', { params });
    return response.data;
  },

  // Weather Data
  async getWeather(cities = null) {
    const params = cities ? { cities } : {};
    const response = await api.get('/weather', { params });
    return response.data;
  },

  // News
  async getNews(hoursBack = 24, maxArticles = 10) {
    const params = {
      hours_back: hoursBack,
      max_articles: maxArticles,
    };
    const response = await api.get('/news', { params });
    return response.data;
  },

  // Accuracy Tracking
  async getAccuracyStats(days = 30) {
    const params = { days };
    const response = await api.get('/accuracy/stats', { params });
    return response.data;
  },

  // Model Training
  async trainModels() {
    const response = await api.post('/train-models');
    return response.data;
  },

  // Wildcard optimization
  async optimizeWildcardTeam(config) {
    const response = await api.post('/wildcard/optimize', config);
    return response.data;
  },

  // Mistral AI optimization
  async optimizeWithMistral(config) {
    const response = await api.post('/wildcard/optimize-mistral', config);
    return response.data;
  },

  // Age Performance Analysis
  async getAgeAnalysis(metric = 'points', position = null, useQuick = true, playerIds = null) {
    const params = { metric };
    if (position) params.position = position;
    if (playerIds && playerIds.length > 0) {
      params.player_ids = playerIds.join(',');
    }
    
    // Use quick endpoint by default to avoid timeouts
    const endpoint = useQuick ? '/age-analysis/quick' : '/age-analysis';
    const response = await api.get(endpoint, { params });
    return response.data;
  },

  async getAgeAnalysisWithAI(metric = 'points', position = null, enableEnrichment = false, enableSummary = true) {
    const params = { 
      metric,
      ai_enrichment: enableEnrichment.toString(),
      ai_summary: enableSummary.toString()
    };
    if (position) params.position = position;
    const response = await api.get('/age-analysis', { params });
    return response.data;
  },

  async predictPerformanceByAge(age, metric = 'points', position = null, playerIds = null) {
    const params = { age, metric };
    if (position) params.position = position;
    if (playerIds && playerIds.length > 0) {
      params.player_ids = playerIds.join(',');
    }
    const response = await api.get('/age-analysis/predict', { params });
    return response.data;
  },

  async comparePlayersByAge(playerIds, metric = 'points') {
    const response = await api.post('/age-analysis/compare-players', {
      player_ids: playerIds,
      metric
    });
    return response.data;
  },

  async getAgeAnalysisMetrics() {
    const response = await api.get('/age-analysis/metrics');
    return response.data;
  },

  async enrichPlayersData(playerIds) {
    const response = await api.post('/age-analysis/enrich-players', {
      player_ids: playerIds
    });
    return response.data;
  },

  async getEnrichedPlayersData() {
    const response = await api.get('/age-analysis/enriched-data');
    return response.data;
  },

  async startBackgroundAIAnalysis(metric = 'points', position = null) {
    const data = { metric };
    if (position) data.position = position;
    const response = await api.post('/age-analysis/background', data);
    return response.data;
  },

  async getBackgroundTaskStatus(taskId) {
    const response = await api.get(`/age-analysis/status/${taskId}`);
    return response.data;
  },

  async listBackgroundTasks() {
    const response = await api.get('/age-analysis/tasks');
    return response.data;
  },

  async bulkEnrichPlayerAges(limit = 50, forceRefresh = false) {
    const data = { limit, force_refresh: forceRefresh };
    const response = await api.post('/age-analysis/bulk-enrich', data);
    return response.data;
  },

  async importPlayerDataFromCSV(season = '2024-25', url = null) {
    const data = { season };
    if (url) data.url = url;
    const response = await api.post('/age-analysis/import-csv', data);
    return response.data;
  },

  async getSeasonCorrelationAnalysis() {
    const response = await api.get('/age-analysis/season-correlation');
    return response.data;
  },

  // Utility methods
  async checkConnection() {
    try {
      await this.getHealth();
      return true;
    } catch (error) {
      return false;
    }
  },

  // Error handling helper
  handleError(error, fallbackMessage = 'An error occurred') {
    if (error.response?.data?.error) {
      return error.response.data.error;
    } else if (error.response?.data?.message) {
      return error.response.data.message;
    } else if (error.message) {
      return error.message;
    } else {
      return fallbackMessage;
    }
  },
};

// Convenience exports for common operations
export const fetchPredictions = () => apiService.predictPlayerPoints();
export const fetchCaptainRecommendation = (teamId) => apiService.getCaptainRecommendation(teamId);
export const trainModels = () => apiService.trainModels();
export const fetchWeather = (cities) => apiService.getWeather(cities);
export const fetchNews = (hoursBack, maxArticles) => apiService.getNews(hoursBack, maxArticles);
export const fetchAccuracyStats = (days) => apiService.getAccuracyStats(days);
export const fetchGameweekScores = () => apiService.getGameweekScorePredictions();
export const fetchTeamScore = (teamId) => apiService.getTeamScorePrediction(teamId);

export default apiService;