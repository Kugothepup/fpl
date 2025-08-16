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