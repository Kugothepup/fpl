/**
 * FPL Manager v3 - News Widget
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Alert,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Chip,
  Link,
  Divider,
} from '@mui/material';
import { Newspaper, TrendingUp, Warning, Info } from '@mui/icons-material';
import { fetchNews } from '../../services/api';

const NewsWidget = () => {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadNews();
  }, []);

  const loadNews = async () => {
    try {
      setLoading(true);
      const response = await fetchNews(24, 5); // Last 24 hours, max 5 articles
      setNews(response.data || []);
      setError(null);
    } catch (err) {
      setError('News service unavailable - API key required');
      setNews([]);
    } finally {
      setLoading(false);
    }
  };

  const getNewsIcon = (category) => {
    switch (category?.toLowerCase()) {
      case 'injury':
        return <Warning color="warning" />;
      case 'transfer':
        return <TrendingUp color="primary" />;
      default:
        return <Info color="info" />;
    }
  };

  const getSeverityColor = (importance) => {
    if (importance >= 8) return 'error';
    if (importance >= 6) return 'warning';
    return 'info';
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Newspaper sx={{ mr: 1 }} />
          <Typography variant="h6" component="div">
            FPL News
          </Typography>
        </Box>

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
            <CircularProgress size={24} />
          </Box>
        ) : error ? (
          <Alert severity="warning" sx={{ fontSize: '0.8rem' }}>
            {error}
            <br />
            <Typography variant="caption">
              Add PERPLEXITY_API_KEY to enable news integration
            </Typography>
          </Alert>
        ) : news.length === 0 ? (
          <Alert severity="info" sx={{ fontSize: '0.8rem' }}>
            No recent FPL news found
          </Alert>
        ) : (
          <List dense sx={{ p: 0 }}>
            {news.slice(0, 4).map((article, index) => (
              <Box key={index}>
                <ListItem sx={{ px: 0, py: 1 }}>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: 0.5 }}>
                        {getNewsIcon(article.category)}
                        <Typography variant="body2" sx={{ fontWeight: 500, lineHeight: 1.3 }}>
                          {article.title || article.summary}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      <Box sx={{ mt: 0.5 }}>
                        <Box sx={{ display: 'flex', gap: 0.5, mb: 0.5, flexWrap: 'wrap' }}>
                          {article.category && (
                            <Chip
                              label={article.category}
                              size="small"
                              variant="outlined"
                              sx={{ height: 16, fontSize: '0.65rem' }}
                            />
                          )}
                          {article.importance && (
                            <Chip
                              label={`Priority ${article.importance}/10`}
                              size="small"
                              color={getSeverityColor(article.importance)}
                              sx={{ height: 16, fontSize: '0.65rem' }}
                            />
                          )}
                        </Box>
                        <Typography variant="caption" color="textSecondary">
                          {article.timestamp ? new Date(article.timestamp).toLocaleTimeString() : 'Recent'}
                          {article.players && article.players.length > 0 && (
                            <span> • Affects: {article.players.slice(0, 2).join(', ')}</span>
                          )}
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
                {index < news.length - 1 && <Divider />}
              </Box>
            ))}
          </List>
        )}

        {news.length > 4 && (
          <Box sx={{ mt: 1, textAlign: 'center' }}>
            <Link component="button" variant="caption" onClick={() => {}}>
              View all news
            </Link>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default NewsWidget;