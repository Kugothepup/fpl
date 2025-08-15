/**
 * FPL Manager v3 - News Widget (Placeholder)
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Alert,
} from '@mui/material';
import { Newspaper } from '@mui/icons-material';

const NewsWidget = () => {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Newspaper sx={{ mr: 1 }} />
          <Typography variant="h6" component="div">
            FPL News
          </Typography>
        </Box>
        <Alert severity="info">
          News integration with Perplexity AI will be implemented in the next phase.
          This will show injury updates, transfer news, and team selection insights.
        </Alert>
      </CardContent>
    </Card>
  );
};

export default NewsWidget;