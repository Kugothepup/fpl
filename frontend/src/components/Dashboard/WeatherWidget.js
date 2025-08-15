/**
 * FPL Manager v3 - Weather Widget (Placeholder)
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Alert,
} from '@mui/material';
import { Cloud } from '@mui/icons-material';

const WeatherWidget = () => {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Cloud sx={{ mr: 1 }} />
          <Typography variant="h6" component="div">
            Weather Impact
          </Typography>
        </Box>
        <Alert severity="info">
          Weather integration will be implemented in the next phase.
          This will show how weather conditions affect player performance.
        </Alert>
      </CardContent>
    </Card>
  );
};

export default WeatherWidget;