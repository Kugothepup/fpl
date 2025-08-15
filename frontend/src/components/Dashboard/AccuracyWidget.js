/**
 * FPL Manager v3 - Accuracy Widget (Placeholder)
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Alert,
} from '@mui/material';
import { Analytics } from '@mui/icons-material';

const AccuracyWidget = () => {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Analytics sx={{ mr: 1 }} />
          <Typography variant="h6" component="div">
            Prediction Accuracy
          </Typography>
        </Box>
        <Alert severity="info">
          Accuracy tracking will show how well our ML predictions perform.
          This includes historical accuracy, model performance, and learning insights.
        </Alert>
      </CardContent>
    </Card>
  );
};

export default AccuracyWidget;