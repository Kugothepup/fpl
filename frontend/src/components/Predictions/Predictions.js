/**
 * FPL Manager v3 - Predictions Component (Placeholder)
 */

import React from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Alert,
  Button,
  Box,
} from '@mui/material';
import { ArrowBack } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const Predictions = () => {
  const navigate = useNavigate();

  return (
    <Container maxWidth="xl" sx={{ mt: 4 }}>
      <Box sx={{ mb: 3 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/')}
          sx={{ mb: 2 }}
        >
          Back to Dashboard
        </Button>
        <Typography variant="h4" component="h1" gutterBottom>
          ML Predictions
        </Typography>
      </Box>

      <Card>
        <CardContent>
          <Alert severity="info">
            Detailed predictions interface will be implemented in the next phase.
            This will include player points predictions, captain recommendations, transfer suggestions, and formation optimization.
          </Alert>
        </CardContent>
      </Card>
    </Container>
  );
};

export default Predictions;