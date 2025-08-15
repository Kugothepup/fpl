/**
 * FPL Manager v3 - Reports Component (Placeholder)
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

const Reports = () => {
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
          Team Reports
        </Typography>
      </Box>

      <Card>
        <CardContent>
          <Alert severity="info">
            Team reports will be implemented in the next phase.
            This will include performance analysis, fixture difficulty analysis, and detailed team statistics.
          </Alert>
        </CardContent>
      </Card>
    </Container>
  );
};

export default Reports;