/**
 * FPL Manager v3 - Team Management Component (Placeholder)
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

const TeamManagement = () => {
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
          Team Management
        </Typography>
      </Box>

      <Card>
        <CardContent>
          <Alert severity="info">
            Team Management features will be implemented in the next phase.
            This will include team selection, formation optimization, and player analysis.
          </Alert>
        </CardContent>
      </Card>
    </Container>
  );
};

export default TeamManagement;