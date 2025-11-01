import React, { useState, useRef } from 'react';
import {
  Box,
  Button,
  Container,
  Paper,
  TextField,
  Typography,
  CircularProgress,
  Alert,
  IconButton,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import { PhotoCamera, Delete } from '@mui/icons-material';
import { analyzeText, analyzeImage } from '../services/api';

// Use MuiGrid directly without custom styling

interface Symptom {
  symptom: string;
  category?: string;
  confidence?: number;
  context?: string;
}

interface Condition {
  condition: string;
  score: number;
}

interface Analysis {
  risk_score: number;
  detected_symptoms: Symptom[];
  severity: string;
  duration: string;
  recommendations: string[];
  emergency_level: string;
  confidence: number;
  conditions: Condition[];
  risk_level: string;
  contributing_factors: string[];
  warning?: string;
}

interface ImageAnalysis {
  detected_conditions: string[];
  confidence_scores: number[];
  visual_markers: string[];
  model_type: string;
  warning?: string;
}

const RiskDashboard: React.FC = () => {
  const [textInput, setTextInput] = useState('');
  const [age, setAge] = useState('');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [imageAnalysis, setImageAnalysis] = useState<ImageAnalysis | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

    const handleImageUpload = async () => {
    if (!selectedImage) {
      setError('Please select an image to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await analyzeImage({
        image: selectedImage
      });
      setImageAnalysis(data);
    } catch (err) {
      console.error('Image analysis error:', err);
      setError(err instanceof Error ? err.message : 'Failed to analyze image');
      setImageAnalysis(null);
    } finally {
      setLoading(false);
    }
  };

  const handleTextAnalysis = async () => {
    if (!textInput) {
      setError('Please enter symptoms description');
      return;
    }

    setLoading(true);
    setError(null);
    setAnalysis(null); // Clear previous analysis

    console.log('Submitting text analysis:', {
      age: age ? parseInt(age) : undefined,
      symptoms_description: textInput,
    });

    try {
      const response = await fetch('http://localhost:8000/analyze-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          age: age ? parseInt(age) : undefined,
          symptoms_description: textInput,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Analysis response:', data);
      setAnalysis(data);
    } catch (err) {
      console.error('Text analysis error:', err);
      setError(err instanceof Error ? err.message : 'Failed to analyze symptoms');
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setImageAnalysis(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <Box>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h4" component="h1" gutterBottom>
              Patient Risk Analysis Dashboard
            </Typography>
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}
            {(analysis?.warning || imageAnalysis?.warning) && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                {analysis?.warning || imageAnalysis?.warning}
              </Alert>
            )}
          </Paper>
        </Box>

        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
          {/* Text Input Section */}
          <Box sx={{ flex: 1 }}>
            <Paper sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Patient Information
              </Typography>
              <TextField
                fullWidth
                label="Age"
                type="number"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Describe your symptoms and medical history"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                sx={{ mb: 2 }}
              />
              <Button
                variant="contained"
                onClick={handleTextAnalysis}
                disabled={loading || !textInput}
                fullWidth
              >
                {loading ? <CircularProgress size={24} /> : 'Analyze Symptoms'}
              </Button>
            </Paper>
          </Box>

          {/* Image Upload Section */}
          <Box sx={{ flex: 1 }}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Image Analysis
            </Typography>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              style={{ display: 'none' }}
              ref={fileInputRef}
            />
            <Box
              sx={{
                border: '2px dashed #ccc',
                borderRadius: 1,
                p: 2,
                mb: 2,
                textAlign: 'center',
                position: 'relative',
              }}
            >
              {selectedImage ? (
                <>
                  <img
                    src={selectedImage}
                    alt="Selected"
                    style={{ maxWidth: '100%', maxHeight: '200px' }}
                  />
                  <IconButton
                    sx={{ position: 'absolute', top: 8, right: 8 }}
                    onClick={handleClearImage}
                  >
                    <Delete />
                  </IconButton>
                </>
              ) : (
                <Button
                  startIcon={<PhotoCamera />}
                  onClick={() => fileInputRef.current?.click()}
                >
                  Upload Image
                </Button>
              )}
            </Box>
            <Button
              variant="contained"
              onClick={handleImageUpload}
              disabled={loading || !selectedImage}
              fullWidth
            >
              {loading ? <CircularProgress size={24} /> : 'Analyze Image'}
            </Button>
          </Paper>
          </Box>
        </Box>

        {/* Analysis Results */}
        {(analysis || imageAnalysis) && (
          <Box>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Analysis Results
              </Typography>
              
              {analysis && (
                <Card sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Risk Analysis Results
                    </Typography>
                    
                    {analysis.warning && (
                      <Alert severity="warning" sx={{ mb: 2 }}>
                        {analysis.warning}
                      </Alert>
                    )}
                    
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="h6" color="error">
                        Risk Level: {analysis.risk_level.toUpperCase()}
                      </Typography>
                      <Typography variant="subtitle1">
                        Risk Score: {(analysis.risk_score * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="subtitle2">
                        Confidence: {(analysis.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    
                    <Typography variant="subtitle1" gutterBottom>
                      Detected Symptoms:
                    </Typography>
                    <List dense>
                      {analysis.detected_symptoms.map((symptom, index) => (
                        <ListItem key={index}>
                          <ListItemText 
                            primary={symptom.symptom}
                            secondary={symptom.context || symptom.category}
                          />
                        </ListItem>
                      ))}
                    </List>
                    
                    {analysis.conditions.length > 0 && (
                      <>
                        <Typography variant="subtitle1" gutterBottom>
                          Possible Conditions:
                        </Typography>
                        <List dense>
                          {analysis.conditions.map((condition, index) => (
                            <ListItem key={index}>
                              <ListItemText
                                primary={condition.condition}
                                secondary={`Confidence: ${(condition.score * 100).toFixed(1)}%`}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </>
                    )}
                    
                    <Typography variant="subtitle1" gutterBottom>
                      Contributing Factors:
                    </Typography>
                    <List dense>
                      {analysis.contributing_factors.map((factor, index) => (
                        <ListItem key={index}>
                          <ListItemText primary={factor} />
                        </ListItem>
                      ))}
                    </List>
                    
                    <Typography variant="subtitle1" gutterBottom>
                      Recommendations:
                    </Typography>
                    <List dense>
                      {analysis.recommendations.map((rec, index) => (
                        <ListItem key={index}>
                          <ListItemText primary={rec} />
                        </ListItem>
                      ))}
                    </List>
                    
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Duration: {analysis.duration}
                      </Typography>
                      <Typography variant="subtitle2" color="text.secondary">
                        Emergency Level: {analysis.emergency_level}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              )}

              {imageAnalysis && (
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Image Analysis Results
                    </Typography>
                    <List dense>
                      {imageAnalysis.detected_conditions.map((condition, index) => (
                        <ListItem key={index}>
                          <ListItemText
                            primary={condition}
                            secondary={`Confidence: ${(
                              imageAnalysis.confidence_scores[index] * 100
                            ).toFixed(1)}%`}
                          />
                        </ListItem>
                      ))}
                    </List>
                    <Typography variant="subtitle1" gutterBottom>
                      Visual Markers:
                    </Typography>
                    <List dense>
                      {imageAnalysis.visual_markers.map((marker, index) => (
                        <ListItem key={index}>
                          <ListItemText primary={marker} />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              )}
            </Paper>
          </Box>
        )}
      </Box>
    </Container>
  );
};

export default RiskDashboard;