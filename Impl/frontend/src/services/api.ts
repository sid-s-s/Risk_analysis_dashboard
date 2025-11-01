import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

interface TextAnalysisRequest {
  age?: number;
  gender?: string;
  symptoms_description: string;
}

interface ImageAnalysisRequest {
  image: string;
}

export const analyzeText = async (data: TextAnalysisRequest) => {
  try {
    console.log('Making API request to /analyze-text:', data);
    const response = await api.post('/analyze-text', data);
    console.log('API response:', response.data);
    return response.data;
  } catch (error: any) {
    console.error('API error:', {
      message: error.message,
      response: error.response?.data,
      status: error.response?.status
    });
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Failed to analyze text: ' + (error.message || 'Unknown error'));
  }
};

export const analyzeImage = async (data: ImageAnalysisRequest) => {
  try {
    const response = await api.post('/analyze-image', data);
    return response.data;
  } catch (error: any) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Failed to analyze image: ' + (error.message || 'Unknown error'));
  }
};