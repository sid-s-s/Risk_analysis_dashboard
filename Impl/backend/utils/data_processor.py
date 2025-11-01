import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess_patient_data(self, data: Dict[str, Union[float, str, int]]) -> np.ndarray:
        """
        Preprocess patient data for risk analysis.
        
        Args:
            data: Dictionary containing patient health data
                Expected fields:
                - age: int
                - blood_pressure_systolic: int
                - blood_pressure_diastolic: int
                - heart_rate: int
                - cholesterol: float
                - blood_sugar: float
                - bmi: float
                
        Returns:
            Preprocessed data as numpy array
        """
        required_fields = [
            'age', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'heart_rate', 'cholesterol', 'blood_sugar', 'bmi'
        ]
        
        # Validate input data
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Extract numerical features
        features = np.array([[
            data['age'],
            data['blood_pressure_systolic'],
            data['blood_pressure_diastolic'],
            data['heart_rate'],
            data['cholesterol'],
            data['blood_sugar'],
            data['bmi']
        ]])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features
    
    def calculate_risk_score(self, processed_data: np.ndarray) -> float:
        """
        Calculate risk score based on processed patient data.
        This is a simplified version - in production, this would use a trained model.
        
        Args:
            processed_data: Preprocessed patient data
            
        Returns:
            Risk score between 0 and 100
        """
        # Simple weighted sum for demonstration
        # In production, this would use a trained model
        weights = np.array([0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.1])
        base_score = np.dot(processed_data, weights)
        
        # Normalize to 0-100 range
        normalized_score = (base_score + 2) * 25  # Assuming scaled data is roughly -2 to 2
        return float(np.clip(normalized_score, 0, 100))