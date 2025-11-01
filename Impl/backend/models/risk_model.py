import tensorflow as tf
import numpy as np
from typing import Tuple, List

class RiskAnalysisModel:
    def __init__(self):
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        Build and compile the risk analysis model.
        This is a simplified version for demonstration.
        In production, this would be a more sophisticated model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def predict_risk(self, processed_data: np.ndarray) -> float:
        """
        Predict risk score using the model.
        
        Args:
            processed_data: Preprocessed patient data
            
        Returns:
            Risk score between 0 and 100
        """
        # Get raw prediction
        prediction = self.model.predict(processed_data)
        
        # Convert to risk score (0-100)
        risk_score = float(prediction[0][0] * 100)
        return risk_score
    
    def generate_explanation(self, processed_data: np.ndarray) -> Tuple[str, List[str]]:
        """
        Generate explanation for the risk prediction.
        This is a simplified version - in production, this would use more sophisticated
        explainable AI techniques (e.g., LIME, SHAP).
        
        Args:
            processed_data: Preprocessed patient data
            
        Returns:
            Tuple of (explanation string, list of contributing factors)
        """
        # Get feature importance through a simple gradient-based approach
        with tf.GradientTape() as tape:
            input_tensor = tf.convert_to_tensor(processed_data, dtype=tf.float32)
            tape.watch(input_tensor)
            prediction = self.model(input_tensor)
            
        gradients = tape.gradient(prediction, input_tensor)
        feature_importance = np.abs(gradients.numpy()[0])
        
        # Map features to their names
        feature_names = [
            'age', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'heart_rate', 'cholesterol', 'blood_sugar', 'bmi'
        ]
        
        # Get top contributing factors
        contributing_factors = []
        for importance, feature in sorted(zip(feature_importance, feature_names), reverse=True):
            if importance > 0.1:  # Threshold for significance
                contributing_factors.append(f"{feature.replace('_', ' ').title()}")
        
        # Generate explanation
        if len(contributing_factors) > 0:
            explanation = f"The main factors contributing to this risk assessment are: {', '.join(contributing_factors)}"
        else:
            explanation = "No single factor shows significant contribution to the risk assessment."
            
        return explanation, contributing_factors