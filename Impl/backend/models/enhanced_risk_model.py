import tensorflow as tf
import numpy as np
from typing import Tuple, List, Dict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class RiskAnalysisModel:
    def __init__(self):
        self.deep_model = self._build_deep_model()
        self.gradient_booster = self._initialize_gradient_booster()
        self.scaler = StandardScaler()
        
    def _build_deep_model(self) -> tf.keras.Model:
        """
        Build a more sophisticated deep learning model for risk analysis.
        Uses a combination of dense layers with skip connections.
        """
        # Input layer
        inputs = tf.keras.Input(shape=(20,))  # Increased input features
        
        # First block
        x1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.Dropout(0.3)(x1)
        
        # Second block with skip connection
        x2 = tf.keras.layers.Dense(128, activation='relu')(x1)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        x2 = tf.keras.layers.Dropout(0.3)(x2)
        x2 = tf.keras.layers.Add()([x1, x2])
        
        # Third block
        x3 = tf.keras.layers.Dense(64, activation='relu')(x2)
        x3 = tf.keras.layers.BatchNormalization()(x3)
        x3 = tf.keras.layers.Dropout(0.2)(x3)
        
        # Output block
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x3)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with advanced optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )
        
        return model
        
    def _initialize_gradient_booster(self):
        """
        Initialize a gradient boosting model for ensemble predictions
        """
        try:
            if os.path.exists('risk_booster.joblib'):
                return joblib.load('risk_booster.joblib')
        except:
            pass
            
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
    def process_features(self, patient_data: Dict) -> np.ndarray:
        """
        Process and normalize patient data into model features
        """
        features = []
        
        # Basic demographic features
        features.extend([
            patient_data.get('age', 0) / 100.0,  # Normalize age
            1 if patient_data.get('gender') == 'male' else 0,
            patient_data.get('bmi', 0) / 50.0  # Normalize BMI
        ])
        
        # Vital signs
        vitals = patient_data.get('vitals', {})
        features.extend([
            vitals.get('blood_pressure', {}).get('systolic', 120) / 200.0,
            vitals.get('blood_pressure', {}).get('diastolic', 80) / 150.0,
            vitals.get('pulse', 70) / 200.0,
            vitals.get('temperature', 37) / 43.0,
            vitals.get('oxygen', 98) / 100.0
        ])
        
        # Symptom features
        symptoms = patient_data.get('symptoms', [])
        symptom_categories = {
            'pain': 0, 'fever': 0, 'respiratory': 0,
            'gastrointestinal': 0, 'neurological': 0,
            'musculoskeletal': 0, 'cardiovascular': 0,
            'dermatological': 0
        }
        
        for symptom in symptoms:
            if symptom['category'] in symptom_categories:
                symptom_categories[symptom['category']] += 1
                
        features.extend(list(symptom_categories.values()))
        
        # Duration features
        duration = patient_data.get('duration', {})
        total_days = (
            duration.get('days', 0) +
            duration.get('weeks', 0) * 7 +
            duration.get('months', 0) * 30 +
            duration.get('years', 0) * 365
        )
        features.append(min(total_days / 365.0, 1.0))  # Normalize to maximum of 1 year
        
        # Emergency level
        features.append(patient_data.get('emergency_level', 0))
        
        return np.array(features).reshape(1, -1)
        
    def predict_risk(self, processed_data: np.ndarray) -> Tuple[float, Dict]:
        """
        Predict risk score using ensemble of models
        
        Args:
            processed_data: Preprocessed patient data
            
        Returns:
            Tuple of (risk score, detailed analysis)
        """
        # Get predictions from both models
        deep_pred = self.deep_model.predict(processed_data)
        boost_pred = self.gradient_booster.predict(processed_data)
        
        # Weighted ensemble
        risk_score = float(0.7 * deep_pred + 0.3 * boost_pred) * 100
        
        # Generate detailed analysis
        analysis = self._generate_detailed_analysis(processed_data, deep_pred, boost_pred)
        
        return risk_score, analysis
        
    def _generate_detailed_analysis(self, 
                                  processed_data: np.ndarray,
                                  deep_pred: np.ndarray,
                                  boost_pred: np.ndarray) -> Dict:
        """
        Generate detailed analysis of the risk prediction
        """
        # Use gradient tape for feature importance
        with tf.GradientTape() as tape:
            input_tensor = tf.convert_to_tensor(processed_data, dtype=tf.float32)
            tape.watch(input_tensor)
            prediction = self.deep_model(input_tensor)
            
        gradients = tape.gradient(prediction, input_tensor)
        feature_importance = np.abs(gradients.numpy()[0])
        
        # Feature names for interpretation
        feature_names = [
            'age', 'gender', 'bmi',
            'systolic_bp', 'diastolic_bp', 'pulse', 'temperature', 'oxygen',
            'pain_symptoms', 'fever_symptoms', 'respiratory_symptoms',
            'gastrointestinal_symptoms', 'neurological_symptoms',
            'musculoskeletal_symptoms', 'cardiovascular_symptoms',
            'dermatological_symptoms',
            'duration',
            'emergency_level'
        ]
        
        # Get top contributing factors
        contributing_factors = []
        risk_levels = {
            (0, 20): "Low",
            (20, 40): "Moderate-Low",
            (40, 60): "Moderate",
            (60, 80): "Moderate-High",
            (80, 100): "High"
        }
        
        # Find risk level
        risk_value = float(deep_pred[0][0] * 100)
        risk_level = next(
            level for (lower, upper), level in risk_levels.items()
            if lower <= risk_value < upper
        )
        
        # Analyze feature importance
        for importance, feature in sorted(zip(feature_importance, feature_names), 
                                       key=lambda x: x[0], 
                                       reverse=True):
            if importance > 0.1:
                contributing_factors.append({
                    'factor': feature.replace('_', ' ').title(),
                    'importance': float(importance),
                    'contribution': 'Major' if importance > 0.3 else 'Moderate'
                })
                
        # Generate recommendations based on risk level and factors
        recommendations = self._generate_recommendations(
            risk_level,
            contributing_factors,
            processed_data
        )
        
        return {
            'risk_level': risk_level,
            'contributing_factors': contributing_factors,
            'model_confidence': {
                'deep_model': float(deep_pred[0][0]),
                'gradient_boost': float(boost_pred[0]),
                'ensemble_agreement': float(1 - abs(deep_pred[0][0] - boost_pred[0]))
            },
            'recommendations': recommendations
        }
        
    def _generate_recommendations(self, 
                                risk_level: str,
                                contributing_factors: List[Dict],
                                processed_data: np.ndarray) -> List[str]:
        """
        Generate personalized recommendations based on risk analysis
        """
        recommendations = []
        
        # Basic recommendation based on risk level
        if risk_level in ["High", "Moderate-High"]:
            recommendations.append("Immediate medical consultation recommended")
            
        # Check vital signs
        vitals_idx = {'systolic_bp': 3, 'diastolic_bp': 4, 'pulse': 5, 
                     'oxygen': 7}
        
        for factor in contributing_factors:
            factor_name = factor['factor'].lower().replace(' ', '_')
            if factor_name in vitals_idx:
                idx = vitals_idx[factor_name]
                value = processed_data[0][idx]
                if value > 0.8:  # Normalized value > 80%
                    recommendations.append(
                        f"Monitor {factor['factor']} closely - currently elevated"
                    )
                    
        # Symptom-based recommendations
        symptom_idx = range(8, 16)  # Indices for symptom features
        symptom_values = processed_data[0][symptom_idx]
        if np.any(symptom_values > 0.5):
            recommendations.append(
                "Multiple symptoms detected - keep detailed symptom diary"
            )
            
        # Duration-based recommendations
        duration_value = processed_data[0][16]
        if duration_value > 0.25:  # More than 3 months
            recommendations.append(
                "Long-term condition detected - consider specialist consultation"
            )
            
        return recommendations