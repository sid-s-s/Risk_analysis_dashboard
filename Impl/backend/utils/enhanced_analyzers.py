import numpy as np
from PIL import Image
from io import BytesIO
import base64
import os
import warnings
warnings.filterwarnings('ignore')

# Optional deep learning dependencies
try:
    import tensorflow as tf
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.applications.densenet import preprocess_input
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not available. Using fallback models.")

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Transformers/PyTorch not available. Using fallback models.")
    print("TensorFlow not available, using simplified model")

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Transformers not available, using simplified text analysis")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Scikit-learn not available, using simplified analysis")

class ImageProcessor:
    def __init__(self):
        self.model = self._initialize_model()
        self.feature_extractor = self._initialize_feature_extractor()
        
    def _initialize_model(self):
        if HAS_TENSORFLOW:
            try:
                # Using DenseNet121 as it's proven effective for medical image analysis
                base_model = DenseNet121(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
                
                # Add custom layers for medical image analysis
                x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
                x = tf.keras.layers.Dense(1024, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(512, activation='relu')(x)
                outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
                
                model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
                
                if os.path.exists('medical_model_weights.h5'):
                    model.load_weights('medical_model_weights.h5')
                    
                return model
                
            except Exception as e:
                print(f"Warning: Could not initialize deep learning model: {e}")
        
        # Fallback to a simple classifier
        class SimpleClassifier:
            def predict(self, x):
                # Return mock predictions for demonstration
                return np.array([[0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]])
                
        return SimpleClassifier()
            
    def _initialize_feature_extractor(self):
        return tf.keras.applications.DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
    def process_image(self, image_data: str) -> tuple[list[str], list[float], list[str]]:
        try:
            # Decode and preprocess image
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(BytesIO(image_bytes))
            image = image.convert('RGB')
            image = image.resize((224, 224))
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = tf.expand_dims(image_array, 0)
            image_array = preprocess_input(image_array)
            
            # Extract features
            features = self.feature_extractor.predict(image_array)
            features_flat = tf.keras.layers.GlobalAveragePooling2D()(features)
            
            # Get predictions
            predictions = self.model.predict(image_array)
            
            # Map predictions to medical conditions
            condition_map = {
                0: "Dermatological condition",
                1: "Inflammation",
                2: "Vascular abnormality",
                3: "Tissue damage",
                4: "Infection signs",
                5: "Abnormal growth",
                6: "Chronic condition indicators",
                7: "Acute condition indicators",
                8: "Normal tissue",
                9: "Requires further analysis"
            }
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[-3:][::-1]  # Top 3 predictions
            detected_conditions = [condition_map[i] for i in top_indices]
            confidence_scores = [float(predictions[0][i]) for i in top_indices]
            
            # Generate visual markers using feature maps
            visual_markers = self._generate_visual_markers(features[0])
            
            return detected_conditions, confidence_scores, visual_markers
            
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
            
    def _generate_visual_markers(self, feature_maps):
        # Analyze feature maps to identify regions of interest
        feature_sum = np.sum(feature_maps, axis=2)
        threshold = np.percentile(feature_sum, 95)
        
        markers = []
        regions = [
            ("Upper left", feature_sum[:112, :112]),
            ("Upper right", feature_sum[:112, 112:]),
            ("Lower left", feature_sum[112:, :112]),
            ("Lower right", feature_sum[112:, 112:]),
            ("Center", feature_sum[56:168, 56:168])
        ]
        
        for region_name, region_data in regions:
            if np.max(region_data) > threshold:
                markers.append(f"Anomaly detected in {region_name} region")
                
        return markers

class TextAnalyzer:
    def __init__(self):
        self.symptom_classifier = None
        self.scaler = None
        self.nlp = None
        
        # Try to initialize ML components if available
        try:
            self.symptom_classifier = self._initialize_symptom_classifier()
        except Exception as e:
            print(f"Warning: Could not initialize symptom classifier: {e}")
            
        if HAS_TRANSFORMERS:
            try:
                self.nlp = pipeline("zero-shot-classification")
            except Exception as e:
                print(f"Warning: Could not initialize NLP pipeline: {e}")
        else:
            print("Warning: Transformers not available, using basic text analysis")
        
    def _initialize_symptom_classifier(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            
            # Load pre-trained classifier if available
            if os.path.exists('symptom_classifier.joblib'):
                import joblib
                return joblib.load('symptom_classifier.joblib')
                
            # Create a new classifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        except ImportError:
            print("Warning: scikit-learn not available. Using basic text analysis.")
            return None
        
    def extract_medical_info(self, text: str) -> dict:
        medical_info = {
            'severity': 'unknown',
            'conditions': [],
            'symptoms': [],
            'confidence': 0.0
        }
        
        # Use advanced NLP if available
        if self.nlp is not None:
            try:
                categories = [
                    "acute condition",
                    "chronic condition",
                    "emergency",
                    "mild condition",
                    "moderate condition",
                    "severe condition"
                ]
                
                result = self.nlp(text, categories, multi_label=True)
                medical_info['conditions'] = [
                    {'condition': label, 'score': score}
                    for label, score in zip(result['labels'], result['scores'])
                    if score > 0.3
                ]
                medical_info['confidence'] = max(result['scores'])
                
                # Determine severity based on highest scoring category
                severity_scores = {
                    label: score for label, score in zip(result['labels'], result['scores'])
                }
                if severity_scores.get('emergency', 0) > 0.5:
                    medical_info['severity'] = 'high'
                elif severity_scores.get('severe condition', 0) > 0.5:
                    medical_info['severity'] = 'high'
                elif severity_scores.get('moderate condition', 0) > 0.5:
                    medical_info['severity'] = 'medium'
                else:
                    medical_info['severity'] = 'low'
                    
            except Exception as e:
                print(f"Warning: Advanced NLP analysis failed: {e}")
                
        # Simple text-based analysis as fallback
        if not medical_info['conditions']:
            # Basic keyword matching
            emergency_keywords = ['severe', 'emergency', 'critical', 'extreme']
            moderate_keywords = ['moderate', 'concerning', 'significant']
            mild_keywords = ['mild', 'minor', 'slight']
            
            text_lower = text.lower()
            if any(word in text_lower for word in emergency_keywords):
                medical_info['severity'] = 'high'
                medical_info['confidence'] = 0.7
                medical_info['emergency_level'] = 'high'
            elif any(word in text_lower for word in moderate_keywords):
                medical_info['severity'] = 'medium'
                medical_info['confidence'] = 0.6
                medical_info['emergency_level'] = 'medium'
            elif any(word in text_lower for word in mild_keywords):
                medical_info['severity'] = 'low'
                medical_info['confidence'] = 0.6
                medical_info['emergency_level'] = 'low'
            else:
                medical_info['emergency_level'] = 'unknown'
                
        # Add duration based on keywords
        duration_keywords = {
            'long': ['chronic', 'months', 'years', 'persistent', 'ongoing'],
            'medium': ['weeks', 'several days', 'recurring'],
            'short': ['recent', 'today', 'yesterday', 'acute', 'sudden']
        }
        
        text_lower = text.lower()
        for duration, keywords in duration_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                medical_info['duration'] = duration
                break
        else:
                medical_info['duration'] = 'unknown'
            
        # Extract symptoms using common medical terminology
        symptoms = self._extract_symptoms(text)
        medical_info['symptoms'] = symptoms
            
        return medical_info
        
    def _extract_symptoms(self, text: str) -> list:
        """Extract symptoms from text using available tools."""
        symptoms = []
        
        # Common medical symptoms and their variations
        symptom_patterns = {
            'pain': ['pain', 'ache', 'hurt', 'sore', 'discomfort'],
            'fever': ['fever', 'temperature', 'hot', 'chills'],
            'fatigue': ['fatigue', 'tired', 'exhausted', 'weak', 'low energy'],
            'cough': ['cough', 'coughing', 'hack'],
            'breathing': ['breath', 'breathing', 'shortness of breath', 'wheezing'],
            'headache': ['headache', 'migraine', 'head pain'],
            'dizziness': ['dizzy', 'dizziness', 'vertigo', 'lightheaded'],
            'nausea': ['nausea', 'nauseous', 'sick to stomach', 'queasy'],
            'vomiting': ['vomit', 'vomiting', 'throwing up'],
            'diarrhea': ['diarrhea', 'loose stool'],
            'rash': ['rash', 'hives', 'skin irritation', 'itchy skin'],
            'swelling': ['swelling', 'swollen', 'inflammation'],
        }
        
        text_lower = text.lower()
        
        # Check for each symptom pattern
        for symptom_name, variations in symptom_patterns.items():
            if any(var in text_lower for var in variations):
                # Find the specific variation that matched
                matched_variation = next(var for var in variations if var in text_lower)
                # Get some context around the symptom
                words = text_lower.split()
                try:
                    symptom_index = next(i for i, word in enumerate(words) if matched_variation in word)
                    context_start = max(0, symptom_index - 2)
                    context_end = min(len(words), symptom_index + 3)
                    context = ' '.join(words[context_start:context_end])
                except StopIteration:
                    context = matched_variation
                    
                symptoms.append({
                    'symptom': symptom_name,
                    'confidence': 0.8,  # High confidence for direct mentions
                    'context': context
                })
                
        return symptoms
        severity_scores = {
            label: score for label, score in zip(result['labels'], result['scores'])
            if 'condition' in label
        }
        severity = max(severity_scores.items(), key=lambda x: x[1])[0]
        
        # Extract duration and temporal information
        duration = self._extract_duration(text)
        
        # Extract numeric values and vital signs
        vitals = self._extract_vitals(text)
        
        # Calculate risk factors
        risk_factors = self._calculate_risk_factors(symptoms, severity_scores)
        
        return {
            'symptoms': symptoms,
            'severity': severity,
            'duration': duration,
            'vitals': vitals,
            'risk_factors': risk_factors,
            'emergency_level': result['scores'][categories.index("emergency")]
        }
        
    def _extract_symptoms(self, text):
        # Enhanced symptom extraction using medical terminology
        medical_terms = {
            'pain': ['pain', 'ache', 'discomfort', 'soreness'],
            'fever': ['fever', 'temperature', 'chills'],
            'respiratory': ['cough', 'shortness of breath', 'wheezing'],
            'gastrointestinal': ['nausea', 'vomiting', 'diarrhea'],
            'neurological': ['headache', 'dizziness', 'confusion'],
            'musculoskeletal': ['weakness', 'stiffness', 'swelling'],
            'cardiovascular': ['chest pain', 'palpitations', 'irregular heartbeat'],
            'dermatological': ['rash', 'itching', 'skin changes']
        }
        
        found_symptoms = []
        text_lower = text.lower()
        
        for category, terms in medical_terms.items():
            for term in terms:
                if term in text_lower:
                    found_symptoms.append({
                        'category': category,
                        'symptom': term,
                        'context': self._extract_context(text_lower, term)
                    })
                    
        return found_symptoms
        
    def _extract_context(self, text, term):
        # Extract surrounding context for the symptom
        try:
            index = text.index(term)
            start = max(0, index - 30)
            end = min(len(text), index + len(term) + 30)
            return text[start:end].strip()
        except:
            return ""
            
    def _extract_duration(self, text):
        import re
        duration_patterns = {
            'days': r'(\d+)\s*(day|days)',
            'weeks': r'(\d+)\s*(week|weeks)',
            'months': r'(\d+)\s*(month|months)',
            'years': r'(\d+)\s*(year|years)'
        }
        
        durations = {}
        for unit, pattern in duration_patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                durations[unit] = int(match.group(1))
                
        return durations
        
    def _extract_vitals(self, text):
        import re
        vitals_patterns = {
            'blood_pressure': r'(?:bp|blood pressure)[:\s]*(\d+)/(\d+)',
            'temperature': r'(?:temp|temperature)[:\s]*(\d+\.?\d*)',
            'pulse': r'(?:pulse|heart rate)[:\s]*(\d+)',
            'oxygen': r'(?:o2|oxygen|spo2)[:\s]*(\d+)',
            'weight': r'(?:weight)[:\s]*(\d+\.?\d*)',
            'height': r'(?:height)[:\s]*(\d+)',
            'bmi': r'(?:bmi)[:\s]*(\d+\.?\d*)'
        }
        
        vitals = {}
        for vital, pattern in vitals_patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                if vital == 'blood_pressure':
                    vitals[vital] = {
                        'systolic': int(match.group(1)),
                        'diastolic': int(match.group(2))
                    }
                else:
                    vitals[vital] = float(match.group(1))
                    
        return vitals
        
    def _calculate_risk_factors(self, symptoms, severity_scores):
        risk_factors = []
        
        # Check for multiple symptoms in same category
        symptom_categories = {}
        for symptom in symptoms:
            category = symptom['category']
            if category in symptom_categories:
                symptom_categories[category] += 1
            else:
                symptom_categories[category] = 1
                
        # Identify high-risk combinations
        for category, count in symptom_categories.items():
            if count > 2:
                risk_factors.append(f"Multiple {category} symptoms")
                
        # Check severity scores
        if severity_scores.get('severe condition', 0) > 0.5:
            risk_factors.append("High severity indicators")
            
        # Add specific symptom combinations
        symptom_names = [s['symptom'] for s in symptoms]
        if 'chest pain' in symptom_names and 'shortness of breath' in symptom_names:
            risk_factors.append("Cardiac warning signs")
            
        return risk_factors
        
    def _extract_symptoms(self, text: str) -> list:
        """Extract symptoms from text using available tools."""
        symptoms = []
        
        # Common medical symptoms for basic keyword matching
        common_symptoms = [
            'fever', 'cough', 'headache', 'pain', 'fatigue', 'nausea',
            'vomiting', 'dizziness', 'shortness of breath', 'chest pain',
            'weakness', 'sweating', 'anxiety', 'depression', 'insomnia'
        ]
        
        text_lower = text.lower()
        
        # Use ML-based classifier if available
        if self.symptom_classifier is not None and self.scaler is not None:
            try:
                # Convert text to feature vector (simplified for example)
                features = np.array([
                    [1 if symptom in text_lower else 0 for symptom in common_symptoms]
                ])
                features_scaled = self.scaler.transform(features)
                
                # Get symptom predictions
                predictions = self.symptom_classifier.predict_proba(features_scaled)
                for symptom, prob in zip(common_symptoms, predictions[0]):
                    if prob > 0.3:  # Confidence threshold
                        symptoms.append({'symptom': symptom, 'confidence': float(prob)})
                        
            except Exception as e:
                print(f"Warning: ML-based symptom extraction failed: {e}")
                
        # Fallback to basic keyword matching
        if not symptoms:
            for symptom in common_symptoms:
                if symptom in text_lower:
                    symptoms.append({'symptom': symptom, 'confidence': 0.6})
                    
        return symptoms