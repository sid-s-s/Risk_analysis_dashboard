import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import base64

class ImageProcessor:
    def __init__(self):
        # Initialize the model (in production, load a proper medical image analysis model)
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        # Placeholder for actual medical image analysis model
        # In production, this would load a trained medical imaging model
        try:
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=True,
                weights='imagenet'
            )
        except Exception as e:
            print(f"Warning: Could not load MobileNetV2 weights: {e}")
            # Create a simple model for demonstration
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(1000, activation='softmax')(x)
            base_model = tf.keras.Model(inputs, outputs)
            
        return base_model
    
    def process_image(self, image_data: str) -> tuple[list[str], list[float], list[str]]:
        """
        Process the base64 encoded image and return analysis results.
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            Tuple of (detected conditions, confidence scores, visual markers)
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Preprocess image
            image = image.convert('RGB')
            image = image.resize((224, 224))
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = tf.expand_dims(image_array, 0)
            image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
            
            # Get predictions
            predictions = self.model.predict(image_array)
            
            # Convert predictions to health-related insights
            # This is a placeholder - in production, use actual medical condition detection
            detected_conditions = ["Skin condition detected", "Possible inflammation"]
            confidence_scores = [0.85, 0.72]
            visual_markers = ["Region 1: Upper left quadrant", "Region 2: Center"]
            
            return detected_conditions, confidence_scores, visual_markers
            
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

class TextAnalyzer:
    def __init__(self):
        # Initialize NLP components (in production, use proper medical NLP models)
        self.keywords = {
            'symptoms': ['pain', 'fever', 'cough', 'headache', 'fatigue'],
            'severity': ['mild', 'moderate', 'severe'],
            'duration': ['days', 'weeks', 'months', 'years']
        }
    
    def extract_medical_info(self, text: str) -> dict:
        """
        Extract medical information from text description.
        
        Args:
            text: Patient's description of symptoms and conditions
            
        Returns:
            Dictionary containing extracted medical information
        """
        text = text.lower()
        
        # Extract symptoms
        symptoms = [word for word in self.keywords['symptoms'] if word in text]
        
        # Extract severity
        severity = next((word for word in self.keywords['severity'] if word in text), None)
        
        # Extract duration
        duration = next((word for word in self.keywords['duration'] if word in text), None)
        
        # Extract numeric values (e.g., age, measurements)
        import re
        numbers = re.findall(r'\d+', text)
        age = next((int(num) for num in numbers if int(num) < 120), None)  # Assume first number < 120 is age
        
        return {
            'symptoms': symptoms,
            'severity': severity,
            'duration': duration,
            'age': age
        }