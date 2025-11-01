from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from utils.enhanced_analyzers import ImageProcessor, TextAnalyzer
from models.enhanced_risk_model import RiskAnalysisModel

app = FastAPI(title="Patient Risk Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors and models
image_processor = ImageProcessor()
text_analyzer = TextAnalyzer()
risk_model = RiskAnalysisModel()

class ImageData(BaseModel):
    image: str

class TextData(BaseModel):
    age: Optional[int]
    gender: Optional[str]
    symptoms_description: str

@app.post("/analyze-image")
async def analyze_image(data: ImageData):
    try:
        conditions, scores, markers = image_processor.process_image(data.image)
        response = {
            "detected_conditions": conditions,
            "confidence_scores": scores,
            "visual_markers": markers,
            "model_type": "deep_learning" if image_processor.has_deep_learning else "basic"
        }
        
        # Add warning if using fallback model
        if not image_processor.has_deep_learning:
            response["warning"] = ("Using basic image analysis. Install TensorFlow "
                                 "for more accurate results.")
        
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(e)}"
        )

@app.post("/analyze-text")
async def analyze_text(data: TextData):
    try:
        print(f"Received text analysis request: {data}")
        
        # Extract medical information from text
        medical_info = text_analyzer.extract_medical_info(data.symptoms_description)
        print(f"Medical info extracted: {medical_info}")
        
        # Add demographic information for risk analysis
        analysis_data = {
            "symptoms": medical_info["symptoms"],
            "severity": medical_info["severity"],
            "age": data.age,
            "gender": data.gender,
            "conditions": medical_info.get("conditions", [])
        }
        print(f"Analysis data prepared: {analysis_data}")
        
        # Process features and get risk prediction
        try:
            processed_features = risk_model.process_features(analysis_data)
            risk_score, detailed_analysis = risk_model.predict_risk(processed_features)
        except Exception as e:
            print(f"Risk model error: {e}")
            # Fallback to basic risk assessment if ML model fails
            risk_score = 0.5  # Default medium risk
            if medical_info["severity"] == "high":
                risk_score = 0.8
            elif medical_info["severity"] == "low":
                risk_score = 0.2
                
            detailed_analysis = {
                "risk_level": "medium" if risk_score == 0.5 else medical_info["severity"],
                "contributing_factors": [
                    {"factor": f"Reported {s['symptom']}", "impact": s['confidence']}
                    for s in medical_info["symptoms"]
                ],
                "recommendations": ["Please consult a healthcare professional for a thorough evaluation."]
            }
        
        # Construct response with all required fields
        response = {
            "risk_score": risk_score,
            "detected_symptoms": [
                {"symptom": s["symptom"], "category": "reported"}
                for s in medical_info["symptoms"]
            ],
            "severity": medical_info["severity"],
            "duration": medical_info.get("duration", "unknown"),
            "risk_level": detailed_analysis["risk_level"],
            "contributing_factors": [
                f["factor"] for f in detailed_analysis.get("contributing_factors", [])
            ],
            "recommendations": detailed_analysis.get("recommendations", [
                "Seek immediate medical attention if symptoms worsen."
            ]),
            "emergency_level": medical_info.get("emergency_level", medical_info["severity"]),
            "confidence": medical_info["confidence"],
            "conditions": medical_info["conditions"]
        }
        
        # Add warning if using basic analysis
        if not text_analyzer.nlp:
            response["warning"] = "Using basic text analysis. Install transformers for more accurate results."
            
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
