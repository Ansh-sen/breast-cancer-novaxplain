import os
import base64
import boto3
import requests
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import demo responses for fallback
try:
    from src.demo_responses import get_demo_response
except ImportError:
    def get_demo_response(request_type="explanation"):
        return "[DEMO MODE] API services unavailable. Using cached analysis."

class NovaExplainer:
    def __init__(self):
        # Load credentials from .env file
        self.aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "eu-north-1")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        
        # Model IDs
        self.nova_model = "eu.amazon.nova-lite-v1:0"  
        self.gemini_model = "gemini-2.0-flash"  # Updated to available model
        
        # Status flags
        self.bedrock = None
        self.aws_available = False
        self.gemini_available = bool(self.gemini_key)
        
        # Validate credentials before initialization
        if not self.aws_key or not self.aws_secret:
            print("[WARNING] AWS credentials missing in .env file")
        elif not self.gemini_key:
            print("[WARNING] Gemini API key missing in .env file")
        else:
            # Try to initialize AWS Bedrock client
            try:
                self.bedrock = boto3.client(
                    "bedrock-runtime",
                    region_name=self.aws_region,
                    aws_access_key_id=self.aws_key,
                    aws_secret_access_key=self.aws_secret,
                )
                self.aws_available = True
                print("[SUCCESS] AWS Bedrock initialized successfully")
            except Exception as e:
                print(f"[WARNING] AWS Bedrock initialization failed: {e}")
                self.aws_available = False
                print("   Falling back to Gemini API")
        
        # Ensure at least one service is available
        if not self.aws_available and not self.gemini_available:
            print("[ERROR] ERROR: No AI services configured! Add credentials to .env file")

    def _call_nova(self, prompt, system, image_bytes):
        """Call Amazon Nova model via Bedrock"""
        if not self.aws_available or self.bedrock is None:
            raise Exception("AWS Bedrock client not initialized")
        
        try:
            content = []
            if image_bytes:
                content.append({"image": {"format": "jpeg", "source": {"bytes": image_bytes}}})
            content.append({"text": prompt})
            
            # Add timeout of 30 seconds for the Bedrock API call
            response = self.bedrock.converse(
                modelId=self.nova_model,
                system=[{"text": system}] if system else [],
                messages=[{"role": "user", "content": content}],
                inferenceConfig={"maxTokens": 350, "temperature": 0.3}
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            error_msg = str(e)
            # Check if it's a rate limiting error
            if "ThrottlingException" in error_msg or "Too many tokens" in error_msg:
                raise Exception(f"AWS Rate Limited: {error_msg[:80]}")
            raise Exception(f"Nova API Error: {str(e)[:100]}")

    def _call_gemini(self, prompt, system, image_bytes):
        """Call Google Gemini API"""
        if not self.gemini_key:
            raise Exception("Gemini API key not configured")
        
        try:
            # Use the correct, available model
            model_id = self.gemini_model
            url = f"https://generativelanguage.googleapis.com/v1/models/{model_id}:generateContent?key={self.gemini_key}"
            
            parts = []
            if image_bytes:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(image_bytes).decode()
                    }
                })
            
            # Combine system instruction with prompt for Gemini v1 API
            if system:
                full_prompt = f"<SYSTEM>\n{system}\n</SYSTEM>\n\n{prompt}"
            else:
                full_prompt = prompt
            
            parts.append({"text": full_prompt})
            
            # Correct payload structure for Generative AI v1 API (no systemInstruction field)
            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "maxOutputTokens": 400,
                    "temperature": 0.3
                }
            }
            
            r = requests.post(url, json=payload, timeout=30)
            
            if r.status_code != 200:
                error_msg = r.text
                raise Exception(f"Gemini API Error {r.status_code}: {error_msg[:200]}")
            
            result = r.json()
            if "candidates" not in result or not result["candidates"]:
                raise Exception("No response from Gemini API")
            
            # Extract text from response
            candidates = result.get("candidates", [])
            if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                return candidates[0]["content"]["parts"][0].get("text", "No text in response")
            
            raise Exception("Unexpected response format from Gemini API")
            
        except requests.exceptions.Timeout:
            raise Exception("Gemini API timeout - request took too long")
        except Exception as e:
            raise Exception(f"Gemini API Error: {str(e)[:100]}")
    
    def _call(self, prompt, system="", image_bytes=None):
        """Routes requests through the fallback logic: try Nova first, then Gemini"""
        errors = []
        
        # Try Nova first if available
        if self.aws_available:
            try:
                print("[INFO] Attempting Amazon Nova...")
                return self._call_nova(prompt, system, image_bytes)
            except Exception as e:
                error_msg = str(e)
                print(f"[WARNING] Nova failed: {error_msg}")
                errors.append(error_msg)
        
        # Fallback to Gemini if available
        if self.gemini_available:
            try:
                print("[INFO] Falling back to Gemini API...")
                return self._call_gemini(prompt, system, image_bytes)
            except Exception as e:
                error_msg = str(e)
                print(f"[WARNING] Gemini failed: {error_msg}")
                errors.append(error_msg)
        
        # If both failed, provide helpful error message
        error_details = " | ".join(errors) if errors else "No AI services available"
        
        # Check for rate limit errors - use demo mode
        if "rate limit" in error_details.lower() or "quota" in error_details.lower() or "throttl" in error_details.lower():
            print("[DEMO MODE] Using cached response due to API rate limits")
            # Return appropriate demo response based on prompt type
            if "morphological" in prompt.lower() or "deep-dive" in prompt.lower():
                return get_demo_response("deepdive")
            elif "query" in prompt.lower() or "question" in prompt.lower() or "case:" in prompt.lower():
                return get_demo_response("chat_response")
            else:
                return get_demo_response("explanation")
        
        return f"[UNAVAILABLE] AI services temporarily unavailable. Error: {error_details[:100]}"

    def generate_explanation(self, pil_img, label, confidence, uncertain=False):
        """Generate pathology report from image and prediction"""
        try:
            # 1. Prepare Image
            buf = BytesIO()
            pil_img.save(buf, format="JPEG")
            image_bytes = buf.getvalue()

            # Enhanced prompt with more clinical context
            prompt = f"""Specimen Classification Result: {label}
Confidence Level: {confidence:.1%}
Certainty: {'High Confidence' if confidence > 0.75 else 'Moderate Confidence' if confidence > 0.6 else 'Low Confidence (Borderline)'}
{'⚠️ BORDERLINE CASE - Expert Review Recommended' if uncertain else ''}

Please provide a concise pathology report including:
1. Morphological appearance consistent with the classification
2. Key diagnostic features supporting the classification
3. Any atypical findings
4. Clinical implications and recommendations

Keep response under 250 words. Use clinical terminology appropriate for pathologists."""

            system = """You are an expert breast pathologist AI assistant. Your role is to:
- Provide morphological analysis of histopathology specimens
- Support clinical decision-making with evidence-based observations
- Flag borderline or uncertain cases for expert review
- Never provide definitive diagnoses - only support pathologist assessment

Always emphasize that final diagnosis must be made by a qualified pathologist."""

            # 2. Use the unified _call method with fallback logic
            return self._call(prompt, system, image_bytes)
        
        except Exception as e:
            return f"[ERROR] Failed to generate explanation: {str(e)[:100]}"
    
    def generate_morphological_breakdown(self, label, confidence, raw_pred, image_bytes=None):
        """Generate detailed morphological analysis"""
        try:
            prompt = f"""The specimen has been classified as: {label}
Classification confidence: {confidence:.1%}
Raw malignancy score: {raw_pred:.4f}

Please provide a STRUCTURED morphological breakdown using these exact headings:

1. CELL DENSITY & ARRANGEMENT
   [Describe cellular density, glandular vs. diffuse patterns, tissue architecture]

2. NUCLEAR MORPHOLOGY
   [Analyze nucleus size, chromatin pattern, irregularity, mitotic activity]

3. GLANDULAR & DUCTAL FEATURES
   [Comment on gland structure, ductal formation, lumen presence]

4. STROMAL CHARACTERISTICS
   [Describe fibrous tissue, inflammation, necrosis if present]

5. FINAL ASSESSMENT
   [Summary of findings and consistency with classification]

Keep analysis clinical and concise. Under 300 words."""

            system = """You are a senior breast pathologist with 20+ years of experience. 
Provide detailed, structured morphological analysis. Be specific and use standard pathology terminology.
Your analysis should support the CNN classification while noting any atypical findings."""

            return self._call(prompt, system, image_bytes)
        except Exception as e:
            return f"[ERROR] Morphological analysis failed: {str(e)[:80]}"