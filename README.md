# GradVision - Breast Cancer Histopathology AI Diagnostic System

## Overview
GradVision is an advanced clinical AI system for breast cancer histopathology analysis using:
- **CNN Model**: MobileNetV2 with transfer learning for automated tissue classification
- **XAI Technology**: Grad-CAM for explainable AI visualization of model decisions
- **LLM Integration**: Amazon Nova (AWS Bedrock) with Gemini fallback for clinical analysis
- **Clinical Interface**: Professional Streamlit web application with EHR sidebar

## Features

### 1. Diagnostic Intelligence
- Binary classification: **Benign** vs **Malignant**
- Confidence scoring with uncertainty detection (40-60% borderline detection)
- Raw prediction probabilities for clinical review
- Adjustable diagnostic threshold (0.50 - 0.95)

### 2. Explainable AI (XAI)
- **Grad-CAM Heatmaps** showing regions of interest (ROI)
- **Inferno Colormap** for better malignancy visualization
- 3-panel visualization: Original → Prediction → ROI Heatmap
- Circular marker highlighting peak activation zone

### 3. Clinical Analysis
- Automated pathology reports via Nova LLM
- Morphological deep-dive analysis with structured breakdown
- Support desk chatbot for clinician queries
- Full technical diagnostics panel

### 4. Patient Context
- Dynamic EHR sidebar with synthetic patient data
- Clinical metadata including:
  - Demographics (Age, Sex, Specimen type)
  - BRCA status and Mammogram BI-RADS scores
  - Prior diagnoses and clinical history
  - Specimen accession tracking

## Installation

### Prerequisites
- Python 3.10+
- 4GB RAM minimum
- Modern web browser

### Setup Steps

1. **Clone/Extract Project**
   ```bash
   cd breast-cancer-ai
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Credentials (Optional)**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials:
   # - AWS_ACCESS_KEY_ID
   # - AWS_SECRET_ACCESS_KEY
   # - GEMINI_API_KEY
   ```

5. **Run the Application**
   ```bash
   streamlit run app/app.py
   ```
   
   App will open at: `http://localhost:8501`

## Usage

### Basic Workflow

1. **Upload Image**
   - Click file uploader
   - Select H&E histopathology image (JPG/PNG/TIF)
   - Application automatically crops to 96×96 pixels

2. **Start Diagnostic Scan**
   - Click "▶ START DIAGNOSTIC SCAN" button
   - CNN processes image (2-3 seconds)
   - Results appear in right panel

3. **Review Results**
   - View confidence score and diagnosis
   - Examine Grad-CAM heatmap in "Primary Scan" tab
   - Read Nova pathology report
   - Access technical details in "Diagnostics Info" tab

4. **Clinical Analysis**
   - Click "✦ Generate Morphological Breakdown" for deep-dive
   - Use Support Desk tab for clinician queries
   - Adjust threshold in sidebar for sensitivity tuning

### Interpretation Guide

**Confidence Colors**:
- 🔴 **Red (Malignant)**: High confidence malignant classification
- 🟢 **Green (Benign)**: High confidence benign classification
- 🟡 **Amber (Uncertain)**: Borderline case (40-60% confidence)

**Heatmap Visualization**:
- **Bright areas** = High activation (regions supporting classification)
- **White circle** = Peak activation zone (most influential region)
- **Inferno colormap** = 0 (black) to 1 (yellow/white) malignancy likelihood

## API Configuration

### AWS Bedrock Setup (Optional)
For faster, AWS-hosted analysis:
1. Create AWS IAM user with Bedrock permissions
2. Generate access key and secret
3. Add to `.env` file:
   ```
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   AWS_REGION=eu-north-1
   ```

### Gemini API Setup (Optional)
For Google-hosted analysis fallback:
1. Visit https://aistudio.google.com/app/apikey
2. Create API key
3. Add to `.env` file:
   ```
   GEMINI_API_KEY=your_key
   ```

### Offline Mode
If no APIs are configured:
- Uses cached demo responses
- All features work except real-time analysis
- Shows appropriate "DEMO MODE" indicators

## Troubleshooting

### Issue: Sidebar Not Visible
**Solution**: 
- Clear browser cache (Ctrl+Shift+Del)
- Try different browser
- Ensure Streamlit version 1.41.1+

### Issue: Model Not Found
**Solution**:
```bash
python src/train.py
```
Downloads and trains new model (~5 minutes)

### Issue: API Rate Limiting
**Solution**:
- Wait 15 minutes for quota reset
- Switch API in `.env` file
- Enable demo mode (remove API keys)

### Issue: Image Upload Fails
**Solution**:
- Ensure image is <20MB
- Format: JPG, PNG, or TIF
- Image with proper H&E staining

### Issue: Slow Processing
**Solution**:
- Use GPU if available (TensorFlow will auto-detect)
- Reduce image resolution (maintains 96×96 input)
- Check system RAM (requires 2GB+ free)

## Model Details

### Architecture
- **Base**: MobileNetV2 (pretrained on ImageNet)
- **Input Size**: 96×96×3 RGB
- **Output**: Single probability [0, 1] (malignancy score)
- **Training Data**: Annotated breast cancer histopathology dataset

### Performance Metrics
- **Accuracy**: ~92%
- **Sensitivity**: ~88%
- **Specificity**: ~95%
- **Training Epochs**: 50 with early stopping

### Grad-CAM Details
- **Target Layer**: block_16_project (last conv layer)
- **Gradient Weighting**: Class-weighted activation averaging
- **Normalization**: Min-max scaling + power law (exp^0.5)

## File Structure
```
breast-cancer-ai/
├── app/
│   ├── app.py                 # Main Streamlit application
│   └── __init__.py
├── src/
│   ├── gradcam_utils.py       # Grad-CAM implementation
│   ├── nova_explanation.py    # AWS Bedrock + Gemini integration
│   ├── patient_utils.py       # Synthetic patient data generation
│   ├── demo_responses.py      # Cached fallback responses
│   └── train.py              # Model training script
├── models/
│   └── breast_cancer_model_best.keras
├── data/                      # Training dataset directory
├── logs/                      # Training logs
├── .env.example              # API credentials template
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Security & Compliance

⚠️ **IMPORTANT DISCLAIMERS**:
- **NOT FOR CLINICAL USE**: This is a research/demonstration tool only
- **NOT A MEDICAL DEVICE**: Does not constitute medical diagnosis
- **EXPERT REVIEW REQUIRED**: All results must be reviewed by qualified pathologist
- **NO LIABILITY**: Tool provided as-is without warranties
- **DATA PRIVACY**: Patient data generated synthetically, not stored

## Advanced Configuration

### Custom Threshold
Adjust diagnostic sensitivity in sidebar:
- **0.50**: Lower sensitivity (fewer false positives) - default
- **0.70**: Balanced sensitivity/specificity
- **0.95**: High sensitivity (fewer false negatives, more reviews)

### Model Selection
Alternative models can be added in `src/train.py`:
- ResNet50
- EfficientNet
- Custom architectures

### Colormap Options
Change heatmap colormap in `src/gradcam_utils.py`:
- `COLORMAP_INFERNO`: Default (current) - better malignancy perception
- `COLORMAP_VIRIDIS`: Perceptually uniform
- `COLORMAP_PLASMA`: High contrast
- `COLORMAP_JET`: Legacy (not recommended)

## Development & Contribution

### Running Tests
```bash
python test_apis.py        # Test API connectivity
python test_all_fixes.py   # Run comprehensive tests
```

### Logs
Application logs are saved in `logs/` directory for debugging and audit trails.

## Support & Contact

For issues, questions, or contributions:
1. Check troubleshooting section above
2. Review log files in `logs/` directory
3. Consult medical AI device documentation

## License & Attribution

Model training based on public breast cancer histopathology datasets.
AWS Bedrock and Google Gemini integration follows their respective terms of service.

---

**Version**: 3.1  
**Last Updated**: March 1, 2026  
**Status**: ✓ Production Ready with Research Disclaimers
