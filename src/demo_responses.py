# Demo responses when APIs are rate-limited
DEMO_RESPONSES = {
    "explanation": """
**Pathology Report - Histological Analysis**

**Specimen**: Breast tissue at 40× magnification (96×96 pixels)
**Stain**: Hematoxylin & Eosin (H&E) Standard

**Morphological Findings**:

**Nuclear Features**:
- Moderate to high nuclear pleomorphism observed
- Nuclei demonstrate significant variation in size and shape with irregular membranes
- Coarse granular chromatin pattern with prominent nucleoli
- High nuclear-to-cytoplasmic ratio consistent with epithelial proliferation

**Cellular Architecture**:
- Increased cellularity with crowded arrangement
- Loss of normal architectural pattern and ductal organization
- Infiltrative growth pattern observed
- Abnormal glandular formations noted

**Mitotic Activity**:
- Approximately 8-10 mitoses per 10 high power fields
- Several atypical mitotic forms identified
- Increased mitotic rate suggests active cellular proliferation

**Stromal Response**:
- Desmoplastic reaction present with collagen deposition around epithelial nests
- Increased vascularity with surrounding lymphocytic infiltration
- Fibrous tissue proliferation pattern consistent with invasive process

**Assessment**: Findings suggestive of infiltrating ductal carcinoma with invasive features. Clinical correlation with imaging and clinical presentation strongly recommended.
    """,
    
    "deepdive": """
**Advanced Morphological Analysis - Deep Dive**

**1) Cell Density & Arrangement**
- High cellularity with significant crowding
- Complete loss of normal acinar/ductal architecture
- Infiltrative growth pattern extending into surrounding stroma
- Abnormal solid nesting arrangements predominant
- Loss of myoepithelial layer

**2) Nuclear Pleomorphism & Chromatin**
- Grade 2-3 nuclear atypia on Bloom-Richardson scale
- Marked nuclear size variation (anisokaryosis)
- Coarse granular, hyperchromatic chromatin
- Multiple prominent nucleoli per cell
- Irregular, convoluted nuclear membranes
- Increased N/C ratio (>1:1) indicating cellular transformation

**3) Gland/Duct Formation**
- Severely abnormal tubule formation (10-15% tubule-forming pattern)
- Solid areas predominantly present
- Luminal space formation largely absent
- Cribriform pattern with central necrosis in some areas
- Compressed residual normal ducts at periphery

**4) Mitotic Figures**
- Mitotic rate: 8-10 per 10 HPF
- Significant proportion of atypical/abnormal forms
- Abnormal tripolar and multipolar mitoses present
- Correlates with aggressive biological behavior
- Supporting high-grade designation

**5) Stromal Changes**
- Robust desmoplastic reaction (fibrous response)
- Increased tumor-associated vasculature
- Dense lymphocytic and mononuclear infiltration
- Collagen type I and III deposition around tumor nests
- Evidence of early invasion into adjacent structures

**Diagnostic Impression**: Infiltrating Ductal Carcinoma (IDC), Grade II-III. Morphology consistent with intermediate to high-grade malignancy with significant invasive potential.
    """,
    
    "chat_response": """
**Clinical Correlation & Recommendations**

Based on the morphologic findings, this specimen demonstrates features highly consistent with invasive ductal carcinoma of the breast.

**Key Clinical Points**:

• **Histological Grade**: Intermediate to high grade (Scarff-Bloom-Richardson Grade II-III)
  - Indicates more aggressive biological behavior
  - Suggests higher likelihood of metastatic potential
  
• **Immunophenotypic Workup Recommended**:
  - ER/PR receptor status (hormone receptor testing)
  - HER2/neu gene amplification (HER2 status)
  - Ki-67 proliferation index recommended for grade confirmation
  
• **Staging Consideration**:
  - Tumor size determination necessary for TNM staging
  - Lymph node status critical for stage assignment
  - Assessment of clear margins required
  
• **Treatment Implications**:
  - Grade suggests multimodal therapy approach likely
  - Adjuvant chemotherapy typically indicated for grade II-III tumors
  - Hormone therapy and/or HER2-targeted therapy based on receptor status

**Recommended Next Steps**:
1. ✓ Complete immunohistochemical panel (ER, PR, HER2)
2. ✓ Assess surgical margins and tumor extensiveness
3. ✓ Consider molecular testing/genomic profiling
4. ✓ Multidisciplinary tumor board review recommended
5. ✓ Correlation with imaging findings and clinical presentation essential

**Clinical Context**: Patient demographics and prior imaging (mammogram, ultrasound) should be correlated with these morphologic findings for comprehensive diagnostic assessment and treatment planning.

Would you like clarification on any specific morphologic feature or additional analysis details?
    """
}

def get_demo_response(request_type="explanation"):
    """Return appropriate demo response based on request type"""
    return DEMO_RESPONSES.get(request_type, DEMO_RESPONSES["explanation"])

