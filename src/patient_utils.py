# Patient data utilities for dynamic EHR

import random
import hashlib
from datetime import datetime, timedelta

def generate_patient_id():
    """Generate a unique patient ID based on timestamp"""
    ts = datetime.now().timestamp()
    hash_val = hashlib.md5(str(ts).encode()).hexdigest()[:5].upper()
    return f"PT-{datetime.now().year}-{hash_val}"

def generate_patient_data(filename=""):
    """Generate synthetic patient data based on upload"""
    patient_id = generate_patient_id()
    
    # Vary patient characteristics
    age = random.randint(35, 75)
    sex_list = ["Female", "Male"]
    sex = random.choice(sex_list)
    
    brca_status = ["BRCA1 Negative", "BRCA2 Negative", "BRCA1 Positive", "BRCA2 Positive"][random.randint(0, 3)]
    
    mammo_scores = ["BI-RADS 2 (Benign)", "BI-RADS 3 (Probably benign)", "BI-RADS 4A", "BI-RADS 4B", "BI-RADS 4C", "BI-RADS 5 (Malignant)"]
    mammo = random.choice(mammo_scores)
    
    prior_dx_list = [
        "No prior abnormalities",
        "Atypical Ductal Hyperplasia",
        "Lobular Carcinoma In-Situ (LCIS)",
        "Fibroadenoma (benign)",
        "Previous cancer (contralateral)"
    ]
    prior_dx = random.choice(prior_dx_list)
    
    referring_doctors = ["Dr. Sharma", "Dr. Chen", "Dr. Patel", "Dr. Kumar", "Dr. Williams", "Dr. Johnson"]
    referring = random.choice(referring_doctors)
    
    specimens = ["Core Needle Biopsy", "Excisional Biopsy", "Fine Needle Aspiration", "Mastectomy", "Lumpectomy"]
    specimen = random.choice(specimens)
    
    # Generate accession number
    acc_num = f"SRG-{datetime.now().year}-{random.randint(1000, 9999)}"
    
    return {
        "patient_id": patient_id,
        "age": age,
        "sex": sex,
        "specimen": specimen,
        "stain": "H&E Standard",
        "prior_dx": prior_dx,
        "brca_status": brca_status,
        "mammogram": mammo,
        "referring": f"{referring} · Oncology",
        "accession": acc_num,
        "timestamp": datetime.now().strftime("%d %b %Y %H:%M"),
        "filename": filename or "specimen.jpg"
    }

def get_patient_display_html(patient_data):
    """Generate HTML for patient EHR display"""
    if not patient_data:
        return """
        <div class="ehr-section">
            <div class="ehr-section-title">EHR · Patient Context</div>
            <div style="text-align:center;padding:1rem;color:var(--text-3);">
                <div style="font-size:2rem;margin-bottom:0.5rem;">📋</div>
                <div style="font-size:0.9rem;">No specimen uploaded</div>
                <div style="font-family:var(--font-mono);font-size:0.65rem;margin-top:0.3rem;">
                    Upload image to generate patient context</div>
            </div>
        </div>
        """
    
    p = patient_data
    
    # Color coding for BRCA and Mammogram
    brca_class = "alert" if "Positive" in p["brca_status"] else "ok"
    mammo_prefix = p["mammogram"].split()[0]
    mammo_class = "alert" if mammo_prefix in ["BI-RADS 4C", "BI-RADS 5"] else "warn" if mammo_prefix in ["BI-RADS 4A", "BI-RADS 4B"] else "ok"
    
    html = f"""
    <div class="ehr-section">
        <div class="ehr-section-title">🏥 Patient Demographics</div>
        <div class="ehr-row">
            <span class="ehr-key">ID</span>
            <span class="ehr-val">{p['patient_id']}</span>
        </div>
        <div class="ehr-row">
            <span class="ehr-key">Age / Sex</span>
            <span class="ehr-val">{p['age']} yrs · {p['sex']}</span>
        </div>
        <div class="ehr-row">
            <span class="ehr-key">Specimen Type</span>
            <span class="ehr-val">{p['specimen']}</span>
        </div>
        <div class="ehr-row">
            <span class="ehr-key">Stain</span>
            <span class="ehr-val">{p['stain']}</span>
        </div>
    </div>
    
    <div class="ehr-section">
        <div class="ehr-section-title">📊 Clinical Context</div>
        <div class="ehr-row">
            <span class="ehr-key">Referring</span>
            <span class="ehr-val" style="font-size:0.72rem;">{p['referring']}</span>
        </div>
        <div class="ehr-row">
            <span class="ehr-key">Prior Dx</span>
            <span class="ehr-val warn" style="font-size:0.72rem;">{p['prior_dx']}</span>
        </div>
        <div class="ehr-row">
            <span class="ehr-key">BRCA Status</span>
            <span class="ehr-val {brca_class}" style="font-size:0.72rem;">{p['brca_status']}</span>
        </div>
        <div class="ehr-row">
            <span class="ehr-key">Mammogram</span>
            <span class="ehr-val {mammo_class}" style="font-size:0.72rem;">{p['mammogram']}</span>
        </div>
    </div>
    
    <div class="ehr-section">
        <div class="ehr-section-title">📄 Accession Info</div>
        <div class="ehr-row">
            <span class="ehr-key">Accession #</span>
            <span class="ehr-val" style="font-size:0.7rem;">{p['accession']}</span>
        </div>
        <div class="ehr-row">
            <span class="ehr-key">Date/Time</span>
            <span class="ehr-val" style="font-size:0.7rem;">{p['timestamp']}</span>
        </div>
    </div>
    """
    return html
