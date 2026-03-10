"""
Professional AI Medical Report Generator

Generates pathology-style medical reports summarizing AI analysis results
and exports them as downloadable PDF documents using ReportLab.
"""

import io
from datetime import datetime
from typing import Dict, Optional, Any, Tuple
from PIL import Image
import numpy as np

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
        PageBreak, Image as RLImage, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_medical_report(
    result_dict: Dict[str, Any],
    heatmap_img: Optional[Image.Image] = None,
    tumor_img: Optional[Image.Image] = None
) -> Tuple[Optional[bytes], str]:
    """
    Generates a professional pathology-style medical report as PDF.
    
    Args:
        result_dict: Dictionary containing analysis results with keys:
            - prediction: str ("Malignant" or "Benign")
            - confidence: float (0.0 to 1.0)
            - cell_count: int (number of detected cells)
            - cell_density: float (cells per pixel area)
            - irregular_nuclei_ratio: float (0.0 to 1.0)
            - cluster_count: int (number of clusters)
            - largest_cluster: int (size of largest cluster)
            - suspicion_level: str ("Low", "Moderate", "High", "Unknown")
            - image_name: str (optional, name of analyzed image)
        
        heatmap_img: PIL Image of GradCAM heatmap (optional)
        tumor_img: PIL Image of tumor localization (optional)
    
    Returns:
        Tuple of (pdf_bytes, status_message):
            - pdf_bytes: bytes object containing PDF or None if failed
            - status_message: str describing success or error
    """
    
    if not REPORTLAB_AVAILABLE:
        return None, "❌ ReportLab not installed. Install: pip install reportlab"
    
    try:
        # Extract data from result_dict
        prediction = result_dict.get('prediction', 'Unknown')
        confidence = result_dict.get('confidence', 0.0)
        cell_count = result_dict.get('cell_count', 0)
        cell_density = result_dict.get('cell_density', 0.0)
        irregular_ratio = result_dict.get('irregular_nuclei_ratio', 0.0)
        cluster_count = result_dict.get('cluster_count', 0)
        largest_cluster = result_dict.get('largest_cluster', 0)
        suspicion_level = result_dict.get('suspicion_level', 'Unknown')
        image_name = result_dict.get('image_name', 'Unknown')
        
        # Create PDF buffer
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Build story (list of flowable elements)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            borderPadding=6
        )
        
        body_style = ParagraphStyle(
            'Body',
            parent=styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leading=14
        )
        
        mono_style = ParagraphStyle(
            'Mono',
            parent=styles['BodyText'],
            fontSize=9,
            textColor=colors.HexColor('#555555'),
            fontName='Courier',
            spaceAfter=6
        )
        
        # 1. HEADER
        header_text = "AI-Assisted Breast Cancer Histopathology Report"
        story.append(Paragraph(header_text, title_style))
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = f"Report Generated: {timestamp} | Model: MobileNetV2 CNN | Analysis Type: Assisted Diagnosis"
        story.append(Paragraph(metadata, mono_style))
        story.append(Spacer(1, 0.2*inch))
        
        # 2. DIAGNOSTIC SUMMARY SECTION
        story.append(Paragraph("DIAGNOSTIC SUMMARY", section_style))
        
        # Create diagnostic summary table
        confidence_pct = round(confidence * 100, 1)
        diag_data = [
            ["Prediction:", f"<b>{prediction}</b>"],
            ["Confidence Score:", f"<b>{confidence_pct}%</b>"],
            ["Specimen:", image_name],
            ["Analysis Date:", timestamp.split()[0]]
        ]
        
        diag_table = Table(diag_data, colWidths=[2*inch, 3*inch])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 0.2*inch))
        
        # 3. TUMOR LOCALIZATION FINDINGS
        story.append(Paragraph("TUMOR LOCALIZATION FINDINGS", section_style))
        
        if prediction.upper() == "MALIGNANT":
            localization_text = (
                "Suspicious tissue region detected during AI-assisted analysis. "
                "The highlighted region represents areas that contributed most strongly to the malignancy prediction. "
                "These regions exhibit morphological patterns consistent with potential malignant transformation."
            )
        else:
            localization_text = (
                "No strong malignant tissue patterns were detected in the analyzed region. "
                "The tissue demonstrates patterns consistent with normal or benign histopathology."
            )
        
        story.append(Paragraph(localization_text, body_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Add heatmap image if available
        if heatmap_img is not None:
            try:
                hm_buffer = _convert_pil_to_bytes(heatmap_img)
                story.append(Paragraph("<b>GradCAM Heatmap - Attention Visualization</b>", body_style))
                rl_heatmap = RLImage(hm_buffer, width=4.5*inch, height=3*inch)
                story.append(rl_heatmap)
                story.append(Spacer(1, 0.1*inch))
            except Exception as e:
                print(f"Warning: Could not embed heatmap image: {str(e)}")
        
        # Add tumor localization image if available
        if tumor_img is not None:
            try:
                tm_buffer = _convert_pil_to_bytes(tumor_img)
                story.append(Paragraph("<b>Tumor Localization - Bounding Box Overlay</b>", body_style))
                rl_tumor = RLImage(tm_buffer, width=4.5*inch, height=3*inch)
                story.append(rl_tumor)
                story.append(Spacer(1, 0.1*inch))
            except Exception as e:
                print(f"Warning: Could not embed tumor image: {str(e)}")
        
        story.append(Spacer(1, 0.2*inch))
        
        # 4. CELL MORPHOLOGY ANALYSIS
        story.append(Paragraph("CELL MORPHOLOGY ANALYSIS", section_style))
        
        irregular_pct = round(irregular_ratio * 100, 1)
        morphology_data = [
            ["Cells Detected:", f"{cell_count}"],
            ["Cell Density:", f"{cell_density:.6f} cells/pixel"],
            ["Irregular Nuclei Ratio:", f"{irregular_pct}%"],
            ["Cell Clusters:", f"{cluster_count}"],
            ["Largest Cluster:", f"{largest_cluster} cells"]
        ]
        
        morph_table = Table(morphology_data, colWidths=[2*inch, 3*inch])
        morph_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        story.append(morph_table)
        story.append(Spacer(1, 0.15*inch))
        
        # Morphology interpretation
        morphology_interpretation = _generate_morphology_interpretation(
            suspicion_level, cell_count, irregular_pct, cluster_count
        )
        story.append(Paragraph(morphology_interpretation, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # 5. AI ASSESSMENT
        story.append(Paragraph("AI ASSESSMENT", section_style))
        
        ai_assessment = _generate_ai_assessment(
            prediction, confidence_pct, suspicion_level, irregular_pct
        )
        story.append(Paragraph(ai_assessment, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # 6. RECOMMENDATIONS & DISCLAIMER
        story.append(Paragraph("RECOMMENDATIONS & DISCLAIMER", section_style))
        
        disclaimer_text = (
            "<b>Important Notice:</b> This AI-assisted analysis is intended for research and clinical decision-support purposes only. "
            "This report is <u>not</u> a substitute for professional pathological examination. "
            "Final diagnosis must be confirmed by a qualified pathologist following standard clinical protocols. "
            "The AI model predictions should be considered as supplementary evidence supporting, but not replacing, "
            "traditional histopathological analysis. All results should be interpreted in the context of clinical presentation and additional investigations."
        )
        story.append(Paragraph(disclaimer_text, body_style))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_text = (
            "<b>GradVision Clinical Intelligence System</b><br/>"
            "Advanced Histopathology Analysis Platform<br/>"
            f"Powered by TensorFlow MobileNetV2 | Generated {timestamp}"
        )
        story.append(Paragraph(footer_text, mono_style))
        
        # Build PDF
        doc.build(story)
        
        pdf_bytes = pdf_buffer.getvalue()
        return pdf_bytes, "✅ Medical report generated successfully"
        
    except Exception as e:
        error_msg = f"❌ Report generation failed: {str(e)}"
        print(error_msg)
        return None, error_msg


def _convert_pil_to_bytes(pil_image: Image.Image) -> io.BytesIO:
    """
    Converts a PIL Image to BytesIO stream for ReportLab embedding.
    
    Args:
        pil_image: PIL Image object
    
    Returns:
        BytesIO stream containing JPEG image data
    """
    img_buffer = io.BytesIO()
    # Convert RGBA to RGB if necessary
    if pil_image.mode in ('RGBA', 'LA', 'P'):
        rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
        rgb_image.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
        rgb_image.save(img_buffer, format='JPEG', quality=85)
    else:
        pil_image.save(img_buffer, format='JPEG', quality=85)
    img_buffer.seek(0)
    return img_buffer


def _generate_morphology_interpretation(
    suspicion_level: str,
    cell_count: int,
    irregular_pct: float,
    cluster_count: int
) -> str:
    """
    Generates readable interpretation of morphology findings.
    
    Args:
        suspicion_level: "High", "Moderate", "Low", or "Unknown"
        cell_count: Number of detected cells
        irregular_pct: Percentage of irregular nuclei
        cluster_count: Number of abnormal clusters
    
    Returns:
        str with morphology interpretation
    """
    
    base_text = f"Morphological analysis identified {cell_count} cell nuclei with {irregular_pct}% irregular patterns. "
    
    if suspicion_level == "High":
        interpretation = (
            base_text +
            "High cellular density combined with significant irregular nuclei patterns suggest possible malignant tissue structures. "
            f"The detection of {cluster_count} abnormal cell clusters further supports concerning morphological features."
        )
    elif suspicion_level == "Moderate":
        interpretation = (
            base_text +
            "Moderate irregular cellular patterns are observed, which may warrant additional clinical correlation. "
            "The presence of cell clustering indicates areas of biological interest."
        )
    elif suspicion_level == "Low":
        interpretation = (
            base_text +
            "Cell morphology appears consistent with normal tissue structure, with minimal irregular nuclei. "
            "Overall morphological features do not suggest malignant transformation."
        )
    else:
        interpretation = (
            base_text +
            "Morphological analysis could not be completed or insufficient data was available. "
            "Results should be interpreted with clinical context."
        )
    
    return interpretation


def _generate_ai_assessment(
    prediction: str,
    confidence_pct: float,
    suspicion_level: str,
    irregular_pct: float
) -> str:
    """
    Generates comprehensive AI assessment combining all analysis streams.
    
    Args:
        prediction: "Malignant" or "Benign"
        confidence_pct: Confidence percentage (0-100)
        suspicion_level: Morphology suspicion level
        irregular_pct: Percentage of irregular nuclei
    
    Returns:
        str with comprehensive assessment
    """
    
    pred_lower = prediction.lower()
    
    if pred_lower == "malignant":
        assessment = (
            f"Based on deep learning analysis and morphological indicators, "
            f"the tissue sample demonstrates patterns consistent with <b>malignant</b> histopathological characteristics. "
            f"The CNN classifier achieved {confidence_pct}% confidence in this prediction, supported by morphological findings showing "
            f"{irregular_pct}% irregular nuclei and a suspicion level of {suspicion_level}. "
            f"The combination of computational vision features and morphological abnormalities suggests significant biological evidence of malignancy."
        )
    elif pred_lower == "benign":
        assessment = (
            f"Based on deep learning analysis and morphological indicators, "
            f"the tissue sample demonstrates patterns consistent with <b>benign</b> histopathological characteristics. "
            f"The CNN classifier achieved {confidence_pct}% confidence in this prediction, with morphological patterns showing "
            f"minimal irregular nuclei ({irregular_pct}%) and a suspicion level of {suspicion_level}. "
            f"Overall biological indicators support a benign classification."
        )
    else:
        assessment = (
            f"AI assessment could not be completed due to insufficient or unclear analysis data. "
            f"Prediction: {prediction} | Confidence: {confidence_pct}% | Morphology Suspicion: {suspicion_level}. "
            f"Clinical interpretation is recommended."
        )
    
    return assessment


def save_report_to_file(pdf_bytes: bytes, filename: str = "medical_report.pdf") -> str:
    """
    Saves PDF bytes to a file on disk.
    
    Args:
        pdf_bytes: Bytes object containing PDF data
        filename: Output filename
    
    Returns:
        str with file path or error message
    """
    try:
        with open(filename, 'wb') as f:
            f.write(pdf_bytes)
        return f"✅ Report saved to {filename}"
    except Exception as e:
        return f"❌ Failed to save report: {str(e)}"
