import os

file_path = "d:/python_projects/breast-cancer-ai/app/app.py"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find where the HEADER starts
header_idx = -1
for i, line in enumerate(lines):
    if "# ── HEADER " in line:
        header_idx = i
        break

head_lines = lines[:header_idx]

# Replace background color in CSS
for i, line in enumerate(head_lines):
    if "--bg:" in line and "0b0f1a" in line:
        head_lines[i] = line.replace("0b0f1a", "0E1117")

# Generate new layout
new_layout = """
# ── SIDEBAR NAVIGATION ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('''
    <div class="ehr-header" style="margin-bottom:1rem;">
        <div class="gv-wordmark">🔬 GradVision</div>
        <div class="gv-tagline">Clinical Intelligence · v4.0</div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("### Navigation")
    nav_selection = st.radio(
        "Navigation", 
        ["Upload Images", "Model Information", "Batch Results", "Diagnostics", "Support"],
        label_visibility="collapsed"
    )

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown(f'''
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:1.8rem 0 1.4rem;border-bottom:1px solid var(--border);
     margin-bottom:1.8rem;">
    <div>
        <div class="hero-title" style="font-size:2.2rem;">GradVision – AI Breast Cancer Detection System</div>
        <div class="hero-sub" style="font-size:0.8rem; letter-spacing:1px;">AI-powered histopathology diagnostic assistant.</div>
    </div>
    <div style="text-align:right;font-family:var(--font-mono);font-size:0.65rem;
         color:var(--text-3);line-height:1.8;">
        <div><span style="color:var(--amber);">SRG-2024-1101</span></div>
        <div>{NOW}</div>
        <div style="color:var(--green);">
            <span class="status-dot dot-green"></span>SYSTEM ACTIVE</div>
    </div>
</div>
''', unsafe_allow_html=True)


# ── PAGE CONTENT ───────────────────────────────────────────────────────────────

if nav_selection == "Upload Images":
    st.container()
    st.markdown("### Patient Diagnostics Dashboard")
    
    top_c1, top_c2, top_c3 = st.columns(3, gap="large")
    
    with top_c1:
        st.markdown('<div class="section-label">📥 Upload Image</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload image",
            type=["jpg","jpeg","png","tif"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
    with top_c2:
        st.markdown('<div class="section-label">🔖 Case ID</div>', unsafe_allow_html=True)
        if uploaded_files:
            st.info("PT-2024-08863")
        else:
            st.markdown("<div style='padding:1rem;border:1px dashed var(--border-hi);border-radius:8px;text-align:center;color:var(--text-3);'>No Case Active</div>", unsafe_allow_html=True)
            
    with top_c3:
        st.markdown('<div class="section-label">⚙ Model Status</div>', unsafe_allow_html=True)
        is_available = nova.aws_available or nova.gemini_available
        nova_status = "ONLINE" if is_available else "OFFLINE"
        
        st.markdown(f'''
        <div style="padding:0.75rem;background:var(--surface-2);border-radius:var(--radius-sm);border-left:3px solid var(--green);">
            <div style="font-size:0.85rem;color:var(--text-1);"><strong>CNN Engine:</strong> LOADED</div>
            <div style="font-size:0.85rem;color:var(--text-1);margin-top:0.3rem;"><strong>Threshold:</strong> {threshold}</div>
            <div style="font-size:0.85rem;color:var(--text-1);margin-top:0.3rem;"><strong>Nova AI:</strong> {nova_status}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    
    if uploaded_files:
        is_batch_mode = len(uploaded_files) > 1
        
        if is_batch_mode:
            st.warning(f"Batch mode detected: {len(uploaded_files)} images. Please click analyze and view results in the 'Batch Results' tab.")
            if st.button("▶ ANALYZE BATCH", use_container_width=True):
                with st.spinner(f"Analyzing {len(uploaded_files)} specimens…"):
                    analyzer = BatchAnalyzer(model, IMG_SIZE, threshold)
                    batch_result = analyzer.analyze_batch(uploaded_files)
                    st.session_state.batch_result = batch_result
                    st.session_state.batch_done = True
                    st.session_state.done = False
                    st.session_state.pred = None
                    st.rerun()
                    
        else:
            uploaded_file = uploaded_files[0]
            if st.button("▶ START DIAGNOSTIC SCAN", use_container_width=True):
                with st.spinner("Processing specimen…"):
                    result = process_image(uploaded_file, uploaded_file.name, model, IMG_SIZE, threshold, nova)
                    if result is not None and isinstance(result, dict):
                        st.session_state.pred = result
                        st.session_state.done = True
                        st.session_state.batch_done = False
                    else:
                        st.warning("⚠️ Could not process image.")
                        st.session_state.done = False

    if st.session_state.done and st.session_state.pred:
        r = st.session_state.pred
        
        st.markdown("---")
        
        # PREDICTION RESULT CARDS
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            is_malignant = r.get("is_malignant", False)
            pred_label = "MALIGNANT" if is_malignant else "BENIGN"
            pred_color = "var(--red)" if is_malignant else "var(--green)"
            st.markdown(f'''
            <div style="background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1rem;text-align:center;">
                <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-3);letter-spacing:2px;margin-bottom:0.5rem;">PREDICTION</div>
                <div style="font-size:2rem;font-weight:900;color:{pred_color};">{pred_label}</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with rc2:
            conf_val = r.get("confidence", 0) * 100
            st.markdown(f'''
            <div style="background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1rem;text-align:center;">
                <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-3);letter-spacing:2px;margin-bottom:0.5rem;">CONFIDENCE</div>
                <div style="font-size:2rem;font-weight:900;color:var(--text-1);">{conf_val:.1f}%</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with rc3:
            severity = r.get("severity", "LOW RISK")
            sev_color = "🔴" if severity == "HIGH RISK" else "🟡" if severity == "MEDIUM RISK" else "🟢"
            txt_color = "var(--red)" if severity == "HIGH RISK" else "var(--amber)" if severity == "MEDIUM RISK" else "var(--green)"
            st.markdown(f'''
            <div style="background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1rem;text-align:center;">
                <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-3);letter-spacing:2px;margin-bottom:0.5rem;">SEVERITY LEVEL</div>
                <div style="font-size:2rem;font-weight:900;color:{txt_color};">{sev_color} {severity}</div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        
        # IMAGE VISUALIZATION PANEL (3 COLUMNS)
        st.markdown('<div class="section-label">👁 Image Visualization Panel</div>', unsafe_allow_html=True)
        img1, img2, img3 = st.columns(3)
        
        with img1:
            st.markdown('<div style="text-align:center;font-weight:bold;margin-bottom:0.5rem;">Original Image</div>', unsafe_allow_html=True)
            st.image(r.get('pil_img'), use_container_width=True)
            
        with img2:
            st.markdown('<div style="text-align:center;font-weight:bold;margin-bottom:0.5rem;">GradCAM Heatmap</div>', unsafe_allow_html=True)
            if r.get('figure'):
                st.image(r.get('figure'), use_container_width=True)
                
        with img3:
            st.markdown('<div style="text-align:center;font-weight:bold;margin-bottom:0.5rem;">Tumor Localization</div>', unsafe_allow_html=True)
            tumor_res = r.get('tumor_result', {})
            if tumor_res and tumor_res.get('tumor_box_image'):
                st.image(tumor_res['tumor_box_image'], use_container_width=True)
            else:
                st.info("No clearly defined tumor bounds isolated.")

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        
        # MORPHOLOGY SECTION
        with st.expander("🔬 Morphology Analysis", expanded=True):
            morph = r.get('morphology_result', {})
            if morph:
                m1, m2, m3 = st.columns(3)
                m1.metric("Texture Irregularity", morph.get('texture_irregularity', 0))
                m2.metric("Nucleus Density", morph.get('cell_density', 0))
                m3.metric("Tissue Uniformity", morph.get('tissue_uniformity', 0))
            else:
                st.warning("Morphology data unavailable.")
                
        # REPORT SECTION
        with st.expander("📄 AI Diagnostic Report", expanded=True):
            st.markdown(f'<div class="nova-block-text">{r.get("report", "No report available.")}</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            
            if st.button("Generate & Download PDF", key="pdf_dl"):
                with st.spinner("Generating PDF..."):
                    report_data = {
                        'prediction': r.get('label', 'Unknown'),
                        'confidence': r.get('confidence', 0.0),
                        'image_name': r.get('image_name', 'Specimen'),
                        'cell_count': morph.get('cell_count', 0) if morph else 0,
                        'cell_density': morph.get('cell_density', 0.0) if morph else 0.0,
                        'irregular_nuclei_ratio': morph.get('irregular_nuclei_ratio', 0.0) if morph else 0.0,
                        'cluster_count': morph.get('cluster_count', 0) if morph else 0,
                        'largest_cluster': morph.get('largest_cluster', 0) if morph else 0,
                        'suspicion_level': morph.get('suspicion_level', 'Unknown') if morph else 'Unknown'
                    }
                    pdf_bytes, json_str, status_msg = generate_medical_report(
                        report_data,
                        heatmap_img=r.get('figure'),
                        tumor_img=r.get('tumor_result', {}).get('tumor_box_image') if r.get('tumor_result') else None
                    )
                    st.session_state.pdf_bytes = pdf_bytes
                    st.success("Report generated!")
            
            if st.session_state.get('pdf_bytes'):
                st.download_button(
                    label="⬇️ Download PDF",
                    data=st.session_state.pdf_bytes,
                    file_name=f"AI_Pathology_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

elif nav_selection == "Batch Results":
    st.container()
    st.markdown("### Batch Results Section")
    
    if st.session_state.batch_done and st.session_state.batch_result:
        br = st.session_state.batch_result
        
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Total Images", br['total_count'])
        col_b2.metric("Malignant Count", br['malignant_count'])
        col_b3.metric("Benign Count", br['benign_count'])
        col_b4.metric("Consensus Result", br['final_diagnosis'])
        
        # Simple Chart
        st.markdown("#### Malignant vs Benign Ratio")
        chart_data = {"Malignant": br['malignant_count'], "Benign": br['benign_count']}
        st.bar_chart(chart_data)
        
        st.markdown("#### Detailed Results")
        for i, res in enumerate(br['results']):
            if res['status'] == 'success':
                st.write(f"**{res['file_name']}** - {res.get('label', 'UNKNOWN')} ({res.get('confidence', 0)*100:.1f}%)")

    else:
        st.info("No batch results available. Please upload multiple images in the 'Upload Images' section.")

elif nav_selection == "Model Information":
    st.container()
    st.markdown("### Model Information")
    st.write("**Model Architecture:** MobileNetV2 with Transfer Learning")
    st.write("**XAI Method:** Grad-CAM (Gradient-weighted Class Activation Mapping)")
    st.write("**Input Size:** 96×96 pixels")
    st.write("**Analysis Type:** Binary Classification (Benign vs Malignant)")
    st.code("Current Threshold: " + str(threshold))

elif nav_selection == "Diagnostics":
    st.container()
    st.markdown("### Advanced Diagnostics")
    st.info("Use the primary 'Upload Images' tab to view individual prediction panels.")
    if st.session_state.done and "wsi_result" in st.session_state.pred:
        wsi = st.session_state.pred["wsi_result"]
        st.write(f"WSI Total Patches: **{wsi['total_patches']}**")
        st.write(f"WSI Tumor Patches: **{wsi['tumor_patches']}**")
        if wsi.get("heatmap_image"):
            st.image(wsi["heatmap_image"], caption="Whole Slide Scan Heatmap")

elif nav_selection == "Support":
    st.container()
    st.markdown("### Clinician Support Desk")
    st.write("For automated analysis assistance, view the AI Diagnostic Report in the Upload page.")
    chat_box = st.container(height=400)
    for msg in st.session_state.chat:
        with chat_box.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if q := st.chat_input("Query Nova..."):
        st.session_state.chat.append({"role": "user", "content": q})
        st.rerun()

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='gv-footer'>"
    "For Research &amp; Demonstration Only &nbsp;·&nbsp; "
    "Not a Clinical Diagnostic Tool &nbsp;·&nbsp; "
    "Consult a Qualified Pathologist"
    "</div>",
    unsafe_allow_html=True)
"""

with open("d:/python_projects/breast-cancer-ai/app/update_script.py", "w", encoding="utf-8") as fw:
    pass

full_new_content = "".join(head_lines) + new_layout
with open(file_path, "w", encoding="utf-8") as f:
    f.write(full_new_content)

print(f"Successfully wrote {len(full_new_content)} characters to {file_path}")
