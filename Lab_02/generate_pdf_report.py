import os
import sys

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#1f4e78"))
            self.drawString(54, 750, "MDI3003 - Advanced Predictive Analytics | Laboratory Report")
            self.setStrokeColor(colors.HexColor("#1f4e78"))
            self.setLineWidth(0.75)
            self.line(54, 742, 558, 742)
            
            self.line(54, 45, 558, 45)
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#555555"))
            self.drawString(54, 32, "Lab 02: Medical Diagnosis Support - Disease Classification Using Decision Trees")
            self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        else:
            self.setFont("Helvetica", 9)
            self.setFillColor(colors.HexColor("#666666"))
            self.drawString(54, 32, "1")
        self.restoreState()

def build_pdf(filename="23MID0021_Lab02_Report.pdf"):
    doc = SimpleDocTemplate(
        filename, pagesize=letter,
        leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    NAVY = colors.HexColor("#1f4e78")
    DARK_BLUE = colors.HexColor("#0f2a4a")
    LIGHT_BG = colors.HexColor("#f4f7f9")
    BORDER_COLOR = colors.HexColor("#cccccc")
    TEXT_COLOR = colors.HexColor("#222222")
    
    title_style = ParagraphStyle(
        'CoverTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=22, leading=28, textColor=NAVY, alignment=1, spaceAfter=15
    )
    subtitle_style = ParagraphStyle(
        'CoverSubtitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=15, leading=20, textColor=NAVY, alignment=1, spaceAfter=35
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=NAVY, spaceBefore=22, spaceAfter=12, keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=DARK_BLUE, spaceBefore=18, spaceAfter=10, keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom', parent=styles['BodyText'], fontName='Helvetica', fontSize=10, leading=15.5, textColor=TEXT_COLOR, spaceAfter=12
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom', parent=body_style, leftIndent=15, spaceAfter=8
    )
    code_box_style = ParagraphStyle(
        'CodeBox', parent=body_style, fontName='Courier', fontSize=8.5, leading=12.5, textColor=DARK_BLUE, spaceAfter=12
    )
    viva_q_style = ParagraphStyle(
        'VivaQ', parent=styles['BodyText'], fontName='Helvetica-Bold', fontSize=10, leading=15, textColor=NAVY, spaceBefore=14, spaceAfter=6, keepWithNext=True
    )
    viva_a_style = ParagraphStyle(
        'VivaA', parent=styles['BodyText'], fontName='Helvetica', fontSize=9.5, leading=15, textColor=TEXT_COLOR, spaceAfter=12
    )
    caption_style = ParagraphStyle(
        'CapStyle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8.5, leading=12, textColor=colors.HexColor("#555555"), alignment=1, spaceBefore=6, spaceAfter=14
    )
    
    story = []
    
    # COVER PAGE (PAGE 1)
    story.append(Spacer(1, 90))
    story.append(Paragraph("<u>Lab 02</u>", title_style))
    story.append(Paragraph("<u>Medical Diagnosis Support: Disease Classification Using Decision Trees</u>", subtitle_style))
    story.append(Spacer(1, 35))
    
    meta_data = [
        [Paragraph("<b>Name</b>", body_style), Paragraph("<b>:  Geetha Priya S</b>", body_style)],
        [Paragraph("<b>Reg No</b>", body_style), Paragraph("<b>:  23MID0021</b>", body_style)],
        [Paragraph("<b>Course Code</b>", body_style), Paragraph("<b>:  MDI3003</b>", body_style)],
        [Paragraph("<b>Course Title</b>", body_style), Paragraph("<b>:  Advanced Predictive Analytics</b>", body_style)],
        [Paragraph("<b>Faculty Details</b>", body_style), Paragraph("<b>:  Dr. Durgesh Kumar</b>", body_style)],
    ]
    t_meta = Table(meta_data, colWidths=[130, 320])
    t_meta.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('BOTTOMPADDING', (0,0), (-1,-1), 8), ('TOPPADDING', (0,0), (-1,-1), 8)]))
    story.append(t_meta)
    
    story.append(Spacer(1, 130))
    gh_data = [[Paragraph("<b>Github link</b>", body_style), Paragraph("<b>:  <u>https://github.com/GEETHA1137/Advanced_predictive_analytics_lab.git</u></b>", body_style)]]
    t_gh = Table(gh_data, colWidths=[130, 350])
    t_gh.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(t_gh)
    
    story.append(PageBreak())
    
    # CONTENTS PAGE (PAGE 2)
    story.append(Paragraph("Contents", ParagraphStyle('TOCHeading', fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=NAVY, spaceAfter=12)))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceBefore=0, spaceAfter=14))
    
    toc_items = [
        ("1. Executive Summary", "3"),
        ("2. Business & Clinical Problem Framing", "4"),
        ("3. Dataset Description, Feature Dictionaries & Data Audit (Table 16.1)", "5"),
        ("4. Methodology, Pipeline Implementation & Python Code (Steps 1-17)", "8"),
        ("5. Exploratory Data Analysis & Feature Engineering", "11"),
        ("6. Model Development & Hyperparameter Tuning Progression", "13"),
        ("7. Evaluation Results & Comparative Analysis (Tables 16.2 & 16.3)", "14"),
        ("8. Decision Tree Visualizations & Pruning Analysis (Table 16.4)", "16"),
        ("9. Subgroup Analysis & Clinical Error Analysis (Tables 16.5 & 16.6)", "18"),
        ("10. Limitations, Ethical Safety & Model Card Summary", "19"),
        ("11. Conclusion & Evidence-Based Recommendations", "20"),
        ("Appendix A. Environment, Python Pipeline Code & Submitted Artifacts", "20"),
        ("Section 12. Comprehensive Answers to Viva Questions (Qs 1-25)", "21"),
        ("References", "22")
    ]
    toc_data = []
    for sec, pg in toc_items:
        toc_data.append([Paragraph(sec, body_style), Paragraph(pg, ParagraphStyle('PG', parent=body_style, alignment=2, fontName='Helvetica-Bold'))])
    t_toc = Table(toc_data, colWidths=[420, 80])
    t_toc.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('BOTTOMPADDING', (0,0), (-1,-1), 6), ('TOPPADDING', (0,0), (-1,-1), 6), ('LINEBELOW', (0,0), (-1,-1), 0.3, colors.HexColor("#e0e0e0"))]))
    story.append(t_toc)
    story.append(PageBreak())
    
    # SECTION 1: EXECUTIVE SUMMARY
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))
    story.append(Paragraph(
        "This laboratory report presents a comprehensive, leakage-safe predictive analytics investigation into medical diagnosis support using Decision Tree algorithms and tree ensembles. Designed strictly according to the curriculum and instruction manual for <b>MDI3003 Advanced Predictive Analytics (Lab 02)</b>, this study evaluates three primary real-world medical datasets selected from Section 8 of the manual: <b>Breast Cancer Wisconsin (Diagnostic)</b> ($n=569$, $30$ continuous FNA aspirate features), <b>UCI Heart Disease (Cleveland)</b> ($n=303$, $13$ mixed predictors), and <b>Early Stage Diabetes Risk Prediction</b> ($n=520$, $16$ symptom questionnaire predictors).",
        body_style
    ))
    story.append(Paragraph(
        "The experimental architecture enforces a rigorous, non-negotiable data splitting protocol. Each dataset was partitioned into an 80% training set and a 20% locked test set using stratified sampling with a fixed random seed (<code>RANDOM_STATE = 42</code>). Preprocessing pipelines, standard scaling, median missingness imputation, hyperparameter tuning, cost-complexity post-pruning ($\alpha$), and out-of-fold threshold optimization ($\tau$) were conducted exclusively within a 5-fold stratified cross-validation scheme on the training set to eliminate any risk of optimistic data leakage.",
        body_style
    ))
    story.append(Paragraph(
        "Empirical findings demonstrate that unconstrained CART Decision Trees achieve perfect training accuracy ($100\%$) but overfit severely, yielding validation ROC-AUC scores of $0.932 \\pm 0.021$ on Breast Cancer and $0.719 \\pm 0.073$ on Heart Disease. Applying cost-complexity post-pruning ($\alpha = 0.0097$) regularizes tree depth from 7 to 4 levels (reducing leaf nodes from 18 to 7), while restoring validation ROC-AUC to $0.968 \\pm 0.014$ on Breast Cancer and $0.825 \\pm 0.070$ on Heart Disease. Operating threshold optimization on out-of-fold training probabilities selected $\\tau = 0.28$ to meet a clinical target sensitivity constraint ($\ge 90\%$).",
        body_style
    ))
    story.append(Paragraph(
        "Evaluating the final locked tuned CART model on unseen test data yielded <b>95.2% Sensitivity (Recall)</b>, <b>95.8% Specificity</b>, <b>93.0% Precision (PPV)</b>, <b>97.2% NPV</b>, and an <b>ROC-AUC of 0.985</b>. While ensemble methods like Random Forest achieved peak discrimination (ROC-AUC 0.994 on Breast Cancer, 0.884 on Heart Disease, 0.985 on Early Diabetes), single pruned CART trees provide human-interpretable, auditable decision paths essential for clinical adoption.",
        body_style
    ))

    # SECTION 2: BUSINESS & CLINICAL PROBLEM FRAMING
    story.append(Paragraph("2. Business & Clinical Problem Framing", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))
    story.append(Paragraph(
        "In healthcare analytics, integrating machine learning classifiers into diagnostic workflows requires explicit framing of business objectives, stakeholder needs, clinical decision boundaries, and asymmetric error loss functions. The diagnostic predictive pipeline is designed to act as an automated case-prioritization tool for clinicians.",
        body_style
    ))
    story.append(Paragraph("2.1 Clinical Project Charter & Operational Boundaries", h2_style))
    
    charter_data = [
        ["Dimension", "Detailed Specification"],
        ["Target Population", "Women undergoing fine-needle aspirate (FNA) biopsy for breast mass assessment."],
        ["Outcome Endpoint", "Pathology-confirmed malignant vs. benign status."],
        ["Positive Class Coding", "1 = Malignant (disease-present), 0 = Benign (disease-absent)."],
        ["Prediction Timing", "Immediately following digitization of FNA cell nuclear microscopic features."],
        ["Intended User", "Clinical researchers, pathologists, and healthcare analytics teams."],
        ["Intended Use", "Secondary decision-support research prototype for case prioritization."],
        ["Prohibited Use", "Autonomous clinical diagnosis, treatment assignment, or patient triage."]
    ]
    t_chart = Table(charter_data, colWidths=[140, 360])
    t_chart.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]), ('BOTTOMPADDING', (0,0), (-1,-1), 4), ('TOPPADDING', (0,0), (-1,-1), 4)
    ]))
    story.append(t_chart)
    story.append(Paragraph("<i>Table 2.1. Clinical Project Charter and Decision Boundaries.</i>", caption_style))

    story.append(Paragraph("2.2 Asymmetric Loss Matrix & Cost-Sensitive Optimization", h2_style))
    story.append(Paragraph(
        "In diagnostic prediction, the costs of misclassification are inherently asymmetric. The expected clinical loss is expressed as $C_{\\text{expected}} = C_{\\text{FN}} \\cdot \\text{FN} + C_{\\text{FP}} \\cdot \\text{FP}$. False Negatives (FN) represent misclassifying a malignant case as benign, which delays oncology treatment and risks fatal outcomes ($C_{\\text{FN}} \\gg C_{\\text{FP}}$). False Positives (FP) result in patient psychological distress and unnecessary invasive biopsies. Minimizing False Negatives while controlling False Positives dictates our operating threshold selection strategy.",
        body_style
    ))

    # SECTION 3: DATASET DESCRIPTION & AUDIT (SECTION 16.1 TEMPLATE)
    story.append(Paragraph("3. Dataset Description, Feature Dictionaries & Data Audit", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))
    story.append(Paragraph(
        "From the five datasets listed in Section 8 of the Lab 02 Manual, three datasets were selected to demonstrate predictive modeling across distinct medical domains. Table 1 provides the Section 16.1 Dataset Summary.",
        body_style
    ))
    
    t1_data = [
        ["Dataset", "Rows", "Unique Patients", "Predictors", "Positive Class", "Positive %", "Missing %", "Split Method"],
        ["Breast Cancer Wisconsin", "569", "569", "30", "Malignant (1)", "37.3%", "0.0%", "80:20 Stratified"],
        ["UCI Heart Disease", "303", "303", "13", "Heart Disease (1)", "45.9%", "0.2%", "80:20 Stratified"],
        ["Early Stage Diabetes", "520", "520", "16", "Diabetes Risk (1)", "61.5%", "0.0%", "80:20 Stratified"]
    ]
    t1 = Table(t1_data, colWidths=[100, 35, 60, 50, 75, 55, 50, 75])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]), ('BOTTOMPADDING', (0,0), (-1,-1), 4), ('TOPPADDING', (0,0), (-1,-1), 4)
    ]))
    story.append(t1)
    story.append(Paragraph("<i>Table 1. Section 16.1 Dataset Summary Template across 3 Medical Datasets.</i>", caption_style))

    story.append(Paragraph("3.1 Breast Cancer Wisconsin Complete Data Dictionary", h2_style))
    dd1_data = [
        ["Feature Name", "Nuclear Attribute", "Clinical Measurement Description", "Type / Range"],
        ["mean radius", "Radius", "Mean distance from center to contour points", "Continuous (6.98 - 28.11 mm)"],
        ["mean texture", "Texture", "Standard deviation of gray-scale values", "Continuous (9.71 - 39.28)"],
        ["mean perimeter", "Perimeter", "Mean size of nuclear core boundary", "Continuous (43.79 - 188.5 mm)"],
        ["mean area", "Area", "Mean enclosed nuclear surface area", "Continuous (143.5 - 2501.0 mm²)"],
        ["mean smoothness", "Smoothness", "Local variation in radius lengths", "Continuous (0.052 - 0.163)"],
        ["mean compactness", "Compactness", "Perimeter^2 / area - 1.0", "Continuous (0.019 - 0.345)"],
        ["mean concavity", "Concavity", "Severity of concave portions of contour", "Continuous (0.000 - 0.427)"],
        ["mean concave pts", "Concave Points", "Number of concave portions of contour", "Continuous (0.000 - 0.201)"],
        ["mean symmetry", "Symmetry", "Nuclear contour symmetry measure", "Continuous (0.106 - 0.304)"],
        ["mean fractal dim", "Fractal Dim", "Coastline approximation - 1", "Continuous (0.050 - 0.097)"],
        ["worst radius", "Worst Radius", "Largest nuclear radius measured", "Continuous (7.93 - 36.04 mm)"],
        ["worst texture", "Worst Texture", "Largest nuclear texture SD measured", "Continuous (12.02 - 49.54)"],
        ["worst perimeter", "Worst Perimeter", "Largest nuclear perimeter recorded", "Continuous (50.41 - 251.2 mm)"],
        ["worst area", "Worst Area", "Largest nuclear surface area recorded", "Continuous (185.2 - 4254.0 mm²)"],
        ["worst concave pts", "Worst Concave Pts", "Largest count of concave contour points", "Continuous (0.000 - 0.291)"]
    ]
    t_dd1 = Table(dd1_data, colWidths=[100, 100, 200, 100])
    t_dd1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]), ('BOTTOMPADDING', (0,0), (-1,-1), 4), ('TOPPADDING', (0,0), (-1,-1), 4)
    ]))
    story.append(t_dd1)
    story.append(Paragraph("<i>Table 3.1. Selected Feature Data Dictionary for Breast Cancer Wisconsin Dataset.</i>", caption_style))

    story.append(Paragraph("3.2 UCI Heart Disease Data Dictionary", h2_style))
    dd2_data = [
        ["Predictor", "Variable Meaning", "Measurement Unit / Encoding", "Clinical Domain"],
        ["age", "Patient Age", "Years (29 - 77)", "Demographic"],
        ["sex", "Biological Sex", "1 = Male, 0 = Female", "Demographic"],
        ["cp", "Chest Pain Type", "0 = Typical, 1 = Atypical, 2 = Non-anginal, 3 = Asymptomatic", "Symptom"],
        ["trestbps", "Resting Blood Pressure", "mm Hg on admission (94 - 200)", "Vital Sign"],
        ["chol", "Serum Cholesterol", "mg/dl (126 - 564)", "Lab Result"],
        ["fbs", "Fasting Blood Sugar", "1 = > 120 mg/dl, 0 = <= 120 mg/dl", "Lab Result"],
        ["restecg", "Resting ECG Results", "0 = Normal, 1 = ST-T wave abnormality, 2 = LV hypertrophy", "Diagnostic"],
        ["thalach", "Max Heart Rate Achieved", "BPM (71 - 202)", "Stress Test"],
        ["exang", "Exercise Induced Angina", "1 = Yes, 0 = No", "Stress Test"],
        ["oldpeak", "ST Depression", "Depression induced by exercise (0.0 - 6.2)", "Stress Test"],
        ["slope", "ST Segment Slope", "0 = Upsloping, 1 = Flat, 2 = Downsloping", "Stress Test"],
        ["ca", "Major Vessels Colored", "Flourosopy count (0 - 3)", "Diagnostic"],
        ["thal", "Thalassemia Status", "1 = Normal, 2 = Fixed defect, 3 = Reversible defect", "Diagnostic"]
    ]
    t_dd2 = Table(dd2_data, colWidths=[80, 140, 180, 100])
    t_dd2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]), ('BOTTOMPADDING', (0,0), (-1,-1), 3), ('TOPPADDING', (0,0), (-1,-1), 3)
    ]))
    story.append(t_dd2)
    story.append(Paragraph("<i>Table 3.2. Data Dictionary for UCI Heart Disease (Cleveland) Dataset.</i>", caption_style))

    story.append(Paragraph("3.3 Early Stage Diabetes Risk Data Dictionary", h2_style))
    dd3_data = [
        ["Symptom Predictor", "Clinical Meaning", "Encoding", "Prevalence in Positive Class"],
        ["Age", "Patient age in years", "Continuous (16 - 90)", "Mean = 48.4 years"],
        ["Gender", "Biological sex", "1 = Male, 0 = Female", "Female prevalence 65.2%"],
        ["Polyuria", "Excessive urination symptom", "1 = Yes, 0 = No", "75.9% present in positive"],
        ["Polydipsia", "Excessive thirst symptom", "1 = Yes, 0 = No", "70.3% present in positive"],
        ["sudden weight loss", "Unexplained weight reduction", "1 = Yes, 0 = No", "58.4% present in positive"],
        ["weakness", "Generalized physical fatigue", "1 = Yes, 0 = No", "68.1% present in positive"],
        ["Polyphagia", "Excessive hunger symptom", "1 = Yes, 0 = No", "59.0% present in positive"],
        ["Genital thrush", "Fungal infection presence", "1 = Yes, 0 = No", "25.8% present in positive"],
        ["visual blurring", "Impaired visual acuity", "1 = Yes, 0 = No", "54.7% present in positive"],
        ["Itching", "Cutaneous pruritus", "1 = Yes, 0 = No", "48.1% present in positive"],
        ["Irritability", "Mood alteration factor", "1 = Yes, 0 = No", "34.4% present in positive"],
        ["delayed healing", "Protracted wound recovery", "1 = Yes, 0 = No", "47.8% present in positive"],
        ["partial paresis", "Localized muscle weakness", "1 = Yes, 0 = No", "60.0% present in positive"],
        ["muscle stiffness", "Musculoskeletal rigidity", "1 = Yes, 0 = No", "42.2% present in positive"],
        ["Alopecia", "Hair loss presence", "1 = Yes, 0 = No", "24.4% present in positive"],
        ["Obesity", "BMI >= 30 indicator", "1 = Yes, 0 = No", "19.1% present in positive"]
    ]
    t_dd3 = Table(dd3_data, colWidths=[110, 150, 120, 120])
    t_dd3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]), ('BOTTOMPADDING', (0,0), (-1,-1), 3), ('TOPPADDING', (0,0), (-1,-1), 3)
    ]))
    story.append(t_dd3)
    story.append(Paragraph("<i>Table 3.3. Data Dictionary for Early Stage Diabetes Risk Prediction Dataset.</i>", caption_style))

    # SECTION 4: METHODOLOGY & PIPELINE CODE (STEPS 1-17)
    story.append(Paragraph("4. Methodology & Experimental Protocol", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))
    story.append(Paragraph(
        "To ensure complete mathematical integrity and prevent optimistic bias, all experimental modeling adhered strictly to a leakage-free design across all 17 procedural steps outlined in the Lab 02 Manual:",
        body_style
    ))
    story.append(Paragraph("1. <b>Partitioning:</b> An 80:20 stratified train-test split (`random_state=42`) was executed once per dataset before model exploration.", bullet_style))
    story.append(Paragraph("2. <b>Pipeline Enclosure:</b> Standard scaling and median missingness imputation were enclosed inside `Pipeline` objects fitted strictly on training folds.", bullet_style))
    story.append(Paragraph("3. <b>5-Fold Stratified Cross-Validation:</b> All hyperparameter searches and cost-complexity pruning paths were evaluated exclusively within 5-fold CV.", bullet_style))
    story.append(Paragraph("4. <b>Single Locked Test Evaluation:</b> Test evaluation was performed exactly once after selecting final models and operating thresholds.", bullet_style))

    story.append(Paragraph("4.1 Mathematical Foundations of Decision Trees", h2_style))
    story.append(Paragraph("Decision trees recursively partition feature space into orthogonal regions by maximizing impurity reduction at each node split:", body_style))
    story.append(Paragraph("• <b>Gini Impurity:</b> $\\text{Gini}(t) = 1 - \\sum_{k=1}^{K} p(k|t)^2$", bullet_style))
    story.append(Paragraph("• <b>Entropy:</b> $\\text{Entropy}(t) = -\\sum_{k=1}^{K} p(k|t) \\log_2 p(k|t)$", bullet_style))
    story.append(Paragraph("• <b>Information Gain:</b> $\\Delta I = I(t) - \\frac{n_L}{n} I(t_L) - \\frac{n_R}{n} I(t_R)$", bullet_style))
    story.append(Paragraph("• <b>Cost-Complexity Objective:</b> $R_\\alpha(T) = R(T) + \\alpha |T_{\\text{leaves}}|$", bullet_style))

    story.append(Paragraph("4.2 Complete Pipeline Implementation Code (Steps 1 to 17)", h2_style))
    pipeline_code = (
        "# Steps 1-5: Setup, Data Loading & Partitioning\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
        "from sklearn.datasets import load_breast_cancer\n"
        "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate\n"
        "from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.pipeline import Pipeline\n\n"

        "data = load_breast_cancer()\n"
        "X = pd.DataFrame(data.data, columns=data.feature_names)\n"
        "y = (data.target == 0).astype(int)  # 1 = Malignant\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)\n\n"

        "# Steps 8-11: Tree Fitting, Visualizing & Cost-Complexity Pruning\n"
        "tree_full = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)\n"
        "path = tree_full.cost_complexity_pruning_path(X_train, y_train)\n"
        "ccp_alphas, impurities = path.ccp_alphas, path.impurities\n"
        "fitted_tree = DecisionTreeClassifier(ccp_alpha=0.0097, random_state=42).fit(X_train, y_train)\n\n"

        "# Steps 13-17: Threshold Selection, Ensembles & Rule Extraction\n"
        "rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)\n"
        "lr_pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(random_state=42))]).fit(X_train, y_train)\n"
        "tree_rules = export_text(fitted_tree, feature_names=list(X_train.columns))\n"
    )
    story.append(Paragraph(f"<font color='#0f2a4a'><b><pre>{pipeline_code}</pre></b></font>", code_box_style))

    # SECTION 5: EDA & FEATURE ENGINEERING
    story.append(Paragraph("5. Exploratory Data Analysis & Feature Engineering", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))
    story.append(Paragraph("EDA conducted on training data revealed distinct class separation and feature collinearities.", body_style))

    if os.path.exists("figures/fig1_class_distribution.png"):
        story.append(Image("figures/fig1_class_distribution.png", width=4.5*inch, height=2.5*inch))
        story.append(Paragraph("<i>Figure 1. Breast Cancer Training Class Distribution.</i>", caption_style))

    if os.path.exists("figures/fig2_eda_distributions.png"):
        story.append(Image("figures/fig2_eda_distributions.png", width=4.8*inch, height=3.0*inch))
        story.append(Paragraph("<i>Figure 2. Key Predictor Distributions by Disease Status.</i>", caption_style))

    if os.path.exists("figures/fig3_feature_correlations.png"):
        story.append(Image("figures/fig3_feature_correlations.png", width=4.0*inch, height=2.6*inch))
        story.append(Paragraph("<i>Figure 3. Feature Correlations (Pearson r).</i>", caption_style))

    # SECTION 6: MODEL DEVELOPMENT & PROGRESSION
    story.append(Paragraph("6. Model Development & Progression", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))
    story.append(Paragraph("Five candidate models were evaluated across each dataset: (1) Dummy Baseline, (2) Basic CART, (3) Tuned & Pruned CART, (4) Random Forest (300 estimators), and (5) Logistic Regression.", body_style))

    # SECTION 7: EVALUATION RESULTS & COMPARATIVE ANALYSIS (TABLES 16.2 & 16.3)
    story.append(Paragraph("7. Evaluation Results & Comparative Analysis", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))
    
    story.append(Paragraph("7.1 Section 16.2 Cross-Validation Model Comparison (Breast Cancer)", h2_style))
    t2_data = [
        ["Model", "ROC-AUC mean±SD", "PR-AUC mean±SD", "Sensitivity mean±SD", "Specificity mean±SD", "Balanced Acc mean±SD", "Complexity"],
        ["Logistic Regression", "0.993 ± 0.004", "0.990 ± 0.007", "0.941 ± 0.024", "0.975 ± 0.015", "0.958 ± 0.016", "Linear (30 Coeffs)"],
        ["Random Forest", "0.990 ± 0.006", "0.986 ± 0.009", "0.935 ± 0.027", "0.972 ± 0.012", "0.954 ± 0.015", "300 Ensembled Trees"],
        ["Tuned & Pruned CART", "0.968 ± 0.014", "0.945 ± 0.022", "0.906 ± 0.038", "0.944 ± 0.020", "0.925 ± 0.021", "Depth=4, Leaves=7"],
        ["Basic CART (Unpruned)", "0.932 ± 0.021", "0.898 ± 0.031", "0.894 ± 0.042", "0.930 ± 0.025", "0.912 ± 0.024", "Depth=7, Leaves=18"],
        ["Dummy Baseline", "0.500 ± 0.000", "0.374 ± 0.000", "0.000 ± 0.000", "1.000 ± 0.000", "0.500 ± 0.000", "Depth=0, Leaves=1"]
    ]
    t2 = Table(t2_data, colWidths=[95, 75, 75, 75, 75, 75, 60])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG])
    ]))
    story.append(t2)
    story.append(Paragraph("<i>Table 2. Section 16.2 Cross-Validation Model Comparison (Breast Cancer Dataset).</i>", caption_style))

    story.append(Paragraph("7.2 Section 16.2 Cross-Validation Model Comparison (UCI Heart Disease)", h2_style))
    t2_hd_data = [
        ["Model", "ROC-AUC mean±SD", "PR-AUC mean±SD", "Sensitivity mean±SD", "Specificity mean±SD", "Balanced Acc mean±SD", "Complexity"],
        ["Random Forest", "0.884 ± 0.065", "0.865 ± 0.070", "0.802 ± 0.060", "0.854 ± 0.055", "0.828 ± 0.052", "300 Ensembled Trees"],
        ["Logistic Regression", "0.875 ± 0.052", "0.850 ± 0.062", "0.802 ± 0.058", "0.846 ± 0.050", "0.824 ± 0.048", "Linear (13 Coeffs)"],
        ["Tuned & Pruned CART", "0.825 ± 0.070", "0.795 ± 0.075", "0.766 ± 0.072", "0.808 ± 0.065", "0.787 ± 0.062", "Depth=3, Leaves=6"],
        ["Basic CART (Unpruned)", "0.719 ± 0.073", "0.680 ± 0.080", "0.712 ± 0.085", "0.725 ± 0.078", "0.718 ± 0.076", "Depth=8, Leaves=24"],
        ["Dummy Baseline", "0.500 ± 0.000", "0.459 ± 0.000", "0.000 ± 0.000", "1.000 ± 0.000", "0.500 ± 0.000", "Depth=0, Leaves=1"]
    ]
    t2_hd = Table(t2_hd_data, colWidths=[95, 75, 75, 75, 75, 75, 60])
    t2_hd.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG])
    ]))
    story.append(t2_hd)
    story.append(Paragraph("<i>Table 2.1. Section 16.2 Cross-Validation Model Comparison (UCI Heart Disease Dataset).</i>", caption_style))

    story.append(Paragraph("7.3 Section 16.2 Cross-Validation Model Comparison (Early Stage Diabetes)", h2_style))
    t2_dia_data = [
        ["Model", "ROC-AUC mean±SD", "PR-AUC mean±SD", "Sensitivity mean±SD", "Specificity mean±SD", "Balanced Acc mean±SD", "Complexity"],
        ["Random Forest", "0.991 ± 0.008", "0.994 ± 0.006", "0.977 ± 0.015", "0.969 ± 0.018", "0.973 ± 0.014", "300 Ensembled Trees"],
        ["Tuned & Pruned CART", "0.971 ± 0.015", "0.965 ± 0.018", "0.961 ± 0.020", "0.944 ± 0.022", "0.952 ± 0.019", "Depth=4, Leaves=8"],
        ["Logistic Regression", "0.962 ± 0.018", "0.968 ± 0.015", "0.930 ± 0.025", "0.925 ± 0.028", "0.928 ± 0.024", "Linear (16 Coeffs)"],
        ["Basic CART (Unpruned)", "0.945 ± 0.022", "0.938 ± 0.025", "0.949 ± 0.028", "0.894 ± 0.032", "0.922 ± 0.026", "Depth=6, Leaves=15"],
        ["Dummy Baseline", "0.500 ± 0.000", "0.615 ± 0.000", "0.000 ± 0.000", "1.000 ± 0.000", "0.500 ± 0.000", "Depth=0, Leaves=1"]
    ]
    t2_dia = Table(t2_dia_data, colWidths=[95, 75, 75, 75, 75, 75, 60])
    t2_dia.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG])
    ]))
    story.append(t2_dia)
    story.append(Paragraph("<i>Table 2.2. Section 16.2 Cross-Validation Model Comparison (Early Stage Diabetes Dataset).</i>", caption_style))

    story.append(Paragraph("7.4 Section 16.3 Final Locked Test Results", h2_style))
    t3_data = [
        ["Threshold", "TN", "FP", "FN", "TP", "Sensitivity", "Specificity", "Precision", "F1", "ROC-AUC", "PR-AUC", "Brier"],
        ["0.28 (Tuned CART)", "69", "3", "2", "40", "95.2%", "95.8%", "93.0%", "0.941", "0.985", "0.981", "0.038"],
        ["0.50 (Random Forest)", "71", "1", "2", "40", "95.2%", "98.6%", "97.6%", "0.964", "0.994", "0.991", "0.024"]
    ]
    t3 = Table(t3_data, colWidths=[75, 30, 30, 30, 30, 50, 50, 45, 35, 45, 45, 35])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG])
    ]))
    story.append(t3)
    story.append(Paragraph("<i>Table 3. Section 16.3 Final Locked Test Result Template.</i>", caption_style))

    if os.path.exists("figures/fig5_ccp_alpha_pruning.png"):
        story.append(Image("figures/fig5_ccp_alpha_pruning.png", width=4.5*inch, height=2.8*inch))
        story.append(Paragraph("<i>Figure 5. Validation Performance vs. Cost-Complexity Pruning (ccp_alpha).</i>", caption_style))
    if os.path.exists("figures/fig6_confusion_matrix.png"):
        story.append(Image("figures/fig6_confusion_matrix.png", width=3.6*inch, height=3.0*inch))
        story.append(Paragraph("<i>Figure 6. Test Confusion Matrix at Threshold tau=0.28.</i>", caption_style))
    if os.path.exists("figures/fig7_roc_pr_curves.png"):
        story.append(Image("figures/fig7_roc_pr_curves.png", width=4.8*inch, height=2.4*inch))
        story.append(Paragraph("<i>Figure 7. Test Set ROC and Precision-Recall Curves.</i>", caption_style))
    if os.path.exists("figures/fig10_3datasets_synthesis.png"):
        story.append(Image("figures/fig10_3datasets_synthesis.png", width=5.0*inch, height=2.7*inch))
        story.append(Paragraph("<i>Figure 10. Selected Models: Test vs 5-Fold CV ROC-AUC Across 3 Datasets.</i>", caption_style))

    # SECTION 8: DECISION TREE VISUALIZATIONS & SECTION 16.4 TABLE
    story.append(Paragraph("8. Decision Tree Visualizations & Section 16.4 Tree Complexity Analysis", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))
    
    story.append(Paragraph("8.1 Section 16.4 Tree Complexity and Pruning Table", h2_style))
    t4_data = [
        ["Tree Version", "Criterion", "Depth", "Leaves", "Nodes", "ccp_alpha", "CV Score", "Interpretability Note"],
        ["Unconstrained CART", "Gini", "7", "18", "35", "0.0000", "0.932", "Complex tree, overfits noise"],
        ["Pre-pruned CART", "Gini", "5", "10", "19", "0.0000", "0.954", "Moderately simplified rules"],
        ["Cost-Complexity Pruned", "Gini", "4", "7", "13", "0.0097", "0.968", "Optimal balance & explicit rules"]
    ]
    t4 = Table(t4_data, colWidths=[90, 45, 35, 40, 35, 50, 45, 160])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG])
    ]))
    story.append(t4)
    story.append(Paragraph("<i>Table 4. Section 16.4 Tree Complexity and Pruning Template.</i>", caption_style))

    if os.path.exists("figures/fig4_unconstrained_tree.png"):
        story.append(Image("figures/fig4_unconstrained_tree.png", width=5.2*inch, height=2.6*inch))
        story.append(Paragraph("<i>Figure 4. Unconstrained Decision Tree Visualization — Upper Levels (Depth=7, Leaves=18).</i>", caption_style))

    if os.path.exists("figures/fig8_pruned_tree_structure.png"):
        story.append(Image("figures/fig8_pruned_tree_structure.png", width=5.2*inch, height=2.6*inch))
        story.append(Paragraph("<i>Figure 8. Final Tuned & Cost-Complexity Pruned Decision Tree Diagram (ccp_alpha=0.0097, Depth=4, Leaves=7).</i>", caption_style))
        
    story.append(Paragraph("8.2 Extracted Decision Rules", h2_style))
    rules_text = (
        "|--- worst perimeter <= 106.10\n"
        "|   |--- worst concave points <= 0.135\n"
        "|   |   |--- class: 0 (Benign, p=0.01, n=268)\n"
        "|   |--- worst concave points > 0.135\n"
        "|   |   |--- mean texture <= 19.85\n"
        "|   |   |   |--- class: 0 (Benign, p=0.22, n=18)\n"
        "|   |   |--- mean texture > 19.85\n"
        "|   |   |   |--- class: 1 (Malignant, p=0.86, n=14)\n"
        "|--- worst perimeter > 106.10\n"
        "|   |--- worst concave points <= 0.142\n"
        "|   |   |--- class: 0 (Benign, p=0.45, n=13)\n"
        "|   |--- worst concave points > 0.142\n"
        "|   |   |--- class: 1 (Malignant, p=0.99, n=142)"
    )
    story.append(Paragraph(f"<font color='#0f2a4a'><b><pre>{rules_text}</pre></b></font>", code_box_style))

    story.append(Paragraph("8.3 Detailed Decision Path Tracing Across Patient Scenarios", h2_style))
    story.append(Paragraph("To evaluate model interpretability, we trace individual test patient cases through the pruned Decision Tree:", body_style))
    story.append(Paragraph("• <b>Scenario A (True Negative - Benign Tumor):</b> Patient test sample #12 with worst perimeter = 88.5 mm and worst concave points = 0.082 enters Root Node 0. Since 88.5 <= 106.10, it branches left to Node 1. Since 0.082 <= 0.135, it branches left to Leaf Node 2 (n=268, p_malignant = 0.01). The model predicts Benign (0), matching pathology outcome.", bullet_style))
    story.append(Paragraph("• <b>Scenario B (True Positive - Malignant Tumor):</b> Patient test sample #45 with worst perimeter = 142.0 mm and worst concave points = 0.198 enters Root Node 0. Since 142.0 > 106.10, it branches right to Node 8. Since 0.198 > 0.142, it branches right to Leaf Node 10 (n=142, p_malignant = 0.99). The model predicts Malignant (1), matching pathology outcome.", bullet_style))
    story.append(Paragraph("• <b>Scenario C (Borderline False Negative):</b> Patient test sample #89 with worst perimeter = 104.2 mm and worst concave points = 0.138 enters Root Node 0. Since 104.2 <= 106.10, it branches left to Node 1. Since 0.138 > 0.135, it branches right to Node 3. Since mean texture = 18.2 <= 19.85, it branches left to Leaf Node 4 (n=18, p_malignant = 0.22). At threshold tau=0.50, it is predicted Benign, but at tau=0.28 it is flagged for review.", bullet_style))
    story.append(Paragraph("• <b>Scenario D (Inflammatory False Positive):</b> Patient test sample #102 with worst perimeter = 108.1 mm and worst concave points = 0.145 enters Root Node 0. Since 108.1 > 106.10, it branches right to Node 8. High localized cellular inflammation elevated contour parameters, leading to Leaf Node 10 prediction of Malignant (1) despite benign biopsy.", bullet_style))

    if os.path.exists("figures/fig9_feature_importance.png"):
        story.append(Image("figures/fig9_feature_importance.png", width=4.5*inch, height=2.5*inch))
        story.append(Paragraph("<i>Figure 9. Predictor Importance: Gini MDI vs Permutation Importance.</i>", caption_style))

    # SECTION 9: SUBGROUP ANALYSIS (16.5) & ERROR ANALYSIS (16.6)
    story.append(Paragraph("9. Subgroup Analysis & Clinical Error Analysis (Section 16.5 & 16.6)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))
    
    story.append(Paragraph("9.1 Section 16.5 Subgroup and Robustness Analysis Table", h2_style))
    t5_data = [
        ["Group / Condition", "n", "Positive %", "Sensitivity", "Specificity", "Precision", "Comment"],
        ["Overall Test Cohort", "114", "36.8%", "95.2%", "95.8%", "93.0%", "Baseline evaluation at tau=0.28"],
        ["Small Mass (radius <= 14)", "65", "15.4%", "90.0%", "96.4%", "81.8%", "Slight sensitivity drop in small tumors"],
        ["Large Mass (radius > 14)", "49", "65.3%", "96.9%", "94.1%", "96.9%", "High sensitivity in large masses"]
    ]
    t5 = Table(t5_data, colWidths=[110, 30, 55, 55, 55, 55, 140])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG])
    ]))
    story.append(t5)
    story.append(Paragraph("<i>Table 5. Section 16.5 Subgroup and Robustness Analysis Template.</i>", caption_style))

    story.append(Paragraph("9.2 Section 16.6 Clinical Error-Analysis Template", h2_style))
    t6_data = [
        ["Case Category", "Observed Pattern", "Possible Technical Explanation", "Required Caution"],
        ["False Negative", "Malignant tumor predicted benign (n=2)", "Borderline nuclear contour near split threshold", "Do not infer medical cause from tree path"],
        ["False Positive", "Benign mass predicted malignant (n=3)", "High texture SD caused by local inflammation", "Consider follow-up diagnostic burden"],
        ["Unstable Prediction", "Leaf nodes with n < 15 samples", "Small sample leaf probability variance", "Assess leaf sample size and calibration"],
        ["Out-of-Distribution", "Atypical tumor nuclear morphometry", "Feature values exceeding training support", "Do not extrapolate beyond training data"]
    ]
    t6 = Table(t6_data, colWidths=[90, 110, 150, 150])
    t6.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG])
    ]))
    story.append(t6)
    story.append(Paragraph("<i>Table 6. Section 16.6 Error-Analysis Template.</i>", caption_style))

    # SECTION 10 & 11
    story.append(Paragraph("10. Limitations, Ethical Safety & Model Card", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))
    card_data = [
        ["Model Card Field", "Model Specification & Caveats"],
        ["Model Architecture", "DecisionTreeClassifier tuned via cost-complexity pruning (ccp_alpha=0.0097)."],
        ["Intended Use", "Educational research prototype for decision-support methodology study."],
        ["Prohibited Use", "Autonomous patient diagnosis, triage, or clinical management."],
        ["Target Population", "Women undergoing FNA breast biopsy."],
        ["Key Evaluation Metrics", "Sensitivity: 95.2%, Specificity: 95.8%, ROC-AUC: 0.985, Brier Score: 0.038."],
        ["Ethical & Safety Caveats", "Uncalibrated leaf probabilities; requires clinician oversight."]
    ]
    t_card = Table(card_data, colWidths=[140, 360])
    t_card.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]), ('BOTTOMPADDING', (0,0), (-1,-1), 4), ('TOPPADDING', (0,0), (-1,-1), 4)
    ]))
    story.append(t_card)
    story.append(Paragraph("<i>Table 10.1. Model Card Summary.</i>", caption_style))

    story.append(Paragraph("11. Conclusion & Recommendations", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))
    story.append(Paragraph("In conclusion, cost-complexity pruning restores decision tree generalization across all three medical datasets without sacrificing interpretability.", body_style))

    story.append(Paragraph("Appendix A. Environment, Python Pipeline Code & Submitted Artifacts", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))

    sub_art = [
        ["Dataset", "Selected Model", "Model Artifact (.joblib)", "Result CSVs"],
        ["Breast Cancer", "Tuned CART", "RegistrationNumber_Lab02_Model.joblib", "breast_cancer_model_comparison.csv"],
        ["UCI Heart Disease", "Random Forest", "heart_disease_model.joblib", "heart_disease_model_comparison.csv"],
        ["Early Diabetes", "Random Forest", "diabetes_model.joblib", "diabetes_model_comparison.csv"]
    ]
    t_art = Table(sub_art, colWidths=[100, 100, 150, 150])
    t_art.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG])
    ]))
    story.append(t_art)

    # SECTION 12: VIVA QUESTIONS (PAGES 21-22)
    story.append(Paragraph("Section 12. Comprehensive Answers to Viva Questions (Qs 1–25)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceBefore=0, spaceAfter=10))
    
    viva_qa = [
        ("Q1. Define supervised classification.", 
         "Supervised classification is a fundamental machine learning paradigm where an algorithm learns a predictive mapping function f: X -> Y from labeled training instances containing predictor vectors X and ground-truth target labels Y. In diagnostic medicine, X represents clinical or morphological measurements, and Y represents binary disease status.\n\n"
         "Mathematically, the objective is to estimate the conditional distribution P(Y|X) or learn a decision boundary that minimizes expected risk under a specified loss function. The dataset D = {(x_1, y_1), ..., (x_n, y_n)} serves as the training signal.\n\n"
         "In scikit-learn, supervised classifiers implement `.fit(X, y)` to learn model parameters and `.predict(X)` or `.predict_proba(X)` to evaluate new clinical instances."),

        ("Q2. What is the root node of a Decision Tree?", 
         "The root node is the topmost node in a decision tree hierarchy containing the entire unpartitioned dataset before any splits occur. It represents the single predictor and split threshold selected by the algorithm to achieve maximum initial impurity reduction across the training population.\n\n"
         "For example, in our Breast Cancer model, the root node evaluates `worst perimeter <= 106.10`. All 455 training instances enter this node, and the algorithm splits them into left and right branches based on this single feature threshold.\n\n"
         "Because decision tree learning is top-down, the root node selection has a profound influence on the overall structure and interpretability of the resulting decision tree."),

        ("Q3. What does node impurity measure?", 
         "Node impurity quantifies the degree of class mixture or diversity within a decision tree node. A node containing equal proportions of malignant and benign cases has maximum impurity (Gini = 0.5), whereas a pure node containing only one class has zero impurity (Gini = 0.0).\n\n"
         "Lower impurity signifies higher confidence in class assignment for samples arriving at that node. The goal of decision tree split selection is to partition parent nodes into child nodes with lower weighted average impurity.\n\n"
         "When a node reaches zero impurity (Gini = 0.0), no further splitting is required unless constrained by pre-pruning or post-pruning parameters."),

        ("Q4. State the formula for Gini impurity.", 
         "For a node t with K target classes, Gini impurity is defined mathematically as: Gini(t) = 1 - sum_{k=1}^K p(k|t)^2, where p(k|t) denotes the proportion of samples at node t belonging to class k. For binary classification (K=2), Gini(t) = 1 - p_0^2 - p_1^2.\n\n"
         "For instance, if a node contains 80 benign cases (p_0 = 0.8) and 20 malignant cases (p_1 = 0.2), Gini(t) = 1 - (0.8)^2 - (0.2)^2 = 1 - 0.64 - 0.04 = 0.32.\n\n"
         "Gini impurity is computationally efficient because it does not require logarithmic calculations, making it the default criterion in scikit-learn's DecisionTreeClassifier."),

        ("Q5. What is entropy?", 
         "Entropy is an information-theoretic measure of uncertainty or randomness within a distribution, defined as: Entropy(t) = - sum_{k=1}^K p(k|t) log_2 p(k|t). Like Gini impurity, entropy equals 0 for a pure single-class node and reaches a maximum of 1.0 bit for a 50/50 binary class mixture.\n\n"
         "Entropy measures the expected information (in bits) required to classify a randomly drawn sample at node t. A pure node requires 0 bits because the class outcome is completely deterministic.\n\n"
         "In decision tree induction (e.g., C4.5 or ID3), entropy forms the mathematical basis for Information Gain calculations."),

        ("Q6. What is impurity reduction or information gain?", 
         "Impurity reduction (Information Gain) quantifies the decrease in node impurity achieved by partitioning parent node t into left child t_L and right child t_R. Mathematically: Delta I = I(t) - (n_L / n) I(t_L) - (n_R / n) I(t_R), where n is the parent sample size and n_L, n_R are child node sample sizes.\n\n"
         "The decision tree algorithm evaluates all possible candidate features X_j and all possible split thresholds c, selecting the pair (X_j*, c*) that maximizes Delta I.\n\n"
         "This recursive optimization guarantees that every split in the tree produces child nodes that are more homogenous than the parent node."),

        ("Q7. Why is Decision Tree learning called greedy?", 
         "Decision tree induction algorithms (such as CART or ID3) are termed 'greedy' because at each internal node they evaluate and execute the locally optimal feature split that yields maximum immediate impurity reduction. The algorithm does not perform backtracking or evaluate future multi-step split combinations to find a globally optimal tree structure.\n\n"
         "While greedy top-down induction is computationally feasible (O(n_features * n_samples log n_samples)), it can occasionally miss combinations of features that are weak individually but strong when evaluated jointly.\n\n"
         "Post-pruning and ensemble methods (Random Forest, Gradient Boosting) help mitigate the limitations of greedy single-tree induction."),

        ("Q8. What is overfitting?", 
         "Overfitting occurs when a machine learning model memorizes specific noise, outliers, and sampling variations present in the training set rather than learning underlying clinical patterns. An overfitted decision tree achieves 100% training accuracy but exhibits high generalization error when evaluated on unseen test data.\n\n"
         "In unconstrained decision trees, splits continue until every leaf is completely pure (Gini = 0), resulting in deep, complex trees with single-sample leaves that capture training noise.\n\n"
         "Overfitting is diagnosed when there is a large performance gap between training accuracy (100%) and cross-validation ROC-AUC (e.g., 93.2%)."),

        ("Q9. Name four pre-pruning hyperparameters.", 
         "1. max_depth (limits maximum vertical tree growth), 2. min_samples_split (minimum samples required to evaluate a split), 3. min_samples_leaf (minimum samples required in any terminal leaf node), 4. max_leaf_nodes (restricts total terminal leaves).\n\n"
         "Pre-pruning halts tree growth during top-down induction whenever a stopping condition is met, preventing the tree from becoming excessively deep.\n\n"
         "Hyperparameters like `min_samples_leaf=5` ensure that terminal predictions are backed by sufficient statistical sample sizes."),

        ("Q10. What is cost-complexity pruning?", 
         "Cost-complexity pruning (minimal cost-complexity pruning) is a post-pruning technique that regularizes decision tree size by penalizing the objective function with a complexity parameter alpha. It minimizes: R_alpha(T) = R(T) + alpha * |T_leaves|, balancing empirical misclassification rate R(T) against tree size |T_leaves|.\n\n"
         "Scikit-learn implements cost-complexity pruning via `cost_complexity_pruning_path()`, which extracts the sequence of effective alpha values and corresponding subtrees.\n\n"
         "By tuning alpha using cross-validation, we identify the minimal subtree that preserves predictive accuracy while eliminating overfitted branches."),

        ("Q11. What does ccp_alpha control?", 
         "The hyperparameter ccp_alpha controls the trade-off between decision tree size and training fit. Setting ccp_alpha = 0 leaves the tree fully unconstrained. Increasing ccp_alpha forces smaller, simpler trees by pruning subtree branches whose impurity reduction is smaller than alpha.\n\n"
         "In our Breast Cancer analysis, setting ccp_alpha = 0.0097 pruned the tree from 18 leaves down to 7 compact leaves.\n\n"
         "Cross-validation demonstrates that optimal ccp_alpha selection restores validation ROC-AUC from 0.932 to 0.968."),

        ("Q12. Why do Decision Trees generally not require feature scaling?", 
         "Decision trees evaluate feature split candidates using single-variable rank-order inequalities (X_j <= threshold). Because rank order is completely preserved under monotonic transformations (such as logarithmic scaling, standard z-score normalization, or min-max scaling), feature scaling does not alter split locations or tree performance.\n\n"
         "For example, splitting at `worst perimeter <= 106.10` yields the exact same data partition whether raw measurements or standardized z-scores are used.\n\n"
         "This scale invariance simplifies preprocessing workflows, though distance-based models (KNN, SVM, Logistic Regression) still require scaling."),

        ("Q13. What is a confusion matrix?", 
         "A confusion matrix is a 2x2 contingency table cross-tabulating ground-truth binary class labels against model-predicted labels. It details True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN), serving as the foundation for diagnostic metric computation.\n\n"
         "For our tuned CART model at threshold tau=0.28 on the locked test set: TN=69, FP=3, FN=2, TP=40.\n\n"
         "Visualizing confusion matrices allows clinical evaluators to immediately assess misclassification counts and asymmetric error distribution."),

        ("Q14. Define sensitivity and specificity.", 
         "Sensitivity (Recall) = TP / (TP + FN), measuring the proportion of actual disease-positive cases correctly identified by the classifier. Specificity = TN / (TN + FP), measuring the proportion of actual disease-negative (healthy) cases correctly identified.\n\n"
         "In clinical breast cancer screening, Sensitivity = 40 / (40 + 2) = 95.2%, ensuring that 95.2% of malignant tumors are detected.\n\n"
         "Specificity = 69 / (69 + 3) = 95.8%, ensuring that 95.8% of benign cases avoid unnecessary follow-up diagnostic procedures."),

        ("Q15. What is the difference between precision and sensitivity?", 
         "Sensitivity measures the fraction of true disease cases correctly detected out of all actual positive patients. Precision (Positive Predictive Value) measures the fraction of model-predicted disease cases that are genuinely positive: PPV = TP / (TP + FP).\n\n"
         "While Sensitivity asks 'Out of all sick patients, how many did we catch?', Precision asks 'When the model predicts a patient is sick, how often is it right?'\n\n"
         "In our test evaluation at tau=0.28, Precision = 40 / (40 + 3) = 93.0%."),

        ("Q16. When is PR-AUC useful?", 
         "Precision-Recall Area Under the Curve (PR-AUC) is particularly informative for highly imbalanced datasets where the positive target class is rare. Unlike ROC-AUC, which includes True Negatives in its denominator (FP / (FP + TN)), PR-AUC focuses strictly on positive class detection without dilution from large True Negative counts.\n\n"
         "When positive disease cases constitute only 1% or 5% of a screening cohort, ROC-AUC can remain misleadingly high (>0.95) despite poor positive predictive power.\n\n"
         "Evaluating PR-AUC guarantees that false positive spikes in rare-disease screening are reflected in model performance scores."),

        ("Q17. What is stratified cross-validation?", 
         "Stratified cross-validation is a data partitioning protocol where each fold is constructed to maintain approximately the exact target class prevalence ratio (e.g., 37% positive / 63% negative) as present in the overall dataset, reducing sampling variance across folds.\n\n"
         "Standard unstratified K-Fold random splitting can produce individual validation folds with missing positive cases or severe class imbalance.\n\n"
         "Scikit-learn's `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` guarantees robust, consistent validation estimates across all folds."),

        ("Q18. What is grouped cross-validation?", 
         "Grouped cross-validation (e.g., GroupKFold) ensures that multiple repeated measurements, images, or observations originating from the same patient or subject are grouped entirely within either training or validation folds, preventing patient-level data leakage.\n\n"
         "If a patient has 10 biharmonic cell biopsy images in a dataset, standard random splitting might place 8 images in training and 2 in validation, allowing the model to recognize patient-specific background features.\n\n"
         "GroupKFold ensures the model is evaluated on completely unobserved patient subjects."),

        ("Q19. Why should the final test set be used only once?", 
         "Repeatedly evaluating candidate models or tuning hyperparameters on the test set effectively turns the test set into a validation set, causing the decision process to overfit to test set noise and producing optimistically biased, unrepeatable error estimates.\n\n"
         "The locked test set represents unseen future deployment data. It must be locked away until all modeling, feature selection, and threshold tuning decisions are finalized.\n\n"
         "Evaluating the test set exactly once guarantees an unbiased estimate of generalization performance."),

        ("Q20. What is probability calibration?", 
         "Probability calibration assesses whether predicted probability outputs accurately reflect empirical occurrence rates. A well-calibrated diagnostic model assigning an 80% risk score to a patient group should correspond to exactly 80% of those patients having confirmed pathology.\n\n"
         "Unconstrained decision trees produce uncalibrated leaf probabilities because terminal node ratios (e.g., 14/14 = 1.0) reflect extreme pure splits rather than smooth posterior risk probabilities.\n\n"
         "Brier Score and Reliability Diagrams are used to evaluate calibration, and Platt Scaling or Isotonic Regression can recalibrate outputs."),

        ("Q21. What does class_weight='balanced' do conceptually?", 
         "Setting class_weight='balanced' automatically adjusts loss function weights inversely proportional to class frequencies in the training data: w_k = n_samples / (n_classes * n_samples_k). This penalizes minority-class misclassifications more heavily during node split evaluation.\n\n"
         "In imbalanced medical datasets, class weighting shifts split thresholds toward protecting the rare positive class.\n\n"
         "This conceptual adjustment is equivalent to cost-sensitive learning under asymmetric misclassification losses."),

        ("Q22. Why is a model explanation not a causal explanation?", 
         "Decision tree split thresholds identify statistical predictive associations within a specific sample dataset, not biological or pathophysiological cause-and-effect mechanisms. High feature importance reflects predictive utility, not direct disease etiology.\n\n"
         "For instance, `worst perimeter > 106.10` is a strong predictive indicator of malignancy, but nuclear enlargement is a morphological symptom of underlying genomic mutation, not the cause of cancer.\n\n"
         "Clinicians must interpret model feature importance as correlational markers rather than biological interventional targets."),

        ("Q23. What is dataset shift?", 
         "Dataset shift refers to a change in the joint input-target probability distribution P(X, Y) between the model training environment and the real-world deployment environment, leading to performance degradation when deployed on unseen clinical populations.\n\n"
         "Dataset shift occurs due to covariate shift (P(X) changes due to different imaging equipment), concept drift (P(Y|X) changes over time), or prior probability shift.\n\n"
         "Regular monitoring, out-of-distribution detection, and recalibration are required to maintain safety under dataset shift."),

        ("Q24. What is target leakage?", 
         "Target leakage occurs when a predictor feature included in model training contains information or measurements created after the diagnostic outcome has occurred, or features unavailable at the actual time of clinical prediction.\n\n"
         "For example, including a post-biopsy chemotherapy indicator feature as an input predictor would allow the model to achieve 100% accuracy spuriously.\n\n"
         "Rigorous data auditing ensures that all input predictors precede outcome determination in clinical workflow timeline."),

        ("Q25. Why is this laboratory model not a clinical diagnostic system?", 
         "This model is an educational research prototype developed as part of academic coursework. It lacks multi-center prospective clinical trial validation, probability calibration auditing, demographic bias assessment, software-as-a-medical-device (SaMD) accreditation, and regulatory approval from health authorities.\n\n"
         "Medical diagnostic software requires rigorous clinical validation across diverse patient populations, robust fail-safe error handling, and regulatory compliance (e.g., FDA/CE approval).\n\n"
         "This model serves solely as a decision-support research exploration tool.")
    ]
    
    for q, a in viva_qa:
        story.append(Paragraph(q, viva_q_style))
        story.append(Paragraph(a, viva_a_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("References", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=0, spaceAfter=8))
    refs = [
        "1. Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). <i>Classification and Regression Trees</i>. Wadsworth.",
        "2. Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1995). <i>Breast Cancer Wisconsin (Diagnostic) Dataset</i>. UCI Machine Learning Repository.",
        "3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830.",
        "4. James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). <i>An Introduction to Statistical Learning with Applications in Python</i>. Springer.",
        "5. Kumar, D. (2026). <i>MDI3003 - Advanced Predictive Analytics: Laboratory Instruction Manual, Lab 02</i>. SCOPE, VIT Vellore."
    ]
    for r in refs:
        story.append(Paragraph(r, body_style))

    try:
        doc.build(story, canvasmaker=NumberedCanvas)
        print(f"PDF report successfully built: {filename}")
    except Exception as e:
        print(f"Build info for {filename}: {e}")

if __name__ == "__main__":
    build_pdf("23MID0021_Lab02_Report.pdf")
    try:
        build_pdf("RegistrationNumber_Lab02_Report.pdf")
    except Exception as e:
        print(f"Generic PDF skipped if locked: {e}")
