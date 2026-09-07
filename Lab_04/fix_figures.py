"""
Fix: replace [Figure: X] placeholder paragraphs in the docx with actual embedded images.
Keeps all existing content/formatting; only swaps the placeholder runs for picture runs.
"""
import os, copy
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import win32com.client

BASE      = r'c:\Users\I768951\OneDrive - SAP SE\SAP\Exam'
DOCX_PATH = os.path.join(BASE, '23MID0021_Lab04_Report.docx')
PDF_PATH  = os.path.join(BASE, '23MID0021_Lab04_Report.pdf')
FIG_DIR   = os.path.join(BASE, 'lab04_figs')

# Map placeholder alt-text → figure file
PLACEHOLDER_MAP = {
    'Class Frequency Distribution':    os.path.join(FIG_DIR, 'fig1_class_dist.png'),
    'Age Distribution by Segment':     os.path.join(FIG_DIR, 'fig2_age_dist.png'),
    'Categorical Features by Segment': os.path.join(FIG_DIR, 'fig3_categorical.png'),
    'Missing Value Map':               os.path.join(FIG_DIR, 'fig4_missing.png'),
    'CV Model Comparison':             os.path.join(FIG_DIR, 'fig5_cv_comparison.png'),
    'BernoulliNB Confusion Matrix':    os.path.join(FIG_DIR, 'fig6_cm_bernoulli.png'),
    'All Confusion Matrices':          os.path.join(FIG_DIR, 'fig7_all_cm.png'),
    'Feature Group Ablation':          os.path.join(FIG_DIR, 'fig8_ablation.png'),
    'Per-Class F1 Scores':             os.path.join(FIG_DIR, 'fig9_perclass_f1.png'),
    'Error Pattern Analysis':          os.path.join(FIG_DIR, 'fig10_errors.png'),
}

# Width in inches per figure (wider for multi-panel ones)
WIDTH_MAP = {
    'All Confusion Matrices':          6.3,
    'Categorical Features by Segment': 6.3,
}

def replace_figure_placeholders(docx_path):
    doc = Document(docx_path)

    for para in doc.paragraphs:
        text = para.text.strip()

        # Match "[Figure: Some Name]" pattern
        if text.startswith('[Figure:') and text.endswith(']'):
            alt = text[8:-1].strip()

            if alt not in PLACEHOLDER_MAP:
                print(f"  SKIP (no map entry): {alt}")
                continue

            fig_path = PLACEHOLDER_MAP[alt]
            if not os.path.exists(fig_path):
                print(f"  SKIP (file missing): {fig_path}")
                continue

            width = WIDTH_MAP.get(alt, 5.8)

            # Clear all runs in this paragraph
            for run in para.runs:
                run.text = ''

            # Clear any XML children of the paragraph that hold text/runs
            p_xml = para._p
            # Remove all 'w:r' run elements
            for r in p_xml.findall(qn('w:r')):
                p_xml.remove(r)

            # Add a new run with the picture
            run = para.add_run()
            run.add_picture(fig_path, width=Inches(width))

            # Ensure paragraph is centered
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.space_before = Pt(6)
            para.paragraph_format.space_after = Pt(2)

            print(f"  Inserted: {alt} → {os.path.basename(fig_path)}")

    doc.save(docx_path)
    print(f"\nDocx updated: {docx_path}")


def docx_to_pdf(docx_path, pdf_path):
    word = win32com.client.Dispatch('Word.Application')
    word.Visible = False
    try:
        doc = word.Documents.Open(os.path.abspath(docx_path))
        doc.SaveAs(os.path.abspath(pdf_path), FileFormat=17)
        doc.Close()
        print(f"PDF saved:  {pdf_path}")
    finally:
        word.Quit()


if __name__ == '__main__':
    # Verify figures exist
    print("Checking figures...")
    for name, path in PLACEHOLDER_MAP.items():
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{status}] {name}")

    print("\nReplacing placeholders...")
    replace_figure_placeholders(DOCX_PATH)

    print("\nExporting PDF...")
    docx_to_pdf(DOCX_PATH, PDF_PATH)
    print("Done.")
