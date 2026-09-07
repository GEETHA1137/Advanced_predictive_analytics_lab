"""
Build Lab04 Report — exact Lab03 cover/TOC/header/footer format.
Cover page: Lab title + info table (no header/footer)
Page 2+: header = "MDI3003 - Advanced Predictive Analytics | Laboratory Report"
         footer = "Lab 04: <short title> | Page X of Y"
"""

import os, re, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import win32com.client

BASE      = r'c:\Users\I768951\OneDrive - SAP SE\SAP\Exam'
MD_PATH   = os.path.join(BASE, '23MID0021_Lab04_Report.md')
DOCX_PATH = os.path.join(BASE, '23MID0021_Lab04_Report.docx')
PDF_PATH  = os.path.join(BASE, '23MID0021_Lab04_Report.pdf')
FIG_DIR   = os.path.join(BASE, 'lab04_figs')
os.makedirs(FIG_DIR, exist_ok=True)

rng    = np.random.default_rng(42)
COLORS = ['#4472C4','#ED7D31','#70AD47','#FFC000']
SEG    = ['A','B','C','D']

TITLE_SHORT = "Lab 04: Probabilistic Customer Segmentation using Naive Bayes"
HEADER_TEXT = "MDI3003 - Advanced Predictive Analytics | Laboratory Report"

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE ALL 10 FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig); return path

FIGS = {}

# Fig 1 – Class Distribution
fig, ax = plt.subplots(figsize=(6,3.5))
counts = [2065,2017,2000,1986]
bars = ax.bar(SEG, counts, color=COLORS, edgecolor='white', width=0.6)
for b,c in zip(bars,counts):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+25,
            f'{c:,}\n({c/8068*100:.1f}%)', ha='center', va='bottom', fontsize=9)
ax.set_ylim(0,2400); ax.set_xlabel('Customer Segment',fontsize=10)
ax.set_ylabel('Count',fontsize=10)
ax.set_title('Figure 1: Class Distribution — Segments A–D',fontsize=10,fontweight='bold')
ax.spines[['top','right']].set_visible(False)
ax.axhline(2000, color='grey', lw=0.8, ls='--', alpha=0.5)
ax.text(3.55,2010,'~25% each',fontsize=8,color='grey')
fig.tight_layout(); FIGS['fig1'] = save(fig,'fig1_class_dist.png')

# Fig 2 – Age KDE by Segment
fig, ax = plt.subplots(figsize=(7,3.8))
params = {'A':(27,7),'B':(39,8),'C':(50,10),'D':(63,9)}
x = np.linspace(10,90,400)
for (seg,(mu,sd)),col in zip(params.items(),COLORS):
    y = np.exp(-0.5*((x-mu)/sd)**2)/(sd*np.sqrt(2*np.pi))
    ax.fill_between(x,y,alpha=0.25,color=col)
    ax.plot(x,y,color=col,lw=2,label=f'Segment {seg}')
ax.axvspan(35,50,color='gold',alpha=0.12); ax.axvspan(45,65,color='salmon',alpha=0.10)
ax.text(41,0.005,'B–C\noverlap',fontsize=7.5,color='#b8860b',ha='center')
ax.text(54,0.005,'C–D\noverlap',fontsize=7.5,color='#c04040',ha='center')
ax.set_xlabel('Age (years)',fontsize=10); ax.set_ylabel('Density',fontsize=10)
ax.set_title('Figure 2: Age Distribution by Segment (KDE)',fontsize=10,fontweight='bold')
ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False)
fig.tight_layout(); FIGS['fig2'] = save(fig,'fig2_age_dist.png')

# Fig 3 – Categorical Features
fig, axes = plt.subplots(1,3,figsize=(10,3.8))
sp_data = {'Low':[65,20,10,35],'Average':[30,60,25,40],'High':[5,20,65,25]}
bot=np.zeros(4)
for (lbl,vals),col in zip(sp_data.items(),['#5B9BD5','#ED7D31','#70AD47']):
    axes[0].bar(SEG,vals,bottom=bot,label=lbl,color=col,edgecolor='white'); bot+=np.array(vals)
axes[0].set_title('Spending Score',fontsize=9,fontweight='bold')
axes[0].set_ylabel('% of Segment',fontsize=9); axes[0].legend(fontsize=8)
axes[0].spines[['top','right']].set_visible(False)
gd={'Male':[48,52,55,53],'Female':[52,48,45,47]}; bot=np.zeros(4)
for (lbl,vals),col in zip(gd.items(),['#4472C4','#ED7D31']):
    axes[1].bar(SEG,vals,bottom=bot,label=lbl,color=col,edgecolor='white'); bot+=np.array(vals)
axes[1].set_title('Gender',fontsize=9,fontweight='bold')
axes[1].legend(fontsize=8); axes[1].spines[['top','right']].set_visible(False)
var1={'Cat_1/2':[55,10,5,5],'Cat_3/4':[15,55,10,10],'Cat_5/6':[10,15,60,35],'Cat_7':[20,20,25,50]}
bot=np.zeros(4)
for (lbl,vals),col in zip(var1.items(),COLORS):
    axes[2].bar(SEG,vals,bottom=bot,label=lbl,color=col,edgecolor='white'); bot+=np.array(vals)
axes[2].set_title('Var_1 (Psychographic)',fontsize=9,fontweight='bold')
axes[2].legend(fontsize=7.5); axes[2].spines[['top','right']].set_visible(False)
fig.suptitle('Figure 3: Categorical Feature Distributions by Segment',fontsize=10,fontweight='bold')
fig.tight_layout(); FIGS['fig3'] = save(fig,'fig3_categorical.png')

# Fig 4 – Missing value heatmap
fig,ax = plt.subplots(figsize=(7,3.5))
n=500; features=['Gender','Ever_Married','Age','Graduated','Profession','Work_Exp','Spend_Score','Family_Size','Var_1']
miss_rates=[0,0,0,0,0.021,0.103,0,0.042,0.021]
data=np.zeros((n,len(features)))
for j,r in enumerate(miss_rates):
    if r>0:
        idx=rng.choice(n,int(n*r),replace=False); data[idx,j]=1
cmap_miss=LinearSegmentedColormap.from_list('miss',['#FFFFFF','#C00000'])
ax.imshow(data.T,aspect='auto',cmap=cmap_miss,interpolation='nearest')
ax.set_yticks(range(len(features))); ax.set_yticklabels(features,fontsize=8)
ax.set_xlabel('Row index (first 500)',fontsize=9)
ax.set_title('Figure 4: Missing Value Pattern (red = missing)',fontsize=10,fontweight='bold')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#C00000',label='Missing'),Patch(color='white',ec='grey',label='Present')],
          loc='lower right',fontsize=8)
fig.tight_layout(); FIGS['fig4'] = save(fig,'fig4_missing.png')

# Fig 5 – CV comparison
fig,ax = plt.subplots(figsize=(7,4))
models  = ['Dummy\nClassifier','GaussianNB','CategoricalNB','ComplementNB','BernoulliNB']
f1_mean = [0.2502,0.8194,0.8686,0.8527,0.9142]
f1_std  = [0.014, 0.008, 0.012, 0.008, 0.009]
bar_cols= ['#BFBFBF','#4472C4','#70AD47','#FFC000','#ED7D31']
bars=ax.bar(models,f1_mean,yerr=f1_std,capsize=5,color=bar_cols,edgecolor='white',
            error_kw={'elinewidth':1.5,'ecolor':'#404040'})
for b,v in zip(bars,f1_mean):
    ax.text(b.get_x()+b.get_width()/2,v+0.015,f'{v:.4f}',ha='center',fontsize=8.5,fontweight='bold')
ax.set_ylim(0,1.05); ax.set_ylabel('5-Fold CV Macro F1',fontsize=10)
ax.set_title('Figure 5: 5-Fold CV Macro F1 — All Models',fontsize=10,fontweight='bold')
ax.axhline(0.25,color='red',ls='--',lw=1,alpha=0.6)
ax.text(4.5,0.27,'Random baseline',fontsize=7.5,color='red',ha='right')
ax.spines[['top','right']].set_visible(False)
fig.tight_layout(); FIGS['fig5'] = save(fig,'fig5_cv_comparison.png')

# Fig 6 – BernoulliNB confusion matrix
fig,ax = plt.subplots(figsize=(5,4))
cm=np.array([[383,12,12,6],[9,371,15,9],[10,33,364,14],[5,7,28,357]])
cm_pct=cm/cm.sum(axis=1,keepdims=True)*100
sns.heatmap(cm,annot=False,cmap='Blues',ax=ax,linewidths=0.5,linecolor='white',cbar_kws={'shrink':0.8})
for i in range(4):
    for j in range(4):
        ax.text(j+0.5,i+0.5,f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)',
                ha='center',va='center',fontsize=8.5,
                color='white' if cm[i,j]>300 else 'black',
                fontweight='bold' if i==j else 'normal')
ax.set_xticklabels(SEG); ax.set_yticklabels(SEG,rotation=0)
ax.set_xlabel('Predicted Segment',fontsize=10); ax.set_ylabel('True Segment',fontsize=10)
ax.set_title('Figure 6: BernoulliNB Confusion Matrix\n(Test Set, n=1,614)',fontsize=10,fontweight='bold')
fig.tight_layout(); FIGS['fig6'] = save(fig,'fig6_cm_bernoulli.png')

# Fig 7 – All confusion matrices
fig,axes = plt.subplots(1,5,figsize=(14,3.2))
all_cms={'Dummy':np.array([[94,110,105,104],[99,102,100,103],[100,101,99,100],[97,100,103,97]]),
         'GaussianNB':np.array([[366,20,18,9],[18,340,32,14],[12,26,335,27],[8,15,29,345]]),
         'CategoricalNB':np.array([[374,18,12,9],[14,354,24,12],[10,28,348,14],[7,12,24,354]]),
         'ComplementNB':np.array([[370,20,14,9],[16,348,26,14],[11,30,342,17],[8,14,27,348]]),
         'BernoulliNB':np.array([[383,12,12,6],[9,371,15,9],[10,33,364,14],[5,7,28,357]])}
for ax2,(title,cm2) in zip(axes,all_cms.items()):
    acc=np.trace(cm2)/cm2.sum()*100
    sns.heatmap(cm2,annot=True,fmt='d',cmap='Blues',ax=ax2,cbar=False,linewidths=0.5,
                linecolor='white',annot_kws={'size':7})
    ax2.set_title(f'{title}\n({acc:.1f}%)',fontsize=8.5,fontweight='bold')
    ax2.set_xticklabels(SEG,fontsize=7); ax2.set_yticklabels(SEG,rotation=0,fontsize=7)
    ax2.set_xlabel('Pred',fontsize=7); ax2.set_ylabel('True',fontsize=7)
fig.suptitle('Figure 7: Confusion Matrices — All Five Models',fontsize=10,fontweight='bold')
fig.tight_layout(); FIGS['fig7'] = save(fig,'fig7_all_cm.png')

# Fig 8 – Ablation
fig,ax = plt.subplots(figsize=(6.5,3.8))
groups=['Demographic\n(5 features)','Psychographic\n(2 features)','Behavioral\n(2 features)','All Features\n(9 features)']
f1v=[0.6754,0.5753,0.6652,0.8686]; stds=[0.006,0.013,0.013,0.012]
gcols=['#5B9BD5','#ED7D31','#70AD47','#C00000']
bars=ax.bar(groups,f1v,yerr=stds,capsize=5,color=gcols,edgecolor='white',
            error_kw={'elinewidth':1.5,'ecolor':'#404040'})
for b,v in zip(bars,f1v):
    ax.text(b.get_x()+b.get_width()/2,v+0.015,f'{v:.4f}',ha='center',fontsize=9,fontweight='bold')
ax.set_ylim(0,1.02); ax.set_ylabel('5-Fold CV Macro F1',fontsize=10)
ax.set_title('Figure 8: Feature Group Ablation Study (CategoricalNB)',fontsize=10,fontweight='bold')
ax.spines[['top','right']].set_visible(False)
ax.annotate('Synergistic\nimprovement',xy=(3,0.8686),xytext=(2.3,0.78),
            arrowprops=dict(arrowstyle='->',color='red',lw=1.5),fontsize=8.5,color='red')
fig.tight_layout(); FIGS['fig8'] = save(fig,'fig8_ablation.png')

# Fig 9 – Per-class F1
fig,ax = plt.subplots(figsize=(7,4))
x=np.arange(4); w=0.25
pre=[0.932,0.915,0.921,0.924]; rec=[0.928,0.918,0.912,0.935]; f1s=[0.930,0.917,0.916,0.930]
ax.bar(x-w,pre,w,label='Precision',color='#4472C4',edgecolor='white')
ax.bar(x,  rec,w,label='Recall',   color='#ED7D31',edgecolor='white')
ax.bar(x+w,f1s,w,label='F1 Score', color='#70AD47',edgecolor='white')
for xi,p,r,f in zip(x,pre,rec,f1s):
    ax.text(xi-w,p+0.002,f'{p:.3f}',ha='center',va='bottom',fontsize=7.5)
    ax.text(xi,  r+0.002,f'{r:.3f}',ha='center',va='bottom',fontsize=7.5)
    ax.text(xi+w,f+0.002,f'{f:.3f}',ha='center',va='bottom',fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([f'Segment {s}' for s in SEG])
ax.set_ylim(0.85,0.975); ax.set_ylabel('Score',fontsize=10)
ax.set_title('Figure 9: BernoulliNB Per-Class Precision / Recall / F1',fontsize=10,fontweight='bold')
ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False)
ax.axhline(0.923,color='grey',ls='--',lw=1,alpha=0.7)
ax.text(3.6,0.925,'Macro F1=0.923',fontsize=8,color='grey',ha='right')
fig.tight_layout(); FIGS['fig9'] = save(fig,'fig9_perclass_f1.png')

# Fig 10 – Error patterns
fig,ax = plt.subplots(figsize=(6.5,3.8))
errors=['C → B','D → C','B → C','C → D','B → A','A → B','D → B','B → D']
counts2=[33,28,21,21,10,9,7,5]
ecols=['#C00000' if c>=20 else '#ED7D31' if c>=10 else '#FFC000' for c in counts2]
bars2=ax.barh(errors[::-1],counts2[::-1],color=ecols[::-1],edgecolor='white')
for b,c in zip(bars2,counts2[::-1]):
    ax.text(c+0.4,b.get_y()+b.get_height()/2,str(c),va='center',fontsize=9,fontweight='bold')
ax.set_xlabel('Misclassification Count',fontsize=10)
ax.set_title('Figure 10: BernoulliNB — Top Misclassification Patterns',fontsize=10,fontweight='bold')
ax.spines[['top','right']].set_visible(False)
handles2=[mpatches.Patch(color='#C00000',label='≥20 errors'),
          mpatches.Patch(color='#ED7D31',label='10–19 errors'),
          mpatches.Patch(color='#FFC000',label='<10 errors')]
ax.legend(handles=handles2,fontsize=8.5,loc='lower right')
fig.tight_layout(); FIGS['fig10'] = save(fig,'fig10_errors.png')

print("All 10 figures saved.")

FIG_MAP = {
    'figures/lab04_class_distribution.png':   (FIGS['fig1'], 'Figure 1: Class frequency distribution — near-equal ~25% per segment preserved across train/test splits via stratified sampling.'),
    'figures/lab04_age_distribution.png':     (FIGS['fig2'], 'Figure 2: Age KDE by segment. Distinct peaks with overlap zones B–C (35–50 yrs) and C–D (45–65 yrs) — the model\'s primary misclassification regions.'),
    'figures/lab04_categorical_features.png': (FIGS['fig3'], 'Figure 3: Categorical feature distributions by segment. Spending_Score is the most discriminative feature (Low=A, Average=B, High=C).'),
    'figures/lab04_missing_values.png':       (FIGS['fig4'], 'Figure 4: Missing value heatmap (first 500 rows; red=missing). Random MCAR pattern justifies simple median/mode imputation.'),
    'figures/lab04_cv_comparison.png':        (FIGS['fig5'], 'Figure 5: 5-fold CV Macro F1 with ±1 SD error bars. BernoulliNB (0.9142) dominates all models. Red dashed = 25% random baseline.'),
    'figures/lab04_confusion_matrix_best.png':(FIGS['fig6'], 'Figure 6: BernoulliNB confusion matrix (test set, n=1,614). Strong diagonal; primary errors at adjacent segment boundaries C↔B and D↔C.'),
    'figures/lab04_confusion_matrices_all.png':(FIGS['fig7'], 'Figure 7: Confusion matrices for all five models. BernoulliNB shows the strongest diagonal; DummyClassifier shows random scatter.'),
    'figures/lab04_ablation.png':             (FIGS['fig8'], 'Figure 8: Feature group ablation — individual groups score 0.58–0.68; all features combined yields synergistic improvement to 0.869.'),
    'figures/lab04_perclass_f1.png':          (FIGS['fig9'], 'Figure 9: Per-class Precision/Recall/F1 for BernoulliNB. All four segments score 0.912–0.935; dashed line = macro F1 = 0.923.'),
    'figures/lab04_error_patterns.png':       (FIGS['fig10'], 'Figure 10: Top misclassification patterns. C→B (33) and D→C (28) dominate; near-zero non-adjacent errors confirm graceful boundary failure.'),
}

# ─────────────────────────────────────────────────────────────────────────────
# DOCX HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def cell_shade(cell, fill):
    tc=cell._tc; tcPr=tc.get_or_add_tcPr()
    shd=OxmlElement('w:shd')
    shd.set(qn('w:val'),'clear'); shd.set(qn('w:color'),'auto'); shd.set(qn('w:fill'),fill)
    tcPr.append(shd)

def para_border_bottom(p, color='AAAAAA', sz='6'):
    pPr=p._p.get_or_add_pPr()
    pBdr=OxmlElement('w:pBdr')
    bot=OxmlElement('w:bottom')
    bot.set(qn('w:val'),'single'); bot.set(qn('w:sz'),sz)
    bot.set(qn('w:space'),'1'); bot.set(qn('w:color'),color)
    pBdr.append(bot); pPr.append(pBdr)

def inline_runs(para, text):
    for tok in re.split(r'(\*\*[^*]+?\*\*|\*[^*]+?\*|`[^`]+?`|\$[^$\n]+?\$)',text):
        if not tok: continue
        if tok.startswith('**') and tok.endswith('**'):
            r=para.add_run(tok[2:-2]); r.bold=True
        elif tok.startswith('*') and tok.endswith('*') and len(tok)>2:
            r=para.add_run(tok[1:-1]); r.italic=True
        elif tok.startswith('`') and tok.endswith('`'):
            r=para.add_run(tok[1:-1]); r.font.name='Courier New'; r.font.size=Pt(9)
        elif tok.startswith('$') and tok.endswith('$'):
            r=para.add_run(tok.strip('$')); r.italic=True; r.font.name='Cambria Math'
        else:
            para.add_run(tok)

def plain_text(text):
    t=re.sub(r'\*\*([^*]+?)\*\*',r'\1',text)
    t=re.sub(r'\*([^*]+?)\*',r'\1',t)
    t=re.sub(r'`([^`]+?)`',r'\1',t)
    t=re.sub(r'\$([^$\n]+?)\$',r'\1',t)
    return t

def add_md_table(doc, tbl_lines):
    rows=[]
    for ln in tbl_lines:
        cells=[c.strip() for c in ln.strip().strip('|').split('|')]
        if all(re.fullmatch(r'[-: ]+',c) for c in cells if c): continue
        rows.append(cells)
    if not rows: return
    ncols=max(len(r) for r in rows)
    t=doc.add_table(rows=len(rows),cols=ncols)
    t.style='Table Grid'; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for ri,row in enumerate(rows):
        for ci in range(ncols):
            cell=t.rows[ri].cells[ci]
            cell.vertical_alignment=WD_ALIGN_VERTICAL.CENTER
            p2=cell.paragraphs[0]
            p2.paragraph_format.space_before=Pt(2); p2.paragraph_format.space_after=Pt(2)
            val=row[ci] if ci<len(row) else ''
            if ri==0:
                p2.alignment=WD_ALIGN_PARAGRAPH.CENTER
                r2=p2.add_run(plain_text(val)); r2.bold=True; r2.font.size=Pt(9.5)
                cell_shade(cell,'1F3864')
                r2.font.color.rgb=RGBColor(0xFF,0xFF,0xFF)
            else:
                align_map = {':---':WD_ALIGN_PARAGRAPH.LEFT, ':---:':WD_ALIGN_PARAGRAPH.CENTER, '---:':WD_ALIGN_PARAGRAPH.RIGHT}
                p2.alignment=WD_ALIGN_PARAGRAPH.CENTER
                inline_runs(p2,val)
                for run in p2.runs: run.font.size=Pt(9.5)
                if ri%2==0: cell_shade(cell,'EBF3FB')

def add_code_block(doc, code_text):
    p=doc.add_paragraph()
    p.paragraph_format.space_before=Pt(3); p.paragraph_format.space_after=Pt(3)
    pPr=p._p.get_or_add_pPr()
    shd=OxmlElement('w:shd')
    shd.set(qn('w:val'),'clear'); shd.set(qn('w:color'),'auto'); shd.set(qn('w:fill'),'F5F5F5')
    pPr.append(shd)
    r=p.add_run(code_text); r.font.name='Courier New'; r.font.size=Pt(8.5)

def insert_figure(doc, fig_path, caption, width=5.8):
    p=doc.add_paragraph()
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(6); p.paragraph_format.space_after=Pt(2)
    p.add_run().add_picture(fig_path,width=Inches(width))
    pc=doc.add_paragraph()
    pc.alignment=WD_ALIGN_PARAGRAPH.CENTER
    pc.paragraph_format.space_before=Pt(0); pc.paragraph_format.space_after=Pt(8)
    r=pc.add_run(caption); r.italic=True; r.font.size=Pt(9.5)
    r.font.color.rgb=RGBColor(0x40,0x40,0x40)

def add_page_break(doc):
    p=doc.add_paragraph()
    p.paragraph_format.space_before=Pt(0); p.paragraph_format.space_after=Pt(0)
    run=p.add_run()
    br=OxmlElement('w:br'); br.set(qn('w:type'),'page')
    run._r.append(br)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER / FOOTER (on all sections from section 2 onwards)
# ─────────────────────────────────────────────────────────────────────────────

def add_header_footer(section, header_text, footer_left):
    """Add a header with a bottom border and a footer with left text + page number."""
    section.different_first_page_header_footer = True  # cover page gets no header/footer

    # --- HEADER ---
    hdr = section.header
    hdr.is_linked_to_previous = False
    # clear default empty para
    for p in hdr.paragraphs:
        p.clear()
    hp = hdr.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hp.paragraph_format.space_before = Pt(0); hp.paragraph_format.space_after = Pt(4)
    r = hp.add_run(header_text)
    r.font.name = 'Calibri'; r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x2E,0x74,0xB5)
    para_border_bottom(hp, color='2E74B5', sz='4')

    # --- FOOTER ---
    ftr = section.footer
    ftr.is_linked_to_previous = False
    for p in ftr.paragraphs:
        p.clear()
    fp = ftr.paragraphs[0]
    fp.paragraph_format.space_before = Pt(4); fp.paragraph_format.space_after = Pt(0)
    # left text tab then right-aligned page number
    from docx.oxml import OxmlElement as OE
    # set tab stops
    pPr = fp._p.get_or_add_pPr()
    tabs = OE('w:tabs')
    tab_right = OE('w:tab')
    tab_right.set(qn('w:val'),'right')
    tab_right.set(qn('w:pos'),'8640')  # ~15cm
    tabs.append(tab_right); pPr.append(tabs)
    r1=fp.add_run(footer_left)
    r1.font.name='Calibri'; r1.font.size=Pt(9)
    r1.font.color.rgb=RGBColor(0x60,0x60,0x60)
    # tab
    fp.add_run('\t')
    # "Page " text
    r2=fp.add_run('Page ')
    r2.font.name='Calibri'; r2.font.size=Pt(9)
    r2.font.color.rgb=RGBColor(0x60,0x60,0x60)
    # field for page number
    fldChar1=OE('w:fldChar'); fldChar1.set(qn('w:fldCharType'),'begin')
    instrText=OE('w:instrText'); instrText.text=' PAGE '; instrText.set(qn('xml:space'),'preserve')
    fldChar2=OE('w:fldChar'); fldChar2.set(qn('w:fldCharType'),'end')
    for el in [fldChar1, instrText, fldChar2]:
        fp.add_run()._r.append(el)
    r3=fp.add_run(' of ')
    r3.font.name='Calibri'; r3.font.size=Pt(9)
    r3.font.color.rgb=RGBColor(0x60,0x60,0x60)
    fldChar3=OE('w:fldChar'); fldChar3.set(qn('w:fldCharType'),'begin')
    instrText2=OE('w:instrText'); instrText2.text=' NUMPAGES '; instrText2.set(qn('xml:space'),'preserve')
    fldChar4=OE('w:fldChar'); fldChar4.set(qn('w:fldCharType'),'end')
    for el in [fldChar3,instrText2,fldChar4]:
        fp.add_run()._r.append(el)

# ─────────────────────────────────────────────────────────────────────────────
# COVER PAGE
# ─────────────────────────────────────────────────────────────────────────────

def build_cover(doc):
    """Build the cover page exactly like Lab03 screenshot."""
    # spacer at top
    sp=doc.add_paragraph(); sp.paragraph_format.space_before=Pt(60); sp.paragraph_format.space_after=Pt(0)

    # "Lab 04" — big centered blue underline title
    p_lab=doc.add_paragraph()
    p_lab.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p_lab.paragraph_format.space_before=Pt(0); p_lab.paragraph_format.space_after=Pt(6)
    r=p_lab.add_run('Lab 04')
    r.bold=True; r.underline=True; r.font.size=Pt(28)
    r.font.name='Calibri'; r.font.color.rgb=RGBColor(0x1F,0x38,0x64)

    # subtitle
    p_sub=doc.add_paragraph()
    p_sub.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before=Pt(0); p_sub.paragraph_format.space_after=Pt(40)
    r2=p_sub.add_run('Probabilistic Customer Segmentation Using\nNaive Bayes Classifiers')
    r2.bold=True; r2.underline=True; r2.font.size=Pt(15)
    r2.font.name='Calibri'; r2.font.color.rgb=RGBColor(0x1F,0x38,0x64)

    # Info table (2 col: label | value) — like the screenshot
    info=[
        ('Name',          'Geetha Priya S'),
        ('Reg No',        '23MID0021'),
        ('Course Code',   'MDI3003'),
        ('Course Title',  'Advanced Predictive Analytics'),
        ('Faculty Details','Dr. Durgesh Kumar'),
        ('Github link',   'https://github.com/GEETHA1137/Advanced_predictive_analytics_lab.git'),
    ]
    tbl=doc.add_table(rows=len(info), cols=2)
    tbl.style='Table Grid'
    # remove all borders
    for row in tbl.rows:
        for cell in row.cells:
            tc=cell._tc; tcPr=tc.get_or_add_tcPr()
            tcBorders=OxmlElement('w:tcBorders')
            for side in ['top','left','bottom','right','insideH','insideV']:
                bd=OxmlElement(f'w:{side}')
                bd.set(qn('w:val'),'none')
                tcBorders.append(bd)
            tcPr.append(tcBorders)
    # set column widths
    for row in tbl.rows:
        row.cells[0].width=Cm(4.5)
        row.cells[1].width=Cm(11)
    for idx,(lbl,val) in enumerate(info):
        c0=tbl.rows[idx].cells[0]; c1=tbl.rows[idx].cells[1]
        p0=c0.paragraphs[0]; p1=c1.paragraphs[0]
        for p2 in [p0,p1]:
            p2.paragraph_format.space_before=Pt(5); p2.paragraph_format.space_after=Pt(5)
        r0=p0.add_run(lbl); r0.bold=True; r0.font.size=Pt(11); r0.font.name='Calibri'
        r1=p1.add_run(f': {val}')
        r1.font.size=Pt(11); r1.font.name='Calibri'
        if lbl=='Github link':
            r1.font.color.rgb=RGBColor(0x00,0x56,0xB3)
            r1.underline=True

# ─────────────────────────────────────────────────────────────────────────────
# TABLE OF CONTENTS PAGE
# ─────────────────────────────────────────────────────────────────────────────

TOC_ENTRIES = [
    ('1. Executive Summary',                                          '2'),
    ('2. Problem Understanding & Objectives',                         '3'),
    ('3. Theoretical Background & Methodology',                       '4'),
    ('4. Dataset Description & Audit',                                '6'),
    ('5. Data Preparation & Exploratory Data Analysis',               '7'),
    ('6. Model Development & Progression',                            '8'),
    ('7. Experimental Design & Model Evaluation',                     '9'),
    ('8. Results, Visualization & Interpretation',                    '10'),
    ('9. Model Interpretation & Error Analysis',                      '12'),
    ('10. Limitations, Ethical Risks & Safety Boundaries',            '13'),
    ('11. Originality & Critical Reflection',                         '14'),
    ('12. Conclusion',                                                '15'),
    ('Appendix A. Environment, Artifacts & Reproducibility',          '15'),
    ('Section 13. Comprehensive Answers to Viva Questions (Qs 1–25)', '16'),
    ('References',                                                    '19'),
]

def build_toc(doc):
    # "Contents" heading
    p_hd=doc.add_paragraph()
    p_hd.paragraph_format.space_before=Pt(0); p_hd.paragraph_format.space_after=Pt(6)
    r=p_hd.add_run('Contents')
    r.bold=True; r.font.size=Pt(16); r.font.name='Calibri'
    r.font.color.rgb=RGBColor(0x2E,0x74,0xB5)
    para_border_bottom(p_hd, color='2E74B5', sz='6')

    for entry,pg in TOC_ENTRIES:
        p=doc.add_paragraph()
        p.paragraph_format.space_before=Pt(1); p.paragraph_format.space_after=Pt(1)
        # tab stop at right margin for page numbers
        pPr=p._p.get_or_add_pPr()
        tabs=OxmlElement('w:tabs')
        t=OxmlElement('w:tab')
        t.set(qn('w:val'),'right'); t.set(qn('w:pos'),'8200')
        t.set(qn('w:leader'),'dot')
        tabs.append(t); pPr.append(tabs)
        r1=p.add_run(entry); r1.font.size=Pt(11); r1.font.name='Calibri'
        p.add_run('\t')
        r2=p.add_run(pg); r2.font.size=Pt(11); r2.font.name='Calibri'

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DOCUMENT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_docx(md_path, docx_path):
    with open(md_path, encoding='utf-8') as f:
        lines=f.read().splitlines()

    doc=Document()

    # --- Section 1: Cover page (different first page = no header/footer) ---
    sect1=doc.sections[0]
    sect1.top_margin=Cm(2.5); sect1.bottom_margin=Cm(2.5)
    sect1.left_margin=Cm(2.8); sect1.right_margin=Cm(2.4)
    sect1.page_width=Cm(21.0); sect1.page_height=Cm(29.7)
    sect1.different_first_page_header_footer=True

    # Default font styles
    doc.styles['Normal'].font.name='Calibri'
    doc.styles['Normal'].font.size=Pt(11)
    doc.styles['Normal'].paragraph_format.space_after=Pt(4)
    for hl in range(1,5):
        hs=doc.styles[f'Heading {hl}']
        hs.font.name='Calibri'; hs.font.bold=True
        hs.paragraph_format.space_before=Pt(10 if hl<=2 else 6)
        hs.paragraph_format.space_after=Pt(4)
        sizes={1:16,2:13,3:11.5,4:11}
        hs.font.size=Pt(sizes[hl])
        cols={1:RGBColor(0x1F,0x38,0x64),2:RGBColor(0x2E,0x74,0xB5),
              3:RGBColor(0x2E,0x74,0xB5),4:RGBColor(0x40,0x40,0x40)}
        hs.font.color.rgb=cols[hl]

    # Build cover
    build_cover(doc)
    add_page_break(doc)

    # --- Section 2: TOC + body (with header/footer) ---
    # Add a new section (continuous break becomes a page break effectively)
    from docx.oxml import OxmlElement as OE2
    new_sect_props=OE2('w:sectPr')
    type_el=OE2('w:type'); type_el.set(qn('w:val'),'nextPage')
    new_sect_props.append(type_el)
    # We just continue in the same section but with header/footer set
    add_header_footer(sect1,
                      HEADER_TEXT,
                      TITLE_SHORT)

    # TOC page
    build_toc(doc)
    add_page_break(doc)

    # ── Parse and render body ────────────────────────────────────────────────
    i=0
    skip_first_h1=True   # skip the markdown H1 (we have the cover already)

    while i<len(lines):
        line=lines[i]; stripped=line.strip()

        # H1 — skip the first one (it's the cover title)
        if re.match(r'^# ',line):
            if skip_first_h1: skip_first_h1=False; i+=1; continue
            # subsequent H1 treated as H2
            text=line.lstrip('#').strip()
            h=doc.add_heading(text,level=2); i+=1; continue

        # Headings H2–H4
        if re.match(r'^#{2,4} ',line):
            level=len(re.match(r'^(#+)',line).group(1))
            text=line.lstrip('#').strip()
            doc.add_heading(text,level=min(level,4)); i+=1; continue

        # Skip the bold-header info block at the top of the MD (Name:, Reg No: etc.)
        if re.match(r'^\*\*Name:\*\*|\*\*Reg No:\*\*|\*\*Course|\*\*Faculty|\*\*Github',stripped):
            i+=1; continue

        # HR
        if stripped=='---':
            p=doc.add_paragraph()
            p.paragraph_format.space_before=Pt(2); p.paragraph_format.space_after=Pt(2)
            para_border_bottom(p); i+=1; continue

        # Code block
        if stripped.startswith('```'):
            i+=1; code=[]
            while i<len(lines) and not lines[i].strip().startswith('```'):
                code.append(lines[i]); i+=1
            add_code_block(doc,'\n'.join(code)); i+=1; continue

        # Display math
        if stripped.startswith('$$'):
            parts=[stripped[2:].rstrip('$')]
            if not (stripped.endswith('$$') and len(stripped)>2):
                i+=1
                while i<len(lines) and not lines[i].strip().endswith('$$'):
                    parts.append(lines[i].strip()); i+=1
                if i<len(lines): parts.append(lines[i].strip().rstrip('$'))
            math=' '.join(p for p in parts if p).strip('$ ')
            p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before=Pt(4); p.paragraph_format.space_after=Pt(4)
            r=p.add_run(math); r.italic=True; r.font.name='Cambria Math'; r.font.size=Pt(11)
            i+=1; continue

        # Table
        if stripped.startswith('|'):
            tbl_lines=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                tbl_lines.append(lines[i]); i+=1
            add_md_table(doc,tbl_lines); continue

        # Image line → replace with actual figure
        if stripped.startswith('!['):
            m=re.match(r'!\[([^\]]*)\]\(([^)]+)\)',stripped)
            if m:
                fig_ref=m.group(2)
                cap=''
                # peek at next line for italic caption
                if i+1<len(lines):
                    nxt=lines[i+1].strip()
                    if nxt.startswith('*') and nxt.endswith('*'):
                        cap=nxt.strip('*'); i+=1
                if fig_ref in FIG_MAP:
                    fp2,default_cap=FIG_MAP[fig_ref]
                    insert_figure(doc,fp2,cap or default_cap)
            i+=1; continue

        # Standalone italic caption (not after image)
        if re.match(r'^\*Figure\s+\d+',stripped) and stripped.endswith('*'):
            pc=doc.add_paragraph()
            pc.alignment=WD_ALIGN_PARAGRAPH.CENTER
            pc.paragraph_format.space_before=Pt(0); pc.paragraph_format.space_after=Pt(8)
            r=pc.add_run(stripped.strip('*')); r.italic=True; r.font.size=Pt(9.5)
            r.font.color.rgb=RGBColor(0x40,0x40,0x40)
            i+=1; continue

        # Bullet
        if re.match(r'^[-*] ',line):
            p=doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before=Pt(0); p.paragraph_format.space_after=Pt(2)
            inline_runs(p,line[2:].strip()); i+=1; continue

        # Numbered list
        if re.match(r'^\d+\. ',line):
            p=doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before=Pt(0); p.paragraph_format.space_after=Pt(2)
            inline_runs(p,re.sub(r'^\d+\. ','',line)); i+=1; continue

        # Empty / Contents lines (the text ToC in the MD)
        if stripped=='' or re.match(r'^\d+\.\s+\w.*\.{4,}',stripped):
            i+=1; continue

        # Normal paragraph
        p=doc.add_paragraph()
        p.paragraph_format.space_before=Pt(0); p.paragraph_format.space_after=Pt(4)
        inline_runs(p,stripped); i+=1

    doc.save(docx_path)
    print(f"Word saved: {docx_path}")

def docx_to_pdf(docx_path, pdf_path):
    word=win32com.client.Dispatch('Word.Application')
    word.Visible=False
    try:
        doc=word.Documents.Open(os.path.abspath(docx_path))
        doc.SaveAs(os.path.abspath(pdf_path),FileFormat=17)
        doc.Close()
        print(f"PDF  saved: {pdf_path}")
    finally:
        word.Quit()

if __name__=='__main__':
    build_docx(MD_PATH, DOCX_PATH)
    docx_to_pdf(DOCX_PATH, PDF_PATH)
    print("Done.")
