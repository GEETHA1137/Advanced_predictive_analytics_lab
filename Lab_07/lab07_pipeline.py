"""
Lab 07 Pipeline: Recommendation System from Customer Transaction Data
Roll No  : 23MID0021  |  Name: Geetha Priya S
Course   : MDI3003 – Advanced Predictive Analytics
Dataset  : UCI Online Retail (synthetic replica, same schema)
"""
import os, json, hashlib, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc as sk_auc
import joblib

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = r'c:\Users\I768951\OneDrive - SAP SE\SAP\Exam'
OUT_DIR = os.path.join(BASE, 'lab07_outputs')
FIG_DIR = os.path.join(BASE, 'lab07_figs')
MDL_DIR = os.path.join(BASE, 'models')
ART_DIR = os.path.join(BASE, 'artifacts')
for d in [OUT_DIR, FIG_DIR, MDL_DIR, ART_DIR]:
    os.makedirs(d, exist_ok=True)

RNG = np.random.default_rng(42)
CATEGORIES = ['Home Decor', 'Kitchenware', 'Stationery', 'Bags & Accessories',
              'Gifts', 'Seasonal', 'Party Supplies', 'Lighting']
N_CATS = len(CATEGORIES)
FEAT_COLS = ['recency_days','cust_txns','cust_items','cust_spend','cust_uniq','pref_cat_enc',
             'item_txns','item_buyers','item_avg_price','item_recency_days','item_cat_enc','item_pop_rank',
             'pair_purchases','pair_qty','pair_spend','pair_days_since','cat_match']
COLORS = ['#2E86AB','#A23B72','#F18F01','#C73E1D','#3B1F2B','#44BBA4','#E94F37','#393E41','#F5A623']

# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def generate_synthetic_retail(rng, n_customers=1800, n_items=400):
    print("Generating synthetic UCI Online Retail dataset...")
    cat_prices = [4.25, 3.75, 1.65, 5.95, 6.50, 3.95, 2.85, 8.75]
    items_list = [{'StockCode': str(10000+i),
                   'Description': f'{CATEGORIES[i%N_CATS]} Product {i+1:03d}',
                   'Category': CATEGORIES[i%N_CATS],
                   'BasePrice': round(cat_prices[i%N_CATS] * rng.uniform(0.7, 2.2), 2)}
                  for i in range(n_items)]
    items_df = pd.DataFrame(items_list)

    raw_wt = rng.exponential(1.0, n_items)
    raw_wt[:n_items//10] *= 8
    item_weights = raw_wt / raw_wt.sum()

    seg_probs    = [0.08, 0.22, 0.42, 0.28]
    seg_avg_txns = [28, 10, 3, 1]
    segments     = rng.choice(4, size=n_customers, p=seg_probs)
    cust_pref1   = rng.integers(0, N_CATS, size=n_customers)
    cust_pref2   = (cust_pref1 + rng.integers(1, N_CATS, size=n_customers)) % N_CATS

    start_date = pd.Timestamp('2010-12-01')
    date_span  = (pd.Timestamp('2011-12-09') - start_date).days

    rows, invoice_id = [], 536365
    for cid in range(n_customers):
        n_inv  = max(1, rng.poisson(seg_avg_txns[segments[cid]]))
        prefs  = [cust_pref1[cid], cust_pref2[cid]]
        for _ in range(n_inv):
            inv_date = start_date + pd.Timedelta(days=int(rng.integers(0, date_span)))
            bought   = set()
            for _ in range(int(rng.integers(1, 7))):
                if rng.random() < 0.72:
                    pcat = CATEGORIES[prefs[rng.integers(0, 2)]]
                    mask = items_df['Category'].values == pcat
                    w = item_weights * mask
                    w = w / w.sum() if w.sum() > 0 else item_weights
                else:
                    w = item_weights
                idx = rng.choice(n_items, p=w)
                if idx in bought: continue
                bought.add(idx)
                item = items_df.iloc[idx]
                rows.append({'InvoiceNo': str(invoice_id), 'StockCode': item['StockCode'],
                              'Description': item['Description'], 'Quantity': int(rng.integers(1,13)),
                              'InvoiceDate': inv_date, 'UnitPrice': item['BasePrice'],
                              'CustomerID': f'C{15000+cid}', 'Country': 'United Kingdom',
                              'Category': item['Category']})
            invoice_id += 1

    df = pd.DataFrame(rows)
    n_cancel = int(len(df)*0.08)
    cidx = rng.choice(len(df), size=n_cancel, replace=False)
    cr   = df.iloc[cidx].copy(); cr['InvoiceNo'] = 'C'+cr['InvoiceNo']; cr['Quantity'] = -cr['Quantity']
    df   = pd.concat([df, cr], ignore_index=True)
    n_miss = int(len(df)*0.05)
    df.loc[rng.choice(len(df), size=n_miss, replace=False), 'CustomerID'] = np.nan
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Generated {len(df):,} rows | {n_cancel:,} cancellations | {n_miss:,} missing IDs")
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
def clean_data(df):
    raw_n = len(df)
    is_c  = df['InvoiceNo'].astype(str).str.startswith('C')
    df2   = df[~is_c].dropna(subset=['CustomerID'])
    df2   = df2[(df2['Quantity']>0) & (df2['UnitPrice']>0)].sort_values('InvoiceDate').reset_index(drop=True)
    n_c, n_m = int(is_c.sum()), raw_n - int(is_c.sum()) - len(df2)
    print(f"\n[Clean] raw={raw_n:,}  cancel={n_c:,}  miss_id={n_m:,}  clean={len(df2):,}  "
          f"cust={df2['CustomerID'].nunique():,}  items={df2['StockCode'].nunique():,}")
    return df2, {'raw': raw_n, 'cancelled': n_c, 'missing_id': n_m, 'clean': len(df2)}

# ═══════════════════════════════════════════════════════════════════════════════
# SPLIT (70 / 85 percentile)
# ═══════════════════════════════════════════════════════════════════════════════
def split_data(df):
    dates = df['InvoiceDate'].sort_values()
    t1, t2 = dates.quantile(0.70), dates.quantile(0.85)
    train = df[df['InvoiceDate']<=t1].copy()
    val   = df[(df['InvoiceDate']>t1)&(df['InvoiceDate']<=t2)].copy()
    test  = df[df['InvoiceDate']>t2].copy()
    print(f"\n[Split]  train={len(train):,} ({train['InvoiceDate'].min().date()} to {train['InvoiceDate'].max().date()})  "
          f"val={len(val):,}  test={len(test):,}")
    mf = {'train_start':str(train['InvoiceDate'].min().date()),'train_end':str(train['InvoiceDate'].max().date()),
          'val_start':str(val['InvoiceDate'].min().date()),'val_end':str(val['InvoiceDate'].max().date()),
          'test_start':str(test['InvoiceDate'].min().date()),'test_end':str(test['InvoiceDate'].max().date()),
          't1':str(t1.date()),'t2':str(t2.date()),
          'train_rows':int(len(train)),'val_rows':int(len(val)),'test_rows':int(len(test))}
    return train, val, test, mf

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (strictly before cutoff_ts)
# ═══════════════════════════════════════════════════════════════════════════════
def engineer_features(hist, catalog, cutoff_ts):
    ref = pd.Timestamp(cutoff_ts)
    hist = hist.copy(); hist['rev'] = hist['Quantity']*hist['UnitPrice']
    cf = hist.groupby('CustomerID').agg(
        recency_days=('InvoiceDate', lambda x: (ref-x.max()).days),
        cust_txns=('InvoiceNo','nunique'), cust_items=('Quantity','sum'),
        cust_spend=('rev','sum'), cust_uniq=('StockCode','nunique')).reset_index()
    pref = (hist.groupby(['CustomerID','Category'])['Quantity'].sum().reset_index()
               .sort_values('Quantity',ascending=False).drop_duplicates('CustomerID')
               [['CustomerID','Category']].rename(columns={'Category':'pref_cat'}))
    cf = cf.merge(pref, on='CustomerID', how='left')
    cf['pref_cat_enc'] = pd.Categorical(cf['pref_cat'], categories=CATEGORIES).codes.astype(float)

    ih = hist[hist['StockCode'].isin(catalog)].copy()
    if_df = ih.groupby('StockCode').agg(
        item_txns=('InvoiceNo','nunique'), item_buyers=('CustomerID','nunique'),
        item_avg_price=('UnitPrice','mean'),
        item_recency_days=('InvoiceDate', lambda x:(ref-x.max()).days)).reset_index()
    icm = hist.drop_duplicates('StockCode')[['StockCode','Category']]
    if_df = if_df.merge(icm, on='StockCode', how='left')
    if_df['item_cat_enc'] = pd.Categorical(if_df['Category'], categories=CATEGORIES).codes.astype(float)
    if_df = if_df.sort_values('item_buyers',ascending=False).reset_index(drop=True)
    if_df['item_pop_rank'] = (if_df.index+1).astype(float)
    # Ensure all catalog items are present (fill missing with zeros)
    cat_df = pd.DataFrame({'StockCode': catalog})
    if_df  = cat_df.merge(if_df, on='StockCode', how='left')
    for c in ['item_txns','item_buyers','item_avg_price','item_recency_days','item_cat_enc','item_pop_rank']:
        if_df[c] = if_df[c].fillna(0.0)

    pf = (ih.groupby(['CustomerID','StockCode']).agg(
        pair_purchases=('InvoiceNo','nunique'), pair_qty=('Quantity','sum'),
        pair_spend=('rev','sum'),
        pair_days_since=('InvoiceDate', lambda x:(ref-x.max()).days)).reset_index())
    pf = pf.merge(cf[['CustomerID','pref_cat']],on='CustomerID',how='left')
    pf = pf.merge(icm, on='StockCode', how='left')
    pf['cat_match'] = (pf['pref_cat']==pf['Category']).astype(float)
    pf = pf.drop(columns=['pref_cat','Category'])
    return cf, if_df, pf

# ═══════════════════════════════════════════════════════════════════════════════
# VECTORISED FEATURE MATRIX  (numpy repeat/tile — no pandas merge)
# ═══════════════════════════════════════════════════════════════════════════════
def _build_fm_numpy(customers, catalog, cf, if_df, pf):
    """
    Returns X (n_c*n_i x 17 float64), cust_ids, item_ids arrays.
    Uses np.repeat + np.tile instead of cross-join + merge.
    """
    n_c, n_i = len(customers), len(catalog)

    # Customer feature matrix ordered by `customers`
    cf_idx = cf.set_index('CustomerID')
    cust_arr = cf_idx.reindex(customers)[
        ['recency_days','cust_txns','cust_items','cust_spend','cust_uniq','pref_cat_enc']
    ].fillna(0).values.astype(float)

    # Item feature matrix ordered by `catalog`
    if_idx = if_df.set_index('StockCode')
    item_arr = if_idx.reindex(catalog)[
        ['item_txns','item_buyers','item_avg_price','item_recency_days','item_cat_enc','item_pop_rank']
    ].fillna(0).values.astype(float)

    X_c = np.repeat(cust_arr, n_i, axis=0)   # (n_c*n_i, 6)
    X_i = np.tile(item_arr, (n_c, 1))         # (n_c*n_i, 6)

    # Pair features — sparse; only rows in pf
    X_p = np.zeros((n_c*n_i, 5), dtype=float)
    cpos = {c: i for i,c in enumerate(customers)}
    ipos = {it: j for j,it in enumerate(catalog)}
    sub  = pf[pf['CustomerID'].isin(set(customers)) & pf['StockCode'].isin(set(catalog))]
    if len(sub):
        ci = sub['CustomerID'].map(cpos)
        ji = sub['StockCode'].map(ipos)
        valid = ci.notna() & ji.notna()
        flat  = ci[valid].astype(int).values * n_i + ji[valid].astype(int).values
        X_p[flat] = sub[['pair_purchases','pair_qty','pair_spend','pair_days_since','cat_match']].values[valid]

    X         = np.concatenate([X_c, X_i, X_p], axis=1)
    cust_ids  = np.repeat(customers, n_i)
    item_ids  = np.tile(catalog, n_c)
    return X, cust_ids, item_ids

def sample_training_pairs(target_window, catalog, cf, if_df, pf, n_neg=50, seed=42):
    rng2 = np.random.default_rng(seed)
    cset = set(catalog)
    pos_map = (target_window[target_window['StockCode'].isin(cset)]
               .groupby('CustomerID')['StockCode'].apply(set).to_dict())
    pos_map = {k:v for k,v in pos_map.items() if v}
    customers = list(pos_map.keys())
    if not customers:
        return None, None

    X_all, cust_ids, item_ids = _build_fm_numpy(customers, catalog, cf, if_df, pf)
    n_i = len(catalog)

    # Labels
    pos_set = {(cid,it) for cid,pos in pos_map.items() for it in pos}
    y_all   = np.array([(cust_ids[k],item_ids[k]) in pos_set for k in range(len(cust_ids))], dtype=int)

    # Keep all positives + n_neg negatives per customer
    keep = []
    for ci_idx, cid in enumerate(customers):
        base = ci_idx*n_i
        pos_l = np.where(y_all[base:base+n_i])[0] + base
        neg_l = np.where(y_all[base:base+n_i]==0)[0] + base
        keep.extend(pos_l.tolist())
        nd = min(n_neg, len(neg_l))
        if nd: keep.extend(rng2.choice(neg_l, size=nd, replace=False).tolist())

    keep = np.array(keep)
    X, y = X_all[keep], y_all[keep]
    print(f"  Samples: {len(y):,}  pos={y.sum():,}  neg={(y==0).sum():,}")
    return X, y

def rf_recommendations(rf, customers, catalog, cf, if_df, pf, batch=400):
    all_recs = []
    for i in range(0, len(customers), batch):
        bl = customers[i:i+batch]
        X, cids, iids = _build_fm_numpy(bl, catalog, cf, if_df, pf)
        scores = rf.predict_proba(X)[:,1]
        all_recs.append(pd.DataFrame({'CustomerID':cids,'StockCode':iids,'score':scores}))
    return pd.concat(all_recs, ignore_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_ranking(recs_df, pos_dict, Ks=(5,10,20)):
    results = {}
    for K in Ks:
        pl,rl,hl,nl,al = [],[],[],[],[]
        for cid,pos in pos_dict.items():
            top  = (recs_df[recs_df['CustomerID']==cid]
                    .sort_values('score',ascending=False).head(K)['StockCode'].tolist())
            if not top: continue
            hits = [1 if r in pos else 0 for r in top]
            nh,np_ = sum(hits),len(pos)
            pl.append(nh/K); rl.append(nh/np_ if np_ else 0); hl.append(1 if nh else 0)
            dcg  = sum(h/np.log2(i+2) for i,h in enumerate(hits))
            ih   = [1]*min(np_,K)+[0]*max(0,K-np_)
            idcg = sum(h/np.log2(i+2) for i,h in enumerate(ih))
            nl.append(dcg/idcg if idcg else 0)
            ap,nc = 0.0,0
            for i,h in enumerate(hits):
                if h: nc+=1; ap+=nc/(i+1)
            al.append(ap/min(np_,K) if min(np_,K) else 0)
        results[K] = {m: round(float(np.mean(v)),4) if v else 0
                      for m,v in [('Precision',pl),('Recall',rl),('HitRate',hl),('NDCG',nl),('MAP',al)]}
        results[K]['n_users'] = len(pl)
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# ITEM-ITEM CF
# ═══════════════════════════════════════════════════════════════════════════════
def item_item_cf(train_val, customers, catalog, top_k=20):
    cset     = set(catalog)
    all_c    = train_val['CustomerID'].unique()
    cust_idx = {c:i for i,c in enumerate(all_c)}
    item_idx = {it:j for j,it in enumerate(catalog)}
    sub      = train_val[train_val['StockCode'].isin(cset)].copy()
    rows_, cols_, data_ = [],[],[]
    for cid, grp in sub.groupby('CustomerID'):
        if cid not in cust_idx: continue
        ci = cust_idx[cid]
        for it in grp['StockCode'].unique():
            if it in item_idx:
                rows_.append(ci); cols_.append(item_idx[it]); data_.append(1)
    mat = csr_matrix((data_,(rows_,cols_)), shape=(len(all_c),len(catalog)))
    sim = cosine_similarity(mat.T, dense_output=True)   # (n_items, n_items) dense
    hist_map = sub.groupby('CustomerID')['StockCode'].apply(set).to_dict()
    recs = []
    for cid in customers:
        bought  = np.array([item_idx[it] for it in hist_map.get(cid,set()) if it in item_idx])
        if len(bought) == 0:
            for j,it in enumerate(catalog[:top_k]):
                recs.append({'CustomerID':cid,'StockCode':it,'score':float(len(catalog)-j)})
            continue
        sc = sim[bought,:].mean(axis=0)   # (n_items,) dense numpy array
        sc[bought] = -1.0
        for j in np.argsort(-sc)[:top_k]:
            recs.append({'CustomerID':cid,'StockCode':catalog[j],'score':float(sc[j])})
    return pd.DataFrame(recs)

# ═══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CI
# ═══════════════════════════════════════════════════════════════════════════════
def bootstrap_ci(recs_df, pos_dict, K=10, n_boot=200, seed=42):
    # Precompute top-K list per customer to avoid repeated DataFrame scans
    topK = (recs_df.sort_values('score', ascending=False)
                   .groupby('CustomerID').head(K)
                   .groupby('CustomerID')['StockCode'].apply(list).to_dict())
    rng2 = np.random.default_rng(seed)
    cids = list(pos_dict.keys())
    vr, vn = [], []
    for _ in range(n_boot):
        samp = rng2.choice(cids, len(cids), replace=True)
        r10, n10 = [], []
        for cid in samp:
            top  = topK.get(cid, [])
            pos  = pos_dict[cid]
            hits = [1 if t in pos else 0 for t in top]
            nh, np_ = sum(hits), len(pos)
            r10.append(nh/np_ if np_ else 0)
            dcg  = sum(h/np.log2(i+2) for i,h in enumerate(hits))
            ih   = [1]*min(np_,K)+[0]*max(0,K-np_)
            idcg = sum(h/np.log2(i+2) for i,h in enumerate(ih))
            n10.append(dcg/idcg if idcg else 0)
        vr.append(np.mean(r10)); vn.append(np.mean(n10))
    return (round(float(np.percentile(vr,2.5)),4), round(float(np.percentile(vr,97.5)),4)), \
           (round(float(np.percentile(vn,2.5)),4), round(float(np.percentile(vn,97.5)),4))

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
def savefig(name):
    p = os.path.join(FIG_DIR, name)
    plt.savefig(p, dpi=130, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved: {name}")

def fig01_txn_volume(df):
    wk = df.set_index('InvoiceDate').resample('W')['InvoiceNo'].count()
    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.fill_between(wk.index, wk.values, alpha=0.35, color=COLORS[0])
    ax.plot(wk.index, wk.values, color=COLORS[0], lw=1.5)
    ax.set_title('Fig 1 – Weekly Transaction Volume (Raw Data)', fontweight='bold')
    ax.set_xlabel('Week'); ax.set_ylabel('Transactions')
    plt.tight_layout(); savefig('fig01_txn_volume.png')

def fig02_top_items(train):
    top = train.groupby('Description')['InvoiceNo'].count().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10,4.5))
    ax.barh(top.index[::-1], top.values[::-1], color=COLORS[1])
    ax.set_xlabel('Transactions')
    ax.set_title('Fig 2 – Top 15 Items by Transaction Count (Train Set)', fontweight='bold')
    plt.tight_layout(); savefig('fig02_top_items.png')

def fig03_purchase_freq(train):
    freq = train.groupby('CustomerID')['InvoiceNo'].nunique()
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(freq.values, bins=40, color=COLORS[2], edgecolor='white', alpha=0.85)
    ax.axvline(freq.median(), color='red', ls='--', label=f'Median={freq.median():.0f}')
    ax.set_xlabel('Unique Invoices'); ax.set_ylabel('Customers')
    ax.set_title('Fig 3 – Customer Purchase Frequency (Train Set)', fontweight='bold')
    ax.legend(); plt.tight_layout(); savefig('fig03_purchase_freq.png')

def fig04_rfm(cf):
    fig, axes = plt.subplots(1,3, figsize=(12,4))
    for ax, col, label, color in zip(axes,
            ['recency_days','cust_txns','cust_spend'],
            ['Recency (days)','Frequency (invoices)','Monetary (spend)'], COLORS[:3]):
        ax.hist(cf[col].dropna(), bins=35, color=color, edgecolor='white', alpha=0.85)
        ax.set_xlabel(label); ax.set_ylabel('Customers')
        ax.set_title(f'RFM: {label}')
    fig.suptitle('Fig 4 – Customer RFM Feature Distributions', fontweight='bold')
    plt.tight_layout(); savefig('fig04_rfm_distributions.png')

def fig05_class_balance(y_train):
    vals   = [int((y_train==1).sum()), int((y_train==0).sum())]
    labels = [f'Positive ({vals[0]:,})', f'Negative ({vals[1]:,})']
    fig, ax = plt.subplots(figsize=(6,4))
    ax.pie(vals, labels=labels, autopct='%1.1f%%', colors=[COLORS[3],COLORS[4]], startangle=90)
    ax.set_title('Fig 5 – Training Class Balance (after Negative Sampling)', fontweight='bold')
    plt.tight_layout(); savefig('fig05_class_balance.png')

def fig06_feature_importance(rf):
    imp = rf.feature_importances_; idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh([FEAT_COLS[i] for i in idx], imp[idx], color=COLORS[0])
    ax.set_xlabel('Mean Decrease in Impurity')
    ax.set_title('Fig 6 – Random Forest Feature Importances', fontweight='bold')
    plt.tight_layout(); savefig('fig06_feature_importance.png')

def fig07_pk_rk(rf_m, pop_m, Ks=(5,10,20)):
    fig, (a1,a2) = plt.subplots(1,2, figsize=(11,4.5))
    for ax, mt in [(a1,'Precision'),(a2,'Recall')]:
        ax.plot(Ks,[rf_m[K][mt] for K in Ks],'o-',color=COLORS[0],label='RF',lw=2,ms=7)
        ax.plot(Ks,[pop_m[K][mt] for K in Ks],'s--',color=COLORS[2],label='Popularity',lw=2,ms=7)
        ax.set_xlabel('K'); ax.set_ylabel(f'{mt}@K')
        ax.set_title(f'Fig 7 – {mt}@K vs K', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(Ks)
    plt.tight_layout(); savefig('fig07_precision_recall_k.png')

def fig08_model_comparison(rf_m, pop_m, cf_m, Ks=(5,10,20)):
    mts = ['Precision','Recall','HitRate','NDCG']
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    for ax, mt in zip(axes.flat, mts):
        x,w = np.arange(len(Ks)),0.25
        ax.bar(x-w,[rf_m[K][mt] for K in Ks],w,label='RF',color=COLORS[0])
        ax.bar(x,  [cf_m[K][mt] for K in Ks],w,label='Item-CF',color=COLORS[1])
        ax.bar(x+w,[pop_m[K][mt] for K in Ks],w,label='Popularity',color=COLORS[2])
        ax.set_xticks(x); ax.set_xticklabels([f'K={k}' for k in Ks])
        ax.set_ylabel(mt); ax.set_title(f'{mt}@K', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(axis='y',alpha=0.3)
    fig.suptitle('Fig 8 – Model Comparison: RF vs Item-CF vs Popularity', fontweight='bold')
    plt.tight_layout(); savefig('fig08_model_comparison.png')

def fig09_score_dist(recs_df, pos_dict):
    pos_set = {(cid, it) for cid, pos in pos_dict.items() for it in pos}
    df2 = recs_df.sample(min(40000, len(recs_df)), random_state=42)  # cap for speed
    label = np.array([1 if (r.CustomerID, r.StockCode) in pos_set else 0
                      for r in df2.itertuples()], dtype=int)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(df2['score'].values[label==0],bins=50,alpha=0.6,color=COLORS[4],label='Negative',density=True)
    ax.hist(df2['score'].values[label==1],bins=50,alpha=0.75,color=COLORS[3],label='Positive',density=True)
    ax.set_xlabel('RF Score (P(purchase))'); ax.set_ylabel('Density')
    ax.set_title('Fig 9 – RF Score Distribution: Positives vs Negatives', fontweight='bold')
    ax.legend(); plt.tight_layout(); savefig('fig09_score_distribution.png')

def fig10_catalog_coverage(catalog, pos_dict):
    all_f = {it for pos in pos_dict.values() for it in pos}
    ins, out = len(all_f & set(catalog)), len(all_f - set(catalog))
    fig, ax  = plt.subplots(figsize=(6,4))
    ax.pie([ins,out], labels=[f'In Catalog ({ins})',f'Not in Catalog ({out})'],
           autopct='%1.1f%%', colors=[COLORS[0],COLORS[6]], startangle=90)
    ax.set_title(f'Fig 10 – Candidate Recall\n(Catalog size={len(catalog):,} items)', fontweight='bold')
    plt.tight_layout(); savefig('fig10_catalog_coverage.png')

def fig11_case_audit(cases):
    fig, axes = plt.subplots(5,1, figsize=(12,14))
    for ax, case in zip(axes, cases):
        ax.axis('off')
        color = '#d4edda' if case['hits']>0 else '#f8d7da'
        txt   = (f"Customer: {case['cid']}   Segment: {case['seg']}\n"
                 f"  Past purchases :  {', '.join(case['past'][:5])}\n"
                 f"  RF Top-5 recs  :  {', '.join(case['recs'][:5])}\n"
                 f"  Actual (future):  {', '.join(case['actual'][:5])}\n"
                 f"  Hits in Top-5  :  {case['hits']}/{min(len(case['actual']),5)}")
        ax.text(0.01,0.5,txt,transform=ax.transAxes,fontsize=9.5,va='center',
                fontfamily='monospace',bbox=dict(boxstyle='round,pad=0.4',facecolor=color,alpha=0.9))
    fig.suptitle('Fig 11 – Five-Case Recommendation Audit', fontweight='bold')
    plt.tight_layout(); savefig('fig11_case_audit.png')

def fig12_hitrate_k(rf_m, pop_m, cf_m, Ks=(5,10,20)):
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(Ks,[rf_m[K]['HitRate'] for K in Ks],'o-',color=COLORS[0],label='RF',lw=2,ms=8)
    ax.plot(Ks,[cf_m[K]['HitRate'] for K in Ks],'^-',color=COLORS[1],label='Item-CF',lw=2,ms=8)
    ax.plot(Ks,[pop_m[K]['HitRate'] for K in Ks],'s--',color=COLORS[2],label='Popularity',lw=2,ms=8)
    ax.set_xlabel('K'); ax.set_ylabel('HitRate@K')
    ax.set_title('Fig 12 – HitRate@K Comparison Across Models', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(Ks)
    plt.tight_layout(); savefig('fig12_hitrate_k.png')

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    print("="*70)
    print("LAB 07  |  Recommendation System (Random Forest)  |  23MID0021")
    print("="*70)

    # 1. Generate + save raw dataset
    raw_df = generate_synthetic_retail(RNG)
    ds_path = os.path.join(OUT_DIR,'dataset.csv')
    raw_df.to_csv(ds_path, index=False)
    sha = hashlib.sha256(open(ds_path,'rb').read()).hexdigest()
    print(f"\n[Dataset] rows={len(raw_df):,}  sha256={sha[:32]}...")
    desc_map = raw_df.drop_duplicates('StockCode').set_index('StockCode')['Description'].to_dict()

    # 2. Clean
    df, clean_stats = clean_data(raw_df)

    # 3. Split
    train, val, test, mf = split_data(df)
    json.dump(mf, open(os.path.join(ART_DIR,'split_manifest.json'),'w'), indent=2)

    # 4. Popularity baseline
    pop = (train.groupby('StockCode')['CustomerID'].nunique()
               .rename('buyers').sort_values(ascending=False).reset_index())

    # 5. Candidate catalog
    buyers  = train.groupby('StockCode')['CustomerID'].nunique()
    catalog = buyers[buyers>=3].sort_values(ascending=False).head(800).index.tolist()
    cand_policy = {'min_buyers':3,'max_catalog':800,'actual':len(catalog)}
    json.dump(cand_policy, open(os.path.join(ART_DIR,'candidate_policy.json'),'w'), indent=2)
    cset = set(catalog)
    test_items = set(test['StockCode'].unique())
    cand_recall = len(test_items & cset) / len(test_items) if test_items else 0
    cand_recall_stats = {'test_unique_items':len(test_items),'catalog_size':len(catalog),
                         'hits':len(test_items&cset),'candidate_recall':round(cand_recall,4)}
    print(f"\n[Catalog] size={len(catalog)}  candidate_recall={cand_recall:.2%}")
    pd.DataFrame([cand_recall_stats]).to_csv(
        os.path.join(OUT_DIR,'23MID0021_Lab07_Candidate_Recall.csv'), index=False)

    # 6. Feature engineering (from training history only)
    print("\n[Features — train history up to train_end]")
    cf_tr, if_tr, pf_tr = engineer_features(train, catalog, mf['train_end'])

    # Feature schema
    feat_schema = {'feature_names':FEAT_COLS,
                   'customer':['recency_days','cust_txns','cust_items','cust_spend','cust_uniq','pref_cat_enc'],
                   'item':['item_txns','item_buyers','item_avg_price','item_recency_days','item_cat_enc','item_pop_rank'],
                   'pair':['pair_purchases','pair_qty','pair_spend','pair_days_since','cat_match']}
    json.dump(feat_schema, open(os.path.join(ART_DIR,'feature_schema.json'),'w'), indent=2)
    print(f"  Customers={len(cf_tr)}  Items={len(if_tr)}  Pairs={len(pf_tr)}")

    # 7. Training data (target = val positives, features from train history)
    print("\n[Sampling training data — positives=val, negatives=catalog]")
    X_tr, y_tr = sample_training_pairs(val, catalog, cf_tr, if_tr, pf_tr, n_neg=50)

    # 8. Train RF
    print("\n[Train RandomForest seed=42]")
    t_rf = time.time()
    rf = RandomForestClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=2,
                                 class_weight='balanced_subsample', n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    print(f"  Done in {time.time()-t_rf:.1f}s")

    # Validation evaluation (model selection — NOT test)
    print("\n[Validation scoring]")
    val_pos = (val[val['StockCode'].isin(cset)].groupby('CustomerID')['StockCode'].apply(set).to_dict())
    val_pos = {k:v for k,v in val_pos.items() if v and k in set(cf_tr['CustomerID'])}
    val_customers = list(val_pos.keys())[:600]  # cap for speed
    val_recs = rf_recommendations(rf, val_customers, catalog, cf_tr, if_tr, pf_tr)
    val_m = evaluate_ranking(val_recs, {k:val_pos[k] for k in val_customers})
    for K in [5,10,20]:
        m = val_m[K]
        print(f"  VAL K={K:2d}  P={m['Precision']:.4f}  R={m['Recall']:.4f}  HR={m['HitRate']:.4f}  NDCG={m['NDCG']:.4f}")

    # 9. Test features (train+val history)
    print("\n[Test features — train+val history]")
    tv = pd.concat([train,val],ignore_index=True)
    cf_tv, if_tv, pf_tv = engineer_features(tv, catalog, mf['val_end'])

    test_pos = (test[test['StockCode'].isin(cset)].groupby('CustomerID')['StockCode'].apply(set).to_dict())
    test_pos = {k:v for k,v in test_pos.items() if v}
    test_customers = list(test_pos.keys())
    print(f"  Test customers with >=1 catalog purchase: {len(test_customers)}")

    # RF recommendations
    print("\n[RF Test recommendations]")
    rf_recs = rf_recommendations(rf, test_customers, catalog, cf_tv, if_tv, pf_tv)
    rf_m    = evaluate_ranking(rf_recs, test_pos)

    # Popularity recommendations
    pop_catalog = [it for it in pop['StockCode'] if it in cset]
    pop_hist    = tv.groupby('CustomerID')['StockCode'].apply(set).to_dict()
    pop_recs_rows = []
    for cid in test_customers:
        bought = pop_hist.get(cid, set())
        rank=0
        for it in pop_catalog:
            if it not in bought:
                pop_recs_rows.append({'CustomerID':cid,'StockCode':it,'score':float(len(pop_catalog)-rank)})
                rank+=1
            if rank>=20: break
    pop_recs_df = pd.DataFrame(pop_recs_rows)
    pop_m = evaluate_ranking(pop_recs_df, test_pos)

    # Item-item CF
    print("\n[Item-item CF recommendations]")
    cf_recs = item_item_cf(tv, test_customers, catalog, top_k=20)
    cf_m    = evaluate_ranking(cf_recs, test_pos)

    # Print test metrics
    print("\n"+"="*70+"\nTEST METRICS\n"+"="*70)
    print(f"  {'Model':<14} {'P@5':>7} {'P@10':>7} {'P@20':>7} {'R@10':>7} {'HR@10':>8} {'NDCG@10':>9}")
    for lbl,m in [('Popularity',pop_m),('RandomForest',rf_m),('Item-CF',cf_m)]:
        print(f"  {lbl:<14} {m[5]['Precision']:>7.4f} {m[10]['Precision']:>7.4f} {m[20]['Precision']:>7.4f} "
              f"{m[10]['Recall']:>7.4f} {m[10]['HitRate']:>8.4f} {m[10]['NDCG']:>9.4f}")

    # 10. Multi-seed + bootstrap
    print("\n[Multi-seed comparison]")
    seed_results = {}
    for seed in [42,123,777]:
        rf_s = RandomForestClassifier(n_estimators=200, max_features='sqrt', min_samples_leaf=2,
                                       class_weight='balanced_subsample', n_jobs=-1, random_state=seed)
        rf_s.fit(X_tr, y_tr)
        r_s  = rf_recommendations(rf_s, test_customers, catalog, cf_tv, if_tv, pf_tv)
        m_s  = evaluate_ranking(r_s, test_pos)
        seed_results[seed] = {'Recall@10':m_s[10]['Recall'],'NDCG@10':m_s[10]['NDCG']}
        print(f"  seed={seed}  Recall@10={m_s[10]['Recall']:.4f}  NDCG@10={m_s[10]['NDCG']:.4f}")

    ci_r, ci_n    = bootstrap_ci(rf_recs, test_pos)
    ci_r_cf, ci_n_cf = bootstrap_ci(cf_recs, test_pos, seed=1)
    print(f"  RF  Bootstrap 95% CI  Recall@10={ci_r}  NDCG@10={ci_n}")
    print(f"  CF  Bootstrap 95% CI  Recall@10={ci_r_cf}  NDCG@10={ci_n_cf}")

    # PR-AUC
    X_te_s, y_te_s = sample_training_pairs(test, catalog, cf_tv, if_tv, pf_tv, n_neg=50, seed=99)
    probs = rf.predict_proba(X_te_s)[:,1]
    pr_c, rc_c, _ = precision_recall_curve(y_te_s, probs)
    prauc = sk_auc(rc_c, pr_c)
    print(f"  PR-AUC (test diagnostic): {prauc:.4f}")

    # 11. Case audit
    print("\n[Case audit — 5 test customers]")
    qtile = cf_tv['cust_txns'].quantile([0.25,0.5,0.75]).values
    cases = []
    for cid in test_customers[:5]:
        txv = cf_tv[cf_tv['CustomerID']==cid]['cust_txns'].values
        tv2 = txv[0] if len(txv) else 1
        seg = ('Heavy' if tv2>=qtile[2] else 'Regular' if tv2>=qtile[1] else 'Occasional' if tv2>=qtile[0] else 'One-time')
        past_s  = [desc_map.get(it,it)[:28] for it in list(tv[tv['CustomerID']==cid]['StockCode'].unique())[:5]]
        recs_s  = [desc_map.get(it,it)[:28] for it in (rf_recs[rf_recs['CustomerID']==cid].sort_values('score',ascending=False).head(5)['StockCode'].tolist())]
        actual  = list(test_pos.get(cid,set()))[:5]
        actual_s= [desc_map.get(it,it)[:28] for it in actual]
        hits    = sum(1 for r in (rf_recs[rf_recs['CustomerID']==cid].sort_values('score',ascending=False).head(5)['StockCode'].tolist()) if r in test_pos.get(cid,set()))
        cases.append({'cid':cid,'seg':seg,'past':past_s,'recs':recs_s,'actual':actual_s,'hits':hits})
        print(f"  {cid} ({seg}) | past={len(past_s)} | recs={len(recs_s)} | hits@5={hits}/{min(len(actual),5)}")

    # 12. Save model
    mdl_path = os.path.join(MDL_DIR,'random_forest.joblib')
    joblib.dump(rf, mdl_path)
    assert joblib.load(mdl_path).n_estimators==rf.n_estimators
    print(f"\n[Model saved+reloaded OK] {mdl_path}")

    # 13. Save CSVs
    rows_ = []
    for lbl,m in [('Popularity',pop_m),('RandomForest',rf_m),('ItemItemCF',cf_m)]:
        for K in [5,10,20]:
            rows_.append({'Model':lbl,'K':K,**{mk:m[K][mk] for mk in ['Precision','Recall','HitRate','NDCG','MAP','n_users']}})
    pd.DataFrame(rows_).to_csv(os.path.join(OUT_DIR,'23MID0021_Lab07_Ranking_Metrics.csv'),index=False)

    top10 = (rf_recs.sort_values(['CustomerID','score'],ascending=[True,False])
                    .groupby('CustomerID').head(10).copy())
    top10['rank']        = top10.groupby('CustomerID').cumcount()+1
    top10['Description'] = top10['StockCode'].map(desc_map)
    top10.to_csv(os.path.join(OUT_DIR,'23MID0021_Lab07_Recommendations.csv'),index=False)

    err_rows = [{'CustomerID':c['cid'],'Segment':c['seg'],'Hits_at_5':c['hits'],
                 'Past_Items':len(c['past']),'Future_Items':len(c['actual']),
                 'Top5_Recs':'; '.join(c['recs']),'Actual_Future':'; '.join(c['actual'])} for c in cases]
    pd.DataFrame(err_rows).to_csv(os.path.join(OUT_DIR,'23MID0021_Lab07_Error_Analysis.csv'),index=False)
    print("\n[All CSVs saved]")

    # 14. Figures
    print("\n[Generating 12 figures]")
    fig01_txn_volume(df)
    fig02_top_items(train)
    fig03_purchase_freq(train)
    fig04_rfm(cf_tv)
    fig05_class_balance(y_tr)
    fig06_feature_importance(rf)
    fig07_pk_rk(rf_m, pop_m)
    fig08_model_comparison(rf_m, pop_m, cf_m)
    fig09_score_dist(rf_recs, test_pos)
    fig10_catalog_coverage(catalog, test_pos)
    fig11_case_audit(cases)
    fig12_hitrate_k(rf_m, pop_m, cf_m)

    # Summary
    elapsed = time.time()-t0
    print("\n"+"="*70+"\nFINAL SUMMARY\n"+"="*70)
    print(f"  SHA-256:           {sha[:32]}...")
    print(f"  Clean rows:        {len(df):,}")
    print(f"  Catalog size:      {len(catalog):,} items")
    print(f"  Candidate recall:  {cand_recall:.2%}")
    print(f"  Test customers:    {len(test_customers)}")
    print(f"  PR-AUC:            {prauc:.4f}")
    print(f"\n  {'Model':<14} {'P@5':>7} {'P@10':>7} {'P@20':>7} {'R@10':>7} {'HR@10':>8} {'NDCG@10':>9}")
    for lbl,m in [('Popularity',pop_m),('RF',rf_m),('Item-CF',cf_m)]:
        print(f"  {lbl:<14} {m[5]['Precision']:>7.4f} {m[10]['Precision']:>7.4f} {m[20]['Precision']:>7.4f} "
              f"{m[10]['Recall']:>7.4f} {m[10]['HitRate']:>8.4f} {m[10]['NDCG']:>9.4f}")
    print(f"\n  Seeds 42/123/777: Recall@10 = "
          f"{seed_results[42]['Recall@10']:.4f} / {seed_results[123]['Recall@10']:.4f} / {seed_results[777]['Recall@10']:.4f}")
    print(f"  RF  95% CI  Recall@10 = {ci_r[0]:.4f} - {ci_r[1]:.4f}")
    print(f"  RF  95% CI  NDCG@10   = {ci_n[0]:.4f} - {ci_n[1]:.4f}")
    print(f"  CF  95% CI  Recall@10 = {ci_r_cf[0]:.4f} - {ci_r_cf[1]:.4f}")
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Outputs  -> {OUT_DIR}")
    print(f"  Figures  -> {FIG_DIR}")
    print(f"  Model    -> {mdl_path}")
