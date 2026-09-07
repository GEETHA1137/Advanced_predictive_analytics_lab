"""
Lab 06 Pipeline: Time-Series Analysis and Forecasting of Reported Crime Incidents
AR and ARIMA Models on Three Urban Crime Datasets
Geetha Priya S — 23MID0021 | MDI3003 Advanced Predictive Analytics
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

BASE     = r'c:\Users\I768951\OneDrive - SAP SE\SAP\Exam'
FIG_DIR  = os.path.join(BASE, 'lab06_figs')
OUT_DIR  = os.path.join(BASE, 'lab06_outputs')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SEED       = 42
TEST_WEEKS = 52   # hold-out: last 52 weeks (~1 year)
ROLL_FOLDS = 6    # rolling-origin folds
np.random.seed(SEED)

COLORS = {
    'D1 Chicago District 1': '#2471A3',
    'D2 NYPD Manhattan':     '#1E8449',
    'D3 SFPD Mission Dist':  '#CB4335',
}
MODEL_COLORS = {'Naive': '#95A5A6', 'AR': '#E67E22', 'ARIMA': '#8E44AD'}

# =============================================================================
# 1.  SYNTHETIC DATA GENERATION
# =============================================================================

def generate_crime_series(n_weeks, base_mean, ar_coefs, noise_frac,
                          covid_start=115, covid_len=35, covid_depth=0.28,
                          seed_off=0):
    """
    Realistic synthetic weekly crime count series.
    Combines annual seasonality, COVID dip, AR dependence, and Gaussian noise.
    All fitting is done on training data only — the test period is held out.
    """
    rng = np.random.default_rng(SEED + seed_off)
    t   = np.arange(n_weeks)

    # Annual (52-week) seasonal component — summer peaks
    seasonal = 0.12 * np.sin(2 * np.pi * t / 52 + 1.2)

    # Slight downward drift post-2020
    trend = -0.00010 * t

    # COVID-19 dip: smooth bell-curve shape March–Aug 2020
    covid = np.zeros(n_weeks)
    for i in range(covid_start, min(covid_start + covid_len, n_weeks)):
        prog = (i - covid_start) / covid_len
        covid[i] = -covid_depth * np.sin(np.pi * prog)

    # Stationary AR noise component
    p     = len(ar_coefs)
    innov = rng.normal(0, 0.06, n_weeks)
    ar    = np.zeros(n_weeks)
    for i in range(p, n_weeks):
        ar[i] = sum(ar_coefs[j] * ar[i - j - 1] for j in range(p)) + innov[i]

    # Observation noise
    obs_noise = rng.normal(0, noise_frac, n_weeks)

    log_vals = np.log(base_mean) + seasonal + trend + covid + ar + obs_noise
    counts   = np.exp(log_vals).clip(min=1)
    return np.round(counts).astype(int)


print("=" * 60)
print("Lab 06 Pipeline — Crime Forecasting (AR / ARIMA)")
print("=" * 60)
print("\n[1] Generating synthetic weekly crime series (2018-2023)...")

start = pd.Timestamp('2018-01-07')
dates = pd.date_range(start, periods=312, freq='W-SUN')
N     = len(dates)

chi_raw = generate_crime_series(N, 85,  [0.45, 0.20],       0.09, covid_depth=0.28, seed_off=0)
nyp_raw = generate_crime_series(N, 452, [0.50, 0.18, 0.10], 0.08, covid_depth=0.32, seed_off=1)
sfp_raw = generate_crime_series(N, 120, [0.38, 0.22],       0.11, covid_depth=0.22, seed_off=2)

datasets = {
    'D1 Chicago District 1': pd.Series(chi_raw, index=dates, name='crime_count'),
    'D2 NYPD Manhattan':     pd.Series(nyp_raw, index=dates, name='crime_count'),
    'D3 SFPD Mission Dist':  pd.Series(sfp_raw, index=dates, name='crime_count'),
}

for dname, s in datasets.items():
    print(f"  {dname}: n={len(s)}, mean={s.mean():.1f}, std={s.std():.1f}, "
          f"min={s.min()}, max={s.max()}")

# =============================================================================
# 2.  EVALUATION HELPERS
# =============================================================================

def mae(actual, pred):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))

def rmse(actual, pred):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))

def naive_forecast(train, n):
    return np.full(n, float(train.iloc[-1]))

def fit_ar_model(train, max_lag=12):
    best_aic, best_lag = np.inf, 1
    for lag in range(1, min(max_lag + 1, len(train) // 4)):
        try:
            res = AutoReg(train.values, lags=lag, old_names=False).fit()
            if res.aic < best_aic:
                best_aic, best_lag = res.aic, lag
        except Exception:
            pass
    res = AutoReg(train.values, lags=best_lag, old_names=False).fit()
    return res, best_lag

def ar_predict_oos(res, train_len, test_len):
    pred = res.predict(start=train_len, end=train_len + test_len - 1)
    return np.asarray(pred).clip(min=0)

def fit_arima_model(train, max_p=4, max_q=4):
    # Determine differencing order from ADF on training data
    adf_res  = adfuller(train.values, autolag='AIC')
    adf_stat = float(adf_res[0])
    adf_p    = float(adf_res[1])
    d_order  = 0 if adf_p <= 0.05 else 1

    best_aic, best_ord, best_res = np.inf, (1, d_order, 1), None
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                r = ARIMA(train.values, order=(p, d_order, q)).fit()
                if r.aic < best_aic:
                    best_aic, best_ord, best_res = r.aic, (p, d_order, q), r
            except Exception:
                pass

    resid_clean = pd.Series(best_res.resid).dropna()
    lb   = acorr_ljungbox(resid_clean, lags=[10], return_df=True)
    lb_p = float(lb['lb_pvalue'].values[0])

    return best_res, best_ord, adf_stat, adf_p, lb_p

def arima_predict_oos(res, test_len):
    fc = res.forecast(steps=test_len)
    return np.asarray(fc).clip(min=0)

def rolling_origin_cv(series, test_len, n_folds, ar_lag, arima_order):
    """Rolling-origin backtesting with fixed model orders."""
    N = len(series)
    min_train = max(test_len * 3, 80)
    origins   = np.linspace(min_train, N - test_len - 1, n_folds).astype(int)

    ar_maes, arima_maes = [], []
    for origin in origins:
        train_f  = series.iloc[:origin]
        actual   = series.iloc[origin:origin + test_len].values
        try:
            ar_r  = AutoReg(train_f.values, lags=ar_lag, old_names=False).fit()
            ar_p  = ar_r.predict(start=len(train_f), end=len(train_f)+test_len-1)
            ar_maes.append(mae(actual, np.asarray(ar_p).clip(0)))
        except Exception:
            pass
        try:
            arima_r = ARIMA(train_f.values, order=arima_order).fit()
            arima_p = arima_r.forecast(steps=test_len)
            arima_maes.append(mae(actual, np.asarray(arima_p).clip(0)))
        except Exception:
            pass

    return np.array(ar_maes), np.array(arima_maes)

# =============================================================================
# 3.  FIT MODELS ON EACH DATASET
# =============================================================================

print("\n[2] Fitting models (Naive / AR / ARIMA) on each dataset...")
res_all = {}

for ds_name, series in datasets.items():
    print(f"\n  >>> {ds_name}")
    train = series.iloc[:-TEST_WEEKS]
    test  = series.iloc[-TEST_WEEKS:]

    # ADF on training data
    adf_train      = adfuller(train.values, autolag='AIC')
    adf_full       = adfuller(series.values, autolag='AIC')

    # Naive
    nv_pred  = naive_forecast(train, TEST_WEEKS)
    nv_mae   = mae(test.values, nv_pred)
    nv_rmse  = rmse(test.values, nv_pred)

    # AR
    ar_res, ar_lag = fit_ar_model(train, max_lag=12)
    ar_pred  = ar_predict_oos(ar_res, len(train), TEST_WEEKS)
    ar_mae   = mae(test.values, ar_pred)
    ar_rmse  = rmse(test.values, ar_pred)

    # ARIMA
    arima_res, arima_ord, adf_s, adf_p, lb_p = fit_arima_model(train)
    arima_pred = arima_predict_oos(arima_res, TEST_WEEKS)
    arima_mae  = mae(test.values, arima_pred)
    arima_rmse = rmse(test.values, arima_pred)

    # Rolling-origin CV
    ro_ar_maes, ro_arima_maes = rolling_origin_cv(series, 8, ROLL_FOLDS,
                                                   ar_lag, arima_ord)

    res_all[ds_name] = dict(
        series=series, train=train, test=test,
        adf_stat_full=float(adf_full[0]),   adf_p_full=float(adf_full[1]),
        adf_stat_train=adf_s,               adf_p_train=adf_p,
        ar_lag=ar_lag, ar_res=ar_res,
        ar_aic=float(ar_res.aic),           ar_bic=float(ar_res.bic),
        arima_ord=arima_ord, arima_res=arima_res,
        arima_aic=float(arima_res.aic),     arima_bic=float(arima_res.bic),
        lb_p=lb_p,
        nv_pred=nv_pred, ar_pred=ar_pred, arima_pred=arima_pred,
        nv_mae=nv_mae,   nv_rmse=nv_rmse,
        ar_mae=ar_mae,   ar_rmse=ar_rmse,
        arima_mae=arima_mae, arima_rmse=arima_rmse,
        ro_ar_maes=ro_ar_maes, ro_arima_maes=ro_arima_maes,
    )
    r = res_all[ds_name]
    print(f"    Naive:         MAE={nv_mae:.2f}  RMSE={nv_rmse:.2f}")
    print(f"    AR({ar_lag}):        MAE={ar_mae:.2f}  RMSE={ar_rmse:.2f}  AIC={r['ar_aic']:.1f}  BIC={r['ar_bic']:.1f}")
    print(f"    ARIMA{arima_ord}: MAE={arima_mae:.2f}  RMSE={arima_rmse:.2f}  AIC={r['arima_aic']:.1f}  BIC={r['arima_bic']:.1f}")
    print(f"    ADF(train): stat={adf_s:.4f}, p={adf_p:.4f}  |  Ljung-Box p={lb_p:.4f}")


# =============================================================================
# 4.  FIGURE GENERATION (12 figures)
# =============================================================================
print("\n[3] Generating figures...")

# ── Figure 01: Raw weekly crime series (3-panel) ─────────────────────────────
print("  fig01: raw series 3-panel")
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
fig.suptitle('Weekly Reported Crime Counts — Three Urban Datasets (2018–2023)',
             fontsize=14, fontweight='bold', y=1.00)

for ax, (dname, r) in zip(axes, res_all.items()):
    s       = r['series']
    split_d = r['test'].index[0]
    col     = COLORS[dname]
    ax.fill_between(s.index, s.values, alpha=0.18, color=col)
    ax.plot(s.index, s.values, color=col, linewidth=1.3, label='Observed')
    ax.axvline(split_d, color='red', linestyle='--', linewidth=1.5,
               label=f'Train/Test split ({split_d.strftime("%b %Y")})')
    ax.set_title(dname, fontsize=11, fontweight='bold', color=col)
    ax.set_ylabel('Incident Count', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Annotate COVID period
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-09-01'),
               alpha=0.09, color='red', label='COVID-19 period')

axes[-1].set_xlabel('Date', fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig01_raw_series.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 02: Stationarity inspection (rolling mean / std) — Chicago ────────
print("  fig02: rolling mean/std stationarity")
chi_s = res_all['D1 Chicago District 1']['series']
roll_mean = chi_s.rolling(12).mean()
roll_std  = chi_s.rolling(12).std()

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig.suptitle('D1 Chicago: Rolling Statistics for Stationarity Inspection\n(12-week window)',
             fontsize=13, fontweight='bold')

axes[0].plot(chi_s.index, chi_s.values, color='#2471A3', alpha=0.55, linewidth=1.0, label='Observed')
axes[0].plot(roll_mean.index, roll_mean.values, color='#922B21', linewidth=2.0, label='12-week rolling mean')
axes[0].set_ylabel('Incident Count', fontsize=10)
axes[0].legend(fontsize=9); axes[0].grid(axis='y', alpha=0.3)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

axes[1].plot(roll_std.index, roll_std.values, color='#1E8449', linewidth=1.8, label='12-week rolling std')
axes[1].axhline(roll_std.mean(), color='gray', linestyle=':', linewidth=1.2, label=f'Mean std={roll_std.mean():.1f}')
axes[1].set_ylabel('Rolling Std', fontsize=10)
axes[1].set_xlabel('Date', fontsize=10)
axes[1].legend(fontsize=9); axes[1].grid(axis='y', alpha=0.3)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig02_stationarity.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 03: ACF / PACF — Chicago training data ────────────────────────────
print("  fig03: ACF/PACF")
chi_train = res_all['D1 Chicago District 1']['train']
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
fig.suptitle('D1 Chicago: Sample ACF and PACF of Training Series\n(Used to identify AR lag order)',
             fontsize=13, fontweight='bold')
plot_acf(chi_train.values, lags=40, ax=axes[0], color='#2471A3', title='')
axes[0].set_title('Autocorrelation Function (ACF)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Lag (weeks)', fontsize=10)
axes[0].set_ylabel('ACF', fontsize=10)
plot_pacf(chi_train.values, lags=20, ax=axes[1], method='ywm', color='#E67E22', title='')
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Lag (weeks)', fontsize=10)
axes[1].set_ylabel('PACF', fontsize=10)
for ax in axes:
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig03_acf_pacf.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 04: ADF test summary ───────────────────────────────────────────────
print("  fig04: ADF test summary")
ds_labels  = ['Chicago D1', 'NYPD D2', 'SFPD D3']
ds_keys    = list(res_all.keys())
adf_stats  = [res_all[k]['adf_stat_train'] for k in ds_keys]
adf_ps     = [res_all[k]['adf_p_train']    for k in ds_keys]
crit_5pct  = [-2.87] * 3   # approximate ADF 5% critical value

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Augmented Dickey-Fuller (ADF) Stationarity Test — Training Sets',
             fontsize=13, fontweight='bold')

x = np.arange(3)
c_list = list(COLORS.values())
bars = axes[0].bar(x, adf_stats, color=c_list, alpha=0.85, edgecolor='white', width=0.5)
axes[0].axhline(-2.87, color='red', linestyle='--', linewidth=1.5, label='5% critical (−2.87)')
axes[0].axhline(-3.44, color='darkorange', linestyle=':', linewidth=1.2, label='1% critical (−3.44)')
for bar, val in zip(bars, adf_stats):
    axes[0].text(bar.get_x() + bar.get_width()/2, val - 0.15,
                 f'{val:.3f}', ha='center', va='top', fontsize=9, fontweight='bold', color='white')
axes[0].set_xticks(x); axes[0].set_xticklabels(ds_labels, fontsize=10)
axes[0].set_ylabel('ADF Test Statistic', fontsize=10)
axes[0].set_title('ADF Statistic (more negative = more stationary)', fontsize=10)
axes[0].legend(fontsize=9)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

bar_colors = ['#27AE60' if p <= 0.05 else '#E74C3C' for p in adf_ps]
bars2 = axes[1].bar(x, adf_ps, color=bar_colors, alpha=0.85, edgecolor='white', width=0.5)
axes[1].axhline(0.05, color='red', linestyle='--', linewidth=1.5, label='α=0.05 threshold')
for bar, val in zip(bars2, adf_ps):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1].set_xticks(x); axes[1].set_xticklabels(ds_labels, fontsize=10)
axes[1].set_ylabel('p-value', fontsize=10)
axes[1].set_title('ADF p-value (< 0.05 = stationary)', fontsize=10)
axes[1].legend(fontsize=9)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig04_adf_summary.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 05: Forecast comparison — Chicago ─────────────────────────────────
print("  fig05: forecast comparison Chicago")
r_chi   = res_all['D1 Chicago District 1']
test_idx = r_chi['test'].index

fig, ax = plt.subplots(figsize=(13, 5))
# Show last 40 training weeks for context
ctx = r_chi['train'].iloc[-40:]
ax.plot(ctx.index, ctx.values, color='#2471A3', linewidth=1.6, label='Training data (last 40 weeks)')
ax.plot(test_idx, r_chi['test'].values, color='black', linewidth=2.2, label='Actual (test)', zorder=5)
ax.plot(test_idx, r_chi['nv_pred'],    color=MODEL_COLORS['Naive'], linewidth=1.5,
        linestyle='--', label=f"Naive  (MAE={r_chi['nv_mae']:.1f})")
ax.plot(test_idx, r_chi['ar_pred'],    color=MODEL_COLORS['AR'],    linewidth=1.8,
        linestyle='-',  label=f"AR({r_chi['ar_lag']})    (MAE={r_chi['ar_mae']:.1f})")
ax.plot(test_idx, r_chi['arima_pred'], color=MODEL_COLORS['ARIMA'], linewidth=1.8,
        linestyle='-.',  label=f"ARIMA{r_chi['arima_ord']} (MAE={r_chi['arima_mae']:.1f})")
ax.axvline(test_idx[0], color='red', linestyle=':', linewidth=1.4, alpha=0.7, label='Train/Test boundary')
ax.set_title('D1 Chicago District 1 — Test-Period Forecast Comparison',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date', fontsize=10); ax.set_ylabel('Weekly Crime Count', fontsize=10)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig05_forecast_chicago.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 06: ARIMA residuals — Chicago ──────────────────────────────────────
print("  fig06: ARIMA residuals Chicago")
arima_res_chi = r_chi['arima_res']
resid = pd.Series(arima_res_chi.resid).dropna()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(f'D1 Chicago: ARIMA{r_chi["arima_ord"]} Residual Diagnostics',
             fontsize=13, fontweight='bold')

axes[0].plot(resid.values, color='#2471A3', linewidth=1.0, alpha=0.8)
axes[0].axhline(0, color='red', linestyle='--', linewidth=1.2)
axes[0].set_title('Residual Time Plot', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Week index'); axes[0].set_ylabel('Residual')
axes[0].grid(axis='y', alpha=0.3)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

plot_acf(resid.values, lags=20, ax=axes[1], color='#E67E22', title='')
axes[1].set_title('Residual ACF', fontsize=10, fontweight='bold')
axes[1].set_xlabel('Lag (weeks)'); axes[1].set_ylabel('ACF')
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

axes[2].hist(resid.values, bins=25, color='#8E44AD', alpha=0.75, edgecolor='white', density=True)
import scipy.stats as sp_stats
x_r = np.linspace(resid.min(), resid.max(), 200)
axes[2].plot(x_r, sp_stats.norm.pdf(x_r, resid.mean(), resid.std()),
             color='red', linewidth=2, label='Normal fit')
axes[2].set_title('Residual Histogram', fontsize=10, fontweight='bold')
axes[2].set_xlabel('Residual value'); axes[2].set_ylabel('Density')
axes[2].legend(fontsize=9)
axes[2].spines['top'].set_visible(False); axes[2].spines['right'].set_visible(False)
lb_chi = r_chi['lb_p']
axes[2].text(0.97, 0.94, f'Ljung-Box p={lb_chi:.4f}', transform=axes[2].transAxes,
             ha='right', va='top', fontsize=8.5,
             color='green' if lb_chi > 0.05 else 'red',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig06_residuals_arima.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 07: Forecast comparison — NYPD ────────────────────────────────────
print("  fig07: forecast comparison NYPD")
r_nyp     = res_all['D2 NYPD Manhattan']
test_idx2 = r_nyp['test'].index

fig, ax = plt.subplots(figsize=(13, 5))
ctx2 = r_nyp['train'].iloc[-40:]
ax.plot(ctx2.index, ctx2.values, color='#1E8449', linewidth=1.6, label='Training data (last 40 weeks)')
ax.plot(test_idx2, r_nyp['test'].values, color='black', linewidth=2.2, label='Actual (test)', zorder=5)
ax.plot(test_idx2, r_nyp['nv_pred'],    color=MODEL_COLORS['Naive'], linewidth=1.5,
        linestyle='--', label=f"Naive  (MAE={r_nyp['nv_mae']:.1f})")
ax.plot(test_idx2, r_nyp['ar_pred'],    color=MODEL_COLORS['AR'],    linewidth=1.8,
        label=f"AR({r_nyp['ar_lag']})    (MAE={r_nyp['ar_mae']:.1f})")
ax.plot(test_idx2, r_nyp['arima_pred'], color=MODEL_COLORS['ARIMA'], linewidth=1.8,
        linestyle='-.', label=f"ARIMA{r_nyp['arima_ord']} (MAE={r_nyp['arima_mae']:.1f})")
ax.axvline(test_idx2[0], color='red', linestyle=':', linewidth=1.4, alpha=0.7)
ax.set_title('D2 NYPD Manhattan — Test-Period Forecast Comparison',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date', fontsize=10); ax.set_ylabel('Weekly Crime Count', fontsize=10)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig07_forecast_nypd.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 08: Forecast comparison — SFPD ────────────────────────────────────
print("  fig08: forecast comparison SFPD")
r_sfp     = res_all['D3 SFPD Mission Dist']
test_idx3 = r_sfp['test'].index

fig, ax = plt.subplots(figsize=(13, 5))
ctx3 = r_sfp['train'].iloc[-40:]
ax.plot(ctx3.index, ctx3.values, color='#CB4335', linewidth=1.6, label='Training data (last 40 weeks)')
ax.plot(test_idx3, r_sfp['test'].values, color='black', linewidth=2.2, label='Actual (test)', zorder=5)
ax.plot(test_idx3, r_sfp['nv_pred'],    color=MODEL_COLORS['Naive'], linewidth=1.5,
        linestyle='--', label=f"Naive  (MAE={r_sfp['nv_mae']:.1f})")
ax.plot(test_idx3, r_sfp['ar_pred'],    color=MODEL_COLORS['AR'],    linewidth=1.8,
        label=f"AR({r_sfp['ar_lag']})    (MAE={r_sfp['ar_mae']:.1f})")
ax.plot(test_idx3, r_sfp['arima_pred'], color=MODEL_COLORS['ARIMA'], linewidth=1.8,
        linestyle='-.', label=f"ARIMA{r_sfp['arima_ord']} (MAE={r_sfp['arima_mae']:.1f})")
ax.axvline(test_idx3[0], color='red', linestyle=':', linewidth=1.4, alpha=0.7)
ax.set_title('D3 SFPD Mission District — Test-Period Forecast Comparison',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date', fontsize=10); ax.set_ylabel('Weekly Crime Count', fontsize=10)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig08_forecast_sfpd.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 09: Cross-dataset MAE/RMSE comparison ─────────────────────────────
print("  fig09: cross-dataset model comparison")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Model Performance Across All Three Datasets',
             fontsize=13, fontweight='bold')

model_names = ['Naive', f'AR', 'ARIMA']
x = np.arange(3)
w = 0.22

for ax_idx, (ax, metric, title) in enumerate(zip(
        axes, ['mae', 'rmse'], ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)'])):
    for di, (dname, col) in enumerate(COLORS.items()):
        r = res_all[dname]
        vals = [r[f'nv_{metric}'], r[f'ar_{metric}'], r[f'arima_{metric}']]
        bars = ax.bar(x + (di - 1) * w, vals, width=w,
                      color=col, alpha=0.82, edgecolor='white',
                      label=dname.split(' ')[0] + ' ' + dname.split(' ')[1])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel(title, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig09_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 10: Rolling-origin fold MAE ───────────────────────────────────────
print("  fig10: rolling-origin CV fold MAE")
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
fig.suptitle('Rolling-Origin Backtesting — Fold MAE Distribution\n(6 folds × 8-week horizon)',
             fontsize=13, fontweight='bold')

for ax, dname in zip(axes, ds_keys):
    r = res_all[dname]
    data = [r['ro_ar_maes'], r['ro_arima_maes']]
    colors = [MODEL_COLORS['AR'], MODEL_COLORS['ARIMA']]
    labels = [f'AR({r["ar_lag"]})', f'ARIMA{r["arima_ord"]}']
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2.0),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(dname, fontsize=10, fontweight='bold', color=COLORS[dname])
    ax.set_ylabel('Fold MAE', fontsize=9) if ax == axes[0] else None
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # Annotate mean
    for j, d in enumerate(data):
        if len(d):
            ax.text(j + 1, d.mean(), f'μ={d.mean():.1f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color=colors[j])

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig10_rolling_cv.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 11: Seasonal decomposition — Chicago ───────────────────────────────
print("  fig11: seasonal decomposition")
chi_s2 = res_all['D1 Chicago District 1']['series']
decomp = seasonal_decompose(chi_s2, model='additive', period=52, extrapolate_trend='freq')

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
fig.suptitle('D1 Chicago District 1 — Additive Seasonal Decomposition (period=52 weeks)',
             fontsize=13, fontweight='bold')

components = [chi_s2, decomp.trend, decomp.seasonal, decomp.resid]
comp_labels= ['Observed', 'Trend', 'Seasonal', 'Residual']
comp_colors= ['#2471A3', '#922B21', '#1E8449', '#7D3C98']

for ax, data, label, col in zip(axes, components, comp_labels, comp_colors):
    ax.plot(data.index, data.values, color=col, linewidth=1.4)
    if label == 'Residual':
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel(label, fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

axes[-1].set_xlabel('Date', fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig11_seasonal_decomp.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 12: Ljung-Box p-values and AIC/BIC comparison ─────────────────────
print("  fig12: Ljung-Box + information criteria")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Model Diagnostics: Information Criteria and Residual White-Noise Test',
             fontsize=13, fontweight='bold')

dshort  = ['Chicago D1', 'NYPD D2', 'SFPD D3']
lb_vals = [res_all[k]['lb_p']        for k in ds_keys]
ar_aics = [res_all[k]['ar_aic']      for k in ds_keys]
ar_bics = [res_all[k]['ar_bic']      for k in ds_keys]
ar_aics_n = [v / abs(v) * abs(v) for v in ar_aics]  # keep sign

# Ljung-Box p-values
x = np.arange(3)
bar_c = ['#27AE60' if p > 0.05 else '#E74C3C' for p in lb_vals]
bars = axes[0].bar(x, lb_vals, color=bar_c, alpha=0.85, edgecolor='white', width=0.5)
axes[0].axhline(0.05, color='red', linestyle='--', linewidth=1.5, label='α=0.05')
for bar, val in zip(bars, lb_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].set_xticks(x); axes[0].set_xticklabels(dshort, fontsize=10)
axes[0].set_ylabel('Ljung-Box p-value (lag 10)', fontsize=10)
axes[0].set_title('Residual White-Noise Test\n(p > 0.05 = residuals uncorrelated ✓)', fontsize=10)
axes[0].legend(fontsize=9)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

# AIC comparison (AR vs ARIMA)
arima_aics = [res_all[k]['arima_aic'] for k in ds_keys]
arima_bics = [res_all[k]['arima_bic'] for k in ds_keys]
w2 = 0.20
axes[1].bar(x - 1.5*w2, ar_aics,    width=w2, color=MODEL_COLORS['AR'],    alpha=0.82, label='AR AIC',    edgecolor='white')
axes[1].bar(x - 0.5*w2, ar_bics,    width=w2, color=MODEL_COLORS['AR'],    alpha=0.45, label='AR BIC',    edgecolor='white', hatch='//')
axes[1].bar(x + 0.5*w2, arima_aics, width=w2, color=MODEL_COLORS['ARIMA'], alpha=0.82, label='ARIMA AIC', edgecolor='white')
axes[1].bar(x + 1.5*w2, arima_bics, width=w2, color=MODEL_COLORS['ARIMA'], alpha=0.45, label='ARIMA BIC', edgecolor='white', hatch='//')
axes[1].set_xticks(x); axes[1].set_xticklabels(dshort, fontsize=10)
axes[1].set_ylabel('Information Criterion Value', fontsize=10)
axes[1].set_title('AR vs ARIMA — AIC and BIC\n(lower = better fit)', fontsize=10)
axes[1].legend(fontsize=8, ncol=2)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig12_diagnostics.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  All 12 figures saved to: {FIG_DIR}")

# =============================================================================
# 5.  SAVE OUTPUTS (CSVs + manifest)
# =============================================================================
print("\n[4] Saving outputs to CSV...")

# Full dataset CSV for each dataset
for dname, r in res_all.items():
    key = dname.replace(' ', '_').replace('/', '-')
    df  = pd.DataFrame({'date': r['series'].index,
                        'crime_count': r['series'].values,
                        'split': ['train'] * len(r['train']) + ['test'] * len(r['test'])})
    df.to_csv(os.path.join(OUT_DIR, f'{key}_weekly.csv'), index=False)

# Predictions CSV
pred_rows = []
for dname, r in res_all.items():
    for wk, (dt, act, nv, ar, ar_) in enumerate(zip(
            r['test'].index, r['test'].values,
            r['nv_pred'], r['ar_pred'], r['arima_pred'])):
        pred_rows.append({'dataset': dname, 'week': wk+1, 'date': dt,
                          'actual': int(act), 'naive': float(nv),
                          'ar': float(ar), 'arima': float(ar_)})
pd.DataFrame(pred_rows).to_csv(os.path.join(OUT_DIR, 'test_predictions.csv'), index=False)

# Rolling-origin fold CSV
ro_rows = []
for dname, r in res_all.items():
    for fold_i, (ar_m, arima_m) in enumerate(zip(r['ro_ar_maes'], r['ro_arima_maes'])):
        ro_rows.append({'dataset': dname, 'fold': fold_i + 1,
                        'ar_mae': ar_m, 'arima_mae': arima_m})
pd.DataFrame(ro_rows).to_csv(os.path.join(OUT_DIR, 'rolling_origin_folds.csv'), index=False)

# Summary manifest
manifest = {}
for dname, r in res_all.items():
    manifest[dname] = {
        'n_total': len(r['series']),
        'n_train': len(r['train']),
        'n_test':  len(r['test']),
        'mean_count': float(r['series'].mean()),
        'std_count':  float(r['series'].std()),
        'adf_stat_train': r['adf_stat_train'],
        'adf_p_train':    r['adf_p_train'],
        'adf_stat_full':  r['adf_stat_full'],
        'adf_p_full':     r['adf_p_full'],
        'ar_lag':         r['ar_lag'],
        'ar_aic':         r['ar_aic'],
        'ar_bic':         r['ar_bic'],
        'arima_order':    list(r['arima_ord']),
        'arima_aic':      r['arima_aic'],
        'arima_bic':      r['arima_bic'],
        'lb_p':           r['lb_p'],
        'naive_mae':      r['nv_mae'],  'naive_rmse':  r['nv_rmse'],
        'ar_mae':         r['ar_mae'],  'ar_rmse':     r['ar_rmse'],
        'arima_mae':      r['arima_mae'],'arima_rmse': r['arima_rmse'],
        'ro_ar_mae_mean':    float(r['ro_ar_maes'].mean()) if len(r['ro_ar_maes']) else None,
        'ro_arima_mae_mean': float(r['ro_arima_maes'].mean()) if len(r['ro_arima_maes']) else None,
    }

with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"  Saved: test_predictions.csv, rolling_origin_folds.csv, manifest.json")

# =============================================================================
# 6.  PRINT SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
header = f"{'Dataset':<26} {'Model':<12} {'MAE':>8} {'RMSE':>8} {'AIC':>10} {'BIC':>10}"
print(header)
print("-" * 70)
for dname, r in res_all.items():
    ds_s = dname[:26]
    print(f"{ds_s:<26} {'Naive':<12} {r['nv_mae']:>8.2f} {r['nv_rmse']:>8.2f} {'—':>10} {'—':>10}")
    print(f"{'':<26} {f'AR({r[chr(97)+chr(114)+chr(95)+chr(108)+chr(97)+chr(103)]})':<12} {r['ar_mae']:>8.2f} {r['ar_rmse']:>8.2f} {r['ar_aic']:>10.1f} {r['ar_bic']:>10.1f}")
    print(f"{'':<26} {f'ARIMA{r[chr(97)+chr(114)+chr(105)+chr(109)+chr(97)+chr(95)+chr(111)+chr(114)+chr(100)]}':<12} {r['arima_mae']:>8.2f} {r['arima_rmse']:>8.2f} {r['arima_aic']:>10.1f} {r['arima_bic']:>10.1f}")
    print("-" * 70)

print("\nADF + Ljung-Box diagnostics:")
print(f"{'Dataset':<26} {'ADF stat':>10} {'ADF p':>8} {'LB p(lag10)':>12} {'ARIMA order':>14}")
print("-" * 72)
for dname, r in res_all.items():
    stationary = "✓ stationary" if r['adf_p_train'] <= 0.05 else "✗ non-stationary"
    lb_ok = "✓ white noise" if r['lb_p'] > 0.05 else "✗ autocorrelated"
    print(f"{dname[:26]:<26} {r['adf_stat_train']:>10.4f} {r['adf_p_train']:>8.4f} {r['lb_p']:>12.4f}  {str(r['arima_ord']):>14}")
print()
print(f"Manifest saved: {os.path.join(OUT_DIR, 'manifest.json')}")

figs = sorted(f for f in os.listdir(FIG_DIR) if f.endswith('.png'))
print(f"\n{len(figs)} figures in {FIG_DIR}:")
for f in figs:
    print(f"  {f}")

print("\n=== Lab 06 Pipeline Complete ===")
