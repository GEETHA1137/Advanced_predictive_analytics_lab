#!/usr/bin/env python3
"""Lab 08 – Agricultural Predictive Analytics: Crop Yield Prediction
MDI3003 Experiment 08 reference implementation (Appendix A style).
Stages: validate (chronological split + rolling-origin CV + model selection)
        test     (one-shot test evaluation on held-out years)
"""
import argparse, hashlib, json, pathlib, sys, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

REG    = ['state', 'district', 'season', 'year']
TARGET = 'yield_t_ha'
STUDENT = '23MID0021'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(obj, path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2, default=str)


def _save_fig(fig, dest):
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(dest)


def load_data(cfg):
    path = cfg['data']['path']
    df = pd.read_csv(path)
    required = ['row_id', 'crop', 'state', 'district', 'season', 'year', 'yield_t_ha']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    crop = cfg.get('crop', 'Rice')
    df = df[df['crop'] == crop].copy()
    if df.empty:
        raise ValueError(f'No rows with crop={crop!r}')
    df = df[df['yield_t_ha'].notna() & (df['yield_t_ha'] >= 0)].copy()
    if df.duplicated(REG).any():
        dup = df[df.duplicated(REG, keep=False)].head(3)[REG].to_dict('records')
        raise ValueError(f'Repeated district-season-year keys: {dup}')
    return df


def candidates(advanced=None):
    cat_cols = ['state', 'district', 'season']
    num_cols = ['year']
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    cat_pipe = Pipeline([('ohe', ohe)])
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())])
    preproc  = ColumnTransformer([
        ('cat', cat_pipe, cat_cols),
        ('num', num_pipe, num_cols),
    ])
    def _make(mdl):
        return Pipeline([('pre', clone(preproc)), ('mdl', mdl)])
    return {
        'median':      _make(DummyRegressor(strategy='median')),
        'ridge_trend': _make(Ridge(alpha=1.0, solver='lsqr')),
        'tree':        _make(DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=42)),
        'forest':      _make(RandomForestRegressor(n_estimators=60, max_depth=12,
                                                   min_samples_leaf=5, n_jobs=2, random_state=42)),
    }


def reg_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4)}


def validate(cfg, out, df, features, target):
    """Chronological split + rolling-origin CV + model selection."""
    out = pathlib.Path(out)
    for sub in ('models', 'figures', 'artifacts'):
        (out / sub).mkdir(parents=True, exist_ok=True)

    years = sorted(df['year'].unique())
    n = len(years)
    if n < 7:
        raise ValueError(f'Need ≥7 unique harvest years, got {n}')

    test_years  = list(years[-2:])
    val_years   = list(years[-4:-2])
    train_years = list(years[:-4])

    df_train = df[df['year'].isin(train_years)].copy()
    df_val   = df[df['year'].isin(val_years)].copy()

    manifest = {
        'train_years': train_years, 'val_years': val_years, 'test_years': test_years,
        'train_rows': int(len(df_train)), 'val_rows': int(len(df_val)),
        'test_rows': int(len(df[df['year'].isin(test_years)])),
    }
    dump(manifest, out / 'artifacts' / 'split_manifest.json')

    # Rolling-origin CV within training years (3 origins)
    n_origins = min(3, max(1, len(train_years) - 2))
    cv_rows = []
    for origin in range(n_origins):
        split_idx   = len(train_years) - n_origins + origin
        cv_tr_years = train_years[:split_idx]
        cv_va_years = train_years[split_idx:split_idx + 1]
        if not cv_tr_years or not cv_va_years:
            continue
        cv_tr = df[df['year'].isin(cv_tr_years)]
        cv_va = df[df['year'].isin(cv_va_years)]
        pipes = candidates(cfg.get('advanced', {}))
        for name, pipe in pipes.items():
            pipe.fit(cv_tr[features], cv_tr[target])
            pred = pipe.predict(cv_va[features])
            m = reg_metrics(cv_va[target].values, pred)
            cv_rows.append({'origin': origin, 'model': name, **m})
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out / 'artifacts' / 'rolling_origins.csv', index=False)

    # Full train -> validate
    pipes = candidates(cfg.get('advanced', {}))
    val_metrics = {}
    for name, pipe in pipes.items():
        pipe.fit(df_train[features], df_train[target])
        joblib.dump(pipe, out / 'models' / f'{name}.joblib')
        pred_v = pipe.predict(df_val[features])
        val_metrics[name] = reg_metrics(df_val[target].values, pred_v)

    # Validation results CSV
    val_df = df_val[['row_id', 'state', 'district', 'season', 'year', target]].copy().reset_index(drop=True)
    for name, pipe in pipes.items():
        val_df[f'pred_{name}'] = pipe.predict(df_val[features])
    val_df = val_df.rename(columns={target: 'actual_yield'})
    val_df.to_csv(out / f'{STUDENT}_Lab08_Validation_Results.csv', index=False)

    # Select best by MAE
    best_name = min(val_metrics, key=lambda k: val_metrics[k]['mae'])
    best_pipe  = pipes[best_name]

    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    selection = {
        'selected_model': best_name,
        'val_metrics': val_metrics,
        'config_sha256': cfg_sha,
        'features': features,
        'target': target,
        'train_years': train_years,
        'val_years': val_years,
        'test_years': test_years,
    }
    dump(selection, out / 'artifacts' / 'selection.json')
    bundle = {'pipeline': best_pipe, 'selection': selection, 'features': features, 'target': target}
    joblib.dump(bundle, out / 'models' / 'selected_bundle.joblib')

    # ── Figures ─────────────────────────────────────────────────────────────
    # 1. Training yield distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df_train[target].values, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Yield (t/ha)', fontsize=11); ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Training Yield Distribution  (n={len(df_train):,})', fontsize=12)
    plt.tight_layout()
    _save_fig(fig, out / 'figures' / 'target.png')

    # 2. Year coverage
    fig, ax = plt.subplots(figsize=(8, 4))
    yr_cnt = df.groupby('year').size()
    colors = ['#1f77b4' if y in train_years else '#ff7f0e' if y in val_years else '#2ca02c'
              for y in yr_cnt.index]
    bars = ax.bar(yr_cnt.index, yr_cnt.values, color=colors, edgecolor='white', alpha=0.85)
    ax.set_xlabel('Harvest Year', fontsize=11); ax.set_ylabel('Records', fontsize=11)
    ax.set_title('Records per Harvest Year  (blue=train, orange=val, green=test)', fontsize=11)
    plt.tight_layout()
    _save_fig(fig, out / 'figures' / 'feature.png')

    # 3. Validation model comparison
    model_names = list(val_metrics.keys())
    x = np.arange(len(model_names))
    metrics_list = [('MAE', 'mae', 'steelblue'), ('RMSE', 'rmse', 'coral'), ('R²', 'r2', 'seagreen')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (label, key, color) in zip(axes, metrics_list):
        vals = [val_metrics[n][key] for n in model_names]
        bars = ax.bar(x, vals, color=color, edgecolor='white', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=15, fontsize=9)
        ax.set_ylabel(label, fontsize=11); ax.set_title(f'Validation {label}', fontsize=11)
        best_idx = (np.argmin(vals) if key != 'r2' else np.argmax(vals))
        bars[best_idx].set_edgecolor('black'); bars[best_idx].set_linewidth(2.0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    _save_fig(fig, out / 'figures' / 'comparison.png')

    print(f'\n[validate] Best model: {best_name}')
    for name, m in val_metrics.items():
        star = ' <<< SELECTED' if name == best_name else ''
        print(f'  {name:15s}  MAE={m["mae"]:.4f}  RMSE={m["rmse"]:.4f}  R²={m["r2"]:.4f}{star}')
    return selection


def predict_validated(bundle, records):
    """Run inference with a saved bundle on a list of dicts."""
    df = pd.DataFrame(records)
    return bundle['pipeline'].predict(df[bundle['features']])


def test_once(cfg, out, df, features, target):
    """One-shot test evaluation — call only after validate."""
    out = pathlib.Path(out)
    bundle_path = out / 'models' / 'selected_bundle.joblib'
    if not bundle_path.exists():
        raise FileNotFoundError('Run --stage validate first.')

    bundle    = joblib.load(bundle_path)
    selection = bundle['selection']

    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    assert selection['config_sha256'] == cfg_sha, \
        'Config changed since validate — re-run validate first.'

    test_years     = selection['test_years']
    non_test_years = [y for y in sorted(df['year'].unique()) if y not in test_years]

    df_train_val = df[df['year'].isin(non_test_years)].copy()
    df_test      = df[df['year'].isin(test_years)].copy()

    best_pipe = bundle['pipeline']
    best_pipe.fit(df_train_val[features], df_train_val[target])

    pred_test = best_pipe.predict(df_test[features])
    m_test    = reg_metrics(df_test[target].values, pred_test)

    print(f'\n[test] Model : {selection["selected_model"]}')
    print(f'[test] Metrics: MAE={m_test["mae"]:.4f}  RMSE={m_test["rmse"]:.4f}  R²={m_test["r2"]:.4f}')

    # Test results CSV
    test_out = df_test[['row_id', 'state', 'district', 'season', 'year', target]].copy().reset_index(drop=True)
    test_out['predicted_yield'] = pred_test
    test_out['residual']        = test_out[target] - test_out['predicted_yield']
    test_out = test_out.rename(columns={target: 'actual_yield'})
    test_out.to_csv(out / f'{STUDENT}_Lab08_Test_Results.csv', index=False)

    # Error analysis — 5 worst cases
    test_out['abs_error'] = test_out['residual'].abs()
    error_df = test_out.sort_values('abs_error', ascending=False).head(5)
    error_df.to_csv(out / f'{STUDENT}_Lab08_Error_Analysis.csv', index=False)

    # Year robustness
    yr_rows = []
    for yr in test_years:
        sub = test_out[test_out['year'] == yr]
        if len(sub):
            mr = reg_metrics(sub['actual_yield'].values, sub['predicted_yield'].values)
            yr_rows.append({'year': yr, **mr})
    pd.DataFrame(yr_rows).to_csv(out / 'artifacts' / 'year_robustness.csv', index=False)

    # Save final selected model with student prefix
    joblib.dump(best_pipe, out / 'models' / f'{STUDENT}_Lab08_selected_bundle.joblib')

    acceptance = {
        'test_mae': m_test['mae'], 'test_rmse': m_test['rmse'], 'test_r2': m_test['r2'],
        'selected_model': selection['selected_model'],
        'test_years': test_years,
        'passed': m_test['mae'] < 1.5,
    }
    dump(acceptance, out / 'artifacts' / 'acceptance.json')

    # ── Figures ─────────────────────────────────────────────────────────────
    actual = df_test[target].values

    # 4. Actual vs Predicted scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, pred_test, alpha=0.45, s=22, color='steelblue', edgecolors='none')
    lo, hi = min(actual.min(), pred_test.min()), max(actual.max(), pred_test.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual Yield (t/ha)', fontsize=11)
    ax.set_ylabel('Predicted Yield (t/ha)', fontsize=11)
    ax.set_title(f'Test: Actual vs Predicted\n(MAE={m_test["mae"]:.3f} t/ha,  R²={m_test["r2"]:.3f})', fontsize=11)
    ax.legend(fontsize=10); plt.tight_layout()
    _save_fig(fig, out / 'figures' / 'actual_predicted.png')

    # 5. Residual panel
    residuals = actual - pred_test
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(pred_test, residuals, alpha=0.4, s=20, color='coral', edgecolors='none')
    axes[0].axhline(0, color='black', lw=1.2, ls='--')
    axes[0].set_xlabel('Predicted Yield (t/ha)', fontsize=11)
    axes[0].set_ylabel('Residual (t/ha)', fontsize=11)
    axes[0].set_title('Residual vs Predicted', fontsize=11)
    axes[1].hist(residuals, bins=25, color='seagreen', edgecolor='white', alpha=0.85)
    axes[1].axvline(0, color='black', lw=1.2, ls='--')
    axes[1].set_xlabel('Residual (t/ha)', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=11)
    plt.tight_layout()
    _save_fig(fig, out / 'figures' / 'residuals.png')

    print(f'[test] Outputs -> {out}')
    return acceptance


def run(cfg, stage):
    out = pathlib.Path(cfg['output']['dir'])
    (out / 'artifacts').mkdir(parents=True, exist_ok=True)

    import sklearn, matplotlib as mpl
    dump({
        'python':   sys.version.split()[0],
        'numpy':    np.__version__,
        'pandas':   pd.__version__,
        'sklearn':  sklearn.__version__,
        'matplotlib': mpl.__version__,
    }, out / 'artifacts' / 'versions.json')
    dump(cfg, out / 'artifacts' / 'config.json')

    df       = load_data(cfg)
    features = REG

    if stage == 'validate':
        return validate(cfg, out, df, features, TARGET)
    elif stage == 'test':
        return test_once(cfg, out, df, features, TARGET)
    else:
        raise ValueError(f'Unknown stage: {stage!r}')


def main():
    ap = argparse.ArgumentParser(description='Lab 08 – Crop Yield Prediction')
    ap.add_argument('--config', default='config.json', help='Config file path')
    ap.add_argument('--stage', choices=['validate', 'test'], required=True, help='Pipeline stage')
    args = ap.parse_args()
    with open(args.config, encoding='utf-8') as fh:
        cfg = json.load(fh)
    result = run(cfg, args.stage)
    print('\n[done]', json.dumps(result, indent=2, default=str)[:600])


if __name__ == '__main__':
    main()
