#!/usr/bin/env python3
import os, sys, json, time, warnings
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")
SEED = 42

SEYP_DIR = 'results/SE-YP'
OUT_DIR = 'results/fig2'
FEATURES = 'mundlak'
PDP_ROWS = 20000


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


RATE_OF = {'N_rel': ('N', 'N_med'), 'P2O5_rel': ('P2O5', 'P2O5_med'), 'K20_rel': ('K20', 'K20_med')}
DISPLAY_NAME = {
    'N': 'N rate', 'P2O5': 'P2O5 rate', 'K20': 'K2O rate',
    'N_med': 'Medium N rate', 'P2O5_med': 'Medium P2O5 rate', 'K20_med': 'Medium K2O rate',
    'N_rel': 'N rate (relative)', 'P2O5_rel': 'P2O5 rate (relative)', 'K20_rel': 'K2O rate (relative)',
    'Altitude': 'Altitude', 'irrigation': 'Irrigation', 'AP': 'AP', 'precipitation': 'Precipitation',
    'Frost_free_day': 'FFP', 'temperature': 'Tem', 'pH': 'pH', 'AN': 'AN', 'TN': 'TN', 'SOM': 'SOM',
    'AK': 'AK', 'Nsurplus': 'Legacy N', 'crop_type_sw': 'crop_type_sw', 'crop_type_ww': 'crop_type_ww'}
UNITS = {'N': 'kg ha$^{-1}$', 'P2O5': 'kg ha$^{-1}$', 'K20': 'kg ha$^{-1}$',
         'N_med': 'kg ha$^{-1}$', 'P2O5_med': 'kg ha$^{-1}$', 'K20_med': 'kg ha$^{-1}$',
         'N_rel': 'x medium', 'P2O5_rel': 'x medium', 'K20_rel': 'x medium',
         'Altitude': 'm', 'irrigation': 'mm', 'AP': 'mg kg$^{-1}$', 'precipitation': 'mm',
         'Frost_free_day': 'day', 'temperature': '\u2103', 'pH': '', 'AN': 'mg kg$^{-1}$',
         'TN': 'g kg$^{-1}$', 'SOM': 'g kg$^{-1}$', 'AK': 'mg kg$^{-1}$', 'Nsurplus': 'kg ha$^{-1}$'}
GROUP_CLIMATE = {'Precipitation', 'Tem', 'FFP'}
GROUP_MANAGE = {'N rate', 'P2O5 rate', 'K2O rate', 'Medium N rate', 'Medium P2O5 rate', 'Medium K2O rate',
                'N rate (relative)', 'P2O5 rate (relative)', 'K2O rate (relative)',
                'crop_type_sw', 'crop_type_ww', 'Irrigation', 'Legacy N'}
GROUP_SOIL = {'SOM', 'TN', 'AN', 'AP', 'AK', 'pH', 'Altitude'}
COLOR = {'climate': '#DAE3F3', 'manage': '#E2F0D9', 'soil': '#FBE5D6', 'other': '#7f7f7f'}
PDP_PANELS = ['N', 'Altitude', 'irrigation', 'P2O5', 'AP', 'precipitation', 'K20', 'Frost_free_day',
              'temperature', 'pH', 'AN', 'TN', 'SOM', 'AK', 'Nsurplus']
PDP_EXTRA = ['N_med', 'P2O5_med', 'K20_med']


def color_of(display):
    if display in GROUP_CLIMATE: return COLOR['climate']
    if display in GROUP_MANAGE: return COLOR['manage']
    if display in GROUP_SOIL: return COLOR['soil']
    return COLOR['other']


def x_label(feature):
    disp = DISPLAY_NAME.get(feature, feature); u = UNITS.get(feature, '')
    return f'{disp} ({u})' if u else disp


def shap_of_stack(stack, X, X_background, explain='ensemble'):
    """SHAP values (log1p scale) of the stacking ensemble for the rows of X, summed over the base learners with the weights of the meta-learner."""
    import shap
    names = [n for n, _ in stack.estimators]
    fitted = dict(zip(names, stack.estimators_))
    if explain == 'xgb':
        return shap.TreeExplainer(fitted['xgb']).shap_values(X)
    w = stack.final_estimator_.coef_
    total = np.zeros((len(X), X.shape[1]))
    for j, n in enumerate(names):
        est = fitted[n]
        if n == 'lr':
            sv = shap.LinearExplainer(est, shap.maskers.Independent(X_background, max_samples=len(X_background))).shap_values(X)
        else:
            sv = shap.TreeExplainer(est).shap_values(X)
        total += w[j] * sv
    return total


def importance_tables(shap_values, feats):
    """Mean absolute SHAP importance in three groupings of the covariates: all 23, relative rates merged into the applied rates, and medium rates merged as well."""
    sv = pd.DataFrame(shap_values, columns=feats)
    frames = {'raw23': sv}
    g = sv.copy()
    for rel, (rate, med) in RATE_OF.items():
        if rel in g:
            g[rate] = g[rate] + g[rel]; g = g.drop(columns=[rel])
    frames['merged_rel'] = g
    h = g.copy()
    for rel, (rate, med) in RATE_OF.items():
        if med in h:
            h[rate] = h[rate] + h[med]; h = h.drop(columns=[med])
    frames['style17'] = h
    tables = {}
    for key, frame in frames.items():
        mean_abs = frame.abs().mean(axis=0)
        rel = 100.0 * mean_abs / mean_abs.sum()
        t = pd.DataFrame({'feature': mean_abs.index, 'display': [DISPLAY_NAME.get(f, f) for f in mean_abs.index],
                          'mean_abs_shap': mean_abs.values, 'rel_importance_pct': rel.values})
        t = t.sort_values('rel_importance_pct', ascending=False).reset_index(drop=True)
        t.insert(0, 'rank', np.arange(1, len(t) + 1))
        tables[key] = t
    return tables


def plot_importance_bar(table, path, xlabel):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(8, max(6, 0.35 * len(table))))
    ax.barh(table['display'][::-1], table['rel_importance_pct'][::-1],
            color=[color_of(d) for d in table['display'][::-1]], edgecolor='none')
    ax.set_xlabel(xlabel); ax.set_ylabel('')
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close(fig)


def plot_beeswarm(shap_values, X, path, xlabel, scale=1.0):
    import shap
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'
    Xd = X.copy(); Xd.columns = [DISPLAY_NAME.get(c, c) for c in X.columns]
    shap.summary_plot(shap_values * scale, Xd, show=False)
    plt.gca().set_xlabel(xlabel)
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()


def pdp_curve(stack, X, feature, grid_res=25):
    """Partial dependence of predicted yield (t ha-1) on one covariate, with the linked relative rate recomputed when a fertilizer or medium rate is varied."""
    lo, hi = X[feature].min(), X[feature].max()
    grid = np.linspace(lo, hi, grid_res)
    Xg = X.copy(); ys = []
    for v in grid:
        Xg[feature] = v
        for rel, (rate, med) in RATE_OF.items():
            if feature in (rate, med):
                Xg[rel] = (Xg[rate] / Xg[med].replace(0, np.nan)).fillna(0)
        ys.append(np.expm1(stack.predict(Xg)).mean() / 1000.0)
    return grid, np.array(ys)


def plot_pdp_panel(x, y, feature, X, path, highlight):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, linewidth=2)
    if highlight:
        p2_5, p97_5 = X[feature].quantile(0.025), X[feature].quantile(0.975)
        ax.axvspan(p2_5, p97_5, alpha=0.1, color='red')
        y_min, y_max = ax.get_ylim(); y_loc = y_min + 0.05 * (y_max - y_min)
    ax.set_xlabel(x_label(feature)); ax.set_ylabel('Yield (Mg ha$^{-1}$)')
    fig.tight_layout(); plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_pdp_grid(curves, order, X, path, highlight, rows=3, cols=5):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'
    feats = [f for f in order if f in curves][: rows * cols]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.6), constrained_layout=True)
    for i in range(rows * cols):
        r, c = divmod(i, cols); ax = axes[r, c]
        if i >= len(feats):
            ax.axis('off'); continue
        f = feats[i]; x, y = curves[f]
        ax.plot(x, y, linewidth=2)
        if highlight:
            p2_5, p97_5 = X[f].quantile(0.025), X[f].quantile(0.975)
            ax.axvspan(p2_5, p97_5, alpha=0.1, color='red')
            y_min, y_max = ax.get_ylim(); y_loc = y_min + 0.07 * (y_max - y_min)
        ax.set_xlabel(x_label(f), fontsize=9); ax.set_ylabel('Yield (Mg ha$^{-1}$)', fontsize=9)
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)


def main():
    """Compute out-of-fold SHAP values with the five site-fold models and partial dependence with the final model, and write the tables and panels of Figure 2."""
    args = SimpleNamespace(seyp=SEYP_DIR, out=OUT_DIR, features=FEATURES, explain='ensemble', pdp_rows=PDP_ROWS)
    shap_dir = os.path.join(args.out, 'fig2a_shap'); pdp_dir = os.path.join(args.out, 'fig2b_pdp')
    os.makedirs(shap_dir, exist_ok=True); os.makedirs(pdp_dir, exist_ok=True)

    feats = json.load(open(os.path.join(args.seyp, 'models', f'feature_list_{args.features}.json')))
    mm = pd.read_csv(os.path.join(args.seyp, 'oof', f'model_matrix_{args.features}.csv'))
    missing = [f for f in feats if f not in mm.columns]
    assert not missing, f'model matrix lacks {missing}'
    X = mm[feats]; y = mm['yield'].values; sites = mm['site'].values; fold_id = mm['fold_site'].values
    n_splits = int(fold_id.max()) + 1
    log(f'{len(mm):,} plots, {len(np.unique(sites)):,} sites, {len(feats)} features, {n_splits} site folds')

    r2_oof = None
    if True:
        shap_all = np.zeros(X.shape); pred_all = np.full(len(mm), np.nan); fold_tables = []
        for k in range(n_splits):
            tk = time.time()
            m = joblib.load(os.path.join(args.seyp, 'models', f'stack_site_{args.features}_fold{k}.pkl'))
            te = fold_id == k; tr = ~te
            pred_all[te] = np.expm1(m.predict(X[te]))
            bg = X[tr].sample(min(2000, int(tr.sum())), random_state=SEED)
            sv = shap_of_stack(m, X[te], bg, args.explain)
            shap_all[te] = sv
            t = importance_tables(sv, feats)['merged_rel']; t['fold'] = k; fold_tables.append(t)
            r2k = 1 - ((y[te] - pred_all[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
            log(f'fold {k}: held-out plots {int(te.sum()):,} R2 {r2k:.3f} ({time.time() - tk:.0f}s)')
        r2_oof = float(1 - ((y - pred_all) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        log(f'site hold-out R2 (yield scale) {r2_oof:.3f}')
        pd.DataFrame(shap_all, columns=feats).assign(site=sites, fold=fold_id).to_csv(os.path.join(shap_dir, 'shap_values_oof.csv'), index=False)
        pd.DataFrame({'site': sites, 'fold': fold_id, 'y': y, 'yhat_oof': pred_all}).to_csv(os.path.join(shap_dir, 'oof_predictions.csv'), index=False)
        tables = importance_tables(shap_all, feats)
        for key, t in tables.items():
            t.to_csv(os.path.join(shap_dir, f'shap_importance_{key}.csv'), index=False, encoding='utf-8-sig')
        fold_df = pd.concat(fold_tables); fold_df.to_csv(os.path.join(shap_dir, 'shap_importance_by_fold_merged_rel.csv'), index=False, encoding='utf-8-sig')
        piv = fold_df.pivot(index='feature', columns='fold', values='rel_importance_pct')
        ranks = piv.rank(ascending=False); ranks.to_csv(os.path.join(shap_dir, 'shap_rank_by_fold.csv'))
        rho = ranks.corr(method='spearman'); rho.to_csv(os.path.join(shap_dir, 'shap_rank_spearman_between_folds.csv'))
        stab = pd.DataFrame({'feature': piv.index, 'display': [DISPLAY_NAME.get(f, f) for f in piv.index],
                             'rel_importance_mean': piv.mean(1).values, 'rel_importance_sd': piv.std(1).values,
                             'rank_min': ranks.min(1).values, 'rank_max': ranks.max(1).values}).sort_values('rel_importance_mean', ascending=False)
        stab.to_csv(os.path.join(shap_dir, 'shap_importance_stability.csv'), index=False, encoding='utf-8-sig')
        iu = np.triu_indices(n_splits, 1)
        log('fold-to-fold Spearman of importance ranks: min %.3f median %.3f' % (rho.values[iu].min(), np.median(rho.values[iu])))
        xlabel = f'Relative importance (%, site hold-out R2 = {r2_oof:.3f})'
        for key in ['merged_rel', 'raw23', 'style17']:
            plot_importance_bar(tables[key], os.path.join(shap_dir, f'fig2a_importance_bar_{key}.png'), xlabel)
        scale = 100.0 / np.abs(shap_all).mean(axis=0).sum()
        plot_beeswarm(shap_all, X, os.path.join(shap_dir, 'fig2a_beeswarm_raw23_relative.png'), xlabel, scale)
        plot_beeswarm(shap_all, X, os.path.join(shap_dir, 'fig2a_beeswarm_raw23_original.png'), 'SHAP value (log yield)')
        import shap
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for j, f in enumerate(feats):
            plt.figure(figsize=(6, 4.5))
            shap.dependence_plot(j, shap_all, X, show=False, interaction_index=None)
            plt.gca().set_xlabel(x_label(f)); plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f'fig2a_dependence_{f}.png'), dpi=300, bbox_inches='tight'); plt.close()
        log('Top features (merged_rel):\n' + tables['merged_rel'][['rank', 'display', 'rel_importance_pct']].round(2).to_string(index=False))

    if True:
        final = joblib.load(os.path.join(args.seyp, 'models', 'SE_YP_model.pkl'))
        Xp = X.sample(min(args.pdp_rows, len(X)), random_state=SEED).reset_index(drop=True)
        curves = {}
        for f in PDP_PANELS + PDP_EXTRA:
            if f not in feats: continue
            curves[f] = pdp_curve(final, Xp, f)
            pd.DataFrame({'x': curves[f][0], 'yield_Mg_ha': curves[f][1]}).to_csv(os.path.join(pdp_dir, f'pdp_{f}.csv'), index=False)
            plot_pdp_panel(curves[f][0], curves[f][1], f, Xp, os.path.join(pdp_dir, f'fig2b_pdp_{f}.png'), False)
            plot_pdp_panel(curves[f][0], curves[f][1], f, Xp, os.path.join(pdp_dir, f'fig2b_pdp_{f}_highlight.png'), True)
        order_path = os.path.join(shap_dir, 'shap_importance_merged_rel.csv')
        order = list(pd.read_csv(order_path)['feature']) if os.path.exists(order_path) else PDP_PANELS
        order = [f for f in order if f in PDP_PANELS] + [f for f in PDP_PANELS if f not in order]
        plot_pdp_grid(curves, order, Xp, os.path.join(pdp_dir, 'fig2b_pdp_all_5x3.png'), False)
        plot_pdp_grid(curves, order, Xp, os.path.join(pdp_dir, 'fig2b_pdp_all_5x3_highlight.png'), True)
        log('PDP written for ' + ', '.join(curves))

    json.dump(dict(seyp_dir=args.seyp, features=args.features, feature_list=feats, explain=args.explain, pdp_rows=args.pdp_rows,
                   n_plots=int(len(mm)), n_sites=int(len(np.unique(sites))), r2_oof=r2_oof),
              open(os.path.join(args.out, 'fig2_run_config.json'), 'w'), indent=1)
    log('done: ' + args.out)


if __name__ == '__main__':
    main()
