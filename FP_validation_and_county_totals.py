#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL_MATRIX = 'results/SE-YP/oof/model_matrix_mundlak.csv'
OOF_SITE = 'results/SE-YP/oof/oof_site_mundlak_stack.csv'
RECOMMEND = 'results/SE-YP-opt/site_recommendations.csv'
FP_BASELINE = 'results/FP_survey/FP_survey_baseline_by_site.csv'
COUNTY_AREA = 'data/county_wheat_area.csv'
OUT = 'results/letter'
SCENARIOS = ['AOF', 'AOF95', 'EOF', 'ESOF']
SITE_COVARIATES = ['Altitude', 'SOM', 'TN', 'AN', 'AP', 'AK', 'pH', 'irrigation', 'precipitation', 'Frost_free_day',
                   'temperature', 'Nsurplus', 'N_med', 'P2O5_med', 'K20_med']
DISPLAY = {'N': 'N rate', 'P2O5': 'P2O5 rate', 'K20': 'K2O rate', 'N_med': 'Medium N', 'P2O5_med': 'Medium P2O5',
           'K20_med': 'Medium K2O', 'N_rel': 'N relative', 'P2O5_rel': 'P2O5 relative', 'K20_rel': 'K2O relative',
           'Frost_free_day': 'FFP', 'temperature': 'Tem', 'precipitation': 'Precipitation', 'irrigation': 'Irrigation',
           'Nsurplus': 'Legacy N', 'crop_type_sw': 'Spring wheat', 'crop_type_ww': 'Winter wheat'}


def vif_table(X):
    """Variance inflation factor of every column of X (OLS of each column on the others, with intercept)."""
    Xv = X.astype(float).values; n, p = Xv.shape; out = []
    for j in range(p):
        y = Xv[:, j]; A = np.column_stack([np.ones(n), np.delete(Xv, j, axis=1)])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out.append(dict(covariate=X.columns[j], display=DISPLAY.get(X.columns[j], X.columns[j]), VIF=1 / max(1 - r2, 1e-12)))
    return pd.DataFrame(out)


def a12_collinearity(mm):
    """Pairwise correlations among the covariates at the plot and site level, the strongly correlated pairs, and the variance inflation factors."""
    feats = [c for c in mm.columns if c not in ('site', 'yield', 'fold_site')]
    cp = mm[feats].corr()
    cp.to_csv(f'{OUT}/A12_correlation_plots.csv')
    sites = mm.drop_duplicates('site')
    cs = sites[SITE_COVARIATES].corr()
    cs.to_csv(f'{OUT}/A12_correlation_sites.csv')
    pairs = [dict(a=DISPLAY.get(a, a), b=DISPLAY.get(b, b), r_plots=cp.loc[a, b],
                  r_sites=(cs.loc[a, b] if a in cs and b in cs else np.nan))
             for i, a in enumerate(feats) for b in feats[i + 1:] if abs(cp.loc[a, b]) > 0.5]
    pd.DataFrame(pairs).sort_values('r_plots', key=abs, ascending=False).to_csv(f'{OUT}/A12_top_pairs.csv', index=False)
    vif_cols = [f for f in feats if f != 'crop_type_sw']
    v23 = vif_table(mm[vif_cols]); v23['set'] = '23 covariates'
    orig = [f for f in vif_cols if not (f.endswith('_med') or f.endswith('_rel'))]
    v17 = vif_table(mm[orig]); v17['set'] = '17 covariates'
    pd.concat([v23, v17]).to_csv(f'{OUT}/A12_vif.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    lab = [DISPLAY.get(f, f) for f in feats]
    im = ax.imshow(cp.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=90, fontsize=8)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson r (plots)')
    fig.tight_layout(); fig.savefig(f'{OUT}/A12_correlation_heatmap.png', dpi=300); plt.close(fig)
    print('A.12: pairs with |r| > 0.5 (plot level):')
    print(pd.DataFrame(pairs).sort_values('r_plots', key=abs, ascending=False).round(2).to_string(index=False))
    print('A.12: VIF above 5 (23 covariates):')
    print(v23[v23.VIF > 5].round(1).to_string(index=False) if (v23.VIF > 5).any() else '  none')


def fp_validation(rec, bl):
    """Survey fertilizer rates and predicted against surveyed FP yield by province and by province and year."""
    d = rec.merge(bl[['No.', 'province', 'year', 'FPs_N', 'FPs_P2O5', 'FPs_K2O', 'FPs_yield']], left_on='site', right_on='No.')
    g = d.groupby('province').agg(sites=('site', 'size'), N_survey=('FPs_N', 'mean'), P2O5_survey=('FPs_P2O5', 'mean'),
                                  K2O_survey=('FPs_K2O', 'mean'), yield_predicted=('FP_yield', 'mean'),
                                  yield_surveyed=('FPs_yield', 'mean')).sort_values('sites', ascending=False)
    g['ratio_predicted_to_surveyed'] = g['yield_predicted'] / g['yield_surveyed']
    g.loc['All sites'] = dict(sites=len(d), N_survey=d.FPs_N.mean(), P2O5_survey=d.FPs_P2O5.mean(), K2O_survey=d.FPs_K2O.mean(),
                              yield_predicted=d.FP_yield.mean(), yield_surveyed=d.FPs_yield.mean(),
                              ratio_predicted_to_surveyed=d.FP_yield.mean() / d.FPs_yield.mean())
    g.round(3).to_csv(f'{OUT}/table_S14_fp_by_province.csv')
    py = d.dropna(subset=['year']).groupby(['province', 'year']).agg(n=('site', 'size'), predicted=('FP_yield', 'mean'),
                                                                     surveyed=('FPs_yield', 'mean')).reset_index()
    r = np.corrcoef(py['surveyed'], py['predicted'])[0, 1]
    py.to_csv(f'{OUT}/fig_S16_province_year.csv', index=False)
    fig, ax = plt.subplots(figsize=(4.9, 4.7))
    ax.scatter(py['surveyed'] / 1000, py['predicted'] / 1000, s=np.clip(py['n'] / 5, 6, 120), alpha=0.55, edgecolor='k', linewidth=0.3)
    lim = (2, 8.5); ax.plot(lim, lim, '--', color='k', lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('Surveyed farm yield (t ha$^{-1}$)'); ax.set_ylabel('Predicted yield at the survey rates (t ha$^{-1}$)')
    ax.text(0.04, 0.96, f'r = {r:.2f}\n{len(py)} province-years', transform=ax.transAxes, va='top')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_S16.png', dpi=300); plt.close(fig)
    print(f'Table S14 / Fig. S16: predicted FP yield {d.FP_yield.mean():.0f}, surveyed {d.FPs_yield.mean():.0f}, '
          f'{len(py)} province-years, r = {r:.3f}')
    top = g.drop('All sites').head(10)
    print(f'  ratio predicted to surveyed, 10 largest provinces: {top.ratio_predicted_to_surveyed.min():.2f} to {top.ratio_predicted_to_surveyed.max():.2f}')
    return d


def gains(d, base):
    rows = []
    for s in SCENARIOS:
        rows.append(dict(scenario=s, baseline=base,
                         yield_gain_pct=100 * (d[f'{s}_yield'] / d[f'{base}_yield'] - 1).mean(),
                         yield_gain_kg=(d[f'{s}_yield'] - d[f'{base}_yield']).mean(),
                         NFB_gain=(d[f'{s}_NFB'] - d[f'{base}_NFB']).mean(),
                         NEEB_gain=(d[f'{s}_NEEB'] - d[f'{base}_NEEB']).mean(),
                         N_change_pct=100 * (d[f'{s}_N'].mean() / d[f'{base}_N'].mean() - 1),
                         P2O5_change_pct=100 * (d[f'{s}_P2O5'].mean() / d[f'{base}_P2O5'].mean() - 1),
                         K2O_change_pct=100 * (d[f'{s}_K2O'].mean() / d[f'{base}_K2O'].mean() - 1)))
    return pd.DataFrame(rows)


def a18_out_of_range(d, oof):
    """Share of sites whose survey rate exceeds the highest tested rate, and the gains over FP with and without those sites."""
    mx = oof.drop_duplicates('site').set_index('site')[['Nmax', 'Pmax', 'Kmax']]
    d = d.join(mx, on='site')
    over = (d.FPs_N > d.Nmax + 1e-9) | (d.FPs_P2O5 > d.Pmax + 1e-9)
    over_k = d.FPs_K2O > d.Kmax + 1e-9
    g_all = gains(d, 'FP'); g_in = gains(d[~over], 'FP')
    cmp = g_all.merge(g_in, on=['scenario', 'baseline'], suffixes=('_all', '_within_range'))
    cmp['yield_gain_pct_diff_points'] = cmp['yield_gain_pct_all'] - cmp['yield_gain_pct_within_range']
    cmp['NFB_gain_rel_diff_pct'] = 100 * (cmp['NFB_gain_within_range'] / cmp['NFB_gain_all'] - 1)
    cmp['NEEB_gain_rel_diff_pct'] = 100 * (cmp['NEEB_gain_within_range'] / cmp['NEEB_gain_all'] - 1)
    cmp.insert(0, 'share_sites_survey_N_or_P2O5_above_tested_max', over.mean())
    cmp.insert(1, 'share_sites_survey_K2O_above_tested_max', over_k.mean())
    cmp.round(3).to_csv(f'{OUT}/A18_fp_out_of_range.csv', index=False)
    print(f'A.18: surveyed N or P2O5 above the tested maximum at {100 * over.mean():.1f}% of sites (K2O: {100 * over_k.mean():.1f}%)')
    print('  yield gain (points) and benefit gain (%) change when those sites are excluded:')
    print(cmp[['scenario', 'yield_gain_pct_all', 'yield_gain_pct_within_range', 'NFB_gain_rel_diff_pct', 'NEEB_gain_rel_diff_pct']].round(2).to_string(index=False))


def s20_totals(d, bl, area):
    """Sites per county, area coverage, and area-weighted national totals of each scenario over the counties with experimental sites."""
    d = d.merge(bl[['No.', 'County code']], on='No.')
    acol = 'wheat_area_ha' if 'wheat_area_ha' in area.columns else 'area_ha'
    area = area.set_index('County code')[acol]
    cty = d.groupby('County code')
    cov = pd.DataFrame({'sites': cty.size()}); cov['area_ha'] = area.reindex(cov.index).values
    cov.to_csv(f'{OUT}/county_coverage.csv')
    total_area = area.sum(); cov_area = cov.area_ha.sum()
    few = cov[cov.sites < 5]
    print(f'coverage: {len(cov):,} counties with sites, {cov_area / 1e6:.2f} Mha = {100 * cov_area / total_area:.0f}% of the mapped area '
          f'({total_area / 1e6:.2f} Mha, {len(area):,} counties); median {cov.sites.median():.0f} sites per county; '
          f'{len(few)} counties ({100 * len(few) / len(cov):.0f}%) with fewer than 5 sites hold {100 * few.area_ha.sum() / cov_area:.1f}% of the covered area')
    rows = []
    for s in ['FP', 'MED'] + SCENARIOS:
        m = cty[[f'{s}_yield', f'{s}_N', f'{s}_P2O5', f'{s}_K2O', f'{s}_NFB', f'{s}_NEEB']].mean()
        a = cov.area_ha.reindex(m.index).fillna(0)
        rows.append(dict(scenario=s, production_Mt=(m[f'{s}_yield'] * a).sum() / 1e9, N_kt=(m[f'{s}_N'] * a).sum() / 1e6,
                         P2O5_kt=(m[f'{s}_P2O5'] * a).sum() / 1e6, K2O_kt=(m[f'{s}_K2O'] * a).sum() / 1e6,
                         NFB_billion_usd=(m[f'{s}_NFB'] * a).sum() / 1e9, NEEB_billion_usd=(m[f'{s}_NEEB'] * a).sum() / 1e9))
    t = pd.DataFrame(rows)
    for c in ['production_Mt', 'N_kt', 'P2O5_kt', 'K2O_kt', 'NFB_billion_usd', 'NEEB_billion_usd']:
        t[f'{c}_vs_FP'] = t[c] - t.loc[t.scenario == 'FP', c].iloc[0]
    t.round(3).to_csv(f'{OUT}/table_S20_totals_point.csv', index=False)
    print('Table S20 (point estimates, counties with sites):'); print(t.round(2).to_string(index=False))


def main():
    os.makedirs(OUT, exist_ok=True)
    mm = pd.read_csv(MODEL_MATRIX)
    a12_collinearity(mm)
    rec = pd.read_csv(RECOMMEND); bl = pd.read_csv(FP_BASELINE); oof = pd.read_csv(OOF_SITE, usecols=['site', 'Nmax', 'Pmax', 'Kmax'])
    d = fp_validation(rec, bl)
    a18_out_of_range(d, oof)
    s20_totals(d, bl, pd.read_csv(COUNTY_AREA, encoding='utf-8-sig'))
    print('written to', OUT)


if __name__ == '__main__':
    main()
