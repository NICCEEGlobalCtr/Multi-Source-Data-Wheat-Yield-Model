#!/usr/bin/env python3
import os, json, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

SEYP_DIR = 'results/SE-YP'
OPT_FILE = 'results/SE-YP-opt/site_recommendations.csv'
FP_BASELINE = 'results/FP_survey/FP_survey_baseline_by_site.csv'
COUNTY_AREA = 'data/county_wheat_area.csv'
OUT_DIR = 'results/SE-YP-bootstrap-optional'
USE_GPU = True

SEED = 42
N_BOOT = 50
SITE_SUBSAMPLE = 1500
STEP = 5
SITES_PER_CHUNK = 50
AOF95_FRAC = 0.95
XGB_PARAMS = dict(n_estimators=411, max_depth=12, learning_rate=0.021464882631168122,
                  subsample=0.9479577158696486, colsample_bytree=0.7033846712737518)
PRICE_GRAIN, PRICE_N, PRICE_P, PRICE_K = 0.376, 0.78, 0.45, 0.91
CV_PRICE_GRAIN, CV_PRICE_FERT, CV_N_ENV_COST, CV_P_ENV_COST = 0.10, 0.20, 0.50, 0.50
GRAIN_N, GRAIN_P, GRAIN_K, P2O5_TO_P, K2O_TO_K = 0.021, 0.0042, 0.0052, 0.436, 0.830
P_ENV_PER_KG_P2O5 = 0.00247
NR = {'North':     (lambda x: 3.21 + 0.068 * x, lambda x: 4.93 * np.exp(0.0057 * x), lambda x: 0.26 * np.exp(0.0045 * x)),
      'Center':    (lambda x: 2.69 + 0.069 * x, lambda x: 3.63 * np.exp(0.0080 * x), lambda x: 0.50 * np.exp(0.0032 * x)),
      'Yangtze':   (lambda x: 0.61 + 0.13 * x,  lambda x: 1.64 * np.exp(0.0078 * x), lambda x: 0.59 * np.exp(0.0060 * x)),
      'Southwest': (lambda x: 0.61 + 0.13 * x,  lambda x: 1.64 * np.exp(0.0078 * x), lambda x: 0.59 * np.exp(0.0060 * x))}
REGION_ALIAS = {'North': 'North', 'North China': 'North', 'Center': 'Center', 'Center China': 'Center',
                'Yangtze River Plain': 'Yangtze', 'Yangtze_River_Plain': 'Yangtze', 'YangtzeRiverPlain': 'Yangtze',
                'Yangtze': 'Yangtze', 'Southwest': 'Southwest'}
COST_N2O, COST_NO3, COST_NH3, COST_N_PROD, EF_LEACH, EF_NH3 = 11.5, 1.32, 5.41, 0.2228, 0.0075, 0.01
STATIC = ['Altitude', 'SOM', 'TN', 'AN', 'AP', 'AK', 'pH', 'irrigation', 'precipitation', 'Frost_free_day',
          'temperature', 'Nsurplus', 'crop_type_sw', 'crop_type_ww']
RATES, MEDS, RELS = ['N', 'P2O5', 'K20'], ['N_med', 'P2O5_med', 'K20_med'], ['N_rel', 'P2O5_rel', 'K20_rel']
SCEN = ['FP', 'MED', 'AOF', 'AOF95', 'EOF', 'ESOF']


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def make_group_folds(groups, n_splits=5, seed=SEED):
    groups = np.asarray(groups); uniq = np.unique(groups); rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq)); fog = np.empty(len(uniq), int); fog[perm] = np.arange(len(uniq)) % n_splits
    gm = dict(zip(uniq, fog)); f = np.array([gm[g] for g in groups])
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_splits)]


def make_xgb(params, seed=SEED, use_gpu=USE_GPU):
    import xgboost as xgb
    kw = dict(params); kw.update(dict(random_state=seed, n_jobs=-1, tree_method='hist'))
    if use_gpu:
        kw['device'] = 'cuda'
    return xgb.XGBRegressor(**kw)


def make_stack(xgb_params, inner_folds, seed=SEED):
    """Stacking ensemble of XGBoost, LightGBM, random forest and linear regression with a linear meta-learner, fitted with the given inner folds."""
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import LinearRegression
    ests = [('xgb', make_xgb(xgb_params, seed)),
            ('lgb', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=seed, verbose=-1)),
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1)),
            ('lr', LinearRegression())]
    return StackingRegressor(estimators=ests, final_estimator=LinearRegression(), cv=list(inner_folds), n_jobs=1)


def n_cost_terms(reg, n):
    """The three terms of the N environmental cost: direct N2O, leaching with its indirect N2O, and NH3 volatilization with its indirect N2O and the production emissions."""
    nh3_f, no3_f, n2o_f = NR[reg]
    nh3, no3 = nh3_f(n), no3_f(n)
    return COST_N2O * n2o_f(n), (COST_NO3 + COST_N2O * EF_LEACH) * no3, (COST_NH3 + COST_N2O * EF_NH3) * nh3 + COST_N_PROD * n


def n_cost(reg, n, mult=(1.0, 1.0, 1.0)):
    t1, t2, t3 = n_cost_terms(reg, n)
    return mult[0] * t1 + mult[1] * t2 + mult[2] * t3


def grid_values(vmax, med, step=STEP):
    v = np.arange(0, vmax + 1e-9, step, dtype=float)
    return np.unique(np.concatenate([v, [0.0, 0.5 * med, med, vmax]]))


def design(site, n, p, k, feats):
    d = {c: np.full(len(n), site[c]) for c in STATIC + MEDS}
    d['N'], d['P2O5'], d['K20'] = n, p, k
    for rate, med, rel in zip(RATES, MEDS, RELS):
        d[rel] = np.where(d[med] > 0, d[rate] / d[med], 0.0)
    return pd.DataFrame(d)[feats]


def scenario(y, n, p, k, ndep, reg, pg, pn, pp, pk, nmult, pmult):
    """Benefits (NFB, NEEB) and N, P and K surpluses of one scenario under the price and cost draws of a replicate."""
    nfb = y * pg - n * pn - p * pp - k * pk
    ncost = np.array([n_cost(r, x) for r, x in zip(reg, n)]) * nmult
    neeb = nfb - ncost - P_ENV_PER_KG_P2O5 * p * pmult
    return dict(N=n, P2O5=p, K2O=k, yld=y, NFB=nfb, NEEB=neeb,
                Nsur=n + ndep - GRAIN_N * y, Psur=P2O5_TO_P * p - GRAIN_P * y, Ksur=K2O_TO_K * k - GRAIN_K * y)


def agg_eff(v, ndep):
    return (np.sum(GRAIN_N * v['yld']) / np.sum(v['N'] + ndep), np.sum(GRAIN_P * v['yld']) / np.sum(P2O5_TO_P * v['P2O5']),
            np.sum(GRAIN_K * v['yld']) / np.sum(K2O_TO_K * v['K2O']))


def main():
    """Run the bootstrap replicates (resample sites, retrain the ensemble, repeat the grid search at the subsample sites) and write the replicate, site-level and pathway-sensitivity outputs."""
    t0 = time.time()
    os.makedirs(f'{OUT_DIR}/sites', exist_ok=True)
    feats = json.load(open(f'{SEYP_DIR}/models/feature_list_mundlak.json'))
    mm = pd.read_csv(f'{SEYP_DIR}/oof/model_matrix_mundlak.csv')
    X_all, Y_all, site_ids = mm[feats], np.log1p(mm['yield']), mm['site'].values
    oof = pd.read_csv(f'{SEYP_DIR}/oof/oof_site_mundlak_stack.csv')
    sites = mm.drop_duplicates('site').set_index('site')[STATIC + MEDS].join(
        oof.drop_duplicates('site').set_index('site')[['Region', 'Ndep', 'Nmax', 'Pmax', 'Kmax', 'province']])
    bl = pd.read_csv(FP_BASELINE).set_index('No.')
    sites = sites.join(bl[['FPs_N', 'FPs_P2O5', 'FPs_K2O', 'County code']])
    sites['reg'] = sites['Region'].map(REGION_ALIAS)
    area = pd.read_csv(COUNTY_AREA, encoding='utf-8-sig')
    acol = 'wheat_area_ha' if 'wheat_area_ha' in area.columns else 'area_ha'
    area = area.set_index('County code')[acol]
    all_sites = np.array(sites.index)
    rng = np.random.RandomState(SEED)
    sub = rng.choice(all_sites, min(SITE_SUBSAMPLE, len(all_sites)), replace=False)
    S = sites.loc[sub]; ndep = S['Ndep'].values.astype(float); reg = S['reg'].values
    n_c = sites['County code'].value_counts()
    w = np.array([area.get(c, 0.0) / n_c.get(c, 1) for c in S['County code'].values]) * (len(all_sites) / len(sub))
    log(f'{len(mm):,} plots, {len(all_sites):,} sites; subsample {len(sub):,}; weighted area {w.sum() / 1e6:.2f} Mha')

    grids = []
    for s in sub:
        r = S.loc[s]
        n, p, k = np.meshgrid(grid_values(r['Nmax'], r['N_med']), grid_values(r['Pmax'], r['P2O5_med']), grid_values(r['Kmax'], r['K20_med']), indexing='ij')
        grids.append((n.ravel(), p.ravel(), k.ravel()))
    n_grid = sum(len(g[0]) for g in grids)

    def predict_grid(model):
        out = np.empty(n_grid); pos = 0
        for start in range(0, len(sub), SITES_PER_CHUNK):
            Xc = pd.concat([design(S.loc[s_], n_, p_, k_, feats) for s_, (n_, p_, k_) in zip(sub[start:start + SITES_PER_CHUNK], grids[start:start + SITES_PER_CHUNK])], ignore_index=True)
            out[pos:pos + len(Xc)] = np.expm1(model.predict(Xc)); pos += len(Xc)
        return out
    fp_rates = tuple(S[c].values.astype(float) for c in ('FPs_N', 'FPs_P2O5', 'FPs_K2O'))
    med_rates = tuple(S[c].values.astype(float) for c in ('N_med', 'P2O5_med', 'K20_med'))
    def rows_at(rates):
        d = {c: S[c].values for c in STATIC + MEDS}; d['N'], d['P2O5'], d['K20'] = rates
        for rate, med, rel in zip(RATES, MEDS, RELS):
            d[rel] = np.where(d[med] > 0, d[rate] / d[med], 0.0)
        return pd.DataFrame(d)[feats]
    X_fp, X_med = rows_at(fp_rates), rows_at(med_rates)
    log(f'grid: {n_grid:,} combinations for {len(sub):,} sites')

    def optimize(yhat, pn, pp, pk, pg, nmult, pmult, cost_mult=(1.0, 1.0, 1.0)):
        """Select the AOF, AOF95, EOF and ESOF combination of every subsample site from the grid predictions."""
        opt = {o: {'N': [], 'P2O5': [], 'K2O': [], 'yld': []} for o in ('AOF', 'AOF95', 'EOF', 'ESOF')}
        pos = 0
        for s, (n, p, k) in zip(sub, grids):
            m_ = len(n); y = yhat[pos:pos + m_]; pos += m_
            cost = n * pn + p * pp + k * pk; nfb = y * pg - cost
            neeb = nfb - n_cost(S.loc[s, 'reg'], n, cost_mult) * nmult - P_ENV_PER_KG_P2O5 * p * pmult
            ok95 = y >= AOF95_FRAC * y.max()
            picks = {'AOF': int(np.argmax(y)), 'EOF': int(np.argmax(nfb)), 'ESOF': int(np.argmax(neeb)),
                     'AOF95': int(np.flatnonzero(ok95)[np.argmin(cost[ok95])])}
            for o, i in picks.items():
                opt[o]['N'].append(n[i]); opt[o]['P2O5'].append(p[i]); opt[o]['K2O'].append(k[i]); opt[o]['yld'].append(y[i])
        return {o: {kk: np.array(vv) for kk, vv in d.items()} for o, d in opt.items()}

    rep_rows, sd_rows, yhat0 = [], [], None
    for b in range(N_BOOT):
        tb = time.time()
        if b == 0:
            tr = np.arange(len(mm)); pg, pf, nm, pm = PRICE_GRAIN, 1.0, 1.0, 1.0
        else:
            boot = rng.choice(all_sites, len(all_sites), replace=True); counts = pd.Series(boot).value_counts()
            tr = np.concatenate([np.flatnonzero(site_ids == s_).repeat(c) for s_, c in counts.items()])
            pg = PRICE_GRAIN * np.exp(rng.normal(0, CV_PRICE_GRAIN)); pf = np.exp(rng.normal(0, CV_PRICE_FERT))
            nm = np.exp(rng.normal(0, CV_N_ENV_COST)); pm = np.exp(rng.normal(0, CV_P_ENV_COST))
        pn, pp, pk = PRICE_N * pf, PRICE_P * pf, PRICE_K * pf
        model = make_stack(XGB_PARAMS, make_group_folds(site_ids[tr], 5, SEED + b), SEED + b)
        model.fit(X_all.iloc[tr], Y_all.iloc[tr])
        yhat = predict_grid(model)
        if b == 0:
            yhat0 = yhat.copy()
        opt = optimize(yhat, pn, pp, pk, pg, nm, pm)
        sc = {'FP': scenario(np.expm1(model.predict(X_fp)), *fp_rates, ndep, reg, pg, pn, pp, pk, nm, pm),
              'MED': scenario(np.expm1(model.predict(X_med)), *med_rates, ndep, reg, pg, pn, pp, pk, nm, pm)}
        for o in opt:
            sc[o] = scenario(opt[o]['yld'], opt[o]['N'], opt[o]['P2O5'], opt[o]['K2O'], ndep, reg, pg, pn, pp, pk, nm, pm)
        row = dict(rep=b, price_grain=pg, fert_price_mult=pf, n_env_cost_mult=nm, p_env_cost_mult=pm)
        for name, v in sc.items():
            for kk in ('N', 'P2O5', 'K2O', 'yld', 'NFB', 'NEEB', 'Nsur', 'Psur', 'Ksur'):
                row[f'{name}_{kk}'] = float(np.mean(v[kk]))
            row[f'{name}_NUE'], row[f'{name}_PUE'], row[f'{name}_KUE'] = agg_eff(v, ndep)
            for kk in ('yld', 'NFB', 'NEEB'):
                row[f'{name}_tot_{kk}'] = float(np.sum(w * v[kk]) / 1e9)
        for name in SCEN[1:]:
            row[f'{name}_dyield_pct_FP'] = 100 * (row[f'{name}_yld'] / row['FP_yld'] - 1)
            row[f'{name}_dN_pct_FP'] = 100 * (row[f'{name}_N'] / row['FP_N'] - 1)
            for kk in ('NFB', 'NEEB', 'Nsur', 'Psur', 'Ksur'):
                row[f'{name}_d{kk}_FP'] = row[f'{name}_{kk}'] - row[f'FP_{kk}']
            for kk in ('yld', 'NFB', 'NEEB'):
                row[f'{name}_dtot_{kk}_FP'] = row[f'{name}_tot_{kk}'] - row[f'FP_tot_{kk}']
        for name in SCEN[2:]:
            row[f'{name}_dN_MED'] = row[f'{name}_N'] - row['MED_N']
            row[f'{name}_dyield_pct_MED'] = 100 * (row[f'{name}_yld'] / row['MED_yld'] - 1)
            for kk in ('NFB', 'NEEB'):
                row[f'{name}_d{kk}_MED'] = row[f'{name}_{kk}'] - row[f'MED_{kk}']
                row[f'{name}_dtot_{kk}_MED'] = row[f'{name}_tot_{kk}'] - row[f'MED_tot_{kk}']
        row['ESOF_dN_EOF'] = row['ESOF_N'] - row['EOF_N']; row['ESOF_dyield_pct_EOF'] = 100 * (row['ESOF_yld'] / row['EOF_yld'] - 1)
        row['ESOF_dNFB_EOF'] = row['ESOF_NFB'] - row['EOF_NFB']; row['ESOF_dNEEB_EOF'] = row['ESOF_NEEB'] - row['EOF_NEEB']
        rep_rows.append(row)
        pd.DataFrame({'rep': b, 'site': sub, 'county': S['County code'].values, 'weight': w,
                      **{f'{n_}_{kk}': v_[kk] for n_, v_ in sc.items() for kk in ('N', 'P2O5', 'K2O', 'yld', 'NFB', 'NEEB')}}
                     ).to_csv(f'{OUT_DIR}/sites/rep{b:02d}.csv', index=False)
        for o in opt:
            sd_rows.append(pd.DataFrame({'rep': b, 'site': sub, 'objective': o, 'N': opt[o]['N'], 'P2O5': opt[o]['P2O5'], 'K2O': opt[o]['K2O']}))
        pd.DataFrame(rep_rows).to_csv(f'{OUT_DIR}/replicates.csv', index=False)
        log(f'  replicate {b}: EOF N {row["EOF_N"]:.0f}, ESOF N {row["ESOF_N"]:.0f}, EOF dNFB {row["EOF_dNFB_FP"]:.0f} ({time.time() - tb:.0f}s)')

        if b == 0:
            prow = []
            for term in range(3):
                for f_ in (0.5, 1.5):
                    mult = [1.0, 1.0, 1.0]; mult[term] = f_
                    o_ = optimize(yhat0, PRICE_N, PRICE_P, PRICE_K, PRICE_GRAIN, 1.0, 1.0, tuple(mult))['ESOF']
                    esof = scenario(o_['yld'], o_['N'], o_['P2O5'], o_['K2O'], ndep, reg, PRICE_GRAIN, PRICE_N, PRICE_P, PRICE_K, 1.0, 1.0)
                    ncost_e = np.array([n_cost(r_, x, tuple(mult)) for r_, x in zip(reg, o_['N'])])
                    ncost_f = np.array([n_cost(r_, x, tuple(mult)) for r_, x in zip(reg, fp_rates[0])])
                    neeb_e = esof['NFB'] - ncost_e - P_ENV_PER_KG_P2O5 * o_['P2O5']
                    neeb_f = sc['FP']['NFB'] - ncost_f - P_ENV_PER_KG_P2O5 * fp_rates[1]
                    prow.append(dict(term=['direct N2O', 'leaching', 'NH3 and production'][term], multiplier=f_,
                                     ESOF_N=float(np.mean(o_['N'])), ESOF_NEEB_gain_over_FP=float(np.mean(neeb_e - neeb_f))))
            pd.DataFrame(prow).to_csv(f'{OUT_DIR}/pathway_sensitivity.csv', index=False)
            log('pathway sensitivity:\n' + pd.DataFrame(prow).round(1).to_string(index=False))

    reps = pd.DataFrame(rep_rows)
    sd = pd.concat(sd_rows).groupby(['objective', 'site'])[['N', 'P2O5', 'K2O']].std()
    sd.to_csv(f'{OUT_DIR}/site_rate_sd.csv')
    log('median SD of the recommended N rate per site: ' + ', '.join(f'{o} {sd.loc[o, "N"].median():.0f}' for o in ('AOF', 'AOF95', 'EOF', 'ESOF')))

    rec = pd.read_csv(OPT_FILE)
    lo, hi, p0 = reps.quantile(0.025), reps.quantile(0.975), reps.iloc[0]
    def ratio(point, col): return point * lo[col] / p0[col], point * hi[col] / p0[col]
    rec = rec.merge(bl[['County code']], left_on='site', right_index=True)
    cty = rec.groupby('County code')[[f'{s}_{kk}' for s in SCEN for kk in ('yield', 'NFB', 'NEEB')]].mean()
    a = area.reindex(cty.index).fillna(0.0)
    tot = {s: dict(production_Mt=(cty[f'{s}_yield'] * a).sum() / 1e9, NFB_bn=(cty[f'{s}_NFB'] * a).sum() / 1e9, NEEB_bn=(cty[f'{s}_NEEB'] * a).sum() / 1e9) for s in SCEN}
    s20 = []
    for s in SCEN:
        d = dict(scenario=s, **tot[s])
        if s != 'FP':
            for kk, col in (('production_Mt', 'yld'), ('NFB_bn', 'NFB'), ('NEEB_bn', 'NEEB')):
                d[f'{kk}_change_FP'] = tot[s][kk] - tot['FP'][kk]; d[f'{kk}_change_FP_lo'], d[f'{kk}_change_FP_hi'] = ratio(d[f'{kk}_change_FP'], f'{s}_dtot_{col}_FP')
        if s not in ('FP', 'MED'):
            for kk, col in (('NFB_bn', 'NFB'), ('NEEB_bn', 'NEEB')):
                d[f'{kk}_change_MED'] = tot[s][kk] - tot['MED'][kk]; d[f'{kk}_change_MED_lo'], d[f'{kk}_change_MED_hi'] = ratio(d[f'{kk}_change_MED'], f'{s}_dtot_{col}_MED')
        s20.append(d)
    pd.DataFrame(s20).to_csv(f'{OUT_DIR}/table_S20.csv', index=False)

    json.dump(dict(seed=SEED, n_boot=N_BOOT, subsample=SITE_SUBSAMPLE, step=STEP, grid='5 kg steps plus the four tested rates',
                   cv=dict(grain=CV_PRICE_GRAIN, fertilizer=CV_PRICE_FERT, n_env=CV_N_ENV_COST, p_env=CV_P_ENV_COST),
                   prices=dict(grain=PRICE_GRAIN, N=PRICE_N, P2O5=PRICE_P, K2O=PRICE_K),
                   xgb_params=XGB_PARAMS, runtime_min=(time.time() - t0) / 60), open(f'{OUT_DIR}/run_config.json', 'w'), indent=1)
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {OUT_DIR}')


if __name__ == '__main__':
    main()
