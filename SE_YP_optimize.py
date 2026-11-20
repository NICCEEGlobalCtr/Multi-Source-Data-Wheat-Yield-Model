#!/usr/bin/env python3
import os, json, time, warnings
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

SEYP_DIR = 'results/SE-YP'
FEATURES = 'mundlak'
FP_BASELINE = 'results/FP_survey/FP_survey_baseline_by_site.csv'
OUT_DIR = 'results/SE-YP-opt'
STEP = 5
BATCH = 16
AOF95_FRAC = 0.95
TOL_YIELD = 0.01
TOL_DOLLAR = 10.0
SEED = 42
PRICE_GRAIN, PRICE_N, PRICE_P, PRICE_K = 0.376, 0.78, 0.45, 0.91

NR = {'North':     (lambda x: 3.21 + 0.068 * x, lambda x: 4.93 * np.exp(0.0057 * x), lambda x: 0.26 * np.exp(0.0045 * x)),
      'Center':    (lambda x: 2.69 + 0.069 * x, lambda x: 3.63 * np.exp(0.0080 * x), lambda x: 0.50 * np.exp(0.0032 * x)),
      'Yangtze':   (lambda x: 0.61 + 0.13 * x,  lambda x: 1.64 * np.exp(0.0078 * x), lambda x: 0.59 * np.exp(0.0060 * x)),
      'Southwest': (lambda x: 0.61 + 0.13 * x,  lambda x: 1.64 * np.exp(0.0078 * x), lambda x: 0.59 * np.exp(0.0060 * x))}
REGION_ALIAS = {'North': 'North', 'North China': 'North', 'Center': 'Center', 'Center China': 'Center',
                'Yangtze River Plain': 'Yangtze', 'Yangtze_River_Plain': 'Yangtze', 'YangtzeRiverPlain': 'Yangtze',
                'Yangtze': 'Yangtze', 'Southwest': 'Southwest'}
COST_N2O, COST_NO3, COST_NH3, COST_N_PROD = 11.5, 1.32, 5.41, 0.2228
P_ENV_PER_KG_P2O5 = 0.00247

STATIC = ['Altitude', 'SOM', 'TN', 'AN', 'AP', 'AK', 'pH', 'irrigation', 'precipitation', 'Frost_free_day',
          'temperature', 'Nsurplus', 'crop_type_sw', 'crop_type_ww']
RATES, MEDS, RELS = ['N', 'P2O5', 'K20'], ['N_med', 'P2O5_med', 'K20_med'], ['N_rel', 'P2O5_rel', 'K20_rel']
OUT_COLS = ['N', 'P2O5', 'K2O', 'yield', 'NFB', 'NEEB', 'N_env_cost', 'P_env_cost']


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def n_env_cost(region, n):
    """Environmental cost of N ($ ha-1) from the regional NH3, NO3 and N2O loss functions, the indirect N2O and the production emissions."""
    nh3_f, no3_f, n2o_f = NR[region]
    nh3, no3 = nh3_f(n), no3_f(n)
    n2o = n2o_f(n) + 0.0075 * no3 + 0.01 * nh3
    return COST_N2O * n2o + COST_NO3 * no3 + COST_NH3 * nh3 + COST_N_PROD * n


def objectives(n, p, k, y, region):
    """Net farmer benefit (NFB), net ecological-economic benefit (NEEB) and the N and P environmental costs of a fertilizer combination."""
    nfb = PRICE_GRAIN * y - PRICE_N * n - PRICE_P * p - PRICE_K * k
    nc = n_env_cost(region, n); pc = P_ENV_PER_KG_P2O5 * p
    return dict(N=n, P2O5=p, K2O=k, **{'yield': y}, NFB=nfb, NEEB=nfb - nc - pc, N_env_cost=nc, P_env_cost=pc)


def grid_values(vmax, med, step):
    """Multiples of step from 0 to the highest tested rate, plus the four tested rates of the design, so that every tested combination is on the grid."""
    v = np.arange(0, vmax + 1e-9, step, dtype=float)
    return np.unique(np.concatenate([v, [0.0, 0.5 * med, med, vmax]]))


def design(site, n, p, k, feats):
    """Model matrix of one site for a set of N, P2O5 and K2O combinations, with the relative rates recomputed."""
    d = {c: np.full(len(n), site[c]) for c in STATIC + MEDS}
    d['N'], d['P2O5'], d['K20'] = n, p, k
    for rate, med, rel in zip(RATES, MEDS, RELS):
        d[rel] = np.where(d[med] > 0, d[rate] / d[med], 0.0)
    return pd.DataFrame(d)[feats]


def main():
    """At every site, predict yield over the grid with the final model and keep the optimum of each objective (AOF, AOF95, EOF, ESOF) together with the FP and medium-rate scenarios."""
    args = SimpleNamespace(seyp=SEYP_DIR, out=OUT_DIR, fp_baseline=FP_BASELINE, step=STEP, sites=0, batch=BATCH)
    os.makedirs(args.out, exist_ok=True); t0 = time.time()

    model = joblib.load(os.path.join(args.seyp, 'models', 'SE_YP_model.pkl'))
    feats = json.load(open(os.path.join(args.seyp, 'models', f'feature_list_{FEATURES}.json')))
    mm = pd.read_csv(os.path.join(args.seyp, 'oof', f'model_matrix_{FEATURES}.csv'))
    oof = pd.read_csv(os.path.join(args.seyp, 'oof', f'oof_site_{FEATURES}_stack.csv'))
    sites = mm.drop_duplicates('site').set_index('site')[STATIC + MEDS].join(
        oof.drop_duplicates('site').set_index('site')[['Region', 'Ndep', 'Nmax', 'Pmax', 'Kmax']])
    sites['reg'] = sites['Region'].map(REGION_ALIAS)
    assert sites['reg'].notna().all(), f"unknown Region names {sites.loc[sites.reg.isna(), 'Region'].unique()}"
    if args.sites:
        sites = sites.sample(args.sites, random_state=SEED)
    has_fp = os.path.exists(args.fp_baseline)
    if has_fp:
        bl = pd.read_csv(args.fp_baseline).set_index('No.')[['FPs_N', 'FPs_P2O5', 'FPs_K2O']]
        sites = sites.join(bl)
        assert sites['FPs_N'].notna().all(), 'sites missing from the FP baseline file'
    else:
        log(f'WARNING: {args.fp_baseline} not found, FP scenario skipped')
    log(f'{len(sites):,} sites, step {args.step} kg/ha')

    rows, bnd = [], []
    ids = list(sites.index)
    for b0 in range(0, len(ids), args.batch):
        batch = ids[b0:b0 + args.batch]; parts, grids, fixed = [], {}, {}
        for s in batch:
            r = sites.loc[s]
            n, p, k = np.meshgrid(grid_values(r['Nmax'], r['N_med'], args.step), grid_values(r['Pmax'], r['P2O5_med'], args.step),
                                  grid_values(r['Kmax'], r['K20_med'], args.step), indexing='ij')
            grids[s] = (n.ravel(), p.ravel(), k.ravel()); parts.append(design(r, *grids[s], feats))
            fixed[s] = {'MED': (r['N_med'], r['P2O5_med'], r['K20_med'])}
            if has_fp:
                fixed[s]['FP'] = (r['FPs_N'], r['FPs_P2O5'], r['FPs_K2O'])
            for fn, fp, fk in fixed[s].values():
                parts.append(design(r, np.array([fn]), np.array([fp]), np.array([fk]), feats))
        yhat = np.expm1(model.predict(pd.concat(parts, ignore_index=True)))
        pos = 0
        for s in batch:
            r = sites.loc[s]; n, p, k = grids[s]; m = len(n)
            y = yhat[pos:pos + m]; pos += m
            o = objectives(n, p, k, y, r['reg'])
            cost = PRICE_N * n + PRICE_P * p + PRICE_K * k
            ymax = y.max(); ok95 = y >= AOF95_FRAC * ymax
            pick = {'AOF': int(np.argmax(y)), 'AOF95': int(np.flatnonzero(ok95)[np.argmin(cost[ok95])]),
                    'EOF': int(np.argmax(o['NFB'])), 'ESOF': int(np.argmax(o['NEEB']))}
            row = dict(site=s, Ndep=r['Ndep'])
            for name, i in pick.items():
                for c in OUT_COLS:
                    row[f'{name}_{c}'] = float(o[c][i])
            for name, (fn, fp, fk) in fixed[s].items():
                of = objectives(np.array([fn]), np.array([fp]), np.array([fk]), yhat[pos:pos + 1], r['reg']); pos += 1
                for c in OUT_COLS:
                    row[f'{name}_{c}'] = float(of[c][0])
            rows.append(row)
            b = dict(site=s)
            for name, band in [('AOF', y >= (1 - TOL_YIELD) * ymax),
                               ('EOF', o['NFB'] >= o['NFB'][pick['EOF']] - TOL_DOLLAR),
                               ('ESOF', o['NEEB'] >= o['NEEB'][pick['ESOF']] - TOL_DOLLAR)]:
                for nut, arr in [('N', n), ('P2O5', p), ('K2O', k)]:
                    b[f'{name}_{nut}_at_max'] = int(arr[pick[name]] >= arr.max())
                    b[f'{name}_{nut}_at_zero'] = int(arr[pick[name]] == 0)
                    b[f'{name}_{nut}_span'] = float(arr[band].max() - arr[band].min())
            bnd.append(b)
        if (b0 // args.batch) % 50 == 0:
            log(f'  {min(b0 + args.batch, len(ids)):,}/{len(ids):,} sites, {time.time() - t0:.0f}s')

    rec = pd.DataFrame(rows).sort_values('site').reset_index(drop=True); rec.to_csv(os.path.join(args.out, 'site_recommendations.csv'), index=False)
    bd = pd.DataFrame(bnd).sort_values('site').reset_index(drop=True); bd.to_csv(os.path.join(args.out, 'site_boundary_span.csv'), index=False)
    pd.DataFrame([dict(objective=o_, nutrient=nu, pct_at_max=100 * bd[f'{o_}_{nu}_at_max'].mean(),
                       pct_at_zero=100 * bd[f'{o_}_{nu}_at_zero'].mean(), mean_span=bd[f'{o_}_{nu}_span'].mean())
                  for o_ in ['AOF', 'EOF', 'ESOF'] for nu in ['N', 'P2O5', 'K2O']]
                 ).to_csv(os.path.join(args.out, 'table_S15_boundary_span.csv'), index=False)
    s16 = pd.DataFrame([dict(scenario=sc, N=rec[f'{sc}_N'].mean(), P2O5=rec[f'{sc}_P2O5'].mean(), K2O=rec[f'{sc}_K2O'].mean(),
                             yield_kg_ha=rec[f'{sc}_yield'].mean(), NFB=rec[f'{sc}_NFB'].mean(), NEEB=rec[f'{sc}_NEEB'].mean())
                        for sc in ['FP', 'MED', 'AOF', 'AOF95', 'EOF', 'ESOF'] if f'{sc}_N' in rec])
    s16.to_csv(os.path.join(args.out, 'table_S16_means.csv'), index=False)
    json.dump(dict(seyp=args.seyp, features=FEATURES, fp_baseline=args.fp_baseline if has_fp else None, step=args.step,
                   sites=int(len(rec)), aof95_frac=AOF95_FRAC, tol_yield=TOL_YIELD, tol_dollar=TOL_DOLLAR,
                   prices=dict(grain=PRICE_GRAIN, N=PRICE_N, P2O5=PRICE_P, K2O=PRICE_K), runtime_min=(time.time() - t0) / 60),
              open(os.path.join(args.out, 'run_config.json'), 'w'), indent=1)
    log('\n' + s16.round(1).to_string(index=False))
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {args.out}')


if __name__ == '__main__':
    main()
