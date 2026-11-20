#!/usr/bin/env python3
import os
import pandas as pd

SURVEY_XLSX = 'data/统计数据汇总.xlsx'
SITE_FILE = 'data/data_2025_8_6_range.csv'
SITE_WORKBOOK = 'data/试验点尺度推荐施肥数据2025_8_13.xlsx'
OUT = 'results/FP_survey'
YEARS = (2005, 2015)
MU_TO_HA = 15

POSTAL = {'01': 'Inner Mongolia', '02': 'Inner Mongolia', '03': 'Shanxi', '04': 'Shanxi', '05': 'Hebei', '06': 'Hebei',
          '07': 'Hebei', '10': 'Beijing', '30': 'Tianjin', '11': 'Liaoning', '12': 'Liaoning', '13': 'Jilin',
          '15': 'Heilongjiang', '16': 'Heilongjiang', '20': 'Shanghai', '21': 'Jiangsu', '22': 'Jiangsu', '23': 'Anhui',
          '24': 'Anhui', '25': 'Shandong', '26': 'Shandong', '27': 'Shandong', '31': 'Zhejiang', '32': 'Zhejiang',
          '33': 'Jiangxi', '34': 'Jiangxi', '35': 'Fujian', '36': 'Fujian', '40': 'Chongqing', '41': 'Hunan', '42': 'Hunan',
          '43': 'Hubei', '44': 'Hubei', '45': 'Henan', '46': 'Henan', '47': 'Henan', '51': 'Guangdong', '52': 'Guangdong',
          '53': 'Guangxi', '54': 'Guangxi', '55': 'Guizhou', '56': 'Guizhou', '57': 'Hainan', '61': 'Sichuan', '62': 'Sichuan',
          '63': 'Sichuan', '64': 'Sichuan', '65': 'Yunnan', '66': 'Yunnan', '67': 'Yunnan', '71': 'Shaanxi', '72': 'Shaanxi',
          '73': 'Gansu', '74': 'Gansu', '75': 'Ningxia', '81': 'Qinghai', '83': 'Xinjiang', '84': 'Xinjiang', '85': 'Tibet',
          '86': 'Tibet'}


def survey_long(path):
    """Reshape the compiled survey workbook to a long wheat table in kg/ha."""
    raw = pd.read_excel(path, header=None)
    h0, h1 = raw.iloc[0].tolist(), raw.iloc[1].tolist()
    prov, cur = [], None
    for i, v in enumerate(h0):
        if i < 2:
            prov.append(None); continue
        if isinstance(v, str) and v.strip():
            cur = v.strip()
        prov.append(cur)
    cols = [('year', ''), ('crop', '')] + [(prov[i], str(h1[i]).strip()) for i in range(2, len(h0))]
    df = raw.iloc[2:].copy(); df.columns = pd.MultiIndex.from_tuples(cols)
    long = df.set_index([('year', ''), ('crop', '')]).stack(level=0, future_stack=True).reset_index()
    long.columns = ['year', 'crop', 'province_cn'] + list(long.columns[3:])
    long['year'] = pd.to_numeric(long['year'], errors='coerce')
    w = long[long['crop'].astype(str).str.contains('小麦')].copy()
    w = w.rename(columns={'单产(公斤/每亩)': 'yield', 'N(公斤/亩)': 'N', 'P2O5(公斤/亩)': 'P2O5', 'K2O(公斤/亩)': 'K2O'})
    for c in ['yield', 'N', 'P2O5', 'K2O']:
        w[c] = pd.to_numeric(w[c], errors='coerce') * MU_TO_HA
    w['province'] = w['province_cn'].str.replace(r'^[\u4e00-\u9fff]+', '', regex=True).str.strip().replace(
        {'Neimenggu': 'Inner Mongolia', 'Xizang': 'Tibet'})
    w.loc[w['province_cn'].str.startswith('全国'), 'province'] = 'China'
    return w[['province', 'province_cn', 'year', 'N', 'P2O5', 'K2O', 'yield']].dropna(subset=['N']).sort_values(['province', 'year'])


def survey_by_site(survey, site_file):
    """Survey rates and farm yield for every site, by province and trial year with fallbacks."""
    site = pd.read_csv(site_file)
    site['province'] = site['ID'].astype(str).str[:2].map(POSTAL).fillna('Other')
    site['year'] = pd.to_numeric(site['ID'].astype(str).str[7:11], errors='coerce')
    s = survey[(survey.year >= YEARS[0]) & (survey.year <= YEARS[1])]
    py = s.set_index(['province', 'year'])[['N', 'P2O5', 'K2O', 'yield']]
    pm = s[s.province != 'China'].groupby('province')[['N', 'P2O5', 'K2O', 'yield']].mean()
    nat = s[s.province == 'China'].set_index('year')[['N', 'P2O5', 'K2O', 'yield']]
    rows = []
    for _, r in site.iterrows():
        key = (r.province, r.year); src = 'province-year'
        if key in py.index:
            v = py.loc[key]
        elif r.province in pm.index:
            v = pm.loc[r.province]; src = 'province-mean'
        elif r.year in nat.index:
            v = nat.loc[r.year]; src = 'national-year'
        else:
            v = nat.mean(); src = 'national-mean'
        rows.append({'No.': r['No.'], 'province': r.province, 'year': r.year, 'source': src, 'FPs_N': v.N,
                     'FPs_P2O5': v.P2O5, 'FPs_K2O': v.K2O, 'FPs_yield': v['yield']})
    return pd.DataFrame(rows)


def merge_baselines(fp, workbook):
    """Join the survey baseline with the gridded baseline (A, B, C) and the medium rates of the site workbook."""
    m = pd.read_excel(workbook)
    m = m.merge(fp[['No.', 'province', 'year', 'source', 'FPs_N', 'FPs_P2O5', 'FPs_K2O', 'FPs_yield']], on='No.')
    base = m[['No.', 'ID', 'Region', 'Province', 'County code', 'province', 'year', 'source', 'A', 'B', 'C',
              'FPs_N', 'FPs_P2O5', 'FPs_K2O', 'FPs_yield', 'Nf', 'Pf', 'Kf']].rename(
        columns={'A': 'FPn_N', 'B': 'FPn_P2O5', 'C': 'FPn_K2O', 'Nf': 'N_med', 'Pf': 'P_med', 'Kf': 'K_med'})
    comp = base.groupby('province').agg(sites=('No.', 'size'), N_gridded=('FPn_N', 'mean'), N_survey=('FPs_N', 'mean'),
                                        P2O5_gridded=('FPn_P2O5', 'mean'), P2O5_survey=('FPs_P2O5', 'mean'),
                                        K2O_gridded=('FPn_K2O', 'mean'), K2O_survey=('FPs_K2O', 'mean'),
                                        yield_survey=('FPs_yield', 'mean')).sort_values('sites', ascending=False)
    for x in ['N', 'P2O5', 'K2O']:
        comp[f'ratio_{x}_gridded_to_survey'] = comp[f'{x}_gridded'] / comp[f'{x}_survey']
    return base, comp


def main():
    os.makedirs(OUT, exist_ok=True)
    survey = survey_long(SURVEY_XLSX)
    survey.to_csv(f'{OUT}/FP_survey_wheat_province_year.csv', index=False)
    fp = survey_by_site(survey, SITE_FILE)
    base, comp = merge_baselines(fp, SITE_WORKBOOK)
    base.to_csv(f'{OUT}/FP_survey_baseline_by_site.csv', index=False)
    comp.round(3).to_csv(f'{OUT}/FP_province_gridded_vs_survey.csv')
    n = len(base)
    print('matching:', ', '.join(f'{k} {v} ({100 * v / n:.1f}%)' for k, v in base['source'].value_counts().items()))
    print('site-weighted survey FP, N/P2O5/K2O/yield:', base[['FPs_N', 'FPs_P2O5', 'FPs_K2O', 'FPs_yield']].mean().round(0).to_dict())
    print('site-weighted gridded FP, N/P2O5/K2O:', base[['FPn_N', 'FPn_P2O5', 'FPn_K2O']].mean().round(0).to_dict())
    print(f'written to {OUT}: FP_survey_wheat_province_year.csv, FP_survey_baseline_by_site.csv, FP_province_gridded_vs_survey.csv')


if __name__ == '__main__':
    main()
