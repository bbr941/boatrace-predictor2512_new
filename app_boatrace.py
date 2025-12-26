import os
# Force single thread to prevent Streamlit Cloud crashes (OpenMP)
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import datetime
import re
import time
import sys
import itertools

# --- Configuration ---
st.set_page_config(page_title="BoatRace AI Dual System", layout="wide")

if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache Cleared!")

MODEL_HONMEI_PATH = 'model_honmei.txt'
MODEL_ANA_PATH = 'model_ana.txt'
DATA_DIR = 'app_data'
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- 1. Scraper Class ---
class BoatRaceScraper:
    @staticmethod
    def get_soup(url):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
                return BeautifulSoup(resp.text, 'html.parser')
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Data Fetch Error: {e}")
                    return None
                time.sleep(1)
        return None

    @staticmethod
    def parse_float(text):
        try:
            return float(re.search(r'([\d\.]+)', text).group(1))
        except:
            return 0.0

    @staticmethod
    def get_odds(date_str, venue_code, race_no):
        jcd = f"{int(venue_code):02d}"
        url = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={race_no}&jcd={jcd}&hd={date_str}"
        soup = BoatRaceScraper.get_soup(url)
        odds_map = {}
        if soup:
            try:
                tables = soup.select("table.is-w495")
                target_table = None
                for t in tables:
                     if "単勝" in t.get_text():
                         target_table = t
                         break
                
                if target_table:
                     rows = target_table.select("tbody tr")
                     for row in rows:
                         tds = row.select("td")
                         if len(tds) >= 3:
                             try:
                                 bn_txt = tds[0].get_text(strip=True)
                                 bn = int(bn_txt)
                                 val_txt = tds[2].get_text(strip=True)
                                 val = float(val_txt)
                                 if val > 0:
                                     odds_map[bn] = 1.0 / val
                             except: pass
            except: pass
        return odds_map

    @staticmethod
    def get_race_data(date_str, venue_code, race_no):
        jcd = f"{int(venue_code):02d}"
        url_before = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_no}&jcd={jcd}&hd={date_str}"
        url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={race_no}&jcd={jcd}&hd={date_str}"
        
        soup_before = BoatRaceScraper.get_soup(url_before)
        soup_list = BoatRaceScraper.get_soup(url_list)
        
        odds_map = BoatRaceScraper.get_odds(date_str, venue_code, race_no)
        
        if not soup_before or not soup_list:
            return None
            
        weather = {'wind_direction': 0, 'wind_speed': 0.0, 'wave_height': 0.0}
        try:
            w = soup_before.select_one("div.weather1_body")
            if w:
                ws = w.select_one(".is-wind span.weather1_bodyUnitLabelData")
                if ws: weather['wind_speed'] = BoatRaceScraper.parse_float(ws.text)
                wh = w.select_one(".is-wave span.weather1_bodyUnitLabelData")
                if wh: weather['wave_height'] = BoatRaceScraper.parse_float(wh.text)
                wd = w.select_one(".is-windDirection p")
                if wd:
                    cls = wd.get('class', [])
                    d = next((c for c in cls if c.startswith('is-wind') and c != 'is-windDirection'), None)
                    if d: weather['wind_direction'] = int(re.sub(r'\D', '', d))
        except: pass

        boat_before = {}
        try:
            for i, tb in enumerate(soup_before.select("table.is-w748 tbody")):
                tds = tb.select("td")
                if len(tds) >= 5:
                    boat_before[i+1] = {'ex_time': BoatRaceScraper.parse_float(tds[4].text), 'st': 0.20}
            
            for idx, row in enumerate(soup_before.select("table.is-w238 tbody tr")):
                bn_span = row.select_one("span.table1_boatImage1Number")
                if bn_span:
                    b = int(bn_span.text.strip())
                    pred_c = idx + 1
                    st_span = row.select_one("span.table1_boatImage1Time")
                    val = 0.20
                    if st_span:
                        txt_raw = st_span.text.strip()
                        if 'L' in txt_raw: val = 1.0
                        elif 'F' in txt_raw:
                            try:
                                sub = txt_raw.replace('F', '')
                                val = -float(sub)
                            except: val = -0.05
                        else:
                            val = BoatRaceScraper.parse_float(txt_raw)
                    if b not in boat_before: boat_before[b] = {'ex_time': 6.8}
                    boat_before[b]['st'] = val
                    boat_before[b]['pred_course'] = pred_c
        except: pass

        rows = []
        try:
            for i, tb in enumerate(soup_list.select("tbody.is-fs12")):
                bn = i + 1
                if bn > 6: break
                
                racer_id = 9999
                try: 
                    txt = tb.select("td")[2].select_one("div").get_text()
                    racer_id = int(re.search(r'(\d{4})', txt).group(1))
                except: pass

                branch = 'Unknown'
                weight = 52.0
                try:
                    td2 = tb.select("td")[2]
                    txt_full = td2.get_text(" ", strip=True)
                    match_w = re.search(r'(\d{2}\.\d)kg', txt_full)
                    if match_w: weight = float(match_w.group(1))
                    
                    prefectures = r"(群馬|埼玉|東京|福井|静岡|愛知|三重|滋賀|大阪|兵庫|徳島|香川|岡山|広島|山口|福岡|佐賀|長崎)"
                    m = re.search(prefectures, txt_full)
                    if m: branch = m.group(1)
                except: pass

                nat_win_rate = 0.0
                local_win_rate = 0.0
                try:
                    col3_txt = tb.select("td")[3].get_text(" ", strip=True)
                    clean_txt = re.sub(r'[FLK]\d+', '', col3_txt) 
                    nums = re.findall(r'(\d+(?:\.\d+)?)', clean_txt)
                    if len(nums) >= 5:
                        nat_win_rate = float(nums[1])
                        local_win_rate = float(nums[3])
                    elif len(nums) >= 4:
                        nat_win_rate = float(nums[0])
                        local_win_rate = float(nums[2])
                except: pass

                prior_results = ""
                try:
                    rank_row = tb.select_one("tr.is-fBold")
                    if rank_row:
                        res_texts = [td.get_text(strip=True) for td in rank_row.select("td")]
                        cleaned_res = []
                        for t in res_texts:
                            if not t: continue
                            t_norm = t.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
                            if re.match(r'^[1-6FLKS欠失転不]$', t_norm):
                                cleaned_res.append(t_norm)
                        prior_results = " ".join(cleaned_res)
                except: pass

                tds = tb.select("td")
                motor = 30.0
                try:
                    txt = tds[6].get_text(" ", strip=True).replace('%', '')
                    parts = txt.split()
                    if len(parts) >= 2: motor = float(parts[1])
                    else: motor = float(parts[0])
                except: pass
                
                boat = 30.0
                try:
                    txt = tds[7].get_text(" ", strip=True).replace('%', '')
                    parts = txt.split()
                    if len(parts) >= 2: boat = float(parts[1])
                    else: boat = float(parts[0])
                except: pass
                
                row = {
                    'race_id': f"{date_str}_{venue_code}_{race_no}",
                    'boat_number': bn,
                    'racer_id': racer_id,
                    'motor_rate': motor,
                    'boat_rate': boat,
                    'exhibition_time': boat_before.get(bn, {}).get('ex_time', 6.8),
                    'exhibition_start_timing': boat_before.get(bn, {}).get('st', 0.20),
                    'pred_course': boat_before.get(bn, {}).get('pred_course', bn),
                    'wind_direction': weather['wind_direction'],
                    'wind_speed': weather['wind_speed'],
                    'wave_height': weather['wave_height'],
                    'prior_results': prior_results,
                    'branch': branch,
                    'weight': weight,
                    'nat_win_rate': nat_win_rate,
                    'local_win_rate': local_win_rate,
                    'syn_win_rate': odds_map.get(bn, 0.0)
                }
                rows.append(row)
        except Exception as e:
            st.error(f"List Parse Error: {e}")
            return None
            
        return pd.DataFrame(rows)

# --- 2. Feature Engineer ---
class FeatureEngineer:
    @staticmethod
    def process(df, venue_name, debug_mode=False):
        df['venue_name'] = venue_name
        
        try:
            r_course = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_course.csv'))
            r_venue = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_venue.csv'))
            v_course = pd.read_csv(os.path.join(DATA_DIR, 'static_venue_course.csv'))
            r_params = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_params.csv'))
            
            df['racer_id'] = df['racer_id'].astype(int)
            df['pred_course'] = df['pred_course'].astype(int)
            r_course['RacerID'] = r_course['RacerID'].astype(int)
            r_course['Course'] = r_course['Course'].astype(int)
            r_venue['RacerID'] = r_venue['RacerID'].astype(int)
            v_course['course_number'] = v_course['course_number'].astype(int)
            r_params['racer_id'] = r_params['racer_id'].astype(int)

            df = df.merge(r_course, left_on=['racer_id', 'pred_course'], right_on=['RacerID', 'Course'], how='left')
            df.rename(columns={
                'RacesRun': 'course_run_count',
                'QuinellaRate': 'course_quinella_rate',
                'TrifectaRate': 'course_trifecta_rate',
                'FirstPlaceRate': 'course_1st_rate',
                'AvgStartTiming': 'course_avg_st',
                'Nige': 'nige_count', 
                'Makuri': 'makuri_count',
                'Sashi': 'sashi_count'
            }, inplace=True)

            venue_map_rev = {
                '桐生': 1, '戸田': 2, '江戸川': 3, '平和島': 4, '多摩川': 5,
                '浜名湖': 6, '蒲郡': 7, '常滑': 8, '津': 9, '三国': 10,
                'びわこ': 11, '住之江': 12, '尼崎': 13, '鳴門': 14, '丸亀': 15,
                '児島': 16, '宮島': 17, '徳山': 18, '下関': 19, '若松': 20,
                '芦屋': 21, '福岡': 22, '唐津': 23, '大村': 24
            }
            df['venue_code_int'] = df['venue_name'].map(venue_map_rev).fillna(0).astype(int)
            r_venue['Venue'] = pd.to_numeric(r_venue['Venue'], errors='coerce').fillna(0).astype(int)
            
            df = df.merge(r_venue, left_on=['racer_id', 'venue_code_int'], right_on=['RacerID', 'Venue'], how='left')
            
            if 'local_win_rate' in df.columns:
                 df['local_win_rate'] = df['local_win_rate'].replace(0.0, np.nan)
                 if 'WinRate' in df.columns:
                     df['local_win_rate'] = df['local_win_rate'].fillna(df['WinRate'])
            elif 'WinRate' in df.columns:
                 df['local_win_rate'] = df['WinRate']

            df = df.merge(v_course, left_on=['venue_name', 'pred_course'], right_on=['venue_name', 'course_number'], how='left')
            df.rename(columns={'rate_1st': 'venue_course_1st_rate', 'rate_2nd': 'venue_course_2nd_rate', 'rate_3rd': 'venue_course_3rd_rate'}, inplace=True)

            df = df.merge(r_params, on='racer_id', how='left')
            
        except Exception: pass
        
        required_cols = ['makuri_count', 'nige_count', 'sashi_count', 'nat_win_rate', 'course_run_count', 'local_win_rate']
        for c in required_cols:
            if c not in df.columns: df[c] = 0.0
            
        # Features
        def parse_prior(x):
            if isinstance(x, (int, float)): return float(x)
            if not isinstance(x, str): return 3.5
            try:
                x_c = re.sub(r'[欠失FLS]', '', x)
                parts = x_c.split()
                ranks = [float(p) for p in parts if p.isdigit()]
                if ranks: return sum(ranks)/len(ranks)
            except: pass
            return 3.5
            
        df['series_avg_rank'] = df['prior_results'].apply(parse_prior)
        df['makuri_rate'] = df['makuri_count'] / df['course_run_count'].replace(0, 1)
        df['nige_rate'] = df['nige_count'] / df['course_run_count'].replace(0, 1)

        df = df.sort_values('pred_course')
        df['inner_st'] = df['exhibition_start_timing'].shift(1).fillna(0)
        df['inner_st_gap'] = df['exhibition_start_timing'] - df['inner_st']
        df['outer_st'] = df['exhibition_start_timing'].shift(-1).fillna(0)
        avg_neighbor = (df['inner_st'] + df['outer_st']) / 2
        df['slit_formation'] = df['exhibition_start_timing'] - avg_neighbor

        c1_nige = df.loc[df['pred_course']==1, 'nige_rate']
        val = c1_nige.values[0] if len(c1_nige) > 0 else 0.5
        df['anti_nige_potential'] = df['makuri_rate'] * (1 - val)
        
        df['wall_strength'] = df['course_quinella_rate'].shift(1).fillna(0)
        df['follow_potential'] = df['makuri_rate'].shift(1).fillna(0) * df['course_quinella_rate']
        
        mean_t = df['exhibition_time'].mean()
        std_t = df['exhibition_time'].std()
        if std_t == 0: std_t = 1
        df['tenji_z_score'] = (mean_t - df['exhibition_time']) / std_t
        df['linear_rank'] = df['exhibition_time'].rank(method='min', ascending=True)
        df['is_linear_leader'] = (df['linear_rank'] == 1).astype(int)
        
        if 'weight_x' in df.columns: df['weight'] = df['weight_x']
        if 'weight' not in df.columns: df['weight'] = 52.0
        df['weight_diff'] = df['weight'] - df['weight'].mean()
        df['high_wind_alert'] = (df['wind_speed'] >= 5).astype(int)
        
        df['nat_win_rate'] = pd.to_numeric(df['nat_win_rate'], errors='coerce').fillna(0.0)
        df['local_win_rate'] = pd.to_numeric(df['local_win_rate'], errors='coerce').fillna(0.0)
        df['local_perf_diff'] = df['local_win_rate'] - df['nat_win_rate']

        # Wind Vector
        def wind_deg_from_int(x): return (x - 1) * 22.5 if 1 <= x <= 16 else 0
        df['wind_angle_deg'] = df['wind_direction'].apply(wind_deg_from_int)
        venue_tailwind_from = {
            '桐生': 135, '戸田': 90, '江戸川': 180, '平和島': 180, '多摩川': 270,
            '浜名湖': 180, '蒲郡': 270, '常滑': 270, '津': 135, '三国': 180,
            'びわこ': 225, '住之江': 270, '尼崎': 90, '鳴門': 135, '丸亀': 15,
            '児島': 225, '宮島': 270, '徳山': 135, '下関': 270, '若松': 270,
            '芦屋': 135, '福岡': 0, '唐津': 135, '大村': 315
        }
        df['venue_tailwind_deg'] = df['venue_name'].map(venue_tailwind_from).fillna(0)
        angle_diff_rad = np.radians(df['wind_angle_deg'] - df['venue_tailwind_deg'])
        df['wind_vector_long'] = df['wind_speed'] * np.cos(angle_diff_rad)
        df['wind_vector_lat'] = df['wind_speed'] * np.sin(angle_diff_rad)

        if 'race_date' not in df.columns: df['race_date'] = '20000101'
        
        # Win Direction Mapping (Int -> String to match Training Data)
        wind_map = {
            1: '北', 2: '北北東', 3: '北東', 4: '東北東', 5: '東', 6: '東南東', 7: '南東', 8: '南南東',
            9: '南', 10: '南南西', 11: '南西', 12: '西南西', 13: '西', 14: '西北西', 15: '北西', 16: '北北西'
        }
        # Only map if numeric
        if pd.api.types.is_numeric_dtype(df['wind_direction']):
             df['wind_direction'] = df['wind_direction'].map(wind_map).fillna(df['wind_direction'])
             # If mapping leaves numbers (e.g. 0), coerce to string or handle?
             # Train data likely has "南" etc. 0 might be problematic if not in train categories.
             # Convert to string to ensure it becomes category later.
             df['wind_direction'] = df['wind_direction'].astype(str)
             # Handle 'nan' string if any
             df['wind_direction'] = df['wind_direction'].replace('nan', '')

        # Categorical Conversion (Must match train_model.py logic)
        # First, try to convert everything to numeric (like pd.read_csv does)
        # Use errors='coerce' to force non-parseable strings to NaN (float), 
        # preventing them from staying as Object and becoming Category.
        for col in df.columns:
            if col not in ['race_id', 'race_date', 'venue_name', 'prior_results', 'wind_direction', 'branch']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Then convert remaining objects to category
        # train_model.py ignores: ['race_id', 'race_date', 'prior_results']
        ignore_cols = ['race_id', 'race_date', 'prior_results', 'pred_score', 'weight_for_loss', 'relevance', 'rank']
        
        for col in df.columns:
            if col in ignore_cols: continue
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')

        return df

    @staticmethod
    def get_features_subset(df, mode='honmei'):
        base_ignore = [
            'race_id', 'boat_number', 'racer_id', 'rank', 'relevance',
            'race_date', 'venue_name', 'prior_results', 'weight_for_loss', 'pred_score', 'score'
        ]
        odds_features = ['syn_win_rate', 'odds', 'prediction_odds', 'popularity', 'vote_count', 'win_share']
        
        all_cols = df.columns.tolist()
        candidates = [c for c in all_cols if c not in base_ignore]
        
        if mode == 'ana':
            final_feats = []
            for c in candidates:
                is_odds = False
                for o in odds_features:
                    if o in c:
                        is_odds = True
                        break
                if not is_odds: final_feats.append(c)
            return final_feats
        else:
            return candidates

def format_trifecta_box(boats):
    return f"{boats[0]}, {boats[1]}, {boats[2]}"

def calculate_trifecta_scores(scores, boats):
    import itertools
    combos = list(itertools.permutations(boats, 3))
    c_list = []
    for c in combos:
        s = (scores[c[0]] * 4) + (scores[c[1]] * 2) + (scores[c[2]] * 1)
        c_list.append({'combo': f"{c[0]}-{c[1]}-{c[2]}", 'val': s})
    return pd.DataFrame(c_list).sort_values('val', ascending=False)

# --- 3. Main App ---
st.title("🚤 BoatRace AI Dual Strategy System")
st.markdown("Returns specific predictions using two specialized models.")

today = datetime.date.today()
target_date = st.sidebar.date_input("Date", today)
venue_map = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村'
}
venue_code = st.sidebar.selectbox("Venue", list(venue_map.keys()), format_func=lambda x: f"{x:02d}: {venue_map[x]}")
venue_name = venue_map[venue_code]
race_no = st.sidebar.selectbox("Race No", range(1, 13))

debug_mode = st.sidebar.checkbox("Show Debug Info", value=False)

if st.button("Analyze Race", type="primary"):
    st.session_state['run_analysis'] = True
    st.session_state['target_props'] = {
        'date': target_date.strftime('%Y%m%d'),
        'venue': venue_code,
        'race': race_no,
        'v_name': venue_name
    }

if st.session_state.get('run_analysis'):
    props = st.session_state['target_props']
    st.info(f"Analyzing: {props['v_name']} {props['race']}R ({props['date']})")
    
    with st.spinner("Scraping Data..."):
        df_race = BoatRaceScraper.get_race_data(props['date'], props['venue'], props['race'])

    if df_race is not None:
        st.subheader("Race Data")
        st.dataframe(df_race[['boat_number', 'racer_id', 'motor_rate', 'exhibition_time', 'exhibition_start_timing', 'syn_win_rate']])
        
        with st.spinner("Engineering Features..."):
            df_feat = FeatureEngineer.process(df_race, props['v_name'], debug_mode=debug_mode)
        
        # --- Dual Prediction ---
        col1, col2 = st.columns(2)
        
        # Model A: Honmei
        with col1:
            st.markdown("### 🛡️ 本命AI (Accuracy Rules)")
            if os.path.exists(MODEL_HONMEI_PATH):
                try:
                    model_h = lgb.Booster(model_file=MODEL_HONMEI_PATH)
                    feats_h = FeatureEngineer.get_features_subset(df_feat, 'honmei')
                    preds_h = model_h.predict(df_feat[feats_h])
                    df_feat['score_honmei'] = preds_h
                    
                    # Top 5
                    scores_h = dict(zip(df_feat['boat_number'], df_feat['score_honmei']))
                    sorted_boats_h = df_feat.sort_values('score_honmei', ascending=False)['boat_number'].tolist()
                    
                    df_c_h = calculate_trifecta_scores(scores_h, sorted_boats_h)
                    
                    st.success(f"Best: **{df_c_h.iloc[0]['combo']}**")
                    st.markdown("#### Top Recommendations")
                    for i in range(5):
                        st.write(f"{i+1}. {df_c_h.iloc[i]['combo']}")
                        
                    top_h = df_c_h.iloc[0]['combo']
                except Exception as e:
                    st.error(f"Honmei Error: {e}")
                    top_h = None
            else:
                st.warning("Model Honmei not found.")
                top_h = None

        # Model B: Ana
        with col2:
            st.markdown("### 💰 穴AI (High Dividend)")
            if os.path.exists(MODEL_ANA_PATH):
                try:
                    model_a = lgb.Booster(model_file=MODEL_ANA_PATH)
                    feats_a = FeatureEngineer.get_features_subset(df_feat, 'ana')
                    # Ensure prediction works even if feature set differs from train if using Booster?
                    # LGBM Booster check feature names? Yes.
                    # We assume get_features_subset returns same consistent set if df columns are same.
                    # Warning: If `df_feat` has different columns than train, might fail.
                    # We rely on FeatureEngineer.process adding all empty '0' cols.
                    
                    # Also need to ensure column ORDER match? 
                    # Booster.predict(dataframe) usually maps by name.
                    preds_a = model_a.predict(df_feat[feats_a])
                    df_feat['score_ana'] = preds_a
                    
                    scores_a = dict(zip(df_feat['boat_number'], df_feat['score_ana']))
                    sorted_boats_a = df_feat.sort_values('score_ana', ascending=False)['boat_number'].tolist()
                    
                    df_c_a = calculate_trifecta_scores(scores_a, sorted_boats_a)
                    
                    st.success(f"Best: **{df_c_a.iloc[0]['combo']}**")
                    st.markdown("#### Top Recommendations")
                    for i in range(5):
                        st.write(f"{i+1}. {df_c_a.iloc[i]['combo']}")
                        
                    top_a = df_c_a.iloc[0]['combo']

                except Exception as e:
                    st.error(f"Ana Error: {e}")
                    top_a = None
            else:
                st.warning("Model Ana not found.")
                top_a = None

        # Synthesis
        st.divider()
        if top_h and top_a:
            if top_h == top_a:
                 st.markdown(f"## 🔥 激アツ鉄板: {top_h}")
                 st.info("本命・穴モデル共に推奨！自信度MAX")
                 
                 # Data Issue Check
                 if df_feat['score_honmei'].equals(df_feat['score_ana']):
                     st.warning("⚠️ 注意: 両モデルの予測スコアが完全に一致しています。学習データにオッズ情報が含まれていない可能性があります。")
                     
            else:
                 st.info("本命と穴で意見が割れました。オッズと相談して選択してください。")
