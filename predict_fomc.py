"""
FOMC NLP Tail Volatility Forecasting Model: XAU/USD (Gold)
Quantitative NLP framework predicting extreme volatility risks (>= 400 pips)
from Federal Reserve press conference transcripts and Chair speeches.
"""

import os
import re
import sys
import datetime
import pypdf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# Configure Windows console encoding for Unicode support
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Enable ANSI escape code processing in Windows Terminal / CMD
os.system('')

# ANSI Color Palette
GOLD = '\033[38;5;220m'
YELLOW = '\033[1;33m'
GREEN = '\033[1;32m'
CYAN = '\033[1;36m'
BLUE = '\033[1;34m'
MAGENTA = '\033[1;35m'
RED = '\033[1;31m'
GRAY = '\033[90m'
WHITE = '\033[1;37m'
RESET = '\033[0m'
BOLD = '\033[1m'


def strip_ansi(s):
    return re.sub(r'\033\[[0-9;]*m', '', s)


def box_row(content, width=58):
    vis = len(strip_ansi(content))
    pad = max(0, width - vis)
    return f"{GREEN}│{RESET}{content}{' ' * pad}{GREEN}│{RESET}"


def print_fomc_header(dataset_count=45):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ASCII Art Logo (XAU/USD Gold Ingot)
    logo = [
        r"          .______________________.          ",
        r"         /                      /|          ",
        r"        /      X A U / U S D   / |          ",
        r"       /      G O L D  999.9  /  |          ",
        r"      /______________________/   |          ",
        r"      |                      |   |          ",
        r"      |   .--------------.   |   |          ",
        r"      |   |  FINE  GOLD  |   |   |          ",
        r"      |   |   1000 g     |   |   /          ",
        r"      |   '--------------'   |  /           ",
        r"      |     NET LIQUIDITY    | /            ",
        r"      |______________________|/             ",
        r"             [ XAUUSD ]                     ",
        r"         Institutional Desk                 ",
        r"                                            ",
        r"                                            "
    ]

    row_cat  = f" {CYAN}CATALYST  {RESET}  FOMC PRESS CONFERENCE (THE FEDERAL RESERVE)"
    row_mod  = f" {CYAN}MODEL     {RESET}  N-GRAM TF-IDF & COSINE SIMILARITY NLP ENGINE"
    row_ds   = f" {CYAN}DATABASE  {RESET}  {dataset_count} HISTORICAL MEETINGS (2020 - 2026)"
    row_eng  = f" {CYAN}STATUS    {RESET}  ONLINE // AUTOMATIC SPEAKER DIARIZATION"
    row_th   = f" {CYAN}THRESHOLD {RESET}  EXTREME TAIL VOLATILITY >= 400 PIPS ($40.0 USD)"
    row_time = f" {CYAN}TIME      {RESET}  {now_str} WIB"

    right_panel = [
        f"{BOLD}{WHITE}Hey, Trader{RESET}",
        f"{GREEN}┌── FOMC Press Conference NLP Engine ──────────────────────┐{RESET}",
        box_row(row_cat),
        box_row(row_mod),
        box_row(row_ds),
        f"{GREEN}├── NLP Execution Protocol ─────────────────────────────────┤{RESET}",
        box_row(row_eng),
        box_row(row_th),
        box_row(row_time),
        f"{GREEN}└───────────────────────────────────────────────────────────┘{RESET}",
        f"{WHITE}Quantitative Macro NLP Forecaster Online.{RESET}",
        f"{RED}●{RESET} {GREEN}●{RESET} {YELLOW}●{RESET} {BLUE}●{RESET} {MAGENTA}●{RESET} {CYAN}●{RESET} {WHITE}●{RESET}",
        ""
    ]

    print("\n")
    for l, r in zip(logo, right_panel):
        print(f"{YELLOW}{l}{RESET}  {r}")
    print("")


def extract_chair_speech_from_pdf(pdf_path):
    """
    Extract Fed Chair dialogue from PDF (detecting Powell, Warsh, or Yellen).
    Handles patterns like: 'CHAIRMAN WARSH.', 'CHAIR POWELL.', 'CHAIR ...'
    """
    reader = pypdf.PdfReader(pdf_path)
    full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    detected_chair = "CHAIR"
    if "WARSH" in full_text[:1500].upper():
        detected_chair = "CHAIRMAN KEVIN WARSH"
    elif "POWELL" in full_text[:1500].upper():
        detected_chair = "CHAIR JEROME POWELL"

    lines = full_text.split('\n')
    chair_dialogues = []
    is_chair_speaking = False

    for line in lines:
        line_clean = line.strip()
        if re.match(r'^(CHAIR|CHAIRMAN)\s+[A-Z\s]+\.', line_clean):
            is_chair_speaking = True
            text_part = re.sub(r'^(CHAIR|CHAIRMAN)\s+[A-Z\s]+\.\s*', '', line_clean)
            if text_part:
                chair_dialogues.append(text_part)
        elif re.match(r'^[A-Z\s]{3,35}\.', line_clean):
            is_chair_speaking = False
        else:
            if is_chair_speaking and len(line_clean) > 0:
                chair_dialogues.append(line_clean)

    extracted_text = " ".join(chair_dialogues).strip()
    if len(extracted_text.split()) < 50:
        speech_final = full_text
    else:
        speech_final = extracted_text

    return detected_chair, speech_final


def main():
    transcript_folder = "transcript"
    dataset_folder = "dataset"
    gold_cache_file = "gold_prices_cache.csv"

    if not os.path.exists(dataset_folder):
        print(f"[!] Dataset folder '{dataset_folder}' not found.")
        return
    if not os.path.exists(gold_cache_file):
        print(f"[!] Historical gold cache file '{gold_cache_file}' not found.")
        return
    if not os.path.exists(transcript_folder):
        print(f"[!] Transcript folder '{transcript_folder}' not found.")
        return

    pdf_files = [f for f in os.listdir(transcript_folder) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"[!] No PDF files found in '{transcript_folder}'.")
        print("[!] Please place the latest FOMC conference PDF in the 'transcript/' folder.")
        return

    # Select latest PDF by modification timestamp
    pdf_files.sort(key=lambda f: os.path.getmtime(os.path.join(transcript_folder, f)))
    latest_pdf = pdf_files[-1]
    latest_pdf_path = os.path.join(transcript_folder, latest_pdf)

    # 1. Historical Dataset Vectorization
    dataset_files = sorted([f for f in os.listdir(dataset_folder) if f.endswith('.txt') and 'FOMC' in f])
    print_fomc_header(dataset_count=len(dataset_files))

    print(f"[*] Processing latest transcript PDF : {WHITE}{latest_pdf}{RESET}")
    chair_name, speech_text = extract_chair_speech_from_pdf(latest_pdf_path)
    word_count = len(speech_text.split())

    print(f"[*] Detected Fed Chair Speaker       : {GOLD}{chair_name}{RESET}")
    print(f"[*] Extracted Dialogue Length        : {WHITE}{word_count:,} words{RESET}")
    print(f"[*] Aligning against {len(dataset_files)} historical FOMC meetings (2020 - 2026)...")

    transcripts = []
    for f in dataset_files:
        m = re.search(r'(\d{8})', f)
        if not m:
            continue
        with open(os.path.join(dataset_folder, f), 'r', encoding='utf-8', errors='ignore') as fp:
            raw = fp.read()
            dialogs = re.findall(r'<NAME>(.*?)</NAME>[\.\:\s]*(.*?)(?=<NAME>|$)', raw, re.DOTALL)
            p_text = " ".join([sp.strip() for name, sp in dialogs if 'POWELL' in name.upper() or 'WARSH' in name.upper()])
            if len(p_text) > 100:
                transcripts.append({'date_str': m.group(1), 'text': p_text})

    df_hist = pd.DataFrame(transcripts)

    gold = pd.read_csv(gold_cache_file, index_col=0, parse_dates=True)
    pips_list = []
    for _, row in df_hist.iterrows():
        d = pd.to_datetime(row['date_str'], format='%Y%m%d')
        sub = gold.loc[gold.index >= d].head(1)
        if len(sub) > 0:
            pips = round(float(sub.iloc[0]['High'] - sub.iloc[0]['Low']) * 10, 1)
        else:
            pips = 350.0
        pips_list.append(pips)

    df_hist['pips'] = pips_list
    THRESHOLD = 400.0
    df_hist['is_extreme'] = (df_hist['pips'] >= THRESHOLD).astype(int)

    # 2. Domain-Specific TF-IDF Feature Extraction
    base_stops = set(ENGLISH_STOP_WORDS)
    fomc_chatter = {'chair', 'chairman', 'warsh', 'powell', 'thank', 'thanks', 'good', 'afternoon', 'morning', 'today', 'question', 'think'}
    preserve = {'not', 'no', 'never', 'more', 'less', 'above', 'below', 'down', 'up', 'further', 'again', 'very'}
    stops = list((base_stops.union(fomc_chatter)) - preserve)

    vec = TfidfVectorizer(ngram_range=(1, 3), max_features=3000, stop_words=stops, min_df=2, sublinear_tf=True)
    hist_vecs = vec.fit_transform(df_hist['text'])

    # 3. Model Inference on Latest PDF Document
    new_vec = vec.transform([speech_text])
    sims = cosine_similarity(new_vec, hist_vecs)[0]

    top_k = 5
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_sims = sims[top_idx]
    top_labels = df_hist.iloc[top_idx]['is_extreme'].values
    top_pips = df_hist.iloc[top_idx]['pips'].values
    top_dates = df_hist.iloc[top_idx]['date_str'].values

    weights = top_sims / (np.sum(top_sims) + 1e-9)
    prob = float(np.sum(weights * top_labels) * 100)
    expected_pips = float(np.sum(weights * top_pips))
    expected_dollar = expected_pips / 10.0

    # 4. Display Results
    print("\n" + "=" * 82)
    print(f"       {BOLD}{GOLD}HASIL ANALISIS VOLATILITAS MAKRO NLP FOMC: {latest_pdf}{RESET}       ")
    print("=" * 82)

    if prob >= 53:
        status = f"{RED}[!] BAHAYA: POTENSI VOLATILITAS TINGGI (RENTANG >= 400 PIPS){RESET}"
        strategy = (
            f"{RED}Eksekusi STRADDLE PENDING BRACKET (Buy Stop & Sell Stop).{RESET}\n"
            f"    - Pasang pending order 20-30 pips dari spot.\n"
            f"    - Stop Loss: minimal 150-200 pips. JANGAN open posisi instan saat rilis!"
        )
    else:
        status = f"{GREEN}[OK] AMAN: VOLATILITAS NORMAL / SEDANG (< 400 PIPS){RESET}"
        strategy = (
            f"{GREEN}Pasar cenderung tenang / pergerakan terukur.{RESET}\n"
            f"    - Hindari mengejar lonjakan sesaat (fakeout).\n"
            f"    - Tunggu penutupan candle 15m untuk konfirmasi arah."
        )

    print(f"  Status Sinyal Model        : {status}")
    print(f"  Probabilitas Gerak Ekstrem : {BOLD}{prob:.1f}%{RESET} (Peluang pergerakan >= 400 pips / $40 USD)")
    print(f"  Estimasi Rentang Harga Emas: {BOLD}+/- {expected_pips:.0f} Pips{RESET} (+/- $ {expected_dollar:.1f} USD)")
    print("-" * 82)
    print(f"{BOLD}Protokol Rekomendasi Eksekusi MT4 / MT5:{RESET}")
    print(f"  {strategy}")
    print("-" * 82)
    print(f"{BOLD}Top-5 Rapat Historis dengan Karakter Linguistik Paling Mirip:{RESET}")
    for rank in range(top_k):
        tgl = pd.to_datetime(top_dates[rank], format='%Y%m%d').strftime('%Y-%m-%d')
        flag_color = RED if top_labels[rank] == 1 else GREEN
        flag = f"{flag_color}[EKSTREM >= 400p]{RESET}" if top_labels[rank] == 1 else f"{flag_color}[NORMAL < 400p]{RESET}"
        print(f"  {rank+1}. Rapat {tgl} | Cosine Sim: {top_sims[rank]:.4f} | Realisasi: {top_pips[rank]:>6.1f} pips | {flag}")
    print("=" * 82)


if __name__ == '__main__':
    main()
