# ekg_v23.py — Paramedik Hastane Oncesi EKG Asistani
# Amac: EKG kagidi veya monitor goruntusunu en guvenilir sekilde yorumlayip
#        Saglik Bakanligi ASHGM SB-ASH-Y protokollerine gore hastane oncesi
#        mudahale onerileri sunmak.
# ekg_ai_pro.py ayni klasorde olmali.

import os
import io
import re
import json
import base64
import logging
import time
import hashlib
import concurrent.futures
import sqlite3
import traceback
from datetime import datetime
from functools import partial
from typing import List, Dict, Any, Optional, Tuple

import requests
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from google import genai
    from google.genai import types
    _GENAI_MODERN = True
except Exception as e:
    genai = None
    types = None
    _GENAI_MODERN = False

from PIL import Image, ImageEnhance, ImageOps, ImageFilter

try:
    from ekg_ai_pro import (
        TEDAVI_ALGORITMALARI,
        tedavi_algoritmasi_bul,
        MANUEL_ISIMLER,
    )
    print("✓ Sözlük ekg_ai_pro.py'den yüklendi")
except Exception as e:
    raise SystemExit("❌ ekg_ai_pro.py aynı klasörde olmalı! " + str(e))

# ─── API ANAHTARLARI ─────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

GEMINI_CLIENT = None
if GEMINI_API_KEY and _GENAI_MODERN:
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
elif GEMINI_API_KEY:
    raise SystemExit("❌ google-genai paketi eksik. 'pip install google-genai' calistirin.")

DB_PATH = os.environ.get("EKG_DB_PATH", "ekg_paramedik.db")

# ─── KULLANICI REFERANS URL'LERI (internet dogrulamaya entegre edilecek) ─────
USER_REFERENCES = [
    "https://acilci.net/ekg-akil-kartlari-1/",
    "https://aciltip.medicine.ankara.edu.tr/wp-content/uploads/sites/818/2014/11/d5_ekg.pdf",
    "https://ailehekimi.medicine.ankara.edu.tr/wp-content/uploads/sites/581/2015/02/Elektrokardiyografi.pdf",
]

# ─── YAPILANDIRMA ────────────────────────────────────────────────────────────
class Config:
    # En iyi gorsel okuma modelleri (goruntu -> yapi/sayilar/tani)
    BEST_VISION = ["gemini-pro", "gpt-4o", "claude", "gemini-flash", "pixtral", "qwen-vl"]
    # En iyi analiz/hakem modelleri (metin -> konsensus karar)
    BEST_ANALYSIS = ["gemini-pro", "gpt-4o", "claude", "o3"]

    # Hizli tarama icin ucuz/hizli modeller
    FAST_MODELS = ["gemini-flash"]
    # Derin analizde kullanilacak tum okuyucular (API anahtari varsa otomatik eklenir)
    DEEP_MODELS = ["gemini-flash", "gemini-pro"]
    # Hakem (judge) ve dogrulayici (red team)
    JUDGE_MODELS = ["gemini-pro", "gemini-flash"]
    VERIFY_MODELS = ["gemini-pro", "gemini-flash"]

    SCREEN_TIMEOUT = 20
    MODEL_TIMEOUT = 55
    JUDGE_TIMEOUT = 70
    VERIFY_TIMEOUT = 55
    INTERNET_TIMEOUT = 35

    CACHE_TTL_SN = 600
    MIN_AI_RESPONSES = 2

    ST_ELEV_THRESHOLD = 1.0
    V2V3_ST_ELEV = 1.5
    ST_DEP_THRESHOLD = 0.5

    MAX_IMAGE_SIZE = 1800
    BLUR_KOTU = 80
    BLUR_ORTA = 220

    # Lokal yaklastirmali derivasyon okuma (sadece kagit EKG)
    ZOOM_READING = True
    ZOOM_MAX_CROPS = 6
    ZOOM_TARGET_SIZE = (1800, 1800)
    ZOOM_MODELS = ["gemini-pro"]

    # Internet dogrulama (Google Search) - Gemini ile
    INTERNET_DOGRULAMA = True
    INTERNET_MODEL = "gemini-flash-latest"

# ─── KATEGORILER VE KODLAR ───────────────────────────────────────────────────
EMERGENCY = {
    "ANTERIOR_MI", "INFERIOR_MI", "LATERAL_MI", "POSTERIOR_MI",
    "SAG_V_MI", "YAYGIN_ANTERIOR_MI", "VT", "VF", "ASISTOLI", "PEA", "TORSADES",
}
PREARREST = {"BRADIKARDI", "AV_BLOK", "SVT", "AF", "WPW"}
ORTA = {"NSTEMI"}

KEY2CODE = {
    "ANTERIOR_MI": "SB-ASH-Y-02 (AKS)",
    "INFERIOR_MI": "SB-ASH-Y-02 (AKS)",
    "LATERAL_MI": "SB-ASH-Y-02 (AKS)",
    "POSTERIOR_MI": "SB-ASH-Y-02 (AKS)",
    "SAG_V_MI": "SB-ASH-Y-02 (AKS)",
    "YAYGIN_ANTERIOR_MI": "SB-ASH-Y-02 (AKS)",
    "NSTEMI": "SB-ASH-Y-02 (AKS)",
    "AF": "SB-ASH-Y-08",
    "SVT": "SB-ASH-Y-08",
    "VT": "SB-ASH-Y-08",
    "VF": "SB-ASH-Y-11",
    "ASISTOLI": "SB-ASH-Y-10",
    "PEA": "SB-ASH-Y-10",
    "TORSADES": "SB-ASH-Y-08",
    "BRADIKARDI": "SB-ASH-Y-07",
    "AV_BLOK": "SB-ASH-Y-07",
    "WPW": "SB-ASH-Y-08",
    "NORMAL": "SB-ASH-Y-02",
    "GENEL": "SB-ASH-Y-01/02",
    "SOL_ANA_KORONER": "SB-ASH-Y-02 (AKS)",
}

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]
CANON = {l.upper(): l for l in LEADS}

# ─── GOMULU KLINIK BILGI BANKASI (RAG benzeri) ───────────────────────────────
BILGI_BANKASI = {
    "ANTERIOR_MI": "Anterior STEMI: V1-V4'te ST elevasyonu. V2-V3 erkek >=2.5mm, kadin >=1.5mm; digerleri >=1mm. Hedef: kapiya kadinik zaman <90 dk. Aspirin, P2Y12, heparin, nitrat (TA izin verirse), morfin.",
    "YAYGIN_ANTERIOR_MI": "Yaygin anterior MI: V1-V6 + I,aVL'de ST elevasyonu. LAD proksimal tikanikligi. Yuksek risk, kardiogenik sok olabilir. Hizli transport.",
    "INFERIOR_MI": "Inferior STEMI: II, III, aVF elevasyonu. RCA/LCx kaynakli. Mutlaka V4R cek, sag V MI ekarte et. Nitrogliserin hipotansiyonda kontrendike.",
    "LATERAL_MI": "Lateral STEMI: I, aVL, V5-V6 elevasyonu. LCx veya LAD diyagonal tutulumu.",
    "POSTERIOR_MI": "Posterior MI: V1-V3'ta yatay depresyon + R dalgasi + dik T. V7-V9'da elevasyon dogrular. AKS protokolu.",
    "SAG_V_MI": "Sag ventrikul MI: Inferior MI + V4R>=1mm. Hipotansiyon, temiz akciger. NITROGLISERIN YASAK. SF bolus 250-500ml.",
    "NSTEMI": "NSTEMI: ST elevasyonu yok, depresyon/T inversiyonu olabilir. Troponin pozitif. Riskli ama acil olmayabilir. Seri EKG 5-10 dk.",
    "AF": "Atriyal fibrilasyon: duzensiz dar QRS, P yok. Hizli AF'ta hiz kontrolu (diltiazem/amiadoron), antikoagulasyon degerlendirilir. Hemodinamik bozukluk varsa sedelektrik kardiyoversiyon.",
    "SVT": "SVT: duzenli dar QRS tasikardi 150-250/dk. Vagal manevra, adenozin. Genis QRS ise VT dusun. Hemodinamik bozukluk varsa senkronize kardiyoversiyon.",
    "VT": "Ventrikuler tasikardi: genis QRS, duzenli. Nabiz yoksa defibrilasyon. Nabiz varsa amiodaron. Stabil degilse senkronize kardiyoversiyon.",
    "VF": "Ventrikuler fibrilasyon: kaotik ritim, QRS yok. Arrest. Hemen baslangic CPR, defibrilasyon. Adrenalin her 3-5 dk.",
    "ASISTOLI": "Asistoli: duz hat, elektriksel aktivite yok. Arrest. Baslangic CPR, adrenalin. SOK YOK. H/T nedenleri dusun.",
    "PEA": "PEA: Elektriksel aktivite var ama nabiz yok. Yuksek kaliteli CPR, adrenalin, tersine cevrilebilir nedenleri hizli tani ve tedavi et.",
    "TORSADES": "Torsades de Pointes: polimorfik VT, QTc uzama bagli. Magnezyum 2g IV, bradikardi bagimliysa hiz artir, QTc uzatan ilaclari kes.",
    "BRADIKARDI": "Semptomatik bradikardi: atropin 0.5mg IV her 3-5 dk (maks 3mg). Pacing hazir. Infeerior MI ve ilaclar (beta bloker, CCB, digoksin) neden olabilir.",
    "AV_BLOK": "AV blok: PR uzama, Mobitz I/II, tam blok. Mobitz II ve tam blokta transkutan pacing hazir. Atropin tam blokta genelde etkisiz.",
    "WPW": "WPW/preexcitasyon: kisa PR, delta dalgasi. AF+hzl1i ileti -> VF riski. Prokainamid/ibutilid; AV nod yavaslaticilara kacinin.",
    "SOL_ANA_KORONER": "aVR elevasyonu + yaygin ST depresyonu: sol ana koroner veya proksimal LAD kritik darlik. Yuksek riskli AKS, hizli transport.",
    "NORMAL": "Normal sinuz ritmi: duzenli, hiz 60-100, normal P-QRS-T. Semptom varsa seri EKG + troponin.",
}

ACIL_NOTLAR = {
    "VF": ["Hemen CPR baslat", "Defibrilatoru hazirla, sokla", "Adrenalin 1mg IV/IO her 3-5 dk", "H/T nedenlerini dusun"],
    "ASISTOLI": ["Hemen CPR baslat", "Adrenalin 1mg IV/IO her 3-5 dk", "SOK YOK", "H/T nedenlerini dusun (hipotermi, hiperkalemi vb.)"],
    "PEA": ["Yuksek kaliteli CPR", "Adrenalin her 3-5 dk", "Tersine cevrilebilir nedenleri hizli tani/tedavi et", "SOK YOK"],
    "TORSADES": ["Magnezyum 2g IV yavas", "Bradikardi varsa hiz artir", "QTc uzatan ilaclari durdur", "Nabiz yoksa defibrilasyon+CPR"],
    "VT": ["Nabiz kontrolu yap", "Nabiz yoksa arrest algoritmasina gec", "Nabiz varsa amiodaron 150-300mg IV"],
    "BRADIKARDI": ["Atropin 0.5mg IV her 3-5 dk (maks 3mg)", "Transkutan pacing hazir", "Dopamin/epinefrin infuzyonu dusun"],
    "AV_BLOK": ["Mobitz II / tam blokta pacing hazir", "Atropin tam blokta genelde etkisiz", "Hemen transport"],
    "SVT": ["Vagal manevra", "Adenozin 6-12mg IV hizli bolus", "Hemodinamik bozukluk varsa senkronize kardiyoversiyon"],
    "AF": ["Hizli AF'ta hiz kontrolu", "Hemodinamik bozukluk varsa sedelektrik kardiyoversiyon", "Seri EKG ile transport"],
    "WPW": ["AF+hzl1i ileti varsa kardiyoversiyon", "Prokainamid/ibutilid dusun", "AV nod yavaslaticilara KACIN"],
    "ANTERIOR_MI": ["Aspirin 325mg cigneme (alerji yoksa)", "Nitrogliserin sublingual (TA>90 ise)", "Morfin gerekirse", "Hizli STEMI merkezine transport"],
    "INFERIOR_MI": ["V4R cek, sag V MI ekarte et", "TA<90 ise NITROGLISERIN YASAK", "SF bolus hazir tut"],
    "SAG_V_MI": ["NITROGLISERIN KESINLIKLE YASAK", "250-500ml SF bolus", "Hipotansiyon varsa sivi resusitasyonu"],
    "POSTERIOR_MI": ["V7-V9 cek", "V1-V3 ayna goruntusunu dogrula", "AKS protokolu uygula"],
    "YAYGIN_ANTERIOR_MI": ["Yuksek riskli AKS", "TA duserse nitrat kes + sivi", "Acil PCI merkezine transport"],
    "SOL_ANA_KORONER": ["aVR elev + yaygin dep", "Kardiyojenik sok riski yuksek", "Inotrop/sivi + hizli transport"],
    "NSTEMI": ["Aspirin cigneme", "Seri EKG 5-10 dk", "Instabilse yuksek riskli kabul"],
}

# ─── YARDIMCI FONKSIYONLAR ───────────────────────────────────────────────────
def temizle_algoritma(m):
    # Hastane sonrasi asamalari sil (--- ile baslayan HASTANEDE bolumu)
    m = re.sub(r"(?:\n---)?\n?📚 HASTANEDE.*", "", m, flags=re.S)
    m = re.sub(r"(?:\n---)?\n?HASTANEDE.*", "", m, flags=re.S)
    m = re.sub(r"\n{3,}", "\n\n", m)
    return m.strip()


def temiz_algo(k):
    a = TEDAVI_ALGORITMALARI.get(k) or TEDAVI_ALGORITMALARI.get("GENEL")
    if not a:
        return {"aciliyeti": "Bilinmiyor", "algoritma": "Protokol bulunamadi."}
    return {"aciliyeti": a["aciliyeti"], "algoritma": temizle_algoritma(a["algoritma"])}


def strip_sources(text: str) -> str:
    """UI'da gosterilecek metinden kaynak gibi gorunen URL/kaynak satirlarini temizle."""
    if not text:
        return text
    # URL'leri kaldir
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    # Kaynak / References satirlarini kaldir
    text = re.sub(r"(?i)\b(?:kaynak|kaynakça|references?|sources?|doi)\b[\s:]*.*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─── PROMPT'LAR ─────────────────────────────────────────────────────────────
ATLAS = """ATLAS:
Anterior STEMI: V1-V4 elevasyon.
Yaygin Anterior: V1-V6 + I,aVL elevasyon.
Inferior: II,III,aVF elevasyon; V4R kontrol.
Lateral: I,aVL,V5-V6 elevasyon.
Posterior: V1-V3 dep + belirgin R + dik T; V7-V9 elevasyon.
Sag V MI: inferior + V4R>=1mm; hipotansiyon; nitro yasak.
de Winter: V1-V6 upsloping dep 1-3mm + dik T.
Wellens: V2-V3 bifazik/derin negatif T.
NSTEMI: dep/T inversiyonu; elevasyon yok.
AF: duzensiz dar QRS; P yok.
SVT: duzenli dar 150-250/dk.
VT: genis QRS; duzenli; AV disosiasyon.
VF: kaotik; QRS yok.
Asistoli: duz hat.
PEA: elektriksel aktivite var, nabiz yok.
Torsades: polimorfik VT, QRS donuyor.
Blok: PR>200; Mobitz I/II; tam blok.
WPW: kisa PR, delta dalgasi."""


LETHAL_SCAN_PROMPT = """Sen deneyimli bir acil tip ve kardiyologsun. Sana EKG goruntusu verildi.
ONCELIKLE olumcul/prearrest ritimleri tani: VF, VT, asistoli, PEA, semptomatik bradikardi,
AV blok (Mobitz II/tam blok), SVT, AF hizli ventrikuler yantt, Torsades, WPW+AF.

Sadece JSON don:
{"olumcul":"EVET|HAYIR","tahmin_onerisi":"<tani adi>","guven":0-100,
"neden":"1-2 cumle","goruntu_turu":"kagit|monitor|belirsiz"}

Tani adlari: VF, Ventrikuler Tasikardi, Asistoli, PEA, Bradikardi, AV Blok, SVT, AF,
Anterior STEMI, Inferior STEMI, Lateral STEMI, Posterior STEMI, Sag Ventrikul MI,
Yaygin Anterior MI, NSTEMI, Normal Sinuz Ritmi, Torsades de Pointes, WPW, Belirsiz."""


TARAMA_PROMPT = """Kidemli kardiyolog gibi davran. Sana EKG fotografi verildi.
HIZLI karar ver. Bu EKG asagidaki gruptan hangisine yakin?
Sadece JSON:
{"acil":"EVET|BELIRSIZ|HAYIR","tahmin_onerisi":"<listeden>","gerekce":"1 cumle"}

Tani listesi: Anterior STEMI, Yaygin Anterior MI, Inferior STEMI, Lateral STEMI,
Posterior STEMI, Sag Ventrikul MI, NSTEMI, AF, SVT, VT, VF, PEA, Torsades,
Bradikardi, AV Blok, WPW, Asistoli, Normal Sinuz Ritmi.

EVET = hayati tehdit (STEMI, VT, VF, asistoli, PEA, semptomatik blok/bradikardi, Torsades, WPW+AF).
BELIRSIZ = kotti fotograf, borderline, emin degil.
HAYIR = normal, stabil NSTEMI, AF/SVT stabil."""


DETAYLI_PROMPT = """Sana orijinal ve iyilestirilmis EKG goruntuleri verildi.
Kidemli kardiyologsun. HASTANE ONCESI PARAMEDIK gozuyle degerlendir.
ONCE olc, SONRA tanı koy. Ozellikle VF, VT, asistoli, PEA, bradikardi, AV blok,
SVT, AF, Torsades, WPW gibi olumcul/prearrest ritimleri ve MI paternlerini kacirma.

""" + ATLAS + """

TANI LISTESI (tam birini sec):
"Anterior STEMI","Yaygin Anterior MI","Inferior STEMI","Lateral STEMI",
"Posterior STEMI","Sag Ventrikul MI","NSTEMI","AF","SVT","VT","VF",
"PEA","Torsades de Pointes","Bradikardi","AV Blok","WPW","Asistoli","Normal Sinuz Ritmi".

KURALLAR:
- Komsu >=2 derivasyonda >=1mm elev (V2-V3 >=1.5-2mm) = STEMI
- Resiprokal depresyon STEMI'yi guclendirir
- V1-V3 dep + belirgin R + elev yok = Posterior
- V1-V6 upsloping dep + dik T = de Winter (STEMI esdegeri)
- aVR elev + yaygin dep = sol ana koroner
- dar+duzenli = SVT, dar+duzensiz = AF, genis+duzenli = VT
- kaotik QRS yok = VF; duz hat = asistoli; elektrik var nabiz yok = PEA
- kisa PR + delta = WPW

Sadece JSON:
{"kalite":"iyi|orta|kotu","tum_leadler":true|false,
"st_mm":{"I":0,"II":0,"III":0,"aVR":0,"aVL":0,"aVF":0,
"V1":0,"V2":0,"V3":0,"V4":0,"V5":0,"V6":0},
"hiz":int|null,"duzenli":true|false|null,
"pr_ms":int|null,"qrs_ms":int|null,"qtc_ms":int|null,
"tahmin":"<listeden>","stemi":true|false,"guven":0-100,
"detay":"3-5 cumle Turkce, derivasyon+mm belirterek"}"""


SEMA_HAKEM = """Sadece JSON:
{"tahmin":"<listeden>","stemi":true|false,"guven":0-100,
"st_mm":{12 lead mm},
"olcumler":{"hiz":int|null,"pr_ms":int|null,"qrs_ms":int|null,"qtc_ms":int|null},
"detay":"4-6 cumle Turkce","nihai_gerekce":"2-3 cumle"}"""


SEMA_VER = """Sadece JSON:
{"kritik_bulgu":true|false,"oneri_tahmin":"<listeden>",
"guven":0-100,"gerekce":"1-2 cumle"}"""


# ─── GORUNTU KALITE KONTROLU ─────────────────────────────────────────────────
def blur_score(raw: bytes) -> Tuple[float, str]:
    try:
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0, "okunamadi"
        h, w = img.shape
        if max(h, w) > 1200:
            scale = 1200 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        score = float(cv2.Laplacian(img, cv2.CV_64F).var())
        if score < Config.BLUR_KOTU:
            return score, "kotu"
        elif score < Config.BLUR_ORTA:
            return score, "orta"
        return score, "iyi"
    except Exception as e:
        logging.warning("Blur hesaplanamadi: %s", e)
        return 0.0, "bilinmiyor"


# ─── KAGIT vs MONITOR TESPITI ────────────────────────────────────────────────
def detect_image_type(raw: bytes) -> str:
    """EKG kagidi mi (grid, 12 lead) yoksa monitor ekrani mi (koyu zemin, tek lead)?"""
    try:
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return "belirsiz"
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Monitorler genelde koyu zeminli; kagit beyaz/beyazimsi
        mean_bright = float(np.mean(gray))
        # Grid cizgileri (kagit EKG) tespiti: Fourier'de guclu yatay/dikey frekanslar
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=w * 0.3, maxLineGap=10)
        n_lines = len(lines) if lines is not None else 0
        # Kagit EKG: cok sayida paralel grid cizgisi + beyaz zemin
        if mean_bright > 180 and n_lines > 30:
            return "kagit"
        if mean_bright < 100 or n_lines < 10:
            return "monitor"
        return "kagit" if n_lines > 20 else "belirsiz"
    except Exception as e:
        logging.warning("Goruntu turu tespit edilemedi: %s", e)
        return "belirsiz"


# ─── GRID TEMIZLEME ──────────────────────────────────────────────────────────
def remove_grid_fourier(raw: bytes) -> bytes:
    try:
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return raw
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        band = 4
        mask[max(0, crow - band):min(rows, crow + band), 0:max(0, ccol - 15)] = 0
        mask[max(0, crow - band):min(rows, crow + band), min(cols, ccol + 15):] = 0
        mask[0:max(0, crow - 15), max(0, ccol - band):min(cols, ccol + band)] = 0
        mask[min(rows, crow + 15):, max(0, ccol - band):min(cols, ccol + band)] = 0
        fshift = fshift * mask
        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
        img_back = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
        img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_back = cv2.filter2D(img_back, -1, kernel)
        _, buf = cv2.imencode(".png", img_back)
        return buf.tobytes()
    except Exception as e:
        logging.warning("Grid temizligi basarisiz: %s", e)
        return raw


def enhance_monitor(raw: bytes) -> bytes:
    """Monitor ekrani icin: koyu zeminden sinyali belirginlestir."""
    try:
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return raw
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Adaptif esikleme ile sinyali cikar
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enh = clahe.apply(gray)
        # Morfolojik islemlerle ince cizgileri kalinlastir
        kernel = np.ones((2, 2), np.uint8)
        enh = cv2.morphologyEx(enh, cv2.MORPH_CLOSE, kernel)
        enh = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
        _, buf = cv2.imencode(".png", enh)
        return buf.tobytes()
    except Exception as e:
        logging.warning("Monitor iyilestirme basarisiz: %s", e)
        return raw


# ─── DERIVASYON YAKINLASTIRMA (LOCAL OKUMA) ──────────────────────────────────
def _cluster(vals: List[int], tol: int = 30) -> List[int]:
    if not vals:
        return []
    vals = sorted(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(sum(g) / len(g)) for g in groups]


def detect_lead_grid(raw: bytes) -> Optional[List[Tuple[int, int, int, int]]]:
    """EKG kagidindaki grid cizgilerini kullanarak 12 lead bolgesi tespit etmeye calisir."""
    try:
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Grid cizgilerini guclendirmek icin morfolojik islemler
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Yatay cizgiler
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
        # Dikey cizgiler
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 20))
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
        grid = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0)
        edges = cv2.Canny(grid, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                                minLineLength=min(w, h) * 0.12, maxLineGap=20)
        if lines is None or len(lines) < 6:
            return None
        horiz_y = []
        vert_x = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > dy * 3:
                horiz_y.append((y1 + y2) // 2)
            elif dy > dx * 3:
                vert_x.append((x1 + x2) // 2)
        h_lines = _cluster(horiz_y, tol=max(20, h // 30))
        v_lines = _cluster(vert_x, tol=max(20, w // 30))
        # Kagit EKG'de genellikle 3-4 satir ve 3-4 sutun olur
        if len(h_lines) < 2 or len(v_lines) < 2:
            return None
        # Cok fazla cizgi varsa, en buyuk aralikli olanlari sec (lead sinirlari)
        if len(h_lines) > 5:
            h_lines = sorted(h_lines)
        if len(v_lines) > 5:
            v_lines = sorted(v_lines)
        cells = []
        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                cells.append((v_lines[j], h_lines[i], v_lines[j + 1], h_lines[i + 1]))
        return cells
    except Exception as e:
        logging.warning("Lead grid tespiti basarisiz: %s", e)
        return None


def _enhance_crop(crop: np.ndarray) -> np.ndarray:
    """Kirpilmis bolgeyi EKG okumaya uygun iyilestir."""
    if crop is None or crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    enh = cv2.detailEnhance(cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR))
    # Keskinlestirme
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enh = cv2.filter2D(enh, -1, kernel)
    return enh


def quadrant_crops(raw: bytes, max_crops: int = 4) -> List[bytes]:
    """Goruntuyu 4 bolgeye kirparak yakinlastirir."""
    try:
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return []
        h, w = img.shape[:2]
        crops = []
        regions = [
            (0, 0, w // 2, h // 2),
            (w // 2, 0, w, h // 2),
            (0, h // 2, w // 2, h),
            (w // 2, h // 2, w, h),
        ]
        target_w, target_h = Config.ZOOM_TARGET_SIZE
        for x1, y1, x2, y2 in regions[:max_crops]:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = _enhance_crop(crop)
            crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            _, buf = cv2.imencode(".png", crop)
            crops.append(buf.tobytes())
        return crops
    except Exception as e:
        logging.warning("Bolge kirpma basarisiz: %s", e)
        return []


def zoom_lead_crops(raw: bytes, max_crops: int = 6) -> List[bytes]:
    """EKG kagidindaki lead bolgelerini tespit edip yakinlastirir."""
    try:
        if detect_image_type(raw) != "kagit":
            return []
        cells = detect_lead_grid(raw)
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return []
        target_w, target_h = Config.ZOOM_TARGET_SIZE
        crops = []
        if cells and len(cells) >= 4:
            # En buyuk hucreleri sec (bos/etiket alani degil, sinyal bolgesi olsun)
            cells = sorted(cells, key=lambda c: (c[2]-c[0])*(c[3]-c[1]), reverse=True)
            for x1, y1, x2, y2 in cells[:max_crops]:
                # Biraz pay ekle
                pad_x = max(5, (x2 - x1) // 20)
                pad_y = max(5, (y2 - y1) // 20)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(img.shape[1], x2 + pad_x)
                y2 = min(img.shape[0], y2 + pad_y)
                crop = img[y1:y2, x1:x2]
                crop = _enhance_crop(crop)
                crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                _, buf = cv2.imencode(".png", crop)
                crops.append(buf.tobytes())
        else:
            # Grid tespit edilemezse 4 bolgeye kirp
            crops = quadrant_crops(raw, max_crops=4)
        return crops[:max_crops]
    except Exception as e:
        logging.warning("Zoom lead crops basarisiz: %s", e)
        return []


# ─── GORUNTU ISLEME ──────────────────────────────────────────────────────────
def _prep(img: Image.Image, gray: bool = False, max_size: int = Config.MAX_IMAGE_SIZE) -> bytes:
    if gray:
        img = ImageOps.grayscale(img).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        f = max_size / max(w, h)
        img = img.resize((int(w * f), int(h * f)), Image.LANCZOS)
    elif w < 1200:
        f = 1200 / w
        img = img.resize((int(w * f), int(h * f)), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.6)
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def varyantlar(raw: bytes, grid_clean: bool = True) -> Tuple[bytes, bytes, bytes, str]:
    base = Image.open(io.BytesIO(raw)).convert("RGB")
    img_type = detect_image_type(raw)
    orig = _prep(base)
    grid = remove_grid_fourier(orig) if grid_clean else orig
    monitor = enhance_monitor(orig) if img_type == "monitor" else orig
    gray = _prep(base, gray=True)
    return orig, grid, gray, img_type


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:24]


def _tj(t: str) -> Dict[str, Any]:
    t = re.sub(r"```(?:json)?", "", t).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSON bulunamadi")
    return json.loads(t[start:end + 1])


def _retry(fn, *a, tries: int = 2, **kw):
    s = None
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except Exception as e:
            s = e
            time.sleep(1 + i)
    raise s


def _oc(base: str, key: str, model: str, iv: List[bytes],
        prompt: Optional[str] = None, timeout: int = 30,
        extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    h = {"Authorization": f"Bearer {key}"}
    if extra_headers:
        h.update(extra_headers)
    imgs = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(x)}"}} for x in iv]
    content = [{"type": "text", "text": prompt or DETAYLI_PROMPT}] + imgs
    pl = {"model": model, "messages": [{"role": "user", "content": content}]}
    if model.startswith(("o1", "o3", "o4")):
        pl["max_completion_tokens"] = 1200
    else:
        pl["temperature"] = 0.0
        pl["max_tokens"] = 1200
    r = requests.post(base.rstrip("/") + "/chat/completions", headers=h, timeout=timeout, json=pl)
    r.raise_for_status()
    return _tj(r.json()["choices"][0]["message"]["content"])


GEMINI_FALLBACKS = [
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-pro-latest",
    "gemini-2.0-pro-exp-02-05",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
]


def gemini_oku(iv: List[bytes], model: str = "gemini-flash-latest",
               prompt: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
    if not GEMINI_CLIENT:
        raise RuntimeError("Gemini client hazir degil")
    ims = [Image.open(io.BytesIO(x)) for x in iv]
    contents = [prompt or DETAYLI_PROMPT] + ims
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
    )
    last_err = None
    candidates = [model] + [m for m in GEMINI_FALLBACKS if m != model]
    for m in candidates:
        try:
            response = GEMINI_CLIENT.models.generate_content(
                model=m,
                contents=contents,
                config=cfg,
            )
            return _tj(response.text)
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "404" in err_str or "NOT_FOUND" in err_str or "is not found" in err_str:
                logging.warning("Gemini model %s bulunamadi, sonraki deneniyor", m)
                continue
            raise
    raise last_err or RuntimeError("Hicbir Gemini modeli calismadi")


def claude_oku(iv: List[bytes], prompt: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
    blk = []
    for x in iv:
        blk.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(x)}})
    blk.append({"type": "text", "text": prompt or DETAYLI_PROMPT})
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
        timeout=timeout,
        json={"model": "claude-sonnet-4-5", "max_tokens": 1200, "temperature": 0.0,
              "messages": [{"role": "user", "content": blk}]})
    r.raise_for_status()
    return _tj(r.json()["content"][0]["text"])


def groq_oku(iv: List[bytes], prompt: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
    s = None
    for m in ["llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]:
        try:
            return _oc("https://api.groq.com/openai/v1", GROQ_API_KEY, m, iv, prompt, timeout=timeout)
        except Exception as e:
            s = e
            logging.warning("Groq model %s basarisiz: %s", m, e)
    raise s or RuntimeError("Groq ile hicbir vision modeli calismadi")


# ─── INTERNET DOGRULAMA ──────────────────────────────────────────────────────
def bilgi_bankasi_getir(key: str) -> str:
    return BILGI_BANKASI.get(key, BILGI_BANKASI.get("NORMAL", ""))


def fetch_reference_text(url: str, max_chars: int = 2000) -> str:
    """Kullanici referans URL'lerinden basit metin cekmeyi dener."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return ""
        ct = r.headers.get("Content-Type", "")
        if "pdf" in ct.lower() or url.lower().endswith(".pdf"):
            return "[PDF icerigi cevrimici okunamadi, prompt'ta URL olarak referans alindi]"
        text = r.text
        # HTML etiketlerini kaldir
        text = re.sub(r"<script.*?</script>", " ", text, flags=re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        logging.warning("Referans URL okunamadi %s: %s", url, e)
        return ""


def _gemini_internet_search(query: str) -> Optional[str]:
    if not GEMINI_CLIENT:
        return None
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=Config.INTERNET_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        return (response.text or "")[:900]
    except Exception as e:
        logging.warning("Gemini internet arama hatasi: %s", e)
        return None


def internet_dogrulama(tani_key: str, st_mm: Dict[str, float]) -> Dict[str, Any]:
    result = {"ozet": None, "referanslar": [], "uyusma": None}
    if not Config.INTERNET_DOGRULAMA:
        return result

    bilgi = bilgi_bankasi_getir(tani_key)
    st_text = json.dumps(st_mm, ensure_ascii=False)
    ref_ozetler = []
    for url in USER_REFERENCES:
        txt = fetch_reference_text(url)
        if txt:
            ref_ozetler.append(f"[{url}] {txt[:400]}")

    # Internet arama sorgusu (kaynak gostermeden icerik al)
    q = (f"Paramedik hastane oncesi yonetimi: '{tani_key}' EKG tanisi. "
         f"SB-ASH-Y acil saglik hizmetleri protokolune gore olay yerinde ne yapilir? "
         f"ST olculeri: {st_text}. Klinik baglam: {bilgi}")
    net = _gemini_internet_search(q)
    if net:
        result["ozet"] = strip_sources(net)

    # Karsilastirma: ref metinleri ile bilgi bankasi uyusuyor mu?
    if ref_ozetler:
        ref_birlestir = " ".join(ref_ozetler)
        # Basit uyusma: kritik kelimeler
        keywords = {
            "VF": ["defibrilasyon", "CPR", "adrenalin"],
            "ASISTOLI": ["CPR", "adrenalin", "sok yok"],
            "VT": ["amiodaron", "kardiyoversiyon", "defibrilasyon"],
            "SVT": ["adenozin", "vagal", "kardiyoversiyon"],
            "BRADIKARDI": ["atropin", "pacing"],
            "ANTERIOR_MI": ["aspirin", "nitrat", "PCI"],
            "INFERIOR_MI": ["V4R", "sag ventrikul", "nitro"],
        }
        kelimeler = keywords.get(tani_key, [])
        uyusan = sum(1 for k in kelimeler if k.lower() in (ref_birlestir + (result.get("ozet") or "")).lower())
        if kelimeler:
            result["uyusma"] = round(100 * uyusan / len(kelimeler))
        result["referanslar"] = [strip_sources(r) for r in ref_ozetler]

    return result


# ─── MODEL KAYIT DEFTERI ─────────────────────────────────────────────────────
REG: Dict[str, callable] = {}


def _add(n: str, f):
    if f and n not in REG:
        REG[n] = f


if GEMINI_CLIENT:
    _add("gemini-flash", partial(gemini_oku, model="gemini-flash-latest"))
    _add("gemini-pro", partial(gemini_oku, model="gemini-pro-latest"))
if GROQ_API_KEY:
    _add("groq", groq_oku)
if OPENAI_API_KEY:
    _add("gpt-4o", partial(_oc, "https://api.openai.com/v1", OPENAI_API_KEY, "gpt-4o"))
    _add("o3", partial(_oc, "https://api.openai.com/v1", OPENAI_API_KEY, "o3", timeout=Config.JUDGE_TIMEOUT))
if ANTHROPIC_API_KEY:
    _add("claude", claude_oku)
if MISTRAL_API_KEY:
    _add("pixtral", partial(_oc, "https://api.mistral.ai/v1", MISTRAL_API_KEY, "pixtral-large-latest"))
if XAI_API_KEY:
    _add("grok", partial(_oc, "https://api.x.ai/v1", XAI_API_KEY, "grok-2-vision-1212"))
if COHERE_API_KEY:
    _add("aya-vision", partial(_oc, "https://api.cohere.com/v2", COHERE_API_KEY, "aya-vision-32b"))
if OPENROUTER_API_KEY:
    OR = {"HTTP-Referer": "https://ekg.local", "X-Title": "Paramedik EKG"}
    # Ucretsiz OpenRouter modelleri
    OR_FREE_MODELS = [
        ("gemini-2.5-flash-free", "google/gemini-2.5-flash:free"),
        ("gemini-1.5-flash-free", "google/gemini-1.5-flash:free"),
        ("llama-4-scout-free", "meta-llama/llama-4-scout:free"),
        ("llama-3.2-90b-vl-free", "meta-llama/llama-3.2-90b-vision-instruct:free"),
        ("llama-3.2-11b-vl-free", "meta-llama/llama-3.2-11b-vision-instruct:free"),
        ("qwen-2.5-vl-free", "qwen/qwen-2.5-vl-72b-instruct:free"),
        ("deepseek-r1-free", "deepseek/deepseek-r1:free"),
        ("mistral-small-3.1-free", "mistralai/mistral-small-3.1-24b-instruct:free"),
        ("phi-4-multimodal-free", "microsoft/phi-4-multimodal-instruct:free"),
        # Ucretli modeller (denge icin)
        ("gpt-4o-or", "openai/gpt-4o"),
        ("claude-or", "anthropic/claude-sonnet-4"),
        ("gemini-pro-or", "google/gemini-1.5-pro"),
    ]
    for n, s in OR_FREE_MODELS:
        _add(n, partial(_oc, "https://openrouter.ai/api/v1", OPENROUTER_API_KEY, s, extra_headers=OR))

READERS = list(REG.items())
HAKEM = [(n, REG[n]) for n in Config.JUDGE_MODELS if n in REG]
VERIF = [(n, REG[n]) for n in Config.VERIFY_MODELS if n in REG]

# En iyi gorsel okuyucuyu ve analizciyi sec
def pick_best(names: List[str]) -> Optional[str]:
    for n in names:
        if n in REG:
            return n
    return None

BEST_VISION_MODEL = pick_best(Config.BEST_VISION)
BEST_ANALYSIS_MODEL = pick_best(Config.BEST_ANALYSIS)

if not READERS:
    print("⚠️ UYARI: Hicbir AI okuyucu aktif degil. GEMINI_API_KEY (veya diger API anahtarlari) kontrol edin.")
else:
    print(f"✓ {len(READERS)} AI hazir | En iyi gorsel: {BEST_VISION_MODEL or 'YOK'} | En iyi analiz: {BEST_ANALYSIS_MODEL or 'YOK'}")


# ─── ISTATISTIKSEL YARDIMCILAR ───────────────────────────────────────────────
def _med(l: List[float]) -> Optional[float]:
    l = [x for x in l if isinstance(x, (int, float))]
    if not l:
        return None
    srt = sorted(l)
    n = len(srt)
    if n % 2 == 1:
        return round(srt[n // 2], 1)
    return round((srt[n // 2 - 1] + srt[n // 2]) / 2, 1)


def key_of(t: str) -> str:
    a = tedavi_algoritmasi_bul(t)
    for k, v in TEDAVI_ALGORITMALARI.items():
        if v is a:
            return k
    return "GENEL"


def norm_st(d: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        c = CANON.get(str(k).strip().upper())
        if c and isinstance(v, (int, float)):
            out[c] = float(v)
    return out


def medyan_st(liste: List[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    for L in LEADS:
        vals = [d.get(L) for d in liste if d.get(L) is not None]
        if vals:
            out[L] = _med(vals)
    return out


# ─── KURAL MOTORU ────────────────────────────────────────────────────────────
def st_karar(st: Dict[str, float]) -> List[Dict[str, Any]]:
    if not st:
        return []

    def elev(leads: List[str], thr: float = 1.0) -> List[str]:
        return [L for L in leads if (st.get(L) or 0) >= thr]

    def dep(leads: List[str], thr: float = 0.5) -> List[str]:
        return [L for L in leads if (st.get(L) or 0) <= -thr]

    inf_e = elev(["II", "III", "aVF"])
    ant_e_v2v3 = elev(["V2", "V3"], thr=Config.V2V3_ST_ELEV)
    ant_e_rest = elev(["V1", "V4"], thr=Config.ST_ELEV_THRESHOLD)
    ant_e = list(dict.fromkeys(ant_e_v2v3 + ant_e_rest))
    lat_e = elev(["I", "aVL", "V5", "V6"])
    ant_d = dep(["V1", "V2", "V3"])
    lat_d = dep(["I", "aVL", "V5", "V6"])
    inf_d = dep(["II", "III", "aVF"])
    avr = st.get("aVR") or 0

    bul = []
    if len(ant_e) >= 2:
        yay = len([L for L in ["V1", "V2", "V3", "V4", "V5", "V6", "I", "aVL"] if (st.get(L) or 0) >= 1]) >= 6
        bul.append({"bolge": "YAYGIN_ANTERIOR_MI" if yay else "ANTERIOR_MI",
                    "leadler": ant_e, "tip": "elevasyon", "resiprokal": bool(inf_d)})
    if len(inf_e) >= 2:
        bul.append({"bolge": "INFERIOR_MI", "leadler": inf_e, "tip": "elevasyon", "resiprokal": bool(lat_d)})
    if len(lat_e) >= 2:
        bul.append({"bolge": "LATERAL_MI", "leadler": lat_e, "tip": "elevasyon", "resiprokal": bool(inf_d)})
    if len(ant_d) >= 2 and not ant_e:
        bul.append({"bolge": "POSTERIOR_MI", "leadler": ant_d, "tip": "ayna-depresyon", "resiprokal": False})
    vdep = dep(["V2", "V3", "V4", "V5", "V6"], 1.0)
    if len(vdep) >= 4 and not ant_e and not inf_e:
        bul.append({"bolge": "ANTERIOR_MI", "leadler": vdep, "tip": "de-Winter esdegeri", "resiprokal": False})
    yay_dep = len(ant_d) + len(inf_d) + len(lat_d)
    if avr >= 1 and yay_dep >= 4:
        bul.append({"bolge": "SOL_ANA_KORONER", "leadler": ["aVR"], "tip": "aVR elev+yaygin dep", "resiprokal": False})
    return bul


# ─── AKILLI TARAMA ───────────────────────────────────────────────────────────
def quick_screen(iv: List[bytes]) -> Optional[Dict[str, Any]]:
    for model_name in Config.FAST_MODELS:
        if model_name not in REG:
            continue
        try:
            fn = REG[model_name]
            r = _retry(fn, iv, prompt=TARAMA_PROMPT, timeout=Config.SCREEN_TIMEOUT)
            r["_k"] = key_of(r.get("tahmin_onerisi"))
            acil = r.get("acil", "BELIRSIZ").upper()
            return {"acil": acil, "tahmin_key": r["_k"], "tahmin": r.get("tahmin_onerisi"),
                    "gerekce": r.get("gerekce", ""), "raw": r}
        except Exception as e:
            logging.warning("Tarama %s hata: %s", model_name, e)
    return None


def lethal_scan(iv: List[bytes]) -> Optional[Dict[str, Any]]:
    """Hizli olumcul ritim taramasi."""
    for model_name in Config.FAST_MODELS:
        if model_name not in REG:
            continue
        try:
            fn = REG[model_name]
            r = _retry(fn, iv, prompt=LETHAL_SCAN_PROMPT, timeout=Config.SCREEN_TIMEOUT)
            r["_k"] = key_of(r.get("tahmin_onerisi"))
            return {"olumcul": r.get("olumcul", "HAYIR").upper() == "EVET",
                    "tahmin_key": r["_k"], "tahmin": r.get("tahmin_onerisi"),
                    "guven": r.get("guven", 0), "goruntu_turu": r.get("goruntu_turu", "belirsiz"),
                    "neden": r.get("neden", ""), "raw": r}
        except Exception as e:
            logging.warning("Olumcul tarama %s hata: %s", model_name, e)
    return None


# ─── TEK MODEL ANALIZ (HIZLI MOD) ────────────────────────────────────────────
def single_analyze(iv: List[bytes], model_name: str, img_type: str = "belirsiz") -> Dict[str, Any]:
    fn = REG[model_name]
    r = _retry(fn, iv, timeout=Config.MODEL_TIMEOUT)
    r["_k"] = key_of(r.get("tahmin"))
    r["_st"] = norm_st(r.get("st_mm"))
    st_mm = r["_st"]
    karar = st_karar(st_mm)
    return _sonuc_olustur(r.get("tahmin") or MANUEL_ISIMLER.get(r["_k"], r["_k"]),
                          r["_k"], st_mm, karar, r, None, {model_name: r.get("tahmin")},
                          {"hiz": r.get("hiz"), "pr_ms": r.get("pr_ms"),
                           "qrs_ms": r.get("qrs_ms"), "qtc_ms": r.get("qtc_ms")},
                          r.get("kalite", "orta"), None, [], None, None, 0, img_type=img_type)


def _sonuc_olustur(tahmin, anahtar, st_mm, st_karar, temsilci, hakem, oylama,
                   olcumler, kalite, kalite_uyarisi, tutarlilik, arastirma,
                   internet_dogrulama, sure, blur=None, screen=None,
                   lethal=None, img_type: str = "belirsiz") -> Dict[str, Any]:
    final = anahtar
    if st_karar and final not in EMERGENCY:
        for k in st_karar:
            if k["bolge"] in EMERGENCY:
                final = k["bolge"]
                break
    risk = "Yuksek" if final in EMERGENCY else ("Orta" if (final in PREARREST or final in ORTA) else "Dusuk")
    acil = final in EMERGENCY or final in PREARREST or final in ORTA

    notlar = ACIL_NOTLAR.get(final, [])
    if final in ("INFERIOR_MI", "SAG_V_MI"):
        notlar = list(set(notlar + ACIL_NOTLAR.get("SAG_V_MI", []) + ACIL_NOTLAR.get("INFERIOR_MI", [])))

    # Internet dogrulamadan kaynaklari tamamen cikar
    net_text = None
    net_match = None
    if isinstance(internet_dogrulama, dict):
        net_text = strip_sources(internet_dogrulama.get("ozet"))
        net_match = internet_dogrulama.get("uyusma")

    return {
        "tahmin": tahmin,
        "anahtar": final,
        "resmi_protokol": KEY2CODE.get(final, "-"),
        "risk_seviyesi": risk,
        "acil_mudahale": acil,
        "detay": strip_sources((hakem or {}).get("detay") or (temsilci or {}).get("detay", "")),
        "guven": min(100, max(0, int((temsilci or {}).get("guven", 70)))),
        "hakem": None,
        "dogrulayici": {},
        "oylama": oylama,
        "olcumler": olcumler,
        "st_mm": st_mm,
        "st_karar": st_karar,
        "kalite": kalite,
        "kalite_uyarisi": kalite_uyarisi,
        "tutarlilik": tutarlilik,
        "saha_notlari": notlar,
        "arastirma": arastirma,
        "internet_dogrulama": net_text,
        "internet_uyusma": net_match,
        "bilgi_bankasi": strip_sources(bilgi_bankasi_getir(final)),
        "nihai_gerekce": strip_sources((hakem or {}).get("nihai_gerekce", "")),
        "sure_sn": round(sure, 1),
        "blur": blur or {"skor": 0, "durum": "-"},
        "goruntu_turu": img_type,
        "uyari": "Klinik onay zorunludur; nihai degerlendirme hekime aittir.",
        "algoritma": temiz_algo(final),
        "tarama": screen,
        "olumcul_tarama": lethal,
    }


# ─── ENSEMBLE ANALIZ ─────────────────────────────────────────────────────────
_result_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def ensemble_analyze(raw: bytes, mode: str = "auto", grid_clean: bool = True) -> Dict[str, Any]:
    if not READERS:
        raise Exception("API anahtari yok")

    img_hash = _hash(raw)
    cache_key = f"{img_hash}:{mode}:{grid_clean}"
    now = time.time()
    if cache_key in _result_cache:
        ts, res = _result_cache[cache_key]
        if now - ts < Config.CACHE_TTL_SN:
            res["_cached"] = True
            return res

    blur_val, blur_label = blur_score(raw)
    orig, grid, gray, img_type = varyantlar(raw, grid_clean=grid_clean)
    # Kagit EKG ise orijinal+grid+gray; monitor ise orijinal+monitor+gray
    iv = [orig, grid, gray] if img_type != "monitor" else [orig, enhance_monitor(orig), gray]
    # Derivasyon yakinlastirmalari (sadece kagit EKG)
    zooms = zoom_lead_crops(raw, max_crops=Config.ZOOM_MAX_CROPS) if (img_type == "kagit" and Config.ZOOM_READING) else []
    zoom_used = bool(zooms)
    t0 = time.time()

    def _build_iv(model_name: str, base: List[bytes]) -> List[bytes]:
        if zooms and model_name in Config.ZOOM_MODELS:
            return base + zooms
        return base

    # HIZLI MOD
    if mode == "fast":
        model_name = Config.FAST_MODELS[0] if (Config.FAST_MODELS and Config.FAST_MODELS[0] in REG) else READERS[0][0]
        res = single_analyze(iv, model_name, img_type=img_type)
        res["blur"] = {"skor": round(blur_val, 1), "durum": blur_label}
        res["sure_sn"] = round(time.time() - t0, 1)
        _result_cache[cache_key] = (time.time(), res)
        return res

    # OLUMCUL / PREARREST TARAMA (her zaman calis)
    lethal = lethal_scan(iv)

    # AKILLI TARAMA
    screen = None
    if mode == "auto":
        screen = quick_screen(iv)
        if screen and screen["acil"] == "HAYIR" and not (lethal and lethal["olumcul"]):
            model_name = Config.FAST_MODELS[0] if (Config.FAST_MODELS and Config.FAST_MODELS[0] in REG) else READERS[0][0]
            res = single_analyze(iv, model_name, img_type=img_type)
            res["tarama"] = screen
            res["olumcul_tarama"] = lethal
            res["blur"] = {"skor": round(blur_val, 1), "durum": blur_label}
            res["sure_sn"] = round(time.time() - t0, 1)
            _result_cache[cache_key] = (time.time(), res)
            return res

    # DERIN MOD
    oylar: Dict[str, Dict[str, Any]] = {}
    deep_models = [m for m in Config.DEEP_MODELS if m in REG]
    if not deep_models:
        deep_models = [n for n, _ in READERS[:3]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(deep_models), 8)) as ex:
        fs = {ex.submit(_retry, REG[m], _build_iv(m, iv), timeout=Config.MODEL_TIMEOUT): m for m in deep_models}
        for f in concurrent.futures.as_completed(fs, timeout=180):
            try:
                r = f.result()
                r["_k"] = key_of(r.get("tahmin"))
                r["_st"] = norm_st(r.get("st_mm"))
                oylar[fs[f]] = r
            except Exception as e:
                logging.warning("[%s] %s", fs[f], e)

    # Fallback: deep mod basarisizsa fast moda gec
    if not oylar:
        logging.warning("Derin mod basarisiz, fast moda dusuluyor")
        model_name = Config.FAST_MODELS[0] if (Config.FAST_MODELS and Config.FAST_MODELS[0] in REG) else READERS[0][0]
        res = single_analyze(iv, model_name, img_type=img_type)
        res["blur"] = {"skor": round(blur_val, 1), "durum": blur_label}
        res["kalite_uyarisi"] = (res.get("kalite_uyarisi") or "") + " Derin mod calismadi, hizli mod sonucu."
        res["sure_sn"] = round(time.time() - t0, 1)
        _result_cache[cache_key] = (time.time(), res)
        return res

    if len(oylar) < Config.MIN_AI_RESPONSES and Config.FAST_MODELS and Config.FAST_MODELS[0] in REG:
        try:
            r = _retry(REG[Config.FAST_MODELS[0]], _build_iv(Config.FAST_MODELS[0], iv), timeout=Config.MODEL_TIMEOUT)
            r["_k"] = key_of(r.get("tahmin"))
            r["_st"] = norm_st(r.get("st_mm"))
            oylar[Config.FAST_MODELS[0]] = r
        except Exception as e:
            logging.warning("Yedek model hata: %s", e)

    st_mm = medyan_st([v["_st"] for v in oylar.values()])
    karar = st_karar(st_mm)

    say = {}
    for v in oylar.values():
        say[v["_k"]] = say.get(v["_k"], 0) + 1
    # Acil olanlari onceliklendir
    cog = sorted(say.items(), key=lambda kv: (-kv[1], -(kv[0] in EMERGENCY), -(kv[0] in PREARREST)))[0][0]

    kal = [v.get("kalite") for v in oylar.values()]
    kalite = max(set(kal), key=kal.count) if kal else "orta"
    lead_ok = sum(1 for v in oylar.values() if v.get("tum_leadler")) >= len(oylar) / 2

    ot_list = []
    for k, v in oylar.items():
        ot_list.append({"ai": k, "tahmin": v.get("tahmin"), "key": v["_k"],
                        "st_mm": v["_st"], "qrs": v.get("qrs_ms"),
                        "detay": (v.get("detay") or "")[:150]})
    ot = json.dumps(ot_list, ensure_ascii=False)
    bilgi = bilgi_bankasi_getir(cog)

    # Hakem promptuna en iyi analizcinin de girebilecegini belirt
    hp = (
        "BAS HAKEMSIN. HASTANE ONCESI PARAMEDIKSIN. "
        "OBJEKTIF ST OLCUMU (AI medyani, mm):\n" + json.dumps(st_mm)
        + "\nKURAL MOTORU KARARI:\n" + json.dumps(karar, ensure_ascii=False)
        + "\nKANIT TABANLI BILGI BANKASI:\n" + bilgi
        + "\nOYLAR:\n" + ot
        + "\nKURAL: kural motoru komsu elevasyon veya ayna-depresyon bulduysa "
        + "ve goruntu dogruluyorsa bu STEMI'dir; cogunluk NORMAL/NSTEMI dese bile. "
        + "VF, VT, asistoli, PEA, semptomatik bradikardi/blok, Torsades, WPW+AF gibi "
        + "olumcul/prearrest durumlarinda guvensiz bile olsan ACIL kabul et. "
        + "Iki goruntuyu incele, KENDIN karar ver.\n"
        + DETAYLI_PROMPT + "\n" + SEMA_HAKEM
    )

    hk_ad, hk = None, None
    # Hakem olarak once en iyi analiz modelini dene
    for ad in ([BEST_ANALYSIS_MODEL] if BEST_ANALYSIS_MODEL else []) + [n for n, _ in HAKEM]:
        if ad not in REG:
            continue
        try:
            h = _retry(REG[ad], _build_iv(ad, iv), prompt=hp, timeout=Config.JUDGE_TIMEOUT)
            h["_k"] = key_of(h.get("tahmin"))
            h["_st"] = norm_st(h.get("st_mm"))
            hk_ad, hk = ad, h
            break
        except Exception as e:
            logging.warning("Hakem %s %s", ad, e)

    if hk and hk["_st"]:
        st_mm = medyan_st([st_mm, hk["_st"]])
        karar = st_karar(st_mm)

    ref = hk["_k"] if hk else cog
    v_ad, ver, internet_dogrulama_text = None, None, None
    vp = (
        "KIRMIZI EKIPSIN. HASTANE ONCESI PARAMEDIKSIN. "
        "Onceki karar: " + ref
        + ". OBJEKTIF ST: " + json.dumps(st_mm)
        + " KURAL: " + json.dumps(karar, ensure_ascii=False)
        + " BILGI BANKASI: " + bilgi_bankasi_getir(ref)
        + ". Kacirilmis STEMI/VF/asistoli/PEA/bradikardi/blok/Torsades/WPW var mi? "
        + "kritik_bulgu sadece gercek hayati bulguda true.\n"
        + DETAYLI_PROMPT + "\n" + SEMA_VER
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex2:
        fv = None
        for ad, fn in VERIF:
            if ad == hk_ad:
                continue
            fv = ex2.submit(_retry, fn, _build_iv(ad, iv), prompt=vp, timeout=Config.VERIFY_TIMEOUT)
            v_ad = ad
            break
        fi = ex2.submit(internet_dogrulama, ref, st_mm)
        if fv:
            try:
                v = fv.result()
                v["_k"] = key_of(v.get("oneri_tahmin"))
                ver = v
            except Exception as e:
                logging.warning("Verif %s %s", v_ad, e)
        try:
            internet_dogrulama_text = fi.result()
        except Exception as e:
            logging.warning("Internet dogrulama hata: %s", e)

    # Nihai karar - GUVENLIK ONCELIGI
    def _severity_rank(k):
        if k in EMERGENCY: return 3
        if k in PREARREST: return 2
        if k in ORTA: return 1
        return 0

    duz = None
    final = hk["_k"] if hk else cog

    # Olumcul tarama cok guclu ise onu dikkate al
    if lethal and lethal["olumcul"] and lethal["guven"] >= 70 and _severity_rank(lethal["tahmin_key"]) > _severity_rank(final):
        final = lethal["tahmin_key"]
        duz = f"Olumcul/prearrest tarama {final} olarak yuksek guvenle belirledi"

    # Kural motoru acil STEMI bulduysa yukselt
    for k in karar:
        if k["bolge"] in EMERGENCY and final not in EMERGENCY:
            final = k["bolge"]
            duz = f"Objektif ST kural motoru {k['bolge']} buldu -> karar buna yukseltildi"
            break

    # Kirmizi ekip kritik bulguyu dogruladiysa
    if ver and ver.get("kritik_bulgu") and final not in EMERGENCY:
        final = ver["_k"] if ver["_k"] in EMERGENCY else "ANTERIOR_MI"
        duz = f"Dogrulayici ({v_ad}) kritik bulgu -> {final}"

    # Guvenlik: adaylar arasindaki en agir taniyi tercih et
    candidates = [cog]
    if hk:
        candidates.append(hk["_k"])
    if ver and ver.get("_k"):
        candidates.append(ver["_k"])
    if lethal and lethal["tahmin_key"]:
        candidates.append(lethal["tahmin_key"])
    for c in candidates:
        if _severity_rank(c) > _severity_rank(final):
            duz = (duz or "") + f" Guvenlik onceligi: {final} -> {c} yukseltildi."
            final = c

    kalite_uyarisi = None
    if kalite == "kotu" or not lead_ok:
        kalite_uyarisi = "⚠️ EKG kalitesi yetersiz/lead eksik. Standartlara uygun yeniden cek."
    if blur_label == "kotu":
        kalite_uyarisi = (kalite_uyarisi or "") + " ⚠️ Fotograf bulanik. Net cekim yap."

    hm = (hk.get("olcumler") or {}) if hk else {}
    olc = {}
    for a in ["hiz", "pr_ms", "qrs_ms", "qtc_ms"]:
        vals = [v.get(a) for v in oylar.values()]
        olc[a] = hm.get(a) or _med(vals)

    guven = round(100 * say.get(final, 0) / len(oylar))
    if hk and hk["_k"] == final:
        guven = max(guven, 75)
    if any(k["bolge"] == final for k in karar if k["bolge"] in EMERGENCY):
        guven = min(100, guven + 10)
    if ver and (not ver.get("kritik_bulgu") or ver["_k"] == final):
        guven = min(100, guven + 5)
    if lethal and lethal["tahmin_key"] == final and lethal["guven"] >= 70:
        guven = min(100, guven + 10)
    if kalite_uyarisi:
        guven = min(guven, 50)
    if blur_label == "kotu":
        guven = min(guven, 45)

    bay = []
    qrs = olc.get("qrs_ms") or 0
    qtc = olc.get("qtc_ms") or 0
    if final == "SVT" and qrs >= 120:
        bay.append("QRS genis: VT olasiligini gozden gecir")
    if final == "AF" and all(v.get("duzenli") for v in oylar.values() if v.get("duzenli") is not None):
        bay.append("Ritim duzenli: AF'yi dogrula")
    if qtc and qtc >= 500:
        bay.append("QTc >=500 ms: Torsades riski")
    if final not in EMERGENCY and (st_mm.get("aVR") or 0) >= 1:
        bay.append("aVR elevasyonu: sol ana koroner dusun")
    if final == "NSTEMI" and any(k["bolge"] in EMERGENCY for k in karar):
        bay.append("DIKKAT: objektif ST elevasyonu var, NSTEMI degil STEMI olabilir")
    if final == "INFERIOR_MI":
        bay.append("V4R cek, sag ventrikul MI ekarte et")

    temsilci = None
    if hk and hk["_k"] == final:
        temsilci = hk
    if not temsilci:
        for v in oylar.values():
            if v["_k"] == final:
                temsilci = v
                break
    if not temsilci:
        temsilci = next(iter(oylar.values()))

    sure = round(time.time() - t0, 1)

    res = {
        "tahmin": temsilci.get("tahmin") or MANUEL_ISIMLER.get(final, final),
        "anahtar": final,
        "resmi_protokol": KEY2CODE.get(final, "-"),
        "risk_seviyesi": "Yuksek" if final in EMERGENCY else ("Orta" if (final in PREARREST or final in ORTA) else "Dusuk"),
        "acil_mudahale": final in EMERGENCY or final in PREARREST or final in ORTA,
        "detay": strip_sources((hk or {}).get("detay") or temsilci.get("detay") or ""),
        "guven": guven,
        "hakem": hk_ad,
        "dogrulayici": {"model": v_ad, "kritik": bool(ver and ver.get("kritik_bulgu")), "duzeltme": duz},
        "oylama": {k: v.get("tahmin") for k, v in oylar.items()},
        "olcumler": olc,
        "st_mm": st_mm,
        "st_karar": karar,
        "kalite": kalite,
        "kalite_uyarisi": kalite_uyarisi,
        "tutarlilik": bay,
        "saha_notlari": ACIL_NOTLAR.get(final, []),
        "arastirma": None,
        "internet_dogrulama": strip_sources(internet_dogrulama_text.get("ozet") if isinstance(internet_dogrulama_text, dict) else None),
        "internet_uyusma": internet_dogrulama_text.get("uyusma") if isinstance(internet_dogrulama_text, dict) else None,
        "bilgi_bankasi": strip_sources(bilgi_bankasi_getir(final)),
        "nihai_gerekce": strip_sources((hk or {}).get("nihai_gerekce", "")),
        "sure_sn": sure,
        "blur": {"skor": round(blur_val, 1), "durum": blur_label},
        "goruntu_turu": img_type,
        "uyari": "Klinik onay zorunludur; nihai degerlendirme hekime aittir.",
        "algoritma": temiz_algo(final),
        "tarama": screen,
        "olumcul_tarama": lethal,
        "zoom_used": zoom_used,
        "zoom_adet": len(zooms),
    }
    _result_cache[cache_key] = (time.time(), res)
    return res


# ─── VERITABANI ─────────────────────────────────────────────────────────────-
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, image_hash TEXT, filename TEXT, mode TEXT,
        tahmin TEXT, anahtar TEXT, guven INTEGER, risk TEXT, acil INTEGER,
        st_mm TEXT, st_karar TEXT, olcumler TEXT, oylama TEXT,
        dogrulayici TEXT, hakem TEXT, kalite TEXT,
        blur_skor REAL, blur_durum TEXT, detay TEXT,
        resmi_protokol TEXT, nihai_gerekce TEXT,
        internet_dogrulama TEXT, sure_sn REAL, goruntu_turu TEXT
    )''')
    conn.commit()
    conn.close()


def save_analysis(data: Dict[str, Any], image_hash: str, filename: str,
                  blur: Dict[str, Any], mode: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO analyses
            (timestamp, image_hash, filename, mode, tahmin, anahtar, guven, risk, acil,
             st_mm, st_karar, olcumler, oylama, dogrulayici, hakem, kalite,
             blur_skor, blur_durum, detay, resmi_protokol, nihai_gerekce,
             internet_dogrulama, sure_sn, goruntu_turu)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (datetime.now().isoformat(), image_hash, filename, mode,
             data.get("tahmin"), data.get("anahtar"), data.get("guven"),
             data.get("risk_seviyesi"), 1 if data.get("acil_mudahale") else 0,
             json.dumps(data.get("st_mm"), ensure_ascii=False),
             json.dumps(data.get("st_karar"), ensure_ascii=False),
             json.dumps(data.get("olcumler"), ensure_ascii=False),
             json.dumps(data.get("oylama"), ensure_ascii=False),
             json.dumps(data.get("dogrulayici"), ensure_ascii=False),
             data.get("hakem"), data.get("kalite"),
             blur.get("skor"), blur.get("durum"), data.get("detay"),
             data.get("resmi_protokol"), data.get("nihai_gerekce"),
             data.get("internet_dogrulama"), data.get("sure_sn"), data.get("goruntu_turu")))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error("VT kayit hatasi: %s", e)


def get_recent_analyses(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('''SELECT * FROM analyses ORDER BY id DESC LIMIT ?''', (limit,))
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logging.error("VT okuma hatasi: %s", e)
        return []


# ─── HTML ONYUZ (YAN YANA BILGILER) ──────────────────────────────────────────
HTML = r"""<!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Paramedik EKG Asistani v23</title><style>
:root{--bg:#0b1220;--card:rgba(255,255,255,.06);--line:rgba(255,255,255,.12);--txt:#e5e7eb;--mut:#94a3b8;--red:#ef4444;--amb:#f59e0b;--grn:#22c55e}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);min-height:100vh;padding:18px;color:var(--txt)}
.app{max-width:1300px;margin:0 auto;display:flex;flex-direction:column;gap:12px}
.glass{background:var(--card);border:1px solid var(--line);border-radius:16px;backdrop-filter:blur(10px)}
.topbar{display:flex;align-items:center;gap:14px;padding:14px 20px}
.logo{width:44px;height:44px;border-radius:14px;background:linear-gradient(135deg,#dc2626,#7f1d1d);display:grid;place-items:center;font-size:22px}
.topbar h1{font-size:19px;font-weight:800}.topbar .sub{font-size:11px;color:var(--mut)}
.chips{margin-left:auto;font-size:11px;color:#4ade80;text-align:right;max-width:55%}
.warn{padding:12px 18px;font-size:12.5px;color:#fde68a;background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.3)}
.main{padding:20px}.tabs{display:flex;gap:6px;margin-bottom:16px}
.tab{flex:1;padding:12px;border:1px solid var(--line);background:transparent;color:var(--mut);border-radius:10px;font-weight:700;font-size:13px;cursor:pointer}
.tab.active{background:linear-gradient(135deg,#dc2626,#991b1b);color:#fff;border-color:transparent}
.tcontent{display:none}.tcontent.active{display:block}
.upload{border:2px dashed rgba(239,68,68,.45);border-radius:14px;padding:26px;text-align:center;background:rgba(239,68,68,.05)}
input[type=file]{display:none}.urow{display:flex;gap:10px;justify-content:center;margin-top:14px;flex-wrap:wrap}
.btn{padding:12px 22px;border:none;border-radius:10px;font-weight:700;font-size:13px;cursor:pointer}
.btn.red{background:linear-gradient(135deg,#dc2626,#991b1b);color:#fff}
.btn.blue{background:linear-gradient(135deg,#2563eb,#1e40af);color:#fff}
.btn.green{background:linear-gradient(135deg,#16a34a,#166534);color:#fff;font-size:15px;padding:14px 28px}
.btn.gray{background:rgba(255,255,255,.1);color:var(--txt)}
.btn:disabled{opacity:.5;cursor:not-allowed}
select{width:100%;padding:13px;border-radius:10px;background:#111a2e;color:var(--txt);border:1px solid var(--line);font-size:14px}
.preview-img{max-width:100%;max-height:240px;border-radius:12px;margin:14px auto 0;display:none;border:1px solid var(--line)}
.bgroup{display:flex;gap:10px;justify-content:center;margin-top:16px;flex-wrap:wrap}
.result{display:none}.ribbon{border-radius:16px;padding:18px 22px;color:#fff;display:flex;flex-wrap:wrap;gap:16px;align-items:center;justify-content:space-between;box-shadow:0 10px 30px rgba(0,0,0,.35)}
.ribbon.r-red{background:linear-gradient(135deg,#7f1d1d,#dc2626);animation:pulse 1.6s infinite}
.ribbon.r-amb{background:linear-gradient(135deg,#78350f,#d97706)}
.ribbon.r-grn{background:linear-gradient(135deg,#14532d,#16a34a)}
@keyframes pulse{0%,100%{box-shadow:0 10px 30px rgba(220,38,38,.35)}50%{box-shadow:0 10px 44px rgba(220,38,38,.7)}}
.ribbon .dx{font-size:24px;font-weight:900}.ribbon .meta{font-size:12px;opacity:.92;margin-top:4px}
.ribbon .right{text-align:right;min-width:180px}.meter{height:8px;background:rgba(255,255,255,.28);border-radius:6px;margin-top:8px;overflow:hidden}
.meter>div{height:100%;background:#fff}.row{display:flex;gap:14px;flex-wrap:wrap;margin-top:14px}
.col{flex:1;min-width:320px;display:flex;flex-direction:column;gap:14px}
.card{padding:16px}.card .lbl{font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:var(--mut);font-weight:700;margin-bottom:9px}
.card .val{font-size:16px;font-weight:800}.card .sub{font-size:12px;color:var(--mut);margin-top:5px;line-height:1.55}
.full{grid-column:1/-1;width:100%}.mini{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.mini div{background:rgba(255,255,255,.05);border-radius:8px;padding:7px;text-align:center}
.mini b{display:block;font-size:9px;color:var(--mut);text-transform:uppercase}
.lead{display:inline-block;border-radius:8px;padding:5px 8px;margin:3px;font-size:11px;font-weight:700;border:1px solid}
.l-elev{background:rgba(239,68,68,.18);border-color:var(--red);color:#fca5a5}
.l-dep{background:rgba(245,158,11,.15);border-color:var(--amb);color:#fde68a}
.l-norm{background:rgba(34,197,94,.1);border-color:var(--grn);color:#86efac}
.karar{display:inline-block;background:rgba(239,68,68,.18);border:1px solid var(--red);border-radius:8px;padding:5px 10px;margin:3px;font-size:12px;color:#fca5a5;font-weight:700}
.oy{display:inline-block;background:rgba(255,255,255,.07);border:1px solid var(--line);border-radius:8px;padding:4px 9px;margin:3px;font-size:11px}
.not{display:inline-block;background:rgba(37,99,235,.15);border:1px solid rgba(37,99,235,.4);border-radius:8px;padding:5px 10px;margin:3px;font-size:12px;color:#bfdbfe}
.bay{display:inline-block;background:rgba(245,158,11,.15);border:1px solid rgba(245,158,11,.4);border-radius:8px;padding:5px 10px;margin:3px;font-size:12px;color:#fde68a}
.info{display:inline-block;background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.4);border-radius:8px;padding:5px 10px;margin:3px;font-size:12px;color:#86efac}
.analysis{font-size:13.5px;line-height:1.7;white-space:pre-wrap;color:#cbd5e1}
.proto{padding:22px;border-left:4px solid var(--red)}
.proto h3{color:#fca5a5;font-size:15px;margin-bottom:8px}
.proto .urg{display:inline-block;background:rgba(255,255,255,.08);border:1px solid var(--line);padding:6px 12px;border-radius:8px;font-size:12px;font-weight:700;margin-bottom:12px}
.proto pre{font-family:inherit;font-size:14px;line-height:1.85;white-space:pre-wrap}
.footer{text-align:center;font-size:11px;color:var(--mut);padding:8px}
.loading{text-align:center;padding:30px}.spinner{width:44px;height:44px;border:4px solid rgba(255,255,255,.12);border-top-color:var(--red);border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 12px}
@keyframes spin{to{transform:rotate(360deg)}}
.fade{animation:fade .4s ease}@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.mode{display:flex;gap:8px;justify-content:center;margin-bottom:14px;flex-wrap:wrap}
.mode label{font-size:12px;color:var(--mut);display:flex;align-items:center;gap:4px;cursor:pointer}
.history-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
.history-table th,.history-table td{padding:8px;border-bottom:1px solid var(--line);text-align:left}
.history-table th{color:var(--mut);text-transform:uppercase;font-size:10px}
@media(max-width:800px){.col{min-width:100%}}
</style></head><body><div class="app">
<div class="glass topbar"><div class="logo">🫀</div><div><h1>Paramedik EKG Asistani</h1>
<div class="sub">Hastane Oncesi · MI/Ritim Tani · SB-ASH-Y Protokol · Coklu AI · Internet Dogrulama</div></div>
<div class="chips" id="aiChips"></div></div>
<div class="glass warn">⚠️ <b>Klinik Uyari:</b> Bu sistem 112 paramedik yetkileri cercevesinde karar destegidir. Nihai karar hekime aittir.</div>
<div class="glass main">
<div class="tabs">
<button class="tab active" id="t1" onclick="stab(1)">📸 Fotoğraf Analizi</button>
<button class="tab" id="t2" onclick="stab(2)">🎯 Manuel Seçim</button>
<button class="tab" id="t3" onclick="stab(3)">📊 Son Analizler</button>
</div>
<div class="tcontent active" id="c1">
<div class="mode">
<label><input type="radio" name="mode" value="auto" checked> ⚡ Otomatik</label>
<label><input type="radio" name="mode" value="fast"> 🚀 Hizli</label>
<label><input type="radio" name="mode" value="deep"> 🔬 Derin</label>
<label><input type="checkbox" id="gridClean" checked> 🔲 Grid Temizle</label>
</div>
<div class="upload"><div style="font-size:42px">📸</div>
<div style="font-size:16px;font-weight:600;margin-top:6px">EKG Kagidi veya Monitor Goruntusu Yukleyin</div>
<div style="font-size:12px;color:var(--mut);margin-top:4px">Bulaniklik kontrolu · Grid temizligi · Kural motoru · Hakem · Internet dogrulama</div>
<div class="urow">
<button class="btn red" onclick="fg.click()">📁 Galeriden Sec</button>
<button class="btn blue" onclick="fc.click()">📷 Kamera ile Cek</button>
</div>
<input type="file" id="fg" accept="image/*">
<input type="file" id="fc" accept="image/*" capture="environment"></div>
<img id="pv" class="preview-img" alt="">
<div class="bgroup" id="bg" style="display:none">
<button class="btn green" id="ab">🧠 EKG'yi Analiz Et</button>
<button class="btn gray" id="rb">↻ Temizle</button></div>
</div>
<div class="tcontent" id="c2">
<div class="card glass" style="margin-bottom:14px"><div class="lbl">🎯 EKG Ritimi / Tanisi Secin</div>
<select id="ritmSelect"><option value="">-- Bir ritim secin --</option>
<optgroup label="🚨 Kalp Krizleri">
<option value="ANTERIOR_MI">Anterior STEMI</option>
<option value="INFERIOR_MI">Inferior STEMI</option>
<option value="LATERAL_MI">Lateral STEMI</option>
<option value="POSTERIOR_MI">Posterior STEMI</option>
<option value="SAG_V_MI">Sag Ventrikul MI</option>
<option value="YAYGIN_ANTERIOR_MI">Yaygin Anterior MI</option>
<option value="NSTEMI">NSTEMI / USAP</option></optgroup>
<optgroup label="⚠️ Aritmiler / Prearrest">
<option value="AF">AF</option><option value="SVT">SVT</option>
<option value="VT">VT</option><option value="VF">VF (Arrest)</option>
<option value="PEA">PEA</option><option value="TORSADES">Torsades</option>
<option value="BRADIKARDI">Bradikardi</option><option value="AV_BLOK">AV Blok</option>
<option value="WPW">WPW</option><option value="ASISTOLI">Asistoli (Arrest)</option></optgroup>
<optgroup label="✅ Diger">
<option value="NORMAL">Normal Sinuz Ritmi</option></optgroup>
</select></div>
<div class="bgroup"><button class="btn green" id="mb">📖 Paramedik Protokolunu Goster</button></div></div>
<div class="tcontent" id="c3">
<div class="card glass"><div class="lbl">📊 Son Kaydedilen Analizler</div>
<div id="historyArea"><p style="color:var(--mut);font-size:13px">Yukleniyor...</p></div></div></div>
<div class="result" id="rs"><div id="rc"></div></div></div>
<div class="footer">Paramedik EKG Asistani · Hastane Oncesi Karar Destegi</div></div>
<script>
var fg=document.getElementById('fg'),fc=document.getElementById('fc'),pv=document.getElementById('pv'),bg=document.getElementById('bg'),ab=document.getElementById('ab'),rb=document.getElementById('rb'),rs=document.getElementById('rs'),rc=document.getElementById('rc'),mb=document.getElementById('mb');
var sel=null;
fetch('/health').then(r=>r.json()).then(h=>{document.getElementById('aiChips').innerHTML='🟢 '+(h.okuyucular||[]).join(' · ')+'<br>📷 Gorsel: '+(h.best_vision||'-')+' · 🧠 Analiz: '+(h.best_analysis||'-');}).catch(()=>{});
function stab(n){['t1','t2','t3'].forEach((id,i)=>{document.getElementById(id).classList.toggle('active',n===(i+1));});['c1','c2','c3'].forEach((id,i)=>{document.getElementById(id).classList.toggle('active',n===(i+1));});rs.style.display='none';if(n===3)loadHistory();}
function hf(e){var f=e.target.files[0];if(!f)return;sel=f;var r=new FileReader();r.onload=function(ev){pv.src=ev.target.result;pv.style.display='block';bg.style.display='flex';rs.style.display='none';};r.readAsDataURL(f);}
fg.onchange=hf;fc.onchange=hf;
rb.onclick=function(){fg.value='';fc.value='';sel=null;pv.style.display='none';bg.style.display='none';rs.style.display='none';};
function getMode(){return document.querySelector('input[name="mode"]:checked').value;}
function getGrid(){return document.getElementById('gridClean').checked;}
ab.onclick=function(){run('/api/analyze?mode='+getMode()+'&grid='+getGrid(),function(){var fd=new FormData();fd.append('file',sel);return fd;});};
mb.onclick=function(){var s=document.getElementById('ritmSelect').value;if(!s)return alert('Ritim secin!');run('/api/manual/'+s,null);};
function run(url,mkfd){ab.disabled=true;mb.disabled=true;rs.style.display='block';rc.innerHTML='<div class="loading"><div class="spinner"></div><p style="color:var(--mut)">AI\'lar mm olcuyor → Grid temizleniyor → Kural motoru → Hakem → Internet dogrulama...</p></div>';var opt=mkfd?{method:'POST',body:mkfd()}:{};fetch(url,opt).then(r=>r.json()).then(r=>{if(r.status==='success')show(r.prediction);else rc.innerHTML='<div class="glass card" style="border-left:4px solid var(--red)"><b>Hata:</b> '+r.message+'</div>';}).catch(e=>{rc.innerHTML='<div class="glass card" style="border-left:4px solid var(--red)"><b>Hata:</b> '+e.message+'</div>';}).finally(()=>{ab.disabled=false;mb.disabled=false;});}
function leadCls(v){if(v>=1)return 'l-elev';if(v<=-0.5)return 'l-dep';return 'l-norm';}
function fmt(v){if(v==null)return '-';return (v>0?'+':'')+v;}
function loadHistory(){fetch('/api/history').then(r=>r.json()).then(d=>{var area=document.getElementById('historyArea');if(d.status!=='success'){area.innerHTML='<p>Hata</p>';return;}var rows=d.data||[];if(!rows.length){area.innerHTML='<p style="color:var(--mut)">Henuz kayit yok.</p>';return;}var h='<table class="history-table"><tr><th>Zaman</th><th>Tani</th><th>Risk</th><th>Guven</th><th>Mod</th><th>Tur</th><th>Bulaniklik</th><th>Sure</th></tr>';rows.forEach(function(row){h+='<tr><td>'+row.timestamp.substring(0,19).replace('T',' ')+'</td><td><b>'+row.tahmin+'</b></td><td>'+row.risk+'</td><td>%'+row.guven+'</td><td>'+row.mode+'</td><td>'+(row.goruntu_turu||'-')+'</td><td>'+(row.blur_durum||'-')+' ('+(row.blur_skor||0)+')</td><td>'+row.sure_sn+' sn</td></tr>';});h+='</table>';area.innerHTML=h;}).catch(e=>{document.getElementById('historyArea').innerHTML='<p style="color:#fca5a5">'+e.message+'</p>';});}
function show(p){var o=p.olcumler||{},dg=p.dogrulayici||{},sm=p.st_mm||{},bl=p.blur||{};var hasOyl=Object.keys(p.oylama||{}).length>0;var rc_=p.risk_seviyesi==='Yuksek'?'r-red':((p.risk_seviyesi==='Orta'||p.acil_mudahale)?'r-amb':'r-grn');var taramaTxt=p.tarama?(' · Tarama: '+p.tarama.acil+(p.tarama.tahmin?' / '+p.tarama.tahmin:'')):'';var lethalTxt=(p.olumcul_tarama&&p.olumcul_tarama.olumcul)?(' · 🚨 Olumcul tarama: '+p.olumcul_tarama.tahmin):'';var typeTxt=p.goruntu_turu?' · Tur: '+p.goruntu_turu:'';var zoomTxt=p.zoom_used?(' · 🔍 Zoom: '+p.zoom_adet+' bolge'):'';
var h='<div class="fade"><div class="ribbon '+rc_+'"><div>'+'<div class="dx">'+(p.acil_mudahale?'🚨':'✅')+' '+p.tahmin+'</div>'+'<div class="meta">📋 '+p.resmi_protokol+' · '+p.risk_seviyesi+' risk'+(p.sure_sn?' · '+p.sure_sn+' sn':'')+(p.hakem?' · Hakem: '+p.hakem:'')+(dg.model?' · Kirmizi Ekip: '+dg.model+(dg.kritik?' 🚨':''):'')+taramaTxt+lethalTxt+typeTxt+zoomTxt+'</div>'+(dg.duzeltme?'<div class="meta">⚖ '+dg.duzeltme+'</div>':'')+'</div><div class="right"><div style="font-size:12px">GUVEN SKORU</div>'+'<div style="font-size:24px;font-weight:900">%'+(p.guven||0)+'</div>'+'<div class="meter"><div style="width:'+(p.guven||0)+'%"></div></div></div></div>';
h+='<div class="row">';
// Sol kolon
h+='<div class="col">';
h+='<div class="glass card"><div class="lbl">🧮 Objektif ST Haritasi (AI medyani, mm)</div><div>';
["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"].forEach(function(L){var v=sm[L];h+='<span class="lead '+(v==null?'l-norm':leadCls(v))+'">'+L+' '+fmt(v)+'</span>';});
h+='</div>';
if(p.st_karar&&p.st_karar.length){h+='<div style="margin-top:8px">';p.st_karar.forEach(function(k){h+='<span class="karar">⚡ '+k.bolge.replace(/_/g,' ')+' · '+k.leadler.join(',')+' · '+k.tip+(k.resiprokal?' +resiprokal':'')+'</span>';});h+='</div>';}
else h+='<div class="sub" style="margin-top:6px">Kural motoru: ST kriteri karsilanmadi</div>';
h+='</div>';
if(p.internet_dogrulama){h+='<div class="glass card"><div class="lbl">🌐 Internet Dogrulamasi'+(p.internet_uyusma!=null?' (uyusma %'+p.internet_uyusma+')':'')+'</div><div class="sub">'+p.internet_dogrulama+'</div></div>';}
if(p.bilgi_bankasi)h+='<div class="glass card"><div class="lbl">📚 Klinik Ozet</div><div class="sub">'+p.bilgi_bankasi+'</div></div>';
if(p.detay)h+='<div class="glass card"><div class="lbl">📝 EKG Degerlendirmesi</div><div class="analysis">'+p.detay+'</div>'+(p.nihai_gerekce?'<div class="sub" style="margin-top:8px">⚖ '+p.nihai_gerekce+'</div>':'')+'</div>';
h+='</div>';
// Sag kolon
h+='<div class="col">';
h+='<div class="glass card"><div class="lbl">📐 Olcumler</div><div class="mini"><div><b>Hiz</b>'+(o.hiz||'-')+'</div><div><b>PR</b>'+(o.pr_ms||'-')+'</div><div><b>QRS</b>'+(o.qrs_ms||'-')+'</div><div><b>QTc</b>'+(o.qtc_ms||'-')+'</div></div></div>';
h+='<div class="glass card"><div class="lbl">🖼️ Kalite</div><div class="val">'+(p.kalite||'-')+'</div><div class="sub">Bulaniklik: '+(bl.durum||'-')+' ('+(bl.skor||0)+')</div>'+(p.kalite_uyarisi?'<div class="sub" style="color:#fca5a5">'+p.kalite_uyarisi+'</div>':'')+'</div>';
if(hasOyl){h+='<div class="glass card"><div class="lbl">🗳️ AI Oylari</div><div>';for(var k in p.oylama)h+='<span class="oy">'+k+' → <b>'+p.oylama[k]+'</b></span>';h+='</div></div>';}
if((p.saha_notlari&&p.saha_notlari.length)||(p.tutarlilik&&p.tutarlilik.length)){h+='<div class="glass card"><div class="lbl">🚑 HASTANE ONCESI SAHA NOTLARI / UYARILAR</div><div>';(p.saha_notlari||[]).forEach(function(n){h+='<span class="not">📌 '+n+'</span>';});(p.tutarlilik||[]).forEach(function(n){h+='<span class="bay">⚠ '+n+'</span>';});h+='</div></div>';}
h+='</div>';
h+='</div>';
h+='<div class="glass proto" style="margin-top:14px"><h3>🚑 PARAMEDIK MUDAHALE PROTOKOLU (Hastane Oncesi)</h3><div class="urg">'+(p.algoritma&&p.algoritma.aciliyeti||'')+'</div><pre>'+(p.algoritma&&p.algoritma.algoritma||'')+'</pre></div></div>';
rc.innerHTML=h;}
</script></body></html>"""


# ─── FASTAPI UCLARI ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Paramedik EKG", version="23.0")

init_db()


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HTML)


@app.get("/health")
async def health():
    return {
        "okuyucular": [a for a, _ in READERS],
        "best_vision": BEST_VISION_MODEL,
        "best_analysis": BEST_ANALYSIS_MODEL,
        "hakem": HAKEM[0][0] if HAKEM else None,
        "modlar": ["auto", "fast", "deep"],
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), mode: str = "auto", grid: bool = True):
    try:
        c = await file.read()
        if not c:
            raise HTTPException(400, "Dosya bos")
        if mode not in ("auto", "fast", "deep"):
            mode = "auto"
        img_hash = _hash(c)
        p = ensemble_analyze(c, mode=mode, grid_clean=grid)
        save_analysis(p, img_hash, file.filename or "unknown", p.get("blur", {}), mode)
        print(f"✓ [{mode}] {p['tahmin']} | {p['resmi_protokol']} | %{p['guven']} | ST:{len(p['st_karar'])} | {p['sure_sn']} sn | tur:{p['goruntu_turu']}")
        return {"status": "success", "prediction": p}
    except Exception as e:
        logging.error("Analiz hatasi: %s", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/manual/{kod}")
async def manual(kod: str):
    if kod not in TEDAVI_ALGORITMALARI:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Bulunamadi"})
    a = temiz_algo(kod)
    return {"status": "success", "prediction": {
        "tahmin": MANUEL_ISIMLER.get(kod, kod),
        "risk_seviyesi": "Manuel",
        "acil_mudahale": "ACIL" in a["aciliyeti"] or "KIRMIZI" in a["aciliyeti"] or "ARREST" in a["aciliyeti"],
        "detay": "", "uyari": "Ogrenme modu.",
        "resmi_protokol": KEY2CODE.get(kod, "-"),
        "olcumler": {}, "st_mm": {}, "st_karar": [],
        "saha_notlari": ACIL_NOTLAR.get(kod, []),
        "tutarlilik": [], "kalite": "-", "blur": {"skor": 0, "durum": "-"},
        "arastirma": None, "internet_dogrulama": None, "internet_uyusma": None,
        "bilgi_bankasi": strip_sources(bilgi_bankasi_getir(kod)),
        "nihai_gerekce": "",
        "sure_sn": 0, "dogrulayici": {}, "oylama": {},
        "guven": 100, "algoritma": a, "goruntu_turu": "-"}}


@app.get("/api/history")
async def history(limit: int = 50):
    try:
        rows = get_recent_analyses(limit)
        return {"status": "success", "count": len(rows), "data": rows}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


if __name__ == "__main__":
    import uvicorn
    print("📍 http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
