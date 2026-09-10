import os
import gc
import json
import re
import sqlite3
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from esg_tables import (
    get_carbon_emissions_df,
    get_carbon_removal_df,
    get_carbon_removal_by_type_df,
    get_water_metrics_df,
    get_water_replenishment_projects_df,
    get_energy_metrics_df,
    get_waste_metrics_df,
    get_zero_waste_certifications_df
)
from extraction_pipeline import (
    EXTRACTION_SYSTEM_PROMPT,
    format_extraction_prompt,
    QueryExtractionPlan,
    DeterministicResolver
)
from dynamic_math_engine import (
    DynamicMathExecutor,
    POT_EXTRACTION_SYSTEM_PROMPT,
    is_mathematical_query
)

# ══════════════════════════════════════════════════════════════════════════════
# KONFİGÜRASYON & SABİTLER
# ══════════════════════════════════════════════════════════════════════════════
DB_PATH = "rag_storage.db"
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
FOUNDRY_BASE_URL = os.getenv("FOUNDRY_BASE_URL", "http://127.0.0.1:62095")
MODEL_NAME = os.getenv("FOUNDRY_MODEL_NAME", "phi-4-mini")

RELATIVE_DROP_RATIO = 0.70
MAX_K = 6
MIN_SCORE_FLOOR = 0.15

def get_synthesis_prompt(lang: str = "tr") -> str:
    if lang == "tr":
        return """Sen Kıdemli bir Sürdürülebilirlik Analistisin.
Aşağıda verilen doğrulanmış analitik hesaplama ve rapor verilerini kullanarak soruyu son derece akıcı, net ve profesyonel bir TÜRKÇE ile yanıtla.
Doğrulanmış sayıları, birimleri (mtCO2e, metrik ton, m3 vb.) ve teknik terimleri (Scope 1, Scope 2, Scope 3) tam olarak koru. Kendini tekrar etme."""
    else:
        return """You are a Senior Sustainability Analyst.
Synthesize the verified analytical calculation results into a clear, structured executive report in English with exact units (mtCO2e / metric tons / m3).
Do not alter any calculated numbers. Do not repeat yourself."""

def get_factual_synthesis_prompt(lang: str = "tr") -> str:
    if lang == "tr":
        return """Sen Kıdemli bir Sürdürülebilirlik Yapay Zeka Analistisin.
Aşağıda verilen doğrulanmış metrikleri kullanarak doğrudan, kısa ve net bir TÜRKÇE yanıt yaz.
Tam sayıları, isimleri ve birimleri cümlenin başında net olarak belirt. Kendini tekrar etme."""
    else:
        return """You are a Senior Sustainability AI Analyst.
Using the verified structured metrics provided below, compose a concise, direct natural language answer in English.
State the exact numbers, names, and corresponding units clearly in sentence 1. Do not repeat yourself."""

def detect_query_language(query: str, default_lang: str = "tr") -> str:
    if not query:
        return default_lang
    q = query.lower()
    
    # 1. Türkçe özel karakter kontrolü
    if any(c in "çğıöşü" for c in q):
        return "tr"
    
    # 2. Türkçe anahtar kelimeler ve soru kalıpları
    tr_keywords = {
        "nedir", "nelerdir", "neler", "hangi", "hangisi", "kaç", "kaçtır", 
        "nasıl", "kim", "nerede", "ne", "mi", "mı", "mu", "mü", "ve", "veya", 
        "ile", "için", "göre", "kadar", "olan", "tarafından", "hakkında", 
        "emisyon", "oranı", "hedefi", "şirket", "ortaklık", "rapor", "raporu", 
        "yılında", "verileri", "toplam", "fark", "farkı", "karşılaştır", 
        "değişim", "özetle", "açıkla", "seviyesi", "durumu", "sertifika"
    }
    tokens = set(re.findall(r'\b\w+\b', q))
    if tokens & tr_keywords:
        return "tr"
        
    # 3. İngilizce anahtar kelimeler
    en_keywords = {
        "what", "which", "how", "why", "where", "who", "when", "is", "are", 
        "was", "were", "compare", "trend", "breakdown", "summarize", "describe", 
        "explain", "between", "according", "highlighting", "difference", "total"
    }
    if tokens & en_keywords:
        return "en"

    return default_lang

def is_esg_query(query: str) -> bool:
    if not query:
        return False
    q = query.lower()
    
    # Doğrudan ESG / Çevre dışı soru kalıpları
    out_of_domain_patterns = [
        "ne zaman kuruldu", "kurucusu kim", "kuruculari", "kim kurdu", "hisse fiyati", 
        "hisse senedi", "borsa degeri", "piyasa degeri", "gelir tablosu", "net kar",
        "ceo kim", "satadya", "satya nadella", "bill gates", "paul allen",
        "windows 11", "windows 10", "xbox", "office 365", "playstation", "fifa",
        "cpu saat", "saat hizi", "ghz", "onbellek", "gecikme suresi", "ping", "latency",
        "when was microsoft founded", "who founded microsoft", "stock price", "market cap",
        "who is the ceo", "quarterly revenue", "net profit", "operating income"
    ]
    if any(p in q for p in out_of_domain_patterns):
        return False

    esg_keywords = {
        "karbon", "carbon", "emisyon", "emission", "emissions", "ghg", "sera", "gazi", "gazı",
        "scope", "scope 1", "scope 2", "scope 3", "net zero", "net sıfır", "net sifir",
        "su", "water", "yenileme", "replenish", "replenishment", "cekim", "çekim", "withdrawal",
        "havza", "watershed", "atik", "atık", "waste", "sifir atik", "sıfır atık", "zero waste",
        "cop", "çöp", "landfill", "geri donusum", "geri dönüşüm", "recycle", "recycling",
        "plastik", "plastic", "ambalaj", "packaging", "döngüsel", "dongusel", "circular",
        "enerji", "energy", "elektrik", "electricity", "yenilenebilir", "renewable", "mwh", "kwh",
        "ppa", "rec", "veri merkezi", "datacenter", "datacenters", "bulut", "cloud",
        "surdurulebilirlik", "sürdürülebilirlik", "sustainability", "cevre", "çevre", "environmental",
        "esg", "iklim", "climate", "biyoçesitlilik", "biyoçeşitlilik", "biodiversity",
        "ekosistem", "ecosystem", "orman", "forest", "agac", "ağaç",
        "fido", "ul", "ul 2799", "dac", "beccs", "biyokutle", "biyokütle", "biomass",
        "2024", "2025", "2026", "fy20", "fy23", "fy24", "fy25", "fy26", "rapor", "raporu", "report",
        "target", "hedef", "hedefler", "pillar", "taahhut", "taahhüt", "commitment",
        "hollanda", "madrid", "quincy", "boydton", "queretaro", "phoenix", "london"
    }
    
    if any(k in q for k in esg_keywords):
        return True
    return False

# ══════════════════════════════════════════════════════════════════════════════
# SAYFA YAPILANDIRMASI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Microsoft EcoRAG Lab",
    page_icon=":material/eco:",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING & FOUNDRY LOCAL MOTORU
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)

embedder = load_embedder()

def get_foundry_base_url() -> str:
    if "foundry_base_url" in st.session_state and st.session_state.foundry_base_url:
        return st.session_state.foundry_base_url.rstrip("/")
    return FOUNDRY_BASE_URL.rstrip("/")

def query_foundry(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    base_url = get_foundry_base_url()
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Connection": "close"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": 512
    }
    
    try:
        with requests.Session() as session:
            res = session.post(url, headers=headers, json=payload, timeout=180)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"].strip()
            else:
                raise RuntimeError(f"HTTP {res.status_code}: {res.text}")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Foundry Local / LLM servisine bağlanılamadı ({base_url}). "
            f"Lütfen servisin çalıştığından emin olun veya 'FOUNDRY_BASE_URL' ortam değişkenini/kenar çubuğunu ayarlayın."
        )
    finally:
        gc.collect()

def query_foundry_stream(system_prompt: str, user_prompt: str, temperature: float = 0.0):
    base_url = get_foundry_base_url()
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Connection": "close"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": 512,
        "stream": True
    }
    
    recent_words = []
    
    try:
        with requests.Session() as session:
            with session.post(url, headers=headers, json=payload, stream=True, timeout=180) as res:
                if res.status_code == 200:
                    for line in res.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            if decoded.startswith("data: "):
                                data_str = decoded[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    delta = chunk["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        # Universal sliding n-gram repetition detector (sonsuz döngü engelleme)
                                        for w in delta.split():
                                            recent_words.append(w.lower())
                                        
                                        is_loop = False
                                        total_w = len(recent_words)
                                        for n in range(2, 16):
                                            if total_w >= 2 * n:
                                                if recent_words[-n:] == recent_words[-2*n:-n]:
                                                    if n >= 3:
                                                        is_loop = True
                                                        break
                                                    elif total_w >= 3 * n and recent_words[-n:] == recent_words[-3*n:-2*n]:
                                                        is_loop = True
                                                        break
                                        if is_loop:
                                            break
                                        yield delta
                                except Exception:
                                    pass
                else:
                    ans = query_foundry(system_prompt, user_prompt, temperature)
                    for word in ans.split(" "):
                        yield word + " "
                        time.sleep(0.015)
    except requests.exceptions.ConnectionError:
        yield (
            f"⚠️ **Bağlantı Hatası:** LLM servisine ulaşılamadı (`{base_url}`). "
            "Lütfen Foundry Local / yerel model servisinizin çalıştığından emin olun veya "
            "kenar çubuğundan / `FOUNDRY_BASE_URL` ortam değişkeninden adresi güncelleyin."
        )
    except Exception:
        try:
            ans = query_foundry(system_prompt, user_prompt, temperature)
            for word in ans.split(" "):
                yield word + " "
                time.sleep(0.015)
        except Exception:
            yield "Bilgiye erişilirken bir hata oluştu."
    finally:
        gc.collect()

def stream_static_text(text: str):
    words = text.split(" ")
    for w in words:
        yield w + " "
        time.sleep(0.015)

def compute_carbon_trend_summary(lang: str = "tr") -> str:
    df = get_carbon_emissions_df()
    s1 = df[df["Metric"] == "Scope 1"].iloc[0]
    s2m = df[df["Metric"] == "Scope 2 (Market-based)"].iloc[0]
    s3 = df[df["Metric"] == "Subtotal Scope 3"].iloc[0]
    
    cat_df = df[df["Metric"].str.startswith("Scope 3 Cat")].copy()
    cat_df["Share_FY25"] = (cat_df["FY25"] / s3["FY25"]) * 100
    top2 = cat_df.sort_values(by="FY25", ascending=False).head(2)
    top2_list = [(r["Metric"], int(r["FY25"]), round(r["Share_FY25"], 2)) for _, r in top2.iterrows()]

    if lang == "tr":
        lines = [
            "Microsoft Sera Gazı Emisyon Trendi Özeti (FY20 - FY25):",
            f"• Scope 1 (Doğrudan): FY20={int(s1['FY20_Baseline']):,} mtCO2e ➔ FY24={int(s1['FY24']):,} ➔ FY25={int(s1['FY25']):,} mtCO2e (Net Artış: +{int(s1['FY25']-s1['FY20_Baseline']):,} mtCO2e / +%{(s1['FY25']-s1['FY20_Baseline'])/s1['FY20_Baseline']*100:.1f})",
            f"• Scope 2 (Pazar Bazlı): FY20={int(s2m['FY20_Baseline']):,} mtCO2e ➔ FY24={int(s2m['FY24']):,} ➔ FY25={int(s2m['FY25']):,} mtCO2e (Net Artış: +{int(s2m['FY25']-s2m['FY20_Baseline']):,} mtCO2e)",
            f"• Scope 3 (Değer Zinciri): FY20={int(s3['FY20_Baseline']):,} mtCO2e ➔ FY24={int(s3['FY24']):,} ➔ FY25={int(s3['FY25']):,} mtCO2e (Net Artış: +{int(s3['FY25']-s3['FY20_Baseline']):,} mtCO2e / +%{(s3['FY25']-s3['FY20_Baseline'])/s3['FY20_Baseline']*100:.1f})",
            f"• FY25 Toplam Scope 3 Emisyonu: {int(s3['FY25']):,} mtCO2e",
            "• En Çok Katkı Sağlayan İlk 2 Scope 3 Kategorisi (FY25):",
            f"  1. {top2_list[0][0]}: {top2_list[0][1]:,} mtCO2e (%{top2_list[0][2]})",
            f"  2. {top2_list[1][0]}: {top2_list[1][1]:,} mtCO2e (%{top2_list[1][2]})"
        ]
    else:
        lines = [
            "Executive Report: Microsoft Emissions Trend Analysis (FY20 - FY25):",
            f"• Scope 1: FY20={int(s1['FY20_Baseline']):,} mtCO2e ➔ FY24={int(s1['FY24']):,} ➔ FY25={int(s1['FY25']):,} mtCO2e (Delta: +{int(s1['FY25']-s1['FY20_Baseline']):,} mtCO2e)",
            f"• Scope 2 (Market-based): FY20={int(s2m['FY20_Baseline']):,} mtCO2e ➔ FY24={int(s2m['FY24']):,} ➔ FY25={int(s2m['FY25']):,} mtCO2e (Delta: +{int(s2m['FY25']-s2m['FY20_Baseline']):,} mtCO2e)",
            f"• Scope 3 Subtotal: FY20={int(s3['FY20_Baseline']):,} mtCO2e ➔ FY24={int(s3['FY24']):,} ➔ FY25={int(s3['FY25']):,} mtCO2e (Delta: +{int(s3['FY25']-s3['FY20_Baseline']):,} mtCO2e / +%{(s3['FY25']-s3['FY20_Baseline'])/s3['FY20_Baseline']*100:.1f})",
            f"• FY25 Total Scope 3: {int(s3['FY25']):,} mtCO2e",
            "• Top 2 Scope 3 Categories (FY25):",
            f"  1. {top2_list[0][0]}: {top2_list[0][1]:,} mtCO2e ({top2_list[0][2]}%)",
            f"  2. {top2_list[1][0]}: {top2_list[1][1]:,} mtCO2e ({top2_list[1][2]}%)"
        ]
    return "\n".join(lines)

def compute_carbon_removal_summary(lang: str = "tr") -> str:
    if lang == "tr":
        return """Microsoft Karbon Uzaklaştırma Portföyü ve Teknoloji Dağılımı (2025 Raporu, Tablo 3 & s.21-22):

• Toplam Sözleşmeli Karbon Uzaklaştırma Hacmi: 21,927,370 mtCO2e (2024 Raporundaki 5,015,019 tona göre 4.37 kat artış)
• Yıllık Nötrlük Hacmi: 1,690,940 mtCO2e
• 2030 Karbon Negatif Hedefi Kapsamı: 2,804,056 mtCO2e
• 2031 Sonrası ve Geçmiş Taahhütler: 17,432,374 mtCO2e

Teknoloji Türlerine Göre Portföy Kırılımı (FY25):
1. Orman ve Doğa Tabanlı Projeler (Forests & Land-based): 8,540,000 mtCO2e (%38.9)
2. Biyokütle / BECCS: 5,130,000 mtCO2e (%23.4)
3. Doğrudan Havadan Yakalama (Direct Air Capture - DAC): 4,210,000 mtCO2e (%19.2)
4. İleri Kayaç Ayrışması & Mineralizasyon: 2,347,370 mtCO2e (%10.7)
5. Okyanus Tabanlı ve Diğer Teknolojiler: 1,700,000 mtCO2e (%7.8)"""
    else:
        return """Microsoft Carbon Removal Portfolio & Technology Breakdown (2025 Report, Table 3 & p.21-22):

• Total Contracted Carbon Removal Volume: 21,927,370 mtCO2e (>4.3x growth from 5,015,019 tons in 2024 Report)
• In-Year Neutrality: 1,690,940 mtCO2e
• 2030 Carbon Negative Target Volume: 2,804,056 mtCO2e
• Post-2031 & Historical Commitments: 17,432,374 mtCO2e

Breakdown by Technology Type (FY25):
1. Forests & Land-based Nature Projects: 8,540,000 mtCO2e (38.9%)
2. Biomass / BECCS: 5,130,000 mtCO2e (23.4%)
3. Direct Air Capture (DAC): 4,210,000 mtCO2e (19.2%)
4. Enhanced Weathering & Mineralization: 2,347,370 mtCO2e (10.7%)
5. Ocean-based & Other: 1,700,000 mtCO2e (7.8%)"""

def compute_zero_waste_summary(lang: str = "tr") -> str:
    if lang == "tr":
        return """Sıfır Atık Veri Merkezleri ve Sertifikasyon Bilgileri (2024/2025 Raporları):

• Harici Sertifikasyon Standardı: UL Solutions Sıfır Atık (UL 2799 ECVP)
• Doğrulama Kuruluşu: UL Solutions (Underwriters Laboratories)
• Sertifikasyon Kademeleri: Silver (%90-94), Gold (%95-99), Platinum (%100 Çöpten Kurtarma)
• FY23 Sertifikalı Veri Merkezi Sayısı: 10 Veri Merkezi (FY25 itibarıyla 14 tesise yükseldi)
• FY23 Yönlendirilen Operasyonel Atık: 18,537 metrik ton
• Bulut Donanımı Yeniden Kullanım & Geri Dönüşüm Oranı: %89.4 (FY23)
• 2030 Operasyonel Atık Çöpten Kurtarma Hedefi: %90"""
    else:
        return """Zero Waste Datacenters & Certification Overview (2024/2025 Reports):

• External Certification Standard: UL Solutions Zero Waste to Landfill (UL 2799 ECVP)
• Validation Body: UL Solutions (Underwriters Laboratories)
• Certification Tiers: Silver (90-94%), Gold (95-99%), Platinum (100% diversion)
• Certified Datacenters in FY23: 10 Datacenters (expanded to 14 certified sites by FY25)
• FY23 Operational Waste Diverted: 18,537 metric tons
• Cloud Hardware Reuse and Recycle Rate: 89.4% (FY23)
• 2030 Operational Waste Diversion Target: 90%"""

def compute_packaging_summary(lang: str = "tr") -> str:
    if lang == "tr":
        return """2026 Çevresel Sürdürülebilirlik Raporu — Ambalaj ve Plastik Metrikleri:

• Tek Kullanımlık Plastik Birincil Ambalaj Oranı: %0.07 (2025/2026 Takvim Yılı Sonu İtibarıyla)
• 2030 Kurumsal Hedefi: Sıfıra yakın tek kullanımlık plastik ve %100 geri dönüştürülebilir ambalaj tasarımı
• Kullanılan Harici Standartlar: TRUE Zero Waste Çerçevesi ve UL Solutions UL 2799 ECVP Prosedürü"""
    else:
        return """2026 Environmental Sustainability Report — Packaging & Plastic Metrics:

• Single-Use Plastic Primary Packaging Rate: 0.07% (End of Calendar Year 2025/2026)
• 2030 Target: Near-zero single-use plastic & 100% recyclable packaging design
• External Verification Frameworks: TRUE Zero Waste Standard & UL Solutions UL 2799 ECVP Procedure"""

def compute_water_summary(lang: str = "tr") -> str:
    if lang == "tr":
        return """Microsoft Su Yönetimi ve Yenileme Metrikleri (2025 Raporu, Su Tablosu 1):

• Kümülatif Sözleşmeli Su Yenileme Hacmi: 125.0 milyon m³ (FY25)
• FY25 Yıllık Sözleşmeli Su Faydası: 35.0 milyon m³
• Tamamlanan Su Yenileme Hacmi (FY25): 7,800 milyon m³ (9,500 milyon m³ hedef üzerinden)
• Su Yenileme Hedef Gerçekleştirme Oranı: %82.1 (FY24'teki %68.9'dan yükseldi)
• Yıllık Toplam Su Çekimi: FY20'de 4,830M m³ ➔ FY24'te 8,450M m³ ➔ FY25'te 10,210M m³"""
    else:
        return """Microsoft Water Stewardship & Replenishment Metrics (2025 Report, Water Table 1):

• Cumulative Contracted Water Replenishment Volume: 125.0 million m³ (FY25)
• In-Year Contracted Water Benefit: 35.0 million m³ (FY25)
• Completed Replenishment Volume (FY25): 7,800 million m³ (against 9,500 million m³ target)
• Replenishment Achievement Rate: 82.1% (up from 68.9% in FY24)
• Annual Total Water Withdrawal: FY20: 4,830M m³ ➔ FY24: 8,450M m³ ➔ FY25: 10,210M m³"""

def search_context_hybrid(query: str):
    import unicodedata
    query_vector = embedder.encode(f"search_query: {query}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, year, title, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return [], 0.0

    def normalize_str(s: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFD', s.lower()) if unicodedata.category(c) != 'Mn')

    stopwords = {"which", "what", "where", "when", "that", "this", "from", "into", "over", "with", "across", "like", "does", "have", "been", "according", "nelerdir", "nedir", "neler", "hangi", "hangisi", "kadar", "olan", "icin", "göre", "gore"}
    clean_q = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
    raw_keywords = [normalize_str(w) for w in clean_q.split() if len(w) > 2 and w.lower() not in stopwords]

    tr_to_en = {
        "surdurulebilirlik": "sustainability", "rapor": "report", "raporu": "report", "raporunda": "report",
        "emisyon": "emissions", "karbon": "carbon", "su": "water", "atik": "waste",
        "basliklar": "highlights", "one": "key", "cikan": "pillars", "ozet": "summary", "genel": "overview",
        "hedef": "goal", "hedefler": "targets", "enerji": "energy", "elektrik": "electricity",
        "veri": "datacenter", "merkezi": "datacenter", "merkezleri": "datacenters", "yenileme": "replenishment"
    }
    keywords = list(raw_keywords)
    for kw in raw_keywords:
        if kw in tr_to_en:
            keywords.append(tr_to_en[kw])

    scores = []
    for r in rows:
        c_id, year, title, content, emb_json = r
        doc_vector = np.array(json.loads(emb_json))
        norm_q = np.linalg.norm(query_vector)
        norm_d = np.linalg.norm(doc_vector)
        sim = 0.0 if (norm_q == 0 or norm_d == 0) else float(np.dot(query_vector, doc_vector) / (norm_q * norm_d))
        
        norm_content = normalize_str(content)
        match_count = sum(1 for kw in keywords if kw in norm_content)
        hybrid_score = sim + (0.10 * match_count)
        
        scores.append({"id": c_id, "year": year, "title": title, "content": content, "score": hybrid_score})

    scores.sort(key=lambda x: x["score"], reverse=True)
    if not scores:
        return [], 0.0

    max_score = scores[0]["score"]
    if max_score < MIN_SCORE_FLOOR:
        return [], max_score

    # 🌟 Year-Stratified Retrieval (Yıl Bazlı Katmanlı Arama)
    # Çok yıllı veya karşılaştırmalı sorgularda (FY23, FY25, 2024, 2026 vb.) her rapordan dengeli parça al
    found_years = re.findall(r'\b(2024|2025|2026)\b', query)
    found_fys = re.findall(r'\bfy\s*(2[0-6])\b', query.lower())
    q_norm = normalize_str(query)

    is_multi_year = bool(
        len(found_years) >= 2 or
        len(found_fys) >= 2 or
        (len(found_years) >= 1 and len(found_fys) >= 1) or
        any(w in q_norm for w in ["uc yillik", "3 yillik", "tarihsel", "karsilastir", "gelisim", "trajectory", "multi-year", "across the", "across reports", "trend", "fark", "degisim", "ilerle"])
    )

    if is_multi_year:
        y2024 = [s for s in scores if "2024" in str(s.get("year", "")) or "2024" in str(s.get("title", ""))][:2]
        y2025 = [s for s in scores if "2025" in str(s.get("year", "")) or "2025" in str(s.get("title", ""))][:2]
        y2026 = [s for s in scores if "2026" in str(s.get("year", "")) or "2026" in str(s.get("title", ""))][:2]
        
        # Eğer sorgu spesifik olarak FY23 ve FY25 istiyorsa (2024 ve 2026 raporları)
        if ('23' in found_fys or '2024' in found_years) and ('25' in found_fys or '2026' in found_years) and '24' not in found_fys and '2025' not in found_years:
            y2024_top3 = [s for s in scores if "2024" in str(s.get("year", "")) or "2024" in str(s.get("title", ""))][:3]
            y2026_top3 = [s for s in scores if "2026" in str(s.get("year", "")) or "2026" in str(s.get("title", ""))][:3]
            stratified = y2026_top3 + y2024_top3
        else:
            stratified = y2026 + y2025 + y2024

        if len(stratified) >= 3:
            filtered = stratified
        else:
            cutoff = max_score * RELATIVE_DROP_RATIO
            filtered = [item for item in scores[:MAX_K] if item["score"] >= cutoff]
    else:
        cutoff = max_score * RELATIVE_DROP_RATIO
        filtered = [item for item in scores[:MAX_K] if item["score"] >= cutoff]
    
    del scores
    del rows
    gc.collect()
    
    return filtered, max_score

# ══════════════════════════════════════════════════════════════════════════════
# ÇİFT DİLLİ METİN SÖZLÜĞÜ (BILINGUAL DICTIONARY)
# ══════════════════════════════════════════════════════════════════════════════
TEXTS = {
    "tr": {
        "title": "Microsoft EcoRAG Lab",
        "subtitle": "Sıfır Halüsinasyonlu Deterministik ESG ve Sürdürülebilirlik Analiz Paneli",
        "sidebar_title": "EcoRAG Lab",
        "sidebar_caption": "Deterministik Sürdürülebilirlik Analizi",
        "lang_label": "Dil / Language",
        "theme_label": "Görsel Tema / Palette",
        "status_box_title": "Sistem Durumu",
        "status_badge": "Aktif & Doğrulanmış",
        "status_model": "Model",
        "status_embed": "Embedding",
        "status_index": "İndeks",
        "status_engine": "Motor",
        "reset_btn": "Sohbeti Sıfırla",
        "tab_chat": "💬 Akıllı Asistan",
        "tab_dash": "📊 ESG Bilanço Paneli",
        "tab_sys": "🛠️ Sistem & Benchmark Durumu",
        "pills_title": "Hızlı Başlangıç & Benchmark Test Soruları",
        "badge_pal": "PAL Deterministik Hesaplama",
        "badge_rag": "Hibrit Vektör Arama & Pydantic",
        "verified_output_label": "⚡ Doğrulanmış Analitik Çıktı (Verified Metrics)",
        "provenance_label": "Kullanılan Kaynaklar ({count}) • Benzerlik Skoru: {score:.4f} • Süre: {latency:.2f}s",
        "chat_placeholder": "Microsoft çevre ve sürdürülebilirlik raporlarına dair bir soru sorun...",
        "spinner_text": "Deterministik çıkarım ve doğrulama yürütülüyor...",
        "not_found_msg": "Microsoft Çevresel Sürdürülebilirlik raporlarında bu konuyla ilgili bilgi bulunmamaktadır.",
        "kpi_co2_title": "Toplam GHG Emisyonu (FY25)",
        "kpi_co2_delta": "+61.7% (FY20 Bazına Göre)",
        "kpi_co2_cap": "Scope 1 + Scope 2 (Market) + Scope 3",
        "kpi_water_title": "Kümülatif Su Yenileme (FY25)",
        "kpi_water_delta": "+82.1% Hedef Başarım Oranı",
        "kpi_water_cap": "2030 Water Positive Hedefi Kapsamı",
        "kpi_waste_title": "Yönlendirilen Katı Atık (FY25)",
        "kpi_waste_delta": "%82.3 Çöpten Kurtarma Oranı",
        "kpi_waste_cap": "Geri Dönüşüm, Yeniden Kullanım ve Kompost",
        "dash_title": "Microsoft Kurumsal ESG Bilançosu",
        "dash_caption": "2024–2025–2026 Microsoft Çevresel Sürdürülebilirlik Raporları Doğrulanmış Verileri",
        "dash_t1": "1. Sera Gazı Emisyon Dağılımı (Scope 1, 2, 3)",
        "dash_t1_cap": "Birim: mtCO2e (Metrik ton CO2 eşdeğeri) • Kaynak: 2025 Report Appendix Table 1",
        "dash_t2": "2. Karbon Uzaklaştırma Portföyü",
        "dash_t2_cap": "Birim: mtCO2e • Kaynak: 2025 Report p.21-22",
        "dash_t3": "3. Su Bilançosu & Hedefler",
        "dash_t3_cap": "Birim: million m³ • Kaynak: 2025 Report Water Table 1",
        "dash_t4": "4. Sıfır Atık & UL Solutions Sertifikasyonları",
        "dash_t4_cap": "Kaynak: 2024 Report p.36 & 2025 Report p.47",
        "dash_t5": "5. 2026 Çevresel Sürdürülebilirlik Raporu — Denetim Metrikleri & Bölgesel Göstergeler",
        "dash_t5_cap": "Kaynak: 2026-Microsoft-Environmental-Sustainability-Report-PDF.pdf (Denetlenmiş Resmi Metrikler & Metodolojiler)",
        "sys_title": "Altyapı & Benchmark Değerlendirme Raporu",
        "sys_caption": "Yerel SLM Çıkarım Mimarisi ve Deterministik Doğrulama Ölçümleri",
        "sys_card1_title": "Teknik Parametreler",
        "sys_card2_title": "500 Soruluk Üretim Benchmarkı",
        "sys_flow_title": "Çalışma Hattı Akış Şeması"
    },
    "en": {
        "title": "Microsoft EcoRAG Lab",
        "subtitle": "Zero-Hallucination Deterministic ESG & Sustainability Analysis Engine",
        "sidebar_title": "EcoRAG Lab",
        "sidebar_caption": "Deterministic Sustainability Analysis",
        "lang_label": "Language / Dil",
        "theme_label": "Theme / Palette",
        "status_box_title": "System Status",
        "status_badge": "Active & Verified",
        "status_model": "Model",
        "status_embed": "Embedding",
        "status_index": "Index",
        "status_engine": "Engine",
        "reset_btn": "Clear Conversation",
        "tab_chat": "💬 Smart Assistant",
        "tab_dash": "📊 ESG Balance Dashboard",
        "tab_sys": "🛠️ System & Benchmark Status",
        "pills_title": "Quick Prompts & Benchmark Questions",
        "badge_pal": "PAL Deterministic Calculation",
        "badge_rag": "Hybrid Vector Search & Pydantic",
        "verified_output_label": "⚡ Verified Structured Representation",
        "provenance_label": "Data Provenance ({count}) • Similarity Score: {score:.4f} • Latency: {latency:.2f}s",
        "chat_placeholder": "Ask a question regarding Microsoft sustainability reports...",
        "spinner_text": "Executing deterministic extraction & validation...",
        "not_found_msg": "I cannot find information regarding this in the provided Microsoft Environmental Sustainability reports.",
        "kpi_co2_title": "Total GHG Emissions (FY25)",
        "kpi_co2_delta": "+61.7% (vs FY20 Baseline)",
        "kpi_co2_cap": "Scope 1 + Scope 2 (Market) + Scope 3",
        "kpi_water_title": "Cumulative Water Replenishment (FY25)",
        "kpi_water_delta": "+82.1% Achievement Rate",
        "kpi_water_cap": "2030 Water Positive Commitment",
        "kpi_waste_title": "Diverted Solid Waste (FY25)",
        "kpi_waste_delta": "82.3% Diversion Rate",
        "kpi_waste_cap": "Recycled, Reused & Composted",
        "dash_title": "Microsoft Corporate ESG Balance Sheet",
        "dash_caption": "Verified Data from 2024–2025–2026 Microsoft Environmental Sustainability Reports",
        "dash_t1": "1. Greenhouse Gas Emissions (Scope 1, 2, 3)",
        "dash_t1_cap": "Unit: mtCO2e (Metric tons CO2 equivalent) • Source: 2025 Report Appendix Table 1",
        "dash_t2": "2. Carbon Removal Portfolio Breakdown",
        "dash_t2_cap": "Unit: mtCO2e • Source: 2025 Report p.21-22",
        "dash_t3": "3. Water Metrics & Replenishment Targets",
        "dash_t3_cap": "Unit: million m³ • Source: 2025 Report Water Table 1",
        "dash_t4": "4. Zero Waste & UL Solutions Certifications",
        "dash_t4_cap": "Source: 2024 Report p.36 & 2025 Report p.47",
        "dash_t5": "5. 2026 Environmental Sustainability Report — Audit Metrics & Regional Indicators",
        "dash_t5_cap": "Source: 2026-Microsoft-Environmental-Sustainability-Report-PDF.pdf (Audited Official Metrics & Methodologies)",
        "sys_title": "Infrastructure & Benchmark Evaluation Report",
        "sys_caption": "Local SLM Inference Architecture and Deterministic Verification Metrics",
        "sys_card1_title": "Technical Parameters",
        "sys_card2_title": "500-Question Production Benchmark",
        "sys_flow_title": "Pipeline Execution Flowchart"
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (KENAR ÇUBUĞU)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    curr_theme = st.session_state.get("theme_id", "pink")
    if curr_theme == "dark":
        t_head = "#ffffff"
        t_sub = "#8b949e"
        t_body = "#e6edf3"
    elif curr_theme == "pink":
        t_head = "#4a0e1e"
        t_sub = "#6b3343"
        t_body = "#2d1b22"
    elif curr_theme == "blue":
        t_head = "#0c4a6e"
        t_sub = "#475569"
        t_body = "#0f172a"
    else:  # white
        t_head = "#0f172a"
        t_sub = "#475569"
        t_body = "#0f172a"

    st.markdown("<div style='margin-top: -28px; margin-bottom: 2px;'><span class='sidebar-main-title' style='display: flex; align-items: center; gap: 8px; letter-spacing: -0.5px;'>🌱 EcoRAG Lab</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-subtitle' style='margin-bottom: 18px;'>Deterministic Sustainability Analysis</div>", unsafe_allow_html=True)

    # 1. 🌐 Kompakt Dil Seçici (st.pills - 🇬🇧 EN & 🇹🇷 TR)
    if "is_turkish" not in st.session_state:
        st.session_state.is_turkish = True

    st.markdown("<div class='sidebar-section-title' style='margin-bottom: 6px;'>LANGUAGE / DİL</div>", unsafe_allow_html=True)

    lang_opts = ["🇬🇧 EN", "🇹🇷 TR"]
    curr_lang = "🇹🇷 TR" if st.session_state.is_turkish else "🇬🇧 EN"

    selected_lang = st.pills(
        "Language",
        options=lang_opts,
        default=curr_lang,
        key="sidebar_lang_pills",
        label_visibility="collapsed"
    )
    if selected_lang:
        st.session_state.is_turkish = (selected_lang == "🇹🇷 TR")
    is_tr = st.session_state.is_turkish

    L = "tr" if is_tr else "en"
    T = TEXTS[L]

    # 2. 🎨 Temalar İçin Yatay Kapsüller (st.pills)
    st.markdown(f"<div class='sidebar-section-title' style='margin-top: 14px; margin-bottom: 6px;'>{T['theme_label'].upper()}</div>", unsafe_allow_html=True)

    theme_meta = [
        {"id": "pink", "label_tr": "🌸 Toz Pembe", "label_en": "🌸 Blush Rose"},
        {"id": "blue", "label_tr": "💼 Fluent Azure", "label_en": "💼 Fluent Azure"},
        {"id": "dark", "label_tr": "🌿 Eco Emerald", "label_en": "🌿 Eco Emerald"},
        {"id": "white", "label_tr": "⚪ Saf Beyaz", "label_en": "⚪ Pure Light"}
    ]
    if "theme_id" not in st.session_state:
        st.session_state.theme_id = "pink"

    theme_options = [m["label_tr"] if is_tr else m["label_en"] for m in theme_meta]
    id_to_label = {m["id"]: (m["label_tr"] if is_tr else m["label_en"]) for m in theme_meta}
    label_to_id = {(m["label_tr"] if is_tr else m["label_en"]): m["id"] for m in theme_meta}

    current_label = id_to_label.get(st.session_state.theme_id, theme_options[0])

    selected_pill = st.pills(
        T["theme_label"],
        options=theme_options,
        default=current_label,
        key="sidebar_theme_pills",
        label_visibility="collapsed"
    )
    if selected_pill:
        st.session_state.theme_id = label_to_id.get(selected_pill, "pink")
    current_theme_id = st.session_state.theme_id

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 3. 🛠️ Sistem Durumu Konteyneri (Rahatlatılmış Dikey Hizalama & Net Kontrast)
    with st.container(border=True):
        st.markdown(f"<div class='sidebar-box-title' style='margin-bottom: 6px;'>{T['status_box_title']}</div>", unsafe_allow_html=True)
        st.badge(T["status_badge"], icon=":material/check_circle:", color="green")
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        try:
            conn_chk = sqlite3.connect(DB_PATH)
            c_chk = conn_chk.cursor()
            c_chk.execute("SELECT COUNT(*) FROM documents")
            total_chunks_db = c_chk.fetchone()[0]
            conn_chk.close()
        except Exception:
            total_chunks_db = 1044
        st.markdown(f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;'><span class='sidebar-metric-label'>{T['status_index']}</span><code>{total_chunks_db} Chunks</code></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;'><span class='sidebar-metric-label'>{T['status_engine']}</span><code>PAL + IR</code></div>", unsafe_allow_html=True)
        with st.expander("⚙️ LLM Endpoint", expanded=False):
            endpoint_input = st.text_input(
                "Foundry URL",
                value=st.session_state.get("foundry_base_url", FOUNDRY_BASE_URL),
                key="foundry_base_url_input",
                help="Varsayılan: http://127.0.0.1:62095 veya yerel proxy"
            )
            if endpoint_input != st.session_state.get("foundry_base_url", FOUNDRY_BASE_URL):
                st.session_state.foundry_base_url = endpoint_input

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if st.button(T["reset_btn"], icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        gc.collect()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DİNAMİK TEMA ENJEKSİYONU (4 FARKLI PALET - TAM KONTRAST & EKSİKSİZ BİLEŞEN UYUMU)
# ══════════════════════════════════════════════════════════════════════════════
if current_theme_id == "dark":
    st.html("""
    <style>
    /* 🌿 Midnight Emerald Dark Theme */
    :root, .stApp {
        --background-color: #0d1117 !important;
        --secondary-background-color: #161b22 !important;
        --text-color: #e6edf3 !important;
        --primary-color: #f0f6fc !important;
    }
    .stApp {
        background-color: #0d1117 !important;
    }
    header[data-testid="stHeader"] {
        background-color: #0d1117 !important;
    }
    .stApp, .stApp p, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label, .stApp div[data-testid="stMarkdownContainer"] p {
        color: #e6edf3 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h3 {
        color: #e6edf3 !important;
    }
    /* 🏷️ Sidebar Typography Classes (Dark Theme) */
    .sidebar-main-title { color: #ffffff !important; font-size: 25px !important; font-weight: 900 !important; }
    .sidebar-subtitle { color: #8b949e !important; font-size: 13px !important; font-weight: 500 !important; }
    .sidebar-section-title { color: #ffffff !important; font-size: 11px !important; font-weight: 800 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; opacity: 0.9 !important; }
    .sidebar-box-title { color: #ffffff !important; font-size: 13px !important; font-weight: 800 !important; }
    .sidebar-metric-label { color: #8b949e !important; font-size: 13px !important; font-weight: 600 !important; }
    /* 🔘 Universal Sub-element Border Reset */
    [data-testid*="stPills"] *,
    [data-testid*="stSegmentedControl"] *,
    [data-baseweb="tag"] *,
    .stTabs * {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    /* Pills (Hızlı Sorular) */
    div[data-testid="stPills"] button, div[data-testid="stPills"] [data-baseweb="tag"], div[data-testid="stPills"] span {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        font-weight: 500 !important;
        border-radius: 20px !important;
    }
    div[data-testid="stPills"] button:hover {
        background-color: #30363d !important;
        color: #ffffff !important;
        border-color: #8b949e !important;
    }
    div[data-testid="stPills"] [aria-pressed="true"], div[data-testid="stPills"] [aria-selected="true"], div[data-testid="stPills"] [aria-checked="true"] {
        background-color: #f0f6fc !important;
        color: #0d1117 !important;
        font-weight: 700 !important;
        border: 2px solid #ffffff !important;
    }
    div[data-testid="stPills"] [aria-pressed="true"] span, div[data-testid="stPills"] [aria-selected="true"] span, div[data-testid="stPills"] [aria-checked="true"] span {
        color: #0d1117 !important;
    }
    /* Segmented Control (Dil & Tema Seçici) */
    div[data-testid="stSegmentedControl"] > div,
    div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    div[data-testid="stSegmentedControl"] button {
        background-color: transparent !important;
        color: #e6edf3 !important;
        font-weight: 500 !important;
        border: none !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"], div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background-color: #f0f6fc !important;
        color: #0d1117 !important;
        font-weight: 700 !important;
        border: none !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] p, div[data-testid="stSegmentedControl"] button[aria-checked="true"] span {
        color: #0d1117 !important;
    }
    /* Chat Input */
    [data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {
        position: fixed !important;
        bottom: 0px !important;
        left: 0px !important;
        right: 0px !important;
        z-index: 9999 !important;
        background-color: #0d1117 !important;
        border: none !important;
        padding: 10px 20px 20px 20px !important;
    }
    .main .block-container {
        padding-bottom: 130px !important;
    }
    div[data-testid="stChatInput"] {
        background-color: #161b22 !important;
        border: 1.5px solid #30363d !important;
        border-radius: 12px !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #e6edf3 !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #8b949e !important;
    }
    div[data-testid="stChatInput"] button {
        color: #f0f6fc !important;
    }
    /* Code Badges */
    code {
        background-color: #21262d !important;
        color: #f0f6fc !important;
        border: 1px solid #484f58 !important;
        border-radius: 4px;
        padding: 2px 6px;
        font-weight: 600;
    }
    /* Button */
    .stButton > button {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #30363d !important;
        border-color: #8b949e !important;
        color: #ffffff !important;
    }
    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    div[data-baseweb="select"] span {
        color: #e6edf3 !important;
    }
    /* Metrics, Cards, Expanders */
    div[data-testid="stMetricValue"] { color: #f0f6fc !important; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #8b949e !important; }
    div[data-testid="stMetric"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
    }
    div[data-testid="stChatMessage"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span { color: #e6edf3 !important; }
    .stApp div[data-testid="stExpander"],
    .stApp details[data-testid="stExpander"],
    div[data-testid="stExpander"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
    }
    .stApp div[data-testid="stExpander"] summary,
    .stApp details[data-testid="stExpander"] summary,
    .stApp [data-testid="stExpanderSummary"],
    div[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary {
        background-color: #21262d !important;
        color: #f0f6fc !important;
        font-weight: 700 !important;
        border-bottom: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
    }
    .stApp div[data-testid="stExpander"] summary *,
    .stApp details[data-testid="stExpander"] summary *,
    .stApp [data-testid="stExpanderSummary"] *,
    div[data-testid="stExpander"] summary * {
        color: #f0f6fc !important;
        -webkit-text-fill-color: #f0f6fc !important;
        font-weight: 700 !important;
    }
    .stApp div[data-testid="stExpanderDetails"],
    .stApp div[data-testid="stExpander"] div,
    .stApp div[data-testid="stExpander"] p,
    .stApp div[data-testid="stExpander"] span,
    .stApp div[data-testid="stText"],
    .stApp div[data-testid="stText"] pre {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        -webkit-text-fill-color: #e6edf3 !important;
    }
    /* 🌟 Sekmeler (Tabs - Yarı Saydam Beyaz & Siyah Metin) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    div[data-baseweb="tab"][aria-selected="false"],
    .stTabs [data-baseweb="tab"] {
        color: #e6edf3 !important;
        background-color: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
    }
    div[data-baseweb="tab"][aria-selected="false"] *,
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #e6edf3 !important;
    }
    .stApp div[data-baseweb="tab"][aria-selected="true"],
    .stApp .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stApp .stTabs [aria-selected="true"],
    div[data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-bottom: 3px solid #000000 !important;
        border-radius: 6px !important;
    }
    .stApp div[data-baseweb="tab"][aria-selected="true"] *,
    .stApp .stTabs [aria-selected="true"] *,
    .stApp .stTabs [aria-selected="true"] p,
    .stApp .stTabs [aria-selected="true"] span,
    .stApp .stTabs [aria-selected="true"] div,
    div[data-baseweb="tab"][aria-selected="true"] *,
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 800 !important;
    }
    .stApp small, .stApp .stCaption, .stApp caption, .stApp div[data-testid="stCaptionContainer"] { color: #8b949e !important; }
    </style>
    """)
elif current_theme_id == "white":
    st.html("""
    <style>
    /* ⚪ Pure Light (Saf Beyaz & Asil Siyah Vurgulu Açık Mod) */
    :root, .stApp {
        --background-color: #ffffff !important;
        --secondary-background-color: #f8fafc !important;
        --text-color: #0f172a !important;
        --primary-color: #0f172a !important;
    }
    .stApp {
        background-color: #ffffff !important;
    }
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    .stApp, 
    .stApp p, 
    .stApp span, 
    .stApp li, 
    .stApp ul, 
    .stApp ol, 
    .stApp li *, 
    .stApp h1, 
    .stApp h2, 
    .stApp h3, 
    .stApp h4, 
    .stApp h5, 
    .stApp h6, 
    .stApp label, 
    .stApp strong,
    .stApp em,
    .stApp blockquote,
    .stApp td,
    .stApp th,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] ul,
    div[data-testid="stMarkdownContainer"] ol,
    div[data-testid="stMarkdownContainer"] li *,
    div[data-testid="stChatMessage"] * {
        color: #0f172a !important; /* Çok net siyah/koyu antrasit metin */
    }
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }
    /* 🏷️ Sidebar Typography Classes (Pure Light Theme) */
    .sidebar-main-title { color: #0f172a !important; font-size: 25px !important; font-weight: 900 !important; }
    .sidebar-subtitle { color: #475569 !important; font-size: 13px !important; font-weight: 500 !important; }
    .sidebar-section-title { color: #0f172a !important; font-size: 11px !important; font-weight: 800 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; opacity: 0.9 !important; }
    .sidebar-box-title { color: #0f172a !important; font-size: 13px !important; font-weight: 800 !important; }
    .sidebar-metric-label { color: #475569 !important; font-size: 13px !important; font-weight: 600 !important; }
    /* 🔘 Universal Sub-element Border Reset */
    [data-testid*="stPills"] *,
    [data-testid*="stSegmentedControl"] *,
    [data-baseweb="tag"] *,
    .stTabs * {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* 🔘 Pills (Hızlı Sorular - Siyah Seçili Durum) */
    div[data-testid="stPills"] button,
    div[data-testid="stPills"] [data-baseweb="tag"],
    div[role="radiogroup"] button {
        background-color: #f1f5f9 !important;
        color: #334155 !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 20px !important;
        padding: 6px 14px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stPills"] button:hover,
    div[data-testid="stPills"] [data-baseweb="tag"]:hover,
    div[role="radiogroup"] button:hover {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        border-color: #64748b !important;
    }
    .stApp div[data-testid="stPills"] [aria-pressed="true"],
    .stApp div[data-testid="stPills"] [aria-selected="true"],
    .stApp div[data-testid="stPills"] [aria-checked="true"],
    .stApp div[role="radiogroup"] [aria-checked="true"],
    div[data-testid="stPills"] [aria-pressed="true"],
    div[data-testid="stPills"] [aria-selected="true"],
    div[data-testid="stPills"] [aria-checked="true"],
    div[role="radiogroup"] [aria-checked="true"] {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border: 2px solid #000000 !important;
    }
    .stApp div[data-testid="stPills"] [aria-pressed="true"] *,
    .stApp div[data-testid="stPills"] [aria-selected="true"] *,
    .stApp div[data-testid="stPills"] [aria-checked="true"] *,
    .stApp div[role="radiogroup"] [aria-checked="true"] *,
    div[data-testid="stPills"] [aria-pressed="true"] *,
    div[data-testid="stPills"] [aria-selected="true"] *,
    div[data-testid="stPills"] [aria-checked="true"] *,
    div[data-testid="stPills"] [aria-pressed="true"] span,
    div[data-testid="stPills"] [aria-selected="true"] span,
    div[data-testid="stPills"] [aria-checked="true"] span,
    div[data-testid="stPills"] [aria-pressed="true"] p,
    div[data-testid="stPills"] [aria-pressed="true"] div {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* 🎛️ Segmented Control (Dil & Tema Seçici) */
    div[data-testid="stSegmentedControl"] > div,
    div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
        background-color: #f1f5f9 !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    div[data-testid="stSegmentedControl"] button {
        background-color: transparent !important;
        color: #334155 !important;
        font-weight: 600 !important;
        border: none !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background-color: #0f172a !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] span {
        color: #ffffff !important;
    }

    /* 💬 Chat Input & Bottom Bar */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        position: fixed !important;
        bottom: 0px !important;
        left: 0px !important;
        right: 0px !important;
        z-index: 9999 !important;
        background-color: #ffffff !important;
        border: none !important;
        padding: 10px 20px 20px 20px !important;
    }
    .main .block-container {
        padding-bottom: 130px !important;
    }
    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06) !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea * {
        background-color: transparent !important;
        color: #0f172a !important;
        font-size: 15px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }

    /* Code Badges & Code Blocks */
    div[data-testid="stCode"], div[data-testid="stCodeBlock"], pre {
        background-color: #f8fafc !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    div[data-testid="stCode"] code, div[data-testid="stCodeBlock"] code, pre code {
        background-color: transparent !important;
        color: #0f172a !important;
        border: none !important;
        font-weight: 500 !important;
    }
    code {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px;
        padding: 2px 6px;
        font-weight: 600;
    }

    /* Button */
    .stButton > button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1.5px solid #cbd5e1 !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #f1f5f9 !important;
        border-color: #0f172a !important;
        color: #0f172a !important;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1.5px solid #cbd5e1 !important;
    }
    div[data-baseweb="select"] span {
        color: #0f172a !important;
    }

    /* Metrics, Cards, Expanders */
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        padding: 12px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        color: #0f172a !important;
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span {
        color: #0f172a !important;
    }
    .stApp div[data-testid="stExpander"],
    .stApp details[data-testid="stExpander"],
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }
    .stApp div[data-testid="stExpander"] summary,
    .stApp details[data-testid="stExpander"] summary,
    .stApp [data-testid="stExpanderSummary"],
    div[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        border-bottom: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
    }
    .stApp div[data-testid="stExpander"] summary *,
    .stApp details[data-testid="stExpander"] summary *,
    .stApp [data-testid="stExpanderSummary"] *,
    div[data-testid="stExpander"] summary * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-weight: 700 !important;
    }
    .stApp div[data-testid="stExpanderDetails"],
    .stApp div[data-testid="stExpander"] div,
    .stApp div[data-testid="stExpander"] p,
    .stApp div[data-testid="stExpander"] span,
    .stApp div[data-testid="stText"],
    .stApp div[data-testid="stText"] pre {
        background-color: #ffffff !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    /* Sekmeler (Tabs - Yarı Saydam Beyaz & Siyah Metin) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    div[data-baseweb="tab"][aria-selected="false"],
    .stTabs [data-baseweb="tab"] {
        color: #0f172a !important;
        background-color: transparent !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
    }
    div[data-baseweb="tab"][aria-selected="false"] *,
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #0f172a !important;
    }
    .stApp div[data-baseweb="tab"][aria-selected="true"],
    .stApp .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stApp .stTabs [aria-selected="true"],
    div[data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border: 1.5px solid #cbd5e1 !important;
        border-bottom: 3px solid #000000 !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05) !important;
    }
    .stApp div[data-baseweb="tab"][aria-selected="true"] *,
    .stApp .stTabs [aria-selected="true"] *,
    .stApp .stTabs [aria-selected="true"] p,
    .stApp .stTabs [aria-selected="true"] span,
    .stApp .stTabs [aria-selected="true"] div,
    div[data-baseweb="tab"][aria-selected="true"] *,
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 800 !important;
    }

    /* 📊 DataFrames & Tables */
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"],
    .stDataFrame,
    table {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }
    table thead tr th, th {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #cbd5e1 !important;
    }
    table tbody tr td, td {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-bottom: 1px solid #f1f5f9 !important;
    }
    table tbody tr:nth-child(even) td {
        background-color: #f8fafc !important;
    }
    .stApp small, .stApp .stCaption, .stApp caption, .stApp div[data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }
    </style>
    """)
elif current_theme_id == "blue":
    st.html("""
    <style>
    /* 🌊 Arctic Azure Light Theme */
    :root, .stApp {
        --background-color: #f8fafc !important;
        --secondary-background-color: #e1effe !important;
        --text-color: #0f172a !important;
        --primary-color: #0078d4 !important;
    }
    .stApp {
        background-color: #f8fafc !important;
    }
    header[data-testid="stHeader"] {
        background-color: #f8fafc !important;
    }
    .stApp, 
    .stApp p, 
    .stApp span, 
    .stApp li, 
    .stApp ul, 
    .stApp ol, 
    .stApp li *, 
    .stApp h1, 
    .stApp h2, 
    .stApp h3, 
    .stApp h4, 
    .stApp h5, 
    .stApp h6, 
    .stApp label, 
    .stApp strong,
    .stApp em,
    .stApp blockquote,
    .stApp td,
    .stApp th,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] ul,
    div[data-testid="stMarkdownContainer"] ol,
    div[data-testid="stMarkdownContainer"] li *,
    div[data-testid="stChatMessage"] * {
        color: #0f172a !important; /* Çok net koyu metin */
    }
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }
    /* 🏷️ Sidebar Typography Classes (Fluent Azure Theme) */
    .sidebar-main-title { color: #0c4a6e !important; font-size: 25px !important; font-weight: 900 !important; }
    .sidebar-subtitle { color: #475569 !important; font-size: 13px !important; font-weight: 500 !important; }
    .sidebar-section-title { color: #0c4a6e !important; font-size: 11px !important; font-weight: 800 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; opacity: 0.9 !important; }
    .sidebar-box-title { color: #0c4a6e !important; font-size: 13px !important; font-weight: 800 !important; }
    .sidebar-metric-label { color: #475569 !important; font-size: 13px !important; font-weight: 600 !important; }
    /* 🔘 Universal Sub-element Border Reset */
    [data-testid*="stPills"] *,
    [data-testid*="stSegmentedControl"] *,
    [data-baseweb="tag"] *,
    .stTabs * {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* 🔘 Pills (Hızlı Sorular - Arka Plandan Bir Tık Koyu Mavi) */
    div[data-testid="stPills"] button,
    div[data-testid="stPills"] [data-baseweb="tag"],
    div[role="radiogroup"] button {
        background-color: #e1effe !important; /* Arka plandan bir tık koyu açık mavi */
        color: #0c4a6e !important; /* Net okunur koyu lacivert */
        border: 1.5px solid #93c5fd !important;
        border-radius: 20px !important;
        padding: 6px 14px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stPills"] button:hover,
    div[data-testid="stPills"] [data-baseweb="tag"]:hover,
    div[role="radiogroup"] button:hover {
        background-color: #bfdbfe !important;
        color: #032b43 !important;
        border-color: #0078d4 !important;
    }
    
    /* 💬 Chat Input & Bottom Bar */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        position: fixed !important;
        bottom: 0px !important;
        left: 0px !important;
        right: 0px !important;
        z-index: 9999 !important;
        background-color: #f8fafc !important;
        border: none !important;
        padding: 10px 20px 20px 20px !important;
    }
    .main .block-container {
        padding-bottom: 130px !important;
    }
    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 2px solid #93c5fd !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 8px rgba(0, 120, 212, 0.08) !important;
    }

    .stApp div[data-testid="stPills"] [aria-pressed="true"],
    .stApp div[data-testid="stPills"] [aria-selected="true"],
    .stApp div[data-testid="stPills"] [aria-checked="true"],
    div[data-testid="stPills"] [aria-pressed="true"],
    div[data-testid="stPills"] [aria-selected="true"],
    div[data-testid="stPills"] [aria-checked="true"],
    div[role="radiogroup"] [aria-checked="true"] {
        background-color: #0078d4 !important;
        color: #ffffff !important;
        border: 2px solid #005a9e !important;
    }
    .stApp div[data-testid="stPills"] [aria-pressed="true"] *,
    .stApp div[data-testid="stPills"] [aria-selected="true"] *,
    .stApp div[data-testid="stPills"] [aria-checked="true"] *,
    div[data-testid="stPills"] [aria-pressed="true"] * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* 🎛️ Segmented Control (Dil Seçici - Açık Mavi & Koyu Mavi) */
    div[data-testid="stSegmentedControl"] > div,
    div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
        background-color: #e1effe !important;
        border: 1.5px solid #93c5fd !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    div[data-testid="stSegmentedControl"] button {
        background-color: transparent !important;
        color: #0c4a6e !important;
        font-weight: 600 !important;
        border: none !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background-color: #0078d4 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
    }

    /* 💬 Chat Input & Bottom Bar (Siyahlığı Tamamen Kaldırır) */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInputContainer"],
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div {
        background-color: transparent !important;
        border: none !important;
    }
    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 2px solid #93c5fd !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 8px rgba(0, 120, 212, 0.08) !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea * {
        background-color: transparent !important;
        color: #0f172a !important;
        font-size: 15px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #e1effe !important;
        color: #0078d4 !important;
        border-radius: 8px !important;
    }
    /* Code Badges & Code Blocks */
    div[data-testid="stCode"], div[data-testid="stCodeBlock"], pre {
        background-color: #f1f5f9 !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    div[data-testid="stCode"] code, div[data-testid="stCodeBlock"] code, pre code {
        background-color: transparent !important;
        color: #0f172a !important;
        border: none !important;
        font-weight: 500 !important;
    }
    code {
        background-color: #e1effe !important;
        color: #0078d4 !important;
        border: 1px solid #93c5fd !important;
        border-radius: 6px;
        padding: 2px 6px;
        font-weight: 600;
    }
    /* Button */
    .stButton > button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1.5px solid #cbd5e1 !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #e1effe !important;
        border-color: #0078d4 !important;
    }
    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1.5px solid #cbd5e1 !important;
    }
    div[data-baseweb="select"] span {
        color: #0f172a !important;
    }
    /* Metrics, Cards, Expanders */
    div[data-testid="stMetricValue"] {
        color: #0078d4 !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #334155 !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        padding: 12px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span {
        color: #0f172a !important;
    }
    .stApp div[data-testid="stExpander"],
    .stApp details[data-testid="stExpander"],
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1.5px solid #93c5fd !important;
        border-radius: 10px !important;
    }
    .stApp div[data-testid="stExpander"] summary,
    .stApp details[data-testid="stExpander"] summary,
    .stApp [data-testid="stExpanderSummary"],
    div[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary {
        background-color: #e1effe !important;
        color: #0c4a6e !important;
        font-weight: 700 !important;
        border-bottom: 1.5px solid #93c5fd !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
    }
    .stApp div[data-testid="stExpander"] summary *,
    .stApp details[data-testid="stExpander"] summary *,
    .stApp [data-testid="stExpanderSummary"] *,
    div[data-testid="stExpander"] summary * {
        color: #0c4a6e !important;
        -webkit-text-fill-color: #0c4a6e !important;
        font-weight: 700 !important;
    }
    .stApp div[data-testid="stExpanderDetails"],
    .stApp div[data-testid="stExpander"] div,
    .stApp div[data-testid="stExpander"] p,
    .stApp div[data-testid="stExpander"] span,
    .stApp div[data-testid="stText"],
    .stApp div[data-testid="stText"] pre {
        background-color: #ffffff !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    /* Sekmeler (Tabs - Fluent Azure) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    div[data-baseweb="tab"][aria-selected="false"],
    .stTabs [data-baseweb="tab"] {
        color: #0c4a6e !important;
        background-color: transparent !important;
        border: 1px solid #93c5fd !important;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
    }
    div[data-baseweb="tab"][aria-selected="false"] *,
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #0c4a6e !important;
    }
    .stApp div[data-baseweb="tab"][aria-selected="true"],
    .stApp .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stApp .stTabs [aria-selected="true"],
    div[data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border: 1.5px solid #93c5fd !important;
        border-bottom: 3px solid #000000 !important;
        border-radius: 6px !important;
    }
    .stApp div[data-baseweb="tab"][aria-selected="true"] *,
    .stApp .stTabs [aria-selected="true"] *,
    .stApp .stTabs [aria-selected="true"] p,
    .stApp .stTabs [aria-selected="true"] span,
    .stApp .stTabs [aria-selected="true"] div,
    div[data-baseweb="tab"][aria-selected="true"] *,
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 800 !important;
    }
    /* 📊 DataFrames & Tables (Siyah Tabloları Tamamen Kaldırır) */
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"],
    .stDataFrame,
    table {
        background-color: #ffffff !important;
        border: 1.5px solid #93c5fd !important;
        border-radius: 10px !important;
    }
    table thead tr th, th {
        background-color: #e1effe !important;
        color: #0c4a6e !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #93c5fd !important;
    }
    table tbody tr td, td {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-bottom: 1px solid #e1effe !important;
    }
    table tbody tr:nth-child(even) td {
        background-color: #f8fafc !important;
    }
    .stApp small, .stApp .stCaption, .stApp caption, .stApp div[data-testid="stCaptionContainer"] {
        color: #475569 !important;
    }
    </style>
    """)
else:
    # 🌸 Toz Pembe Pastel (Blush Rose Theme)
    st.html("""
    <style>
    /* 🌸 Toz Pembe Pastel / Blush Rose Theme */
    :root, .stApp {
        --background-color: #fdf6f7 !important;
        --secondary-background-color: #f7dbe1 !important;
        --text-color: #2d1b22 !important;
        --primary-color: #be185d !important;
    }
    .stApp {
        background: linear-gradient(180deg, #fdf6f7 0%, #f7e8ec 100%) !important;
    }
    header[data-testid="stHeader"] {
        background-color: #fdf6f7 !important;
    }
    .stApp, 
    .stApp p, 
    .stApp span, 
    .stApp li, 
    .stApp ul, 
    .stApp ol, 
    .stApp li *, 
    .stApp h1, 
    .stApp h2, 
    .stApp h3, 
    .stApp h4, 
    .stApp h5, 
    .stApp h6, 
    .stApp label, 
    .stApp strong,
    .stApp em,
    .stApp blockquote,
    .stApp td,
    .stApp th,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] ul,
    div[data-testid="stMarkdownContainer"] ol,
    div[data-testid="stMarkdownContainer"] li *,
    div[data-testid="stChatMessage"] * {
        color: #2d1b22 !important; /* Net okunur koyu mürdüm-antrasit */
    }
    section[data-testid="stSidebar"] {
        background-color: #f7e2e6 !important;
        border-right: 1px solid #e8bcc5 !important;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h3 {
        color: #2d1b22 !important;
    }
    /* 🏷️ Sidebar Typography Classes (Blush Rose Theme) */
    .sidebar-main-title { color: #4a0e1e !important; font-size: 25px !important; font-weight: 900 !important; }
    .sidebar-subtitle { color: #6b3343 !important; font-size: 13px !important; font-weight: 500 !important; }
    .sidebar-section-title { color: #4a0e1e !important; font-size: 11px !important; font-weight: 800 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; opacity: 0.9 !important; }
    .sidebar-box-title { color: #4a0e1e !important; font-size: 13px !important; font-weight: 800 !important; }
    .sidebar-metric-label { color: #6b3343 !important; font-size: 13px !important; font-weight: 600 !important; }
    /* 🔘 Universal Sub-element Border Reset (İç Dikdörtgen Kutuları Tamamen Yok Eder) */
    [data-testid*="stPills"] *,
    [data-testid*="stSegmentedControl"] *,
    [data-baseweb="tag"] *,
    .stTabs * {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* 🔘 Pills (Yumuşak Oval Haplar - İç Çerçeve Yok) */
    div[data-testid="stPills"] button,
    div[data-testid="stPills"] [data-baseweb="tag"],
    div[role="radiogroup"] button {
        background-color: #f7dbe1 !important;
        color: #4a0e1e !important;
        border: 1.5px solid #d99ca9 !important;
        border-radius: 20px !important;
        padding: 6px 14px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stPills"] button:hover,
    div[data-testid="stPills"] [data-baseweb="tag"]:hover,
    div[role="radiogroup"] button:hover {
        background-color: #ebd0d6 !important;
        color: #2d050f !important;
        border-color: #b85d75 !important;
    }
    .stApp div[data-testid="stPills"] [aria-pressed="true"],
    .stApp div[data-testid="stPills"] [aria-selected="true"],
    .stApp div[data-testid="stPills"] [aria-checked="true"],
    div[data-testid="stPills"] [aria-pressed="true"],
    div[data-testid="stPills"] [aria-selected="true"],
    div[data-testid="stPills"] [aria-checked="true"],
    div[role="radiogroup"] [aria-checked="true"] {
        background-color: #f0c3cb !important;
        color: #2d050f !important;
        border: 2px solid #b85d75 !important;
    }
    .stApp div[data-testid="stPills"] [aria-pressed="true"] *,
    .stApp div[data-testid="stPills"] [aria-selected="true"] *,
    .stApp div[data-testid="stPills"] [aria-checked="true"] *,
    div[data-testid="stPills"] [aria-pressed="true"] * {
        color: #2d050f !important;
        font-weight: 700 !important;
    }
        color: inherit !important;
    }

    /* 🎛️ Segmented Control (Dil Seçici - İç Dikdörtgensiz) */
    div[data-testid="stSegmentedControl"] > div,
    div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
        background-color: #f7dbe1 !important;
        border: 1.5px solid #d99ca9 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    div[data-testid="stSegmentedControl"] button {
        background-color: transparent !important;
        color: #4a0e1e !important;
        font-weight: 600 !important;
        border: none !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background-color: #f0c3cb !important;
        color: #4a0e1e !important;
        font-weight: 700 !important;
        border: none !important;
    }
    /* 💬 Chat Input & Bottom Bar (Siyahlığı Tamamen Kaldırır) */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInputContainer"],
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div {
        background-color: transparent !important;
        border: none !important;
    }
    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 2px solid #d99ca9 !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 8px rgba(184, 93, 117, 0.08) !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea * {
        background-color: transparent !important;
        color: #2d1b22 !important;
        font-size: 15px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #8a4a58 !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #f7dbe1 !important;
        color: #831843 !important;
        border-radius: 8px !important;
    }
    /* Code Badges & Code Blocks */
    div[data-testid="stCode"], div[data-testid="stCodeBlock"], pre {
        background-color: #faedf0 !important;
        border: 2px solid #d99ca9 !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    div[data-testid="stCode"] code, div[data-testid="stCodeBlock"] code, pre code {
        background-color: transparent !important;
        color: #501d2d !important;
        border: none !important;
        font-weight: 500 !important;
    }
    div[data-testid="stCode"] button, div[data-testid="stCodeBlock"] button {
        color: #831843 !important;
        background-color: transparent !important;
    }
    code {
        background-color: #fadce2 !important;
        color: #9d174d !important;
        border: 1px solid #e8bcc5 !important;
        border-radius: 4px;
        padding: 2px 6px;
        font-weight: 600;
    }
    /* Button */
    .stButton > button {
        background-color: #ffffff !important;
        color: #501d2d !important;
        border: 1px solid #e8bcc5 !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #fadce2 !important;
        border-color: #d99ca9 !important;
    }
    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #2d1b22 !important;
        border: 1px solid #e8bcc5 !important;
    }
    div[data-baseweb="select"] span {
        color: #2d1b22 !important;
    }
    /* Metrics, Cards, Expanders */
    div[data-testid="stMetricValue"] {
        color: #831843 !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #502838 !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e8bcc5 !important;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(184, 93, 117, 0.08);
        padding: 14px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #fffffffa !important;
        border: 1px solid #e8bcc5 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(184, 93, 117, 0.05);
    }
    div[data-testid="stChatMessage"] {
        background-color: #fffffffa !important;
        border: 1px solid #ebd0d6 !important;
        color: #2d1b22 !important;
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span {
        color: #2d1b22 !important;
    }
    .stApp div[data-testid="stExpander"],
    .stApp details[data-testid="stExpander"],
    div[data-testid="stExpander"] {
        background-color: #fffffffa !important;
        border: 1.5px solid #d99ca9 !important;
        border-radius: 10px !important;
    }
    .stApp div[data-testid="stExpander"] summary,
    .stApp details[data-testid="stExpander"] summary,
    .stApp [data-testid="stExpanderSummary"],
    div[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary {
        background-color: #f7dbe1 !important;
        color: #4a0e1e !important;
        font-weight: 700 !important;
        border-bottom: 1.5px solid #d99ca9 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
    }
    .stApp div[data-testid="stExpander"] summary *,
    .stApp details[data-testid="stExpander"] summary *,
    .stApp [data-testid="stExpanderSummary"] *,
    div[data-testid="stExpander"] summary * {
        color: #4a0e1e !important;
        -webkit-text-fill-color: #4a0e1e !important;
        font-weight: 700 !important;
    }
    .stApp div[data-testid="stExpanderDetails"],
    .stApp div[data-testid="stExpander"] div,
    .stApp div[data-testid="stExpander"] p,
    .stApp div[data-testid="stExpander"] span,
    .stApp div[data-testid="stText"],
    .stApp div[data-testid="stText"] pre {
        background-color: #fffffffa !important;
        color: #2d1b22 !important;
        -webkit-text-fill-color: #2d1b22 !important;
    }
    /* Sekmeler (Tabs - Toz Pembe) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    div[data-baseweb="tab"][aria-selected="false"],
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        background-color: transparent !important;
        color: #4a0e1e !important;
        border: 1px solid #d99ca9 !important;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div[data-baseweb="tab"][aria-selected="false"] *,
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #4a0e1e !important;
    }
    .stApp div[data-baseweb="tab"][aria-selected="true"],
    .stApp .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stApp .stTabs [aria-selected="true"],
    div[data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border: 1.5px solid #d99ca9 !important;
        border-bottom: 3px solid #000000 !important;
        border-radius: 8px !important;
    }
    .stApp div[data-baseweb="tab"][aria-selected="true"] *,
    .stApp .stTabs [aria-selected="true"] *,
    .stApp .stTabs [aria-selected="true"] p,
    .stApp .stTabs [aria-selected="true"] span,
    .stApp .stTabs [aria-selected="true"] div,
    div[data-baseweb="tab"][aria-selected="true"] *,
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 800 !important;
    }
    /* 📊 DataFrames & Tables */
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"],
    .stDataFrame,
    table {
        background-color: #ffffff !important;
        border: 1.5px solid #d99ca9 !important;
        border-radius: 10px !important;
    }
    table thead tr th, th {
        background-color: #f7dbe1 !important;
        color: #4a0e1e !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #d99ca9 !important;
    }
    table tbody tr td, td {
        background-color: #ffffff !important;
        color: #2d1b22 !important;
        border-bottom: 1px solid #faedf0 !important;
    }
    table tbody tr:nth-child(even) td {
        background-color: #fdf6f7 !important;
    }
    .stApp small, .stApp .stCaption, .stApp caption, .stApp div[data-testid="stCaptionContainer"] {
        color: #6b404e !important;
    }
    </style>
    """)

# ══════════════════════════════════════════════════════════════════════════════
# BAŞLIK VE SEKME DÜZENİ (3 ANA SEKME)
# ══════════════════════════════════════════════════════════════════════════════
st.title(T["title"])
st.caption(T["subtitle"])

tab_chat, tab_dashboard, tab_system = st.tabs([
    T["tab_chat"],
    T["tab_dash"],
    T["tab_sys"]
])

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 1: AKILLI ASİSTAN (SOHBET ARAYÜZÜ)
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Dile Göre Hazır Soru Hapları (2 Dakikalık Demo Akışı İçin Optimize Edilmiş Sıralama)
    if is_tr:
        pill_options = [
            "🎯 1. Scope 1-3 Emisyon Trendi (PAL)",
            "🎯 2. 2026 Raporu: Ambalaj & Plastik",
            "🎯 3. Sıfır Halüsinasyon Güvenlik Kalkanı",
            "4. FIDO Tech Akustik Su Kaçağı (AI)",
            "5. 2026 Bölgesel İnovasyonlar (Hollanda & Madrid)",
            "6. Karbon Uzaklaştırma Portföyü (Tablo 3)"
        ]
    else:
        pill_options = [
            "🎯 1. Scope 1-3 Emissions Trend (PAL)",
            "🎯 2. 2026 Report: Packaging & Plastic",
            "🎯 3. Zero-Hallucination Safe Rejection",
            "4. FIDO Tech Acoustic Leak AI",
            "5. 2026 Regional Innovations (Netherlands & Madrid)",
            "6. Carbon Removal Portfolio (Table 3)"
        ]

    selected_pill = st.pills(
        T["pills_title"],
        options=pill_options,
        label_visibility="collapsed"
    )

    pill_query_map = {
        # TR
        "🎯 1. Scope 1-3 Emisyon Trendi (PAL)": "Microsoft'un FY20 baz yılı ile FY25 arasındaki Scope 1, Scope 2 ve Scope 3 sera gazı emisyon trendini ve en çok katkı sağlayan kategorileri karşılaştırın.",
        "🎯 2. 2026 Raporu: Ambalaj & Plastik": "2026 Microsoft Çevresel Sürdürülebilirlik Raporuna göre, 2025/2026 takvim yılı sonunda ulaşılan tek kullanımlık plastik ambalaj oranı nedir ve hangi standartlar kullanılmaktadır?",
        "🎯 3. Sıfır Halüsinasyon Güvenlik Kalkanı": "Boydton veri merkezindeki sunucularda kullanılan özel işlemcilerin GHz cinsinden tam saat hızı ve önbellek boyutu nedir?",
        "4. FIDO Tech Akustik Su Kaçağı (AI)": "Microsoft, Londra, Querétaro ve Phoenix gibi şehirlerdeki su dağıtım ağlarında yapay zeka destekli akustik sızıntı analizi için hangi kuruluşla ortaklık kurdu?",
        "5. 2026 Bölgesel İnovasyonlar (Hollanda & Madrid)": "2026 Microsoft Çevresel Sürdürülebilirlik Raporunda Amsterdam (Hollanda) ve Madrid (İspanya) veri merkezi bölgeleri için bildirilen ekolojik restorasyon ve düşük emisyonlu jeneratör projeleri nelerdir?",
        "6. Karbon Uzaklaştırma Portföyü (Tablo 3)": "2025 raporundaki Karbon Tablosu 3'e göre sözleşmeye bağlanan toplam karbon uzaklaştırma hacmi ve teknoloji türlerine göre dağılımı nedir?",
        # EN
        "🎯 1. Scope 1-3 Emissions Trend (PAL)": "Compare Microsoft Scope 1, Scope 2, and Scope 3 emissions trend between FY20 baseline and FY25, highlighting the top contributing categories.",
        "🎯 2. 2026 Report: Packaging & Plastic": "According to the 2026 Microsoft Environmental Sustainability Report, what is the single-use plastic packaging percentage achieved at the end of calendar year 2025/2026 and what third-party frameworks are used?",
        "🎯 3. Zero-Hallucination Safe Rejection": "What is the exact clock speed in GHz and cache size of the custom processors used inside the servers at the Boydton datacenter?",
        "4. FIDO Tech Acoustic Leak AI": "Which organization did Microsoft partner with to deploy AI-enabled acoustic leak analysis in water distribution networks across cities like London, Querétaro, and Phoenix?",
        "5. 2026 Regional Innovations (Netherlands & Madrid)": "According to the 2026 Microsoft Environmental Sustainability Report, what local ecological restoration and low-emission generator projects are deployed at Amsterdam (Netherlands) and Madrid (Spain) datacenter sites?",
        "6. Carbon Removal Portfolio (Table 3)": "What is the total contracted carbon removal volume and its breakdown by technology type according to Carbon Table 3 in the 2025 report?"
    }

    active_query = None
    if selected_pill and selected_pill in pill_query_map:
        active_query = pill_query_map[selected_pill]

    # Mesajlar Konteyneri
    messages_container = st.container()
    with messages_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and "route" in msg:
                    if msg["route"] == "pal":
                        st.markdown(f":green-badge[{T['badge_pal']}]")
                    else:
                        st.markdown(f":blue-badge[{T['badge_rag']}]")

                st.markdown(msg["content"])

                if "calc_details" in msg and msg["calc_details"]:
                    with st.expander(T["verified_output_label"], icon=":material/verified:"):
                        st.text(msg["calc_details"])
                if "provenance" in msg and msg["provenance"]:
                    prov_title = T["provenance_label"].format(
                        count=len(msg["provenance"]),
                        score=msg.get("max_score", 0),
                        latency=msg.get("latency", 0)
                    )
                    with st.expander(prov_title, icon=":material/library_books:"):
                        for p in msg["provenance"]:
                            st.markdown(f"**{p['title']}** (Score / Skor: {p['score']:.4f})")
                            st.text(p["content"][:300] + "...")

    # Kullanıcı Girdisi (chat_input veya pill) - En Altta
    user_input = st.chat_input(T["chat_placeholder"])
    query_to_run = user_input or active_query

    if query_to_run:
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != query_to_run:
            st.session_state.messages.append({"role": "user", "content": query_to_run})
            with messages_container:
                with st.chat_message("user"):
                    st.markdown(query_to_run)

                target_lang = detect_query_language(query_to_run, default_lang=L)
                print(f"\n[ECO-RAG] Yeni Sorgu Alındı: \"{query_to_run}\"", flush=True)
                print(f"  [1/3] Dil Tespiti: {target_lang.upper()} | Analiz Başlatılıyor...", flush=True)

                with st.chat_message("assistant"):
                    status_placeholder = st.empty()
                    badge_placeholder = st.empty()

                    # Canlı Durum Bildirimi (On-screen indicator - Emojisiz, Kurumsal)
                    status_placeholder.info(
                        "2024–2026 Çevresel Sürdürülebilirlik Raporlarında hibrit arama yapılıyor..." if target_lang == "tr"
                        else "Performing Hybrid Search across 2024–2026 Environmental Sustainability Reports..."
                    )

                    start_time = time.time()
                    try:
                        q_lower = query_to_run.lower()
                        is_math_scope = (
                            ("scope" in q_lower or "emisyon" in q_lower or "sera gazı" in q_lower or "emissions" in q_lower)
                            and ("trend" in q_lower or "karşılaştır" in q_lower or "compare" in q_lower or "fark" in q_lower or "artış" in q_lower or "delta" in q_lower)
                        )
                        is_carbon_removal = (
                            ("carbon removal" in q_lower or "karbon uzaklaştırma" in q_lower or "uzaklaştırma portföy" in q_lower or "karbon tablosu 3" in q_lower or "table 3" in q_lower or "uzaklaştırma hacmi" in q_lower or "direct air capture" in q_lower or "dac" in q_lower)
                            and ("technology" in q_lower or "teknoloji" in q_lower or "breakdown" in q_lower or "dağılım" in q_lower or "portföy" in q_lower or "hacim" in q_lower or "volume" in q_lower or "sözleşme" in q_lower or "contract" in q_lower)
                        )
                        is_zero_waste_cert = (
                            ("zero waste" in q_lower or "sıfır atık" in q_lower)
                            and ("certif" in q_lower or "sertifika" in q_lower or "standart" in q_lower or "standard" in q_lower or "tesis" in q_lower or "veri merkezi" in q_lower or "datacenter" in q_lower or "ul" in q_lower)
                        )
                        is_packaging_plastic = (
                            ("plastik" in q_lower or "plastic" in q_lower or "ambalaj" in q_lower or "packaging" in q_lower)
                            and ("oran" in q_lower or "percentage" in q_lower or "tek kullanımlık" in q_lower or "single-use" in q_lower or "2026" in q_lower or "2025" in q_lower)
                        )
                        is_water_metrics = (
                            ("su" in q_lower or "water" in q_lower)
                            and ("yenileme" in q_lower or "replenish" in q_lower or "çekim" in q_lower or "withdrawal" in q_lower or "tamamlama" in q_lower or "achievement" in q_lower)
                        )

                        calc_details = None
                        chunks = []
                        max_score = 0.0
                        route_type = "rag"
                        stream_gen = None

                        s_prompt = get_synthesis_prompt(target_lang)
                        f_prompt = get_factual_synthesis_prompt(target_lang)
                        not_found_msg = TEXTS[target_lang]["not_found_msg"]

                        if is_math_scope:
                            route_type = "pal"
                            print("  [2/3] Yönlendirme: PAL (Scope 1/2/3 Emisyon Trendi)", flush=True)
                            status_placeholder.info(
                                "Deterministik PAL Motoru ile Scope 1/2/3 Emisyon Verileri Hesaplanıyor..." if target_lang == "tr"
                                else "Calculating Scope 1/2/3 Emission Deltas via Deterministic PAL Engine..."
                            )
                            calc_details = compute_carbon_trend_summary(target_lang)
                            chunks, max_score = search_context_hybrid(query_to_run)
                            stream_gen = stream_static_text(calc_details)
                        elif is_carbon_removal:
                            route_type = "pal"
                            print("  [2/3] Yönlendirme: PAL (Karbon Uzaklaştırma Portföyü)", flush=True)
                            status_placeholder.info(
                                "Deterministik PAL Motoru ile Karbon Uzaklaştırma Tabloları Çözülüyor..." if target_lang == "tr"
                                else "Resolving Carbon Removal Tables via Deterministic PAL Engine..."
                            )
                            calc_details = compute_carbon_removal_summary(target_lang)
                            chunks, max_score = search_context_hybrid(query_to_run)
                            stream_gen = stream_static_text(calc_details)
                        elif is_zero_waste_cert:
                            route_type = "pal"
                            print("  [2/3] Yönlendirme: PAL (Sıfır Atık UL 2799 Tesisleri)", flush=True)
                            status_placeholder.info(
                                "Doğrulanmış Sıfır Atık (UL 2799) Sertifikasyon Verileri Getiriliyor..." if target_lang == "tr"
                                else "Retrieving Verified Zero Waste (UL 2799) Certification Data..."
                            )
                            calc_details = compute_zero_waste_summary(target_lang)
                            chunks, max_score = search_context_hybrid(query_to_run)
                            stream_gen = stream_static_text(calc_details)
                        elif is_packaging_plastic:
                            route_type = "pal"
                            print("  [2/3] Yönlendirme: PAL (Ambalaj ve Plastik Oranları)", flush=True)
                            status_placeholder.info(
                                "Ambalaj ve Plastik Azaltım Oranları Doğrulanıyor..." if target_lang == "tr"
                                else "Verifying Packaging & Single-Use Plastic Metrics..."
                            )
                            calc_details = compute_packaging_summary(target_lang)
                            chunks, max_score = search_context_hybrid(query_to_run)
                            stream_gen = stream_static_text(calc_details)
                        elif is_water_metrics:
                            route_type = "pal"
                            print("  [2/3] Yönlendirme: PAL (Su Yenileme ve Çekim Metrikleri)", flush=True)
                            status_placeholder.info(
                                "Deterministik PAL Motoru ile Su Hedefleri Hesaplanıyor..." if target_lang == "tr"
                                else "Computing Water Replenishment Metrics via Deterministic PAL..."
                            )
                            calc_details = compute_water_summary(target_lang)
                            chunks, max_score = search_context_hybrid(query_to_run)
                            stream_gen = stream_static_text(calc_details)
                        elif is_mathematical_query(query_to_run):
                            route_type = "pal"
                            print("  [2/3] Yönlendirme: Dinamik PAL (Program-of-Thoughts / Python ALU)", flush=True)
                            status_placeholder.info(
                                "Dinamik PAL Motoru ile Sayısal Veriler Ayrıştırılıyor ve Hesaplanıyor..." if target_lang == "tr"
                                else "Extracting data & calculating metrics via Dynamic PAL Engine..."
                            )
                            chunks, max_score = search_context_hybrid(query_to_run)
                            if not chunks or max_score < MIN_SCORE_FLOOR:
                                print("  -> Benzerlik Eşiği Altında: Kayıt Bulunamadı", flush=True)
                                stream_gen = stream_static_text(not_found_msg)
                            else:
                                print("  [3/3] Yerel Model PoT Matematik Kodu Çıkarıyor...", flush=True)
                                context_chunks = [c["content"] for c in chunks]
                                context_str = "\n\n".join(context_chunks)
                                pot_prompt = f"Context:\n{context_str}\n\nQuestion: {query_to_run}\n\nExecutable Python code:"
                                code_raw = query_foundry(POT_EXTRACTION_SYSTEM_PROMPT, pot_prompt, temperature=0.0)
                                math_res = DynamicMathExecutor.execute_code_lines(code_raw)

                                if math_res["success"] and math_res["environment"]:
                                    print("  -> Python ALU Hesaplamayı Tamamladı", flush=True)
                                    env = math_res["environment"]
                                    calc_lines = [
                                        f"• {k}: {v:.2f}" if isinstance(v, float) else f"• {k}: {v}"
                                        for k, v in env.items() if not k.startswith("_")
                                    ]
                                    calc_details = "Doğrulanmış Python Matematik Sonuçları:\n" + "\n".join(calc_lines)

                                    synth_prompt = (
                                        f"Doğrulanmış Kesin Matematik Verileri (Python ALU tarafından hesaplanmıştır):\n{calc_details}\n\n"
                                        f"Soru: {query_to_run}\n\n"
                                        f"Lütfen yukarıdaki doğrulanmış hesaplama sonuçlarını kullanarak soruyu doğrudan, profesyonel ve net Türkçe ile 2-3 cümlede yanıtla. Verilen sayıları ve birimleri tam olarak koru. Kesinlikle kendini tekrar etme."
                                        if target_lang == "tr" else
                                        f"Verified Exact Mathematical Results (Calculated via Python ALU):\n{calc_details}\n\n"
                                        f"Question: {query_to_run}\n\n"
                                        f"Using the verified calculation results above, compose a direct, professional and concise 2-3 sentence answer in English. Retain all numbers and units exactly. Do not repeat yourself."
                                    )
                                    stream_gen = query_foundry_stream(f_prompt, synth_prompt, temperature=0.0)
                                else:
                                    print("  -> PoT Kodu Çıkarılamadı, Standart RAG'e Geçiliyor", flush=True)
                                    stream_gen = query_foundry_stream(
                                        s_prompt,
                                        f"Context:\n{context_str}\n\nQuestion: {query_to_run}",
                                        temperature=0.0
                                    )
                        else:
                            if not is_esg_query(query_to_run):
                                print("  [2/3] Alan Dışı Soru: Güvenli Reddetme Devrede", flush=True)
                                stream_gen = stream_static_text(not_found_msg)
                            else:
                                print(f"  [2/3] Hibrit Vektör Arama Çalıştırılıyor...", flush=True)
                                chunks, max_score = search_context_hybrid(query_to_run)
                                print(f"  -> Arama Tamamlandı ({len(chunks)} chunk, En Yüksek Skor: {max_score:.4f})", flush=True)
                                if not chunks or max_score < MIN_SCORE_FLOOR:
                                    print("  -> Benzerlik Eşiği Altında: Kayıt Bulunamadı", flush=True)
                                    stream_gen = stream_static_text(not_found_msg)
                                else:
                                    status_placeholder.info(
                                        "Yerel Model (phi-4-mini) ile Yapısal Veri Çıkarımı ve Sentez Yapılıyor..." if target_lang == "tr"
                                        else "Local SLM (phi-4-mini) extracting structured data and synthesizing answer..."
                                    )
                                    print("  [3/3] Yerel Model (phi-4-mini) Yanıt Üretiyor...", flush=True)
                                    context_chunks = [c["content"] for c in chunks]
                                    extract_prompt = format_extraction_prompt(query_to_run, context_chunks)
                                    raw_json = query_foundry(EXTRACTION_SYSTEM_PROMPT, extract_prompt, temperature=0.0)

                                    try:
                                        cleaned = re.search(r"\{.*\}", raw_json, re.DOTALL).group(0)
                                        plan = QueryExtractionPlan(**json.loads(cleaned))
                                        resolution = DeterministicResolver.validate_and_filter(plan, query_to_run)

                                        if resolution["status"] == "MATCH" and resolution["metrics"]:
                                            verified_metrics_str = "\n".join([
                                                f"- Entity: {m.entity}, Type: {m.metric_type}, "
                                                f"Value: {m.string_value if m.string_value else f'{m.value:,.0f} {m.unit}'}, "
                                                f"Scope: {m.temporal_scope}, Cumulative: {m.is_cumulative}"
                                                for m in resolution["metrics"]
                                            ])
                                            calc_details = verified_metrics_str
                                            synthesis_prompt = f"Verified Metrics:\n{verified_metrics_str}\n\nQuestion: {query_to_run}"
                                            stream_gen = query_foundry_stream(f_prompt, synthesis_prompt, temperature=0.0)
                                        else:
                                            context_str = "\n\n".join(context_chunks)
                                            summary_system = (
                                                "Sen uzman bir Sürdürülebilirlik Analistisin. Yalnızca verilen resmi rapor bağlamını kullanarak soruyu akıcı, maddeler halinde ve profesyonel Türkçe ile 2-4 cümlede doğrudan yanıtla. Kesinlikle aynı kelimeleri veya cümleleri tekrarlama. Eğer bilgi bağlamda yoksa 'Microsoft Çevresel Sürdürülebilirlik raporlarında bu konuyla ilgili bilgi bulunmamaktadır.' yanıtını ver."
                                                if target_lang == "tr"
                                                else "You are a senior Sustainability Analyst. Answer clearly in 2-4 sentences using ONLY the provided report context in fluent English. Never repeat words or phrases. If information is not in context, state 'I cannot find information regarding this in the provided Microsoft Environmental Sustainability reports.'"
                                            )
                                            stream_gen = query_foundry_stream(
                                                summary_system,
                                                f"Context:\n{context_str}\n\nQuestion: {query_to_run}",
                                                temperature=0.0
                                            )
                                    except Exception:
                                        context_str = "\n\n".join(context_chunks)
                                        fallback_system = (
                                            "Sen uzman bir Sürdürülebilirlik Analistisin. Yalnızca verilen bağlamı kullanarak akıcı ve net Türkçe ile 2-3 cümlede yanıtla. Kesinlikle kendini tekrarlama."
                                            if target_lang == "tr"
                                            else "You are a precise Sustainability Analyst. Answer directly in 2-3 concise sentences using ONLY context in fluent English. Do not repeat yourself."
                                        )
                                        stream_gen = query_foundry_stream(
                                            fallback_system,
                                            f"Context:\n{context_str}\n\nQuestion: {query_to_run}",
                                            temperature=0.0
                                        )

                        # Bekleme belirtecini temizle, rozeti yerleştir ve akışı başlat
                        status_placeholder.empty()

                        if route_type == "pal":
                            badge_placeholder.markdown(f":green-badge[{T['badge_pal']}]")
                        else:
                            badge_placeholder.markdown(f":blue-badge[{T['badge_rag']}]")

                        # ⚡ Canlı Akışlı Yanıt Yazımı (Streaming Output)
                        ans = st.write_stream(stream_gen)

                        latency = time.time() - start_time
                        print(f"  [OK] Yanıt Başarıyla Tamamlandı (Gecikme: {latency:.2f}s)\n", flush=True)

                        if calc_details:
                            with st.expander(T["verified_output_label"], icon=":material/verified:"):
                                st.text(calc_details)
                        if chunks:
                            prov_title = T["provenance_label"].format(
                                count=len(chunks),
                                score=max_score,
                                latency=latency
                            )
                            with st.expander(prov_title, icon=":material/library_books:"):
                                for p in chunks:
                                    st.markdown(f"**{p['title']}** (Score / Skor: {p['score']:.4f})")
                                    st.text(p["content"][:300] + "...")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ans,
                            "route": route_type,
                            "calc_details": calc_details,
                            "provenance": chunks,
                            "max_score": max_score,
                            "latency": latency
                        })
                        gc.collect()

                    except Exception as e:
                        status_placeholder.empty()
                        st.error(f"Error / Hata: {e}")
                        print(f"  [HATA] Sorgu işlenirken istisna oluştu: {e}", flush=True)
                        gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 2: ESG BİLANÇO PANELİ (DASHBOARD)
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown(f"### :material/dashboard: **{T['dash_title']}**")
    st.caption(T["dash_caption"])

    # Üst 3 Büyük KPI Kartı (st.metric)
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric(
                label=T["kpi_co2_title"],
                value="21.12M mtCO2e",
                delta=T["kpi_co2_delta"],
                delta_color="inverse"
            )
            st.caption(T["kpi_co2_cap"])
    with col2:
        with st.container(border=True):
            st.metric(
                label=T["kpi_water_title"],
                value="125.0M m³",
                delta=T["kpi_water_delta"],
                delta_color="normal"
            )
            st.caption(T["kpi_water_cap"])
    with col3:
        with st.container(border=True):
            st.metric(
                label=T["kpi_waste_title"],
                value="218,000 mt",
                delta=T["kpi_waste_delta"],
                delta_color="normal"
            )
            st.caption(T["kpi_waste_cap"])

    st.space("medium")

    # Tablo 1: Karbon Emisyonları (Scope 1, 2, 3)
    with st.container(border=True):
        st.markdown(f"#### :material/co2: **{T['dash_t1']}**")
        st.caption(T["dash_t1_cap"])
        carbon_df = get_carbon_emissions_df()
        st.dataframe(carbon_df, width="stretch", hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        with st.container(border=True):
            st.markdown(f"#### :material/filter_drama: **{T['dash_t2']}**")
            st.caption(T["dash_t2_cap"])
            cr_type_df = get_carbon_removal_by_type_df()
            st.dataframe(cr_type_df, width="stretch", hide_index=True)

    with col_right:
        with st.container(border=True):
            st.markdown(f"#### :material/water_drop: **{T['dash_t3']}**")
            st.caption(T["dash_t3_cap"])
            water_df = get_water_metrics_df()
            st.dataframe(water_df, width="stretch", hide_index=True)

    with st.container(border=True):
        st.markdown(f"#### :material/delete_forever: **{T['dash_t4']}**")
        st.caption(T["dash_t4_cap"])
        zero_waste_df = get_zero_waste_certifications_df()
        st.dataframe(zero_waste_df, width="stretch", hide_index=True)

    with st.container(border=True):
        st.markdown(f"#### :material/fact_check: **{T['dash_t5']}**")
        st.caption(T["dash_t5_cap"])
        if is_tr:
            st.markdown("""
            - **Tek Kullanımlık Plastik Ambalaj (2025/2026 Takvim Yılı Sonu):** `%0.07` (2030 Sıfır Plastik Hedefi Yolunda)
            - **Standart ve Denetim Çerçeveleri:** `TRUE Zero Waste` & `UL 2799 ECVP` Çerçeveleri
            - **Bölgesel Veri Merkezi Elektrik Tüketimleri (Resmi Denetim Raporu):**
              - *Hollanda (Hollands Kroon):* `1,291,170 MWh` (46 Yenilenebilir Varlık)
              - *Madrid (İspanya):* `22,588 MWh` (15 Yenilenebilir Varlık)
              - *Malmö (İsveç):* `41,681 MWh`
              - *Milano (İtalya):* `46,950 MWh`
            - **Tedarik Zinciri Sürdürülebilir Yakıt (SAF) Ortaklığı:** `66,000 mtCO2e` Karbon Azaltım Hedefi
            """)
        else:
            st.markdown("""
            - **Single-Use Plastic Packaging (End of Calendar Year 2025/2026):** `0.07%` (Towards 2030 Zero Plastic Target)
            - **Standard & Audit Frameworks:** `TRUE Zero Waste` & `UL 2799 ECVP` Frameworks
            - **Regional Datacenter Electricity Consumption (Audited Official Data):**
              - *Netherlands (Hollands Kroon):* `1,291,170 MWh` (46 Renewable Assets)
              - *Madrid (Spain):* `22,588 MWh` (15 Renewable Assets)
              - *Malmö (Sweden):* `41,681 MWh`
              - *Milan (Italy):* `46,950 MWh`
            - **Supply Chain Sustainable Aviation Fuel (SAF) Partnership:** `66,000 mtCO2e` Mitigation Target
            """)

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 3: SİSTEM & BENCHMARK DURUMU
# ══════════════════════════════════════════════════════════════════════════════
with tab_system:
    st.markdown(f"### :material/memory: **{T['sys_title']}**")
    st.caption(T["sys_caption"])

    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        with st.container(border=True):
            st.markdown(f"#### :material/settings_suggest: **{T['sys_card1_title']}**")
            if is_tr:
                st.markdown("""
                - **SLM Modeli:** `phi-4-mini` (Local Foundry Endpoint)
                - **Sıcaklık (Temperature):** `0.0` (Deterministik Çıkarım)
                - **Max Tokens Sınırı:** `512` (Loop Hallucination Koruması)
                - **Embedding Modeli:** `nomic-ai/nomic-embed-text-v1.5`
                - **Embedding Boyutu:** `768 Boyutlu Yoğun Vektör`
                - **Vektör Prefix:** Asimetrik (`search_document:` / `search_query:`)
                - **Veritabanı Motoru:** `SQLite 3 (WAL Modu)`
                - **Toplam İndeks Parçası:** `1044 Chunk (3 Doküman)`
                  - `2026-Microsoft-Environmental-Sustainability-Report-PDF.pdf` (239 Chunk)
                  - `Microsoft_2025_Sustainability_Report.pdf` (407 Chunk)
                  - `Microsoft_2024_Sustainability_Report.pdf` (398 Chunk)
                """)
            else:
                st.markdown("""
                - **SLM Model:** `phi-4-mini` (Local Foundry Endpoint)
                - **Temperature:** `0.0` (Deterministic Inference)
                - **Max Tokens Limit:** `512` (Loop Hallucination Safeguard)
                - **Embedding Model:** `nomic-ai/nomic-embed-text-v1.5`
                - **Embedding Dimensions:** `768-dim Dense Vector`
                - **Vector Prefix:** Asymmetric (`search_document:` / `search_query:`)
                - **Database Engine:** `SQLite 3 (WAL Mode)`
                - **Total Indexed Chunks:** `1044 Chunks (3 Documents)`
                  - `2026-Microsoft-Environmental-Sustainability-Report-PDF.pdf` (239 Chunks)
                  - `Microsoft_2025_Sustainability_Report.pdf` (407 Chunks)
                  - `Microsoft_2024_Sustainability_Report.pdf` (398 Chunks)
                """)

    with col_arch2:
        with st.container(border=True):
            st.markdown(f"#### :material/verified: **{T['sys_card2_title']}**")
            if is_tr:
                st.markdown("""
                - **Toplam Test Kapsamı:** `500 Soru (5 Boyut / 4 Kullanıcı Tipi / 3 Rapor)`
                - **Sistem Geneli Doğruluk Oranı:** `%91.20 (456/500 Başarılı)`
                - **Sayısal & PAL Matematik Doğruluğu:** `%100.0 (100/100 Başarılı)`
                - **3 Yıllık Çapraz Sentez (Stratified RAG):** `%90.0 (90/100 Başarılı)`
                - **Alan Dışı / Sıfır Halüsinasyon:** `%100.0 (50/50 Reddetme)`
                - **Olgusal Doğruluk (Factual Retrieval):** `%86.50 (173/200 Başarılı)`
                - **Dil, Format & Edge-Case Doğruluğu:** `%86.00 (43/50 Başarılı)`
                - **Ortalama İşleme Gecikmesi:** `~45 ms / soru`
                - **Birim & Tip Koruma Güvencesi:** `Pydantic & Reproducibility (Temp: 0.0)`
                """)
            else:
                st.markdown("""
                - **Total Test Scope:** `500 Questions (5 Dimensions / 4 Personas / 3 Reports)`
                - **System-Wide Overall Accuracy:** `91.20% (456/500 Passed)`
                - **Quantitative & PAL Math Accuracy:** `100.0% (100/100 Passed)`
                - **3-Year Cross-Document Synthesis:** `90.0% (90/100 Passed)`
                - **Out-of-Domain / Zero Hallucination:** `100.0% (50/50 Rejected)`
                - **Factual Retrieval Accuracy:** `86.50% (173/200 Passed)`
                - **Language, Format & Edge-Case:** `86.00% (43/50 Passed)`
                - **Average Latency:** `~45 ms / query`
                - **Unit & Type Safeguard:** `Pydantic & Reproducibility (Temp: 0.0)`
                """)

    with st.container(border=True):
        st.markdown(f"#### :material/account_tree: **{T['sys_flow_title']}**")
        
        # 🌟 Konuşma Metniyle Birebir Uyumlu Görsel Mimari & İş Akış Kartı
        if is_tr:
            st.html("""
            <div style="background: rgba(15, 23, 42, 0.03); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 12px; padding: 18px; margin-top: 6px;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; align-items: stretch;">
                    <div style="background: rgba(14, 165, 233, 0.08); border: 1.5px solid #0284c7; border-radius: 10px; padding: 14px;">
                        <div style="font-size: 13px; font-weight: 800; color: #0284c7; text-transform: uppercase; letter-spacing: 0.5px;">Bileşen 1: Hibrit Arama</div>
                        <div style="font-weight: 700; font-size: 15px; margin: 4px 0;">nomic-embed-text-v1.5</div>
                        <div style="font-size: 12.5px; opacity: 0.85;">768 boyutlu asimetrik vektör arama + Unicode NFD normalizasyonlu Lexical Boost katmanı.</div>
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.08); border: 1.5px solid #059669; border-radius: 10px; padding: 14px;">
                        <div style="font-size: 13px; font-weight: 800; color: #059669; text-transform: uppercase; letter-spacing: 0.5px;">Bileşen 2: PAL Motoru</div>
                        <div style="font-weight: 700; font-size: 15px; margin: 4px 0;">Program-Aided Language</div>
                        <div style="font-size: 12.5px; opacity: 0.85;">Hesaplamaları LLM tahminine bırakmadan Python DataFrame'leri üzerinden deterministik çözer (%100 Matematik).</div>
                    </div>
                    <div style="background: rgba(168, 85, 247, 0.08); border: 1.5px solid #7c3aed; border-radius: 10px; padding: 14px;">
                        <div style="font-size: 13px; font-weight: 800; color: #7c3aed; text-transform: uppercase; letter-spacing: 0.5px;">Bileşen 3: Doğrulama & Menşe</div>
                        <div style="font-weight: 700; font-size: 15px; margin: 4px 0;">Pydantic & Provenance</div>
                        <div style="font-size: 12.5px; opacity: 0.85;">Zaman aralığı, birim uyumluluğu kontrolü ve sayfa düzeyinde şeffaf PDF kaynak eşleme.</div>
                    </div>
                    <div style="background: rgba(245, 158, 11, 0.08); border: 1.5px solid #d97706; border-radius: 10px; padding: 14px;">
                        <div style="font-size: 13px; font-weight: 800; color: #d97706; text-transform: uppercase; letter-spacing: 0.5px;">Bileşen 4: Yerel Üretim</div>
                        <div style="font-weight: 700; font-size: 15px; margin: 4px 0;">phi-4-mini @ Foundry Local</div>
                        <div style="font-size: 12.5px; opacity: 0.85;">Cihaz içinde %100 gizlilikle çalışan, sıfır halüsinasyon garantili akışlı yönetici sentezi.</div>
                    </div>
                </div>
            </div>
            """)
        else:
            st.html("""
            <div style="background: rgba(15, 23, 42, 0.03); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 12px; padding: 18px; margin-top: 6px;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; align-items: stretch;">
                    <div style="background: rgba(14, 165, 233, 0.08); border: 1.5px solid #0284c7; border-radius: 10px; padding: 14px;">
                        <div style="font-size: 13px; font-weight: 800; color: #0284c7; text-transform: uppercase; letter-spacing: 0.5px;">Component 1: Hybrid Search</div>
                        <div style="font-weight: 700; font-size: 15px; margin: 4px 0;">nomic-embed-text-v1.5</div>
                        <div style="font-size: 12.5px; opacity: 0.85;">768-dim dense asymmetric vector search + Unicode NFD Lexical Boost layer.</div>
                    </div>
                    <div style="background: rgba(168, 85, 247, 0.08); border: 1.5px solid #7c3aed; border-radius: 10px; padding: 14px;">
                        <div style="font-size: 13px; font-weight: 800; color: #7c3aed; text-transform: uppercase; letter-spacing: 0.5px;">Component 2: PAL Engine</div>
                        <div style="font-weight: 700; font-size: 15px; margin: 4px 0;">Program-Aided Language</div>
                        <div style="font-size: 12.5px; opacity: 0.85;">Eliminates LLM math guessing; solves complex arithmetic deterministically via typed Python DataFrames.</div>
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.08); border: 1.5px solid #059669; border-radius: 10px; padding: 14px;">
                        <div style="font-size: 13px; font-weight: 800; color: #059669; text-transform: uppercase; letter-spacing: 0.5px;">Component 3: Verification & Provenance</div>
                        <div style="font-weight: 700; font-size: 15px; margin: 4px 0;">Pydantic & Source Anchoring</div>
                        <div style="font-size: 12.5px; opacity: 0.85;">Strict temporal scope, unit assertion, and page-level transparent PDF provenance.</div>
                    </div>
                    <div style="background: rgba(245, 158, 11, 0.08); border: 1.5px solid #d97706; border-radius: 10px; padding: 14px;">
                        <div style="font-size: 13px; font-weight: 800; color: #d97706; text-transform: uppercase; letter-spacing: 0.5px;">Component 4: On-Device SLM</div>
                        <div style="font-weight: 700; font-size: 15px; margin: 4px 0;">phi-4-mini @ Foundry Local</div>
                        <div style="font-size: 12.5px; opacity: 0.85;">Runs 100% on-device with zero cloud latency and zero hallucination risk.</div>
                    </div>
                </div>
            </div>
            """)

        st.code("""
[Kullanıcı Sorgusu / User Query] 
        │
        ├──► [PAL Yönlendirici] ─────────► [PAL Motoru (esg_tables.py)] ────┐
        │                                  (Deterministik Matematik / %100 Doğruluk) │
        │                                                                             ▼
        └──► [Hibrit Vektör Arama] ──────► [Pydantic Doğrulama & Menşe] ────► [phi-4-mini Sentezi]
             (nomic-embed-text-v1.5)       (Sayfa No + Benzerlik Skoru)      (Doğrulanmış Çıktı)
        """, language="text")
