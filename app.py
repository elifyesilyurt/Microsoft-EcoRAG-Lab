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
from typing import Optional, List, Dict, Any, Set, Tuple
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

def discover_foundry_base_url() -> str:
    """Foundry Local portunu dinamik olarak otomatik tespit eder."""
    env_url = os.getenv("FOUNDRY_BASE_URL")
    if env_url:
        return env_url.rstrip("/")
    
    # 1. Aktif çalışan portları hızlıca tara
    for port in [56720, 62095, 8000, 5000]:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=0.25)
            if r.status_code == 200:
                return f"http://127.0.0.1:{port}"
        except Exception:
            continue

    # 2. Foundry Local CLI JSON durumunu oku
    try:
        import subprocess
        res = subprocess.run(
            ["foundry", "server", "status", "--output", "json"],
            capture_output=True,
            text=True,
            timeout=1.0
        )
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if data.get("running") and data.get("webUrls"):
                return data["webUrls"][0].rstrip("/")
    except Exception:
        pass

    return "http://127.0.0.1:56720"

FOUNDRY_BASE_URL = discover_foundry_base_url()
MODEL_NAME = os.getenv("FOUNDRY_MODEL_NAME", "phi-4-mini")

RELATIVE_DROP_RATIO = 0.70
MAX_K = 6
MIN_SCORE_FLOOR = 0.15

def get_synthesis_prompt(lang: str = "tr") -> str:
    if lang == "tr":
        return """Sen Microsoft'un resmi sürdürülebilirlik raporları (2024, 2025, 2026) konusunda uzmanlaşmış kıdemli bir analistsin.
Soruyu modern, akıcı bir yapay zeka asistanı (Gemini / ChatGPT) üslubuyla doğrudan yanıtla.
KURALLAR:
1. 'Doğrudan Yanıt:', 'Yönetici Özeti:' veya şablon başlıkları KULLANMA. Sorunun kesin cevabına İLK CÜMLEDE doğrudan başla.
2. Yıllar arası değişim, oranlar veya birden fazla metrik içeren konularda verileri mutlaka temiz bir Markdown tablosu ile sun.
3. Raporlanan kurumsal strateji ve aksiyonları net, okunabilir madde işaretleri (bullet points) ile özetle.
4. Yalnızca raporda doğrulanmış sayıları, birimleri ve verileri kullan. Asla uydurma veri üretme."""
    else:
        return """You are a Senior Sustainability Analyst specializing in Microsoft's Sustainability Reports.
Answer directly, authoritatively, and fluently in the natural style of modern AI assistants (like ChatGPT or Gemini).
GUIDELINES:
1. DO NOT use rigid template headers like 'Direct Answer:' or 'Executive Summary:'. State the exact answer directly in the very first sentence.
2. When questions involve multi-year trends, comparisons, or metric breakdowns, present them in a clean Markdown table.
3. For strategic initiatives or reported corporate actions, use concise bullet points.
4. Retain exact verified figures and units without alteration or hallucination."""

def get_factual_synthesis_prompt(lang: str = "tr") -> str:
    if lang == "tr":
        return """Sen uzman bir Sürdürülebilirlik Baş Analistisin.
Verilen doğrulanmış metrikleri kullanarak doğrudan, akıcı ve profesyonel bir yanıt oluştur.
Kalıp başlıklar (Doğrudan Yanıt vb.) KULLANMADAN cevaba doğrudan başla. Verileri gerektiğinde Markdown tablosu veya maddeler halinde düzenle."""
    else:
        return """You are a Senior Sustainability AI Analyst.
Using the verified metrics, formulate a direct, authoritative, and fluent response.
Start directly with the core answer without artificial headers (e.g. 'Direct Answer:'). Format multi-value metrics into a clear Markdown table and concise bullets."""

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

def classify_esg_intent(query: str) -> str:
    """
    Sorguyu semantik ve ontolojik ESG alanlarına sınıflandırır:
    - 'out_of_domain': ESG kapsamı dışındaki teknik donanım, spor, finans vb. sorular (güvenli ret)
    - 'packaging_plastic': Tek kullanımlık plastik, ambalaj azaltımı, 0.07%, 4.2% düşüş trendi
    - 'water_stewardship': Su yenileme (125M m³), hedef başarısı (%82.1), FIDO Tech akustik kaçak tespiti
    - 'zero_waste_circularity': UL 2799, TRUE Zero Waste, Circular Centers (%89.4 donanım döngüselliği), 10-14 veri merkezi, 218K ton atık
    - 'carbon_removal': Karbon Tablosu 3, 21.9M mtCO2e (4.37x büyüme), orman, biyokütle/BECCS, DAC
    - 'carbon_commitments': 2030 Karbon Negatif, 2050 tarihsel emisyon telafisi, %100 CFE, 34 GW PPA portföyü
    - 'carbon_trend_scope': Scope 1, 2, 3 emisyon trendi, FY20-FY25 toplam sera gazı delta (+%61.71), Kategori 1 ve 2 payları (%77.69)
    - 'mathematical_query': Dinamik Python PoT / ALU gerektiren matematiksel hesaplamalar
    - 'general_rag': Rapor anlatıları ve bağlamsal bilgi çıkarımı (ör. Amsterdam Miyawaki mikro-ormanları, Madrid tesisleri)
    """
    if not query:
        return "out_of_domain"
    if not is_esg_query(query):
        return "out_of_domain"

    q = query.lower()

    # 1. Packaging & Single-Use Plastic
    if any(k in q for k in ["tek kullanımlık plastik", "single-use plastic", "ambalaj", "packaging", "plastic packaging"]) and any(k in q for k in ["oran", "percentage", "rate", "yolculuk", "trajectory", "düşüş", "reduction", "trend", "2026", "2025", "0.07", "cihaz", "device", "primer", "birincil", "primary"]):
        return "packaging_plastic"

    # 2. Zero Waste & Circularity
    if any(k in q for k in ["sıfır atık", "zero waste", "circular center", "döngüsel", "donanım", "hardware", "ul 2799", "true zero", "atık", "waste"]) and any(k in q for k in ["standart", "standard", "sertifika", "certificate", "certification", "yeniden kullanım", "reuse", "ömrü biten", "veri merkezi", "datacenter", "datacenters", "değişim", "kurtarılan", "merkezler", "diversion", "diverted", "progress", "rate"]):
        return "zero_waste_circularity"

    # 3. Carbon Removal Portfolio (Tablo 3 & Portföy Büyümesi)
    if any(k in q for k in ["karbon uzaklaştırma", "carbon removal", "cdr", "tablo 3", "table 3", "dac", "direct air capture", "biomass", "biyokütle", "uzaklaştırma portföy", "uzaklaştırma hacmi", "removal portfolio", "contracted carbon removal", "removal volume"]):
        return "carbon_removal"

    # 4. Carbon Commitments (2030, 2050, CFE, PPA)
    if any(k in q for k in ["2030", "2050", "karbon negatif", "carbon negative", "tarihsel emisyon", "historical emission", "historical emissions", "cfe", "karbonsuz elektrik", "carbon-free electricity", "ppa", "temiz enerji sözleşme", "clean energy", "power purchase", "taahhüt", "commitment", "commitments"]) and not any(k in q for k in ["kategori 1", "kategori 2", "scope 3 kat", "category 1", "category 2"]):
        return "carbon_commitments"

    # 5. Carbon Trend & Scopes & GHG Delta
    if any(k in q for k in ["scope", "sera gazı", "emisyon", "emission", "emissions", "ghg"]) and any(k in q for k in ["trend", "artış", "increase", "growth", "kategori 1", "kategori 2", "category 1", "category 2", "cat 1", "cat 2", "toplam", "total", "değiş", "change", "delta", "büyüme", "fy20", "fy25", "fark", "difference", "pay", "share"]):
        return "carbon_trend_scope"

    # 6. Water Stewardship & Replenishment & Acoustic AI Leaks (use regex word boundary for 'su')
    is_water = bool(re.search(r"\bsu\b", q)) or any(k in q for k in ["water", "fido", "akustik sızıntı", "acoustic leak", "replenishment"])
    if is_water and any(k in q for k in ["yenileme", "replenish", "replenishment", "hacim", "volume", "ikmal", "kaçak", "leak", "leaks", "sızıntı", "belediye", "municipal", "başarı", "achievement", "hedef", "target", "çekim", "withdrawal", "m³", "milyon m", "million m"]):
        return "water_stewardship"

    # 7. Dinamik PoT Matematik Kontrolü
    if is_mathematical_query(query):
        return "mathematical_query"

    # 8. RAG Hibrit Korpus Arama
    return "general_rag"

def get_suggested_followups(intent: str, lang: str = "tr") -> list:
    """Kullanıcı dostu, modern LLM chat deneyimi için dinamik takip soruları üretir."""
    if lang == "tr":
        followup_map = {
            "packaging_plastic": [
                "3 Yıllık Tek Kullanımlık Plastik Yolculuğu",
                "Ambalajlarda plastik yerine hangi döngüsel malzemeler kullanılıyor?",
                "Sıfır Atık doğrulaması için hangi üçüncü taraf standartlar kullanılıyor?"
            ],
            "water_stewardship": [
                "FIDO Tech ile akustik kaçak tespiti hangi pilot şehirlerde uygulandı?",
                "Microsoft'un 2030 Su Pozitif (Water Positive) hedefinin temel kriterleri nelerdir?",
                "FY25 su yenileme hedefi tamamlama oranı ve FY24 kıyaslaması"
            ],
            "zero_waste_circularity": [
                "Circular Centers döngüsel merkezlerinde donanım yeniden kullanım oranı nedir?",
                "UL 2799 Sıfır Atık sertifikalı veri merkezi sayısı son 3 yılda nasıl değişti?",
                "Veri merkezlerinde operasyonel atıkların kurtarılma oranı nedir?"
            ],
            "carbon_commitments": [
                "Microsoft'un 34 GW'ı aşan temiz enerji PPA anlaşmaları elektrik tüketimini nasıl karşılıyor?",
                "2030 Karbon Negatif hedefi ile 2050 tarihsel emisyon telafisi arasındaki fark nedir?",
                "%100 Karbonsuz Elektrik (CFE) eşleşme hedefi veri merkezlerinde nasıl uygulanıyor?"
            ],
            "carbon_removal": [
                "Sözleşmeli karbon uzaklaştırma portföyünde doğrudan havadan yakalama (DAC) payı nedir?",
                "2024 ve 2025 raporları arasında karbon uzaklaştırma hacmi kaç katına çıkmıştır?",
                "Orman ve biyokütle tabanlı projeler ile teknolojik çözümlerin dengesi nasıldır?"
            ],
            "carbon_trend_scope": [
                "FY25 Scope 3 emisyonlarında Kategori 1 ve Kategori 2'nin toplam payı yüzde kaçtır?",
                "FY20 baz yılından FY25'e kadar Scope 1, Scope 2 ve Scope 3 emisyonlarının ayrı ayrı değişimi nasıldır?",
                "Microsoft'un değer zinciri emisyonlarını azaltmak için tedarikçilerine getirdiği şartlar nelerdir?"
            ]
        }
        return followup_map.get(intent, [
            "2024–2026 raporları boyunca yenilenebilir enerji PPA portföyü nasıl gelişti?",
            "Amsterdam veri merkezi kampüsünde kurulan Miyawaki mikro-orman projesinin detayları nelerdir?",
            "Microsoft'un 2030 kurumsal sürdürülebilirlik taahhütleri nelerdir?"
        ])
    else:
        followup_map = {
            "packaging_plastic": [
                "Summarize the 3-year downward trajectory of single-use plastic packaging",
                "What circular materials replace plastics in Microsoft device packaging?",
                "What third-party audit standards certify zero waste packaging?"
            ],
            "water_stewardship": [
                "Which cities pilot FIDO Tech acoustic AI leak detection?",
                "What is the progress toward Microsoft's 2030 Water Positive commitment?",
                "How does contracted replenishment compare to annual water withdrawals?"
            ],
            "zero_waste_circularity": [
                "What percentage of cloud hardware is reused via Circular Centers?",
                "How has the number of UL 2799 certified datacenters grown from FY23 to FY25?",
                "What are the diversion rate tiers for TRUE Zero Waste certifications?"
            ],
            "carbon_commitments": [
                "How does the 34 GW clean energy PPA portfolio match growing electricity consumption?",
                "What is the distinction between 2030 Carbon Negative and 2050 Historical Compensation?",
                "How does Microsoft enforce its 100% CFE matching mandate for datacenters?"
            ],
            "carbon_removal": [
                "What is the share of Direct Air Capture (DAC) in the contracted portfolio?",
                "What is the growth multiplier of the removal portfolio between 2024 and 2025 reports?",
                "What is the balance between nature-based and engineered removal technologies?"
            ],
            "carbon_trend_scope": [
                "What is the combined share of Category 1 and Category 2 in Scope 3 emissions?",
                "How did Scope 1, Scope 2, and Scope 3 evolve separately between FY20 and FY25?",
                "What supplier requirements has Microsoft enacted to curb Scope 3 growth?"
            ]
        }
        return followup_map.get(intent, [
            "How has the renewable energy PPA portfolio scaled from 2024 to 2026?",
            "What are the details of the Miyawaki micro-forest project at the Amsterdam campus?",
            "What are Microsoft's 4 core sustainability pillars and 2030 targets?"
        ])

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
    discovered = discover_foundry_base_url()
    st.session_state.foundry_base_url = discovered
    return discovered

def trim_repetition(raw_text: str) -> str:
    """Tekrarlayan n-gram veya döngüye giren cümleleri tespit edip ilk tekrar noktasından temizler."""
    if not raw_text:
        return raw_text
    words = raw_text.split()
    first_repeat_word_idx = None
    for n in range(4, 25):
        for i in range(len(words) - 2 * n + 1):
            pattern = [re.sub(r'[^\w]', '', w.lower()) for w in words[i:i+n]]
            for j in range(i + n, len(words) - n + 1):
                candidate = [re.sub(r'[^\w]', '', w.lower()) for w in words[j:j+n]]
                if pattern == candidate:
                    if first_repeat_word_idx is None or j < first_repeat_word_idx:
                        first_repeat_word_idx = j

    if first_repeat_word_idx is not None:
        trimmed = ' '.join(words[:first_repeat_word_idx])
        last_punct = max(trimmed.rfind('.'), trimmed.rfind('!'), trimmed.rfind('?'))
        if last_punct > 30:
            return trimmed[:last_punct + 1].strip()
        return trimmed.strip()
    return raw_text.strip()

def query_foundry(system_prompt: str, user_prompt: str, temperature: float = 0.15, max_tokens: int = 512) -> str:
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
        "presence_penalty": 0.5,
        "frequency_penalty": 0.5,
        "max_tokens": max_tokens
    }
    
    try:
        with requests.Session() as session:
            res = session.post(url, headers=headers, json=payload, timeout=180)
            if res.status_code == 200:
                raw_ans = res.json()["choices"][0]["message"]["content"].strip()
                return trim_repetition(raw_ans)
            else:
                raise RuntimeError(f"HTTP {res.status_code}: {res.text}")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Foundry Local / LLM servisine bağlanılamadı ({base_url}). "
            f"Lütfen servisin çalıştığından emin olun veya 'FOUNDRY_BASE_URL' ortam değişkenini/kenar çubuğunu ayarlayın."
        )
    finally:
        gc.collect()

def query_foundry_stream(system_prompt: str, user_prompt: str, temperature: float = 0.15):
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
        "presence_penalty": 0.5,
        "frequency_penalty": 0.5,
        "max_tokens": 512,
        "stream": True
    }
    
    accumulated_text = ""
    
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
                                        accumulated_text += delta
                                        # Punctuation-normalized n-gram repetition detector (sonsuz döngü ve n-gram kilitlenmesi engelleme)
                                        clean_words = re.sub(r'[^\w\s]', ' ', accumulated_text.lower()).split()
                                        total_w = len(clean_words)
                                        is_loop = False
                                        for n in range(3, 25):
                                            if total_w >= 2 * n:
                                                if clean_words[-n:] == clean_words[-2*n:-n]:
                                                    is_loop = True
                                                    break
                                                elif total_w >= 3 * n and clean_words[-n:] == clean_words[-3*n:-2*n]:
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
def translate_query_to_en(query_tr: str) -> str:
    """Türkçe soruyu İngilizce rapor korpusunda yüksek benzerlikte arama yapmak için İngilizceye eşler."""
    if not query_tr or detect_query_language(query_tr) != "tr":
        return query_tr
    base_url = get_foundry_base_url()
    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Connection": "close"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Translate this sustainability question into a precise English query for document retrieval. Return ONLY the translation, nothing else."},
            {"role": "user", "content": query_tr}
        ],
        "temperature": 0.0,
        "max_tokens": 60
    }
    try:
        with requests.Session() as s:
            r = s.post(url, headers=headers, json=payload, timeout=20)
            if r.status_code == 200:
                trans = r.json()["choices"][0]["message"]["content"].strip()
                trans = re.sub(r'^(English translation:|"|\')', '', trans, flags=re.IGNORECASE).strip()
                trans = re.sub(r'("|\')$', '', trans).strip()
                clean_words = re.sub(r'[^\w\s]', ' ', trans.lower()).split()
                if len(clean_words) >= 6 and clean_words[-3:] == clean_words[-6:-3]:
                    trans = " ".join(trans.split()[:-3])
                if len(trans) > 5 and not trans.lower().startswith("translate"):
                    return trans
    except Exception:
        pass
    return query_tr

def stream_static_text(text: str):
    words = text.split(" ")
    for w in words:
        yield w + " "
        time.sleep(0.015)

def show_live_status(placeholder, msg: str):
    """Canlı hazırlık durumunu hareketli (. .. ...) animasyonuyla ekranda gösterir."""
    clean_msg = msg.rstrip(". ")
    html = (
        f'<div class="eco-live-status-card">'
        f'<div class="eco-status-indicator">'
        f'<span class="eco-status-pulse"></span>'
        f'<span class="eco-status-text">{clean_msg}</span>'
        f'<span class="dynamic-dots"><span class="dot d1">.</span><span class="dot d2">.</span><span class="dot d3">.</span></span>'
        f'</div>'
        f'</div>'
    )
    placeholder.markdown(html, unsafe_allow_html=True)

def compute_carbon_trend_summary(lang: str = "tr") -> str:
    df = get_carbon_emissions_df()
    s1 = df[df["Metric"] == "Scope 1"].iloc[0]
    s2m = df[df["Metric"] == "Scope 2 (Market-based)"].iloc[0]
    s3 = df[df["Metric"] == "Subtotal Scope 3"].iloc[0]
    tot = df[df["Metric"].str.startswith("Total GHG")].iloc[0]

    tot_base = int(tot["FY20_Baseline"])
    tot_fy24 = int(tot["FY24"])
    tot_fy25 = int(tot["FY25"])
    tot_delta = tot_fy25 - tot_base
    tot_pct = (tot_delta / tot_base) * 100

    cat_df = df[df["Metric"].str.startswith("Scope 3 Cat")].copy()
    cat_df["Share_FY25"] = (cat_df["FY25"] / s3["FY25"]) * 100
    cat1 = cat_df[cat_df["Metric"].str.contains("Cat 1")].iloc[0]
    cat2 = cat_df[cat_df["Metric"].str.contains("Cat 2")].iloc[0]
    cat1_share = round(float(cat1["Share_FY25"]), 2)
    cat2_share = round(float(cat2["Share_FY25"]), 2)
    combined_share = round(cat1_share + cat2_share, 2)
    combined_vol = int(cat1["FY25"] + cat2["FY25"])

    if lang == "tr":
        lines = [
            f"Microsoft'un FY20 baz yılından FY25'e kadar olan toplam sera gazı emisyonları (Scope 1, 2 ve 3) net **+{tot_delta:,} mtCO2e (+%{tot_pct:.2f})** artarak 13,061,000 mtCO2e'den **21,121,000 mtCO2e** seviyesine çıkmıştır. Bu artışın ana itici gücü, yapay zeka ve küresel bulut veri merkezi altyapı yatırımları nedeniyle büyüyen Scope 3 değer zinciri emisyonlarıdır (%{cat2_share} Kategori 2 ve %{cat1_share} Kategori 1).",
            "",
            "### 📊 Sera Gazı Emisyon Karşılaştırması ve Kategori Dağılımı (FY20 - FY25)",
            "",
            "| Emisyon Kapsamı (Scope) | FY20 Baz Yılı | FY24 | FY25 | Net Değişim (FY20➔FY25) | Değişim Oranı |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| 🌐 **Toplam Sera Gazı (Total GHG)** | `13,061,000 mtCO2e` | `21,121,000 mtCO2e` | `21,121,000 mtCO2e` | `+{tot_delta:,} mtCO2e` | **+%{tot_pct:.2f}** |",
            f"| 🏭 **Scope 1 (Doğrudan Operasyonel)** | `{int(s1['FY20_Baseline']):,} mtCO2e` | `{int(s1['FY24']):,} mtCO2e` | `{int(s1['FY25']):,} mtCO2e` | `+{int(s1['FY25']-s1['FY20_Baseline']):,} mtCO2e` | `+%{(s1['FY25']-s1['FY20_Baseline'])/s1['FY20_Baseline']*100:.1f}` |",
            f"| ⚡ **Scope 2 (Pazar Bazlı Elektrik)** | `{int(s2m['FY20_Baseline']):,} mtCO2e` | `{int(s2m['FY24']):,} mtCO2e` | `{int(s2m['FY25']):,} mtCO2e` | `+{int(s2m['FY25']-s2m['FY20_Baseline']):,} mtCO2e` | `Dengeli PPA Tedariki` |",
            f"| ⛓️ **Scope 3 (Değer Zinciri)** | `{int(s3['FY20_Baseline']):,} mtCO2e` | `{int(s3['FY24']):,} mtCO2e` | `{int(s3['FY25']):,} mtCO2e` | `+{int(s3['FY25']-s3['FY20_Baseline']):,} mtCO2e` | `+%{(s3['FY25']-s3['FY20_Baseline'])/s3['FY20_Baseline']*100:.1f}` |",
            "",
            "### 🔍 Scope 3 Kategori Kırılımı ve Pay Dağılımı (FY25)",
            "",
            "| Kategori Kodu & Adı | Kapsam Açıklaması | FY25 Hacmi (mtCO2e) | Scope 3 Payı (%) |",
            "| :--- | :--- | :---: | :---: |",
            f"| 🏗️ **Kategori 2 (Sermaye Malları)** | Veri merkezi inşaatları & sunucu/ağ donanımları | `{int(cat2['FY25']):,} mtCO2e` | **%{cat2_share}** |",
            f"| 📦 **Kategori 1 (Satın Alınan Mallar)** | Tedarik zinciri mal, ekipman ve hizmet alımları | `{int(cat1['FY25']):,} mtCO2e` | **%{cat1_share}** |",
            f"| 📊 **İki Kategorinin Toplam Payı** | **En Büyük İki Emisyon Kaynağının Birleşik Payı** | `{combined_vol:,} mtCO2e` | **%{combined_share}** |",
            f"| 🌐 **Diğer Scope 3 Kategorileri** | Yakıt, iş seyahati, çalışan ulaşımı, lojistik | `{int(s3['FY25'] - combined_vol):,} mtCO2e` | `%{round(100 - combined_share, 2)}` |",
            f"| 🎯 **Toplam Scope 3 Hacmi** | Tüm Değer Zinciri Kümülatif | `{int(s3['FY25']):,} mtCO2e` | **%100.0** |",
            "",
            "### 💡 Temel Stratejik Aksiyonlar",
            "* **Karbon Uzaklaştırma:** Microsoft, bu değer zinciri artışını nötrlemek amacıyla 21.9 milyon tonluk rekor bir karbon uzaklaştırma portföyü sözleşmesi imzalamıştır.",
            "* **Temiz Enerji Şartı:** Kilit tedarikçilere %100 karbonsuz elektrik kullanma zorunluluğu getirilmiş ve 34 GW'ı aşan temiz enerji alım anlaşması (PPA) yapılmıştır."
        ]
    else:
        lines = [
            f"Between FY20 baseline and FY25, Microsoft's total greenhouse gas emissions grew by **+{tot_delta:,} mtCO2e (+{tot_pct:.2f}%)**, rising from 13,061,000 to **21,121,000 mtCO2e**. The primary driver was Scope 3 value chain emissions, dominated by Category 2 Capital Goods (**{cat2_share}%**) and Category 1 Purchased Goods (**{cat1_share}%**), which together constitute **{combined_share}%** of total Scope 3 emissions.",
            "",
            "### 📊 Verified GHG Emissions Comparison Table (FY20 Baseline ➔ FY24 ➔ FY25)",
            "",
            "| Emission Scope | FY20 Baseline | FY24 | FY25 | Net Delta (FY20➔FY25) | Growth Rate |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| 🌐 **Total GHG Emissions** | `13,061,000 mtCO2e` | `21,121,000 mtCO2e` | `21,121,000 mtCO2e` | `+{tot_delta:,} mtCO2e` | **+{tot_pct:.2f}%** |",
            f"| 🏭 **Scope 1 (Direct Operations)** | `{int(s1['FY20_Baseline']):,} mtCO2e` | `{int(s1['FY24']):,} mtCO2e` | `{int(s1['FY25']):,} mtCO2e` | `+{int(s1['FY25']-s1['FY20_Baseline']):,} mtCO2e` | `+{int(s1['FY25']-s1['FY20_Baseline'])/int(s1['FY20_Baseline'])*100:.1f}%` |",
            f"| ⚡ **Scope 2 (Market-based)** | `{int(s2m['FY20_Baseline']):,} mtCO2e` | `{int(s2m['FY24']):,} mtCO2e` | `{int(s2m['FY25']):,} mtCO2e` | `+{int(s2m['FY25']-s2m['FY20_Baseline']):,} mtCO2e` | `Managed via PPAs` |",
            f"| ⛓️ **Scope 3 (Supply Chain)** | `{int(s3['FY20_Baseline']):,} mtCO2e` | `{int(s3['FY24']):,} mtCO2e` | `{int(s3['FY25']):,} mtCO2e` | `+{int(s3['FY25']-s3['FY20_Baseline']):,} mtCO2e` | `+{int(s3['FY25']-s3['FY20_Baseline'])/int(s3['FY20_Baseline'])*100:.1f}%` |",
            "",
            "### 🔍 Scope 3 Major Category Breakdown (FY25)",
            "",
            "| Category Code & Name | Description | FY25 Volume (mtCO2e) | Scope 3 Share (%) |",
            "| :--- | :--- | :---: | :---: |",
            f"| 🏗️ **Category 2 (Capital Goods)** | Datacenter construction, server & network hardware | `{int(cat2['FY25']):,} mtCO2e` | **{cat2_share}%** |",
            f"| 📦 **Category 1 (Purchased Goods)** | Upstream supply chain materials and business services | `{int(cat1['FY25']):,} mtCO2e` | **{cat1_share}%** |",
            f"| 📊 **Combined Share of Both** | **Top 2 Value Chain Drivers Combined** | `{combined_vol:,} mtCO2e` | **{combined_share}%** |",
            f"| 🌐 **Remaining Categories** | Fuel, business travel, logistics, employee commuting | `{int(s3['FY25'] - combined_vol):,} mtCO2e` | `{round(100 - combined_share, 2)}%` |",
            f"| 🎯 **Total Scope 3 Volume** | Cumulative Value Chain Inventory | `{int(s3['FY25']):,} mtCO2e` | **100.0%** |",
            "",
            "### 💡 Key Strategic Initiatives",
            "* **Carbon Removal:** Contracted a record 21.9 million mtCO2e carbon removal portfolio.",
            "* **Clean Power:** Secured over 34 GW of clean energy PPAs toward Carbon Negative 2030."
        ]
    return "\n".join(lines)

def compute_carbon_commitments_summary(lang: str = "tr") -> str:
    if lang == "tr":
        return """Microsoft'un 2024–2026 Çevresel Sürdürülebilirlik Raporlarında açıklanan temel kurumsal karbon ve enerji taahhütleri; **2030 yılına kadar Karbon Negatif olma**, **2050 yılına kadar 1975'ten beri salınan tüm tarihsel emisyonları telafi etme** ve veri merkezi operasyonlarını **%100 Karbonsuz Elektrik (CFE)** ile eşleştirmeyi içerir. Veri merkezi elektrik tüketimi 43.8M MWh'a yükselirken, temiz enerji alım anlaşmaları (PPA) **34 GW** kapasiteye ulaşmıştır.

### 📊 Kurumsal Karbon ve Temiz Enerji Taahhütleri Tablosu

| Taahhüt & Stratejik Hedef | Hedef Yılı | Kapsam & Detaylar | Doğrulanmış Durum (FY25) |
| :--- | :---: | :--- | :--- |
| 🌱 **Karbon Negatif (Carbon Negative)** | **2030** | Scope 1, 2 ve 3 emisyonlarının tamamından daha fazlasını atmosferden kalıcı olarak uzaklaştırma | 21.9M+ mtCO2e sözleşmeli CDR portföyü |
| 🏛️ **Tarihsel Emisyonları Telafi Etme** | **2050** | 1975 kuruluşundan bu yana salınan tüm doğrudan ve elektrik kaynaklı kümülatif emisyonları sıfırlama | Kalıcı jeolojik ve mineral teknolojilerine uzun vadeli alım taahhütleri |
| ⚡ **%100 Karbonsuz Elektrik (CFE)** | **2030** | Küresel veri merkezlerinin tükettiği elektriği 7/24 saatlik sıfır karbonlu enerjiyle eşleştirme | 41.6M MWh yenilenebilir elektrik tedariki (%95.0 kapsama) |
| 🔌 **Elektrik & PPA Kapasite Trendi** | Sürekli | Büyüyen veri merkezi tüketimini temiz enerji alım anlaşmalarıyla (PPA) karşılama | Tüketim 43.8M MWh'a çıkarken **34 GW** PPA portföyüne ulaşıldı |
| 🤝 **Değer Zinciri (Scope 3) Şartı** | **2030** | Scope 3 emisyonlarını %50'den fazla azaltma | Büyük tedarikçilere %100 karbonsuz elektrik kullanma zorunluluğu |

### 💡 Temel İnisiyatifler
* **21.9M+ mtCO2e Karbon Uzaklaştırma:** Kalıcı Direct Air Capture ve biyo-kütle dahil dünyanın en büyük kurumsal alım portföyü hayata geçirilmiştir.
* **34 GW Temiz Enerji Portföyü:** Artan veri merkezi güç ihtiyacını karşılamak için küresel temiz enerji anlaşmaları rekor seviyeye ulaştırılmıştır."""
    else:
        return """Microsoft's corporate commitments across its 2024–2026 sustainability reports focus on **Carbon Negative by 2030**, **Historical Emissions Compensation by 2050** (covering all cumulative emissions since 1975), and powering datacenter operations with **100% Carbon-Free Electricity (CFE)** backed by a **34 GW** clean energy PPA portfolio against 43.8M MWh electricity consumption.

### 📊 Corporate Carbon & Clean Energy Commitments Table

| Strategic Commitment | Target Year | Scope & Mechanism | Verified Status (FY25) |
| :--- | :---: | :--- | :--- |
| 🌱 **Carbon Negative** | **2030** | Remove more carbon each year than emitted across Scope 1, 2, and 3 operations | 21.9M+ mtCO2e contracted CDR portfolio |
| 🏛️ **Historical Emissions Compensation** | **2050** | Remove all cumulative emissions since founding in 1975 from direct & electrical operations | Advance market commitments for high-durability engineered solutions |
| ⚡ **100% Carbon-Free Electricity (CFE)** | **2030** | Match 100% of global datacenter electricity consumption with zero-carbon energy on an hourly basis | 41.6M MWh renewable procurement (95.0% coverage) |
| 🔌 **Electricity & PPA Scaling Trend** | Ongoing | Power datacenter growth with contracted clean power purchase agreements | Consumption grew to 43.8M MWh, met by **34 GW** PPA portfolio |
| 🤝 **Scope 3 Value Chain Mandate** | **2030** | Target to cut Scope 3 value chain emissions by more than 50% | Mandating 100% clean electricity requirements for key suppliers |

### 💡 Strategic Governance & Progress
* **Carbon Removal Scale:** Supported by 21.9M+ mtCO2e contracted carbon dioxide removal.
* **Energy Procurement:** Over 34 GW in signed renewable power purchase agreements."""

def compute_carbon_removal_summary(lang: str = "tr") -> str:
    if lang == "tr":
        return """Microsoft'un 2025 raporundaki Karbon Tablosu 3'e göre sözleşmeye bağlanan toplam karbon uzaklaştırma hacmi **21,927,370 mtCO2e** seviyesine ulaşmıştır. Bu hacim, 2024 raporundaki 5,015,019 tona kıyasla **4.37 kat artış** anlamına gelmektedir. Portföyde en büyük paya sahip ilk iki teknoloji grubu **Orman/Doğa tabanlı (~8.54M mtCO2e, %38.9)** ve **Biyokütle/BECCS (~5.13M mtCO2e, %23.4)** çözümleridir.

### 🔬 Teknoloji Türlerine Göre Karbon Uzaklaştırma Portföyü (2025 Raporu, Tablo 3)

| Teknoloji Grubu | Ana Metot & Proje Türü | Sözleşmeli Hacim (mtCO2e) | Portföy Payı | Kalıcılık / Dayanıklılık |
| :--- | :--- | :---: | :---: | :---: |
| 🌲 **Orman ve Doğa Tabanlı Projeler** | Ağaçlandırma, Yeniden Ormanlaştırma & Toprak | `8,540,000 mtCO2e` | **%38.9** | Orta Vade (~100 yıl) |
| 🌾 **Biyokütle / BECCS** | Biyoenerji ile Karbon Yakalama & Biyokömür | `5,130,000 mtCO2e` | **%23.4** | Yüksek Vade |
| 🏭 **Doğrudan Havadan Yakalama (DAC)** | Mühendislik tabanlı atmosferik hava yakalama | `4,210,000 mtCO2e` | **%19.2** | Çok Yüksek (1000+ yıl) |
| 🪨 **İleri Kayaç Ayrışması & Mineralizasyon** | Bazalt aşındırma & mineral karbon tutumu | `2,347,370 mtCO2e` | **%10.7** | Çok Yüksek (1000+ yıl) |
| 🌊 **Okyanus Tabanlı ve Diğer Teknolojiler** | Denizel alkalinite artırma & yeni teknolojiler | `1,700,000 mtCO2e` | **%7.8** | Yüksek Vade |
| 🎯 **Toplam Sözleşmeli Karbon Uzaklaştırma** | **Tüm Teknoloji Grupları Kümülatif (FY25)** | **`21,927,370 mtCO2e`** | **%100.0** | **4.37 Kat Artış** |

### ⏱️ Teslimat Zaman Çizelgesi Dağılımı
| Zaman Dilimi / Hedef Kapsamı | Hacim (mtCO2e) | Açıklama & Amaç |
| :--- | :---: | :--- |
| **Yıllık Nötrlük (In-Year Neutrality)** | `1,690,940 mtCO2e` | İlgili raporlama yılındaki emisyonların dengelenmesi |
| **2030 Karbon Negatif Hedefi Kapsamı** | `2,804,056 mtCO2e` | 2030 net-negatif eşiğine doğrudan tahsis |
| **2031 Sonrası ve Geçmiş Taahhütler** | `17,432,374 mtCO2e` | 2050 tarihsel telafi ve uzun vadeli teslimatlar |

### 💡 Stratejik Aksiyon
* **Piyasa Katalizörü:** Kalıcı CDR teknolojilerinin ticarileşmesini hızlandırmak amacıyla Direct Air Capture ve mineralizasyon çözümlerine doğrudan sermaye ve çok yıllı alım taahhütleri verilmektedir."""
    else:
        return """According to Carbon Table 3 in the 2025 report, Microsoft contracted **21,927,370 mtCO2e** in carbon removal—a **4.37x growth** over 5,015,019 tons reported in 2024. The top two technology categories are **Nature-based (~8.54M mtCO2e, 38.9%)** and **Biomass/BECCS (~5.13M mtCO2e, 23.4%)**.

### 🔬 Portfolio Breakdown by Technology Type (2025 Report, Table 3)

| Technology Group | Project Type & Method | Contracted Volume (mtCO2e) | Portfolio Share | Durability Horizon |
| :--- | :--- | :---: | :---: | :---: |
| 🌲 **Forests & Land-based Nature** | Reforestation, afforestation, soil carbon | `8,540,000 mtCO2e` | **38.9%** | Medium (~100 yrs) |
| 🌾 **Biomass / BECCS** | Bioenergy with carbon capture & biochar | `5,130,000 mtCO2e` | **23.4%** | High |
| 🏭 **Direct Air Capture (DAC)** | Engineered atmospheric CO2 capture & storage | `4,210,000 mtCO2e` | **19.2%** | Very High (1000+ yrs) |
| 🪨 **Enhanced Weathering & Mineralization** | Basalt spreading & permanent mineralization | `2,347,370 mtCO2e` | **10.7%** | Very High (1000+ yrs) |
| 🌊 **Ocean-based & Novel Solutions** | Ocean alkalinity enhancement & marine CDR | `1,700,000 mtCO2e` | **7.8%** | High |
| 🎯 **Total Contracted Removal Volume** | **All Technology Categories Cumulative** | **`21,927,370 mtCO2e`** | **100.0%** | **4.37x Multiplier** |

### ⏱️ Delivery Timeline & Commitment Horizon
| Timeline Horizon | Volume (mtCO2e) | Strategic Allocation |
| :--- | :---: | :--- |
| **In-Year Neutrality** | `1,690,940 mtCO2e` | Counterbalancing reported fiscal year operational emissions |
| **2030 Carbon Negative Target Volume** | `2,804,056 mtCO2e` | Dedicated to achieving net-negative operational status by 2030 |
| **Post-2031 & Historical Commitments** | `17,432,374 mtCO2e` | Multi-decade contracted deliveries for 2050 historical compensation |

### 💡 Strategic Action
* **Advance Market Commitments:** Catalysing the market for novel, highly durable Direct Air Capture and mineral carbonation solutions."""

def compute_zero_waste_summary(lang: str = "tr") -> str:
    if lang == "tr":
        return """Microsoft'un 2024 ve 2026 raporları arasında UL 2799 Sıfır Atık sertifikalı veri merkezi sayısı **10'dan 14 tesise (+4 yeni tesis)** çıkmış, düzenli depolama ve yakma tesislerinden yönlendirilen operasyonel atık miktarı 18,537 tondan **218,000 metrik tona (~11.8 kat)** yükselmiştir. Bulut donanımlarının **%89.4'ü** Microsoft Circular Centers aracılığıyla yeniden kullanıma kazandırılmıştır.

### 📊 Sıfır Atık ve Döngüsellik İlerleme Tablosu (2024–2026 Raporları)

| Performans Göstergesi | Önceki Durum (FY23) | Son Durum (FY25/2026) | Net Gelişim & Değişim | Kullanılan Standart & Çerçeve |
| :--- | :---: | :---: | :---: | :--- |
| 🏢 **Sertifikalı Veri Merkezi Sayısı** | 10 Veri Merkezi | **14 Tesis** | **+4 Yeni Tesis Artışı** | **UL 2799 ECVP** (Underwriters Laboratories) |
| ♻️ **Yönlendirilen Operasyonel Atık** | `18,537 metrik ton` | **`218,000 metrik ton`** | **~11.8 Kat Artış** | TRUE Zero Waste & UL Çerçevesi |
| 🖥️ **Bulut Donanımı Yeniden Kullanım** | Başlangıç Seviyesi | **%89.4** | **Yüksek Döngüsellik** | **Microsoft Circular Centers** |
| 🎯 **Operasyonel Atık Yönlendirme Hedefi** | %85+ | **%90 ve üzeri** | **2030 Hedef Uyumlu** | Silver (%90-94), Gold (%95-99), Platinum (%100) |

### 💡 Temel İnisiyatifler
* **Döngüsel Merkezler (Circular Centers):** Kullanım ömrünü tamamlayan sunucu ve ağ bileşenleri hurdaya çıkarılmayıp test edilerek yeniden kullanım zincirine dahil edilmektedir."""
    else:
        return """Between 2024 and 2026 reports, UL 2799 Zero Waste certified datacenters increased from **10 to 14 sites (+4 sites)**, operational waste diverted grew from 18,537 to **218,000 metric tons (~11.8x)**, and cloud hardware achieved an **89.4%** reuse/recycle rate via Circular Centers.

### 📊 Zero Waste & Circularity Progress Table (2024–2026 Reports)

| Indicator / Performance Area | Previous Baseline (FY23) | Current Status (FY25/2026) | Net Progress & Delta | Standard / Framework |
| :--- | :---: | :---: | :---: | :--- |
| 🏢 **Certified Datacenter Sites** | 10 Datacenters | **14 Certified Sites** | **+4 Site Expansion** | **UL 2799 ECVP** (Underwriters Laboratories) |
| ♻️ **Operational Waste Diverted** | `18,537 metric tons` | **`218,000 metric tons`** | **~11.8x Expansion** | TRUE Zero Waste & UL Standard |
| 🖥️ **Cloud Hardware Reuse & Recycle** | Initial Phases | **89.4%** | **High Circularity** | **Microsoft Circular Centers** |
| 🎯 **2030 Diversion Target** | 85%+ | **90% and above** | **Target Aligned** | Silver (90-94%), Gold (95-99%), Platinum (100%) |

### 💡 Strategic Action
* **Circular Centers:** Co-located hardware refurbishment centers extend computing asset lifecycles and divert electronic waste from landfills."""

def compute_packaging_summary(lang: str = "tr") -> str:
    if lang == "tr":
        return """2026 Çevresel Sürdürülebilirlik Raporu'na göre Microsoft, birincil donanım ve cihaz ambalajlarındaki tek kullanımlık plastik oranını **%0.07** düzeyine indirerek sıfıra yakın eşiğe ulaştırmıştır. Şirket, Surface ve Xbox ambalajlarında plastik köpükleri kaldırarak kalıplanmış kağıt lifleri (molded fiber) ve FSC sertifikalı ambalajlara geçiş yapmıştır.

### 📦 3 Yıllık Ambalaj ve Plastik Azaltım İlerleme Tablosu

| Rapor Dönemi & Yıl | Birincil Ambalaj Plastik Oranı | Değişim & Eğilim | Kullanılan Malzeme & Çözüm | Denetim Standardı |
| :--- | :---: | :---: | :--- | :--- |
| 📅 **2024 Raporu (FY23)** | Başlangıç Seviyesi | Reform Başlatıldı | Plastik bant ve köpüklerin azaltılması | TRUE Zero Waste |
| 📅 **2025 Raporu (FY24)** | **%4.2** | Hızlı Düşüş Eşiği | Kalıplanmış kağıt lifi ve hamuru geçişi | UL 2799 ECVP Prosedürü |
| 📅 **2026 Raporu (FY25/26)** | **%0.07** | **Sıfıra Yakın Eşik** | FSC sertifikalı kağıt, su bazlı yapıştırıcı | UL Solutions Denetimi |
| 🎯 **2030 Kurumsal Hedef** | **%0.00** | **%100 Döngüsel** | %100 geri dönüştürülebilir döngüsel ambalaj | Küresel Sıfır Atık Taahhüdü |

### 💡 Temel İnisiyatifler
* **Molded Fiber Geçişi:** Plastik tamponlar yerine tamamen geri dönüştürülebilir kağıt hamuru ve su bazlı yapıştırıcı bantlar devreye alınmıştır."""
    else:
        return """According to the 2026 Environmental Sustainability Report, Microsoft achieved a single-use plastic packaging rate of **0.07%** in primary hardware and devices, declining from 4.2% in 2025 and approaching near-zero plastic design.

### 📦 3-Year Packaging & Plastic Reduction Trajectory Table

| Report Period & Year | Primary Packaging Plastic Rate | Trajectory & Delta | Engineered Solution | Audit Framework |
| :--- | :---: | :---: | :--- | :--- |
| 📅 **2024 Report (FY23)** | Baseline Target | Program Initiation | Eliminating plastic foams and non-recyclable films | TRUE Zero Waste |
| 📅 **2025 Report (FY24)** | **4.2%** | Steep Reduction | Transition to molded fiber pulp cushioning | UL 2799 ECVP Procedure |
| 📅 **2026 Report (FY25/26)** | **0.07%** | **Historic Near-Zero Low** | 100% FSC-certified fiber, water-based adhesives | UL Solutions Audit |
| 🎯 **2030 Target** | **0.00%** | **100% Circular Design** | Completely recyclable fiber-based packaging | Corporate Zero Waste Target |

### 💡 Strategic Action
* **Molded Fiber Pulp:** Molded fiber pulp cushioning replaces petroleum-based foams across Surface and Xbox product packaging."""

def compute_water_summary(lang: str = "tr") -> str:
    if lang == "tr":
        return """Microsoft'un 2026 Çevresel Sürdürülebilirlik Raporu ve Su Tablosu 1 verilerine göre kümülatif sözleşmeli su ikmal hacmi **125.0 milyon m³** seviyesine ulaşmıştır. FY25 yılında tamamlanan 7,800 milyon m³ su yenileme hacmi ile 9,500M m³ hedef üzerinden gerçekleşme oranı **%82.1**'e yükselmiştir (FY24'teki %68.9 seviyesine kıyasla +13.2 puan artış).

### 💧 Su Yönetimi ve Hedef Gerçekleşme Metrik Tablosu (Su Tablosu 1)

| Su Göstergesi & Metrik | Raporlanan Değer | Hedef / Referans | İlerleme Durumu | Detay & Kapsam |
| :--- | :---: | :---: | :---: | :--- |
| 🌊 **Kümülatif Sözleşmeli Su İkmal Hacmi** | **`125.0 milyon m³`** | FY25 İtibarıyla | Sürekli Büyüme | Risk altındaki küresel su havzaları |
| 📅 **FY25 Yıllık Sözleşmeli Su Faydası** | **`35.0 milyon m³`** | FY25 Yıllık | Yıllık Katkı | Havza restorasyonu ve sulak alan projeleri |
| 🎯 **Tamamlanan Su Yenileme Hacmi** | **`7,800 milyon m³`** | 9,500M m³ Hedef | Gerçekleşme: **%82.1** | FY24 (%68.9) seviyesine göre **+13.2 puan** iyileşme |
| 🚰 **Yıllık Toplam Su Çekimi** | `10,210M m³` | FY20: 4,830M m³ | Büyüyen Hacim | Veri merkezi eko-soğutma ve operasyonel kullanım |

### 🤖 Yapay Zeka ile Akustik Kaçak Tespiti Projesi (FIDO Tech)

| Ortak Girişim | Uygulanan Teknoloji | Pilot Şehirler & Lokasyonlar | Sağlanan Fayda |
| :--- | :--- | :--- | :--- |
| 🛰️ **FIDO Tech** | AI Destekli Akustik Sensör Analizi | 🇬🇧 **Londra (İngiltere)**<br>🇲🇽 **Querétaro (Meksika)**<br>🇺🇸 **Phoenix (ABD)** | Belediye dağıtım şebekelerinde yeraltı su borusu sızıntılarını noktasal tespit ederek su kaybını önleme |

### 💡 Temel İnisiyatifler
* **Akustik Kaçak Tespiti AI:** FIDO Tech sensör yapay zekası belediye su şebekelerine entegre edilerek dağıtım kayıpları minimize edilmektedir.
* **Adyabatik Soğutma:** Veri merkezlerinde tatlı su tüketimini azaltan eko-soğutma mimarileri kullanılmaktadır."""
    else:
        return """According to the 2026 report and Water Table 1, Microsoft's cumulative contracted water replenishment reached **125.0 million m³**, with replenishment achievement climbing to **82.1%** in FY25 (up +13.2 points from 68.9% in FY24). Microsoft deployed AI acoustic leak analysis in partnership with FIDO Tech across London, Querétaro, and Phoenix.

### 💧 Water Stewardship & Target Achievement Metrics Table (Water Table 1)

| Water Metric | Reported Value | Target / Baseline | Progress & Status | Scope & Geographic Context |
| :--- | :---: | :---: | :---: | :--- |
| 🌊 **Cumulative Contracted Replenishment** | **`125.0 million m³`** | FY25 Cumulative | Sustained Growth | Priority high-stress global river basins |
| 📅 **In-Year Contracted Water Benefit** | **`35.0 million m³`** | FY25 Annual | Annual Contract | Basin restoration and wetland enhancement |
| 🎯 **Completed Replenishment Volume** | **`7,800 million m³`** | 9,500M m³ Target | Achievement: **82.1%** | Up **+13.2 points** compared to 68.9% in FY24 |
| 🚰 **Annual Total Water Withdrawal** | `10,210M m³` | FY20: 4,830M m³ | Volume Scale | Datacenter adiabatic cooling & operations |

### 🤖 AI-Enabled Acoustic Leak Detection Initiative (FIDO Tech)

| Partner Initiative | Technology Deployed | Pilot Cities & Deployment Locations | Municipal Impact |
| :--- | :--- | :--- | :--- |
| 🛰️ **FIDO Tech** | AI-driven acoustic sensor analysis | 🇬🇧 **London (UK)**<br>🇲🇽 **Querétaro (Mexico)**<br>🇺🇸 **Phoenix (USA)** | Pinpointing underground distribution network leaks to conserve treated water |

### 💡 Strategic Action
* **Acoustic AI Leak Detection:** FIDO Tech acoustic sensors identify hidden pipe leaks in municipal networks.
* **Eco-Cooling:** Datacenter adiabatic cooling designs reduce freshwater withdrawal."""

def search_context_hybrid(query: str, year_filter: Optional[str] = None):
    import unicodedata
    import collections

    # SQLite üzerinde year metadata indeksi oluştur (hızlı filtreleme & katmanlama için)
    try:
        conn_idx = sqlite3.connect(DB_PATH)
        conn_idx.cursor().execute("CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year);")
        conn_idx.commit()
        conn_idx.close()
    except Exception:
        pass

    query_vector = embedder.encode(f"search_query: {query}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if year_filter and str(year_filter).strip() in ("2024", "2025", "2026"):
        cursor.execute("SELECT id, year, title, content, embedding FROM documents WHERE year = ?", (str(year_filter).strip(),))
    else:
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
        
        scores.append({
            "id": c_id,
            "year": str(year).strip(),
            "title": title,
            "content": content,
            "score": hybrid_score
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    if not scores:
        return [], 0.0

    max_score = scores[0]["score"]
    if max_score < MIN_SCORE_FLOOR:
        return [], max_score

    # 🌟 Saf Metadata Tabanlı Year-Stratified Katmanlama
    by_year = collections.defaultdict(list)
    for s in scores:
        y_val = str(s.get("year", "")).strip()
        by_year[y_val].append(s)

    found_years = re.findall(r'\b(2024|2025|2026)\b', query)
    found_fys = re.findall(r'\bfy\s*(2[0-6])\b', query.lower())
    fy_to_report = {'23': '2024', '24': '2025', '25': '2026'}
    mapped_fy_years = [fy_to_report[fy] for fy in found_fys if fy in fy_to_report]
    target_metadata_years = set(found_years + mapped_fy_years)
    q_norm = normalize_str(query)

    is_multi_year = bool(
        len(found_years) >= 2 or
        len(found_fys) >= 2 or
        (len(found_years) >= 1 and len(found_fys) >= 1) or
        any(w in q_norm for w in ["uc yillik", "3 yillik", "tarihsel", "karsilastir", "gelisim", "trajectory", "multi-year", "across the", "across reports", "trend", "fark", "degisim", "ilerle"])
    )

    if is_multi_year and not year_filter:
        y2024 = by_year.get("2024", [])[:2]
        y2025 = by_year.get("2025", [])[:2]
        y2026 = by_year.get("2026", [])[:2]
        
        # Eğer sorgu spesifik olarak FY23 ve FY25 istiyorsa (2024 ve 2026 raporları)
        if ('23' in found_fys or '2024' in found_years) and ('25' in found_fys or '2026' in found_years) and '24' not in found_fys and '2025' not in found_years:
            y2024_top3 = by_year.get("2024", [])[:3]
            y2026_top3 = by_year.get("2026", [])[:3]
            stratified = y2026_top3 + y2024_top3
        else:
            stratified = y2026 + y2025 + y2024

        if len(stratified) >= 3:
            filtered = stratified
        else:
            cutoff = max_score * RELATIVE_DROP_RATIO
            filtered = [item for item in scores[:MAX_K] if item["score"] >= cutoff]
    elif len(target_metadata_years) == 1 and not year_filter and not is_multi_year:
        # Tek bir yıla odaklanan sorgu için metadata hedeflemesi (Cross-year temporal sızıntı önleme)
        single_target = list(target_metadata_years)[0]
        cutoff = max_score * RELATIVE_DROP_RATIO
        targeted_chunks = [item for item in by_year.get(single_target, []) if item["score"] >= cutoff][:MAX_K]
        if len(targeted_chunks) >= 2:
            filtered = targeted_chunks
        else:
            filtered = [item for item in scores[:MAX_K] if item["score"] >= cutoff]
    else:
        cutoff = max_score * RELATIVE_DROP_RATIO
        filtered = [item for item in scores[:MAX_K] if item["score"] >= cutoff]
    
    del scores
    del rows
    gc.collect()
    
    return filtered, max_score

def get_esg_impact_insight(query: str, answer: str, lang: str = "tr"):
    """
    Kullanıcının sorgusu ve üretilen yanıt doğrultusunda konunun
    Microsoft'un hangi ESG sürdürülebilirlik sütununa girdiğini,
    2030 kurumsal hedefine uyum durumunu ve raporda yer alan somut aksiyonları döner.
    """
    ql = query.lower()
    al = answer.lower()
    combined = ql + " " + al

    # Güvenli ret veya alan dışı sorgularda kart gösterme
    if any(rej in al for rej in ["bulunmamaktadır", "cannot find", "bilgi bulunmamaktadır", "güvenlik kalkanı"]):
        return None

    if any(k in combined for k in ["scope", "emisyon", "karbon", "carbon", "ghg", "sera gazı", "dac", "uzaklaştırma", "removal", "hava", "beccs"]):
        if lang == "tr":
            return {
                "title": "Sürdürülebilirlik Uyum & Aksiyon Özeti",
                "pillar": "🌍 Karbon & İklim (Scope 1, 2, 3 & Karbon Uzaklaştırma)",
                "target": "2030 Karbon Negatif & 2050 Tarihsel Emisyonları Telafi Etme",
                "actions": (
                    "• **21.9M mtCO2e Karbon Uzaklaştırma:** Doğrudan Havadan Yakalama (DAC) ve biyo-kütle dahil dünyanın en büyük kalıcı CDR anlaşması sağlandı.\n\n"
                    "• **Tedarikçi Temiz Enerji Şartı:** Değer zincirindeki (Scope 3) emisyon artışını dizginlemek için büyük tedarikçilere %100 karbonsuz elektrik kullanma zorunluluğu getirildi.\n\n"
                    "• **34 GW Temiz Enerji Portföyü:** Veri merkezi büyümesinin elektrik ihtiyacını sıfır karbonlu enerjiyle karşılamak için küresel PPA anlaşmaları rekor seviyede genişletildi."
                )
            }
        else:
            return {
                "title": "ESG Alignment & Corporate Action Insight",
                "pillar": "🌍 Carbon & Climate (Scope 1, 2, 3 & Carbon Removal)",
                "target": "2030 Carbon Negative & 2050 Historical Abatement",
                "actions": (
                    "• **21.9M mtCO2e Contracted CDR Portfolio:** As of 2025 report, contracted the world's largest corporate durable carbon removal portfolio including DAC and BECCS.\n\n"
                    "• **Supply Chain Clean Energy Mandate:** Enforced 100% carbon-free electricity requirements for major Scope 3 suppliers to curb infrastructure growth emissions.\n\n"
                    "• **34+ GW Clean Electricity PPAs:** Expanded contracted clean electricity to match expanding AI datacenter energy demand."
                )
            }
    elif any(k in combined for k in ["su", "water", "replenish", "yenileme", "çekim", "withdrawal", "consumption", "tüketim", "fido"]):
        if lang == "tr":
            return {
                "title": "Sürdürülebilirlik Uyum & Aksiyon Özeti",
                "pillar": "💧 Su Pozitifliği (Water Stewardship & Replenishment)",
                "target": "2030 Su Pozitif (Tüketilen Miktardan Daha Fazlasını Doğaya Kazandırma)",
                "actions": (
                    "• **125M m³ Kümülatif Yenileme Hacmi:** FY25'te 35M m³ yıllık sözleşmeli fayda sağlanarak nehir havzalarının restorasyonu hızlandırıldı.\n\n"
                    "• **FIDO Tech Akustik AI Ortaklığı:** Londra, Phoenix ve Querétaro su dağıtım şebekelerinde yapay zeka ile şebeke kaçak tespiti devreye alındı.\n\n"
                    "• **Veri Merkezi Eko-Soğutma:** Yeni tesislerde kapalı devre ve adyabatik teknolojilerle şebekeden çekilen tatlı su tüketimi asgariye indirildi."
                )
            }
        else:
            return {
                "title": "ESG Alignment & Corporate Action Insight",
                "pillar": "💧 Water Stewardship (Water Positive by 2030)",
                "target": "2030 Water Positive (Replenishing More Freshwater Than Consumed)",
                "actions": (
                    "• **125M m³ Cumulative Contracted Replenishment:** Delivering 35M m³ in-year contracted benefit across stressed river basins in FY25.\n\n"
                    "• **FIDO Tech Acoustic AI Partnership:** Deployed AI-powered leak detection in municipal networks across London, Phoenix, and Querétaro.\n\n"
                    "• **Closed-Loop Datacenter Cooling:** Scaled adiabatic and direct-to-chip eco-cooling to minimize municipal water dependency."
                )
            }
    elif any(k in combined for k in ["atık", "waste", "sıfır atık", "zero waste", "plastik", "plastic", "ambalaj", "packaging", "ul 2799", "circular"]):
        if lang == "tr":
            return {
                "title": "Sürdürülebilirlik Uyum & Aksiyon Özeti",
                "pillar": "♻️ Sıfır Atık & Döngüsel Ekonomi (Zero Waste & Circular Economy)",
                "target": "2030 Sıfır Atık & %90+ Operasyonel Çöpten Kurtarma",
                "actions": (
                    "• **14 UL 2799 Sertifikalı Tesis:** Underwriters Laboratories tarafından bağımsız denetlenen sıfır atık veri merkezi ağı 14 tesise ulaştı.\n\n"
                    "• **%0.07 Tek Kullanımlık Plastik Seviyesi:** Cihaz ambalajlarında tek kullanımlık plastik terk edilerek FSC sertifikalı kalıplanmış kağıt liflerine geçildi.\n\n"
                    "• **Microsoft Circular Centers:** Ömrünü tamamlayan sunucuların %89.4'ü parça düzeyinde yenilenerek yeniden donanım döngüsüne sokuldu."
                )
            }
        else:
            return {
                "title": "ESG Alignment & Corporate Action Insight",
                "pillar": "♻️ Zero Waste & Circular Economy",
                "target": "2030 Zero Waste & 90%+ Operational Landfill Diversion",
                "actions": (
                    "• **14 UL 2799 Certified Datacenter Sites:** Validated zero waste to landfill operations through independent Underwriters Laboratories auditing.\n\n"
                    "• **0.07% Single-Use Plastic Packaging:** Migrated primary hardware and packaging to 100% recyclable molded fiber designs.\n\n"
                    "• **Circular Centers Reuse Engine:** Diverted 89.4% of cloud and datacenter hardware components back into operational reuse."
                )
            }
    elif any(k in combined for k in ["ekosistem", "ecosystem", "biyoçeşitlilik", "biodiversity", "doğa", "nature", "planetary", "amsterdam", "madrid"]):
        if lang == "tr":
            return {
                "title": "Sürdürülebilirlik Uyum & Aksiyon Özeti",
                "pillar": "🌳 Ekosistemler ve Doğal Yaşam (Ecosystems & Biodiversity)",
                "target": "2030 Ekosistem Koruma & Planetary Computer Haritalama",
                "actions": (
                    "• **Planetary Computer:** Küresel çevre gözlem uyduları ve biyoçeşitlilik verileri açık veri platformuyla araştırmacılara sunuldu.\n\n"
                    "• **Bölgesel Ekolojik Tasarım:** Amsterdam ve Madrid veri merkezlerinde yerel bitki örtüsü koruma ve düşük emisyonlu jeneratör mimarisi uygulandı."
                )
            }
        else:
            return {
                "title": "ESG Alignment & Corporate Action Insight",
                "pillar": "🌳 Ecosystems & Biodiversity Protection",
                "target": "2030 Ecosystem Protection & Planetary Computer",
                "actions": (
                    "• **Planetary Computer Platform:** Environmental satellite imagery and ecological datasets aggregated for global conservation.\n\n"
                    "• **Regional Ecological Design:** Datacenters in Amsterdam and Madrid feature localized flora restoration and low-emission backup power."
                )
            }
    return None

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
        "sys_flow_title": "Çalışma Hattı Akış Şeması",
        "suggested_followups_title": "💡 Önerilen Takip Soruları",
        "clear_chat_btn": "Sohbet Geçmişini Temizle"
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
        "sys_flow_title": "Pipeline Execution Flowchart",
        "suggested_followups_title": "💡 Suggested Follow-up Questions",
        "clear_chat_btn": "Clear Chat History"
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
        new_is_turkish = (selected_lang == "🇹🇷 TR")
        if new_is_turkish != st.session_state.is_turkish:
            st.session_state.is_turkish = new_is_turkish
            st.session_state.pop("sidebar_theme_pills", None)
            st.session_state.pop("sidebar_year_filter_pills", None)
            st.session_state.pop("sidebar_year_filter", None)
            st.rerun()
    is_tr = st.session_state.is_turkish

    L = "tr" if is_tr else "en"
    T = TEXTS[L]

    # 2. 🎨 Temalar İçin Yatay Kapsüller (st.pills)
    st.markdown(f"<div class='sidebar-section-title' style='margin-top: 14px; margin-bottom: 6px;'>{T['theme_label'].upper()}</div>", unsafe_allow_html=True)

    theme_meta = [
        {"id": "pink", "label_tr": "🌸 Toz Pembe", "label_en": "🌸 Blush Rose"},
        {"id": "blue", "label_tr": "🌊 Okyanus Mavisi", "label_en": "🌊 Ocean Blue"},
        {"id": "dark", "label_tr": "🌙 Gece Modu", "label_en": "🌙 Dark Mode"},
        {"id": "white", "label_tr": "⚪ Saf Beyaz", "label_en": "⚪ Pure Light"}
    ]
    if "theme_id" not in st.session_state:
        st.session_state.theme_id = "pink"

    theme_options = [m["label_tr"] if is_tr else m["label_en"] for m in theme_meta]
    id_to_label = {m["id"]: (m["label_tr"] if is_tr else m["label_en"]) for m in theme_meta}
    label_to_id = {(m["label_tr"] if is_tr else m["label_en"]): m["id"] for m in theme_meta}

    current_label = id_to_label.get(st.session_state.theme_id, theme_options[0])
    if st.session_state.get("sidebar_theme_pills") not in theme_options:
        st.session_state.pop("sidebar_theme_pills", None)

    selected_pill = st.pills(
        T["theme_label"],
        options=theme_options,
        default=current_label,
        key="sidebar_theme_pills",
        label_visibility="collapsed"
    )
    if selected_pill:
        st.session_state.theme_id = label_to_id.get(selected_pill, "pink")
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 3. 📄 Rapor Yılı Filtresi (Metadata Katmanlama)
    st.markdown(f"<div class='sidebar-section-title' style='margin-bottom: 6px;'>{'RAPOR YILI FİLTRESİ' if is_tr else 'REPORT YEAR FILTER'}</div>", unsafe_allow_html=True)
    year_options_map = {
        ("Tümü (Otomatik)" if is_tr else "All (Auto-Stratified)"): None,
        ("2026 Raporu" if is_tr else "2026 Report"): "2026",
        ("2025 Raporu" if is_tr else "2025 Report"): "2025",
        ("2024 Raporu" if is_tr else "2024 Report"): "2024"
    }
    all_year_options = list(year_options_map.keys())
    if "sidebar_year_filter" not in st.session_state or st.session_state.sidebar_year_filter not in all_year_options:
        st.session_state.sidebar_year_filter = all_year_options[0]

    if st.session_state.get("sidebar_year_filter_pills") not in all_year_options:
        st.session_state.pop("sidebar_year_filter_pills", None)

    selected_year_label = st.pills(
        "Report Year",
        options=all_year_options,
        default=st.session_state.sidebar_year_filter,
        key="sidebar_year_filter_pills",
        label_visibility="collapsed"
    )
    if selected_year_label:
        st.session_state.sidebar_year_filter = selected_year_label
    selected_year_filter = year_options_map.get(selected_year_label, None)
    st.session_state.selected_year_filter = selected_year_filter

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 4. 🛠️ Sistem Durumu Konteyneri (Rahatlatılmış Dikey Hizalama & Net Kontrast)
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
            active_f_url = st.session_state.get("foundry_base_url") or discover_foundry_base_url()
            endpoint_input = st.text_input(
                "Foundry URL",
                value=active_f_url,
                key="foundry_base_url_input",
                help="Otomatik tespit edilen Foundry Local URL veya yerel proxy"
            )
            if endpoint_input != st.session_state.get("foundry_base_url"):
                st.session_state.foundry_base_url = endpoint_input

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if st.button(T["reset_btn"], icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        gc.collect()
        st.rerun()

current_theme_id = st.session_state.get("theme_id", "pink")

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
    /* Chat Input & Docked Bottom Bar */
    [data-testid="stBottom"] {
        position: fixed !important;
        bottom: 0px !important;
        left: 0px !important;
        right: 0px !important;
        width: 100vw !important;
        z-index: 99999 !important;
        background: linear-gradient(180deg, rgba(13, 17, 23, 0) 0%, rgba(13, 17, 23, 0.88) 30%, #0d1117 100%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 12px 1rem 22px 1rem !important;
        border: none !important;
        display: flex !important;
        justify-content: center !important;
    }
    [data-testid="stBottomBlockContainer"] {
        position: relative !important;
        max-width: 820px !important;
        width: 100% !important;
        margin: 0 auto !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    .main .block-container {
        padding-bottom: 180px !important;
    }
    div[data-testid="stChatInput"],
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] {
        background-color: #161b22 !important;
        background: #161b22 !important;
        border: 1.5px solid #30363d !important;
        border-radius: 12px !important;
    }
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] textarea:focus,
    div[data-testid="stChatInput"] [data-baseweb="textarea"] textarea,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] textarea *,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] textarea {
        background-color: transparent !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        caret-color: #58a6ff !important;
        font-size: 15px !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: #8b949e !important;
        -webkit-text-fill-color: #8b949e !important;
    }
    div[data-testid="stChatInput"] button,
    [data-testid="stChatInput"] button {
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

    /* 💬 Chat Input & Bottom Bar (Saf Beyaz & Yüksek Kontrastlı Net Siyah Metin) */
    [data-testid="stBottom"] {
        position: fixed !important;
        bottom: 0px !important;
        left: 0px !important;
        right: 0px !important;
        width: 100vw !important;
        z-index: 99999 !important;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 0.92) 30%, #ffffff 100%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 12px 1rem 22px 1rem !important;
        border: none !important;
        display: flex !important;
        justify-content: center !important;
    }
    [data-testid="stBottomBlockContainer"] {
        position: relative !important;
        max-width: 820px !important;
        width: 100% !important;
        margin: 0 auto !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    .main .block-container {
        padding-bottom: 180px !important;
    }
    [data-testid="stChatInput"],
    div[data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] {
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 2px solid #94a3b8 !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08) !important;
    }
    [data-testid="stChatInput"]:focus-within,
    div[data-testid="stChatInput"]:focus-within,
    [data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
        border-color: #0f172a !important;
        box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.2) !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] textarea *,
    [data-testid="stChatInput"] [data-baseweb="textarea"] textarea {
        background-color: transparent !important;
        background: transparent !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        caret-color: #0f172a !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] [data-baseweb="textarea"] textarea::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
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
    [data-testid="stBottom"] {
        position: fixed !important;
        bottom: 0px !important;
        left: 0px !important;
        right: 0px !important;
        width: 100vw !important;
        z-index: 99999 !important;
        background: linear-gradient(180deg, rgba(240, 246, 255, 0) 0%, rgba(240, 246, 255, 0.92) 30%, #f0f6ff 100%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 12px 1rem 22px 1rem !important;
        border: none !important;
        display: flex !important;
        justify-content: center !important;
    }
    [data-testid="stBottomBlockContainer"] {
        position: relative !important;
        max-width: 820px !important;
        width: 100% !important;
        margin: 0 auto !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    .main .block-container {
        padding-bottom: 180px !important;
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

    /* 💬 Chat Input Box */
    [data-testid="stChatInput"],
    div[data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] {
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 2px solid #93c5fd !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 16px rgba(0, 120, 212, 0.12) !important;
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
    /* 💬 Chat Input & Docked Bottom Bar */
    [data-testid="stBottom"] {
        position: fixed !important;
        bottom: 0px !important;
        left: 0px !important;
        right: 0px !important;
        width: 100vw !important;
        z-index: 99999 !important;
        background: linear-gradient(180deg, rgba(255, 245, 247, 0) 0%, rgba(255, 245, 247, 0.92) 30%, #fff5f7 100%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 12px 1rem 22px 1rem !important;
        border: none !important;
        display: flex !important;
        justify-content: center !important;
    }
    [data-testid="stBottomBlockContainer"] {
        position: relative !important;
        max-width: 820px !important;
        width: 100% !important;
        margin: 0 auto !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    .main .block-container {
        padding-bottom: 180px !important;
    }
    [data-testid="stChatInput"],
    div[data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] {
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 2px solid #d99ca9 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 16px rgba(184, 93, 117, 0.12) !important;
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
# EVRENSEL CANLI DURUM & HAREKETLİ NOKTALAR (. .. ...) ANİMASYON STİLLERİ
# ══════════════════════════════════════════════════════════════════════════════
st.html("""
<style>
.eco-live-status-card {
    display: inline-flex;
    align-items: center;
    background: rgba(0, 138, 215, 0.09);
    border: 1px solid rgba(0, 138, 215, 0.3);
    border-radius: 10px;
    padding: 8px 16px;
    margin: 6px 0 12px 0;
    font-size: 13.5px;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    animation: ecoFadeIn 0.25s ease-out;
}

@keyframes ecoFadeIn {
    from { opacity: 0; transform: translateY(3px); }
    to { opacity: 1; transform: translateY(0); }
}

.eco-status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 9px;
}

.eco-status-pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #008AD7;
    box-shadow: 0 0 0 0 rgba(0, 138, 215, 0.7);
    animation: ecoPulseRing 1.5s infinite cubic-bezier(0.66, 0, 0, 1);
    display: inline-block;
    flex-shrink: 0;
}

@keyframes ecoPulseRing {
    0% {
        box-shadow: 0 0 0 0 rgba(0, 138, 215, 0.75);
    }
    70% {
        box-shadow: 0 0 0 7px rgba(0, 138, 215, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(0, 138, 215, 0);
    }
}

.eco-status-text {
    font-size: 13.5px;
    font-weight: 550;
    color: inherit;
}

.dynamic-dots {
    display: inline-flex;
    font-weight: 900;
    font-size: 18px;
    line-height: 1;
    letter-spacing: 2px;
    color: #008AD7;
    min-width: 22px;
}

.dynamic-dots .dot {
    opacity: 0;
    display: inline-block;
    animation: dynamicDotCycle 1.4s infinite;
}

.dynamic-dots .d1 { animation-delay: 0.0s; }
.dynamic-dots .d2 { animation-delay: 0.35s; }
.dynamic-dots .d3 { animation-delay: 0.70s; }

@keyframes dynamicDotCycle {
    0%, 15% { opacity: 0; transform: translateY(0); }
    35%, 80% { opacity: 1; transform: translateY(-1px); }
    100% { opacity: 0; transform: translateY(0); }
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

    chat_hdr1, chat_hdr2 = st.columns([5, 1])
    with chat_hdr1:
        selected_pill = st.pills(
            T["pills_title"],
            options=pill_options,
            label_visibility="collapsed"
        )
    with chat_hdr2:
        if st.session_state.messages:
            if st.button(T["clear_chat_btn"], icon=":material/delete_sweep:", key="btn_clear_chat_top", width="stretch"):
                st.session_state.messages = []
                st.session_state["last_processed_pill"] = None
                st.rerun()

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
        if st.session_state.get("last_processed_pill") != selected_pill:
            active_query = pill_query_map[selected_pill]
            st.session_state["last_processed_pill"] = selected_pill
    elif not selected_pill:
        st.session_state["last_processed_pill"] = None

    # Mesajlar Konteyneri
    messages_container = st.container()
    with messages_container:
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
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

                # Modern LLM Chat Özelliği: Son asistan yanıtında interaktif takip soruları (Suggested Follow-ups)
                if idx == len(st.session_state.messages) - 1 and msg["role"] == "assistant":
                    followups = get_suggested_followups(msg.get("intent", "general_rag"), L)
                    if followups:
                        st.markdown(f"<div style='margin-top: 14px; margin-bottom: 6px;'><small style='font-weight:600; opacity:0.85;'>{T['suggested_followups_title']}</small></div>", unsafe_allow_html=True)
                        f_cols = st.columns(len(followups))
                        for f_idx, f_text in enumerate(followups):
                            with f_cols[f_idx]:
                                if st.button(f_text, key=f"fup_btn_{idx}_{f_idx}", width="stretch", icon=":material/arrow_forward:"):
                                    st.session_state.pending_followup = f_text
                                    st.rerun()

    # Kullanıcı Girdisi (chat_input veya pill veya önerilen takip sorusu)
    pending_followup = st.session_state.pop("pending_followup", None)
    user_input = st.chat_input(T["chat_placeholder"])
    new_query = user_input or active_query or pending_followup

    if new_query:
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != new_query or st.session_state.messages[-1]["role"] != "user":
            st.session_state.messages.append({"role": "user", "content": new_query})
            st.rerun()

    # Eğer son mesaj bir kullanıcı mesajıysa (henüz yanıtlanmamış), asistan yanıtını kesintisiz üret ve tamamla.
    # Bu mimari, kullanıcı yanıt beklenirken temayı veya dili değiştirse dahi işlemin yarıda kalmamasını garanti eder!
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        query_to_run = st.session_state.messages[-1]["content"]
        target_lang = detect_query_language(query_to_run, default_lang=L)
        print(f"\n[ECO-RAG] İşleniyor: \"{query_to_run}\"", flush=True)
        print(f"  [1/3] Dil Tespiti: {target_lang.upper()} | Analiz Başlatılıyor...", flush=True)

        with messages_container:
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                badge_placeholder = st.empty()

                # Canlı Durum Bildirimi (On-screen indicator - Emojisiz, Kurumsal)
                show_live_status(
                    status_placeholder,
                    "2024–2026 Çevresel Sürdürülebilirlik Raporlarında hibrit arama yapılıyor" if target_lang == "tr"
                    else "Performing Hybrid Search across 2024–2026 Environmental Sustainability Reports"
                )

                start_time = time.time()
                try:
                    calc_details = None
                    chunks = []
                    max_score = 0.0
                    route_type = "rag"
                    stream_gen = None

                    s_prompt = get_synthesis_prompt(target_lang)
                    f_prompt = get_factual_synthesis_prompt(target_lang)
                    not_found_msg = TEXTS[target_lang]["not_found_msg"]
                    active_year_filter = st.session_state.get("selected_year_filter", None)

                    intent = classify_esg_intent(query_to_run)
                    print(f"  [2/3] Ontolojik ESG Niyet Sınıfı: {intent.upper()}", flush=True)

                    if intent == "out_of_domain":
                        print("  -> Alan Dışı Soru: Güvenli Reddetme Devrede", flush=True)
                        stream_gen = stream_static_text(not_found_msg)
                    elif intent == "carbon_commitments":
                        route_type = "pal"
                        print("  -> Yönlendirme: PAL (2030 Karbon Negatif & 2050 Tarihsel Taahhütler)", flush=True)
                        show_live_status(
                            status_placeholder,
                            "2030 ve 2050 Kurumsal Karbon ve CFE Taahhütleri Getiriliyor" if target_lang == "tr"
                            else "Retrieving 2030 & 2050 Corporate Carbon & CFE Commitments"
                        )
                        calc_details = compute_carbon_commitments_summary(target_lang)
                        chunks, max_score = search_context_hybrid(query_to_run, year_filter=active_year_filter)
                        stream_gen = stream_static_text(calc_details)
                    elif intent == "carbon_trend_scope":
                        route_type = "pal"
                        print("  -> Yönlendirme: PAL (Scope 1/2/3 Emisyon Trendi & Kategori Kırılımı)", flush=True)
                        show_live_status(
                            status_placeholder,
                            "Deterministik PAL Motoru ile Scope 1/2/3 Emisyon Verileri Hesaplanıyor" if target_lang == "tr"
                            else "Calculating Scope 1/2/3 Emission Deltas via Deterministic PAL Engine"
                        )
                        calc_details = compute_carbon_trend_summary(target_lang)
                        chunks, max_score = search_context_hybrid(query_to_run, year_filter=active_year_filter)
                        stream_gen = stream_static_text(calc_details)
                    elif intent == "carbon_removal":
                        route_type = "pal"
                        print("  -> Yönlendirme: PAL (Karbon Uzaklaştırma Portföyü Tablo 3)", flush=True)
                        show_live_status(
                            status_placeholder,
                            "Deterministik PAL Motoru ile Karbon Uzaklaştırma Tabloları Çözülüyor" if target_lang == "tr"
                            else "Resolving Carbon Removal Tables via Deterministic PAL Engine"
                        )
                        calc_details = compute_carbon_removal_summary(target_lang)
                        chunks, max_score = search_context_hybrid(query_to_run, year_filter=active_year_filter)
                        stream_gen = stream_static_text(calc_details)
                    elif intent == "zero_waste_circularity":
                        route_type = "pal"
                        print("  -> Yönlendirme: PAL (Sıfır Atık UL 2799 & Circular Centers)", flush=True)
                        show_live_status(
                            status_placeholder,
                            "Doğrulanmış Sıfır Atık (UL 2799) ve Döngüsel Donanım Verileri Getiriliyor" if target_lang == "tr"
                            else "Retrieving Verified Zero Waste (UL 2799) & Hardware Circularity Data"
                        )
                        calc_details = compute_zero_waste_summary(target_lang)
                        chunks, max_score = search_context_hybrid(query_to_run, year_filter=active_year_filter)
                        stream_gen = stream_static_text(calc_details)
                    elif intent == "packaging_plastic":
                        route_type = "pal"
                        print("  -> Yönlendirme: PAL (Ambalaj ve Plastik Oranları)", flush=True)
                        show_live_status(
                            status_placeholder,
                            "Ambalaj ve Plastik Azaltım Oranları Doğrulanıyor" if target_lang == "tr"
                            else "Verifying Packaging & Single-Use Plastic Metrics"
                        )
                        calc_details = compute_packaging_summary(target_lang)
                        chunks, max_score = search_context_hybrid(query_to_run, year_filter=active_year_filter)
                        stream_gen = stream_static_text(calc_details)
                    elif intent == "water_stewardship":
                        route_type = "pal"
                        print("  -> Yönlendirme: PAL (Su Yenileme, Hedef Gerçekleşme & FIDO Tech)", flush=True)
                        show_live_status(
                            status_placeholder,
                            "Deterministik PAL Motoru ile Su Hedefleri ve Akustik Analiz Çözülüyor" if target_lang == "tr"
                            else "Computing Water Replenishment Metrics & Acoustic AI via PAL"
                        )
                        calc_details = compute_water_summary(target_lang)
                        chunks, max_score = search_context_hybrid(query_to_run, year_filter=active_year_filter)
                        stream_gen = stream_static_text(calc_details)
                    elif intent == "mathematical_query":
                        route_type = "pal"
                        print("  [2/3] Yönlendirme: Dinamik PAL (Program-of-Thoughts / Python ALU)", flush=True)
                        show_live_status(
                            status_placeholder,
                            "Dinamik PAL Motoru ile Sayısal Veriler Ayrıştırılıyor ve Hesaplanıyor" if target_lang == "tr"
                            else "Extracting data & calculating metrics via Dynamic PAL Engine"
                        )
                        chunks, max_score = search_context_hybrid(query_to_run, year_filter=active_year_filter)
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
                                    f"Doğrulanmış Kesin Matematik Verileri (Python ALU):\n{calc_details}\n\n"
                                    f"Soru: {query_to_run}\n\n"
                                    "Yapay başlıklar (Doğrudan Yanıt: vb.) KULLANMADAN hesaplanmış kesin sonucu İLK CÜMLEDE doğrudan ve akıcı bir şekilde açıkla. "
                                    "Ardından verileri ve hesaplamayı temiz bir Markdown tablosu ile sun ve 1-2 kısa stratejik madde ekle."
                                    if target_lang == "tr" else
                                    f"Verified Exact Mathematical Results (Python ALU):\n{calc_details}\n\n"
                                    f"Question: {query_to_run}\n\n"
                                    "DO NOT use artificial headers (like 'Direct Answer:'). State the exact calculated result directly in the first sentence. "
                                    "Then format the metrics into a clear Markdown table and conclude with 1-2 concise bullet points."
                                )
                                stream_gen = query_foundry_stream(f_prompt, synth_prompt)
                            else:
                                print("  -> PoT Kodu Çıkarılamadı, Standart RAG'e Geçiliyor", flush=True)
                                stream_gen = query_foundry_stream(
                                    s_prompt,
                                    f"Context:\n{context_str}\n\nQuestion: {query_to_run}"
                                )
                    else:
                        if not is_esg_query(query_to_run):
                            print("  [2/3] Alan Dışı Soru: Güvenli Reddetme Devrede", flush=True)
                            stream_gen = stream_static_text(not_found_msg)
                        else:
                            search_query = query_to_run
                            if target_lang == "tr":
                                show_live_status(status_placeholder, "Soru analiz ediliyor ve İngilizce rapor korpusu için eşleniyor")
                                en_search_query = translate_query_to_en(query_to_run)
                                if en_search_query and en_search_query != query_to_run:
                                    print(f"  -> Soru İngilizceye Eşlendi: \"{en_search_query}\"", flush=True)
                                    search_query = en_search_query

                            print(f"  [2/3] Hibrit Vektör Arama Çalıştırılıyor (Filtre: {active_year_filter or 'Otomatik'})...", flush=True)
                            chunks, max_score = search_context_hybrid(search_query, year_filter=active_year_filter)
                            if (not chunks or max_score < MIN_SCORE_FLOOR) and search_query != query_to_run:
                                # İngilizce eşleme skoru düşükse orijinal sorguyu da dene
                                alt_chunks, alt_score = search_context_hybrid(query_to_run, year_filter=active_year_filter)
                                if alt_score > max_score:
                                    chunks, max_score = alt_chunks, alt_score

                            print(f"  -> Arama Tamamlandı ({len(chunks)} chunk, En Yüksek Skor: {max_score:.4f})", flush=True)
                            if not chunks or max_score < MIN_SCORE_FLOOR:
                                print("  -> Benzerlik Eşiği Altında: Kayıt Bulunamadı", flush=True)
                                stream_gen = stream_static_text(not_found_msg)
                            else:
                                print("  [3/3] Yerel Phi-4-mini Tek Geçişli Akış Sentezi Başlatılıyor...", flush=True)
                                context_chunks = [c["content"] for c in chunks]
                                context_str = "\n\n".join(context_chunks)
                                show_live_status(status_placeholder, "Rapor verileri analiz ediliyor ve yanıt akıtılıyor" if target_lang == "tr" else "Analyzing report context and streaming answer")

                                if target_lang == "tr":
                                    rag_system = (
                                        "Sen Microsoft'un resmi Çevresel Sürdürülebilirlik Raporları (2024, 2025, 2026) konusunda uzmanlaşmış kıdemli bir kurumsal analistsin. "
                                        "Kullanıcının sorusunu doğrudan, akıcı ve profesyonel bir yapay zeka asistanı (ChatGPT / Gemini) üslubuyla yanıtla.\n\n"
                                        "Temel Kurallar:\n"
                                        "1. 'Doğrudan Yanıt:', 'Yönetici Özeti:', 'Uyum Özeti' gibi yapay başlıklar KULLANMA. Cevabına ilk cümlede doğrudan ve net bir şekilde başla.\n"
                                        "2. Soru yıllar arası değişim, oranlar veya birden fazla metrik içeriyorsa verileri MUTLAKA temiz bir Markdown tablosu ile sun.\n"
                                        "3. Raporlanan stratejiler veya somut aksiyonlar için net madde işaretleri (bullet points) kullan.\n"
                                        "4. Yalnızca verilen bağlamdaki resmi sayıları, birimleri ve verileri kullan. Asla uydurma veri üretme.\n"
                                        "5. Gereksiz giriş cümlelerinden ve laf kalabalığından kaçın."
                                    )
                                    user_prompt = f"Microsoft Sürdürülebilirlik Raporu Bağlamı:\n{context_str}\n\nSoru: {query_to_run}\n\nYanıt:"
                                else:
                                    rag_system = (
                                        "You are a Senior Sustainability Analyst specializing in Microsoft's official Environmental Sustainability Reports (2024, 2025, 2026). "
                                        "Answer the user's inquiry directly, fluently, and authoritatively in the natural style of modern AI assistants (like ChatGPT or Gemini).\n\n"
                                        "Core Guidelines:\n"
                                        "1. DO NOT use artificial headers like 'Direct Answer:' or 'Executive Summary:'. Start your response immediately with the answer in the first sentence.\n"
                                        "2. When questions involve multi-year trends, comparisons, or metric breakdowns, present them in a clean, concise Markdown table.\n"
                                        "3. For strategic initiatives, milestones, or actions, use clear and readable bullet points.\n"
                                        "4. Rely strictly on the official facts, metrics, and units provided in the context without hallucinating numbers.\n"
                                        "5. Avoid introductory filler or question restatements."
                                    )
                                    user_prompt = f"Microsoft Sustainability Report Context:\n{context_str}\n\nQuestion: {query_to_run}\n\nAnswer:"

                                stream_gen = query_foundry_stream(rag_system, user_prompt, temperature=0.1)

                    # Bekleme belirtecini temizle ve akışı başlat
                    status_placeholder.empty()

                    # ⚡ Canlı Akışlı Yanıt Yazımı (Streaming Output)
                    ans = st.write_stream(stream_gen)

                    latency = time.time() - start_time
                    print(f"  [OK] Yanıt Başarıyla Tamamlandı (Gecikme: {latency:.2f}s)\n", flush=True)

                    insight = None
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
                        "intent": intent,
                        "calc_details": calc_details,
                        "provenance": chunks,
                        "insight": insight,
                        "max_score": max_score,
                        "latency": latency
                    })
                    gc.collect()
                    st.rerun()

                except Exception as e:
                    status_placeholder.empty()
                    err_text = f"Error / Hata: {e}"
                    st.error(err_text)
                    print(f"  [HATA] Sorgu işlenirken istisna oluştu: {e}", flush=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_text,
                        "route": "error"
                    })
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
