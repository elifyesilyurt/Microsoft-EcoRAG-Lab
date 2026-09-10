"""
run_benchmarks.py — Microsoft EcoRAG Lab v2.1 — 50-Question Multi-Year Benchmark Suite
========================================================================================
Kapsam : 2024, 2025, 2026 Microsoft Çevresel Sürdürülebilirlik Raporları
Zorluk : 5 Kategori (Kolay, Orta, Zor/PAL, Trend/3-Yıl, Alan Dışı/Negatif)
Senaryolar : Karbon (carbon), Su (water), Enerji (energy), Atık (waste), Alan Dışı (negative)
Kullanıcı Tipleri:
  analyst    — ESG/Sürdürülebilirlik Analisti (sayısal hassasiyet, metrik çıkarımı)
  auditor    — Çevre Denetçisi (sertifikasyonlar, doğrulama standartları)
  researcher — Araştırmacı (metodoloji, bölgesel projeler)
  executive  — Üst Yönetim (stratejik hedefler, çok yıllı büyüme oranları)

Kullanım:
  python run_benchmarks.py
  python run_benchmarks.py --difficulty hard
  python run_benchmarks.py --scenario carbon
  python run_benchmarks.py --user-type analyst
  python run_benchmarks.py --only 1,5,12,31,44
"""

import gc
import json
import re
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from extraction_pipeline import (
    EXTRACTION_SYSTEM_PROMPT,
    DeterministicResolver,
    QueryExtractionPlan,
    format_extraction_prompt,
)
from esg_tables import (
    get_carbon_emissions_df,
    get_carbon_removal_by_type_df,
    get_carbon_removal_df,
    get_water_metrics_df,
    get_water_replenishment_projects_df,
    get_energy_metrics_df,
    get_waste_metrics_df,
    get_zero_waste_certifications_df,
)

DB_PATH         = "rag_storage.db"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
FOUNDRY_BASE_URL = os.getenv("FOUNDRY_BASE_URL", "http://127.0.0.1:62095")
FOUNDRY_URL      = f"{FOUNDRY_BASE_URL}/v1/chat/completions"
PHI_MODEL        = os.getenv("FOUNDRY_MODEL_NAME", "phi-4-mini")
MAX_K           = 6
DROP_RATIO      = 0.70
MIN_SCORE       = 0.15
MAX_TOKENS      = 512

SYNTHESIS_PROMPT = (
    "You are a Senior Sustainability Analyst. "
    "Synthesize the verified analytical calculation data into a concise, precise executive answer. "
    "State exact numbers, percentages, and units in sentence 1. Do NOT repeat yourself."
)
FACTUAL_PROMPT = (
    "You are a Senior Sustainability AI Analyst. "
    "Using ONLY the verified structured metrics below, give a direct, concise answer. "
    "State exact numbers, names, and units in sentence 1. Do NOT repeat yourself."
)
GROUNDING_PROMPT = (
    "You are a precise Sustainability Analyst. "
    "Answer using ONLY the context provided. "
    "If context contains [VISUAL REFERENCE] warnings, state that graphical data is not available as text. "
    "If the answer is not in the context, respond EXACTLY: "
    "'I cannot find information regarding this in the provided Microsoft Environmental Sustainability reports.'"
)


class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    GREY    = "\033[90m"
    WHITE   = "\033[97m"


def hr(char="=", width=80):
    return char * width


@dataclass
class BenchmarkQuestion:
    qid:           int
    difficulty:    str
    section:       str
    user_type:     str
    scenario:      str
    question:      str
    expected_hint: str
    report_years:  str
    use_pal:       bool = False
    pal_data:      str  = ""


QUESTIONS: List[BenchmarkQuestion] = [

    # ── KATEGORİ 1: KOLAY / DIRECT FACTUAL (Q01-Q10) ────────────────────────
    BenchmarkQuestion(
        qid=1, difficulty="easy", section="factual",
        user_type="auditor", scenario="waste",
        question="Which external certification does Microsoft use to validate its Zero Waste datacenters, and how many datacenters were certified under this standard in FY23 according to the 2024 report?",
        expected_hint="UL Solutions Zero Waste to Landfill (UL 2799) | 10 datacenters",
        report_years="2024", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=2, difficulty="easy", section="factual",
        user_type="researcher", scenario="energy",
        question="What alternative fuel did Microsoft transition to for backup generators in specific datacenters like Sweden and the Netherlands to eliminate petroleum diesel?",
        expected_hint="Hydrotreated Vegetable Oil (HVO) / renewable diesel",
        report_years="2024 / 2025",
    ),
    BenchmarkQuestion(
        qid=3, difficulty="easy", section="factual",
        user_type="auditor", scenario="energy",
        question="What new requirement did Microsoft introduce for high-impact suppliers starting in FY25 regarding their electricity use?",
        expected_hint="100% carbon-free electricity (CFE) by 2030 / Supplier Code of Conduct",
        report_years="2025",
    ),
    BenchmarkQuestion(
        qid=4, difficulty="easy", section="factual",
        user_type="executive", scenario="carbon",
        question="What is Microsoft's corporate timeline target year for becoming carbon negative across Scope 1, 2, and 3 emissions?",
        expected_hint="2030 (and remove all historical emissions by 2050)",
        report_years="2024 / 2025 / 2026",
    ),
    BenchmarkQuestion(
        qid=5, difficulty="easy", section="factual",
        user_type="analyst", scenario="water",
        question="What is the 2030 target percentage for water use efficiency improvement across Microsoft's owned datacenters, and what baseline year is it measured against?",
        expected_hint="40% improvement / 2022 baseline",
        report_years="2025",
    ),
    BenchmarkQuestion(
        qid=6, difficulty="easy", section="factual",
        user_type="researcher", scenario="water",
        question="Which organization did Microsoft partner with to deploy AI-enabled acoustic leak analysis in water distribution networks across cities like London, Queretaro, and Phoenix?",
        expected_hint="FIDO Tech",
        report_years="2024 / 2025",
    ),
    BenchmarkQuestion(
        qid=7, difficulty="easy", section="factual",
        user_type="auditor", scenario="waste",
        question="What was Microsoft's reuse and recycle rate for servers and components across all cloud hardware in FY23 according to the 2024 report?",
        expected_hint="89.4%",
        report_years="2024", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=8, difficulty="easy", section="factual",
        user_type="executive", scenario="energy",
        question="What renewable energy share percentage did Microsoft achieve in FY25, and how many MWh were purchased through PPAs and RECs?",
        expected_hint="95.0% share | 41,600,000 MWh purchased",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=9, difficulty="easy", section="factual",
        user_type="analyst", scenario="waste",
        question="What is Microsoft's target percentage for single-use plastic packaging elimination across primary packaging by 2025/2026?",
        expected_hint="Near-zero / 0.07% single-use plastic achieved / 100% recyclable goal",
        report_years="2025 / 2026",
    ),
    BenchmarkQuestion(
        qid=10, difficulty="easy", section="factual",
        user_type="analyst", scenario="waste",
        question="What is Microsoft's 2030 target percentage for diversion of operational waste at owned datacenters and campuses?",
        expected_hint="90% diversion by 2030",
        report_years="2024 / 2025", use_pal=True,
    ),

    # ── KATEGORİ 2: ORTA / MULTI-CONDITION & TABULAR (Q11-Q22) ──────────────
    BenchmarkQuestion(
        qid=11, difficulty="medium", section="factual",
        user_type="analyst", scenario="carbon",
        question="What was the contracted carbon removal volume and share percentage for Direct Air Capture (DAC) in the 2025 report portfolio?",
        expected_hint="4,210,000 tons | 19.2%",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=12, difficulty="medium", section="factual",
        user_type="analyst", scenario="carbon",
        question="What was the contracted volume for Forests & Land-based nature projects and for Biomass/BECCS in the 2025 carbon removal portfolio?",
        expected_hint="Forests: 8,540,000 tons (38.9%) | BECCS: 5,130,000 tons (23.4%)",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=13, difficulty="medium", section="factual",
        user_type="auditor", scenario="carbon",
        question="What were the Scope 2 Market-based greenhouse gas emissions for the FY20 baseline and for FY25 in mtCO2e?",
        expected_hint="FY20: 456,119 mtCO2e | FY25: 2,707,428 mtCO2e",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=14, difficulty="medium", section="factual",
        user_type="analyst", scenario="carbon",
        question="What was the Scope 1 direct emissions total in FY24 and in FY25 in mtCO2e according to the GHG emissions table?",
        expected_hint="FY24: 143,510 mtCO2e | FY25: 170,887 mtCO2e",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=15, difficulty="medium", section="factual",
        user_type="researcher", scenario="water",
        question="What was the cumulative contracted water replenishment volume and the in-year specifically contracted benefit by the end of FY25?",
        expected_hint="Cumulative: 125.0 million m3 | In-year FY25: 35.0 million m3",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=16, difficulty="medium", section="factual",
        user_type="analyst", scenario="water",
        question="How many water replenishment projects were completed in North America and in Asia Pacific regions by FY25 according to the 2025 Water Table?",
        expected_hint="North America: 24 projects | Asia Pacific: 10 projects",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=17, difficulty="medium", section="factual",
        user_type="auditor", scenario="waste",
        question="What was the total operational waste generated and the amount diverted from landfills in FY25 according to the waste metrics table?",
        expected_hint="Total generated: 265,000 metric tons | Diverted: 218,000 metric tons",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=18, difficulty="medium", section="factual",
        user_type="executive", scenario="energy",
        question="How did total electricity consumption change from FY20 baseline to FY24 and FY25 in MWh, and what does this reflect about Microsoft datacenter growth?",
        expected_hint="FY20: 10,200,000 MWh -> FY24: 29,500,000 MWh -> FY25: 43,800,000 MWh (4.3x growth vs baseline)",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=19, difficulty="medium", section="factual",
        user_type="auditor", scenario="waste",
        question="What external certification standard and tier structure is used by UL Solutions for certifying Zero Waste to Landfill facilities?",
        expected_hint="UL 2799 ECVP | Silver (90-94%), Gold (95-99%), Platinum (100%)",
        report_years="2024 / 2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=20, difficulty="medium", section="factual",
        user_type="analyst", scenario="carbon",
        question="What was the contracted volume for Enhanced Weathering & Mineralization in the 2025 carbon removal portfolio?",
        expected_hint="2,347,370 tons | 10.7%",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=21, difficulty="medium", section="factual",
        user_type="analyst", scenario="carbon",
        question="What was the total contracted carbon removal volume reported in the 2024 report (Table 3) and how does it compare to the 2025 report total?",
        expected_hint="2024 Report: 5,015,019 tons vs 2025 Report: 21,927,370 tons (more than 4x increase)",
        report_years="2024 / 2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=22, difficulty="medium", section="factual",
        user_type="executive", scenario="water",
        question="What was Microsoft's water replenishment completion rate in FY24 and FY25, and what does this indicate about progress toward water positive goals?",
        expected_hint="FY24: 68.9% | FY25: 82.1% -- significant improvement toward water positive commitment",
        report_years="2025", use_pal=True,
    ),

    # ── KATEGORİ 3: ZOR / MULTI-YEAR MATH & PAL (Q23-Q32) ──────────────────
    BenchmarkQuestion(
        qid=23, difficulty="hard", section="factual",
        user_type="analyst", scenario="carbon",
        question="What is the net delta and percentage increase in Subtotal Scope 3 emissions between the FY20 baseline and FY25?",
        expected_hint="+5,756,000 mtCO2e increase (+46.1%) from 12,487,000 to 18,243,000 mtCO2e",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=24, difficulty="hard", section="factual",
        user_type="analyst", scenario="carbon",
        question="Which top two Scope 3 categories contributed the highest emissions in FY25, and what was their individual percentage share of total Scope 3 emissions?",
        expected_hint="Cat 2 Capital Goods: 9,044,000 mtCO2e (49.6%) | Cat 1 Purchased Goods: 5,129,000 mtCO2e (28.1%)",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=25, difficulty="hard", section="factual",
        user_type="executive", scenario="carbon",
        question="What was the total GHG emissions growth (Scope 1 + 2 Market-based + 3) from the FY20 baseline to FY25 in absolute mtCO2e and percentage change?",
        expected_hint="From 13,061,000 to 21,121,000 mtCO2e | Delta: +8,060,000 mtCO2e | +61.7% increase",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=26, difficulty="hard", section="factual",
        user_type="researcher", scenario="carbon",
        question="What is the combined percentage share of Scope 3 Category 1 (Purchased Goods) and Category 2 (Capital Goods) out of total company-wide FY25 GHG emissions?",
        expected_hint="14,173,000 mtCO2e out of 21,121,000 total -- ~67.1% of total emissions",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=27, difficulty="hard", section="factual",
        user_type="analyst", scenario="carbon",
        question="Compare the total share of engineered technological removals (DAC + BECCS + Mineralization) versus nature-based removals (Forests) in the 2025 carbon removal portfolio.",
        expected_hint="Tech removals: 53.3% (11,687,370 tons) | Nature-based: 38.9% (8,540,000 tons)",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=28, difficulty="hard", section="factual",
        user_type="auditor", scenario="waste",
        question="How did the solid waste diversion rate and absolute volume improve from the FY20 baseline to FY24 and FY25?",
        expected_hint="FY20: 63.0% (119,000 mt) -> FY24: 79.3% (188,000 mt) -> FY25: 82.3% (218,000 mt)",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=29, difficulty="hard", section="factual",
        user_type="executive", scenario="energy",
        question="What was the growth in renewable energy purchased from the FY20 baseline to FY24 and FY25 in MWh, and how did the renewable share percentage change?",
        expected_hint="FY20: 8.1M MWh (79.4%) -> FY24: 23.4M MWh (79.3%) -> FY25: 41.6M MWh (95.0%)",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=30, difficulty="hard", section="factual",
        user_type="researcher", scenario="water",
        question="Compare the water replenishment achievement rate between FY23, FY24, and FY25, and calculate the volumetric progress against the 2030 water positive target.",
        expected_hint="FY23: 68.9% | FY24: 68.9% | FY25: 82.1% | FY25 completed: 7,800M m3 vs 9,500M m3 target",
        report_years="2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=31, difficulty="hard", section="factual",
        user_type="executive", scenario="carbon",
        question="Between the 2024 report and the 2025 report, by how much did the total contracted carbon removal volume grow in absolute mtCO2e and by what multiplier?",
        expected_hint="From 5,015,019 to 21,927,370 tons | +16,912,351 mtCO2e increase | ~4.37x growth",
        report_years="2024 / 2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=32, difficulty="hard", section="factual",
        user_type="auditor", scenario="carbon",
        question="What was the Scope 1 emissions growth from FY20 baseline to FY24 and FY25 in mtCO2e?",
        expected_hint="FY20: 118,100 -> FY24: 143,510 (+21.5%) -> FY25: 170,887 (+44.7% vs baseline)",
        report_years="2025", use_pal=True,
    ),

    # ── KATEGORİ 4: TREND / 3-YILLIK ÇAPRAZ-RAPOR (Q33-Q42) ────────────────
    BenchmarkQuestion(
        qid=33, difficulty="trend", section="factual",
        user_type="executive", scenario="carbon",
        question="Across the 2024, 2025, and 2026 Microsoft sustainability reports, how has the stated ambition and scale of the carbon removal portfolio evolved over the three reporting periods?",
        expected_hint="2024: 5M tons contracted | 2025: 21.9M tons (+4x growth) | 2026 report: continued scaling / long-term contracts",
        report_years="2024 / 2025 / 2026", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=34, difficulty="trend", section="factual",
        user_type="analyst", scenario="energy",
        question="How did Microsoft's total electricity consumption and renewable energy share evolve from FY20 baseline through the period covered by the 2024 and 2025 reports?",
        expected_hint="FY20: 10.2M MWh (79.4%) -> FY24: 29.5M MWh (79.3%) -> FY25: 43.8M MWh (95.0%) -- major acceleration",
        report_years="2024 / 2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=35, difficulty="trend", section="factual",
        user_type="analyst", scenario="carbon",
        question="Comparing the 2024 and 2025 reports, how did Scope 3 Category 1 (Purchased Goods) and Category 2 (Capital Goods) change in mtCO2e?",
        expected_hint="Cat 1: FY20 4.59M -> FY24 5.61M -> FY25 5.13M | Cat 2: FY20 3.43M -> FY24 6.29M -> FY25 9.04M",
        report_years="2024 / 2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=36, difficulty="trend", section="factual",
        user_type="researcher", scenario="water",
        question="Based on the 2024, 2025, and 2026 reports, how has Microsoft's water withdrawal and water replenishment progress tracked against its 2030 water positive commitment?",
        expected_hint="Withdrawal grew FY20: 4,830 -> FY24: 8,450 -> FY25: 10,210 million m3 | Achievement: 68.9% FY24 -> 82.1% FY25",
        report_years="2024 / 2025 / 2026", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=37, difficulty="trend", section="factual",
        user_type="auditor", scenario="waste",
        question="Trace the evolution of Microsoft's Zero Waste certified datacenter count across the 2024 and 2025 reports, and describe the certification standard used.",
        expected_hint="FY23: 10 certified datacenters (2024 report) -> FY25: 14 certified sites | Standard: UL 2799 ECVP by UL Solutions",
        report_years="2024 / 2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=38, difficulty="trend", section="factual",
        user_type="analyst", scenario="energy",
        question="How does Microsoft's electricity consumption in MWh and renewable energy procurement compare across FY20 baseline, FY24, and FY25 in the Energy summary tables?",
        expected_hint="FY20: 10.2M MWh total (8.1M renewable) -> FY24: 29.5M MWh (23.4M renewable) -> FY25: 43.8M MWh (41.6M renewable)",
        report_years="2024 / 2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=39, difficulty="trend", section="factual",
        user_type="analyst", scenario="carbon",
        question="How did the split between in-year carbon neutrality volumes and 2030 carbon negative target volumes change between the 2024 and 2025 carbon removal portfolio tables?",
        expected_hint="2024: In-year 3,549,242 / 2030 target 1,465,777 | 2025: In-year 1,690,940 / 2030 target 2,804,056 / Post-2031: 17,432,374",
        report_years="2024 / 2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=40, difficulty="trend", section="factual",
        user_type="researcher", scenario="energy",
        question="How does Microsoft describe the expansion of on-site renewable generation from FY20 baseline to FY25 across its datacenter and campus operations?",
        expected_hint="FY20: 52,000 MWh -> FY24: 98,000 MWh -> FY25: 130,000 MWh on-site generation",
        report_years="2024 / 2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=41, difficulty="trend", section="factual",
        user_type="researcher", scenario="water",
        question="Across the 2024 and 2025 water tables, how did completed water replenishment project counts change in North America and Latin America regions?",
        expected_hint="North America: FY24 18 -> FY25 24 projects | Latin America: FY24 11 -> FY25 15 projects",
        report_years="2024 / 2025", use_pal=True,
    ),
    BenchmarkQuestion(
        qid=42, difficulty="trend", section="factual",
        user_type="auditor", scenario="carbon",
        question="Across all three years (2024, 2025, 2026 reports), how has Microsoft's Scope 3 Category 2 (Capital Goods) emissions trend changed and what is the stated cause?",
        expected_hint="FY24: 6,291,000 mtCO2e -> FY25: 9,044,000 mtCO2e | +43.8% increase | driven by AI infrastructure / server hardware procurement",
        report_years="2024 / 2025 / 2026", use_pal=True,
    ),

    # ── KATEGORİ 5: ALAN DIŞI / NEGATİF KONTROL (Q43-Q50) ──────────────────
    BenchmarkQuestion(
        qid=43, difficulty="negative", section="negative",
        user_type="analyst", scenario="negative",
        question="What was the average round-trip network latency between the Quincy datacenter and the San Antonio Azure edge site in milliseconds during 2024?",
        expected_hint="REJECT -- network latency not in ESG reports",
        report_years="N/A",
    ),
    BenchmarkQuestion(
        qid=44, difficulty="negative", section="negative",
        user_type="executive", scenario="negative",
        question="What is the exact clock speed in GHz and cache size of the custom processors used inside the servers at the Boydton datacenter?",
        expected_hint="REJECT -- hardware CPU specs not in ESG reports",
        report_years="N/A",
    ),
    BenchmarkQuestion(
        qid=45, difficulty="negative", section="negative",
        user_type="analyst", scenario="negative",
        question="What was Microsoft's total marketing and advertising spend in US dollars during fiscal year 2025?",
        expected_hint="REJECT -- marketing spend not in sustainability reports",
        report_years="N/A",
    ),
    BenchmarkQuestion(
        qid=46, difficulty="negative", section="negative",
        user_type="researcher", scenario="negative",
        question="Who won the FIFA Women's World Cup in 2023, and what was the final score?",
        expected_hint="REJECT -- completely out of domain",
        report_years="N/A",
    ),
    BenchmarkQuestion(
        qid=47, difficulty="negative", section="negative",
        user_type="auditor", scenario="negative",
        question="What was the exact base salary and stock compensation breakdown for Satya Nadella in fiscal year 2025?",
        expected_hint="REJECT -- executive compensation not in environmental reports",
        report_years="N/A",
    ),
    BenchmarkQuestion(
        qid=48, difficulty="negative", section="negative",
        user_type="executive", scenario="negative",
        question="What are the specific kernel bug patch numbers included in the Windows 11 24H2 security update released in 2025?",
        expected_hint="REJECT -- OS patch details not in sustainability reports",
        report_years="N/A",
    ),
    BenchmarkQuestion(
        qid=49, difficulty="negative", section="negative",
        user_type="analyst", scenario="negative",
        question="What was the closing stock price of Microsoft (MSFT) on NASDAQ on March 15, 2026 in USD?",
        expected_hint="REJECT -- daily stock ticker price not in ESG reports",
        report_years="N/A",
    ),
    BenchmarkQuestion(
        qid=50, difficulty="negative", section="negative",
        user_type="researcher", scenario="negative",
        question="What is the maximum color depth and HDR chroma subsampling rate supported by the Xbox Series X HDMI 2.1 port?",
        expected_hint="REJECT -- gaming console video specs not in ESG reports",
        report_years="N/A",
    ),
]


@dataclass
class BenchmarkResult:
    qid:              int
    difficulty:       str
    section:          str
    user_type:        str
    scenario:         str
    question:         str
    expected_hint:    str
    report_years:     str
    answer:           str
    latency_s:        float
    retrieval_chunks: int
    max_score:        float
    pydantic_ok:      Optional[bool] = None
    verified_metrics: int            = 0
    visual_chunks:    int            = 0
    pal_used:         bool           = False
    verdict:          str            = "--"


print(f"{C.CYAN}Embedding modeli yukleniyor: {EMBEDDING_MODEL}...{C.RESET}")
_embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
print(f"{C.GREEN}Model hazir.{C.RESET}\n")


def _call_phi(system, user):
    payload = {
        "model": PHI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }
    try:
        with requests.Session() as sess:
            r = sess.post(
                FOUNDRY_URL, json=payload,
                headers={"Content-Type": "application/json", "Connection": "close"},
                timeout=180,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            return f"[HTTP {r.status_code}] {r.text[:300]}"
    finally:
        gc.collect()


def _normalize_str(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s.lower())
        if unicodedata.category(c) != 'Mn'
    )


def _hybrid_search(query):
    qvec = _embedder.encode(f"search_query: {query}", normalize_embeddings=True).astype(np.float32)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, year, title, content, embedding FROM documents").fetchall()
    conn.close()
    stopwords = {"which", "what", "where", "when", "that", "this", "from", "into",
                 "over", "with", "across", "like", "does", "have", "been", "according"}
    clean_q  = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
    keywords = [_normalize_str(w) for w in clean_q.split()
                if len(w) > 2 and w.lower() not in stopwords]
    scored = []
    for rid, year, title, content, emb_json in rows:
        dvec         = np.array(json.loads(emb_json), dtype=np.float32)
        sim          = float(np.dot(qvec, dvec))
        match_count  = sum(1 for kw in keywords if kw in _normalize_str(content))
        hybrid_score = sim + (0.10 * match_count)
        scored.append({"id": rid, "year": year, "title": title, "content": content, "score": hybrid_score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    del rows; gc.collect()
    if not scored or scored[0]["score"] < MIN_SCORE:
        return [], 0.0
    max_s  = scored[0]["score"]
    cutoff = max_s * DROP_RATIO
    return [x for x in scored[:MAX_K] if x["score"] >= cutoff], max_s


def _is_negative_answer(text):
    neg_phrases = ["cannot find information", "not find information", "does not contain",
                   "not available in", "not covered", "not mentioned", "no information",
                   "outside the scope", "not in the provided"]
    low = text.lower()
    return any(p in low for p in neg_phrases)


def _auto_verdict(q, result):
    if q.section == "negative":
        return "REJECT+OK" if _is_negative_answer(result.answer) else "HALLUCINATION!"
    if _is_negative_answer(result.answer):
        return "FAIL"
    if len(result.answer.strip()) < 30:
        return "CAUTION"
    return "PASS"


def run_single(q):
    t0 = time.time()
    result = BenchmarkResult(
        qid=q.qid, difficulty=q.difficulty, section=q.section,
        user_type=q.user_type, scenario=q.scenario,
        question=q.question, expected_hint=q.expected_hint,
        report_years=q.report_years,
        answer="", latency_s=0.0, retrieval_chunks=0, max_score=0.0,
    )

    if q.use_pal and q.pal_data:
        answer = _call_phi(SYNTHESIS_PROMPT,
                           f"Question: {q.question}\n\nVerified Analytical Calculation Data:\n{q.pal_data}")
        chunks, max_s           = _hybrid_search(q.question)
        result.answer           = answer
        result.latency_s        = round(time.time() - t0, 2)
        result.retrieval_chunks = len(chunks)
        result.max_score        = round(max_s, 4)
        result.pal_used         = True
        result.verdict          = _auto_verdict(q, result)
        return result

    chunks, max_s           = _hybrid_search(q.question)
    result.retrieval_chunks = len(chunks)
    result.max_score        = round(max_s, 4)
    visual_chunks           = [c for c in chunks if "VISUAL" in c["content"][:250]]
    result.visual_chunks    = len(visual_chunks)

    if not chunks or max_s < MIN_SCORE:
        result.answer    = ("I cannot find information regarding this in the provided "
                           "Microsoft Environmental Sustainability reports.")
        result.latency_s = round(time.time() - t0, 2)
        result.verdict   = _auto_verdict(q, result)
        return result

    context = [c["content"] for c in chunks]
    raw_json = _call_phi(EXTRACTION_SYSTEM_PROMPT, format_extraction_prompt(q.question, context))
    try:
        cleaned = re.search(r"\{.*\}", raw_json, re.DOTALL).group(0)
        plan    = QueryExtractionPlan(**json.loads(cleaned))
        res     = DeterministicResolver.validate_and_filter(plan, q.question)
        if res["status"] == "NOT_FOUND":
            result.pydantic_ok = False
            answer = _call_phi(GROUNDING_PROMPT, f"Context:\n{chr(10).join(context)}\n\nQuestion: {q.question}")
        else:
            result.pydantic_ok      = True
            result.verified_metrics = len(res["metrics"])
            verified_str = "\n".join([
                f"- Entity: {m.entity}, Type: {m.metric_type}, "
                f"Value: {m.string_value if m.string_value else f'{m.value:,.0f} {m.unit}'}, "
                f"Scope: {m.temporal_scope}, Cumulative: {m.is_cumulative}"
                for m in res["metrics"]
            ])
            answer = _call_phi(FACTUAL_PROMPT, f"Verified Metrics:\n{verified_str}\n\nQuestion: {q.question}")
    except Exception:
        result.pydantic_ok = False
        answer = _call_phi(GROUNDING_PROMPT, f"Context:\n{chr(10).join(context)}\n\nQuestion: {q.question}")

    result.answer    = answer
    result.latency_s = round(time.time() - t0, 2)
    result.verdict   = _auto_verdict(q, result)
    return result


def _verdict_color(verdict):
    return {"PASS": C.GREEN, "CAUTION": C.YELLOW, "FAIL": C.RED,
            "REJECT+OK": C.GREEN, "HALLUCINATION!": C.RED}.get(verdict, C.WHITE)


def _diff_badge(diff):
    return {"easy": f"{C.GREEN}[KOLAY]{C.RESET}", "medium": f"{C.CYAN}[ORTA]{C.RESET}",
            "hard": f"{C.YELLOW}[ZOR/PAL]{C.RESET}", "trend": f"{C.MAGENTA}[TREND/3-YIL]{C.RESET}",
            "negative": f"{C.RED}[ALAN DISI]{C.RESET}"}.get(diff, f"[{diff}]")


def _user_label(ut):
    return {"analyst": "Analist", "executive": "Yonetim", "researcher": "Arastirmaci",
            "auditor": "Denetci"}.get(ut, ut)


def _scen_label(sc):
    return {"carbon": "Karbon", "water": "Su", "energy": "Enerji",
            "waste": "Atik", "negative": "Disi"}.get(sc, sc)


def print_result(q, r):
    vc = _verdict_color(r.verdict)
    print(f"\n{hr()}")
    print(f" Q{r.qid:02d} {_diff_badge(q.difficulty)}  {vc}{r.verdict}{C.RESET}  "
          f"{C.GREY}Latency: {r.latency_s:.2f}s  Chunks: {r.retrieval_chunks}  Score: {r.max_score:.4f}{C.RESET}")
    print(f"    {C.GREY}Kullanici: {_user_label(q.user_type)}  "
          f"Senaryo: {_scen_label(q.scenario)}  Rapor: {q.report_years}{C.RESET}")
    print(f"{hr('-')}")
    print(f" Soru    : {q.question}")
    print(f" Beklenen: {q.expected_hint}")
    flags = []
    if r.pal_used:          flags.append(f"{C.GREEN}PAL Engine OK{C.RESET}")
    if r.pydantic_ok is True:  flags.append(f"{C.GREEN}Pydantic OK ({r.verified_metrics} metrik){C.RESET}")
    elif r.pydantic_ok is False: flags.append(f"{C.YELLOW}Pydantic->Fallback{C.RESET}")
    if r.visual_chunks > 0: flags.append(f"{C.YELLOW}VisualChunk:{r.visual_chunks}{C.RESET}")
    if flags: print(f" Pipeline: {' | '.join(flags)}")
    print(f"{hr('-')}")
    lines = []
    for word in r.answer.split():
        if not lines or len(lines[-1]) + len(word) + 1 > 115: lines.append(word)
        else: lines[-1] += " " + word
    for line in lines: print(f"  {line}")


def print_summary(results):
    total_lat = sum(r.latency_s for r in results)
    print(f"\n{hr('=')}")
    print(f" Microsoft EcoRAG Lab v2.1 -- 50-Soru Benchmark Ozet Raporu")
    print(f" Raporlar: 2024 | 2025 | 2026 -- Toplam Sure: {total_lat:.1f}s (~{total_lat/60:.1f} dk)")
    print(f"{hr('=')}")

    cat_info = [
        ("easy",     "Kat.1: Kolay / Direct Factual (10 Soru)"),
        ("medium",   "Kat.2: Orta / Multi-Condition & Tabular (12 Soru)"),
        ("hard",     "Kat.3: Zor / Multi-Year Math & PAL (10 Soru)"),
        ("trend",    "Kat.4: Trend / 3-Yillik Capraz Karsilastirma (10 Soru)"),
        ("negative", "Kat.5: Alan Disi / Negatif Kontrol (8 Soru)"),
    ]
    print(f"\n-- ZORLUK KATEGORISi BAZINDA PERFORMANS --")
    for diff, title in cat_info:
        sub = [r for r in results if r.difficulty == diff]
        if not sub: continue
        c_pass = sum(1 for r in sub if r.verdict in ("PASS", "REJECT+OK"))
        c_fail = sum(1 for r in sub if r.verdict in ("FAIL", "HALLUCINATION!"))
        c_caut = sum(1 for r in sub if r.verdict == "CAUTION")
        avg_lat = sum(r.latency_s for r in sub) / len(sub)
        pct    = c_pass / len(sub) * 100
        color  = C.GREEN if pct == 100 else C.YELLOW if pct >= 80 else C.RED
        bar    = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"\n {title}")
        print(f"  [{color}{bar}{C.RESET}] {color}{pct:.0f}%{C.RESET}  "
              f"{C.GREEN}OK:{c_pass}{C.RESET}  {C.RED}FAIL:{c_fail}{C.RESET}  "
              f"{C.YELLOW}CAUTION:{c_caut}{C.RESET}  {C.GREY}avg:{avg_lat:.2f}s{C.RESET}")

    print(f"\n{hr('-')}")
    print(f"-- KULLANICI TiPi BAZINDA PERFORMANS --")
    for ut in ["analyst", "executive", "researcher", "auditor"]:
        sub = [r for r in results if r.user_type == ut and r.section == "factual"]
        if not sub: continue
        c_pass = sum(1 for r in sub if r.verdict == "PASS")
        pct    = c_pass / len(sub) * 100
        color  = C.GREEN if pct == 100 else C.YELLOW if pct >= 80 else C.RED
        print(f"  {_user_label(ut):15s}  {color}{pct:5.1f}%{C.RESET}  "
              f"({c_pass}/{len(sub)} PASS)  "
              f"{C.GREY}avg:{sum(r.latency_s for r in sub)/len(sub):.2f}s{C.RESET}")

    print(f"\n{hr('-')}")
    print(f"-- SENARYO BAZINDA PERFORMANS --")
    for sc in ["carbon", "water", "energy", "waste"]:
        sub = [r for r in results if r.scenario == sc]
        if not sub: continue
        c_pass = sum(1 for r in sub if r.verdict == "PASS")
        pct    = c_pass / len(sub) * 100
        color  = C.GREEN if pct == 100 else C.YELLOW if pct >= 80 else C.RED
        print(f"  {_scen_label(sc):12s}  {color}{pct:5.1f}%{C.RESET}  "
              f"({c_pass}/{len(sub)} PASS)  "
              f"{C.GREY}avg:{sum(r.latency_s for r in sub)/len(sub):.2f}s{C.RESET}")

    print(f"\n{hr('-')}")
    print(f"-- RAPOR YILI BAZINDA ERiSiM KALiTESi --")
    for year_tag, label in [("2024", "2024 Raporu"), ("2025", "2025 Raporu"), ("2026", "2026 Raporu")]:
        sub = [r for r in results if year_tag in r.report_years and r.section == "factual"]
        if not sub: continue
        avg_score  = sum(r.max_score for r in sub) / len(sub)
        avg_chunks = sum(r.retrieval_chunks for r in sub) / len(sub)
        c_pass     = sum(1 for r in sub if r.verdict == "PASS")
        pct        = c_pass / len(sub) * 100
        color      = C.GREEN if pct == 100 else C.YELLOW if pct >= 80 else C.RED
        print(f"  {label}  {color}{pct:.0f}% PASS{C.RESET}  "
              f"Avg Score: {avg_score:.4f}  Avg Chunks: {avg_chunks:.1f}")

    print(f"\n{hr('-')}")
    factual_all  = [r for r in results if r.section == "factual"]
    negative_all = [r for r in results if r.section == "negative"]
    trend_all    = [r for r in results if r.difficulty == "trend" and r.section == "factual"]
    pal_used_all = [r for r in results if r.pal_used]
    f_rate = (sum(1 for r in factual_all if r.verdict == "PASS") / len(factual_all) * 100) if factual_all else 0
    n_rate = (sum(1 for r in negative_all if r.verdict == "REJECT+OK") / len(negative_all) * 100) if negative_all else 0
    t_rate = (sum(1 for r in trend_all if r.verdict == "PASS") / len(trend_all) * 100) if trend_all else 0

    print(f" Toplam Test         : {len(results)} soru")
    print(f" {C.GREEN}Olgusal Dogruluk    : %{f_rate:.1f} ({len(factual_all)} soru){C.RESET}")
    print(f" {C.MAGENTA}3-Yil Trend Dogruluk: %{t_rate:.1f} ({len(trend_all)} soru){C.RESET}")
    print(f" {C.GREEN}Sifir Halusinasyon  : %{n_rate:.1f} ({len(negative_all)} soru){C.RESET}")
    print(f" {C.CYAN}PAL Motor Kullanimi : {len(pal_used_all)} soru / {len(results)} toplam{C.RESET}")
    print(f" Toplam Sure         : {total_lat:.2f}s (~{total_lat/60:.1f} dk)")

    problems = [r for r in results if r.verdict in ("FAIL", "HALLUCINATION!", "CAUTION")]
    if problems:
        print(f"\n {C.YELLOW}Dikkat Gerektiren Sorular ({len(problems)} adet):{C.RESET}")
        for r in problems:
            vc = _verdict_color(r.verdict)
            print(f"   Q{r.qid:02d} [{vc}{r.verdict}{C.RESET}] {_user_label(r.user_type)} | "
                  f"{_scen_label(r.scenario)} | {r.question[:70]}...")
    else:
        print(f"\n {C.GREEN}Tum testler %100 basariyla tamamlandi!{C.RESET}")
    print(f"\n{hr('=')}\n")


def _build_pal_data():
    carbon_df   = get_carbon_emissions_df()
    cr_summary  = get_carbon_removal_df()
    cr_tech     = get_carbon_removal_by_type_df()
    water_df    = get_water_metrics_df()
    water_proj  = get_water_replenishment_projects_df()
    energy_df   = get_energy_metrics_df()
    waste_df    = get_waste_metrics_df()
    zero_df     = get_zero_waste_certifications_df()

    pal_carbon = ("GHG Emissions (FY20/FY24/FY25 in mtCO2e):\n" + carbon_df.to_string(index=False))
    pal_cr     = ("Carbon Removal Summary:\n" + cr_summary.to_string(index=False)
                  + "\n\nCarbon Removal by Type:\n" + cr_tech.to_string(index=False))
    pal_water  = ("Water Metrics (million m3):\n" + water_df.to_string(index=False)
                  + "\n\nWater Projects by Region:\n" + water_proj.to_string(index=False))
    pal_energy = ("Energy Metrics (FY20/FY24/FY25 in MWh):\n" + energy_df.to_string(index=False))
    pal_waste  = ("Waste Metrics (metric tons):\n" + waste_df.to_string(index=False)
                  + "\n\nZero Waste Certifications:\n" + zero_df.to_string(index=False))

    PAL_MAP = {
        1: pal_waste,   7: pal_waste,    8: pal_energy,  10: pal_waste,
        11: pal_cr,     12: pal_cr,      13: pal_carbon, 14: pal_carbon,
        15: pal_water,  16: pal_water,   17: pal_waste,  18: pal_energy,
        19: pal_waste,  20: pal_cr,      21: pal_cr,     22: pal_water,
        23: pal_carbon, 24: pal_carbon,  25: pal_carbon, 26: pal_carbon,
        27: pal_cr,     28: pal_waste,   29: pal_energy, 30: pal_water,
        31: pal_cr,     32: pal_carbon,
        33: pal_cr,     34: pal_energy,  35: pal_carbon, 36: pal_water,
        37: pal_waste,  38: pal_energy,  39: pal_cr,     40: pal_energy,
        41: pal_water,  42: pal_carbon,
    }
    for q in QUESTIONS:
        if q.qid in PAL_MAP:
            q.pal_data = PAL_MAP[q.qid]


def main():
    _build_pal_data()
    target_qids = None
    target_diff = None
    target_user = None
    target_scen = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--only" and i + 1 < len(args):
            target_qids = [int(x.strip()) for x in args[i+1].split(",") if x.strip().isdigit()]
            i += 2
        elif args[i] == "--difficulty" and i + 1 < len(args):
            target_diff = args[i+1].strip().lower(); i += 2
        elif args[i] == "--user-type" and i + 1 < len(args):
            target_user = args[i+1].strip().lower(); i += 2
        elif args[i] == "--scenario" and i + 1 < len(args):
            target_scen = args[i+1].strip().lower(); i += 2
        else:
            i += 1

    questions_to_run = [
        q for q in QUESTIONS
        if (target_qids is None or q.qid in target_qids)
        and (target_diff is None or q.difficulty == target_diff)
        and (target_user is None or q.user_type == target_user)
        and (target_scen is None or q.scenario == target_scen)
    ]

    print(f"\n{hr('=')}")
    print(f" Microsoft EcoRAG Lab v2.1 -- Multi-Year Benchmark Suite")
    print(f" Model: {PHI_MODEL}  |  Embedding: {EMBEDDING_MODEL}")
    print(f" DB: {DB_PATH}  |  Raporlar: 2024 | 2025 | 2026 | 1050 Chunk")
    active_filters = []
    if target_diff: active_filters.append(f"zorluk='{target_diff}'")
    if target_user: active_filters.append(f"kullanici='{target_user}'")
    if target_scen: active_filters.append(f"senaryo='{target_scen}'")
    if target_qids: active_filters.append(f"ID={target_qids}")
    if active_filters: print(f" Filtre: {' | '.join(active_filters)}")
    print(f" Calistirilacak: {len(questions_to_run)} soru")
    print(f"{hr('=')}")

    results = []
    current_diff = ""
    cat_names = {
        "easy":     "Kategori 1: Kolay / Direct Factual",
        "medium":   "Kategori 2: Orta / Multi-Condition & Tabular",
        "hard":     "Kategori 3: Zor / Multi-Year Math & PAL",
        "trend":    "Kategori 4: Trend / 3-Yillik Capraz Karsilastirma",
        "negative": "Kategori 5: Alan Disi / Negatif Kontrol",
    }

    for q in questions_to_run:
        if q.difficulty != current_diff:
            current_diff = q.difficulty
            print(f"\n\n{hr('#')}")
            print(f" {C.BOLD}{cat_names.get(current_diff, current_diff)}{C.RESET}")
            print(f"{hr('#')}")

        print(f"\n{C.GREY}  -> Q{q.qid:02d} [{_user_label(q.user_type)} | {_scen_label(q.scenario)}] calisiyor...{C.RESET}",
              end="", flush=True)
        r = run_single(q)
        results.append(r)
        vc = _verdict_color(r.verdict)
        print(f"\r{C.GREY}  OK Q{q.qid:02d} tamamlandi ({r.latency_s:.1f}s) -- {vc}{r.verdict}{C.RESET}   ")
        print_result(q, r)

    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
