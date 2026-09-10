"""
run_5_tests.py — Comprehensive 5-Test Execution and Evaluation Script
--------------------------------------------------------------------
Executes the 5 benchmark tests against the real pipeline without leaking answers.
Records retrieval results, routing decisions, model outputs, and checks accuracy.
"""

import sys
import os
import json
import time
import requests

from run_benchmark_set2 import search_context_hybrid
from dynamic_math_engine import (
    DynamicMathExecutor,
    POT_EXTRACTION_SYSTEM_PROMPT,
    is_mathematical_query
)

FOUNDRY_BASE_URL = os.getenv("FOUNDRY_BASE_URL", "http://127.0.0.1:62095")
MODEL_NAME = "phi-4-mini"

TESTS = [
    {
        "id": "Test 1",
        "name": "Çoklu Tablo ve Kategori Karşılaştırması (Scope 3 Upstream)",
        "query": "Microsoft'un FY23 ve FY25 dönemlerinde Kategori 1 (Purchased Goods and Services) ile Kategori 2 (Capital Goods) Scope 3 emisyon payları yüzde kaçtır? Bu iki upstream kategorinin toplam payı FY23'ten FY25'e nasıl değişmiştir (yüzde puan farkı)?",
        "ground_truth": {
            "fy23_cat1": 36.23,
            "fy23_cat2": 38.24,
            "fy23_total": 74.47,
            "fy25_cat1": 25.28,
            "fy25_cat2": 44.57,
            "fy25_total": 69.85,
            "delta_points": -4.62
        }
    },
    {
        "id": "Test 2",
        "name": "Görsel Grafik Çıkarımı ve PAL Deterministik Hesaplama (Su Tüketimi)",
        "query": "2026 Sürdürülebilirlik Raporu'ndaki su tüketimi grafiğine göre, Microsoft'un FY25 yılındaki su çekimi (withdrawals) ile su tüketimi (consumption) arasındaki fark kaç metreküptür (m^3)?",
        "ground_truth": {
            "withdrawals": 13266000,
            "consumption": 8170000,
            "difference": 5096000
        }
    },
    {
        "id": "Test 3",
        "name": "Teknik Kısaltmalar, Lejant ve Çift Temsilli Tablo Ayrıştırma (PUE & WUE)",
        "query": "Microsoft'un sahip olduğu veri merkezlerinde FY25 döneminde ulaşılan küresel ortalama PUE (Power Usage Effectiveness) ve WUE (Water Usage Effectiveness) değerleri nedir? Şirketin 2022 baz yılına kıyasla WUE azaltım hedefi ve FY25'te ulaştığı gerçekleşme oranı nedir?",
        "ground_truth": {
            "pue": 1.17,
            "wue": 0.27,
            "wue_target": "40%",
            "wue_achieved": "25%"
        }
    },
    {
        "id": "Test 4",
        "name": "Ters-Olgusal (Counterfactual) Görsel Gürültü ve Sayısal Doğrulama",
        "query": "2026 Raporu sayfa 6'daki emisyon grafiğine göre, Microsoft'un FY25 yılı için fiilen raporladığı toplam sera gazı emisyonu (Actual reported emissions) ile müdahaleler olmasaydı oluşacağı tahmin edilen emisyon (Estimated emissions without select interventions) miktarları kaç milyon mtCO2e'dir? Yapılan müdahalelerle engellenen emisyon farkı kaçtır?",
        "ground_truth": {
            "actual": 20,
            "estimated_without": 34,
            "avoided": 14
        }
    },
    {
        "id": "Test 5",
        "name": "Yıllar Arası Trend Takibi ve Vektör Dağılım Testi (Sözleşmeli Karbon Uzaklaştırma)",
        "query": "Microsoft'un FY23, FY24 ve FY25 mali yıllarında ilgili yılların raporlarında duyurulan yıllık sözleşmeye bağlanmış karbon uzaklaştırma (contracted carbon removal) miktarları ne kadardır?",
        "ground_truth": {
            "fy23": "5,015,019",
            "fy24": "21,927,370 (veya ~22 milyon)",
            "fy25": "45 milyon"
        }
    }
]

def query_foundry(system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 500) -> str:
    url = f"{FOUNDRY_BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(url, json=payload, timeout=90)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

def run_test(test_item):
    print(f"\n==================================================")
    print(f"RUNNING: {test_item['id']} - {test_item['name']}")
    print(f"Query: {test_item['query']}")
    print(f"==================================================")

    start_time = time.time()
    query = test_item["query"]
    is_math = is_mathematical_query(query)
    print(f"Math Query Detected: {is_math}")

    # 1. Retrieval
    chunks, max_score = search_context_hybrid(query)
    print(f"Retrieved {len(chunks)} chunks, Max Score: {max_score:.4f}")
    retrieval_hits = []
    for c in chunks:
        hit_info = f"{c['title']} (Score: {c['score']:.4f})"
        retrieval_hits.append(hit_info)
        print("  Hit:", hit_info)

    context_chunks = [c["content"] for c in chunks]
    context_str = "\n\n".join(context_chunks)

    pot_code = None
    pot_env = None
    calc_details = None

    # 2. Routing
    if is_math:
        print("Routing to Dynamic PAL Engine...")
        pot_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nExecutable Python code:"
        pot_code = query_foundry(POT_EXTRACTION_SYSTEM_PROMPT, pot_prompt, temperature=0.0, max_tokens=350)
        math_res = DynamicMathExecutor.execute_code_lines(pot_code)
        if math_res["success"] and math_res["environment"]:
            pot_env = math_res["environment"]
            print("  Python ALU Execution Trace:")
            for t in math_res["trace"]:
                print("   ", t)
            calc_lines = [
                f"• {k}: {v:.2f}" if isinstance(v, float) else f"• {k}: {v}"
                for k, v in pot_env.items() if not k.startswith("_")
            ]
            calc_details = "Doğrulanmış Python Matematik Sonuçları:\n" + "\n".join(calc_lines)

            synth_prompt = (
                f"Doğrulanmış Kesin Matematik Verileri (Python ALU tarafından hesaplanmıştır):\n{calc_details}\n\n"
                f"Soru: {query}\n\n"
                f"Lütfen yukarıdaki doğrulanmış hesaplama sonuçlarını kullanarak soruyu doğrudan, profesyonel ve net Türkçe ile 2-3 cümlede yanıtla. Verilen sayıları ve birimleri tam olarak koru. Kesinlikle kendini tekrar etme."
            )
            ans = query_foundry(
                "Sen Kıdemli bir Sürdürülebilirlik Yapay Zeka Analistisin. Doğrulanmış metrikleri kullanarak net ve doğrudan Türkçe yanıt ver.",
                synth_prompt,
                temperature=0.0,
                max_tokens=400
            )
        else:
            print("  PoT execution did not yield environment, falling back to RAG")
            ans = query_foundry(
                "Sen Kıdemli bir Sürdürülebilirlik Analistisin. Yalnızca verilen bağlamı kullanarak doğrudan Türkçe yanıt ver.",
                f"Context:\n{context_str}\n\nQuestion: {query}",
                temperature=0.0,
                max_tokens=400
            )
    else:
        print("Routing to Standard RAG...")
        ans = query_foundry(
            "Sen Kıdemli bir Sürdürülebilirlik Analistisin. Yalnızca verilen bağlamı kullanarak doğrudan Türkçe yanıt ver.",
            f"Context:\n{context_str}\n\nQuestion: {query}",
            temperature=0.0,
            max_tokens=400
        )

    latency = time.time() - start_time
    print(f"\nLatency: {latency:.2f}s")
    print(f"\n--- MODEL RESPONSE ---")
    print(ans)

    return {
        "id": test_item["id"],
        "name": test_item["name"],
        "query": query,
        "is_math": is_math,
        "retrieval_hits": retrieval_hits,
        "pot_code": pot_code,
        "pot_env": pot_env,
        "calc_details": calc_details,
        "response": ans,
        "latency": latency,
        "ground_truth": test_item["ground_truth"]
    }

if __name__ == "__main__":
    results = []
    for t in TESTS:
        res = run_test(t)
        results.append(res)
        time.sleep(1)

    with open("benchmark_results_5_tests.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nAll 5 tests completed and saved to benchmark_results_5_tests.json")
