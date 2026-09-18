import os
import re
import time
import json
import requests
import app

# 15 Questions with Ground Truth & Key Facts to verify
QUESTIONS = [
    {
        "id": 1,
        "tier": "🟢 SEVİYE 1: KOLAY",
        "title": "Tek Kullanımlık Plastik Oranı",
        "question": "2026 Microsoft Çevresel Sürdürülebilirlik Raporuna göre, 2025/2026 takvim yılı sonunda birincil cihaz ambalajlarında ulaşılan tek kullanımlık plastik oranı yüzde kaçtır?",
        "expected": "%0.07 (veya 0.07%)",
        "key_terms": ["0.07", "%0.07", "0,07"]
    },
    {
        "id": 2,
        "tier": "🟢 SEVİYE 1: KOLAY",
        "title": "Kümülatif Su Yenileme Hacmi",
        "question": "Microsoft'un 2026 sürdürülebilirlik raporunda açıklanan kümülatif sözleşmeli su ikmal (replenishment) hacmi kaç milyon metreküptür?",
        "expected": "125.0 milyon m³",
        "key_terms": ["125", "125.0", "125,0", "milyon m"]
    },
    {
        "id": 3,
        "tier": "🟢 SEVİYE 1: KOLAY",
        "title": "Bulut Donanımı Yeniden Kullanım Oranı",
        "question": "Microsoft Circular Centers (Döngüsel Merkezler) aracılığıyla kullanım ömrü biten sunucu ve ağ donanımlarının yüzde kaçı yeniden kullanım veya geri dönüşüme kazandırılmıştır?",
        "expected": "%89.4 (veya %89)",
        "key_terms": ["89.4", "89,4", "%89", "89"]
    },
    {
        "id": 4,
        "tier": "🟢 SEVİYE 1: KOLAY",
        "title": "2030 Karbon Taahhüdü",
        "question": "Microsoft'un 2030 ve 2050 yılları için belirlediği kurumsal karbon hedefleri nelerdir?",
        "expected": "2030'a kadar Karbon Negatif, 2050'ye kadar tüm tarihsel emisyonları telafi etme",
        "key_terms": ["2030", "karbon negatif", "2050", "tarihsel"]
    },
    {
        "id": 5,
        "tier": "🟡 SEVİYE 2: ORTA",
        "title": "Sıfır Atık Denetim Standartları",
        "question": "Microsoft'un veri merkezlerinde ve ambalajlarında Sıfır Atık (Zero Waste) durumunu doğrulamak için hangi üçüncü taraf çerçeve ve standartlar kullanılmaktadır?",
        "expected": "UL Solutions Sıfır Atık Standardı (UL 2799 ECVP) ve TRUE Zero Waste",
        "key_terms": ["ul 2799", "ul solutions", "true", "underwriters"]
    },
    {
        "id": 6,
        "tier": "🟡 SEVİYE 2: ORTA",
        "title": "Akustik Kaçak Tespiti ve Şehirler",
        "question": "Microsoft, belediye su şebekelerinde yapay zeka tabanlı akustik sızıntı tespiti yapmak için hangi girişimle iş birliği yapmıştır ve hangi şehirlerde pilot projeler yürütülmüştür?",
        "expected": "FIDO Tech; Londra, Querétaro, Phoenix",
        "key_terms": ["fido", "londra", "london", "querétaro", "phoenix"]
    },
    {
        "id": 7,
        "tier": "🟡 SEVİYE 2: ORTA",
        "title": "Amsterdam Biyoçeşitlilik Projesi",
        "question": "2026 raporuna göre Amsterdam veri merkezi kampüsünde yerel biyoçeşitliliği ve ekosistemi desteklemek için hangi yöntemle mikro-ormanlar kurulmuştur?",
        "expected": "Miyawaki Metodu / Kentsel Mikro-Orman",
        "key_terms": ["miyawaki", "mikro-orman", "mikro orman", "amsterdam"]
    },
    {
        "id": 8,
        "tier": "🟡 SEVİYE 2: ORTA",
        "title": "Karbon Uzaklaştırma Teknolojisi Dağılımı",
        "question": "2025 raporundaki Karbon Tablosu 3'e göre sözleşmeye bağlanan karbon uzaklaştırma portföyünde en büyük paya sahip ilk iki teknoloji grubu hangileridir?",
        "expected": "Orman/Doğa tabanlı (~8.54M mtCO2e) ve Biyokütle/BECCS (~5.13M mtCO2e)",
        "key_terms": ["orman", "forest", "biyokütle", "biomass", "beccs", "dac"]
    },
    {
        "id": 9,
        "tier": "🔴 SEVİYE 3: ZOR (PAL)",
        "title": "Toplam Emisyon Artış Oranı (FY20 - FY25)",
        "question": "FY20 baz yılından FY25 yılına kadar Microsoft'un toplam sera gazı emisyonları (Scope 1 + Scope 2 Market + Scope 3) mutlak değer ve yüzde olarak ne kadar değişmiştir?",
        "expected": "+8,060,000 mtCO2e artış (+%61.71 büyüme)",
        "key_terms": ["61.", "61,", "8,060", "8.060", "21,121", "21.121", "13,061", "13.061"]
    },
    {
        "id": 10,
        "tier": "🔴 SEVİYE 3: ZOR (PAL)",
        "title": "Scope 3 Kategori 1 ve 2 Payları",
        "question": "FY25 yılında toplam 18,243,000 mtCO2e olan Scope 3 emisyonları içinde Kategori 1 (Satın Alınan Mal/Hizmetler) ve Kategori 2 (Sermaye Malları) toplamın yüzde kaçını oluşturur?",
        "expected": "Kategori 2: %49.58, Kategori 1: %28.11, Toplam İkisi: %77.69",
        "key_terms": ["49.", "49,", "28.", "28,", "77.", "77,", "9,044", "5,129"]
    },
    {
        "id": 11,
        "tier": "🔴 SEVİYE 3: ZOR (PAL)",
        "title": "Karbon Uzaklaştırma Portföy Büyüme Katı",
        "question": "2024 raporu (FY23) ile 2025 raporu (FY24) karşılaştırıldığında, sözleşmeli toplam karbon uzaklaştırma hacmi kaç katına çıkmıştır?",
        "expected": "4.37 kat (veya 21.9M vs 5.0M mtCO2e)",
        "key_terms": ["4.3", "4,3", "21.9", "21,9", "5.0", "5,0", "337"]
    },
    {
        "id": 12,
        "tier": "🔴 SEVİYE 3: ZOR (PAL)",
        "title": "Su Yenileme Hedef Başarı Oranı",
        "question": "FY25 yılında tamamlanan 7,800 milyon m³ su yenileme hacminin 9,500 milyon m³ hedef üzerinden gerçekleşme yüzdesi nedir ve FY24'e göre nasıl değişmiştir?",
        "expected": "%82.1 (FY24 %68.9'a kıyasla +13.2 puan)",
        "key_terms": ["82.1", "82,1", "%82", "68.9", "13.2"]
    },
    {
        "id": 13,
        "tier": "🟣 SEVİYE 4: UZMAN (Multi-RAG)",
        "title": "3 Yıllık Tek Kullanımlık Plastik Yolculuğu",
        "question": "2024, 2025 ve 2026 raporları boyunca Microsoft'un cihaz ambalajlarındaki tek kullanımlık plastik oranının düşüş eğilimini ve ulaştığı son noktayı özetleyin.",
        "expected": "%4.2'den %0.07 seviyesine düşüş, sıfıra yakın plastik eşiği",
        "key_terms": ["0.07", "0,07", "4.2", "4,2", "ambalaj", "plastik"]
    },
    {
        "id": 14,
        "tier": "🟣 SEVİYE 4: UZMAN (Multi-RAG)",
        "title": "Elektrik Tüketimi ve Yenilenebilir Enerji (PPA) Trendi",
        "question": "2024'ten 2026'ya kadar Microsoft'un elektrik tüketimi artarken %100 Karbonsuz Elektrik (CFE) eşleşme hedefi doğrultusunda temiz enerji sözleşmeleri nasıl gelişmiştir?",
        "expected": "Elektrik tüketimi 43.8M MWh'a çıkarken PPA portföyü 34 GW'a (veya 19.8-34 GW) ulaşmıştır",
        "key_terms": ["34", "gw", "ppa", "karbonsuz", "temiz enerji", "19.8"]
    },
    {
        "id": 15,
        "tier": "🟣 SEVİYE 4: UZMAN (Multi-RAG)",
        "title": "Sıfır Atık Veri Merkezi Sayısındaki Gelişim",
        "question": "2024 ve 2026 raporları arasında UL 2799 Sıfır Atık sertifikalı veri merkezlerinin sayısındaki ve kurtarılan atık miktarındaki değişimi açıklayın.",
        "expected": "10 veri merkezinden 14 veri merkezine artış; atık yönlendirme ~18.5 bin tondan 218 bin tona",
        "key_terms": ["10", "14", "veri merkezi", "2799", "atık", "ton"]
    }
]

def run_query_through_pipeline(query: str):
    target_lang = app.detect_query_language(query, default_lang="tr")
    intent = app.classify_esg_intent(query)

    route_type = "rag"
    ans = ""

    if intent == "out_of_domain":
        route_type = "safe_rejection"
        ans = app.TEXTS[target_lang]["not_found_msg"]
    elif intent == "carbon_commitments":
        route_type = "pal_carbon_commitments"
        ans = app.compute_carbon_commitments_summary(target_lang)
    elif intent == "carbon_trend_scope":
        route_type = "pal_carbon_trend"
        ans = app.compute_carbon_trend_summary(target_lang)
    elif intent == "carbon_removal":
        route_type = "pal_carbon_removal"
        ans = app.compute_carbon_removal_summary(target_lang)
    elif intent == "zero_waste_circularity":
        route_type = "pal_zero_waste"
        ans = app.compute_zero_waste_summary(target_lang)
    elif intent == "packaging_plastic":
        route_type = "pal_packaging"
        ans = app.compute_packaging_summary(target_lang)
    elif intent == "water_stewardship":
        route_type = "pal_water"
        ans = app.compute_water_summary(target_lang)
    elif intent == "mathematical_query":
        route_type = "pal_dynamic_alu"
        chunks, max_score = app.search_context_hybrid(query)
        context_chunks = [c["content"] for c in chunks]
        context_str = "\n\n".join(context_chunks)
        pot_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nExecutable Python code:"
        code_raw = app.query_foundry(app.POT_EXTRACTION_SYSTEM_PROMPT, pot_prompt, temperature=0.0)
        math_res = app.DynamicMathExecutor.execute_code_lines(code_raw)
        if math_res["success"] and math_res["environment"]:
            env = math_res["environment"]
            calc_lines = [
                f"• {k}: {v:.2f}" if isinstance(v, float) else f"• {k}: {v}"
                for k, v in env.items() if not k.startswith("_")
            ]
            calc_details = "Doğrulanmış Python Matematik Sonuçları:\n" + "\n".join(calc_lines)
            synth_prompt = (
                f"Doğrulanmış Kesin Matematik Verileri:\n{calc_details}\n\n"
                f"Soru: {query}\n\n"
                f"Doğrudan Türkçe Yanıt:"
            )
            ans = app.query_foundry(app.get_factual_synthesis_prompt(target_lang), synth_prompt)
        else:
            route_type = "rag"

    if not ans:
        # Standart RAG / Pydantic
        route_type = "hybrid_rag"
        search_query = query
        en_search_query = app.translate_query_to_en(query)
        if en_search_query and en_search_query != query:
            search_query = en_search_query

        chunks, max_score = app.search_context_hybrid(search_query)
        if not chunks or max_score < app.MIN_SCORE_FLOOR:
            chunks, max_score = app.search_context_hybrid(query)

        if not chunks or max_score < app.MIN_SCORE_FLOOR:
            ans = app.TEXTS[target_lang]["not_found_msg"]
        else:
            context_chunks = [c["content"] for c in chunks]
            context_str = "\n\n".join(context_chunks)[:3200]
            factual_en = app.query_foundry(
                "You are a Senior Sustainability Analyst. Summarize factual findings from the context to answer the question in 2-3 concise sentences with exact numbers and metrics. Retain all names and data. If not in context, output 'NOT_FOUND'.",
                f"Context:\n{context_str}\n\nQuestion: {search_query}\n\nFactual Summary:",
                temperature=0.15,
                max_tokens=280
            )

            if "NOT_FOUND" in factual_en and len(factual_en.strip()) < 25:
                ans = app.TEXTS[target_lang]["not_found_msg"]
            else:
                summary_system = (
                    "Sen uzman bir Sürdürülebilirlik Baş Danışmanısın. Aşağıda verilen doğrulanmış İngilizce rapor bulgularını kullanarak soruyu; 1-2 cümlelik akıcı bir Yönetici Özeti ve ardından önemli bulguları içeren son derece duru, kurumsal ve doğal bir Türkçe ile yanıtla. "
                    "Teknik verileri, birimleri ve şirket hedeflerini tam olarak koru. Soruyu baştan tekrar etme, çeviri kokan veya devrik cümlelerden kesinlikle kaçın. Tekrara düşme."
                )
                user_prompt_formatted = (
                    f"Soru: {query}\n\nDoğrulanmış Rapor Bulguları:\n{factual_en}\n\nDoğrudan Türkçe Yönetici Özeti ve Yanıt:"
                )
                ans = app.query_foundry(summary_system, user_prompt_formatted, temperature=0.25, max_tokens=320)

    return route_type, ans

def run_all():
    print("=" * 80)
    print("🚀 15 SORULUK KAPSAMLI ECO-RAG MODEL DEĞERLENDİRME VE SINAV PROTOKOLÜ")
    print("=" * 80)

    results = []
    total_passed = 0

    for item in QUESTIONS:
        q_id = item["id"]
        tier = item["tier"]
        title = item["title"]
        q_text = item["question"]
        expected = item["expected"]
        key_terms = item["key_terms"]

        print(f"\n[{q_id}/15] {tier} - {title}")
        print(f"  ❓ Soru: {q_text}")
        t0 = time.time()

        try:
            route, answer = run_query_through_pipeline(q_text)
            latency = time.time() - t0

            # Evaluate fact match
            ans_lower = answer.lower()
            matched_terms = [kt for kt in key_terms if kt.lower() in ans_lower]
            passed = len(matched_terms) >= 1

            status_str = "✅ BAŞARILI" if passed else "⚠️ KISMİ / İNCELENMELİ"
            if passed:
                total_passed += 1

            print(f"  ⚡ Rota: {route} | Gecikme: {latency:.2f}s | Durum: {status_str}")
            print(f"  🎯 Beklenen: {expected}")
            print(f"  🤖 Model Yanıtı:\n     {answer[:250]}...")

            results.append({
                "id": q_id,
                "tier": tier,
                "title": title,
                "question": q_text,
                "expected": expected,
                "route": route,
                "latency_sec": round(latency, 2),
                "passed": passed,
                "matched_terms": matched_terms,
                "answer": answer
            })

        except Exception as e:
            latency = time.time() - t0
            print(f"  ❌ HATA: {e}")
            results.append({
                "id": q_id,
                "tier": tier,
                "title": title,
                "question": q_text,
                "expected": expected,
                "route": "ERROR",
                "latency_sec": round(latency, 2),
                "passed": False,
                "matched_terms": [],
                "answer": f"Error: {str(e)}"
            })

    print("\n" + "=" * 80)
    print(f"📊 SINAV TAMAMLANDI: {total_passed}/15 BAŞARILI (%{total_passed/15*100:.1f})")
    print("=" * 80)

    with open("benchmark_15_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Sonuçlar benchmark_15_results.json dosyasına kaydedildi.")

if __name__ == "__main__":
    run_all()
