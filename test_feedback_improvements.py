#!/usr/bin/env python3
"""
Test Suite for User Feedback Improvements:
1. Streamlit config & credentials verification (no email prompt, gatherUsageStats=false)
2. Pure metadata-driven Year-Stratified retrieval & year filtering
3. ESG Impact & Corporate Action Card resolver (get_esg_impact_insight)
4. Humanized & readable compute summaries with 100% mathematical integrity
"""

import os
import sys
import unittest
import toml

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from app import (
    search_context_hybrid,
    get_esg_impact_insight,
    compute_carbon_trend_summary,
    compute_carbon_removal_summary,
    compute_zero_waste_summary,
    compute_packaging_summary,
    compute_water_summary
)

class TestFeedbackImprovements(unittest.TestCase):

    def test_01_streamlit_config_and_credentials(self):
        """1. Streamlit e-posta uyarısı ve telemetri engelleme doğrulaması."""
        config_path = os.path.join(REPO_DIR, ".streamlit", "config.toml")
        cred_path = os.path.join(REPO_DIR, ".streamlit", "credentials.toml")

        self.assertTrue(os.path.exists(config_path), ".streamlit/config.toml mevcut olmalı")
        self.assertTrue(os.path.exists(cred_path), ".streamlit/credentials.toml mevcut olmalı")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = toml.load(f)
        self.assertFalse(cfg.get("browser", {}).get("gatherUsageStats", True))
        self.assertTrue(cfg.get("server", {}).get("headless", False))

        with open(cred_path, "r", encoding="utf-8") as f:
            cred = toml.load(f)
        self.assertEqual(cred.get("general", {}).get("email"), "")
        print("  ✓ [TEST 1] Streamlit e-posta ve telemetri yapılandırması başarıyla doğrulandı.")

    def test_02_metadata_year_stratified_retrieval(self):
        """2. Year-Stratified aramanın yapısal metadata ile çalışması."""
        # A. Manuel filtreleme testi (2024)
        chunks_2024, score_2024 = search_context_hybrid("Scope 1 emissions", year_filter="2024")
        self.assertTrue(len(chunks_2024) > 0)
        for c in chunks_2024:
            self.assertEqual(str(c["year"]).strip(), "2024", "Tüm chunklar 2024 metadata'sına sahip olmalı")

        # B. Manuel filtreleme testi (2026)
        chunks_2026, score_2026 = search_context_hybrid("Scope 3 Capital Goods", year_filter="2026")
        self.assertTrue(len(chunks_2026) > 0)
        for c in chunks_2026:
            self.assertEqual(str(c["year"]).strip(), "2026", "Tüm chunklar 2026 metadata'sına sahip olmalı")

        # C. Çok yıllı otomatik katmanlama testi
        multi_q = "Compare Scope 1 emissions across 2024, 2025, and 2026 sustainability reports"
        chunks_multi, score_multi = search_context_hybrid(multi_q, year_filter=None)
        years_found = {str(c["year"]).strip() for c in chunks_multi}
        self.assertTrue(len(years_found) >= 2, f"Çok yıllı katmanlama en az 2 farklı rapor yılı içermeli, bulunan: {years_found}")
        print("  ✓ [TEST 2] Metadata tabanlı Year-Stratified arama ve filtreleme başarıyla doğrulandı.")

    def test_03_esg_impact_insight_card(self):
        """3. Sürdürülebilirlik Uyum & Aksiyon Özeti kartı üretimi."""
        # Karbon sorgusu
        c_insight = get_esg_impact_insight("Microsoft'un karbon ayak izi ve emisyonları nedir?", "Scope 1 ve Scope 2 emisyonları artmıştır.", "tr")
        self.assertIsNotNone(c_insight)
        self.assertIn("Karbon", c_insight["pillar"])
        self.assertIn("2030 Karbon Negatif", c_insight["target"])
        self.assertIn("21.9M mtCO2e", c_insight["actions"])

        # Su sorgusu
        w_insight = get_esg_impact_insight("Microsoft su yenileme hedefi nedir?", "125M m3 kümülatif yenileme yapılmıştır.", "tr")
        self.assertIsNotNone(w_insight)
        self.assertIn("Su", w_insight["pillar"])
        self.assertIn("2030 Su Pozitif", w_insight["target"])

        # Sıfır atık sorgusu
        z_insight = get_esg_impact_insight("What are the zero waste certifications?", "14 datacenters certified under UL 2799.", "en")
        self.assertIsNotNone(z_insight)
        self.assertIn("Zero Waste", z_insight["pillar"])
        self.assertIn("2030 Zero Waste", z_insight["target"])

        # Güvenli ret / Bilinmeyen soru
        rej_insight = get_esg_impact_insight("Bilinmeyen soru", "Microsoft Çevresel Sürdürülebilirlik raporlarında bu konuyla ilgili bilgi bulunmamaktadır.", "tr")
        self.assertIsNone(rej_insight, "Reddedilen sorgular için insight kartı üretilmemeli")
        print("  ✓ [TEST 3] ESG Uyum ve Aksiyon Kartı üretimi tüm sütunlar için başarıyla doğrulandı.")

    def test_04_humanized_compute_summaries(self):
        """4. Humanize edilmiş PAL özetlerinin biçim ve sayısal kesinlik kontrolü."""
        c_trend = compute_carbon_trend_summary("tr")
        self.assertIn("📊 Sera Gazı Emisyon Karşılaştırması", c_trend)
        self.assertIn("💡 Temel Stratejik Aksiyonlar", c_trend)
        self.assertIn("21,121,000", c_trend)
        self.assertIn("170,887", c_trend)  # FY25 Scope 1
        self.assertIn("2,707,428", c_trend) # FY25 Scope 2 market

        c_rem = compute_carbon_removal_summary("tr")
        self.assertIn("21,927,370 mtCO2e", c_rem)
        self.assertIn("4.37 kat artış", c_rem)

        w_sum = compute_water_summary("tr")
        self.assertIn("125.0 milyon m³", w_sum)
        self.assertIn("%82.1", w_sum)

        p_sum = compute_packaging_summary("tr")
        self.assertIn("%0.07", p_sum)

        z_sum = compute_zero_waste_summary("tr")
        self.assertIn("18,537 metrik ton", z_sum)
        print("  ✓ [TEST 4] Humanize edilmiş PAL özetleri ve kesin matematik verileri doğrulandı.")


if __name__ == "__main__":
    unittest.main()
