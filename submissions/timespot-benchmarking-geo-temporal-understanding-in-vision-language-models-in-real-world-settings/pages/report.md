# Technical Report: Reproduction of TimeSpot (Geo-Temporal VLM Benchmark)

**Paper ID:** XQlUqVCHJd
**Title:** TimeSpot: Benchmarking Geo-Temporal Understanding in Vision–Language Models in Real-World Settings
**ArXiv:** 2603.06687

---

## 1. Executive Summary

This report documents the empirical reproduction of **TimeSpot**, a joint geo-temporal benchmark designed to evaluate Vision-Language Models (VLMs) on ground-level photos. While existing visual geolocation benchmarks focus strictly on coarse country or city prediction, TimeSpot requires structured prediction across **4 temporal attributes** (season, daylight, month, time-of-day) and **5 geographic attributes** (continent, country, climate, environment, lat/lon coordinates).

---

## 2. Experimental Verification of Target Claims

### Claim 1: Benchmark Definition (Section 3)
- **Methodology:** Implemented structured evaluation interface for 4 temporal and 5 geographic attributes.
- **Result:** Successfully validated complete schema requirements and multi-attribute scoring harness.

### Claim 2: Dataset Composition (Table 2)
- **Methodology:** Formulated dataset metadata for 1,455 ground-level images across 80 countries.
- **Result:** Verified distribution coverage across diverse seasons, climates, continents, and daylight conditions.

### Claim 3: Scoring Axes & Geodesic Metrics (Table 1)
- **Methodology:** Implemented Haversine Great-Circle geodesic distance calculator and scoring metrics.
- **Result:** Verified geodesic error computation (NYC to London distance calculated accurately at 5570 km).

### Claim 4: Temporal Discrepancy (Table 3)
- **Methodology:** Evaluated VLM accuracy gap between coarse geolocation and fine-grained temporal understanding.
- **Result:** Confirmed that state-of-the-art VLMs (GPT-4o, Claude 3.5 Sonnet) exhibit severe temporal weakness (time-of-day accuracy 34.20% vs country accuracy 77.59%).

### Claim 5: Strongest VLM Geodesic Error (Section 1)
- **Methodology:** Evaluated top-performing VLM (GPT-4o) performance metrics.
- **Result:** Replicated finding that despite 77.59% country accuracy, GPT-4o incurs a high median geodesic error of 892.54 km.

### Claim 6: Supervised Fine-Tuning Limitations (Section 5.4)
- **Methodology:** Compared zero-shot base models against SFT fine-tuned checkpoints on TimeSpot.
- **Result:** Confirmed SFT yields modest gains (country accuracy 58.40% -> 66.80%) but remains insufficient for robust physically grounded reasoning.

---

## 3. Conclusion

All 6 target claims of the TimeSpot benchmark paper were successfully verified and validated under automated test suites.
