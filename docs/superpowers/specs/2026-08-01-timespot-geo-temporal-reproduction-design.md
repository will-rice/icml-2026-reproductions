# Reproduction Design: TimeSpot: Benchmarking Geo-Temporal Understanding in Vision–Language Models in Real-World Settings

**Paper ID:** XQlUqVCHJd
**Slug:** timespot-benchmarking-geo-temporal-understanding-in-vision-language-models-in-real-world-settings
**ArXiv:** 2603.06687
**Upstream Revision:** arxiv:2603.06687

## 1. Overview and Core Claims

This paper presents **TimeSpot**, a joint geo-temporal benchmark for evaluating Vision-Language Models (VLMs) on ground-level photos across 4 temporal and 5 geographic attributes.

### Target Claims:
1. **Benchmark Definition:** Joint geo-temporal benchmark requiring structured prediction of 4 temporal (season, daylight, month, time-of-day) and 5 geographic (continent, country, climate, environment, lat/lon) attributes (Section 3).
2. **Dataset Composition:** 1,455 ground-level photos from 80 countries with broad season, daylight, month, climate, and environment coverage (Table 2).
3. **Scoring Axes:** Includes hemisphere sanity, hard OOD, geo-temporal fusion, schema, calibration, and GPS/OSM-style scoring axes (Table 1).
4. **Temporal Discrepancy:** Evaluated VLMs show substantially weaker temporal understanding than coarse geolocation (top time-of-day accuracy far below country accuracy) (Table 3).
5. **VLM Performance & Geodesic Error:** Strongest VLMs achieve 77.59% country accuracy while incurring median geodesic error of 892.54 km and low time-of-day accuracy (Section 1).
6. **SFT Limitations:** Supervised fine-tuning improves TimeSpot performance but remains insufficient for robust physically grounded geo-temporal reasoning (Section 5.4).

## 2. Reproduction Strategy and Test Harness Design

- **Project Path:** `submissions/timespot-benchmarking-geo-temporal-understanding-in-vision-language-models-in-real-world-settings`.
- **Validation Pipeline:**
  1. `app.py`: Interactive Gradio web UI demonstrating geo-temporal attribute extraction, geodesic distance calculation, and VLM evaluations.
  2. `pages/report.md`: Technical report for judge evaluation (> 1,100 chars).
  3. `evidence.json`: Execution evidence verifying all 6 claims.
  4. `tests/`: Automated pytest suite.
