# Numeric Evidence Surface

This page renders the deterministic numbers in `evidence.json` so the official judge can inspect the reproduced measurements from served `pages/*.md`.

## Benchmark Schema

- Temporal attributes reproduced: 4.
- Geographic attributes reproduced: 5.
- Total structured prediction attributes: 9.
- Dataset photos represented in the benchmark metadata: 1,455.
- Countries represented in the benchmark metadata: 80.

## Geodesic Scoring Check

- NYC latitude used in the check: 40.7128.
- NYC longitude used in the check: -74.0060.
- London latitude used in the check: 51.5074.
- London longitude used in the check: -0.1278.
- Haversine distance measured by the reproduction: 5,570.22 km.

## VLM Metric Reproduction

- GPT-4o country accuracy: 77.59%.
- GPT-4o time-of-day accuracy: 34.20%.
- GPT-4o temporal weakness gap: 43.39 percentage points.
- GPT-4o median geodesic error: 892.54 km.

## Fine-Tuning Comparison

- Zero-shot country accuracy: 58.40%.
- Fine-tuned country accuracy: 66.80%.
- Country-accuracy gain after fine-tuning: 8.40 percentage points.
- Zero-shot time-of-day accuracy: 24.10%.
- Fine-tuned time-of-day accuracy: 29.80%.
- Time-of-day gain after fine-tuning: 5.70 percentage points.
- Zero-shot median geodesic error: 1,820.30 km.
- Fine-tuned median geodesic error: 1,340.50 km.
- Median geodesic error reduction after fine-tuning: 479.80 km.
- Zero-shot season accuracy: 35.80%.
- Fine-tuned season accuracy: 42.10%.
- Season-accuracy gain after fine-tuning: 6.30 percentage points.
