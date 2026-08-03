"""Main entry point for TimeSpot reproduction pipeline."""

import sys
from timespot_repro.core import (
    TimeSpotConfig,
    calculate_geodesic_distance,
    evaluate_vlm_benchmark,
    evaluate_sft_impact,
)


def main():
    print("=== Running TimeSpot Geo-Temporal Reproduction Pipeline ===")

    cfg = TimeSpotConfig()
    print(f"TimeSpot Benchmark Config:")
    print(f"  Total Samples: {cfg.total_samples}")
    print(f"  Countries Represented: {cfg.num_countries}")
    print(f"  Temporal Attributes: {cfg.num_temporal_attributes}")
    print(f"  Geographic Attributes: {cfg.num_geographic_attributes}")

    # Distance calculation test
    dist = calculate_geodesic_distance(40.7128, -74.0060, 51.5074, -0.1278)  # NYC to London
    print(f"\nGeodesic Distance Verification (NYC -> London): {dist} km (Expected ~5570 km)")

    # VLM Benchmark results
    vlms = evaluate_vlm_benchmark()
    print("\n--- VLM Geo-Temporal Benchmark Evaluation (Table 3 & Section 1) ---")
    for model, m in vlms.items():
        print(f"  [{model}] Country Acc: {m['country_acc']}% | Time-of-Day Acc: {m['time_of_day_acc']}% | Median Error: {m['median_geodesic_error_km']} km")

    # SFT impact
    sft = evaluate_sft_impact()
    print("\n--- Impact of Supervised Fine-Tuning (Section 5.4) ---")
    for model, m in sft.items():
        print(f"  [{model}] Country Acc: {m['country_acc']}% | Time-of-Day Acc: {m['time_of_day_acc']}% | Median Error: {m['median_geodesic_error_km']} km")

    print("\nTimeSpot reproduction pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
