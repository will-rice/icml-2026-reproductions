"""Interactive Gradio Web UI for TimeSpot Reproduction."""

import gradio as gr
from timespot_repro.core import (
    TimeSpotConfig,
    calculate_geodesic_distance,
    evaluate_vlm_benchmark,
    evaluate_sft_impact,
)


def run_distance_calc(lat1, lon1, lat2, lon2):
    dist = calculate_geodesic_distance(float(lat1), float(lon1), float(lat2), float(lon2))
    return f"**Calculated Geodesic Error**: `{dist} km`"


def load_vlm_benchmark():
    vlms = evaluate_vlm_benchmark()
    table = []
    for model, m in vlms.items():
        table.append([
            model,
            f"{m['country_acc']}%",
            f"{m['time_of_day_acc']}%",
            f"{m['season_acc']}%",
            f"{m['hemisphere_sanity']}%",
            f"{m['median_geodesic_error_km']} km",
        ])
    return table


def load_sft_benchmark():
    sft = evaluate_sft_impact()
    table = []
    for model, m in sft.items():
        table.append([
            model,
            f"{m['country_acc']}%",
            f"{m['time_of_day_acc']}%",
            f"{m['season_acc']}%",
            f"{m['median_geodesic_error_km']} km",
        ])
    return table


with gr.Blocks(title="TimeSpot Geo-Temporal Benchmark") as demo:
    gr.Markdown("# TimeSpot: Benchmarking Geo-Temporal Understanding in Vision–Language Models")
    gr.Markdown("Reproduction suite for ICML 2026 Paper **XQlUqVCHJd** (arXiv:2603.06687)")

    with gr.Tab("VLM Geo-Temporal Leaderboard (Table 3)"):
        gr.Markdown("### Evaluation across 1,455 ground-level photos from 80 countries")
        vlm_btn = gr.Button("Load Benchmark Leaderboard")
        vlm_table = gr.Dataframe(
            headers=["Model", "Country Acc %", "Time-of-Day Acc %", "Season Acc %", "Hemisphere Sanity %", "Median Geodesic Error"],
            datatype=["str", "str", "str", "str", "str", "str"],
        )
        vlm_btn.click(load_vlm_benchmark, outputs=[vlm_table])

    with gr.Tab("Geodesic Error Calculator"):
        gr.Markdown("### Haversine Great-Circle Distance Evaluation")
        with gr.Row():
            lat1_in = gr.Number(label="Predicted Lat", value=40.7128)
            lon1_in = gr.Number(label="Predicted Lon", value=-74.0060)
            lat2_in = gr.Number(label="Ground Truth Lat", value=51.5074)
            lon2_in = gr.Number(label="Ground Truth Lon", value=-0.1278)
        calc_btn = gr.Button("Calculate Geodesic Distance")
        dist_out = gr.Markdown()
        calc_btn.click(run_distance_calc, inputs=[lat1_in, lon1_in, lat2_in, lon2_in], outputs=[dist_out])

    with gr.Tab("SFT Fine-Tuning Impact (Section 5.4)"):
        gr.Markdown("### Zero-Shot vs Supervised Fine-Tuning Performance")
        sft_btn = gr.Button("Load SFT Comparison")
        sft_table = gr.Dataframe(
            headers=["Configuration", "Country Acc %", "Time-of-Day Acc %", "Season Acc %", "Median Geodesic Error"],
            datatype=["str", "str", "str", "str", "str"],
        )
        sft_btn.click(load_sft_benchmark, outputs=[sft_table])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
