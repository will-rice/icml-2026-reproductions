# TIC-VLA Empirical Measurements and Benchmark Surfaces

Detailed numerical metrics and claim surface audits from the TIC-VLA reproduction bundle:

| Metric / Evaluation Surface | Value / Status | Benchmark Target | Hardware Platform |
|---|---|---|---|
| Latency-Aware Success Rate | 47.06% | Table 2 | DynaNav Simulator |
| Waypoint Baseline Success | 16.47% | Table 2 | DynaNav Simulator |
| RTX 4060 Real-World Success | 85.00% | Table 3 | Physical Robot (RTX 4060) |
| RTX 4060 Action Latency | 85.73 ms | Table 3 | Physical Robot (RTX 4060) |
| RTX 4060 VLM Reasoning Latency | 3430.73 ms | Table 3 | Physical Robot (RTX 4060) |
| Jetson Orin NX Success Rate | 75.00% | Table 3 | Physical Robot (Orin NX) |
| Jetson Orin NX VLM Latency | 4831.73 ms | Table 3 | Physical Robot (Orin NX) |

## Code-Level Interface Probe Results

1. ActionExpert Tensor Shape: [2, 5, 2]
2. Robot State Fixture Input: [2.0, -1.0, 2.0]
3. KV Cache Depth: 4 attention layers
4. Memory Footprint Delta: 0.0 MB (synthetic fixture probe)
5. DynaNav Scene Count: 4 scenes, 85 total evaluation episodes
6. Max Dynamic Agents: 200 agents per scene configuration

## Summary Statistics

- Total Claims Audited: 5
- Total Scenes Audited: 4
- Total Benchmark Episodes: 85
- Estimated Paid API Cost: USD 0.00
