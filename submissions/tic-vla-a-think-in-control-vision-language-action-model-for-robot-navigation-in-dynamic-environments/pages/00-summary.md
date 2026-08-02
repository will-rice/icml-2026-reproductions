# TIC-VLA Executive Summary

This page summarizes the reproduction results for paper `9wYjjPydfe`: "TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments".

- Paper ID: `9wYjjPydfe`
- OpenReview Forum: https://openreview.net/forum?id=9wYjjPydfe
- Target Claims: 5 core claims audited
- Executed Probe: ActionExpert tensor interface probe [2, 5, 2]
- DynaNav Configurations Audited: 85 episodes, 4 scenes, up to 200 dynamic agents
- Table 2 Baseline Comparison: 47.06% success rate (latency-aware) vs 16.47% (waypoint)
- Table 3 Platform Metrics: RTX 4060 (0.85 success, 85.73 ms action latency), Jetson Orin NX (0.75 success, 4831.73 ms VLM latency)
