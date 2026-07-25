# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2ed1391a3011", "created_at": "2026-07-25T15:19:22+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-25T15:19:22+00:00"}
-->
## Outcome

This CPU-only **released-artifact audit** separates paper-reported context from computed evidence. The released selector contains 14 target datasets, but released dataset classes expose 8 task types rather than the paper context of 10 canonical paradigms, so that claim is **partial**. Exact pinned preprocessing methods are repeat-identical on two synthetic EEG configurations, and all three released harness strategies have source-backed wiring plus finite CPU smoke steps.

**Unavailable:** GPU leaderboard performance and representation analysis require gated raw datasets, checkpoints, and GPU runs.

Sources: [paper](https://arxiv.org/abs/2508.17742v3) · [pinned upstream](https://github.com/xw1216/EEG-FM-Bench/tree/325398d7d057ecc1216fb3510d70c16eb60337cc)


---
<!-- trackio-cell
{"type": "figure", "id": "cell_826f7eb873e1", "created_at": "2026-07-25T15:19:22+00:00", "title": "Evidence poster", "pinned": true, "pinned_at": "2026-07-25T15:19:23+00:00"}
-->
````html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>EEG-FM-Bench released-artifact audit</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #07111f; color: #eaf2ff; }
    main { max-width: 1100px; margin: auto; padding: 36px; }
    h1 { font-size: clamp(2rem, 5vw, 4rem); margin: 0; }
    .kicker { color: #6ee7c7; letter-spacing: .12em; text-transform: uppercase; }
    .scope { color: #b8c7dc; max-width: 75ch; font-size: 1.1rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(240px,1fr)); gap: 16px; margin: 28px 0; }
    article { border: 1px solid #29415e; border-radius: 16px; padding: 20px; background: #0c1b2e; }
    .status { color: #6ee7c7; font-weight: 700; }
    .partial { color: #ffd166; }
    strong { font-size: 1.35rem; display: block; margin: 8px 0; }
    footer { border-top: 1px solid #29415e; padding-top: 18px; color: #b8c7dc; }
    code { color: #9cc2ff; overflow-wrap: anywhere; }
  </style>
</head>
<body>
<main>
  <p class="kicker">ICML 2026 · CPU evidence · USD 0</p>
  <h1>EEG-FM-Bench released-artifact audit</h1>
  <p class="scope">Pinned source census, exact released preprocessing method
  execution on synthetic EEG, and deterministic three-strategy harness smoke.
  This is not a leaderboard reproduction.</p>
  <section class="grid">
    <article>
      <span class="status partial">Partial</span>
      <strong>14 datasets</strong>
      <p>Found in released selectors and builders, spanning
      <b>8 release-defined task types</b>; paper context: 10 canonical paradigms.</p>
    </article>
    <article>
      <span class="status">Verified locally</span>
      <strong>Deterministic preprocessing</strong>
      <p>Two dataset configurations, exact pinned method bodies, repeat-identical
      channel standardization, 256 Hz resampling, and windowing.</p>
    </article>
    <article>
      <span class="status">Verified locally</span>
      <strong>Three harness strategies</strong>
      <p>Frozen single-task, full single-task, and full multi-task paths audited
      with finite deterministic CPU steps.</p>
    </article>
  </section>
  <footer>
    <p><b>GPU leaderboard: unavailable.</b> Gated raw EEG data, foundation-model
    checkpoints, and GPU runs are outside scope.</p>
    <p>Repo <code>325398d7d057ecc1216fb3510d70c16eb60337cc</code> · paper
    <code>arxiv:2508.17742v3</code> · hashes and rerun commands in the evidence bundle.</p>
  </footer>
</main>
</body>
</html>

````
