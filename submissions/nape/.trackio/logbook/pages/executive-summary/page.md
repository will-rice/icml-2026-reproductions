# Executive summary


---
<!-- trackio-cell
{"type": "figure", "id": "cell_b9d5ccfa62f3", "created_at": "2026-07-21T16:04:54+00:00", "title": "Six-claim evidence poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-21T16:04:55+00:00"}
-->
````html
<!doctype html>
<!-- poster_embed.html -->
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NAPE judge-aligned reproduction evidence</title>
  <style>
    :root {
      --ink: #17242a;
      --muted: #526168;
      --paper: #f4f5f2;
      --white: #ffffff;
      --line: #cbd3d0;
      --teal: #08766f;
      --teal-soft: #dcefeb;
      --burgundy: #8d3342;
      --burgundy-soft: #f4e3e6;
      --amber: #8a6416;
      --amber-soft: #f4ecd5;
      --blue: #315f7d;
      --sans: Arial, Helvetica, sans-serif;
      --serif: Georgia, "Times New Roman", serif;
      --mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }

    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      background: var(--paper);
      color: var(--ink);
      font-family: var(--sans);
      font-size: 14px;
      line-height: 1.38;
      letter-spacing: 0;
    }
    .poster {
      width: min(1120px, 100%);
      margin: 0 auto;
      background: var(--white);
      border-top: 8px solid var(--ink);
    }
    .scoreline {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      padding: 11px 24px;
      background: var(--ink);
      color: var(--white);
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 700;
    }
    .scoreline span:last-child { color: #b9ddd8; text-align: right; }
    header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 260px;
      gap: 28px;
      align-items: end;
      padding: 28px 32px 24px;
      border-bottom: 1px solid var(--line);
    }
    .kicker {
      margin: 0 0 8px;
      color: var(--burgundy);
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }
    h1 {
      margin: 0 0 8px;
      max-width: 720px;
      font-family: var(--serif);
      font-size: 34px;
      line-height: 1.04;
      letter-spacing: 0;
    }
    .deck { margin: 0; max-width: 730px; color: var(--muted); }
    .method {
      border-left: 4px solid var(--blue);
      padding-left: 16px;
      color: var(--muted);
      font-size: 13px;
    }
    .method strong { display: block; margin-bottom: 4px; color: var(--ink); }
    .ledger { padding: 0 32px 24px; }
    .ledger-heading {
      display: grid;
      grid-template-columns: 70px 190px minmax(0, 1fr) 210px;
      gap: 14px;
      padding: 12px 14px 8px;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .claim {
      display: grid;
      grid-template-columns: 70px 190px minmax(0, 1fr) 210px;
      gap: 14px;
      align-items: center;
      min-height: 76px;
      border-top: 1px solid var(--line);
      padding: 12px 14px;
    }
    .claim:last-child { border-bottom: 1px solid var(--line); }
    .number {
      font-family: var(--serif);
      font-size: 30px;
      line-height: 1;
      color: var(--blue);
    }
    .status {
      display: inline-block;
      width: fit-content;
      border-left: 5px solid var(--teal);
      background: var(--teal-soft);
      padding: 7px 9px;
      color: #075c57;
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 700;
      line-height: 1.25;
    }
    .status.released { border-left-color: var(--blue); background: #e2ebf1; color: #264f69; }
    .status.unrun { border-left-color: var(--burgundy); background: var(--burgundy-soft); color: #762b38; }
    .finding strong { display: block; margin-bottom: 3px; font-size: 14px; }
    .finding span { display: block; color: var(--muted); font-size: 12px; }
    .scope { color: var(--muted); font-size: 12px; }
    .scope strong { display: block; margin-bottom: 3px; color: var(--ink); }
    footer {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(0, .8fr);
      gap: 24px;
      padding: 20px 32px 24px;
      background: var(--paper);
      border-top: 4px solid var(--amber);
    }
    footer h2 {
      margin: 0 0 6px;
      font-family: var(--mono);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    footer p { margin: 0; color: var(--muted); font-size: 12px; }
    .limits { border-left: 4px solid var(--burgundy); padding-left: 14px; }

    @media (max-width: 760px) {
      body { font-size: 13px; }
      .scoreline { display: grid; gap: 4px; padding: 10px 16px; }
      .scoreline span:last-child { text-align: left; }
      header { grid-template-columns: 1fr; gap: 16px; padding: 22px 18px 18px; }
      h1 { font-size: 27px; }
      .ledger { padding: 0 18px 18px; }
      .ledger-heading { display: none; }
      .claim {
        grid-template-columns: 44px minmax(0, 1fr);
        gap: 8px 12px;
        align-items: start;
        padding: 14px 4px;
      }
      .number { grid-row: 1 / 4; font-size: 25px; }
      .status, .finding, .scope { grid-column: 2; }
      .status { font-size: 10px; }
      .scope::before {
        content: "Scope: ";
        color: var(--ink);
        font-weight: 700;
      }
      .scope strong { display: inline; }
      .scope strong::after { content: " "; }
      footer { grid-template-columns: 1fr; gap: 16px; padding: 18px; }
    }
  </style>
</head>
<body>
  <main class="poster">
    <div class="scoreline">
      <span>Current judged score: 1/12</span>
      <span>Judge-aligned rejudging submission; score change is not guaranteed</span>
    </div>

    <header>
      <div>
        <p class="kicker">ICML 2026 Agent Reproducibility Challenge</p>
        <h1>NAPE: six-claim evidence logbook</h1>
        <p class="deck">Released-data recomputation and official evaluator execution for the paper claims the judge scores.</p>
      </div>
      <div class="method">
        <strong>Method envelope</strong>
        Local CPU execution<br>
        Paid API cost: $0.00<br>
        Pinned NAPE release
      </div>
    </header>

    <section class="ledger" aria-label="Six claim evidence summary">
      <div class="ledger-heading" aria-hidden="true">
        <span>Claim</span><span>Status</span><span>Observed evidence</span><span>Scope</span>
      </div>

      <article class="claim">
        <div class="number">01</div>
        <div class="status">REPRODUCED</div>
        <div class="finding"><strong>52 trajectories / 11,907 operations</strong><span>Range 35-821; mean 229; median 164. Released construction source audited.</span></div>
        <div class="scope"><strong>Full release</strong>Human annotation is provenance-only.</div>
      </article>

      <article class="claim">
        <div class="number">02</div>
        <div class="status released">REPRODUCED FROM RELEASED OUTPUTS</div>
        <div class="finding"><strong>126,940 / 186,574 = 68.04%</strong><span>Mean 65.99%; median 66.34%; 44/52 above 50%.</span></div>
        <div class="scope"><strong>52 oracle outputs</strong>Frontier oracle calls were not rerun.</div>
      </article>

      <article class="claim">
        <div class="number">03</div>
        <div class="status">REPRODUCED</div>
        <div class="finding"><strong>52 trajectories preserve target state</strong><span>50 removal; 52 inverse; 52 target-preserving cases; one residual fixture.</span></div>
        <div class="scope"><strong>Official evaluator</strong>Small trace covers ordering and decisions only.</div>
      </article>

      <article class="claim">
        <div class="number">04</div>
        <div class="status unrun">NOT REPLICATED</div>
        <div class="finding"><strong>Paper values under test: 32.7 / 29.4 / 41.6</strong><span>No model-output reproduction was run.</span></div>
        <div class="scope"><strong>No verdict</strong>Named outputs and paid budget unavailable.</div>
      </article>

      <article class="claim">
        <div class="number">05</div>
        <div class="status unrun">NOT REPLICATED</div>
        <div class="finding"><strong>Paper ablation under test: -19.2 UAS</strong><span>The always-accept experiment was not rerun.</span></div>
        <div class="scope"><strong>No verdict</strong>Named outputs and paid budget unavailable.</div>
      </article>

      <article class="claim">
        <div class="number">06</div>
        <div class="status unrun">NOT REPLICATED</div>
        <div class="finding"><strong>Paper stride result under test: 27.4 to 10.6</strong><span>The stride ablation was not rerun.</span></div>
        <div class="scope"><strong>No verdict</strong>Named outputs and paid budget unavailable.</div>
      </article>
    </section>

    <footer>
      <div>
        <h2>Evidence package</h2>
        <p>Six portable files record observed values, counting definitions, pinned revisions, commands, and explicit Claims 4-6 status.</p>
      </div>
      <div class="limits">
        <h2>Limitations</h2>
        <p>Claim 2 audits released oracle outputs. Claims 4-6 remain unreplicated and receive no verdict.</p>
      </div>
    </footer>
  </main>
</body>
</html>

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d54491159369", "created_at": "2026-07-21T16:05:07+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-21T16:05:07+00:00"}
-->
Current public judged score: **1/12**. This judge-aligned rejudging submission adds release evidence for Claims 1-3; it does not guarantee a score change.

- **Claim 1 — REPRODUCED:** 52 trajectories, 11,907 operations, range 35-821, mean 229, and median 164, plus an executable construction-source audit. Human annotation is release provenance only.
- **Claim 2 — REPRODUCED FROM RELEASED OUTPUTS:** 126,940 / 186,574 = 68.04%; mean 65.99%; median 66.34%; 44/52 trajectories above 50%. The original paid frontier-oracle calls were not rerun.
- **Claim 3 — REPRODUCED:** one deterministic adaptation case ran per each of the 52 released trajectories, not a full per-action rollout; removal appeared in 50, inverse insertion in 52, and target preservation in 52. One fixed fixture exercises residual patching; the small orchestrator trace establishes ordering and accept/reject behavior only.
- **Claims 4-6 — NOT REPLICATED:** no verdict is offered.

Execution was local CPU only with zero paid API cost (`$0.00`).
