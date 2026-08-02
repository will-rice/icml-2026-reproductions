# Repository and Hub Audits

- Paper: `vCc2NAe0OS`
- Attempt: `9fb6688a-2be9-4436-98bf-2525fb2a8df1`

## Hub Assets & Revisions

| Asset | Revision / Value | Status |
| --- | --- | --- |
| Audio Archive | `7ed0f3ac1e166b2e1455cbff550defc618bab25d` | 5,600 TARs (52.99 GB sampled) |
| AudioSep Checkpoint | `113d2e4399a4f19b6a0d567bbde38f2fe1b11794` | Present |
| FlowSep Checkpoint | `7af336090e0c155b1850de37ab310cf36c3e390e` | Present |
| Metadata Dataset | `32b57157653ac31a7b525dbfa57aa03aa4d8e3fd` | Present (1.23 GB parquets) |

## Pipeline & Ontology Audits

- **Purification Pipeline**: 6 stages present (including semantic/acoustic alignment and Apollo super-resolution).
- **Ontology Stats**: 354 total nodes, 295 leaf-like classes vs 283 claimed classes.
- **Tiny Mix Verification**: Reconstructed 2-source mixture successfully (`mix = [1.0, 0.5, -0.5]`).
