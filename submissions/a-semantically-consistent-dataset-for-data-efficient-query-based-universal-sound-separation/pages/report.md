# A Semantically Consistent Dataset for Data-Efficient Query-Based Universal Sound Separation

- Attempt: `9fb6688a-2be9-4436-98bf-2525fb2a8df1`
- Paper: `vCc2NAe0OS`
- Snapshot: `11dbf14c30f9b4573e95f7e6df7227c998a7ef09b854edc38ef541d35245233d`
- Code pin: `github:ShandaAI/Hive@f41b507d6be616ba864a5cd538b071338b6bd90d`

## Claim Evidence

### Claim 1: verified

The Hive construction pipeline mines high-purity single-event segments, aligns them semantically and acoustically, and standardizes audio via super-resolution (Figure 1)

Pinned code exposes all six purification stages, including Qwen/AudioTag alignment and Apollo super-resolution.

### Claim 2: toy

Hive comprises 2,442 hours of raw audio and 19.6 million synthesized mixtures spanning a 283-class ontology (Section 4.2)

Pinned dataset/model cards declare 2,442 hours and 19.6M mixtures, and Hub metadata exposes large metadata/audio repositories. This run did not download or sum the full audio archive; the naive ontology leaf count is 295 vs claimed 283.

### Claim 3: inconclusive

A semantic compatibility matrix is used to avoid implausible event co-occurrences during mixture synthesis (Figure 5)

The README describes logic-based co-occurrence constraints, but the pinned source audit did not find a machine-readable compatibility matrix artifact.

### Claim 4: inconclusive

Enforcing semantic-consistency constraints yields consistent gains over random mixtures built from the same purified single-event sources (Table 3)

No released Table 3 result file comparing semantic-consistency constraints against random mixtures was found.

### Claim 5: toy

Hive-trained AudioSep and FlowSep are compared against original checkpoints and SAM-Audio on the Hive test set and third-party out-of-distribution benchmarks (Tables 4 and 5)

Hive-trained AudioSep and FlowSep checkpoint repositories and inference wrappers are present; benchmark Tables 4 and 5 were not recomputed.

### Claim 6: inconclusive

Paired co-occurrence/decorrelation tests show Hive training reduces shortcut reliance while controlling target identity, source count, and SNR (Table 6)

No released paired co-occurrence/decorrelation result artifact for Table 6 was found.

## Audits

```json
{
  "hub": {
    "audio_archive_bytes_sampled": 52992204800,
    "audio_archive_tars": 5600,
    "hive_checkpoints": "present",
    "metadata_bytes": 1232988399,
    "metadata_parquets": "present",
    "revisions": {
      "audio_archive": "7ed0f3ac1e166b2e1455cbff550defc618bab25d",
      "audiosep_model": "113d2e4399a4f19b6a0d567bbde38f2fe1b11794",
      "flowsep_model": "7af336090e0c155b1850de37ab310cf36c3e390e",
      "metadata_dataset": "32b57157653ac31a7b525dbfa57aa03aa4d8e3fd"
    }
  },
  "ontology": {
    "claimed_classes": 283,
    "leaf_like_count": 295,
    "matches_claimed_classes": false,
    "node_count": 354
  },
  "pipeline": {
    "missing_stages": [],
    "semantic_acoustic_alignment": "present",
    "stage_count": 6,
    "super_resolution": "present"
  },
  "results": {
    "benchmark_result_files": [],
    "table3_result_files": [],
    "table6_result_files": []
  },
  "tiny_mix": {
    "mix": [
      1.0,
      0.5,
      -0.5
    ],
    "source_count": 2
  }
}
```

No paper-reported benchmark value is presented as a recomputed measurement.
