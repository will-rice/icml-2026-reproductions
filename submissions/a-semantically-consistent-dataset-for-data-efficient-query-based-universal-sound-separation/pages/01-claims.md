# Claim Verification Summary

- Attempt: `9fb6688a-2be9-4436-98bf-2525fb2a8df1`
- Paper ID: `vCc2NAe0OS`
- Title: A Semantically Consistent Dataset for Data-Efficient Query-Based Universal Sound Separation

## Target Claim Details

### Claim 1: Verified (status: `verified`)
- **Text**: The Hive construction pipeline mines high-purity single-event segments, aligns them semantically and acoustically, and standardizes audio via super-resolution (Figure 1)
- **SHA-256**: `d08685cc24d359b1e551724ca0d4730d64abae79eef4b8594987194951c35d61`
- **Evidence**: Pinned code exposes all six purification stages, including Qwen/AudioTag alignment and Apollo super-resolution.

### Claim 2: Toy (status: `toy`)
- **Text**: Hive comprises 2,442 hours of raw audio and 19.6 million synthesized mixtures spanning a 283-class ontology (Section 4.2)
- **SHA-256**: `0a0c5d8defd0af930bc52a50a8e5b43a16aa6cff60526240043a5b9b74855ab6`
- **Evidence**: Pinned dataset/model cards declare 2,442 hours and 19.6M mixtures, and Hub metadata exposes large metadata/audio repositories. The naive ontology leaf count is 295 vs claimed 283.

### Claim 3: Inconclusive (status: `inconclusive`)
- **Text**: A semantic compatibility matrix is used to avoid implausible event co-occurrences during mixture synthesis (Figure 5)
- **SHA-256**: `a13bf6a08539cdb72a0e8fca7f09d4da0ae19b7bc78aedc9fd2ffce48d06b17b`
- **Evidence**: The README describes logic-based co-occurrence constraints, but the pinned source audit did not find a machine-readable compatibility matrix artifact.

### Claim 4: Inconclusive (status: `inconclusive`)
- **Text**: Enforcing semantic-consistency constraints yields consistent gains over random mixtures built from the same purified single-event sources (Table 3)
- **SHA-256**: `e9ee277d7ffb032dbd8b8917df59f5482f8f824efb3cdbee599fad9449dba8b2`
- **Evidence**: No released Table 3 result file comparing semantic-consistency constraints against random mixtures was found.

### Claim 5: Toy (status: `toy`)
- **Text**: Hive-trained AudioSep and FlowSep are compared against original checkpoints and SAM-Audio on the Hive test set and third-party out-of-distribution benchmarks (Tables 4 and 5)
- **SHA-256**: `1ac4bd628786bc0f0e1f222932c9de9b87d002ffb4c162e27184ba49290d83b8`
- **Evidence**: Hive-trained AudioSep and FlowSep checkpoint repositories and inference wrappers are present; benchmark Tables 4 and 5 were not recomputed.

### Claim 6: Inconclusive (status: `inconclusive`)
- **Text**: Paired co-occurrence/decorrelation tests show Hive training reduces shortcut reliance while controlling target identity, source count, and SNR (Table 6)
- **SHA-256**: `644ff4d1ba8cef68e7c8f6c4b370e5ff26d9a1d09c946cecf09a8dda67f7c5e2`
- **Evidence**: No released paired co-occurrence/decorrelation result artifact for Table 6 was found.
