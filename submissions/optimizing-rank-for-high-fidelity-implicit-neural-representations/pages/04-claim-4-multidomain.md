# Claim 4 — Multi-domain extension

> The reported improvements extend to natural images, medical images, audio, super-resolution, and novel-view synthesis, with up to about +9 dB PSNR over the same architecture (Tables 1-6).

**Status: `partially_reproduced` (toy-scale: 4 of the paper's modalities; novel-view synthesis not attempted).**

Four modalities are exercised with the same paired-initialization protocol.
Super-resolution trains on a 16x16 coordinate subgrid and is evaluated on the
full 32x32 grid, so its PSNR measures generalization to unseen coordinates
rather than memorization.

| Domain | Adam PSNR | Muon PSNR | Gain |
| --- | --- | --- | --- |
| natural image | 14.01 dB | 14.03 dB | +0.02 dB |
| medical phantom | 11.54 dB | 9.74 dB | -1.8 dB |
| audio 1d | 3.52 dB | 2.22 dB | -1.29 dB |
| super resolution | 14.01 dB | 14.03 dB | +0.02 dB |

- Largest observed gain: **0.02 dB**.
- All domains improved: **False**.
- Domains where Muon beat Adam:
  **2 of 4**.

The claim reproduces only partially, and the failures are reported as
measured. Muon wins narrowly on the two image-grid domains
(**0.02 dB**) but *loses* on the medical phantom
(-1.8 dB) and on the 1-D audio
signal (-1.29 dB). The paper's headline
"up to about +9 dB" is a dataset-scale claim; the only place this
reproduction sees a gap of that size is the Siren image-overfitting result on
the previous page, not this multi-domain sweep.

Novel-view synthesis is not attempted at all: it requires multi-view 3D data
outside this CPU budget. No figure is reported for it.
