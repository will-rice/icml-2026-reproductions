# TextAtlas5M Reproduction Design

Attempt: `c21b9754-4227-4943-b26f-ac9afd5712a4`
Paper: `5vufrrbi4N`

This recovery design reconstructs the lost validated source for the existing TextAtlas5M lane. The evidence is intentionally scoped to deterministic public metadata checks because the full TextAtlas5M image corpus is approximately terabyte scale and model fine-tuning claims are outside the CPU-only budget.

Target checks:

- Verify the released TextAtlas5M metadata scale against the pinned Hugging Face dataset revision `f9f2a0f5000fbb078f718197acb45cfb9ceed551`.
- Verify the public TextAtlasEval count and four-domain structure as metadata evidence.
- Render unsupported benchmark and fine-tuning claims as unavailable rather than reproduced.

The implementation writes a machine-readable JSON bundle and two judge-visible Markdown pages. It does not report paper numbers as reproduced model metrics.
