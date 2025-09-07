# Open Datasets (Hugging Face) — Audio/Video Sources

Goal: curate audio and video datasets with captions to maximize cross‑modal connectivity (audio↔text↔video, later ↔image via WIT), while allowing unmatched items to fill to 50k per modality.

Recommended sources (HF dataset IDs)
- Audio + captions
  - `confit/audiocaps` — AudioCaps (YouTube audio excerpts with captions).
  - `CLAPv2/clotho_full` — Clotho (audio with 5 human captions).
- Video + captions
  - `friedrichor/MSR-VTT` (or `AlexZigma/msr-vtt`) — MSR‑VTT videos with captions (train/val/test splits).
  - `HuggingFaceM4/vatex` — VATEX (videos with multilingual captions).
  - `gigant/webvid-mini` or `qingy2024/webvid-mini-100k-scored` — WebVid subsets with captions; pick a scored top slice.

Links (examples)
- AudioCaps search: https://huggingface.co/datasets?search=audiocaps
- Clotho search: https://huggingface.co/datasets?search=clotho
- MSR‑VTT search: https://huggingface.co/datasets?search=MSR-VTT
- VATEX search: https://huggingface.co/datasets?search=vatex
- WebVid search: https://huggingface.co/datasets?search=webvid

Selection policy (50k target each)
- Audio: 30k AudioCaps + 20k Clotho (dedup by caption hash; English first). If fewer, fill to 50k from remaining split(s).
- Video: 20k MSR‑VTT + 15k VATEX + 15k WebVid (top scored). If fewer, fill to 50k from remaining.
- Images (WIT): choose 50k lines most similar to the union of audio/video captions to maximize bridging.
- Unmatched items are not excluded; they fill remainder to reach 50k.

Implementation helpers
- Fetch: `knowledge3d/tools/hf_fetch_multimodal.py` — downloads/batches audio/video + captions to local folders and JSON metadata.
- Match: `knowledge3d/tools/match_crossmodal.py` — aligns audio↔video via TF‑IDF over captions; produces matched id pairs and a caption pool for WIT selection.
- Build GLBs: `knowledge3d/tools/build_multimodal_50k.py` — orchestrates embedding and GLB generation.

Notes
- Licenses vary; verify dataset licenses before redistribution. Prefer subsets that host media on HF to simplify fetching. When only URLs are provided, the fetcher will skip items without accessible files unless URL download is permitted.
