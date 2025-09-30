# Training Progress Summary

- Multi-trainer (k3d_multi_x2): started 2x50 epochs with beam decoding (RPN). Running now.
- Shapes consistency (k3d_shapes_100): started 100 epochs with GLB previews.
- AV consistency (k3d_av_30): scanning ~2000 audio and ~2000 video files; training 30 epochs.

For detailed epoch-by-epoch logs, this run will append records into docs/benchmarks/progress_log.json; aggregate with:

    PYTHONPATH=. python -m knowledge3d.tools.phase25.progress_dashboard

This writes docs/benchmarks/progress_summary.json and updates this summary.
