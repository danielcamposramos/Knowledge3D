"""Registry of standard open formats and MIME types."""

from __future__ import annotations


FILE_FORMATS: dict[str, dict[str, str]] = {
    "glb": {"mime": "model/gltf-binary", "domain": "Visual", "description": "GL Transmission Format Binary"},
    "gltf": {"mime": "model/gltf+json", "domain": "Visual", "description": "GL Transmission Format JSON"},
    "obj": {"mime": "model/obj", "domain": "Visual", "description": "Wavefront OBJ"},
    "fbx": {"mime": "application/octet-stream", "domain": "Visual", "description": "Filmbox"},
    "usdz": {"mime": "model/vnd.usdz+zip", "domain": "Visual", "description": "Universal Scene Description"},
    "stl": {"mime": "model/stl", "domain": "Visual", "description": "Stereolithography"},
    "pdf": {"mime": "application/pdf", "domain": "Language", "description": "Portable Document Format"},
    "html": {"mime": "text/html", "domain": "Language", "description": "HyperText Markup Language"},
    "md": {"mime": "text/markdown", "domain": "Language", "description": "Markdown"},
    "tex": {"mime": "application/x-tex", "domain": "Mathematics", "description": "LaTeX"},
    "epub": {"mime": "application/epub+zip", "domain": "Language", "description": "Electronic Publication"},
    "png": {"mime": "image/png", "domain": "Visual", "description": "Portable Network Graphics"},
    "svg": {"mime": "image/svg+xml", "domain": "Visual", "description": "Scalable Vector Graphics"},
    "jpg": {"mime": "image/jpeg", "domain": "Visual", "description": "JPEG Image"},
    "webp": {"mime": "image/webp", "domain": "Visual", "description": "WebP Image"},
    "wav": {"mime": "audio/wav", "domain": "Audio", "description": "Waveform Audio"},
    "mp3": {"mime": "audio/mpeg", "domain": "Audio", "description": "MPEG Audio Layer 3"},
    "ogg": {"mime": "audio/ogg", "domain": "Audio", "description": "Ogg Vorbis"},
    "flac": {"mime": "audio/flac", "domain": "Audio", "description": "Free Lossless Audio Codec"},
    "opus": {"mime": "audio/opus", "domain": "Audio", "description": "Opus Audio"},
    "mp4": {"mime": "video/mp4", "domain": "Audio", "description": "MPEG-4 Video"},
    "webm": {"mime": "video/webm", "domain": "Audio", "description": "WebM Video"},
    "mkv": {"mime": "video/x-matroska", "domain": "Audio", "description": "Matroska Video"},
    "json": {"mime": "application/json", "domain": "Tools", "description": "JavaScript Object Notation"},
    "csv": {"mime": "text/csv", "domain": "Tools", "description": "Comma-Separated Values"},
    "xml": {"mime": "application/xml", "domain": "Tools", "description": "Extensible Markup Language"},
    "yaml": {"mime": "application/x-yaml", "domain": "Tools", "description": "YAML Ain't Markup Language"},
    "toml": {"mime": "application/toml", "domain": "Tools", "description": "Tom's Obvious Minimal Language"},
    "py": {"mime": "text/x-python", "domain": "Tools", "description": "Python Source"},
    "ts": {"mime": "text/typescript", "domain": "Tools", "description": "TypeScript Source"},
    "rs": {"mime": "text/x-rust", "domain": "Tools", "description": "Rust Source"},
    "cu": {"mime": "text/x-cuda", "domain": "Tools", "description": "CUDA Source"},
    "ptx": {"mime": "text/x-ptx", "domain": "Tools", "description": "Parallel Thread Execution"},
}


def iter_format_entries() -> list[tuple[str, dict[str, str]]]:
    return [(key, FILE_FORMATS[key]) for key in sorted(FILE_FORMATS.keys())]


__all__ = ["FILE_FORMATS", "iter_format_entries"]
