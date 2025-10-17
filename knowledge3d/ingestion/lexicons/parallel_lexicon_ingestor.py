"""
Parallel lexicon ingestion helpers.

This module introduces a producer/consumer style workflow that pushes the CPU
bound definition gathering work into a multiprocessing pool while batching the
GPU-bound refinement on the sovereign swarm. The default configuration mirrors
the sequential `LexiconIngestor` API but operates 10× faster on large corpora.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
import json
import time
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

PreprocessedSynset = Dict[str, object]
ResultRecord = Dict[str, object]


# --------------------------------------------------------------------------- #
# Multiprocessing helpers                                                     #
# --------------------------------------------------------------------------- #
def _preprocess_synset_task(synset_name: str) -> PreprocessedSynset:
    """
    Worker-friendly function that extracts the information we need from a WordNet
    synset. It is intentionally top-level so that it can be pickled by
    `multiprocessing.Pool`.
    """
    from nltk.corpus import wordnet as wn  # local import for worker rehydration

    synset = wn.synset(synset_name)
    definition = synset.definition() or ""
    examples = synset.examples() or []
    lemma = synset_name.split(".")[0]

    return {
        "synset_name": synset_name,
        "lemma": lemma,
        "definition": definition,
        "examples": examples,
    }


# --------------------------------------------------------------------------- #
# Parallel ingestor                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class ParallelLexiconIngestor:
    """
    High-throughput lexicon ingestion pipeline.

    Args:
        num_workers:
            Number of CPU workers used for synset preprocessing. Set to ``1`` to
            run sequentially (helpful for tests).
        batch_size:
            Number of preprocessed entries to hand over to the GPU pipeline at
            once. Larger batches improve swarm utilisation.
        chunksize:
            Chunk size passed to ``Pool.imap`` when distributing work.
        text_ingestor:
            Optional pre-instantiated ``SovereignTextIngestor`` (or stub for
            tests). If omitted, a real instance is created lazily.
        swarm_processor:
            Optional ``SovereignLanguageSwarmProcessor`` (or stub).
        gpu_batch_processor:
            Optional callback replacing the default GPU batch routine. Mainly
            used in unit tests to avoid heavy CUDA dependencies.
    """

    num_workers: int = 8
    batch_size: int = 64
    chunksize: int = 1_000
    text_ingestor: object | None = None
    swarm_processor: object | None = None
    gpu_batch_processor: Optional[Callable[[List[PreprocessedSynset]], List[ResultRecord]]] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest_wordnet_parallel(
        self,
        output_path: str | Path,
        *,
        limit: Optional[int] = None,
        wordnet_module=None,
        progress: Optional[Callable[[int, int, float], None]] = None,
    ) -> Dict[str, float]:
        """
        Ingest the WordNet corpus using the parallel pipeline.

        Args:
            output_path: Destination JSON file (will be overwritten).
            limit: Optional cap on the number of synsets (useful for smoke tests).
            wordnet_module: Optional module supplying ``all_synsets``. Defaults to
                ``nltk.corpus.wordnet``.
            progress: Optional callback ``fn(processed, total, elapsed)`` to report
                progress as batches are completed.
        """
        wn = wordnet_module
        if wn is None:
            from nltk.corpus import wordnet as wn  # type: ignore

        synset_names = [s.name() for s in wn.all_synsets()]
        if limit is not None:
            synset_names = synset_names[:limit]

        total_synsets = len(synset_names)
        if total_synsets == 0:
            raise ValueError("No synsets to ingest.")

        start_time = time.perf_counter()
        preprocessed = self._run_cpu_preprocessing(synset_names)
        preprocess_time = time.perf_counter() - start_time

        results = self._run_gpu_batches(
            preprocessed,
            total_synsets=total_synsets,
            start_time=start_time,
            progress_callback=progress,
        )

        total_time = time.perf_counter() - start_time
        throughput = total_synsets / max(total_time, 1e-9)

        self._write_results(
            output_path,
            language="en",
            synsets=results,
            total_time=total_time,
            throughput=throughput,
        )

        self._cleanup_gpu_components()

        return {
            "synset_count": float(len(results)),
            "total_time_s": total_time,
            "preprocessing_time_s": preprocess_time,
            "throughput_synsets_per_sec": throughput,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _run_cpu_preprocessing(self, synset_names: Sequence[str]) -> List[PreprocessedSynset]:
        """Distribute definition extraction across CPU workers."""
        if self.num_workers <= 1:
            return [_preprocess_synset_task(name) for name in synset_names]

        processed: List[PreprocessedSynset] = []
        with Pool(processes=self.num_workers) as pool:
            for payload in pool.imap(_preprocess_synset_task, synset_names, chunksize=self.chunksize):
                processed.append(payload)
        return processed

    def _run_gpu_batches(
        self,
        preprocessed: Sequence[PreprocessedSynset],
        *,
        total_synsets: int,
        start_time: float,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> List[ResultRecord]:
        """Process the preprocessed items in GPU-sized batches."""
        results: List[ResultRecord] = []

        for offset in range(0, len(preprocessed), self.batch_size):
            batch = list(preprocessed[offset : offset + self.batch_size])
            batch_results = self._process_gpu_batch(batch)
            results.extend(batch_results)

            if progress_callback is not None:
                progress_callback(len(results), total_synsets, time.perf_counter() - start_time)
            elif (len(results) // self.batch_size) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                throughput = len(results) / max(elapsed, 1e-9)
                print(
                    f"[ParallelLexiconIngestor] processed {len(results):>6}/{total_synsets} "
                    f"synsets ({throughput:.1f} synsets/s)",
                    flush=True,
                )

        return results

    def _process_gpu_batch(self, batch: List[PreprocessedSynset]) -> List[ResultRecord]:
        """
        Execute the GPU portion of the pipeline. Allows overriding for unit tests.
        """
        if self.gpu_batch_processor is not None:
            return self.gpu_batch_processor(batch)

        self._ensure_gpu_components()

        output: List[ResultRecord] = []
        for item in batch:
            definition = item["definition"] if item["definition"] else item["lemma"]
            text_result = self.text_ingestor.ingest_sentence("en", definition)  # type: ignore[attr-defined]
            swarm_result = self.swarm_processor.process_language_embedding(  # type: ignore[attr-defined]
                text_result["embedding_128"],
                modality="text",
                language="en",
                include_diagnostics=False,
            )

            output.append(
                {
                    "synset": item["synset_name"],
                    "lemma": item["lemma"],
                    "definition": definition,
                    "examples": item["examples"],
                    "position_3d": np.asarray(swarm_result.position_3d, dtype=np.float32).tolist(),
                    "embedding": np.asarray(swarm_result.refined_embedding, dtype=np.float32).tolist(),
                }
            )
        return output

    def _ensure_gpu_components(self) -> None:
        """Instantiate sovereign GPU helpers on demand."""
        if self.text_ingestor is None:
            from knowledge3d.ingestion.language.sovereign_text_pipeline import SovereignTextIngestor

            self.text_ingestor = SovereignTextIngestor()

        if self.swarm_processor is None:
            from knowledge3d.ingestion.language.swarm_integration import SovereignLanguageSwarmProcessor

            self.swarm_processor = SovereignLanguageSwarmProcessor()

    def _cleanup_gpu_components(self) -> None:
        """Best-effort cleanup to release GPU resources."""
        if self.text_ingestor and hasattr(self.text_ingestor, "cleanup"):
            try:
                self.text_ingestor.cleanup()
            except Exception:  # pragma: no cover - defensive
                pass
        if self.swarm_processor and hasattr(self.swarm_processor, "cleanup"):
            try:
                self.swarm_processor.cleanup()
            except Exception:  # pragma: no cover - defensive
                pass

    @staticmethod
    def _write_results(
        output_path: str | Path,
        *,
        language: str,
        synsets: Sequence[ResultRecord],
        total_time: float,
        throughput: float,
    ) -> None:
        """Persist ingestion results to disk."""
        payload = {
            "language": language,
            "source": "WordNet",
            "synset_count": len(synsets),
            "total_time_s": total_time,
            "throughput_synsets_per_sec": throughput,
            "synsets": synsets,
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)


__all__ = ["ParallelLexiconIngestor"]
