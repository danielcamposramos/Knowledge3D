"""Sovereign boot-time GPU star materialization bridge."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess
import zlib

from knowledge3d.cranium.sovereign import loader


CUDA_DIR = Path(__file__).resolve().parents[1] / "cranium" / "cuda"
PTX_DIR = Path(__file__).resolve().parents[1] / "cranium" / "ptx"
DEVICE_HEADER = CUDA_DIR / "device_functions.cuh"
BOOT_FINALIZE_SOURCE = CUDA_DIR / "boot_star_finalize.cu"
BUILD_DECODE_SOURCE = CUDA_DIR / "catalog_build_decode.cu"
STAR_MATERIALIZER_SOURCE = CUDA_DIR / "star_materializer.cu"
REF_CSR_SOURCE = CUDA_DIR / "ref_csr_builder.cu"
STAR_HASH_INDEX_SOURCE = CUDA_DIR / "star_hash_index.cu"
REF_HASH_RESOLVE_SOURCE = CUDA_DIR / "ref_hash_resolve.cu"
REVERSE_SYMLINK_SOURCE = CUDA_DIR / "reverse_symlink_expand.cu"
REVERSE_REF_HASH_SOURCE = CUDA_DIR / "reverse_ref_hash_expand.cu"
ROUTE_CAPABILITY_TRIT_SOURCE = CUDA_DIR / "route_capability_trit.cu"
BOOT_FINALIZE_PTX = PTX_DIR / "boot_star_finalize.ptx"
BUILD_DECODE_PTX = PTX_DIR / "catalog_build_decode.ptx"
STAR_MATERIALIZER_PTX = PTX_DIR / "star_materializer.ptx"
REF_CSR_PTX = PTX_DIR / "ref_csr_builder.ptx"
STAR_HASH_INDEX_PTX = PTX_DIR / "star_hash_index.ptx"
REF_HASH_RESOLVE_PTX = PTX_DIR / "ref_hash_resolve.ptx"
REVERSE_SYMLINK_PTX = PTX_DIR / "reverse_symlink_expand.ptx"
REVERSE_REF_HASH_PTX = PTX_DIR / "reverse_ref_hash_expand.ptx"
ROUTE_CAPABILITY_TRIT_PTX = PTX_DIR / "route_capability_trit.ptx"

RAW_CATALOG_INPUT_ENTRY_BYTES = 176
FINALIZED_CATALOG_INPUT_ENTRY_BYTES = 152
CATALOG_INPUT_ENTRY_BYTES = RAW_CATALOG_INPUT_ENTRY_BYTES
REF_TUPLE_BYTES = 16
BUILD_REF_HASH_BYTES = 16
FEED_SOURCE_REF_BYTES = 24


class StarMaterializerBridge:
    """Load and launch sovereign boot kernels for star materialization."""

    def __init__(self) -> None:
        self._finalizer_ptx = self._ensure_ptx(BOOT_FINALIZE_SOURCE, BOOT_FINALIZE_PTX)
        self._decode_ptx = self._ensure_ptx(BUILD_DECODE_SOURCE, BUILD_DECODE_PTX)
        self._materializer_ptx = self._ensure_ptx(STAR_MATERIALIZER_SOURCE, STAR_MATERIALIZER_PTX)
        self._csr_ptx = self._ensure_ptx(REF_CSR_SOURCE, REF_CSR_PTX)
        self._hash_index_ptx = self._ensure_ptx(STAR_HASH_INDEX_SOURCE, STAR_HASH_INDEX_PTX)
        self._ref_hash_resolve_ptx = self._ensure_ptx(REF_HASH_RESOLVE_SOURCE, REF_HASH_RESOLVE_PTX)
        self._reverse_symlink_ptx = self._ensure_ptx(REVERSE_SYMLINK_SOURCE, REVERSE_SYMLINK_PTX)
        self._reverse_ref_hash_ptx = self._ensure_ptx(REVERSE_REF_HASH_SOURCE, REVERSE_REF_HASH_PTX)
        self._route_capability_trit_ptx = self._ensure_ptx(ROUTE_CAPABILITY_TRIT_SOURCE, ROUTE_CAPABILITY_TRIT_PTX)
        self.finalizer_module = loader.load_module_from_file(str(self._finalizer_ptx))
        self.decode_module = loader.load_module_from_file(str(self._decode_ptx))
        self.materializer_module = loader.load_module_from_file(str(self._materializer_ptx))
        self.csr_module = loader.load_module_from_file(str(self._csr_ptx))
        self.hash_index_module = loader.load_module_from_file(str(self._hash_index_ptx))
        self.ref_hash_resolve_module = loader.load_module_from_file(str(self._ref_hash_resolve_ptx))
        self.reverse_symlink_module = loader.load_module_from_file(str(self._reverse_symlink_ptx))
        self.reverse_ref_hash_module = loader.load_module_from_file(str(self._reverse_ref_hash_ptx))
        self.route_capability_trit_module = loader.load_module_from_file(str(self._route_capability_trit_ptx))
        self.finalizer_kernel = loader.get_function(self.finalizer_module, "boot_star_finalize")
        self.decode_kernel = loader.get_function(self.decode_module, "catalog_build_decode")
        self.materializer_kernel = loader.get_function(self.materializer_module, "star_materializer")
        self.ref_count_kernel = loader.get_function(self.csr_module, "ref_count_refs")
        self.ref_scan_kernel = loader.get_function(self.csr_module, "ref_scan_offsets")
        self.ref_scatter_kernel = loader.get_function(self.csr_module, "ref_scatter_refs")
        self.hash_index_kernel = loader.get_function(self.hash_index_module, "star_hash_index_build")
        self.ref_hash_resolve_kernel = loader.get_function(self.ref_hash_resolve_module, "ref_hash_resolve")
        self.reverse_symlink_kernel = loader.get_function(self.reverse_symlink_module, "reverse_symlink_expand")
        self.reverse_ref_hash_kernel = loader.get_function(self.reverse_ref_hash_module, "reverse_ref_hash_expand")
        self.route_capability_trit_kernel = loader.get_function(self.route_capability_trit_module, "route_capability_trit")
        self.finalizer_ptx_signature = self._ptx_signature(self._finalizer_ptx)
        self.decode_ptx_signature = self._ptx_signature(self._decode_ptx)
        self.materializer_ptx_signature = self._ptx_signature(self._materializer_ptx)
        self.csr_builder_ptx_signature = self._ptx_signature(self._csr_ptx)
        self.hash_index_ptx_signature = self._ptx_signature(self._hash_index_ptx)
        self.ref_hash_resolve_ptx_signature = self._ptx_signature(self._ref_hash_resolve_ptx)
        self.reverse_symlink_ptx_signature = self._ptx_signature(self._reverse_symlink_ptx)
        self.reverse_ref_hash_ptx_signature = self._ptx_signature(self._reverse_ref_hash_ptx)
        self.route_capability_trit_ptx_signature = self._ptx_signature(self._route_capability_trit_ptx)

    @staticmethod
    def _ensure_ptx(source: Path, target: Path) -> Path:
        PTX_DIR.mkdir(parents=True, exist_ok=True)
        newest_source_mtime = max(
            source.stat().st_mtime,
            DEVICE_HEADER.stat().st_mtime if DEVICE_HEADER.exists() else 0.0,
        )
        if target.exists() and target.stat().st_mtime >= newest_source_mtime:
            return target
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError(f"nvcc_not_found_for_{source.stem}")
        subprocess.run(
            [
                nvcc,
                "-ptx",
                "-arch=sm_86",
                "--compiler-bindir",
                "/usr/bin/gcc-13",
                "-o",
                str(target),
                str(source),
            ],
            check=True,
        )
        return target

    @staticmethod
    def _ptx_signature(path: Path) -> str:
        payload = path.read_bytes()
        return f"{path.name}:{zlib.crc32(payload) & 0xFFFFFFFF:08x}"

    def ptx_signatures(self) -> dict[str, str]:
        return {
            "boot_finalize_ptx_signature": str(self.finalizer_ptx_signature),
            "build_decode_ptx_signature": str(self.decode_ptx_signature),
            "materializer_ptx_signature": str(self.materializer_ptx_signature),
            "csr_builder_ptx_signature": str(self.csr_builder_ptx_signature),
            "hash_index_ptx_signature": str(self.hash_index_ptx_signature),
            "ref_hash_resolve_ptx_signature": str(self.ref_hash_resolve_ptx_signature),
            "reverse_symlink_ptx_signature": str(self.reverse_symlink_ptx_signature),
            "reverse_ref_hash_ptx_signature": str(self.reverse_ref_hash_ptx_signature),
            "route_capability_trit_ptx_signature": str(self.route_capability_trit_ptx_signature),
        }

    def decode_build_rows(
        self,
        *,
        build_rows_ptr,
        raw_input_ptr,
        entry_count: int,
        stream=None,
        block_size: int = 128,
    ) -> None:
        total = max(0, int(entry_count))
        if total <= 0:
            return
        loader.launch(
            self.decode_kernel,
            ((total + block_size - 1) // block_size, 1, 1),
            (block_size, 1, 1),
            [
                build_rows_ptr,
                raw_input_ptr,
                ctypes.c_uint(total),
            ],
            stream=stream,
        )

    def finalize_chunk(
        self,
        *,
        raw_input_ptr,
        finalized_input_ptr,
        entry_count: int,
        stream=None,
        block_size: int = 128,
    ) -> None:
        total = max(0, int(entry_count))
        if total <= 0:
            return
        loader.launch(
            self.finalizer_kernel,
            ((total + block_size - 1) // block_size, 1, 1),
            (block_size, 1, 1),
            [
                raw_input_ptr,
                finalized_input_ptr,
                ctypes.c_uint(total),
            ],
            stream=stream,
        )

    def materialize_chunk(
        self,
        *,
        galaxy_table_ptr,
        input_ptr,
        entry_count: int,
        star_offset: int,
        router_offsets_ptr,
        router_counts_ptr,
        executor_offsets_ptr,
        executor_counts_ptr,
        validator_offsets_ptr,
        validator_counts_ptr,
        anti_pattern_offsets_ptr,
        anti_pattern_counts_ptr,
        stream=None,
        block_size: int = 128,
    ) -> None:
        total = max(0, int(entry_count))
        if total <= 0:
            return
        loader.launch(
            self.materializer_kernel,
            ((total + block_size - 1) // block_size, 1, 1),
            (block_size, 1, 1),
            [
                galaxy_table_ptr,
                input_ptr,
                ctypes.c_uint(total),
                ctypes.c_uint(int(star_offset)),
                router_offsets_ptr,
                router_counts_ptr,
                executor_offsets_ptr,
                executor_counts_ptr,
                validator_offsets_ptr,
                validator_counts_ptr,
                anti_pattern_offsets_ptr,
                anti_pattern_counts_ptr,
            ],
            stream=stream,
        )

    def count_refs(
        self,
        *,
        ref_tuples_ptr,
        ref_count: int,
        star_count: int,
        router_counts_ptr,
        executor_counts_ptr,
        validator_counts_ptr,
        anti_pattern_counts_ptr,
        stream=None,
        block_size: int = 128,
    ) -> None:
        total = max(0, int(ref_count))
        if total <= 0:
            return
        loader.launch(
            self.ref_count_kernel,
            ((total + block_size - 1) // block_size, 1, 1),
            (block_size, 1, 1),
            [
                ref_tuples_ptr,
                ctypes.c_uint(total),
                ctypes.c_uint(int(star_count)),
                router_counts_ptr,
                executor_counts_ptr,
                validator_counts_ptr,
                anti_pattern_counts_ptr,
            ],
            stream=stream,
        )

    def scan_offsets(
        self,
        *,
        star_count: int,
        router_counts_ptr,
        router_offsets_ptr,
        executor_counts_ptr,
        executor_offsets_ptr,
        validator_counts_ptr,
        validator_offsets_ptr,
        anti_pattern_counts_ptr,
        anti_pattern_offsets_ptr,
        stream=None,
    ) -> None:
        loader.launch(
            self.ref_scan_kernel,
            (1, 1, 1),
            (1, 1, 1),
            [
                ctypes.c_uint(int(star_count)),
                router_counts_ptr,
                router_offsets_ptr,
                executor_counts_ptr,
                executor_offsets_ptr,
                validator_counts_ptr,
                validator_offsets_ptr,
                anti_pattern_counts_ptr,
                anti_pattern_offsets_ptr,
            ],
            stream=stream,
        )

    def scatter_refs(
        self,
        *,
        galaxy_table_ptr,
        ref_indices_ptr,
        ref_tuples_ptr,
        ref_count: int,
        star_count: int,
        router_offsets_ptr,
        executor_offsets_ptr,
        validator_offsets_ptr,
        anti_pattern_offsets_ptr,
        stream=None,
        block_size: int = 128,
    ) -> None:
        total = max(0, int(ref_count))
        if total <= 0:
            return
        loader.launch(
            self.ref_scatter_kernel,
            ((total + block_size - 1) // block_size, 1, 1),
            (block_size, 1, 1),
            [
                galaxy_table_ptr,
                ref_indices_ptr,
                ref_tuples_ptr,
                ctypes.c_uint(total),
                ctypes.c_uint(int(star_count)),
                router_offsets_ptr,
                executor_offsets_ptr,
                validator_offsets_ptr,
                anti_pattern_offsets_ptr,
            ],
            stream=stream,
        )

    def build_star_hash_index(
        self,
        *,
        galaxy_table_ptr,
        star_count: int,
        hash_keys_ptr,
        hash_values_ptr,
        table_capacity: int,
        collision_flag_ptr,
        stream=None,
        block_size: int = 128,
    ) -> None:
        total = max(0, int(star_count))
        if total <= 0:
            return
        loader.launch(
            self.hash_index_kernel,
            ((total + block_size - 1) // block_size, 1, 1),
            (block_size, 1, 1),
            [
                galaxy_table_ptr,
                ctypes.c_uint(total),
                hash_keys_ptr,
                hash_values_ptr,
                ctypes.c_uint(max(1, int(table_capacity))),
                collision_flag_ptr,
            ],
            stream=stream,
        )

    def resolve_ref_hashes(
        self,
        *,
        ref_hash_rows_ptr,
        ref_count: int,
        star_count: int,
        hash_keys_ptr,
        hash_values_ptr,
        table_capacity: int,
        router_counts_ptr,
        executor_counts_ptr,
        validator_counts_ptr,
        anti_pattern_counts_ptr,
        resolved_ref_tuples_ptr,
        unresolved_source_ptr,
        unresolved_target_hash_ptr,
        unresolved_count_ptr,
        stream=None,
        block_size: int = 128,
    ) -> None:
        total = max(0, int(ref_count))
        if total <= 0:
            return
        loader.launch(
            self.ref_hash_resolve_kernel,
            ((total + block_size - 1) // block_size, 1, 1),
            (block_size, 1, 1),
            [
                ref_hash_rows_ptr,
                ctypes.c_uint(total),
                ctypes.c_uint(max(1, int(star_count))),
                hash_keys_ptr,
                hash_values_ptr,
                ctypes.c_uint(max(1, int(table_capacity))),
                router_counts_ptr,
                executor_counts_ptr,
                validator_counts_ptr,
                anti_pattern_counts_ptr,
                resolved_ref_tuples_ptr,
                unresolved_source_ptr,
                unresolved_target_hash_ptr,
                unresolved_count_ptr,
            ],
            stream=stream,
        )

    def expand_reverse_symlinks(
        self,
        *,
        forward_ref_tuples_ptr,
        ref_count: int,
        star_count: int,
        router_counts_ptr,
        executor_counts_ptr,
        validator_counts_ptr,
        anti_pattern_counts_ptr,
        expanded_ref_tuples_ptr,
        stream=None,
        block_size: int = 128,
    ) -> None:
        total = max(0, int(ref_count))
        if total <= 0:
            return
        loader.launch(
            self.reverse_symlink_kernel,
            ((total + block_size - 1) // block_size, 1, 1),
            (block_size, 1, 1),
            [
                forward_ref_tuples_ptr,
                ctypes.c_uint(total),
                ctypes.c_uint(max(1, int(star_count))),
                router_counts_ptr,
                executor_counts_ptr,
                validator_counts_ptr,
                anti_pattern_counts_ptr,
                expanded_ref_tuples_ptr,
            ],
            stream=stream,
        )

    def expand_reverse_ref_hashes(
        self,
        *,
        feed_source_ref_rows_ptr,
        ref_count: int,
        star_count: int,
        hash_keys_ptr,
        hash_values_ptr,
        table_capacity: int,
        galaxy_table_ptr,
        build_ref_hash_rows_ptr,
        unresolved_source_ptr,
        unresolved_target_hash_ptr,
        unresolved_count_ptr,
        stream=None,
        block_size: int = 128,
    ) -> None:
        total = max(0, int(ref_count))
        if total <= 0:
            return
        loader.launch(
            self.reverse_ref_hash_kernel,
            ((total + block_size - 1) // block_size, 1, 1),
            (block_size, 1, 1),
            [
                feed_source_ref_rows_ptr,
                ctypes.c_uint(total),
                ctypes.c_uint(max(1, int(star_count))),
                hash_keys_ptr,
                hash_values_ptr,
                ctypes.c_uint(max(1, int(table_capacity))),
                galaxy_table_ptr,
                build_ref_hash_rows_ptr,
                unresolved_source_ptr,
                unresolved_target_hash_ptr,
                unresolved_count_ptr,
            ],
            stream=stream,
        )

    def audit_route_capability(
        self,
        *,
        raw_input_ptr,
        star_count: int,
        route_trits_ptr,
        audit_counts_ptr,
        stream=None,
        block_size: int = 128,
    ) -> None:
        total = max(0, int(star_count))
        if total <= 0:
            return
        loader.launch(
            self.route_capability_trit_kernel,
            ((total + block_size - 1) // block_size, 1, 1),
            (block_size, 1, 1),
            [
                raw_input_ptr,
                ctypes.c_uint(total),
                route_trits_ptr,
                audit_counts_ptr,
            ],
            stream=stream,
        )
