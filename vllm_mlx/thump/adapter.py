"""ctypes wrapper over the thump runtime adapter library."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DEFAULT_LIB_PATH = Path(
    os.environ.get(
        "VLLM_MLX_THUMP_LIB",
        "/Users/David/code/thump-stack/build/libthump_runtime.dylib",
    )
)


class ThumpRuntimeError(RuntimeError):
    """Raised when the adapter library returns a non-zero status."""


class _CRopeConfig(ctypes.Structure):
    _fields_ = [
        ("variant", ctypes.c_uint8),
        ("scaling_mode", ctypes.c_uint8),
        ("theta", ctypes.c_float),
        ("partial_rotary_factor", ctypes.c_float),
        ("scaling_factor", ctypes.c_float),
        ("original_max_position_embeddings", ctypes.c_uint32),
        ("beta_fast", ctypes.c_float),
        ("beta_slow", ctypes.c_float),
        ("mscale", ctypes.c_float),
        ("mscale_all_dim", ctypes.c_float),
    ]


class _CBlockGeometry(ctypes.Structure):
    _fields_ = [
        ("block_size_tokens", ctypes.c_uint32),
        ("num_kv_heads", ctypes.c_uint32),
        ("head_dim", ctypes.c_uint32),
        ("group_size", ctypes.c_uint32),
        ("rope", _CRopeConfig),
    ]


class _CSessionMetadata(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version_major", ctypes.c_uint16),
        ("version_minor", ctypes.c_uint16),
        ("flags", ctypes.c_uint32),
        ("model_id_hash", ctypes.c_uint64),
        ("session_id", ctypes.c_uint64),
        ("layer_index", ctypes.c_uint32),
        ("prompt_tokens", ctypes.c_uint32),
        ("generated_tokens", ctypes.c_uint32),
        ("saved_sequence_length", ctypes.c_uint32),
        ("saved_sequence_version", ctypes.c_uint32),
        ("saved_block_count", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
    ]


class _CSessionManifest(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version_major", ctypes.c_uint16),
        ("version_minor", ctypes.c_uint16),
        ("flags", ctypes.c_uint32),
        ("model_id_hash", ctypes.c_uint64),
        ("session_id", ctypes.c_uint64),
        ("sequence_id", ctypes.c_uint64),
        ("prompt_tokens", ctypes.c_uint32),
        ("generated_tokens", ctypes.c_uint32),
        ("bank_count", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
    ]


THUMP_RT_SESSION_BANK_PATH_MAX = 192


class _CSessionBankEntry(ctypes.Structure):
    _fields_ = [
        ("layer_index", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("bank_relpath", ctypes.c_char * THUMP_RT_SESSION_BANK_PATH_MAX),
    ]


@dataclass(frozen=True)
class RopeConfig:
    variant: int
    scaling_mode: int = 0
    theta: float = 0.0
    partial_rotary_factor: float = 1.0
    scaling_factor: float = 1.0
    original_max_position_embeddings: int = 0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 0.0

    def to_c(self) -> _CRopeConfig:
        return _CRopeConfig(
            variant=self.variant,
            scaling_mode=self.scaling_mode,
            theta=self.theta,
            partial_rotary_factor=self.partial_rotary_factor,
            scaling_factor=self.scaling_factor,
            original_max_position_embeddings=self.original_max_position_embeddings,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            mscale=self.mscale,
            mscale_all_dim=self.mscale_all_dim,
        )

    @classmethod
    def from_c(cls, cfg: _CRopeConfig) -> "RopeConfig":
        return cls(
            variant=int(cfg.variant),
            scaling_mode=int(cfg.scaling_mode),
            theta=float(cfg.theta),
            partial_rotary_factor=float(cfg.partial_rotary_factor),
            scaling_factor=float(cfg.scaling_factor),
            original_max_position_embeddings=int(cfg.original_max_position_embeddings),
            beta_fast=float(cfg.beta_fast),
            beta_slow=float(cfg.beta_slow),
            mscale=float(cfg.mscale),
            mscale_all_dim=float(cfg.mscale_all_dim),
        )


@dataclass(frozen=True)
class BlockGeometry:
    block_size_tokens: int
    num_kv_heads: int
    head_dim: int
    group_size: int
    rope: RopeConfig

    @property
    def block_elements(self) -> int:
        return self.block_size_tokens * self.num_kv_heads * self.head_dim

    def to_c(self) -> _CBlockGeometry:
        return _CBlockGeometry(
            block_size_tokens=self.block_size_tokens,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            group_size=self.group_size,
            rope=self.rope.to_c(),
        )

    @classmethod
    def from_c(cls, geom: _CBlockGeometry) -> "BlockGeometry":
        return cls(
            block_size_tokens=int(geom.block_size_tokens),
            num_kv_heads=int(geom.num_kv_heads),
            head_dim=int(geom.head_dim),
            group_size=int(geom.group_size),
            rope=RopeConfig.from_c(geom.rope),
        )


@dataclass(frozen=True)
class SessionMetadata:
    flags: int
    model_id_hash: int
    session_id: int
    layer_index: int
    prompt_tokens: int
    generated_tokens: int
    saved_sequence_length: int = 0
    saved_sequence_version: int = 0
    saved_block_count: int = 0

    def to_c(self) -> _CSessionMetadata:
        return _CSessionMetadata(
            magic=0,
            version_major=0,
            version_minor=0,
            flags=self.flags,
            model_id_hash=self.model_id_hash,
            session_id=self.session_id,
            layer_index=self.layer_index,
            prompt_tokens=self.prompt_tokens,
            generated_tokens=self.generated_tokens,
            saved_sequence_length=self.saved_sequence_length,
            saved_sequence_version=self.saved_sequence_version,
            saved_block_count=self.saved_block_count,
            reserved0=0,
        )

    @classmethod
    def from_c(cls, meta: _CSessionMetadata) -> "SessionMetadata":
        return cls(
            flags=int(meta.flags),
            model_id_hash=int(meta.model_id_hash),
            session_id=int(meta.session_id),
            layer_index=int(meta.layer_index),
            prompt_tokens=int(meta.prompt_tokens),
            generated_tokens=int(meta.generated_tokens),
            saved_sequence_length=int(meta.saved_sequence_length),
            saved_sequence_version=int(meta.saved_sequence_version),
            saved_block_count=int(meta.saved_block_count),
        )


@dataclass(frozen=True)
class SessionManifest:
    flags: int
    model_id_hash: int
    session_id: int
    sequence_id: int
    prompt_tokens: int
    generated_tokens: int
    bank_count: int = 0

    def to_c(self, *, bank_count: int | None = None) -> _CSessionManifest:
        return _CSessionManifest(
            magic=0,
            version_major=0,
            version_minor=0,
            flags=self.flags,
            model_id_hash=self.model_id_hash,
            session_id=self.session_id,
            sequence_id=self.sequence_id,
            prompt_tokens=self.prompt_tokens,
            generated_tokens=self.generated_tokens,
            bank_count=self.bank_count if bank_count is None else bank_count,
            reserved0=0,
        )

    @classmethod
    def from_c(cls, manifest: _CSessionManifest) -> "SessionManifest":
        return cls(
            flags=int(manifest.flags),
            model_id_hash=int(manifest.model_id_hash),
            session_id=int(manifest.session_id),
            sequence_id=int(manifest.sequence_id),
            prompt_tokens=int(manifest.prompt_tokens),
            generated_tokens=int(manifest.generated_tokens),
            bank_count=int(manifest.bank_count),
        )


@dataclass(frozen=True)
class SessionBankEntry:
    layer_index: int
    bank_relpath: str

    def to_c(self) -> _CSessionBankEntry:
        encoded = os.fsencode(self.bank_relpath)
        if len(encoded) >= THUMP_RT_SESSION_BANK_PATH_MAX:
            raise ValueError("bank_relpath exceeds adapter limit")
        entry = _CSessionBankEntry()
        entry.layer_index = self.layer_index
        entry.reserved0 = 0
        entry.bank_relpath = encoded
        return entry

    @classmethod
    def from_c(cls, entry: _CSessionBankEntry) -> "SessionBankEntry":
        raw = bytes(entry.bank_relpath)
        path = raw.split(b"\0", 1)[0].decode()
        return cls(layer_index=int(entry.layer_index), bank_relpath=path)


def _load_library(path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    lib_path = Path(path) if path is not None else _DEFAULT_LIB_PATH
    lib = ctypes.CDLL(str(lib_path))

    void_pp = ctypes.POINTER(ctypes.c_void_p)

    lib.thump_rt_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.POINTER(_CBlockGeometry),
        void_pp,
    ]
    lib.thump_rt_create.restype = ctypes.c_int
    lib.thump_rt_attach.argtypes = [ctypes.c_char_p, void_pp]
    lib.thump_rt_attach.restype = ctypes.c_int
    lib.thump_rt_close.argtypes = [ctypes.c_void_p]
    lib.thump_rt_close.restype = None
    lib.thump_rt_get_geometry.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_CBlockGeometry),
    ]
    lib.thump_rt_get_geometry.restype = ctypes.c_int
    lib.thump_rt_total_bytes.argtypes = [ctypes.c_void_p]
    lib.thump_rt_total_bytes.restype = ctypes.c_uint64
    lib.thump_rt_sequence_id.argtypes = [ctypes.c_void_p]
    lib.thump_rt_sequence_id.restype = ctypes.c_uint64
    lib.thump_rt_set_sequence_id.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.thump_rt_set_sequence_id.restype = ctypes.c_int
    lib.thump_rt_get_session_metadata.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_CSessionMetadata),
    ]
    lib.thump_rt_get_session_metadata.restype = ctypes.c_int
    lib.thump_rt_set_session_metadata.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_CSessionMetadata),
    ]
    lib.thump_rt_set_session_metadata.restype = ctypes.c_int
    lib.thump_rt_validate_session_snapshot.argtypes = [ctypes.c_void_p]
    lib.thump_rt_validate_session_snapshot.restype = ctypes.c_int
    lib.thump_rt_write_session_manifest.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_CSessionManifest),
        ctypes.POINTER(_CSessionBankEntry),
    ]
    lib.thump_rt_write_session_manifest.restype = ctypes.c_int
    lib.thump_rt_read_session_manifest.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_CSessionManifest),
        ctypes.POINTER(_CSessionBankEntry),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.thump_rt_read_session_manifest.restype = ctypes.c_int
    lib.thump_rt_validate_session_manifest.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_CSessionManifest),
        ctypes.POINTER(_CSessionBankEntry),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.thump_rt_validate_session_manifest.restype = ctypes.c_int
    lib.thump_rt_block_capacity.argtypes = [ctypes.c_void_p]
    lib.thump_rt_block_capacity.restype = ctypes.c_uint32
    lib.thump_rt_sequence_length.argtypes = [ctypes.c_void_p]
    lib.thump_rt_sequence_length.restype = ctypes.c_uint32
    lib.thump_rt_sequence_version.argtypes = [ctypes.c_void_p]
    lib.thump_rt_sequence_version.restype = ctypes.c_uint32
    lib.thump_rt_block_elements.argtypes = [ctypes.c_void_p]
    lib.thump_rt_block_elements.restype = ctypes.c_uint32
    lib.thump_rt_alloc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.thump_rt_alloc.restype = ctypes.c_int
    lib.thump_rt_free_block.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.thump_rt_free_block.restype = ctypes.c_int
    lib.thump_rt_set_sequence.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    lib.thump_rt_set_sequence.restype = ctypes.c_int
    lib.thump_rt_splice_insert.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.thump_rt_splice_insert.restype = ctypes.c_int
    lib.thump_rt_splice_replace_equal_length.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.thump_rt_splice_replace_equal_length.restype = ctypes.c_int
    lib.thump_rt_splice_mixed.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.thump_rt_splice_mixed.restype = ctypes.c_int
    lib.thump_rt_write_blocks_fp16.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.thump_rt_write_blocks_fp16.restype = ctypes.c_int
    lib.thump_rt_materialize_range.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.thump_rt_materialize_range.restype = ctypes.c_int
    return lib


def _check(status: int, op: str) -> None:
    if status != 0:
        raise ThumpRuntimeError(f"{op} failed with status {status}")


def _as_u16(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint16:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(arr.astype(np.float16, copy=False).view(np.uint16))


def _manifest_round_trip(
    path: str | os.PathLike[str],
    *,
    lib_path: str | os.PathLike[str] | None = None,
    validate: bool,
) -> tuple[SessionManifest, list[SessionBankEntry]]:
    lib = _load_library(lib_path)
    manifest = _CSessionManifest()
    bank_count = ctypes.c_uint32(0)
    fn = (
        lib.thump_rt_validate_session_manifest
        if validate
        else lib.thump_rt_read_session_manifest
    )
    _check(
        fn(
            os.fsencode(path),
            ctypes.byref(manifest),
            None,
            0,
            ctypes.byref(bank_count),
        ),
        (
            "thump_rt_validate_session_manifest"
            if validate
            else "thump_rt_read_session_manifest"
        ),
    )
    count = int(bank_count.value)
    banks_array = (_CSessionBankEntry * count)() if count else None
    _check(
        fn(
            os.fsencode(path),
            ctypes.byref(manifest),
            banks_array,
            count,
            ctypes.byref(bank_count),
        ),
        (
            "thump_rt_validate_session_manifest"
            if validate
            else "thump_rt_read_session_manifest"
        ),
    )
    banks: list[SessionBankEntry] = []
    if banks_array is not None:
        banks = [SessionBankEntry.from_c(entry) for entry in banks_array]
    return SessionManifest.from_c(manifest), banks


def write_session_manifest(
    path: str | os.PathLike[str],
    manifest: SessionManifest,
    banks: list[SessionBankEntry],
    *,
    lib_path: str | os.PathLike[str] | None = None,
) -> None:
    lib = _load_library(lib_path)
    manifest_c = manifest.to_c(bank_count=len(banks))
    bank_array = (_CSessionBankEntry * len(banks))(*[bank.to_c() for bank in banks])
    _check(
        lib.thump_rt_write_session_manifest(
            os.fsencode(path),
            ctypes.byref(manifest_c),
            bank_array,
        ),
        "thump_rt_write_session_manifest",
    )


def read_session_manifest(
    path: str | os.PathLike[str],
    *,
    lib_path: str | os.PathLike[str] | None = None,
) -> tuple[SessionManifest, list[SessionBankEntry]]:
    return _manifest_round_trip(path, lib_path=lib_path, validate=False)


def validate_session_manifest(
    path: str | os.PathLike[str],
    *,
    lib_path: str | os.PathLike[str] | None = None,
) -> tuple[SessionManifest, list[SessionBankEntry]]:
    return _manifest_round_trip(path, lib_path=lib_path, validate=True)


class RuntimeHandle:
    """Python owner for one thump runtime bank."""

    def __init__(self, handle: ctypes.c_void_p, lib: ctypes.CDLL):
        self._handle = handle
        self._lib = lib

    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        block_count: int,
        geometry: BlockGeometry,
        *,
        lib_path: str | os.PathLike[str] | None = None,
    ) -> "RuntimeHandle":
        lib = _load_library(lib_path)
        handle = ctypes.c_void_p()
        geom = geometry.to_c()
        _check(
            lib.thump_rt_create(
                os.fsencode(path),
                block_count,
                ctypes.byref(geom),
                ctypes.byref(handle),
            ),
            "thump_rt_create",
        )
        return cls(handle, lib)

    @classmethod
    def attach(
        cls,
        path: str | os.PathLike[str],
        *,
        lib_path: str | os.PathLike[str] | None = None,
    ) -> "RuntimeHandle":
        lib = _load_library(lib_path)
        handle = ctypes.c_void_p()
        _check(
            lib.thump_rt_attach(
                os.fsencode(path),
                ctypes.byref(handle),
            ),
            "thump_rt_attach",
        )
        return cls(handle, lib)

    def close(self) -> None:
        if self._handle:
            self._lib.thump_rt_close(self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def geometry(self) -> BlockGeometry:
        geom = _CBlockGeometry()
        _check(
            self._lib.thump_rt_get_geometry(self._handle, ctypes.byref(geom)),
            "thump_rt_get_geometry",
        )
        return BlockGeometry.from_c(geom)

    @property
    def total_bytes(self) -> int:
        return int(self._lib.thump_rt_total_bytes(self._handle))

    @property
    def sequence_id(self) -> int:
        return int(self._lib.thump_rt_sequence_id(self._handle))

    @sequence_id.setter
    def sequence_id(self, value: int) -> None:
        _check(
            self._lib.thump_rt_set_sequence_id(self._handle, value),
            "thump_rt_set_sequence_id",
        )

    @property
    def block_capacity(self) -> int:
        return int(self._lib.thump_rt_block_capacity(self._handle))

    @property
    def sequence_length(self) -> int:
        return int(self._lib.thump_rt_sequence_length(self._handle))

    @property
    def sequence_version(self) -> int:
        return int(self._lib.thump_rt_sequence_version(self._handle))

    @property
    def block_elements(self) -> int:
        return int(self._lib.thump_rt_block_elements(self._handle))

    def get_session_metadata(self) -> SessionMetadata:
        meta = _CSessionMetadata()
        _check(
            self._lib.thump_rt_get_session_metadata(self._handle, ctypes.byref(meta)),
            "thump_rt_get_session_metadata",
        )
        return SessionMetadata.from_c(meta)

    def set_session_metadata(self, meta: SessionMetadata) -> None:
        meta_c = meta.to_c()
        _check(
            self._lib.thump_rt_set_session_metadata(self._handle, ctypes.byref(meta_c)),
            "thump_rt_set_session_metadata",
        )

    def validate_session_snapshot(self) -> None:
        _check(
            self._lib.thump_rt_validate_session_snapshot(self._handle),
            "thump_rt_validate_session_snapshot",
        )

    def alloc(self, count: int) -> np.ndarray:
        ids = np.zeros(count, dtype=np.uint32)
        _check(
            self._lib.thump_rt_alloc(
                self._handle,
                count,
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ),
            "thump_rt_alloc",
        )
        return ids

    def set_sequence(self, ids: np.ndarray) -> None:
        ids = np.ascontiguousarray(ids.astype(np.uint32, copy=False))
        _check(
            self._lib.thump_rt_set_sequence(
                self._handle,
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                ids.shape[0],
            ),
            "thump_rt_set_sequence",
        )

    def splice_insert(self, insert_at: int, count: int) -> np.ndarray:
        ids = np.zeros(count, dtype=np.uint32)
        _check(
            self._lib.thump_rt_splice_insert(
                self._handle,
                insert_at,
                count,
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ),
            "thump_rt_splice_insert",
        )
        return ids

    def splice_replace_equal_length(self, replace_at: int, count: int) -> np.ndarray:
        ids = np.zeros(count, dtype=np.uint32)
        _check(
            self._lib.thump_rt_splice_replace_equal_length(
                self._handle,
                replace_at,
                count,
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ),
            "thump_rt_splice_replace_equal_length",
        )
        return ids

    def write_blocks(
        self, block_ids: np.ndarray, k_fp16: np.ndarray, v_fp16: np.ndarray
    ) -> None:
        block_ids = np.ascontiguousarray(block_ids.astype(np.uint32, copy=False))
        k_bits = _as_u16(k_fp16)
        v_bits = _as_u16(v_fp16)
        _check(
            self._lib.thump_rt_write_blocks_fp16(
                self._handle,
                block_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                block_ids.shape[0],
                k_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                v_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ),
            "thump_rt_write_blocks_fp16",
        )

    def materialize_range(
        self, start_index: int, count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        elems = self.block_elements * count
        k_bits = np.zeros(elems, dtype=np.uint16)
        v_bits = np.zeros(elems, dtype=np.uint16)
        _check(
            self._lib.thump_rt_materialize_range(
                self._handle,
                start_index,
                count,
                k_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                v_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ),
            "thump_rt_materialize_range",
        )
        return k_bits.view(np.float16), v_bits.view(np.float16)
