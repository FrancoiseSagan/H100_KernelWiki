---
name: h100-sm90a-kernel-wiki
description: Direct programming reference for NVIDIA H100 Hopper SM90A CUDA/PTX kernels. Use when writing, reviewing, or debugging code with sm_90a, TMA tensor maps, cp.async.bulk.tensor, tensormap.replace, WGMMA instruction operands, WGMMA matrix descriptors, WGMMA accumulator/register fragments, setmaxnreg, mbarrier, proxy fences, distributed shared memory, thread-block clusters, or warp-specialized Hopper kernels.
---

# H100 SM90A Kernel Programming

Sources checked: NVIDIA PTX ISA 9.3 and CUDA Driver API Tensor Memory material on 2026-07-08.

## Working Rules

- Prefer existing CUTLASS/CuTe wrappers when present. Drop to inline PTX only for missing coverage, reviewing generated PTX, or fixing Hopper-specific hazards.
- Treat `sm_90a` code as an H100 fast path, not a portable `compute_90` path.
- Distinguish three proxies: generic, async, and tensormap. Synchronization in one proxy does not automatically order accesses in another proxy.
- Do not emit illustrative PTX with brace alternatives such as `{.global|.shared::cta}` unless explicitly marking it as a syntax schema. Recipes below use concrete spellings.
- For any instruction with `.sync.aligned`, all required threads must execute the same instruction under uniform control flow.
- Exclude post-Hopper variants in H100 code: no TMA `.cta_group`, no `.tile::gather4`/`.tile::scatter4`, no `.im2col::w`, and no `.im2col::w::128`.

## Warp Specialization And setmaxnreg

- One WGMMA warpgroup is 4 contiguous warps. Use warpgroup-aligned partitions such as `0..3`, `4..7`.
- A producer warp that only issues TMA should release registers with `setmaxnreg.dec`; a WGMMA consumer warpgroup with large accumulators may request more with `setmaxnreg.inc`.
- Producers and consumers coordinate through shared-memory mbarriers or pipeline state. `__syncthreads()` alone does not wait for TMA or WGMMA.

`setmaxnreg`:

```ptx
// every warp in the warpgroup executes the same instruction
setmaxnreg.dec.sync.aligned.u32 64;
setmaxnreg.inc.sync.aligned.u32 192;
```

Constraints:

- Target `sm_90a`.
- Immediate register count is 24..256 and multiple of 8.
- `.inc` blocks until registers are available in the CTA pool; newly acquired registers are undefined.
- `.dec`/`.inc` are warpgroup-aligned. If only one warp in a warpgroup executes it, behavior is undefined.
- After a `setmaxnreg`, explicitly synchronize before another `setmaxnreg`.
- For one producer lane per warp, use `elect.sync leader|elected, 0xffffffff;` then guard the TMA issue with `@elected`.
- `elect.sync d|p, membermask` requires every executing lane to be in `membermask`; lanes in the mask wait until the same instruction executes.

## Inline PTX Operands

```cuda
static __device__ __forceinline__ uint32_t smem_u32(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
```

- Shared-memory operands to TMA and mbarrier PTX are 32-bit shared addresses when using `.shared::*` address operands.
- Tensor maps in global/param/const are generic addresses passed with a 64-bit operand.
- WGMMA descriptors are `.b64` register values, passed with an `l` constraint in inline asm.
- WGMMA immediate operands (`imm_scale_*`, `imm_trans_*`, `N` in `wait_group`) must be compile-time constants.
- `scale-d` is a predicate or immediate accepted by the opcode; use `0` when overwriting accumulators and `1` when accumulating.

## Target, Launch, Special Registers

- Compile WGMMA, `tensormap.replace`, and `setmaxnreg` paths with `-arch=sm_90a` or `-gencode arch=compute_90a,code=sm_90a`.
- `cp.async.bulk.tensor` requires `sm_90` or higher; `.multicast::cluster` is the H100-style fast path and should be used with `sm_90a`.
- Launch clusters with CUDA cluster attributes such as `__cluster_dims__` or `cudaLaunchKernelEx` cluster dimensions before relying on `%cluster_*`, DSM, multicast, or remote shared memory.
- `setmaxnreg` needs a valid compile-time register ceiling for the kernel; do not combine it with an accidental low `-maxrregcount`.
- Useful special registers: `%laneid`, `%warpid`, `%cluster_ctaid`, `%cluster_nctaid`, `%cluster_ctarank`, `%cluster_nctarank`, `%total_smem_size`, `%dynamic_smem_size`.

## Proxy And Barrier Model

Proxy ordering matrix:

| Producer | Consumer | Required ordering |
| --- | --- | --- |
| Generic SMEM stores | WGMMA reads SMEM | `fence.proxy.async.shared::cta` then a CTA or warpgroup-safe barrier |
| Generic SMEM stores | TMA store reads SMEM | `fence.proxy.async.shared::cta` then ensure all writers reached it |
| TMA GMEM to SMEM | Generic or WGMMA reads SMEM | Wait on mbarrier phase; WGMMA still needs `wgmma.fence` for register ordering |
| WGMMA writes D registers | Generic code reads D | `wgmma.commit_group` then `wgmma.wait_group` covering the group |
| Generic writes tensor map | TMA reads tensor map | `tensormap.cp_fenceproxy` or `fence.proxy.tensormap::generic.acquire` as appropriate |

TMA complete-tx does a release operation at cluster scope. A successful mbarrier wait is the point where consumers can rely on async completion. `bar.sync` is only a thread rendezvous.

## mbarrier State Machine

One shared mbarrier tracks one phase at a time.

```ptx
// one thread initializes, then all users rendezvous
mbarrier.init.shared::cta.b64 [bar], arrival_count;
bar.sync 0;

// producer for each phase
mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 state, [bar], tx_bytes;
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.tile
  [dst_smem], [tmap, {x, y}], [bar];

// consumers spin on current parity
wait:
mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [bar], parity;
@!done bra wait;
```

Rules:

- `arrival_count` is the number of explicit arrivals expected for the phase. TMA completion contributes transaction bytes, not thread arrivals.
- `expect_tx` adds expected transaction bytes for the phase. Match actual completion bytes or the phase may hang or complete incorrectly.
- Toggle software parity after each completed phase.
- For N-stage pipelines, keep one mbarrier and one parity bit per stage; the stage index is not the phase.
- Do not re-arm a stage until all consumers have left the previous phase for that stage.
- For remote cluster barriers, use `.shared::cluster` address operands and sink destination `_` when a return state is not available.
- Use `mbarrier.inval.shared::cta.b64 [bar]` before reusing the storage for other data.

Remote arrival pattern:

```ptx
// local shared address of CTA rank r
mapa.shared::cluster.u32 remote_bar, local_bar_smem_u32, r;
mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [remote_bar], tx_bytes;
```

## TMA Tensor Map Construction

The tensor map is a 1024-bit / 128-byte opaque object. CUDA names it `CUtensorMap`. PTX consumes it through the tensormap proxy.

Host encode for dense tiled copies:

```cuda
CUtensorMap map;
cuTensorMapEncodeTiled(&map, data_type, rank, global_address,
    global_dim, global_stride, box_dim, element_stride,
    interleave, swizzle, l2_promotion, oob_fill);
```

For row-major `T A[M][K]`, logical fastest dimension is K: `global_dim={K,M}`, `global_stride={K*sizeof(T)}`, `box_dim={tile_K,tile_M}`, `tensorCoords={k0,m0}`.

Field rules:

- `global_dim`, `box_dim`, and `element_stride` are element counts.
- `global_stride` is in bytes and excludes the fastest dimension.
- The first tensor coordinate is the fastest dimension.
- `CUtensorMap` storage is 64B aligned. `global_address` is at least 16B aligned; interleave or packed sub-byte types may require 32B.
- `global_stride` entries are multiples of 16B; `global_dim` entries are element counts and need not be 16B multiples.
- `box_dim[i]` and `element_stride[i]` are non-zero; tiled maps limit `box_dim[i] <= 256` and `element_stride[i] <= 8`. If `element_stride[i] != 1`, TMA loads `ceil(box_dim[i] / element_stride[i])`; set `box_dim[i] = logical_count * element_stride[i]`. With no interleave, `element_stride[0]` is ignored.
- With no interleave, `box_dim[0] * sizeof(element)` must be a multiple of 16B. Use 128B-aligned SMEM tile buffers when the tile will feed WGMMA or TMA swizzle.
- `swizzle` is the shared-memory layout that TMA writes. It must agree with the WGMMA descriptor layout when the tile feeds WGMMA.
- TMA tensor-map swizzle values differ from WGMMA descriptor swizzle values; use the cross-check table in the WGMMA descriptor section.
- H100 TMA swizzle modes are none/32B/64B/128B. Do not use 96B or `.swizzle_atomicity` for SM90A.
- For TMA swizzle, box dimension 0 in bytes is `<= swizzle_bytes`; allocate and index shared storage with the full swizzle span when the logical tile is smaller.

## Device-Side Tensor Map Mutation

Use this only for per-CTA descriptor changes that cannot be represented by coordinates.

Concrete shared-memory forms:

```ptx
tensormap.replace.tile.global_address.shared::cta.b1024.b64 [smap], addr64;
tensormap.replace.tile.rank.shared::cta.b1024.b32           [smap], rank_minus_1;
tensormap.replace.tile.global_dim.shared::cta.b1024.b32     [smap], 0, dim0;
tensormap.replace.tile.global_stride.shared::cta.b1024.b64  [smap], 0, stride0_bytes;
tensormap.replace.tile.box_dim.shared::cta.b1024.b32        [smap], 0, box0;
tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [smap], 0, elem_stride0;
tensormap.replace.tile.elemtype.shared::cta.b1024.b32       [smap], 6;
tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [smap], 0;
tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32   [smap], 3;
tensormap.replace.tile.fill_mode.shared::cta.b1024.b32      [smap], 0;
```

Concrete global-memory forms replace `.shared::cta` with `.global`.

Rules:

- `ord` is an immediate dimension ordinal in `0..4`.
- `.rank` stores `rank - 1`.
- `.global_address` and `.global_stride` are `.b64`; other fields are `.b32`.
- `.elemtype`, `.interleave_layout`, `.swizzle_mode`, and `.fill_mode` values must be immediates.
- `tensormap.replace` is a weak operation on the entire 128-byte map.
- The `.elemtype` immediates below are PTX field values, not CUDA Driver API enum values.

Important immediate values:

| Field | Values |
| --- | --- |
| `elemtype` | 0 `u8`, 1 `u16`, 2 `u32`, 3 `s32`, 4 `u64`, 5 `s64`, 6 `f16`, 7 `f32`, 8 `f32.ftz`, 9 `f64`, 10 `bf16`, 11 `tf32`, 12 `tf32.ftz` |
| `interleave_layout` | 0 none, 1 16B, 2 32B |
| `swizzle_mode` | 0 none, 1 32B, 2 64B, 3 128B |
| `fill_mode` | 0 zero, 1 OOB-NaN |

Safe shared-to-global publish:

```ptx
// Copy 128B map from shared to global and release generic writes to tensormap proxy.
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned
  [gmap], [smap], 128;

// Consumer before using gmap in cp.async.bulk.tensor:
fence.proxy.tensormap::generic.acquire.gpu [gmap], 128;
```

## TMA Load, Store, Prefetch, Reduce

Common exact forms:

```ptx
// global -> CTA/cluster shared
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.tile
  [dst_smem], [tmap, {d0, d1}], [bar];

// global -> cluster shared multicast
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
  [dst_cluster_smem], [tmap, {d0, d1}], [bar], ctaMask;

// CTA shared -> global
fence.proxy.async.shared::cta;
bar.sync 0;
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
  [tmap, {d0, d1}], [src_smem];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group.read 0;
```

Use `.shared::cluster.global` for cluster/multicast-capable SM90 TMA wrappers. The CTA-local `.shared::cta.global` variant is valid for non-multicast local loads. For multicast, the mbarrier operand names the same shared offset in every destination CTA, so every destination must initialize that offset before issue.

Notes:

- If `.load_mode` is absent, it defaults to `.tile`; include `.tile` for clarity.
- H100 tensor modes to use are `.tile` and base `.im2col`. Do not use `.tile::gather4`, `.tile::scatter4`, `.im2col::w`, `.im2col::w::128`, or `.cta_group::*` in SM90A code.
- `.im2col` requires at least 3 dimensions and carries an extra `im2colInfo` vector.
- `ctaMask` is 16 bits; bit position equals `%cluster_ctarank` of a destination CTA. Data lands at the same shared offset in each destination CTA.
- TMA store waits: `.read` only waits until the operation has read the tensor map and source memory. Use `cp.async.bulk.wait_group 0` when the issuing thread needs global writes visible to itself.
- `cp.async.bulk.prefetch.tensor` warms the tensor path without producing SMEM data. Use it for future tiles when descriptor and coordinates are known.
- `cp.reduce.async.bulk.tensor` performs shared-to-global tensor reductions. It uses bulk-group completion and has the same source-SMEM proxy fence need as TMA store.

Useful exact forms:

```ptx
cp.async.bulk.prefetch.tensor.2d.L2.global
  [tmap, {d0, d1}];

cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group
  [tmap, {d0, d1}], [src_smem];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group.read 0;
```

Byte-counted non-tensor bulk forms:

```ptx
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
  [dst_smem], [src_gmem], size, [bar];
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
  [dst_cluster_smem], [src_gmem], size, [bar], ctaMask;
cp.async.bulk.global.shared::cta.bulk_group
  [dst_gmem], [src_smem], size;
```

Use these for 16B-aligned byte copies without tensor maps or coordinates. Shared-to-global still needs `fence.proxy.async.shared::cta`, `cp.async.bulk.commit_group`, and a bulk wait before source reuse.

## WGMMA Instruction Forms

WGMMA requires `sm_90a`. B is always in shared memory and is passed as a 64-bit matrix descriptor. A is either a shared-memory matrix descriptor or a register fragment. D is an accumulator register tuple and may also be an input when `scale-d` is true.

Core sequence:

```ptx
// If normal stores produced shared A/B, first fence into async proxy.
fence.proxy.async.shared::cta;
bar.sync 0;

wgmma.fence.sync.aligned;
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
  {d0, ..., d63}, a_desc, b_desc,
  scale_d, 1, 1, imm_trans_a, imm_trans_b;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
```

Operand schemas. Instantiate `shape`, `dtype`, `atype`, and `btype` before emitting PTX; braces mark optional syntax, not literal text.

```ptx
// f16: dtype in {f16,f32}, N = 8..256 step 8
wgmma.mma_async.sync.aligned.shape.dtype.f16.f16 d, a-desc, b-desc,
  scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;
wgmma.mma_async.sync.aligned.shape.dtype.f16.f16 d, a, b-desc,
  scale-d, imm-scale-a, imm-scale-b, imm-trans-b;

// bf16: dtype is f32
wgmma.mma_async.sync.aligned.shape.f32.bf16.bf16 d, a-desc, b-desc,
  scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;
wgmma.mma_async.sync.aligned.shape.f32.bf16.bf16 d, a, b-desc,
  scale-d, imm-scale-a, imm-scale-b, imm-trans-b;

// tf32: dtype is f32, K = 8
wgmma.mma_async.sync.aligned.shape.f32.tf32.tf32 d, a-desc, b-desc,
  scale-d, imm-scale-a, imm-scale-b;
wgmma.mma_async.sync.aligned.shape.f32.tf32.tf32 d, a, b-desc,
  scale-d, imm-scale-a, imm-scale-b;

// fp8: atype/btype in {e4m3,e5m2}, dtype in {f16,f32}, K = 32
wgmma.mma_async.sync.aligned.shape.dtype.atype.btype d, a-desc_or_a, b-desc,
  scale-d, imm-scale-a, imm-scale-b;

// int8: atype/btype in {s8,u8}, dtype is s32, K = 32; emit .satfinite after btype
wgmma.mma_async.sync.aligned.shape.s32.atype.btype{.satfinite} d, a-desc_or_a, b-desc,
  scale-d;

// b1: K = 256, H100 WGMMA uses .and.popc
wgmma.mma_async.sync.aligned.shape.s32.b1.b1.and.popc d, a-desc_or_a, b-desc, scale-d;
```

Shape rules:

- Floating dense: `m=64`, `N = 8*i` for `i=1..32`.
- `.f16/.bf16`: `K=16`; `.tf32`: `K=8`; fp8/int8: `K=32`; b1: `K=256`.
- int8 `N`: `8,16,24,32` and `48..224` by 16. b1 `N`: `8,16,24,32` and `48..256` by 16.
- Sparse WGMMA uses `wgmma.mma_async.sp`; A is 50% structured sparse, packed to half K, with metadata operand and selector. Use official sparse forms, do not reuse dense A layouts.

Shared-memory major-ness:

- `imm-trans-a = 0` means A is K-major; `1` means A is M-major.
- `imm-trans-b = 0` means B is K-major; `1` means B is N-major.
- `imm-scale-a` and `imm-scale-b` are immediate `1` or `-1` where present; use `1, 1` unless deliberately negating inputs.
- Register-A f16/bf16 forms only take `imm-trans-b`; tf32/fp8/int8/b1 register-A forms take no transpose operands.

## WGMMA Shared Matrix Descriptor

The descriptor is a uniform `.b64` value.

| Bits | Field |
| --- | --- |
| 13:0 | `(matrix_start_addr & 0x3ffff) >> 4` |
| 29:16 | `(leading_byte_offset & 0x3ffff) >> 4` |
| 45:32 | `(stride_byte_offset & 0x3ffff) >> 4` |
| 51:49 | matrix base offset, ignored for no-swizzle |
| 63:62 | swizzle: 0 none, 1 128B, 2 64B, 3 32B |

Swizzle encoding cross-check:

| Swizzle bytes | TMA `swizzle_mode` | WGMMA bits 63:62 |
| --- | --- | --- |
| none | 0 | 0 |
| 32B | 1 | 3 |
| 64B | 2 | 2 |
| 128B | 3 | 1 |

```cuda
static __device__ __forceinline__ uint64_t wgmma_desc(
    void const* smem, uint32_t lbo, uint32_t sbo,
    uint32_t base_offset, uint32_t swizzle) {
  uint64_t a = (uint64_t)__cvta_generic_to_shared(smem);
  auto enc = [] __device__ (uint64_t x) { return (x & 0x3ffffull) >> 4; };
  return (enc(a) << 0) |
         (enc(lbo) << 16) |
         (enc(sbo) << 32) |
         ((uint64_t(base_offset) & 0x7ull) << 49) |
         ((uint64_t(swizzle) & 0x3ull) << 62);
}
```

Base offset:

```text
base_offset = 0 when swizzle pattern starts at:
  128B swizzle: 1024B boundary
  64B swizzle:   512B boundary
  32B swizzle:   256B boundary
otherwise: (pattern_start_addr >> 7) & 0x7
```

LBO/SBO rules:

- Shared matrix start addresses must be 16B aligned.
- K-major, no swizzle: LBO is offset from first column to second column of the 8x2 tile after 128-bit element normalization.
- K-major, swizzled: LBO is not used and is assumed 1.
- MN-major, no/interleaved: LBO is offset from first 8 columns to next 8 columns.
- MN-major, swizzled: LBO is offset from first `(swizzle_bytes/16)` rows to next group.
- SBO is offset from first 8 rows to next 8 rows for K-major; for MN no/interleaved it is first row to next row; for MN swizzled it is first 8 columns to next 8 columns.

Canonical layout hints with `T = 128 / element_bits`:

| Major | Swizzle | Layout hint |
| --- | --- | --- |
| MN | none/interleave | `((T,1,m),(8,k)):((1,T,SBO),(T,LBO))` |
| MN | 32B/64B/128B | `((T,S,m),(8,k)):((1,T,LBO),(S*T,SBO))`, `S=2/4/8` |
| K | none/interleave | `((8,m),(T,2k)):((T,SBO),(1,LBO))` |
| K | 32B/64B/128B | `((8,m),(T,2k)):((S*T,SBO),(1,T))`, `S=2/4/8` |

## WGMMA Register Fragments

Register fragment counts per thread:

| Shape family | Register A per thread | D per thread |
| --- | --- | --- |
| `m64nNk16` f16/bf16 register A | 4 `.f16x2` regs = 8 elements | f32: `N/2` regs; f16: `N/4` `.f16x2` regs |
| `m64nNk8` tf32 register A | 4 `.b32` regs = 4 elements | f32: `N/2` regs |
| `m64nNk32` int8/fp8 register A | 4 `.b32` regs = 16 elements | s32/f32: `N/2` regs; f16: `N/4` `.f16x2` regs |
| `m64nNk256` b1 register A | 4 `.b32` regs = 128 bits | s32: `N/2` regs |

Register-A fragment mapping, when A uses the register form:

```text
t = threadIdx.x % 128
warp = t / 32
lane = t % 32
row0 = 16*warp + lane/4
row1 = row0 + 8

m64nNk16 f16/bf16, 4 .f16x2 regs:
  c = 2*(lane % 4)
  a[0].lo/hi -> A[row0, c + 0/1]
  a[1].lo/hi -> A[row1, c + 0/1]
  a[2].lo/hi -> A[row0, c + 8/9]
  a[3].lo/hi -> A[row1, c + 8/9]

m64nNk8 tf32, 4 .b32 regs:
  c = lane % 4
  a[0] -> A[row0, c];     a[1] -> A[row1, c]
  a[2] -> A[row0, c + 4]; a[3] -> A[row1, c + 4]

m64nNk32 int8/fp8, 4 .b32 regs packed as 4 elements each:
  c = 4*(lane % 4)
  a[0] -> A[row0, c..c+3];      a[1] -> A[row1, c..c+3]
  a[2] -> A[row0, c+16..c+19];  a[3] -> A[row1, c+16..c+19]

m64nNk256 b1, 4 .b32 regs packed as 32 bits each:
  c = 32*(lane % 4)
  a[0] -> A[row0, c..c+31];       a[1] -> A[row1, c..c+31]
  a[2] -> A[row0, c+128..c+159];  a[3] -> A[row1, c+128..c+159]
```

For `.f32` and `.s32` D, the PTX D fragment figures follow this practical mapping for each 8-column group:

```text
t = threadIdx.x % 128
warp = t / 32
lane = t % 32
row0 = 16*warp + lane/4
row1 = row0 + 8
base_col = 2*(lane % 4)

for g in 0 .. N/8-1:
  d[4*g + 0] -> D[row0, 8*g + base_col + 0]
  d[4*g + 1] -> D[row0, 8*g + base_col + 1]
  d[4*g + 2] -> D[row1, 8*g + base_col + 0]
  d[4*g + 3] -> D[row1, 8*g + base_col + 1]
```

For `.f16` D, each `.f16x2` packs two adjacent D elements; use the same row/column grouping and split halves carefully during epilogue. Example D counts for f32 accumulators: `m64n8` -> 4 regs, `m64n64` -> 32, `m64n128` -> 64, `m64n256` -> 128.

## WGMMA Ordering Hazards

- `wgmma.fence.sync.aligned` is required before the first WGMMA in a warpgroup.
- Issue another `wgmma.fence` between ordinary register access and later WGMMA access to the same D or register-A operand, except accumulator reuse across multiple WGMMA instructions of the same shape.
- `wgmma.fence` does not replace `fence.proxy.async.shared::cta`.
- `wgmma.commit_group.sync.aligned` batches prior uncommitted WGMMA ops into a per-warpgroup group.
- `wgmma.wait_group.sync.aligned N` waits until only `N` newest groups remain pending. `0` waits all prior groups.
- Reading or overwriting D or register-A before a wait covering that group is undefined.
- ptxas may warn about WGMMA serialization if dependency analysis is unclear. Use explicit fences/waits and keep accumulator lifetimes simple.

## Cluster And DSM

Addressing:

```ptx
// Convert a local shared address to same shared offset in CTA rank r.
mapa.shared::cluster.u32 remote_smem, local_smem, r;
getctarank.shared::cluster.u32 owner, remote_smem;
st.shared::cluster.u32 [remote_smem], val;
ld.shared::cluster.u32 val, [remote_smem];
```

Cluster synchronization:

```ptx
barrier.cluster.arrive.aligned;
barrier.cluster.wait.aligned;
```

Rules:

- Remote CTA shared memory is valid only while that CTA exists in the cluster.
- `barrier.cluster.wait` orders normal memory accesses around the cluster barrier; it does not wait for TMA or WGMMA async completion.
- Use a cluster barrier before any CTA exits if peers may still read or write its DSM.
- Remote generic DSM stores should be followed by cluster synchronization before peer generic loads rely on them.
- For TMA multicast, every destination CTA must have initialized its mbarrier before the source CTA issues multicast.
- `ctaMask` bits correspond to destination `%cluster_ctarank`.
- Do not use TMA `.cta_group::*` in SM90A code; it is a post-Hopper qualifier.
- For async-proxy data consumed through generic loads after cluster communication, use the documented acquire fence sequence at cluster scope, not only `barrier.cluster`.

## Final Review Checklist

- Compile target is `sm_90a` for WGMMA, `tensormap.replace`, or `setmaxnreg`.
- Warpgroup boundaries are correct and all `.aligned` instructions are uniform.
- TMA tensor coordinates are fastest-dimension-first; global strides are bytes.
- Tensor map publish/acquire uses tensormap proxy fences.
- TMA load expected bytes match actual copied bytes and phase parity is advanced exactly once.
- TMA store/reduce has generic-to-async shared proxy fence before issue and bulk wait before source reuse.
- TMA and WGMMA swizzle numeric encodings are not confused.
- WGMMA descriptor start/LBO/SBO are 16B aligned byte values encoded with `>> 4`.
- WGMMA D register tuple length matches shape and dtype.
- D/register-A are not read or overwritten before the required `wgmma.wait_group`.
- Cluster code uses `mapa`, CTA ranks, masks, and cluster barriers deliberately; no pointer arithmetic guesses peer shared addresses.
