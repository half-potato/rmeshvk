import gzip
import struct
import sys
import numpy as np

TAGGED_MAGIC = 0x524D5458


def read_u32(raw, off):
    return struct.unpack_from("<I", raw, off)[0], off + 4


def check_rmesh(path):
    print(f"Reading: {path}")
    with open(path, "rb") as f:
        compressed = f.read()
    print(f"Compressed size: {len(compressed) / 1e6:.1f} MB")

    raw = gzip.decompress(compressed)
    total = len(raw)
    print(f"Decompressed size: {total / 1e6:.1f} MB")

    off = 0

    # Header
    N, M, deg, k = struct.unpack_from("<4I", raw, off); off += 16
    print(f"\n=== Header ===")
    print(f"  vertices:     {N:,}")
    print(f"  tets:         {M:,}")
    print(f"  sh_degree:    {deg}")
    print(f"  k_components: {k}")

    # Start pose
    pose = struct.unpack_from("<8f", raw, off); off += 32
    print(f"\n=== Start Pose ===")
    print(f"  position: ({pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f})")
    print(f"  quaternion: ({pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f}, {pose[6]:.4f})")

    # Vertices [N x 3] f32
    verts_bytes = N * 3 * 4
    print(f"\n=== Vertices ===")
    print(f"  offset: {off:,}  size: {verts_bytes:,} bytes ({verts_bytes/1e6:.1f} MB)")
    if off + 12 <= total:
        s = np.frombuffer(raw[off:off+12], dtype=np.float32)
        print(f"  first: ({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f})")
    off += verts_bytes

    # Indices [M x 4] u32
    idx_bytes = M * 4 * 4
    print(f"\n=== Indices ===")
    print(f"  offset: {off:,}  size: {idx_bytes:,} bytes ({idx_bytes/1e6:.1f} MB)")
    if off + 16 <= total:
        s = np.frombuffer(raw[off:off+16], dtype=np.uint32)
        print(f"  first tet: ({s[0]}, {s[1]}, {s[2]}, {s[3]})")
        max_idx = np.frombuffer(raw[off:off+idx_bytes], dtype=np.uint32).max()
        print(f"  max index: {max_idx} (vertices: {N})")
    off += idx_bytes

    # Densities [M] u8
    print(f"\n=== Densities ===")
    print(f"  offset: {off:,}  size: {M:,} bytes ({M/1e6:.1f} MB)")
    if off + 4 <= total:
        s = list(raw[off:off+8])
        print(f"  first 8 u8: {s}")
    off += M

    # Align
    off = (off + 3) & ~3

    remaining = total - off
    print(f"\n  after densities+align: offset={off:,}, remaining={remaining:,} bytes ({remaining/1e6:.1f} MB)")

    # Check if SH data exists (k > 0), or if tagged magic is here
    has_sh = k > 0
    magic_here = (off + 4 <= total and struct.unpack_from("<I", raw, off)[0] == TAGGED_MAGIC)

    if magic_here:
        print("\n=== Tagged magic found immediately after densities (no SH, no color_grads) ===")
        parse_tagged_sections(raw, off, total)
        return

    if has_sh:
        num_coeffs = (deg + 1) ** 2
        total_dims = num_coeffs * 3
        mean_bytes = total_dims * 2
        basis_bytes = k * total_dims * 2
        weights_bytes = M * k * 2
        sh_total = mean_bytes + basis_bytes + weights_bytes

        print(f"\n=== SH (PCA compressed) ===")
        print(f"  num_coeffs: {num_coeffs}  total_dims: {total_dims}")
        print(f"  mean:    offset={off:,}  size={mean_bytes:,} bytes")
        if off + min(20, mean_bytes) <= total:
            s = np.frombuffer(raw[off:off+min(20, mean_bytes)], dtype=np.float16)
            print(f"           first: {s}")
        off += mean_bytes

        print(f"  basis:   offset={off:,}  size={basis_bytes:,} bytes")
        off += basis_bytes

        print(f"  weights: offset={off:,}  size={weights_bytes:,} bytes ({weights_bytes/1e6:.1f} MB)")
        if off + min(20, weights_bytes) <= total:
            s = np.frombuffer(raw[off:off+min(20, weights_bytes)], dtype=np.float16)
            print(f"           first: {s}")
        off += weights_bytes

        off = (off + 3) & ~3
        remaining = total - off
        print(f"\n  after SH+align: offset={off:,}, remaining={remaining:,} bytes ({remaining/1e6:.1f} MB)")
    else:
        print(f"\n=== No SH data (degree={deg}, k={k}) ===")

    # Check for tagged magic (PBR sections before color_grads)
    magic_here = (off + 4 <= total and struct.unpack_from("<I", raw, off)[0] == TAGGED_MAGIC)
    if magic_here:
        print("\n=== Tagged magic found (no color_grads) ===")
        parse_tagged_sections(raw, off, total)
        return

    # Color grads [M x 3] f16
    grads_bytes = M * 3 * 2
    print(f"\n=== Color Grads ===")
    print(f"  expected: {grads_bytes:,} bytes ({grads_bytes/1e6:.1f} MB)")
    if off + grads_bytes <= total:
        print(f"  offset: {off:,}  size: {grads_bytes:,} bytes — OK")
        if off + 12 <= total:
            s = np.frombuffer(raw[off:off+12], dtype=np.float16)
            print(f"  first 2 triplets: {s}")
        off += grads_bytes
    else:
        avail = total - off
        full_triplets = avail // 6
        print(f"  *** PARTIAL: only {avail:,} bytes available (short by {grads_bytes - avail:,})")
        print(f"  *** {full_triplets:,} complete (r,g,b) triplets out of {M:,} tets")
        if off + 12 <= total:
            s = np.frombuffer(raw[off:off+12], dtype=np.float16)
            print(f"  first values: {s}")
        # Scan for tagged magic within this region
        magic_bytes = struct.pack("<I", TAGGED_MAGIC)
        search_start = off
        pos = raw[search_start:].find(magic_bytes)
        if pos >= 0 and pos % 4 == 0:
            abs_pos = search_start + pos
            grads_actual = pos
            print(f"\n  *** Found TAGGED_MAGIC at offset {abs_pos:,} ({grads_actual:,} bytes of color_grads)")
            print(f"  *** That's {grads_actual // 6:,} complete triplets")
            off = abs_pos
        else:
            off += avail

    # Align
    off = (off + 3) & ~3
    remaining = total - off
    print(f"\n  after color_grads+align: offset={off:,}, remaining={remaining:,} bytes")

    if remaining == 0:
        print("\n=== End of file (no tagged extension sections) ===")
        return

    # Check for tagged magic
    magic_here = (off + 4 <= total and struct.unpack_from("<I", raw, off)[0] == TAGGED_MAGIC)
    if magic_here:
        parse_tagged_sections(raw, off, total)
    else:
        val = struct.unpack_from("<I", raw, off)[0]
        print(f"\n=== Unknown trailing data ===")
        print(f"  next u32: 0x{val:08X} (not TAGGED_MAGIC 0x{TAGGED_MAGIC:08X})")
        # Brute-force search
        magic_bytes = struct.pack("<I", TAGGED_MAGIC)
        pos = raw[off:].find(magic_bytes)
        if pos >= 0:
            print(f"  Found TAGGED_MAGIC at offset {off + pos:,} ({pos:,} bytes into remaining)")
        else:
            print(f"  TAGGED_MAGIC not found anywhere in remaining {remaining:,} bytes")
            # Dump hex of first and last bytes
            print(f"  First 32 bytes: {raw[off:off+32].hex()}")
            if remaining > 64:
                print(f"  Last 32 bytes:  {raw[total-32:total].hex()}")


def parse_tagged_sections(raw, off, total):
    off += 4  # skip magic
    num_sections, off = read_u32(raw, off)
    print(f"\n=== Tagged Extension Sections: {num_sections} ===")

    for i in range(num_sections):
        if off + 16 > total:
            print(f"  *** Truncated at section {i} header")
            break

        tag = raw[off:off+16].decode("ascii", errors="replace").rstrip("\x00")
        off += 16

        dtype, off = read_u32(raw, off)
        shape_rank, off = read_u32(raw, off)
        shape = []
        for _ in range(shape_rank):
            dim, off = read_u32(raw, off)
            shape.append(dim)
        data_bytes, off = read_u32(raw, off)

        dtype_names = {0: "f16", 1: "f32", 2: "u8", 3: "i32"}
        dtype_name = dtype_names.get(dtype, f"unknown({dtype})")

        print(f"\n  [{i}] '{tag}'")
        print(f"      dtype: {dtype_name}  shape: {shape}  payload: {data_bytes:,} bytes")

        if off + data_bytes > total:
            print(f"      *** TRUNCATED (need {data_bytes:,}, have {total - off:,})")
            break

        payload = raw[off:off+data_bytes]

        # Show sample + stats
        if dtype == 0 and data_bytes >= 2:
            arr = np.frombuffer(payload, dtype=np.float16).astype(np.float32)
            print(f"      count: {len(arr):,}  min: {arr.min():.4f}  max: {arr.max():.4f}  mean: {arr.mean():.4f}")
            print(f"      first: {arr[:min(8, len(arr))]}")
        elif dtype == 1 and data_bytes >= 4:
            arr = np.frombuffer(payload, dtype=np.float32)
            print(f"      count: {len(arr):,}  min: {arr.min():.4f}  max: {arr.max():.4f}  mean: {arr.mean():.4f}")
            print(f"      first: {arr[:min(8, len(arr))]}")
        elif dtype == 2 and data_bytes >= 1:
            arr = np.frombuffer(payload, dtype=np.uint8)
            print(f"      count: {len(arr):,}  min: {arr.min()}  max: {arr.max()}")
        elif tag == "brdf_mlp":
            parse_mlp_section(payload)
        else:
            print(f"      (raw, {data_bytes} bytes)")

        off += data_bytes
        off = (off + 3) & ~3  # align

    remaining = total - off
    print(f"\n=== End of tagged sections: offset={off:,}, remaining={remaining:,} bytes ===")


def parse_mlp_section(payload):
    """Parse and display BRDF MLP layer info."""
    off = 0
    num_layers = struct.unpack_from("<I", payload, off)[0]; off += 4
    print(f"      MLP layers: {num_layers}")
    for j in range(num_layers):
        if off + 9 > len(payload):
            print(f"        *** truncated at layer {j}")
            break
        in_dim = struct.unpack_from("<I", payload, off)[0]; off += 4
        out_dim = struct.unpack_from("<I", payload, off)[0]; off += 4
        has_bias = payload[off]; off += 1
        w_bytes = out_dim * in_dim * 2
        print(f"        layer {j}: {in_dim} -> {out_dim}  bias={bool(has_bias)}  weights={w_bytes} bytes")
        off += w_bytes
        if has_bias:
            b_bytes = out_dim * 2
            off += b_bytes


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.rmesh>")
        sys.exit(1)
    check_rmesh(sys.argv[1])
