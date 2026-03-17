//! Overdraw instrumentation test.
//!
//! Renders a scene via the tiled pipeline and reads back per-pixel debug stats
//! to measure ray miss / ghost / occluded / useful tet-pixel pair breakdown.
//!
//! Run with an .rmesh file:
//!   RMESH_PATH=/path/to/scene.rmesh cargo test -p rmesh-render --test overdraw_stats -- --nocapture
//!
//! Or without RMESH_PATH to use a synthetic multi-tet scene.

mod common;

use common::*;
use glam::Vec3;

const W: u32 = 256;
const H: u32 = 256;

fn setup_camera(eye: Vec3, target: Vec3) -> (glam::Mat4, glam::Mat3, [f32; 4]) {
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0);
    let view = look_at(eye, target, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    let (c2w, intrinsics) = test_camera_c2w_intrinsics(
        eye, target, std::f32::consts::FRAC_PI_2, W as f32, H as f32,
    );
    (vp, c2w, intrinsics)
}

const TILE_SIZE: u32 = 12;

fn print_stats(stats: &[[u32; 4]], w: u32, h: u32) {
    let pixel_count = (w * h) as usize;
    let mut total_miss: u64 = 0;
    let mut total_ghost: u64 = 0;
    let mut total_occluded: u64 = 0;
    let mut total_useful: u64 = 0;
    let mut per_pixel_total: Vec<u32> = Vec::with_capacity(pixel_count);
    let mut per_pixel_useful: Vec<u32> = Vec::with_capacity(pixel_count);
    let mut max_total: u32 = 0;

    for s in stats {
        total_miss += s[0] as u64;
        total_ghost += s[1] as u64;
        total_occluded += s[2] as u64;
        total_useful += s[3] as u64;
        let t = s[0] + s[1] + s[2] + s[3];
        per_pixel_total.push(t);
        per_pixel_useful.push(s[3]);
        max_total = max_total.max(t);
    }

    let grand_total = total_miss + total_ghost + total_occluded + total_useful;
    let pct = |v: u64| if grand_total > 0 { v as f64 / grand_total as f64 * 100.0 } else { 0.0 };

    println!("\n===== Overdraw Stats ({w}x{h} = {pixel_count} pixels) =====");
    println!("Total tet-pixel pairs: {grand_total}");
    println!("  Ray miss:   {:>12} ({:5.1}%)", total_miss, pct(total_miss));
    println!("  Ghost:      {:>12} ({:5.1}%)", total_ghost, pct(total_ghost));
    println!("  Degen slab: {:>12} ({:5.1}%)  (valid slab, t_min ≈ t_max)", total_occluded, pct(total_occluded));
    println!("  Useful:     {:>12} ({:5.1}%)", total_useful, pct(total_useful));

    // Per-pixel statistics
    per_pixel_total.sort_unstable();
    per_pixel_useful.sort_unstable();
    let non_zero: Vec<&u32> = per_pixel_total.iter().filter(|&&v| v > 0).collect();
    let non_zero_count = non_zero.len();

    if non_zero_count > 0 {
        let mean_total = grand_total as f64 / pixel_count as f64;
        let median_total = per_pixel_total[pixel_count / 2];
        let p95_total = per_pixel_total[(pixel_count as f64 * 0.95) as usize];
        let p99_total = per_pixel_total[(pixel_count as f64 * 0.99) as usize];

        println!("\nPer-pixel total (all categories):");
        println!("  Mean:   {mean_total:.1}");
        println!("  Median: {median_total}");
        println!("  P95:    {p95_total}");
        println!("  P99:    {p99_total}");
        println!("  Max:    {max_total}");
        println!("  Non-zero pixels: {non_zero_count} / {pixel_count}");

        // Histogram of per-pixel total counts
        let buckets = [0u32, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, u32::MAX];
        println!("\nHistogram (per-pixel total tet evaluations):");
        for i in 0..buckets.len() - 1 {
            let lo = buckets[i];
            let hi = buckets[i + 1];
            let count = per_pixel_total.iter().filter(|&&v| v >= lo && v < hi).count();
            if count > 0 {
                let label = if hi == u32::MAX {
                    format!("{lo}+")
                } else {
                    format!("{lo}-{}", hi - 1)
                };
                let bar_len = (count as f64 / pixel_count as f64 * 60.0) as usize;
                let bar: String = "#".repeat(bar_len);
                println!("  {label:>8}: {count:>6} ({:5.1}%) {bar}", count as f64 / pixel_count as f64 * 100.0);
            }
        }
    } else {
        println!("\nNo pixels had any tet evaluations.");
    }

    // --- Per-tile statistics ---
    let ts = TILE_SIZE;
    let tiles_x = (w + ts - 1) / ts;
    let tiles_y = (h + ts - 1) / ts;
    let num_tiles = (tiles_x * tiles_y) as usize;

    // Aggregate per-pixel stats into per-tile totals
    // Per tile: [miss, ghost, occluded, useful, total_pairs, max_pixel_total]
    let mut tile_stats: Vec<[u64; 6]> = vec![[0u64; 6]; num_tiles];
    for py in 0..h {
        for px in 0..w {
            let pixel_idx = (py * w + px) as usize;
            let s = &stats[pixel_idx];
            let tx = px / ts;
            let ty = py / ts;
            let tile_idx = (ty * tiles_x + tx) as usize;
            tile_stats[tile_idx][0] += s[0] as u64;
            tile_stats[tile_idx][1] += s[1] as u64;
            tile_stats[tile_idx][2] += s[2] as u64;
            tile_stats[tile_idx][3] += s[3] as u64;
            let pixel_total = (s[0] + s[1] + s[2] + s[3]) as u64;
            tile_stats[tile_idx][4] += pixel_total;
            tile_stats[tile_idx][5] = tile_stats[tile_idx][5].max(pixel_total);
        }
    }

    let mut per_tile_total: Vec<u64> = tile_stats.iter().map(|t| t[4]).collect();
    per_tile_total.sort_unstable();
    let non_zero_tiles = per_tile_total.iter().filter(|&&v| v > 0).count();

    println!("\n----- Per-tile Stats ({tiles_x}x{tiles_y} = {num_tiles} tiles, {ts}x{ts} px) -----");
    if non_zero_tiles > 0 {
        let tile_mean = grand_total as f64 / num_tiles as f64;
        let tile_median = per_tile_total[num_tiles / 2];
        let tile_p95 = per_tile_total[((num_tiles as f64) * 0.95) as usize];
        let tile_p99 = per_tile_total[((num_tiles as f64) * 0.99) as usize];
        let tile_max = per_tile_total[num_tiles - 1];

        println!("  Total pairs per tile:");
        println!("    Mean:   {tile_mean:.0}");
        println!("    Median: {tile_median}");
        println!("    P95:    {tile_p95}");
        println!("    P99:    {tile_p99}");
        println!("    Max:    {tile_max}");
        println!("    Non-zero tiles: {non_zero_tiles} / {num_tiles}");

        // Histogram of per-tile total pairs
        let buckets: [u64; 12] = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, u64::MAX];
        println!("\n  Histogram (per-tile total tet-pixel pairs):");
        for i in 0..buckets.len() - 1 {
            let lo = buckets[i];
            let hi = buckets[i + 1];
            let count = per_tile_total.iter().filter(|&&v| v >= lo && v < hi).count();
            if count > 0 {
                let label = if hi == u64::MAX {
                    format!("{}+", lo)
                } else {
                    format!("{lo}-{}", hi - 1)
                };
                let bar_len = (count as f64 / num_tiles as f64 * 50.0) as usize;
                let bar: String = "#".repeat(bar_len);
                println!("    {label:>12}: {count:>5} ({:5.1}%) {bar}", count as f64 / num_tiles as f64 * 100.0);
            }
        }

        // Category breakdown for hottest tiles (top 5)
        let mut indexed: Vec<(usize, &[u64; 6])> = tile_stats.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1[4].cmp(&a.1[4]));
        println!("\n  Top 10 hottest tiles:");
        println!("    {:>4} {:>4}  {:>8}  {:>7} {:>7} {:>7} {:>7}  {:>6}",
            "tx", "ty", "total", "miss", "ghost", "degen", "useful", "max_px");
        for &(tile_idx, ts_data) in indexed.iter().take(10) {
            let tx = tile_idx as u32 % tiles_x;
            let ty = tile_idx as u32 / tiles_x;
            println!("    {:>4} {:>4}  {:>8}  {:>7} {:>7} {:>7} {:>7}  {:>6}",
                tx, ty, ts_data[4], ts_data[0], ts_data[1], ts_data[2], ts_data[3], ts_data[5]);
        }
    } else {
        println!("  No tiles had any tet evaluations.");
    }
    println!("==========================================\n");
}

#[test]
fn overdraw_stats_synthetic() {
    // Use a multi-tet scene so we get meaningful overdraw
    let mut rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(42);
    let scene = random_single_tet_scene(&mut rng, 1.0);

    let eye = Vec3::new(2.0, 2.0, 2.0);
    let target = Vec3::ZERO;
    let (vp, c2w, intrinsics) = setup_camera(eye, target);

    let result = gpu_tiled_render_with_stats(&scene, eye, vp, c2w, intrinsics, W, H);
    let Some((image, stats)) = result else {
        println!("No GPU adapter available — skipping overdraw_stats_synthetic");
        return;
    };

    print_stats(&stats, W, H);

    // Sanity: useful count should be > 0 for pixels that have non-zero alpha
    let mut pixels_with_alpha = 0u32;
    let mut pixels_with_useful = 0u32;
    for (i, px) in image.iter().enumerate() {
        if px[3] > 0.001 {
            pixels_with_alpha += 1;
            if stats[i][3] > 0 {
                pixels_with_useful += 1;
            }
        }
    }
    if pixels_with_alpha > 0 {
        println!("Pixels with alpha > 0: {pixels_with_alpha}");
        println!("Of those, pixels with useful > 0: {pixels_with_useful}");
        assert!(
            pixels_with_useful > 0,
            "Bug: pixels have alpha but no useful tet evaluations"
        );
    }
}

#[test]
fn overdraw_stats_rmesh() {
    let path = match std::env::var("RMESH_PATH") {
        Ok(p) => p,
        Err(_) => {
            println!("RMESH_PATH not set — skipping overdraw_stats_rmesh");
            return;
        }
    };

    let data = std::fs::read(&path).expect("Failed to read .rmesh file");
    let (scene, _sh) = rmesh_data::load_rmesh(&data).expect("Failed to parse .rmesh file");
    println!("Loaded scene: {} verts, {} tets", scene.vertices.len() / 3, scene.tet_count);

    let eye = Vec3::new(0.0, -3.0, 1.5);
    let target = Vec3::ZERO;
    let (vp, c2w, intrinsics) = setup_camera(eye, target);

    let result = gpu_tiled_render_with_stats(&scene, eye, vp, c2w, intrinsics, W, H);
    let Some((_image, stats)) = result else {
        println!("No GPU adapter available — skipping overdraw_stats_rmesh");
        return;
    };

    print_stats(&stats, W, H);
}
