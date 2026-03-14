// Sovereign ARC grid operations (rotate/flip/translate/recolor) on GPU.
//
// Input and output grids are uint8 color indices (0-255).
// Supported ops:
//   0: rotate 90° CW
//   1: rotate 180°
//   2: rotate 270° CW
//   3: flip horizontally
//   4: flip vertically
//   5: translate (p1=dx, p2=dy, fill=0)
//   6: recolor (p1=src, p2=dst)
//   7: periodic tile repeat
//   8: checker tile repeat with horizontal flip on odd tile rows
//   9: connect same-color pairs with straight lines
//  10: periodic consensus cleanup
//  11: fill enclosed regions based on border size
//  12: pack color components diagonally
//  13: self-pattern complement tiling
//  14: marker axis crop
//  15: separator bridge projection
//  16: anchor spiral pair

#include <cuda_runtime.h>

#define ARC_MAX_GRID_CELLS 1024
#define ARC_MAX_COMPONENTS 128

extern "C" __global__
void arc_grid_op(const unsigned char* input,
                 unsigned char* output,
                 int src_w,
                 int src_h,
                 int dst_w,
                 int dst_h,
                 int op,
                 int p1,
                 int p2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) {
        return;
    }

    int src_x = x;
    int src_y = y;

    switch (op) {
        case 0: { // rotate 90 CW
            // dst_w = src_h, dst_h = src_w
            src_x = y;
            src_y = src_h - 1 - x;
            break;
        }
        case 1: { // rotate 180
            src_x = src_w - 1 - x;
            src_y = src_h - 1 - y;
            break;
        }
        case 2: { // rotate 270 CW (90 CCW)
            // dst_w = src_h, dst_h = src_w
            src_x = src_w - 1 - y;
            src_y = x;
            break;
        }
        case 3: { // flip horizontal
            src_x = src_w - 1 - x;
            src_y = y;
            break;
        }
        case 4: { // flip vertical
            src_x = x;
            src_y = src_h - 1 - y;
            break;
        }
        case 5: { // translate (dx=p1, dy=p2)
            src_x = x - p1;
            src_y = y - p2;
            break;
        }
        case 6: { // recolor (src=p1, dst=p2)
            int idx = y * src_w + x;
            unsigned char v = input[idx];
            output[y * dst_w + x] = (v == (unsigned char)p1) ? (unsigned char)p2 : v;
            return;
        }
        case 7: { // periodic tile repeat
            src_x = x % src_w;
            src_y = y % src_h;
            break;
        }
        case 8: { // checker tile repeat with horizontal flip on odd tile rows
            int tile_row = y / src_h;
            src_y = y % src_h;
            int local_x = x % src_w;
            src_x = (tile_row & 1) ? (src_w - 1 - local_x) : local_x;
            break;
        }
        case 9: { // connect same-color pairs with straight lines
            unsigned char result = input[y * src_w + x];

            for (int color = 1; color <= 9; ++color) {
                int count = 0;
                int x0 = -1, y0 = -1, x1 = -1, y1 = -1;
                for (int yy = 0; yy < src_h; ++yy) {
                    for (int xx = 0; xx < src_w; ++xx) {
                        unsigned char v = input[yy * src_w + xx];
                        if (v != (unsigned char)color) {
                            continue;
                        }
                        if (count == 0) {
                            x0 = xx;
                            y0 = yy;
                        } else if (count == 1) {
                            x1 = xx;
                            y1 = yy;
                        }
                        count += 1;
                    }
                }
                if (count == 2 && y0 == y1 && y == y0) {
                    int left = x0 < x1 ? x0 : x1;
                    int right = x0 > x1 ? x0 : x1;
                    if (x >= left && x <= right && (result == 0 || result == (unsigned char)color)) {
                        result = (unsigned char)color;
                    }
                }
            }

            for (int color = 1; color <= 9; ++color) {
                int count = 0;
                int x0 = -1, y0 = -1, x1 = -1, y1 = -1;
                for (int yy = 0; yy < src_h; ++yy) {
                    for (int xx = 0; xx < src_w; ++xx) {
                        unsigned char v = input[yy * src_w + xx];
                        if (v != (unsigned char)color) {
                            continue;
                        }
                        if (count == 0) {
                            x0 = xx;
                            y0 = yy;
                        } else if (count == 1) {
                            x1 = xx;
                            y1 = yy;
                        }
                        count += 1;
                    }
                }
                if (count == 2 && x0 == x1 && x == x0) {
                    int top = y0 < y1 ? y0 : y1;
                    int bottom = y0 > y1 ? y0 : y1;
                    if (y >= top && y <= bottom) {
                        result = (unsigned char)color;
                    }
                }
            }

            output[y * dst_w + x] = result;
            return;
        }
        case 10: { // periodic consensus cleanup
            int best_px = 0;
            int best_px_score = -2147483647;
            for (int candidate = 2; candidate < src_w; ++candidate) {
                int full_w = (src_w / candidate) * candidate;
                if (full_w < candidate * 2) {
                    continue;
                }
                int score = 0;
                for (int yy = 0; yy < src_h; ++yy) {
                    for (int xx = 0; xx + candidate < full_w; ++xx) {
                        unsigned char a = input[yy * src_w + xx];
                        unsigned char b = input[yy * src_w + xx + candidate];
                        if (a == 0 && b == 0) {
                            continue;
                        }
                        if (a != 0 && b != 0 && a == b) {
                            score += 3;
                        } else if (a != 0 && b != 0) {
                            score -= 2;
                        } else {
                            score -= 1;
                        }
                    }
                }
                if (score > best_px_score) {
                    best_px_score = score;
                    best_px = candidate;
                }
            }

            int best_py = 0;
            int best_py_score = -2147483647;
            for (int candidate = 2; candidate < src_h; ++candidate) {
                int full_h = (src_h / candidate) * candidate;
                if (full_h < candidate * 2) {
                    continue;
                }
                int score = 0;
                for (int yy = 0; yy + candidate < full_h; ++yy) {
                    for (int xx = 0; xx < src_w; ++xx) {
                        unsigned char a = input[yy * src_w + xx];
                        unsigned char b = input[(yy + candidate) * src_w + xx];
                        if (a == 0 && b == 0) {
                            continue;
                        }
                        if (a != 0 && b != 0 && a == b) {
                            score += 3;
                        } else if (a != 0 && b != 0) {
                            score -= 2;
                        } else {
                            score -= 1;
                        }
                    }
                }
                if (score > best_py_score) {
                    best_py_score = score;
                    best_py = candidate;
                }
            }

            if (best_px <= 0 || best_py <= 0) {
                output[y * dst_w + x] = input[y * src_w + x];
                return;
            }

            int full_w = (src_w / best_px) * best_px;
            int full_h = (src_h / best_py) * best_py;
            if (x >= full_w || y >= full_h) {
                output[y * dst_w + x] = 0;
                return;
            }

            int counts[10];
            for (int i = 0; i < 10; ++i) {
                counts[i] = 0;
            }
            int nonzero_samples = 0;
            int phase_x = x % best_px;
            int phase_y = y % best_py;
            for (int yy = phase_y; yy < full_h; yy += best_py) {
                for (int xx = phase_x; xx < full_w; xx += best_px) {
                    unsigned char v = input[yy * src_w + xx];
                    if (v <= 9) {
                        counts[(int)v] += 1;
                        if (v != 0) {
                            nonzero_samples += 1;
                        }
                    }
                }
            }

            int best_color = 0;
            int best_count = counts[0];
            for (int color = 1; color <= 9; ++color) {
                if (counts[color] > best_count || (counts[color] == best_count && best_color == 0)) {
                    best_color = color;
                    best_count = counts[color];
                }
            }
            output[y * dst_w + x] = (unsigned char)((best_color != 0 && best_count >= 2 && nonzero_samples >= 2) ? best_color : 0);
            return;
        }
        case 11: { // fill enclosed regions based on border size
            unsigned char current = input[y * src_w + x];
            if (current != 0) {
                output[y * dst_w + x] = current;
                return;
            }

            int best_area = 1 << 30;
            unsigned char fill = 0;
            for (int top = 0; top < y; ++top) {
                if (input[top * src_w + x] != 2) {
                    continue;
                }
                for (int bottom = y + 1; bottom < src_h; ++bottom) {
                    if (input[bottom * src_w + x] != 2) {
                        continue;
                    }
                    for (int left = 0; left < x; ++left) {
                        if (input[y * src_w + left] != 2) {
                            continue;
                        }
                        for (int right = x + 1; right < src_w; ++right) {
                            if (input[y * src_w + right] != 2) {
                                continue;
                            }
                            bool valid = true;
                            for (int xx = left; xx <= right; ++xx) {
                                if (input[top * src_w + xx] != 2 || input[bottom * src_w + xx] != 2) {
                                    valid = false;
                                    break;
                                }
                            }
                            if (!valid) {
                                continue;
                            }
                            for (int yy = top; yy <= bottom; ++yy) {
                                if (input[yy * src_w + left] != 2 || input[yy * src_w + right] != 2) {
                                    valid = false;
                                    break;
                                }
                            }
                            if (!valid) {
                                continue;
                            }
                            int area = (right - left + 1) * (bottom - top + 1);
                            if (area < best_area) {
                                best_area = area;
                                int inner_w = right - left - 1;
                                int inner_h = bottom - top - 1;
                                int scale = inner_w > inner_h ? inner_w : inner_h;
                                fill = (unsigned char)((scale <= 3) ? 8 : (scale <= 5) ? 4 : 3);
                            }
                        }
                    }
                }
            }
            output[y * dst_w + x] = fill;
            return;
        }
        case 12: { // pack color components diagonally
            int found_count = 0;
            int colors[9];
            int min_x[10];
            int min_y[10];
            int max_x[10];
            int max_y[10];
            int start_x[9];
            int start_y[9];
            int box_w[9];
            int box_h[9];

            for (int color = 0; color <= 9; ++color) {
                min_x[color] = src_w;
                min_y[color] = src_h;
                max_x[color] = -1;
                max_y[color] = -1;
            }

            for (int yy = 0; yy < src_h; ++yy) {
                for (int xx = 0; xx < src_w; ++xx) {
                    unsigned char v = input[yy * src_w + xx];
                    if (v == 0 || v > 9) {
                        continue;
                    }
                    int color = (int)v;
                    if (xx < min_x[color]) {
                        min_x[color] = xx;
                    }
                    if (yy < min_y[color]) {
                        min_y[color] = yy;
                    }
                    if (xx > max_x[color]) {
                        max_x[color] = xx;
                    }
                    if (yy > max_y[color]) {
                        max_y[color] = yy;
                    }
                }
            }

            for (int color = 1; color <= 9; ++color) {
                if (max_x[color] < 0 || max_y[color] < 0) {
                    continue;
                }
                int insert_at = found_count;
                while (insert_at > 0) {
                    int prev_color = colors[insert_at - 1];
                    bool should_swap = (min_x[color] < min_x[prev_color]) ||
                        (min_x[color] == min_x[prev_color] && min_y[color] < min_y[prev_color]) ||
                        (min_x[color] == min_x[prev_color] && min_y[color] == min_y[prev_color] && color < prev_color);
                    if (!should_swap) {
                        break;
                    }
                    colors[insert_at] = colors[insert_at - 1];
                    insert_at -= 1;
                }
                colors[insert_at] = color;
                found_count += 1;
            }

            for (int idx = 0; idx < found_count; ++idx) {
                int color = colors[idx];
                box_w[idx] = max_x[color] - min_x[color] + 1;
                box_h[idx] = max_y[color] - min_y[color] + 1;
                if (idx == 0) {
                    start_x[idx] = 0;
                    start_y[idx] = 0;
                } else {
                    start_x[idx] = start_x[idx - 1] + box_w[idx - 1] - 1;
                    start_y[idx] = start_y[idx - 1] + box_h[idx - 1] - 1;
                }
            }

            unsigned char result = 0;
            for (int idx = 0; idx < found_count; ++idx) {
                int color = colors[idx];
                int local_x = x - start_x[idx];
                int local_y = y - start_y[idx];
                if (local_x < 0 || local_y < 0 || local_x >= box_w[idx] || local_y >= box_h[idx]) {
                    continue;
                }
                unsigned char sampled = input[(min_y[color] + local_y) * src_w + (min_x[color] + local_x)];
                if (sampled != 0) {
                    result = sampled;
                }
            }
            output[y * dst_w + x] = result;
            return;
        }
        case 13: { // self-pattern complement tiling
            if (src_w <= 0 || src_h <= 0) {
                output[y * dst_w + x] = 0;
                return;
            }
            int block_x = x / src_w;
            int block_y = y / src_h;
            if (block_x >= src_w || block_y >= src_h) {
                output[y * dst_w + x] = 0;
                return;
            }
            unsigned char anchor = input[block_y * src_w + block_x];
            unsigned char pattern = input[(y % src_h) * src_w + (x % src_w)];
            output[y * dst_w + x] = (unsigned char)((anchor != 0 && pattern == 0) ? anchor : 0);
            return;
        }
        case 14: { // marker axis crop
            if (x != 0 || y != 0) {
                return;
            }
            for (int i = 0; i < dst_w * dst_h; ++i) {
                output[i] = 0;
            }

            const int marker_color = 8;
            const int mirror_margin = 2;
            unsigned char visited[ARC_MAX_GRID_CELLS];
            int queue_r[ARC_MAX_GRID_CELLS];
            int queue_c[ARC_MAX_GRID_CELLS];
            int cell_count = src_w * src_h;
            for (int i = 0; i < cell_count; ++i) {
                visited[i] = 0;
            }

            int best_top = -1;
            int best_left = -1;
            int best_height = -1;
            int best_width = -1;
            int best_area = -1;
            for (int yy = 0; yy < src_h; ++yy) {
                for (int xx = 0; xx < src_w; ++xx) {
                    int start_idx = yy * src_w + xx;
                    if (visited[start_idx] || input[start_idx] != marker_color) {
                        continue;
                    }
                    int q_head = 0;
                    int q_tail = 0;
                    queue_r[q_tail] = yy;
                    queue_c[q_tail] = xx;
                    q_tail += 1;
                    visited[start_idx] = 1;
                    int top = yy;
                    int bottom = yy;
                    int left = xx;
                    int right = xx;
                    int comp_area = 0;
                    while (q_head < q_tail) {
                        int cr = queue_r[q_head];
                        int cc = queue_c[q_head];
                        q_head += 1;
                        comp_area += 1;
                        if (cr < top) top = cr;
                        if (cr > bottom) bottom = cr;
                        if (cc < left) left = cc;
                        if (cc > right) right = cc;
                        const int dr[4] = {-1, 1, 0, 0};
                        const int dc[4] = {0, 0, -1, 1};
                        for (int dir = 0; dir < 4; ++dir) {
                            int nr = cr + dr[dir];
                            int nc = cc + dc[dir];
                            if (nr < 0 || nr >= src_h || nc < 0 || nc >= src_w) {
                                continue;
                            }
                            int nidx = nr * src_w + nc;
                            if (visited[nidx] || input[nidx] != marker_color) {
                                continue;
                            }
                            visited[nidx] = 1;
                            queue_r[q_tail] = nr;
                            queue_c[q_tail] = nc;
                            q_tail += 1;
                        }
                    }
                    int height = bottom - top + 1;
                    int width = right - left + 1;
                    int rect_area = height * width;
                    if (rect_area != comp_area) {
                        continue;
                    }
                    if (rect_area > best_area) {
                        best_area = rect_area;
                        best_top = top;
                        best_left = left;
                        best_height = height;
                        best_width = width;
                    }
                }
            }
            if (best_area <= 0) {
                return;
            }

            int top = best_top;
            int left = best_left;
            int height = best_height;
            int width = best_width;
            int total_h = src_h;
            int total_w = src_w;

            if (height > width && (left == 0 || left + width == total_w)) {
                int crop_top = left;
                if (crop_top < 0) crop_top = 0;
                if (crop_top > total_h - width) crop_top = total_h - width;
                int crop_left = top;
                if (crop_left < 0) crop_left = 0;
                if (crop_left > total_w - height) crop_left = total_w - height;
                for (int rr = 0; rr < width; ++rr) {
                    for (int cc = 0; cc < height; ++cc) {
                        output[cc * dst_w + rr] = input[(crop_top + rr) * src_w + (crop_left + cc)];
                    }
                }
                return;
            }

            int right_gap = total_w - (left + width);
            int bottom_gap = total_h - (top + height);
            int side_gap = left < right_gap ? left : right_gap;

            if (side_gap <= width) {
                int crop_left = total_w - width - left;
                if (left > right_gap) {
                    crop_left += mirror_margin;
                } else {
                    crop_left -= mirror_margin;
                }
                if (crop_left < 0) crop_left = 0;
                if (crop_left > total_w - width) crop_left = total_w - width;
                for (int rr = 0; rr < height; ++rr) {
                    for (int cc = 0; cc < width; ++cc) {
                        output[rr * dst_w + cc] = input[(top + rr) * src_w + (crop_left + (width - 1 - cc))];
                    }
                }
                return;
            }

            int crop_top = total_h - height - top;
            if (top > bottom_gap) {
                crop_top += mirror_margin;
            } else {
                crop_top -= mirror_margin;
            }
            if (crop_top < 0) crop_top = 0;
            if (crop_top > total_h - height) crop_top = total_h - height;
            for (int rr = 0; rr < height; ++rr) {
                for (int cc = 0; cc < width; ++cc) {
                    output[rr * dst_w + cc] = input[(crop_top + (height - 1 - rr)) * src_w + (left + cc)];
                }
            }
            return;
        }
        case 15: { // separator bridge projection
            if (x != 0 || y != 0) {
                return;
            }
            int total_cells = dst_w * dst_h;
            for (int i = 0; i < total_cells; ++i) {
                output[i] = input[i];
            }

            int sep_row = -1;
            int sep_col = -1;
            for (int yy = 0; yy < src_h; ++yy) {
                bool full = true;
                for (int xx = 0; xx < src_w; ++xx) {
                    if (input[yy * src_w + xx] != 8) {
                        full = false;
                        break;
                    }
                }
                if (full) {
                    sep_row = yy;
                    break;
                }
            }
            if (sep_row < 0) {
                for (int xx = 0; xx < src_w; ++xx) {
                    bool full = true;
                    for (int yy = 0; yy < src_h; ++yy) {
                        if (input[yy * src_w + xx] != 8) {
                            full = false;
                            break;
                        }
                    }
                    if (full) {
                        sep_col = xx;
                        break;
                    }
                }
            }
            if (sep_row < 0 && sep_col < 0) {
                return;
            }

            unsigned char visited[ARC_MAX_GRID_CELLS];
            int queue_r[ARC_MAX_GRID_CELLS];
            int queue_c[ARC_MAX_GRID_CELLS];
            int comp_top[ARC_MAX_COMPONENTS];
            int comp_bottom[ARC_MAX_COMPONENTS];
            int comp_left[ARC_MAX_COMPONENTS];
            int comp_right[ARC_MAX_COMPONENTS];
            int cell_count = src_w * src_h;
            int comp_count = 0;
            int before_count = 0;
            int after_count = 0;
            for (int i = 0; i < cell_count; ++i) {
                visited[i] = 0;
            }

            for (int yy = 0; yy < src_h; ++yy) {
                for (int xx = 0; xx < src_w; ++xx) {
                    int start_idx = yy * src_w + xx;
                    if (visited[start_idx] || input[start_idx] != 4) {
                        continue;
                    }
                    if (comp_count >= ARC_MAX_COMPONENTS) {
                        return;
                    }
                    int q_head = 0;
                    int q_tail = 0;
                    queue_r[q_tail] = yy;
                    queue_c[q_tail] = xx;
                    q_tail += 1;
                    visited[start_idx] = 1;
                    int top = yy;
                    int bottom = yy;
                    int left = xx;
                    int right = xx;
                    while (q_head < q_tail) {
                        int cr = queue_r[q_head];
                        int cc = queue_c[q_head];
                        q_head += 1;
                        if (cr < top) top = cr;
                        if (cr > bottom) bottom = cr;
                        if (cc < left) left = cc;
                        if (cc > right) right = cc;
                        const int dr[4] = {-1, 1, 0, 0};
                        const int dc[4] = {0, 0, -1, 1};
                        for (int dir = 0; dir < 4; ++dir) {
                            int nr = cr + dr[dir];
                            int nc = cc + dc[dir];
                            if (nr < 0 || nr >= src_h || nc < 0 || nc >= src_w) {
                                continue;
                            }
                            int nidx = nr * src_w + nc;
                            if (visited[nidx] || input[nidx] != 4) {
                                continue;
                            }
                            visited[nidx] = 1;
                            queue_r[q_tail] = nr;
                            queue_c[q_tail] = nc;
                            q_tail += 1;
                        }
                    }
                    if (sep_row >= 0) {
                        if (bottom < sep_row) {
                            before_count += 1;
                        } else if (top > sep_row) {
                            after_count += 1;
                        } else {
                            return;
                        }
                    } else {
                        if (right < sep_col) {
                            before_count += 1;
                        } else if (left > sep_col) {
                            after_count += 1;
                        } else {
                            return;
                        }
                    }
                    comp_top[comp_count] = top;
                    comp_bottom[comp_count] = bottom;
                    comp_left[comp_count] = left;
                    comp_right[comp_count] = right;
                    comp_count += 1;
                }
            }
            if (comp_count == 0) {
                return;
            }
            if ((before_count > 0 && after_count > 0) || (before_count == 0 && after_count == 0)) {
                return;
            }
            bool source_before = before_count > 0;

            for (int i = 0; i < cell_count; ++i) {
                if (input[i] == 4) {
                    output[i] = 3;
                }
            }

            for (int comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
                int top = comp_top[comp_idx];
                int bottom = comp_bottom[comp_idx];
                int left = comp_left[comp_idx];
                int right = comp_right[comp_idx];
                int box_h = bottom - top + 1;
                int box_w = right - left + 1;

                if (sep_row >= 0) {
                    int row_start = source_before ? (sep_row + 1) : 0;
                    int row_stop = source_before ? src_h : sep_row;
                    int pattern_top = src_h;
                    int pattern_bottom = -1;
                    unsigned char pattern[ARC_MAX_GRID_CELLS];
                    for (int i = 0; i < ARC_MAX_GRID_CELLS; ++i) {
                        pattern[i] = 0;
                    }
                    for (int rr = row_start; rr < row_stop; ++rr) {
                        for (int cc = left; cc <= right; ++cc) {
                            if (input[rr * src_w + cc] == 2) {
                                if (rr < pattern_top) pattern_top = rr;
                                if (rr > pattern_bottom) pattern_bottom = rr;
                            }
                        }
                    }
                    int pattern_h = (pattern_bottom >= pattern_top) ? (pattern_bottom - pattern_top + 1) : 0;
                    if (pattern_h > 0) {
                        for (int rr = pattern_top; rr <= pattern_bottom; ++rr) {
                            for (int cc = left; cc <= right; ++cc) {
                                if (input[rr * src_w + cc] == 2) {
                                    int local_r = rr - pattern_top;
                                    int local_c = cc - left;
                                    pattern[local_r * box_w + local_c] = 1;
                                }
                            }
                        }
                    }

                    if (source_before) {
                        for (int rr = bottom + 1; rr < sep_row; ++rr) {
                            for (int cc = left; cc <= right; ++cc) {
                                output[rr * dst_w + cc] = 4;
                            }
                        }
                        int copy_top = pattern_h > 0 ? (src_h - pattern_h) : (src_h - box_h);
                        if (copy_top < 0) copy_top = 0;
                        for (int rr = sep_row + 1; rr < copy_top; ++rr) {
                            for (int cc = left; cc <= right; ++cc) {
                                output[rr * dst_w + cc] = 8;
                            }
                        }
                        for (int rr = copy_top; rr < src_h; ++rr) {
                            for (int cc = left; cc <= right; ++cc) {
                                output[rr * dst_w + cc] = 8;
                            }
                        }
                        if (pattern_h > 0) {
                            for (int rr = 0; rr < pattern_h; ++rr) {
                                for (int cc = 0; cc < box_w; ++cc) {
                                    if (pattern[rr * box_w + cc]) {
                                        output[(copy_top + rr) * dst_w + (left + cc)] = 2;
                                    }
                                }
                            }
                        }
                    } else {
                        int copy_bottom = pattern_h > 0 ? (pattern_h - 1) : (box_h - 1);
                        if (copy_bottom >= src_h) copy_bottom = src_h - 1;
                        for (int rr = 0; rr <= copy_bottom; ++rr) {
                            for (int cc = left; cc <= right; ++cc) {
                                output[rr * dst_w + cc] = 8;
                            }
                        }
                        if (pattern_h > 0) {
                            for (int rr = 0; rr < pattern_h; ++rr) {
                                for (int cc = 0; cc < box_w; ++cc) {
                                    if (pattern[rr * box_w + cc]) {
                                        output[rr * dst_w + (left + cc)] = 2;
                                    }
                                }
                            }
                        }
                        for (int rr = copy_bottom + 1; rr < sep_row; ++rr) {
                            for (int cc = left; cc <= right; ++cc) {
                                output[rr * dst_w + cc] = 8;
                            }
                        }
                        for (int rr = sep_row + 1; rr < top; ++rr) {
                            for (int cc = left; cc <= right; ++cc) {
                                output[rr * dst_w + cc] = 4;
                            }
                        }
                    }
                } else {
                    int col_start = source_before ? (sep_col + 1) : 0;
                    int col_stop = source_before ? src_w : sep_col;
                    int pattern_left = src_w;
                    int pattern_right = -1;
                    unsigned char pattern[ARC_MAX_GRID_CELLS];
                    for (int i = 0; i < ARC_MAX_GRID_CELLS; ++i) {
                        pattern[i] = 0;
                    }
                    for (int rr = top; rr <= bottom; ++rr) {
                        for (int cc = col_start; cc < col_stop; ++cc) {
                            if (input[rr * src_w + cc] == 2) {
                                if (cc < pattern_left) pattern_left = cc;
                                if (cc > pattern_right) pattern_right = cc;
                            }
                        }
                    }
                    int pattern_w = (pattern_right >= pattern_left) ? (pattern_right - pattern_left + 1) : 0;
                    if (pattern_w > 0) {
                        for (int rr = top; rr <= bottom; ++rr) {
                            for (int cc = pattern_left; cc <= pattern_right; ++cc) {
                                if (input[rr * src_w + cc] == 2) {
                                    int local_r = rr - top;
                                    int local_c = cc - pattern_left;
                                    pattern[local_r * pattern_w + local_c] = 1;
                                }
                            }
                        }
                    }

                    if (source_before) {
                        for (int rr = top; rr <= bottom; ++rr) {
                            for (int cc = right + 1; cc < sep_col; ++cc) {
                                output[rr * dst_w + cc] = 4;
                            }
                        }
                        int copy_left = pattern_w > 0 ? (src_w - pattern_w) : (src_w - box_w);
                        if (copy_left < 0) copy_left = 0;
                        for (int rr = top; rr <= bottom; ++rr) {
                            for (int cc = sep_col + 1; cc < copy_left; ++cc) {
                                output[rr * dst_w + cc] = 8;
                            }
                            for (int cc = copy_left; cc < src_w; ++cc) {
                                output[rr * dst_w + cc] = 8;
                            }
                        }
                        if (pattern_w > 0) {
                            for (int rr = 0; rr < box_h; ++rr) {
                                for (int cc = 0; cc < pattern_w; ++cc) {
                                    if (pattern[rr * pattern_w + cc]) {
                                        output[(top + rr) * dst_w + (copy_left + cc)] = 2;
                                    }
                                }
                            }
                        }
                    } else {
                        int copy_right = pattern_w > 0 ? (pattern_w - 1) : (box_w - 1);
                        if (copy_right >= src_w) copy_right = src_w - 1;
                        for (int rr = top; rr <= bottom; ++rr) {
                            for (int cc = 0; cc <= copy_right; ++cc) {
                                output[rr * dst_w + cc] = 8;
                            }
                        }
                        if (pattern_w > 0) {
                            for (int rr = 0; rr < box_h; ++rr) {
                                for (int cc = 0; cc < pattern_w; ++cc) {
                                    if (pattern[rr * pattern_w + cc]) {
                                        output[(top + rr) * dst_w + cc] = 2;
                                    }
                                }
                            }
                        }
                        for (int rr = top; rr <= bottom; ++rr) {
                            for (int cc = copy_right + 1; cc < sep_col; ++cc) {
                                output[rr * dst_w + cc] = 8;
                            }
                            for (int cc = sep_col + 1; cc < left; ++cc) {
                                output[rr * dst_w + cc] = 4;
                            }
                        }
                    }
                }
            }
            return;
        }
        case 16: { // anchor spiral pair
            if (x != 0 || y != 0) {
                return;
            }
            int total_cells = dst_w * dst_h;
            for (int i = 0; i < total_cells; ++i) {
                output[i] = 0;
            }
            if (src_w < 2 || src_h <= 0) {
                return;
            }

            int anchor_r = -1;
            int anchor_c = -1;
            int anchor_count = 0;
            for (int yy = 0; yy < src_h; ++yy) {
                for (int xx = 0; xx < src_w; ++xx) {
                    if (input[yy * src_w + xx] == 1) {
                        anchor_r = yy;
                        anchor_c = xx;
                        anchor_count += 1;
                    }
                }
            }
            if (anchor_count != 1) {
                for (int i = 0; i < total_cells; ++i) {
                    output[i] = input[i];
                }
                return;
            }

            unsigned char primary = input[0];
            unsigned char secondary = input[1];
            if (primary == 0 || primary == 1 || secondary == 0 || secondary == 1 || primary == secondary) {
                for (int i = 0; i < total_cells; ++i) {
                    output[i] = input[i];
                }
                return;
            }

            output[anchor_r * dst_w + anchor_c] = 1;
            int start_c = anchor_c - 1;
            if (start_c < 0) {
                return;
            }
            output[anchor_r * dst_w + start_c] = primary;

            const int dir_r[4] = {0, 1, 0, -1};
            const int dir_c[4] = {-1, 0, 1, 0};
            int current_r = anchor_r;
            int current_c = start_c;
            int segment_length = 2;
            int max_segments = (src_h + src_w) * 2;
            for (int segment_idx = 0; segment_idx < max_segments; ++segment_idx) {
                int desired_steps = (segment_idx == 0) ? (segment_length - 1) : (segment_length + 1);
                if (segment_idx != 0) {
                    segment_length += 1;
                }
                unsigned char color = (segment_idx & 1) ? secondary : primary;
                int actual_steps = 0;
                for (int step = 0; step < desired_steps; ++step) {
                    int next_r = current_r + dir_r[segment_idx % 4];
                    int next_c = current_c + dir_c[segment_idx % 4];
                    if (next_r < 0 || next_r >= src_h || next_c < 0 || next_c >= src_w) {
                        break;
                    }
                    if (next_r == anchor_r && next_c == anchor_c) {
                        break;
                    }
                    current_r = next_r;
                    current_c = next_c;
                    output[current_r * dst_w + current_c] = color;
                    actual_steps += 1;
                }
                if (actual_steps < desired_steps) {
                    break;
                }
            }
            return;
        }
        default:
            // Unsupported op; write zero
            output[y * dst_w + x] = 0;
            return;
    }

    // Bounds check for operations that access source differently
    if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
        output[y * dst_w + x] = 0;
        return;
    }

    int src_idx = src_y * src_w + src_x;
    int dst_idx = y * dst_w + x;
    output[dst_idx] = input[src_idx];
}
