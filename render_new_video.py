import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
cv2.setNumThreads(1)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "assets" / "original-invitation-source.mp4"
OUTPUT = ROOT / "invitation_updated.mp4"
WIDTH = 720
HEIGHT = 1280

FONT_ARABIC = Path(r"C:\Windows\Fonts\trado.ttf")
FONT_NAMES = Path(r"C:\Windows\Fonts\arabtype.ttf")
FONT_LATIN = Path(r"C:\Windows\Fonts\tahoma.ttf")
FONT_DATE = Path(r"C:\Windows\Fonts\georgia.ttf")

INK = (86, 79, 67, 255)
OLIVE = (108, 116, 79, 255)
DYNAMIC_TEXT_MASKS = {
    "opening_names": ([(180, 815, 535, 940)], 8, 5),
    "story_names": (
        [(170, 790, 340, 940), (335, 790, 410, 940), (400, 790, 550, 940)],
        8,
        4,
    ),
    "time": ([(75, 620, 335, 715)], 8, 8),
    "date_day": ([(355, 620, 620, 740)], 8, 8),
    "venue": ([(155, 780, 565, 895)], 8, 8),
}


def load_font(path, size):
    if not path.is_file():
        raise FileNotFoundError(f"Required font not found: {path}")
    return ImageFont.truetype(str(path), size)


def draw_arabic_centered(draw, text, center_x, top, font, color):
    bounds = draw.textbbox((0, 0), text, font=font, direction="rtl", language="ar")
    text_width = bounds[2] - bounds[0]
    draw.text(
        (center_x - text_width / 2, top),
        text,
        font=font,
        fill=color,
        direction="rtl",
        language="ar",
    )


def draw_latin_centered(draw, text, center_x, top, font, color):
    bounds = draw.textbbox((0, 0), text, font=font)
    text_width = bounds[2] - bounds[0]
    draw.text((center_x - text_width / 2, top), text, font=font, fill=color)


def make_layer(draw_callback):
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw_callback(ImageDraw.Draw(image))
    bounds = image.getbbox()
    if bounds is None:
        raise ValueError("An invitation text layer is empty.")
    rgba = np.asarray(image.crop(bounds))
    return (
        bounds,
        rgba[:, :, :3][:, :, ::-1].copy(),
        rgba[:, :, 3].astype(np.float32) / 255.0,
    )


def apply_layer(frame, layer, opacity):
    if opacity <= 0:
        return
    (x1, y1, x2, y2), overlay_bgr, layer_alpha = layer
    base = frame[y1:y2, x1:x2].astype(np.float32)
    alpha = (layer_alpha * opacity)[:, :, None]
    frame[y1:y2, x1:x2] = np.clip(
        base * (1.0 - alpha) + overlay_bgr.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def make_opening_logo_layer():
    def draw(draw):
        font = load_font(FONT_ARABIC, 260)
        draw_arabic_centered(draw, "ش", WIDTH // 2, 450, font, OLIVE)
        draw_arabic_centered(draw, "هـ", WIDTH // 2 + 10, 555, font, OLIVE)

    return make_layer(draw)


def make_names_layer(top, font_size, include_monogram=False):
    def draw(draw):
        font = load_font(FONT_NAMES, font_size)
        draw_arabic_centered(draw, "هيثم", 465, top, font, INK)
        draw_arabic_centered(draw, "شيماء", 255, top, font, INK)
        if include_monogram:
            small_font = load_font(FONT_ARABIC, 80)
            draw_arabic_centered(draw, "ش", WIDTH // 2, top - 20, small_font, OLIVE)
            draw_arabic_centered(draw, "هـ", WIDTH // 2 + 4, top + 10, small_font, OLIVE)
        else:
            draw.line(
                (WIDTH // 2, top + 14, WIDTH // 2, top + font_size - 3),
                fill=OLIVE,
                width=2,
            )

    return make_layer(draw)


def make_event_layers():
    time_font = load_font(FONT_ARABIC, 34)
    day_font = load_font(FONT_ARABIC, 34)
    date_font = load_font(FONT_DATE, 28)
    venue_font = load_font(FONT_LATIN, 25)

    def draw_time(draw):
        draw_arabic_centered(draw, "الثالثة مساءً", 190, 620, time_font, INK)

    def draw_date_and_day(draw):
        draw_arabic_centered(draw, "السبت", 485, 614, day_font, INK)
        draw_latin_centered(draw, "Le 26.12.2026", 485, 665, date_font, INK)

    def draw_venue(draw):
        draw_latin_centered(draw, "Salle des fêtes El Hana", WIDTH // 2, 786, venue_font, INK)
        draw_latin_centered(draw, "Hôtel Royal Elysa", WIDTH // 2, 819, venue_font, INK)

    return make_layer(draw_time), make_layer(draw_date_and_day), make_layer(draw_venue)


def make_footer_brand_layer():
    def draw(draw):
        font = load_font(FONT_DATE, 22)
        draw_latin_centered(draw, "lunevora.studio", WIDTH // 2, 1245, font, INK)

    return make_layer(draw)


def read_frame_at(capture, fps, second):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(second * fps)))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read reference frame at {second} seconds.")
    return frame


def replace_footer_brand(frame, brand_layer):
    x1, y1, x2, y2 = 280, 1238, 470, 1275
    region = frame[y1:y2, x1:x2]
    grayscale = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mask = (grayscale < 150).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    frame[y1:y2, x1:x2] = cv2.inpaint(region, mask, 4, cv2.INPAINT_TELEA)
    apply_layer(frame, brand_layer, 1.0)


def make_text_mask(
    active_frame,
    clean_frame,
    bounds,
    threshold=24,
    spread=1,
):
    x1, y1, x2, y2 = bounds
    difference = cv2.absdiff(active_frame[y1:y2, x1:x2], clean_frame[y1:y2, x1:x2])
    core = (np.max(difference, axis=2) > threshold).astype(np.uint8) * 255
    kernel_size = spread * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    expanded = cv2.dilate(core, kernel, iterations=1)
    feathered = cv2.GaussianBlur(expanded, (0, 0), 1.5).astype(np.float32) / 255.0
    return (bounds, feathered, core > 0, expanded > 0)


def make_dynamic_text_mask(frame, clean_frame, zones, threshold, spread):
    x1 = min(zone[0] for zone in zones)
    y1 = min(zone[1] for zone in zones)
    x2 = max(zone[2] for zone in zones)
    y2 = max(zone[3] for zone in zones)
    bounds = (x1, y1, x2, y2)
    allowed = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for zone_x1, zone_y1, zone_x2, zone_y2 in zones:
        allowed[zone_y1 - y1 : zone_y2 - y1, zone_x1 - x1 : zone_x2 - x1] = 255

    difference = cv2.absdiff(frame[y1:y2, x1:x2], clean_frame[y1:y2, x1:x2])
    core = ((np.max(difference, axis=2) > threshold) & (allowed > 0)).astype(np.uint8) * 255
    kernel_size = spread * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    expanded = cv2.dilate(core, kernel, iterations=1)
    expanded[allowed == 0] = 0
    feathered = cv2.GaussianBlur(expanded, (0, 0), 1.5).astype(np.float32) / 255.0
    return (bounds, feathered, core > 0, expanded > 0)


def restore_text(frame, clean_frame, mask):
    (x1, y1, x2, y2), alpha = mask[:2]
    base = frame[y1:y2, x1:x2].astype(np.float32)
    clean = clean_frame[y1:y2, x1:x2].astype(np.float32)
    alpha = alpha[:, :, None]
    frame[y1:y2, x1:x2] = np.clip(
        base * (1.0 - alpha) + clean * alpha,
        0,
        255,
    ).astype(np.uint8)


def make_clock_center_mask():
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.circle(mask, (223, 589), 13, 255, -1)
    return cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32) / 255.0


def restore_clock_center(frame, clean_frame, mask):
    x1, y1, x2, y2 = 209, 575, 238, 604
    alpha = mask[y1:y2, x1:x2, None]
    frame[y1:y2, x1:x2] = np.clip(
        frame[y1:y2, x1:x2].astype(np.float32) * (1.0 - alpha)
        + clean_frame[y1:y2, x1:x2].astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def draw_clock_hands(frame, opacity):
    if opacity <= 0:
        return
    color = (75, 93, 101)
    center = (223, 589)
    minute_hand = (223, 576)
    hour_hand = (236, 589)
    overlay = frame.copy()
    cv2.line(overlay, center, minute_hand, color, 2, cv2.LINE_AA)
    cv2.line(overlay, center, hour_hand, color, 2, cv2.LINE_AA)
    cv2.circle(overlay, center, 2, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, frame)


def measure_activity(frame, clean_frame, mask):
    x1, y1, x2, y2 = mask[0]
    active_pixels = mask[2]
    if not np.any(active_pixels):
        return 0.0
    difference = cv2.absdiff(frame[y1:y2, x1:x2], clean_frame[y1:y2, x1:x2])
    return float(difference[active_pixels].mean())


def smooth_curve(values):
    if len(values) < 3:
        return values
    return np.convolve(values, np.array([0.2, 0.6, 0.2]), mode="same")


def make_opacity_curve(values, fps, start, end, peak_second, baseline):
    values = smooth_curve(np.asarray(values, dtype=np.float32))
    peak_index = min(len(values) - 1, int(round(peak_second * fps)))
    peak = float(values[peak_index])
    if peak <= baseline:
        raise ValueError(f"No source text was detected for the interval {start}-{end}s.")
    curve = np.clip((values - baseline) / (peak - baseline), 0.0, 1.0)
    first = max(0, int(start * fps))
    last = min(len(curve), int(end * fps))
    curve[:first] = 0.0
    curve[last:] = 0.0
    return curve


def get_mask_specs(capture, fps):
    opening_clean = read_frame_at(capture, fps, 8.0)
    story_clean = read_frame_at(capture, fps, 19.5)
    event_clean = read_frame_at(capture, fps, 21.0)

    opening_active = read_frame_at(capture, fps, 5.0)
    event_active = read_frame_at(capture, fps, 30.0)

    masks = {
        "opening_logo": make_text_mask(
            opening_active, opening_clean, (285, 570, 430, 825)
        ),
        "clock_hands": make_text_mask(
            event_active, event_clean, (209, 575, 238, 604)
        ),
    }

    return opening_clean, story_clean, event_clean, masks


def measure_all_opacities(capture, fps, frame_count, refs, masks):
    opening_clean, story_clean, event_clean = refs
    reference_by_name = {
        "opening_logo": opening_clean,
        "opening_names": opening_clean,
        "story_names": story_clean,
        "time": event_clean,
        "date_day": event_clean,
        "venue": event_clean,
        "clock_hands": event_clean,
    }
    values = {name: [] for name in masks}
    values.update({name: [] for name in DYNAMIC_TEXT_MASKS})
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("The invitation source ended during opacity analysis.")
        for name in values:
            if name in DYNAMIC_TEXT_MASKS:
                zones, threshold, spread = DYNAMIC_TEXT_MASKS[name]
                mask = make_dynamic_text_mask(
                    frame,
                    reference_by_name[name],
                    zones,
                    threshold,
                    spread,
                )
            else:
                mask = masks[name]
            values[name].append(
                measure_activity(frame, reference_by_name[name], mask)
            )

    return {
        "opening_logo": make_opacity_curve(values["opening_logo"], fps, 1.8, 8.3, 5.0, 3.0),
        "opening_names": make_opacity_curve(values["opening_names"], fps, 2.3, 8.3, 5.0, 3.0),
        "story_names": make_opacity_curve(values["story_names"], fps, 14.0, 19.5, 17.5, 3.0),
        "time": make_opacity_curve(values["time"], fps, 21.5, 34.0, 30.0, 3.0),
        "date_day": make_opacity_curve(values["date_day"], fps, 21.5, 34.0, 30.0, 3.0),
        "venue": make_opacity_curve(values["venue"], fps, 21.5, 34.0, 30.0, 3.0),
        "clock_hands": make_opacity_curve(values["clock_hands"], fps, 21.5, 34.0, 30.0, 3.0),
    }


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Original invitation video not found: {SOURCE}")

    capture = cv2.VideoCapture(str(SOURCE))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open the original invitation video: {SOURCE}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width != WIDTH or height != HEIGHT or fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"Unexpected video format: {width}x{height} at {fps} fps")

    opening_clean, story_clean, event_clean, masks = get_mask_specs(capture, fps)
    opacities = measure_all_opacities(
        capture,
        fps,
        frame_count,
        (opening_clean, story_clean, event_clean),
        masks,
    )
    capture.release()
    capture = cv2.VideoCapture(str(SOURCE))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen the original invitation video: {SOURCE}")

    opening_logo_layer = make_opening_logo_layer()
    opening_names_layer = make_names_layer(810, 76)
    story_names_layer = make_names_layer(800, 82, include_monogram=True)
    time_layer, date_day_layer, venue_layer = make_event_layers()
    footer_brand_layer = make_footer_brand_layer()
    clock_center_mask = make_clock_center_mask()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    temporary_output = OUTPUT.with_name("invitation_updated.tmp.mp4")
    log_file = tempfile.NamedTemporaryFile(mode="w+b", suffix=".log", delete=False)
    log_path = Path(log_file.name)
    command = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", str(SOURCE),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-threads", "1",
        "-preset", "ultrafast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(temporary_output),
    ]

    process = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log_file)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read source frame {frame_index}.")
            second = frame_index / fps

            if 1.8 <= second < 8.3:
                restore_text(frame, opening_clean, masks["opening_logo"])
            if 2.3 <= second < 8.3:
                zones, threshold, spread = DYNAMIC_TEXT_MASKS["opening_names"]
                restore_text(
                    frame,
                    opening_clean,
                    make_dynamic_text_mask(
                        frame, opening_clean, zones, threshold, spread
                    ),
                )
            if 1.8 <= second < 8.3:
                apply_layer(frame, opening_logo_layer, opacities["opening_logo"][frame_index])
            if 2.3 <= second < 8.3:
                apply_layer(frame, opening_names_layer, opacities["opening_names"][frame_index])

            if 14.0 <= second < 19.5:
                zones, threshold, spread = DYNAMIC_TEXT_MASKS["story_names"]
                restore_text(
                    frame,
                    story_clean,
                    make_dynamic_text_mask(
                        frame, story_clean, zones, threshold, spread
                    ),
                )
                apply_layer(frame, story_names_layer, opacities["story_names"][frame_index])

            if second >= 21.5:
                for name in ("time", "date_day", "venue"):
                    zones = DYNAMIC_TEXT_MASKS[name][0]
                    mask = make_dynamic_text_mask(
                        frame,
                        event_clean,
                        zones,
                        threshold=4,
                        spread=3,
                    )
                    restore_text(frame, event_clean, mask)
                restore_clock_center(frame, event_clean, clock_center_mask)
                apply_layer(frame, time_layer, opacities["time"][frame_index])
                apply_layer(frame, date_day_layer, opacities["date_day"][frame_index])
                apply_layer(frame, venue_layer, opacities["venue"][frame_index])
                draw_clock_hands(frame, opacities["clock_hands"][frame_index])

            replace_footer_brand(frame, footer_brand_layer)

            try:
                process.stdin.write(frame.tobytes())
            except BrokenPipeError as error:
                process.wait()
                log_file.flush()
                log_file.seek(0)
                details = log_file.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"FFmpeg stopped while encoding the invitation video:\n{details}"
                ) from error

        capture.release()
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            log_file.flush()
            log_file.seek(0)
            details = log_file.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg failed to encode the invitation video:\n{details}")

        os.replace(temporary_output, OUTPUT)
    finally:
        capture.release()
        if process is not None and process.poll() is None:
            if process.stdin:
                process.stdin.close()
            process.wait()
        temporary_output.unlink(missing_ok=True)
        log_file.close()
        log_path.unlink(missing_ok=True)

    print(f"Updated invitation video saved to {OUTPUT}")


if __name__ == "__main__":
    main()
