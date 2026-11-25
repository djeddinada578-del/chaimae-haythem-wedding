import cv2
from pathlib import Path

root = Path(__file__).resolve().parent
video = root / "invitation_updated.mp4"
output = root / "frames"
output.mkdir(exist_ok=True)
for old_frame in output.glob("*.jpg"):
    old_frame.unlink()

capture = cv2.VideoCapture(str(video))
fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("size", width, height, "fps", fps, "frames", frame_count, "duration", frame_count / fps)

step = max(1, int(fps))
index = 0
saved = 0
while True:
    ok, frame = capture.read()
    if not ok:
        break
    if index % step == 0:
        cv2.imwrite(str(output / f"frame_{saved:03d}.jpg"), frame)
        saved += 1
    index += 1

print("saved", saved)
