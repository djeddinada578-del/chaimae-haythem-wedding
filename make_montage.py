import cv2
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent
frames = sorted((root / "frames").glob("*.jpg"))
thumbs = []
for frame in frames:
    image = cv2.imread(str(frame))
    image = cv2.resize(image, (216, 384))
    thumbs.append(image)

rows = []
for i in range(0, len(thumbs), 4):
    row = thumbs[i:i + 4]
    while len(row) < 4:
        row.append(np.zeros_like(row[0]))
    rows.append(cv2.hconcat(row))
montage = cv2.vconcat(rows)
cv2.imwrite(str(root / "montage.jpg"), montage)
