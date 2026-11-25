import cv2

video = r"C:\Users\HP\.copilot\chats\04044a77-7cba-4625-a660-66399a881e0c\invitation.mp4"
capture = cv2.VideoCapture(video)
for second, name in [(12, "render_12s.jpg"), (20, "render_20s.jpg"), (33, "render_33s.jpg")]:
    capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, frame = capture.read()
    if ok:
        cv2.imwrite(r"C:\Users\HP\.copilot\chats\04044a77-7cba-4625-a660-66399a881e0c\\" + name, frame)
capture.release()
