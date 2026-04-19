import cv2
from main import SIFT, compute_homography, apply_homography
import numpy as np
import matplotlib.pyplot as plt
# read the book img and vid
book_img = cv2.imread("cv_cover.jpg")
book_video = cv2.VideoCapture("book.mov")
ar_video = cv2.VideoCapture("ar_source_2.mov")

# book image
h, w = book_img.shape[:2]
# print(f"Book image size: {w}x{h}")
aspect_ratio = w / h
book_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

# video writer configs
fps    = book_video.get(cv2.CAP_PROP_FPS)
width  = int(book_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(book_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

out    = cv2.VideoWriter("ar_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

def SIFT(original, frame, kp1=None, des1=None):
    sift = cv2.SIFT_create()

    if kp1 is None or des1 is None:
        kp1, des1 = sift.detectAndCompute(original, None)

    kp2, des2 = sift.detectAndCompute(frame, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.25 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return src_pts, dst_pts

def overlay_ar_frame(frame, cropped_ar, H, w, h):
    # maps the 0,0 to w,h rectangle to the book's location in the video
    warped_ar = cv2.warpPerspective(cropped_ar, H, (frame.shape[1], frame.shape[0]))

    # creatinga mask for the warped AR frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # mapping 4 corners to find the polygon to fill
    book_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    video_corners = cv2.perspectiveTransform(book_corners, H) # transform the book corners to the video frame using the homography
    
    # to specify the area where the ar frame will be overlaid
    cv2.fillConvexPoly(mask, video_corners.astype(int), 255) # fill the polygon defined by the transformed corners with white (255)

    # combine
    frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    ar_fg    = cv2.bitwise_and(warped_ar, warped_ar, mask=mask)
    return cv2.add(frame_bg, ar_fg)
    

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(book_img, None)

while True:
    ret, frame = book_video.read() # ret is a boolean read/not read
    ret_ar, frame_ar = ar_video.read()
    if not ret or not ret_ar:
        break
    
    h_ar, w_ar = frame_ar.shape[:2]
    # print(f"AR frame size: {w_ar}x{h_ar}")
    ar_aspect  = w_ar / h_ar

    if ar_aspect > aspect_ratio: # ar vid is wider
        new_w = int(h_ar * aspect_ratio) # change width to match book aspect ratio
        cropped_ar = frame_ar[:, (w_ar - new_w) // 2 : (w_ar - new_w) // 2 + new_w]
    else: # ar vid is taller
        new_h = int(w_ar / aspect_ratio)
        cropped_ar = frame_ar[(h_ar - new_h) // 2 : (h_ar - new_h) // 2 + new_h, :] # change height to match book aspect ratio

    cropped_ar = cv2.resize(cropped_ar, (w, h)) #resize to book size

    try:
        src_pts, dst_pts = SIFT(book_img, frame, kp1=kp1, des1=des1)
        H = compute_homography(src_pts, dst_pts)
        result = overlay_ar_frame(frame, cropped_ar, H, w, h)
    except Exception as e:
        print(f"Skipping frame: {e}")
        result = frame  # fallback to original frame

    out.write(result)

out.release()
book_video.release()
ar_video.release()
