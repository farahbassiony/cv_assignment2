import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def SIFT(original,frame):
    sift = cv.SIFT_create()

    keypoints = sift.detect(frame, None)
    img = cv.drawKeypoints(frame, keypoints, None,
                           flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(original, None)
    kp2, des2 = sift.detectAndCompute(frame, None)

    #2 nearest matches (brute force matcher)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2) #Compares every descriptor in image 1 with image 2

    #m = best match, n = second-best match to remove ambiguous matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Take best 50
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    good_matches = good_matches[:50]

    # Draw matches
    matched_img = cv.drawMatches(
        original, kp1,
        frame, kp2,
        good_matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # ---- Added: plot book, frame, matches ----
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.title("Book Image")
    plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("First Frame")
    plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Matches")
    plt.imshow(cv.cvtColor(matched_img, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

    # ---- Added: extract correspondences ----
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return src_pts, dst_pts


def compute_homography(src_pts, dst_pts):

    n = src_pts.shape[0]
    if n < 4:
        print("At least 4 points are required to compute homography.")
        return None

    A = []
    b = []

    for i in range(n):
        x, y = src_pts[i]
        x_p, y_p = dst_pts[i]

        A.append([x, y, 1, 0, 0, 0, -x*x_p, -y*x_p])
        A.append([0, 0, 0, x, y, 1, -x*y_p, -y*y_p])

        b.append(x_p)
        b.append(y_p)

    A = np.array(A)
    b = np.array(b)

    # Solve Ax = b using least squares
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    H = np.array([
        [x[0], x[1], x[2]],
        [x[3], x[4], x[5]],
        [x[6], x[7], 1]
    ])

    return H


# ---- Added: apply homography for verification ----
def apply_homography(H, pts):
    pts_h = np.hstack([pts, np.ones((pts.shape[0],1))])
    mapped = (H @ pts_h.T).T
    mapped = mapped[:, :2] / mapped[:, 2][:, np.newaxis]
    return mapped


if __name__ == "__main__":
    count=0

    video = cv.VideoCapture("book.mov")
    book_img = cv.imread("cv_cover.jpg")

    ret, frame = video.read()
    if not ret:
        print("video not found")

    #correspondences
    src_pts, dst_pts = SIFT(book_img, frame)

    H = compute_homography(src_pts, dst_pts)
    print("Homography Matrix:\n", H)

    mapped_pts = apply_homography(H, src_pts)

    for pt in mapped_pts[:20]:
        cv.circle(frame, tuple(np.int32(pt)), 5, (0,255,0), -1)

    plt.figure(figsize=(6,6))
    plt.title("Mapped Points ")
    plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    video.release()
