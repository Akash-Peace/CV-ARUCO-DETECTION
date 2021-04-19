import cv2, numpy as np, time
def plot(bg_image, fg_image, aruco_value_top_left, aruco_value_top_right, aruco_value_bottom_right, aruco_value_bottom_left):
    img = cv2.imread(bg_image)
    src = cv2.imread(fg_image)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_para = cv2.aruco.DetectorParameters_create()
    corners, ids, rej = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_para)
    if len(corners) != 4:
        exit()
    else:
        ids = ids.flatten()
        ref_pts = []
        for i in (aruco_value_top_left, aruco_value_top_right, aruco_value_bottom_right, aruco_value_bottom_left):  # 923, 1001, 241, 1007
            j = np.squeeze(np.where(ids == i))
            corner = np.squeeze(corners[j])
            ref_pts.append(corner)
        refPtTL, refPtTR, refPtBR, refPtBL = ref_pts
        dst_mat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
        dst_mat = np.array(dst_mat, np.int32)
        src_h, src_w = src.shape[:2]
        img_h, img_w = img.shape[:2]
        src_matrix = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])
        cv2.fillConvexPoly(img, dst_mat, 0)
        homography, _ = cv2.findHomography(src_matrix, dst_mat)
        warped = cv2.warpPerspective(src, homography, (img_w, img_h))
        combined = cv2.add(img, warped)
        cv2.imwrite(f"aruco_photo_embedding_{time.time()}.jpg", combined)
        cv2.imshow("Photo", combined)
        cv2.waitKey(0)