import cv2 as cv
import os
import numpy as np
import sys

sys.path.append(os.path.abspath('src'))

from sfm import compute_all_camera_pairs_data, compute_all_camera_n_tuples_data 


def load_fountain_p11() -> list[np.ndarray]:
    dataset_dir = 'fountain-p11'
    dataset = []
    for i in range(11):
        image_filename = f'{i:04d}.png'
        image_file = os.path.join(dataset_dir, image_filename)
        image = cv.imread(image_file, cv.IMREAD_COLOR)
        dataset.append(image)
    return dataset


if __name__ == '__main__':
    ds = load_fountain_p11()
    ds = ds[1:-1]

    K = np.array([
        [2759.48, 0, 1520.69],
        [0, 2764.16, 1006.81],
        [0, 0, 1], 
    ])

    images = ds[:5]
    feature_extractor = cv.SIFT.create()
    # feature_extractor = cv.xfeatures2d.SURF.create()

    data = []
    for i, im in enumerate(images):
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        kp, des = feature_extractor.detectAndCompute(im_gray, None)
        item = {
            'id': i,
            'kp': kp,
            'des': des,
        }
        data.append(item)

    points, R_rel, t_rel = compute_all_camera_pairs_data(data, K)
    cam_3tuples = compute_all_camera_n_tuples_data(R_rel, t_rel, len(images), 3)

    for key, val in cam_3tuples.items():
        print(key)
