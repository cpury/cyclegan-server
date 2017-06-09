import argparse
import cv2
import numpy as np
import os
import random
import shutil

import openface
import openface.helper
from openface.data import iterImgs

modelDir = '/home/apps/models/'
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def from_model_image(image):
    return ((image + 1) * 127.5).astype(np.uint8)


def to_model_image(image):
    return ((image.astype(float) / 127.5) - 1.0)


def align_face(image, size, landmarks='innerEyesAndBottomLip'):
    dlib_face_predictor = os.path.join(
        dlibModelDir, "shape_predictor_68_face_landmarks.dat"
    )

    landmark_map = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }
    if landmarks not in landmark_map:
        raise Exception("Landmarks unrecognized: {}".format(landmarks))

    landmark_indices = landmark_map[landmarks]

    align = openface.AlignDlib(dlib_face_predictor)

    converted_image = from_model_image(image)
    out_image = align.align(
        size, converted_image,
        landmarkIndices=landmark_indices,
        skipMulti=False,
    )

    if out_image is not None:
        converted_out_image = to_model_image(out_image)
        return converted_out_image
