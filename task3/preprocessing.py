import cv2
import numpy as np
from typing import Callable
from skimage.restoration import denoise_nl_means
from skimage.filters import threshold_yen, threshold_multiotsu

#---------------------------preprocessing functions-------------------------------------------------------

def rescale_image(img: np.ndarray, height=128, width=128) -> np.ndarray:
    return cv2.resize(img, dsize=(height, width), interpolation=cv2.INTER_CUBIC)

def denoise_img(img: np.ndarray, iterations=1) -> np.ndarray:
    for _ in range(iterations):
        img = denoise_nl_means(img)
    img = 255 * img

    return np.round(img).astype(np.uint8)

#-----------------------------augmentation functions------------------------------------------------------

def identity_img(img: np.ndarray) -> np.ndarray:
    '''this function is used to create a channel without augmentation during preprocessing'''
    return img

def blur_img(img: np.ndarray, mode="median") -> np.ndarray:
    if mode == "median":
        blurred = cv2.medianBlur(img,3)
    if mode == "gauss":
        blurred = cv2.GaussianBlur(img, (3,3), 1)

    return blurred

def produce_mask_layer(img: np.ndarray, method='mo', classes_for_mo=3) -> np.ndarray:
    '''tries to segment pixels of image into differenct classes using statistical methods'''
    if method == 'mo':
            thresholds = threshold_multiotsu(img, classes_for_mo)
            img = np.digitize(img, bins=thresholds)
    elif method == 'yen':
            threshold = threshold_yen(img)
            img = np.array(img > threshold, dtype='int')
    else:
         produce_mask_layer(img)

    return img

#---------------------------------------------------------------------------------------------------

class Image_Processor:
     
    def __init__(self, preprocessing_fs: list[Callable], augmentation_fs: list[Callable]):
        self._preprocessing_fs = preprocessing_fs
        self._augmentation_fs = augmentation_fs

    def process_image(self, img: np.ndarray):
        '''
        img: has dimensions (height, width)

        returns: returns np.ndarray with the following dimensions (height, width, channels)
            the amount of channels depends on the amount of augmentation functions
        '''

        preprocessed_img = self._preprocess_image(img)            
        augmentation_img = self._augment_image(preprocessed_img)
        return augmentation_img
    
    def process_label(self, label: np.ndarray):
        '''
        label: has dimensions (height, width)
        '''

        label = np.array(label, dtype=np.uint8)
        label = rescale_image(label)
        return label
    

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        '''
        imgs: dimensions (height, width)
        return: dimensions (height, width)
        '''
        preprocessed_img = img
        for preprocessing_f in self._preprocessing_fs:
            preprocessed_img = preprocessing_f(preprocessed_img)

        return preprocessed_img
    
    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        '''
        imgs: dimensions (height, width)
        return: dimensions (height, width, channels)
        '''
        augmented_imgs = []
        for augmentation_f in self._augmentation_fs:
            augmented_imgs.append(
                augmentation_f(img)                     
            )

        return np.stack(augmented_imgs, axis=2)
    
STANDARD_IMAGE_PROCESSOR = Image_Processor(
    preprocessing_fs=[rescale_image, denoise_img],
    augmentation_fs=[identity_img, blur_img, produce_mask_layer]
)