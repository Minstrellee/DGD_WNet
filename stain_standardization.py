import numpy as np
import cv2
from sklearn.decomposition import FastICA
from pywt import wavedec2, waverec2


def convert_to_optical_density(image, background_intensity=255):

    image = image.astype(np.float32)
    od_image = -np.log((image + 1) / background_intensity)  # Avoid log(0)
    return od_image


def wavelet_transform(image):

    coeffs = []
    for channel in range(3):  # Process each RGB channel
        channel_data = image[:, :, channel]
        coeffs.append(wavedec2(channel_data, wavelet='haar', level=2))
    return coeffs


def normalize_and_sort_subbands(coeffs):

    normalized_coeffs = []
    for channel_coeffs in coeffs:
        normalized_channel = []
        for subband in channel_coeffs:
            subband = (subband - np.mean(subband)) / np.std(subband)  # Normalize
            normalized_channel.append(subband)
        normalized_coeffs.append(normalized_channel)
    return normalized_coeffs


def compute_staining_matrix(od_image):

    ica = FastICA(n_components=3)  # Assume 3 components for RGB
    ica.fit(od_image.reshape(-1, 3))
    return ica.components_


def inverse_beer_lambert_transform(od_norm, background_intensity=255):

    normalized_image = background_intensity * np.exp(-od_norm)
    return np.clip(normalized_image, 0, 255).astype(np.uint8)


def stain_standardization(image):


    od_image = convert_to_optical_density(image)
    wavelet_coeffs = wavelet_transform(od_image)
    normalized_coeffs = normalize_and_sort_subbands(wavelet_coeffs)
    staining_matrix = compute_staining_matrix(od_image)
    od_norm = np.dot(od_image.reshape(-1, 3), np.linalg.inv(staining_matrix))
    od_norm = od_norm.reshape(od_image.shape)
    normalized_image = inverse_beer_lambert_transform(od_norm)

    return normalized_image