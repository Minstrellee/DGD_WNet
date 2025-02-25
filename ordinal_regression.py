import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
import cv2

def get_centroids(mask):
    props = regionprops(mask)
    centroids = []
    for prop in props:
        centroids.append(prop.centroid)
    return centroids

def calculate_distances(mask, centroids, instance_id):
    y, x = np.where(mask == instance_id)
    pixels = np.column_stack((x, y))
    centroid = np.array([centroids[instance_id - 1][::-1]])
    distances = cdist(pixels, centroid, 'euclidean').flatten()
    return distances

def calculate_thresholds(K):
    thresholds = []
    for j in range(1, K + 1):
        s = np.exp(np.log(2) * (1 - K / j)) - 1
        thresholds.append(s)
    thresholds.insert(0, 0)
    return thresholds

def assign_labels(distances, max_distance, thresholds):
    normalized_distances = distances / max_distance
    labels = np.zeros_like(distances, dtype=int)
    for j in range(1, len(thresholds)):
        mask = (normalized_distances >= thresholds[j]) & (normalized_distances < thresholds[j - 1])
        labels[mask] = j
    return labels

def ordinal_regression(mask, K=8):
    centroids = get_centroids(mask)
    height, width = mask.shape
    ordinal_label_map = np.zeros((height, width), dtype=int)
    for instance_id in range(1, len(centroids) + 1):
        y, x = np.where(mask == instance_id)
        distances = calculate_distances(mask, centroids, instance_id)
        max_distance = np.max(distances)
        thresholds = calculate_thresholds(K)
        instance_labels = assign_labels(distances, max_distance, thresholds)
        ordinal_label_map[y, x] = instance_labels
    return ordinal_label_map

def main():
    mask = cv2.imread(" ", cv2.IMREAD_GRAYSCALE)
    ord_labels = ordinal_regression(mask, K=8)
    cv2.imwrite("", ord_labels)

if __name__ == "__main__":
    main()