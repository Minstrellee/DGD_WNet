import numpy as np
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, binary_opening, binary_closing
from scipy.ndimage import label, distance_transform_edt, binary_fill_holes

def generate_markers(foreground_map, ordinal_map, alpha=4):
    markers = np.logical_and(ordinal_map > alpha, foreground_map >= 0.5).astype(np.uint8)
    markers = binary_opening(markers, np.ones((3, 3)))
    markers = binary_closing(markers, np.ones((3, 3)))
    markers = binary_fill_holes(markers)
    return markers

def generate_topology_map(ordinal_map, markers):
    topological_map = -(markers * ordinal_map)
    distance_map = distance_transform_edt(markers)
    topological_map = topological_map * distance_map
    return topological_map

def watershed_segmentation(topological_map, markers, foreground_mask, min_size=50):
    labeled_markers, _ = label(markers)
    segmentation = watershed(topological_map, labeled_markers, mask=foreground_mask)
    segmentation = remove_small_objects(segmentation, min_size=min_size)
    return segmentation

def post_process(foreground_map, ordinal_map, alpha=4, min_size=50):
    markers = generate_markers(foreground_map, ordinal_map, alpha)
    topological_map = generate_topology_map(ordinal_map, markers)
    segmentation = watershed_segmentation(topological_map, markers, foreground_map >= 0.5, min_size)
    return segmentation


def main():
    foreground_map, ordinal_map = load_data()
    segmentation = post_process(foreground_map, ordinal_map, alpha=4, min_size=50)
    import cv2
    cv2.imwrite("", segmentation.astype(np.uint8) * 255)


if __name__ == "__main__":
    main()