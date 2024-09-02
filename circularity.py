import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import distance

def find_and_interpolate_edges(mask):
    height, width = mask.shape
    y_min, y_max = 256 - 40, 256 + 40
    upper_edge = np.full(width, y_min - 1)
    lower_edge = np.full(width, y_max + 1)

    for x in range(width):
        column = mask[y_min:y_max+1, x]
        if np.any(column):
            upper_edge[x] = y_min + np.max(np.where(column)[0])
            lower_edge[x] = y_min + np.min(np.where(column)[0])

    # Interpolate missing points
    x_range = np.arange(width)
    valid_upper = upper_edge != (y_min - 1)
    valid_lower = lower_edge != (y_max + 1)

    if np.any(valid_upper):
        f_upper = interp1d(x_range[valid_upper], upper_edge[valid_upper], kind='linear', fill_value='extrapolate')
        upper_edge = np.clip(f_upper(x_range), y_min, y_max)

    if np.any(valid_lower):
        f_lower = interp1d(x_range[valid_lower], lower_edge[valid_lower], kind='linear', fill_value='extrapolate')
        lower_edge = np.clip(f_lower(x_range), y_min, y_max)

    return upper_edge, lower_edge

def find_max_inscribed_circle(mask, upper_edge, lower_edge):
    y_min, y_max = 256 - 40, 256 + 40
    height, width = mask.shape
    max_radius = 0
    center = None

    for x in range(width):
        for y in range(int(lower_edge[x]), int(upper_edge[x]) + 1):
            if y_min <= y <= y_max and mask[y, x]:
                distances = distance.cdist(np.array([[x, y]]), np.argwhere(mask == 0)).min()
                if distances > max_radius:
                    max_radius = distances
                    center = (x, y)

    return center, max_radius

def visualize_sternum_with_edges_and_circle(slice_image, mask):
    upper_edge, lower_edge = find_and_interpolate_edges(mask)
    center, radius = find_max_inscribed_circle(mask, upper_edge, lower_edge)

    plt.figure(figsize=(10, 10))
    plt.imshow(slice_image, cmap='gray')
    plt.imshow(mask, alpha=0.3, cmap='Reds')
    
    # Plot edges
    x_range = np.arange(len(upper_edge))
    plt.plot(x_range, upper_edge, color='blue', linewidth=2)
    plt.plot(x_range, lower_edge, color='blue', linewidth=2)

    # Plot inscribed circle
    if center is not None and radius > 0:
        circle = plt.Circle(center, radius, color='yellow', fill=False, linewidth=2)
        plt.gca().add_artist(circle)

    plt.title('Sternum with Edges and Max Inscribed Circle')
    plt.ylim(256 + 40, 256 - 40)  # Reverse y-axis to match image coordinates
    plt.tight_layout()

def visualize_results(slices, area_proportion_list, circularity_list, region_list):
    def plot_slice(slice_index):
        plt.figure(figsize=(20, 10))
        
        # Original slice
        plt.subplot(231)
        plt.imshow(slices[slice_index], cmap='gray')
        plt.title(f'Original Slice {slice_index}')
        
        # Sternum region
        plt.subplot(232)
        mask = np.zeros_like(slices[slice_index], dtype=bool)
        for x, y in region_list[slice_index]:
            if 256 - 40 <= y <= 256 + 40:
                mask[y, x] = True
        plt.imshow(slices[slice_index], cmap='gray')
        plt.imshow(mask, cmap='Reds', alpha=0.3)
        plt.title(f'Sternum Region (Slice {slice_index})')
        
        # Metrics
        plt.subplot(233)
        plt.text(0.1, 0.6, f'Area Proportion: {area_proportion_list[slice_index]:.2f}%', fontsize=12)
        plt.text(0.1, 0.4, f'Circularity: {circularity_list[slice_index]:.4f}', fontsize=12)
        plt.axis('off')
        plt.title('Metrics')
        
        # New visualization with edges and inscribed circle
        plt.subplot(212)
        visualize_sternum_with_edges_and_circle(slices[slice_index], mask)
        
        plt.tight_layout()
        plt.show()

    interact(plot_slice, slice_index=IntSlider(min=0, max=len(slices)-1, step=1, value=0))

# メイン処理部分は変更ありません