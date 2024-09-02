これが一番うまくいっていた。黄色い内接円が赤のregionの上縁と下縁をはみ出ないようにして。
内部に少しくらいノイズ由来の非regionを含むのは許容します。

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

def find_and_interpolate_edges(mask):
    height, width = mask.shape
    upper_edge = np.full(width, -1)
    lower_edge = np.full(width, height)

    for x in range(width):
        column = mask[:, x]
        if np.any(column):
            upper_edge[x] = np.max(np.where(column)[0])
            lower_edge[x] = np.min(np.where(column)[0])

    # Interpolate missing points
    x_range = np.arange(width)
    valid_upper = upper_edge != -1
    valid_lower = lower_edge != height

    if np.any(valid_upper):
        f_upper = interp1d(x_range[valid_upper], upper_edge[valid_upper], kind='linear', fill_value='extrapolate')
        upper_edge = f_upper(x_range)

    if np.any(valid_lower):
        f_lower = interp1d(x_range[valid_lower], lower_edge[valid_lower], kind='linear', fill_value='extrapolate')
        lower_edge = f_lower(x_range)

    return upper_edge, lower_edge

def find_max_inscribed_circle(upper_edge, lower_edge):
    points = np.column_stack((np.arange(len(upper_edge)), upper_edge))
    points = np.vstack((points, np.column_stack((np.arange(len(lower_edge)), lower_edge))))
    
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    max_radius = 0
    center = None

    for x in range(len(upper_edge)):
        y_range = np.arange(lower_edge[x], upper_edge[x] + 1)
        for y in y_range:
            point = np.array([x, y])
            distances = np.min(np.abs(np.cross(hull_points[1:] - hull_points[:-1], point - hull_points[:-1])) / 
                               np.linalg.norm(hull_points[1:] - hull_points[:-1], axis=1))
            if distances > max_radius:
                max_radius = distances
                center = point

    return center, max_radius

def visualize_sternum_with_edges_and_circle(slice_image, mask):
    upper_edge, lower_edge = find_and_interpolate_edges(mask)
    center, radius = find_max_inscribed_circle(upper_edge, lower_edge)

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
    plt.axis('off')
    plt.tight_layout()
    plt.show()

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



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from ipywidgets import interact, IntSlider

def find_and_interpolate_edges(mask):
    # この関数は変更なし

def find_max_inscribed_circle(upper_edge, lower_edge):
    points = np.column_stack((np.arange(len(upper_edge)), upper_edge))
    points = np.vstack((points, np.column_stack((np.arange(len(lower_edge)), lower_edge))))
    
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    max_radius = 0
    center = None

    x_min, x_max = 256 - 40, 256 + 40  # x座標の制限

    for x in range(max(0, x_min), min(len(upper_edge), x_max + 1)):
        y_range = np.arange(lower_edge[x], upper_edge[x] + 1)
        for y in y_range:
            point = np.array([x, y])
            distances = np.min(np.abs(np.cross(hull_points[1:] - hull_points[:-1], point - hull_points[:-1])) / 
                               np.linalg.norm(hull_points[1:] - hull_points[:-1], axis=1))
            
            # 円の左端と右端がx_minとx_maxの範囲内に収まるかチェック
            if distances <= x - x_min and distances <= x_max - x:
                if distances > max_radius:
                    max_radius = distances
                    center = point

    return center, max_radius

def visualize_sternum_with_edges_and_circle(slice_image, mask):
    upper_edge, lower_edge = find_and_interpolate_edges(mask)
    center, radius = find_max_inscribed_circle(upper_edge, lower_edge)

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

    # Plot x-coordinate constraints
    plt.axvline(x=256-40, color='green', linestyle='--', linewidth=1)
    plt.axvline(x=256+40, color='green', linestyle='--', linewidth=1)

    plt.title('Sternum with Edges and Max Inscribed Circle')
    plt.axis('off')
    plt.tight_layout()

def visualize_results(slices, area_proportion_list, circularity_list, region_list):
    # この関数は変更なし

# メイン処理部分は変更ありません