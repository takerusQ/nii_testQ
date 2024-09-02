各スライスに色つきで表示されているregionの上縁と下縁の曲線(ノイズなどで途切れていたら補完する)をブルーで、そこに内接できる最大の半径の円をイエローで追加のCT画像に表示する関数を追加して

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









import os
import pydicom
import cv2
import numpy as np
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import ipywidgets as widgets

# 既存の関数（extract_bones, extract_specific_y, extract_specific_x, analyze_sternum）はそのまま使用すると仮定します

def process_dicom_directory(dicom_dir_path):
    dicom_files = sorted([f for f in os.listdir(dicom_dir_path) if f.endswith('.dcm')])
    d = pydicom.dcmread(os.path.join(dicom_dir_path, dicom_files[0]))

    slices = [cv2.resize(
        pydicom.dcmread(os.path.join(dicom_dir_path, f)).pixel_array,
        dsize=None,
        #fx=(d.ReconstructionDiameter / 512),
        #fy=(d.ReconstructionDiameter / 512),
        fx=(512 / 512),
        fy=(512 / 512),
        interpolation=cv2.INTER_CUBIC)
        for f in dicom_files]

    return slices[:23]  # スライスを23枚に制限

def analyze_slices(slices):
    area_proportion_list = []
    circularity_list = []
    region_list = []

    for slice in slices:
        image = slice
        bones = extract_bones(image)
        bones_specific_y = extract_specific_y(bones, y_min=256-40, y_max=256+40)
        bones_specific_xy = extract_specific_x(bones_specific_y, x_min=0, x_max=256)
        bones_specific_xy_mask_median_3 = median_filter(bones_specific_xy, size=3)

        sternum_area, region_points, circularity = analyze_sternum(bones_specific_xy_mask_median_3)

        area_proportion = np.sum(bones_specific_xy_mask_median_3 >= 200) / sternum_area * 100 if sternum_area > 0 else 0

        area_proportion_list.append(area_proportion)
        circularity_list.append(circularity)
        region_list.append(region_points)

    return area_proportion_list, circularity_list, region_list

def visualize_results(slices, area_proportion_list, circularity_list, region_list):
    def plot_slice(slice_index):
        plt.figure(figsize=(15, 5))

        # Original slice
        plt.subplot(131)
        plt.imshow(slices[slice_index], cmap='gray')
        plt.title(f'Original Slice {slice_index}')

        # Sternum region
        plt.subplot(132)
        mask = np.zeros_like(slices[slice_index], dtype=bool)
        for x, y in region_list[slice_index]:
            mask[y, x] = True
        plt.imshow(slices[slice_index], cmap='gray')
        plt.imshow(mask, cmap='Reds', alpha=0.3)
        plt.title(f'Sternum Region (Slice {slice_index})')

        # Metrics
        plt.subplot(133)
        plt.text(0.1, 0.6, f'Area Proportion: {area_proportion_list[slice_index]:.2f}%', fontsize=12)
        plt.text(0.1, 0.4, f'Circularity: {circularity_list[slice_index]:.4f}', fontsize=12)
        plt.axis('off')
        plt.title('Metrics')

        plt.tight_layout()
        plt.show()

    interact(plot_slice, slice_index=IntSlider(min=0, max=len(slices)-1, step=1, value=0))

# メイン処理
example_person_dicom_dir_path = "/content/drive/Shareddrives/複数手法を用いた256画素CT画像の主観評価/本番環境/data/2_1_島村先生にアップロード頂いたファイルの整理dataのうち必要な患者のseries2のdata/0030_20200805_2"

slices = process_dicom_directory(example_person_dicom_dir_path)
area_proportion_list, circularity_list, region_list = analyze_slices(slices)

# 結果の表示
print("Area Proportion List:", area_proportion_list)
print("Circularity List:", circularity_list)

# インタラクティブな可視化
visualize_results(slices, area_proportion_list, circularity_list, region_list)
