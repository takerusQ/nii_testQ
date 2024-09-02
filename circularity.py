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
