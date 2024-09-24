import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cv2
from ipywidgets import interact, IntSlider
import numpy as np
import pydicom
import os

def process_dicom_directory(directory_path):
    slices = []
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith('.dcm'):
            file_path = os.path.join(directory_path, filename)
            dcm = pydicom.dcmread(file_path)
            slices.append(dcm.pixel_array)
    return slices

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

def find_max_inscribed_circle(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0

    x_min, x_max = 256 - 40, 256 + 40
    dist_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    dist_map[:, :x_min] = 0
    dist_map[:, x_max:] = 0
    max_val, max_loc = cv2.minMaxLoc(dist_map)

    return max_loc, max_val

def calculate_improved_distance_map(mask):
    inverted_mask = cv2.bitwise_not(mask.astype(np.uint8))
    dist_map = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist_map

def visualize_sternum_with_edges_and_circle(slice_image, mask):
    upper_edge, lower_edge = find_and_interpolate_edges(mask)
    center, radius = find_max_inscribed_circle(mask)

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

def analyze_slices(slices):
    area_proportion_list = []
    area_proportion_in_circle_list = []
    region_list = []

    for slice_image in slices:
        mask = (slice_image > 150) & (slice_image < 300)
        area_proportion = np.sum(mask) / mask.size * 100
        area_proportion_list.append(area_proportion)

        center, radius = find_max_inscribed_circle(mask)
        if center is not None and radius > 0:
            y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
            dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            circle_mask = dist_from_center <= radius
            area_proportion_in_circle = np.sum(mask & circle_mask) / np.sum(circle_mask)
        else:
            area_proportion_in_circle = 0
        area_proportion_in_circle_list.append(area_proportion_in_circle)

        region = np.argwhere(mask)
        region_list.append(region)

    return area_proportion_list, area_proportion_in_circle_list, region_list

def visualize_results(slices, area_proportion_list, area_proportion_in_circle_list, region_list):
    def plot_slice(slice_index):
        fig = plt.figure(figsize=(20, 15))
        
        # Original slice
        ax1 = fig.add_subplot(231)
        ax1.imshow(slices[slice_index], cmap='gray')
        ax1.set_title(f'Original Slice {slice_index}')
        ax1.axis('off')

        # Sternum region
        ax2 = fig.add_subplot(232)
        mask = np.zeros_like(slices[slice_index], dtype=bool)
        for y, x in region_list[slice_index]:
            mask[y, x] = True
        ax2.imshow(slices[slice_index], cmap='gray')
        ax2.imshow(mask, cmap='Reds', alpha=0.3)
        ax2.set_title(f'Sternum Region (Slice {slice_index})')
        ax2.axis('off')

        # Metrics
        ax3 = fig.add_subplot(233)
        ax3.text(0.1, 0.6, f'Area Proportion: {area_proportion_list[slice_index]:.2f}%', fontsize=12)
        ax3.text(0.1, 0.4, f'area_proportion_in_circle: {area_proportion_in_circle_list[slice_index]:.4f}', fontsize=12)
        ax3.axis('off')
        ax3.set_title('Metrics')

        # New visualization with edges and inscribed circle
        ax4 = fig.add_subplot(234)
        visualize_sternum_with_edges_and_circle(slices[slice_index], mask)
        ax4.set_title('Sternum with Edges and Circle')

        # Area Proportion Plot
        ax5 = fig.add_subplot(235)
        ax5.plot(area_proportion_list)
        ax5.set_title('Area Proportion')
        ax5.set_xlabel('Slice Index')
        ax5.set_ylabel('Area Proportion (%)')
        max_area_index = np.argmax(area_proportion_list)
        max_area_index_partial = np.argmax(area_proportion_list[10:]) + 10
        ax5.text(0.65, 0.95, f'Max (All): {max_area_index}', transform=ax5.transAxes, fontsize=10)
        ax5.text(0.65, 0.85, f'Max (10+): {max_area_index_partial}', transform=ax5.transAxes, fontsize=12, fontweight='bold')
        ax5.axvline(x=slice_index, color='r', linestyle='--')
        ax5.axvline(x=10, color='g', linestyle=':')

        # Improved Distance map
        ax6 = fig.add_subplot(236)
        improved_dist_map = calculate_improved_distance_map(mask)
        im = ax6.imshow(improved_dist_map, cmap='jet')
        ax6.set_title('Improved Distance Map')
        ax6.axis('off')
        fig.colorbar(im, ax=ax6, label='Distance (pixels)')

        plt.tight_layout()
        plt.show()

    interact(plot_slice, slice_index=IntSlider(min=0, max=len(slices)-1, step=1, value=0))

# メイン処理
example_person_dicom_dir_path = "/content/drive/Shareddrives/複数手法を用いた256画素CT画像の主観評価/本番環境/data/2_1_島村先生にアップロード頂いたファイルの整理dataのうち必要な患者のseries2のdata/0017_20190829_2"
slices = process_dicom_directory(example_person_dicom_dir_path)
area_proportion_list, area_proportion_in_circle_list, region_list = analyze_slices(slices)

# 結果の表示
print("Area Proportion List:", area_proportion_list)
print("area_proportion_in_circle List:", area_proportion_in_circle_list)

# インタラクティブな可視化
visualize_results(slices, area_proportion_list, area_proportion_in_circle_list, region_list)













import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cv2
from ipywidgets import interact, IntSlider
import numpy as np

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

def find_max_inscribed_circle(mask):
    # マスクの輪郭を取得
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0

    # x座標の制限（256±40）
    x_min, x_max = 256 - 40, 256 + 40

    # 距離マップを計算
    dist_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

    # x座標制限を適用
    dist_map[:, :x_min] = 0
    dist_map[:, x_max:] = 0

    # 最大距離とその位置を取得
    max_val, max_loc = cv2.minMaxLoc(dist_map)

    return max_loc, max_val

def visualize_sternum_with_edges_and_circle(slice_image, mask):
    upper_edge, lower_edge = find_and_interpolate_edges(mask)
    center, radius = find_max_inscribed_circle(mask)

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

def visualize_results(slices, area_proportion_list, area_proportion_in_circle_list, region_list):
    def plot_slice(slice_index):
        fig = plt.figure(figsize=(20, 15))

        # Original slice
        ax1 = fig.add_subplot(231)
        ax1.imshow(slices[slice_index], cmap='gray')
        ax1.set_title(f'Original Slice {slice_index}')
        ax1.axis('off')

        # Sternum region
        ax2 = fig.add_subplot(232)
        mask = np.zeros_like(slices[slice_index], dtype=bool)
        for y, x in region_list[slice_index]:
            mask[y, x] = True
        ax2.imshow(slices[slice_index], cmap='gray')
        ax2.imshow(mask, cmap='Reds', alpha=0.3)
        ax2.set_title(f'Sternum Region (Slice {slice_index})')
        ax2.axis('off')

        # Metrics
        ax3 = fig.add_subplot(233)
        ax3.text(0.1, 0.6, f'Area Proportion: {area_proportion_list[slice_index]:.2f}%', fontsize=12)
        ax3.text(0.1, 0.4, f'area_proportion_in_circle: {area_proportion_in_circle_list[slice_index]:.4f}', fontsize=12)
        ax3.axis('off')
        ax3.set_title('Metrics')

        # New visualization with edges and inscribed circle
        ax4 = fig.add_subplot(234)
        visualize_sternum_with_edges_and_circle(slices[slice_index], mask)
        ax4.set_title('Sternum with Edges and Circle')

        # Area Proportion Plot
        ax5 = fig.add_subplot(235)
        ax5.plot(area_proportion_list)
        ax5.set_title('Area Proportion')
        ax5.set_xlabel('Slice Index')
        ax5.set_ylabel('Area Proportion (%)')
        max_area_index = np.argmax(area_proportion_list)
        max_area_index_partial = np.argmax(area_proportion_list[10:]) + 10
        ax5.text(0.65, 0.95, f'Max (All): {max_area_index}', transform=ax5.transAxes, fontsize=10)
        ax5.text(0.65, 0.85, f'Max (10+): {max_area_index_partial}', transform=ax5.transAxes, fontsize=12, fontweight='bold')
        ax5.axvline(x=slice_index, color='r', linestyle='--')
        ax5.axvline(x=10, color='g', linestyle=':')

        # Distance map
        ax6 = fig.add_subplot(236)
        dist_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        ax6.imshow(dist_map, cmap='jet')
        ax6.set_title('Distance Map')
        ax6.axis('off')

        plt.tight_layout()
        plt.show()

    interact(plot_slice, slice_index=IntSlider(min=0, max=len(slices)-1, step=1, value=0))

# 以下は変更なし
example_person_dicom_dir_path = "/content/drive/Shareddrives/複数手法を用いた256画素CT画像の主観評価/本番環境/data/2_1_島村先生にアップロード頂いたファイルの整理dataのうち必要な患者のseries2のdata/0017_20190829_2"
slices = process_dicom_directory(example_person_dicom_dir_path)
area_proportion_list, area_proportion_in_circle_list, region_list = analyze_slices(slices)

# 結果の表示
print("Area Proportion List:", area_proportion_list)
print("area_proportion_in_circle List:", area_proportion_in_circle_list)

# インタラクティブな可視化
visualize_results(slices, area_proportion_list, area_proportion_in_circle_list, region_list)









import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cv2
from ipywidgets import interact, IntSlider

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

def find_max_inscribedcircle(mask):
    # マスクの輪郭を取得
    contours,  = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0

    # x座標の制限（256±40）
    x_min, x_max = 256 - 40, 256 + 40

    # 距離マップを計算
    dist_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

    # x座標制限を適用
    dist_map[:, :x_min] = 0
    dist_map[:, x_max:] = 0

    # 最大距離とその位置を取得
    maxval, max_loc = cv2.minMaxLoc(dist_map)

    return max_loc, max_val

def visualize_sternum_with_edges_and_circle(slice_image, mask):
    upper_edge, lower_edge = find_and_interpolate_edges(mask)
    center, radius = find_max_inscribed_circle(mask)

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

def visualize_results(slices, area_proportion_list, area_proportion_in_circle_list, region_list):
    def plot_slice(slice_index):
        plt.figure(figsize=(20, 10))

        # Original slice
        plt.subplot(231)
        plt.imshow(slices[slice_index], cmap='gray')
        plt.title(f'Original Slice {slice_index}')

        # Sternum region
        plt.subplot(232)
        mask = np.zeros_like(slices[slice_index], dtype=bool)
        for y, x in region_list[slice_index]:
            mask[y, x] = True
        plt.imshow(slices[slice_index], cmap='gray')
        plt.imshow(mask, cmap='Reds', alpha=0.3)
        plt.title(f'Sternum Region (Slice {slice_index})')

        # Metrics
        plt.subplot(233)
        plt.text(0.1, 0.6, f'Area Proportion: {area_proportion_list[slice_index]:.2f}%', fontsize=12)
        plt.text(0.1, 0.4, f'area_proportion_in_circle: {area_proportion_in_circle_list[slice_index]:.4f}', fontsize=12)
        plt.axis('off')
        plt.title('Metrics')

        # New visualization with edges and inscribed circle
        plt.subplot(212)
        visualize_sternum_with_edges_and_circle(slices[slice_index], mask)

        plt.tight_layout()
        plt.show()

        # Area Proportion Plot
        plt.subplot(234)
        plt.plot(area_proportion_list)
        plt.title('Area Proportion')
        plt.xlabel('Slice Index')
        plt.ylabel('Area Proportion (%)')
        max_area_index = np.argmax(area_proportion_list)
        max_area_index_partial = np.argmax(area_proportion_list[10:]) + 10
        plt.text(0.65, 0.95, f'Max (All): {max_area_index}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.65, 0.85, f'Max (10+): {max_area_index_partial}', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
        plt.axvline(x=slice_index, color='r', linestyle='--')
        plt.axvline(x=10, color='g', linestyle=':')

        # area_proportion_in_circle_list Plot
        plt.subplot(235)
        plt.plot(area_proportion_in_circle_list)
        plt.title('area_proportion_in_circle')
        plt.xlabel('Slice Index')
        plt.ylabel('area_proportion_in_circle')
        max_circularity_index = np.argmax(area_proportion_in_circle_list)
        max_circularity_index_partial = np.argmax(area_proportion_in_circle_list[10:]) + 10
        #plt.text(0.65, 0.95, f'Max (All): {max_index}', transform=plt.gca().transAxes, fontsize=10)
        #plt.text(0.65, 0.85, f'Max (10+): {max_index_partial}', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
        plt.axvline(x=slice_index, color='r', linestyle='--')
        plt.axvline(x=10, color='g', linestyle=':')

        plt.tight_layout()
        plt.show()

    interact(plot_slice, slice_index=IntSlider(min=0, max=len(slices)-1, step=1, value=0))

example_person_dicom_dir_path = "/content/drive/Shareddrives/複数手法を用いた256画素CT画像の主観評価/本番環境/data/2_1_島村先生にアップロード頂いたファイルの整理dataのうち必要な患者のseries2のdata/0017_20190829_2"
slices = process_dicom_directory(example_person_dicom_dir_path)
area_proportion_list, area_proportion_in_circle_list, region_list = analyze_slices(slices)

# 結果の表示
print("Area Proportion List:", area_proportion_list)
print("area_proportion_in_circle List:", area_proportion_in_circle_list)

# インタラクティブな可視化
visualize_results(slices, area_proportion_list, area_proportion_in_circle_list, region_list)


















import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import matplotlib.patches as patches

def analyze_slices(slices):
    area_proportion_list = []
    circularity_list = []
    inside_circle_proportion_list = []
    region_list = []

    for slice in slices:
        mask = create_sternum_mask(slice)  # この関数は別途定義が必要です
        center, radius = find_max_inscribed_circle(mask)
        
        # 胸骨領域の面積
        sternum_area = np.sum(mask)
        
        # 画像全体の面積
        total_area = mask.size
        
        # Area Proportion（既存の定義）
        area_proportion = sternum_area / total_area * 100
        
        if center is not None and radius > 0:
            # 内接円の面積
            circle_area = np.pi * radius ** 2
            
            # 新しいCircularity定義
            circularity = circle_area / sternum_area * 100 if sternum_area > 0 else 0
            
            # Inside Circle Proportion
            y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
            circle_mask = ((x - center[0])**2 + (y - center[1])**2 <= radius**2)
            inside_circle_area = np.sum(mask & circle_mask)
            inside_circle_proportion = inside_circle_area / circle_area * 100 if circle_area > 0 else 0
        else:
            circularity = 0
            inside_circle_proportion = 0

        area_proportion_list.append(area_proportion)
        circularity_list.append(circularity)
        inside_circle_proportion_list.append(inside_circle_proportion)
        region_list.append(np.argwhere(mask))

    return area_proportion_list, circularity_list, inside_circle_proportion_list, region_list


def visualize_results(slices, area_proportion_list, circularity_list, inside_circle_proportion_list, region_list):
    def plot_slice(slice_index):
        fig, axs = plt.subplots(3, 3, figsize=(20, 20))
        fig.suptitle(f'Sternum Analysis - Slice {slice_index}', fontsize=16)
        
        # Original slice
        axs[0, 0].imshow(slices[slice_index], cmap='gray')
        axs[0, 0].set_title('Original Slice')
        
        # Sternum region
        mask = np.zeros_like(slices[slice_index], dtype=bool)
        for x, y in region_list[slice_index]:
            mask[y, x] = True
        axs[0, 1].imshow(slices[slice_index], cmap='gray')
        axs[0, 1].imshow(mask, cmap='Reds', alpha=0.3)
        axs[0, 1].set_title('Sternum Region')
        
        # Metrics
        axs[0, 2].axis('off')
        axs[0, 2].text(0.1, 0.7, f'Area Proportion: {area_proportion_list[slice_index]:.2f}%', fontsize=12)
        axs[0, 2].text(0.1, 0.5, f'Circularity: {circularity_list[slice_index]:.2f}%', fontsize=12)
        axs[0, 2].text(0.1, 0.3, f'Inside Circle Proportion: {inside_circle_proportion_list[slice_index]:.2f}%', fontsize=12)
        axs[0, 2].set_title('Metrics')

        # Area Proportion Plot
        axs[1, 0].plot(area_proportion_list)
        axs[1, 0].set_title('Area Proportion')
        axs[1, 0].set_xlabel('Slice Index')
        axs[1, 0].set_ylabel('Area Proportion (%)')
        max_area_index = np.argmax(area_proportion_list)
        max_area_index_partial = np.argmax(area_proportion_list[10:]) + 10
        axs[1, 0].text(0.65, 0.95, f'Max (All): {max_area_index}', transform=axs[1, 0].transAxes, fontsize=10)
        axs[1, 0].text(0.65, 0.85, f'Max (10+): {max_area_index_partial}', transform=axs[1, 0].transAxes, fontsize=12, fontweight='bold')
        axs[1, 0].axvline(x=slice_index, color='r', linestyle='--')
        axs[1, 0].axvline(x=10, color='g', linestyle=':')

        # Circularity Plot
        axs[1, 1].plot(circularity_list)
        axs[1, 1].set_title('Circularity')
        axs[1, 1].set_xlabel('Slice Index')
        axs[1, 1].set_ylabel('Circularity (%)')
        max_circularity_index = np.argmax(circularity_list)
        max_circularity_index_partial = np.argmax(circularity_list[10:]) + 10
        axs[1, 1].text(0.65, 0.95, f'Max (All): {max_circularity_index}', transform=axs[1, 1].transAxes, fontsize=10)
        axs[1, 1].text(0.65, 0.85, f'Max (10+): {max_circularity_index_partial}', transform=axs[1, 1].transAxes, fontsize=12, fontweight='bold')
        axs[1, 1].axvline(x=slice_index, color='r', linestyle='--')
        axs[1, 1].axvline(x=10, color='g', linestyle=':')

        # Inside Circle Proportion Plot
        axs[1, 2].plot(inside_circle_proportion_list)
        axs[1, 2].set_title('Inside Circle Proportion')
        axs[1, 2].set_xlabel('Slice Index')
        axs[1, 2].set_ylabel('Inside Circle Proportion (%)')
        max_inside_circle_index = np.argmax(inside_circle_proportion_list)
        max_inside_circle_index_partial = np.argmax(inside_circle_proportion_list[10:]) + 10
        axs[1, 2].text(0.65, 0.95, f'Max (All): {max_inside_circle_index}', transform=axs[1, 2].transAxes, fontsize=10)
        axs[1, 2].text(0.65, 0.85, f'Max (10+): {max_inside_circle_index_partial}', transform=axs[1, 2].transAxes, fontsize=12, fontweight='bold')
        axs[1, 2].axvline(x=slice_index, color='r', linestyle='--')
        axs[1, 2].axvline(x=10, color='g', linestyle=':')

        # New visualization with edges and inscribed circle
        visualize_sternum_with_edges_and_circle(slices[slice_index], mask, ax=axs[2, 1])
        
        # Remove empty subplots
        axs[2, 0].axis('off')
        axs[2, 2].axis('off')

        plt.tight_layout()
        plt.show()

    interact(plot_slice, slice_index=IntSlider(min=0, max=len(slices)-1, step=1, value=0))


def visualize_sternum_with_edges_and_circle(slice_image, mask, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    upper_edge, lower_edge = find_and_interpolate_edges(mask)
    center, radius = find_max_inscribed_circle(mask)

    ax.imshow(slice_image, cmap='gray')
    ax.imshow(mask, alpha=0.3, cmap='Reds')
    
    # Plot edges
    #x_range = np.arange(len(upper_edge))
    #ax.plot(x_range, upper_edge, color='blue', linewidth=2)
    #ax.plot(x_range, lower_edge, color='blue', linewidth=2)

    # Plot inscribed circle
    if center is not None and radius > 0:
        try:
            circle = patches.Circle(center, radius, color='yellow', fill=False, linewidth=2)
            ax.add_patch(circle)
        except Exception as e:
            print(f"Error drawing circle: {e}")
            print(f"Center: {center}, Radius: {radius}")

    # Plot x-coordinate constraints
    ax.axvline(x=256-40, color='green', linestyle='--', linewidth=1)
    ax.axvline(x=256+40, color='green', linestyle='--', linewidth=1)

    ax.set_title('Sternum with Edges and Max Inscribed Circle')
    ax.axis('off')


# メイン処理
example_person_dicom_dir_path = "/content/drive/Shareddrives/複数手法を用いた256画素CT画像の主観評価/本番環境/data/2_1_島村先生にアップロード頂いたファイルの整理dataのうち必要な患者のseries2のdata/0024_20201210_2"

slices = process_dicom_directory(example_person_dicom_dir_path)
area_proportion_list, circularity_list, inside_circle_proportion_list, region_list = analyze_slices(slices)

# 結果の表示
print("Area Proportion List:", area_proportion_list)
print("Circularity List:", circularity_list)
print("Inside Circle Proportion List:", inside_circle_proportion_list)

# インタラクティブな可視化
visualize_results(slices, area_proportion_list, circularity_list, inside_circle_proportion_list, region_list)
