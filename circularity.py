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
