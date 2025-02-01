import os
import slicer
import time
import datetime
from io import StringIO
import sys
import qt
import ctk
import gc

class Logger:
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.log = StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def setup_logging(log_folder):
    """ログ出力の設定"""
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_folder, f"segmentation_log_{timestamp}.txt")
    sys.stdout = Logger(log_file_path)
    print(f"ログ記録開始: {log_file_path}")
    print(f"実行日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

def find_widget(parent, widget_type, name):
    """指定した型と名前のウィジェットを検索"""
    if parent is None:
        return None
    children = parent.findChildren(widget_type)
    for child in children:
        if child.objectName == name:
            return child
    return None

def dump_widget_info(widget, level=0):
    """ウィジェットの情報を表示（デバッグ用）"""
    if widget:
        indent = "  " * level
        print(f"{indent}Name: {widget.objectName}, Type: {type(widget).__name__}")
        for child in widget.children():
            dump_widget_info(child, level + 1)

def enable_cpu_mode(module_widget):
    """CPUモードを有効化"""
    try:
        # 詳細設定ボタンを展開
        advanced_button = find_widget(module_widget, ctk.ctkCollapsibleButton, 'advancedCollapsibleButton')
        if advanced_button:
            advanced_button.setChecked(True)
            print("詳細設定パネルを展開しました")
            slicer.app.processEvents()
            time.sleep(0.5)  # UIの更新を待機
            
            # CPUチェックボックスを見つけて設定
            cpu_checkbox = find_widget(module_widget, qt.QCheckBox, 'cpuCheckBox')
            if cpu_checkbox:
                current_state = cpu_checkbox.isChecked()
                print(f"現在のCPU設定: {'有効' if current_state else '無効'}")
                
                if not current_state:
                    cpu_checkbox.setChecked(True)
                    print("CPUモードを有効化しました")
                    slicer.app.processEvents()
                    time.sleep(0.5)  # 設定の反映を待機
                
                # 設定の確認
                if cpu_checkbox.isChecked():
                    print("CPU設定が正しく適用されました")
                else:
                    print("警告: CPU設定の適用に失敗した可能性があります")
            else:
                print("警告: CPUチェックボックスが見つかりません")
                print("利用可能なウィジェット:")
                dump_widget_info(module_widget)
    except Exception as e:
        print(f"CPUモード設定エラー: {str(e)}")
        import traceback
        print(traceback.format_exc())

def load_dicom_volume(input_path):
    """DICOMデータを読み込み"""
    try:
        dicom_files = [f for f in os.listdir(input_path) if f.endswith('.dcm')]
        if not dicom_files:
            print("DICOMファイルが見つかりません")
            return None
            
        first_file = os.path.join(input_path, dicom_files[0])
        volume_node = slicer.util.loadVolume(first_file)
        print(f"ボリューム読み込み成功: {volume_node.GetName()}")
        return volume_node
        
    except Exception as e:
        print(f"DICOMの読み込みエラー: {str(e)}")
        return None

def print_segment_names(node):
    """セグメンテーションノードの中身を確認"""
    if node and node.GetSegmentation():
        print(f"セグメント一覧 ({node.GetName()}):")
        for i in range(node.GetSegmentation().GetNumberOfSegments()):
            segment_id = node.GetSegmentation().GetNthSegmentID(i)
            segment_name = node.GetSegmentation().GetSegment(segment_id).GetName()
            print(f"- {segment_name}")

def create_new_segmentation(name, model_name, volume_node):
    """新しいセグメンテーションを作成してモデルを適用"""
    try:
        # セグメンテーションノードの作成
        segmentation_node = slicer.vtkMRMLSegmentationNode()
        slicer.mrmlScene.AddNode(segmentation_node)
        segmentation_node.SetName(name)
        segmentation_node.CreateDefaultDisplayNodes()

        # リファレンスジオメトリを設定
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

        # モジュールウィジェットの取得
        module_widget = slicer.modules.monaiauto3dseg.widgetRepresentation()
        if not module_widget:
            print("エラー: MONAIモジュールが見つかりません")
            return None
            
        print("MONAIモジュールを取得しました")
        
        # リモート処理を無効化
        remote_checkbox = find_widget(module_widget, qt.QCheckBox, 'remoteProcessingCheckBox')
        if remote_checkbox:
            current_state = remote_checkbox.isChecked()
            print(f"現在のリモート処理設定: {'有効' if current_state else '無効'}")
            if current_state:
                remote_checkbox.setChecked(False)
                print("リモート処理を無効化しました")
                slicer.app.processEvents()
                time.sleep(0.5)  # 設定の反映を待機
        else:
            print("警告: リモート処理チェックボックスが見つかりません")
        
        # CPUモードを有効化
        enable_cpu_mode(module_widget)
        
        # 入力ノードの設定
        input_selector = find_widget(module_widget, slicer.qMRMLNodeComboBox, 'inputNodeSelector0')
        if input_selector:
            input_selector.setCurrentNode(volume_node)
            print(f"入力ボリューム設定: {volume_node.GetName()}")
            
        # 出力セグメンテーションの設定
        output_selector = find_widget(module_widget, slicer.qMRMLNodeComboBox, 'outputSegmentationSelector')
        if output_selector:
            output_selector.setCurrentNode(segmentation_node)
            print(f"出力セグメンテーション設定: {segmentation_node.GetName()}")
            
        # モデルの選択
        model_list = find_widget(module_widget, qt.QListWidget, 'modelComboBox')
        if model_list:
            for i in range(model_list.count):
                if model_name in model_list.item(i).text():
                    model_list.setCurrentRow(i)
                    print(f"モデル選択: {model_name}")
                    break
                    
        # Applyボタンの実行
        apply_button = find_widget(module_widget, qt.QPushButton, 'applyButton')
        if apply_button:
            print("セグメンテーション処理開始")
            apply_button.click()
            
            # 進行状況の監視
            progress_bar = find_widget(module_widget, qt.QProgressBar, 'progressBar')
            start_time = time.time()
            last_progress = -1
            last_activity_time = start_time
            
            while progress_bar:
                current_progress = progress_bar.value
                current_time = time.time()
                
                # 進捗の変化を検出
                if current_progress != last_progress:
                    print(f"進行状況: {current_progress}%")
                    last_progress = current_progress
                    last_activity_time = current_time
                
                # 完了確認
                if current_progress == 100:
                    print("処理が完了しました")
                    time.sleep(5)  # 完了後の安定化待機
                    break
                    
                # 進捗が一定時間更新されない場合のタイムアウト（10分）
                if current_time - last_activity_time > 600:
                    print("警告: 進捗が10分間更新されていません")
                    break
                    
                # 全体のタイムアウト（30分）
                if current_time - start_time > 1800:
                    print("警告: 処理が30分のタイムアウト時間を超過しました")
                    break
                    
                slicer.app.processEvents()
                time.sleep(1)
            
            print("セグメンテーション処理完了")
        
        # セグメントの確認
        print_segment_names(segmentation_node)
        
        return segmentation_node
        
    except Exception as e:
        print(f"セグメンテーション作成エラー: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def copy_segments(target_node, source_node, segment_names):
    """セグメントをコピー"""
    try:
        if not target_node or not source_node:
            print("無効なノード")
            return
            
        print("元のセグメント:")
        print_segment_names(source_node)
            
        for segment_name in segment_names:
            source_id = source_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
            if source_id:
                print(f"セグメントをコピー: {segment_name}")
                target_node.GetSegmentation().CopySegmentFromSegmentation(
                    source_node.GetSegmentation(), source_id)
            else:
                print(f"セグメントが見つかりません: {segment_name}")
    except Exception as e:
        print(f"セグメントコピーエラー: {str(e)}")

def export_segmentation(node, output_path, reference_volume_node):
    """セグメンテーションをNIFTIとしてエクスポート"""
    try:
        # セグメンテーションのラベル値を設定
        segments = node.GetSegmentation()
        for i in range(segments.GetNumberOfSegments()):
            segment_id = segments.GetNthSegmentID(i)
            segment = segments.GetSegment(segment_id)
            segment_name = segment.GetName()

            # ラベル値の割り当て
            if segment_name == "L2 vertebra":
                segment.SetTag("Label", "1")
            elif segment_name == "L1 vertebra":
                segment.SetTag("Label", "2")
            elif segment_name == "T12 vertebra":
                segment.SetTag("Label", "3")
            elif segment_name == "T11 vertebra":
                segment.SetTag("Label", "4")
            elif segment_name == "T10 vertebra":
                segment.SetTag("Label", "5")
            elif segment_name == "T9 vertebra":
                segment.SetTag("Label", "6")
            elif segment_name == "T8 vertebra":
                segment.SetTag("Label", "7")
            elif segment_name == "T7 vertebra":
                segment.SetTag("Label", "8")
            elif segment_name == "T6 vertebra":
                segment.SetTag("Label", "9")
            elif segment_name == "T5 vertebra":
                segment.SetTag("Label", "10")
            elif segment_name == "T4 vertebra":
                segment.SetTag("Label", "11")
            elif segment_name == "T3 vertebra":
                segment.SetTag("Label", "12")
            elif segment_name == "T2 vertebra":
                segment.SetTag("Label", "13")
            elif segment_name == "T1 vertebra":
                segment.SetTag("Label", "14")
            elif segment_name == "C7 vertebra":
                segment.SetTag("Label", "15")
            elif segment_name == "left deep back muscle":
                segment.SetTag("Label", "16")
            elif segment_name == "right deep back muscle":
                segment.SetTag("Label", "17")
            elif segment_name == "aorta":
                segment.SetTag("Label", "18")

        # セグメンテーションをバイナリラベルマップ形式に変換
        node.GetSegmentation().CreateRepresentation("Binary labelmap")
        time.sleep(1)  # 変換待機

        # リファレンスジオメトリを設定
        node.SetReferenceImageGeometryParameterFromVolumeNode(reference_volume_node)
        
        # ラベルマップノードを作成
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        
        # セグメンテーションをラベルマップに変換
        converter = slicer.vtkSlicerSegmentationsModuleLogic()
        converter.ExportVisibleSegmentsToLabelmapNode(node, labelmap_node, reference_volume_node)
        
        # NIFTIとして保存
        properties = {
            'useCompression': 1,
            'referenceImageGeometry': reference_volume_node
        }
        slicer.util.saveNode(labelmap_node, output_path, properties)
        
        # 一時ノードを削除
        slicer.mrmlScene.RemoveNode(labelmap_node)
        print(f"エクスポート完了: {output_path}")
        
    except Exception as e:
        print(f"エクスポートエラー: {str(e)}")
        import traceback
        print(traceback.format_exc())

def process_single_case(volume_node, output_folder, case_id):
    """1ケースの処理を実行"""
    try:
        print(f"ケース {case_id} の処理開始...")
        
        # 各セグメンテーションの作成
        muscle_node = create_new_segmentation("muscle", "Muscles TS2", volume_node)
        if muscle_node:
            print("筋肉セグメンテーション完了")
            
        vertebra_node = create_new_segmentation("vertebra", "Vertebrae TS2", volume_node)
        if vertebra_node:
            print("脊椎セグメンテーション完了")
            
        aorta_node = create_new_segmentation("aorta", "Aorta", volume_node)
        if aorta_node:
            print("大動脈セグメンテーション完了")

        # セグメンテーション処理の後に追加
        print("\nセグメント確認:")
        for i in range(vertebra_node.GetSegmentation().GetNumberOfSegments()):
            segment_id = vertebra_node.GetSegmentation().GetNthSegmentID(i)
            segment = vertebra_node.GetSegmentation().GetSegment(segment_id)
            print(f"セグメント {i+1}:")
            print(f"  名前: {segment.GetName()}")
            print(f"  ID: {segment_id}")
            print(f"  ラベル値: {segment.GetTag('Label')}")
      
        # セグメントの統合
        if all([vertebra_node, muscle_node, aorta_node]):
            print("セグメントの統合開始...")
            copy_segments(vertebra_node, muscle_node, 
                        ["left deep back muscle", "right deep back muscle"])
            copy_segments(vertebra_node, aorta_node, ["aorta"])
            
            # エクスポート
            print("エクスポート開始...")
            output_path = os.path.join(output_folder, f"segmentation_ver2_{case_id}.nii.gz")
            export_segmentation(vertebra_node, output_path, volume_node)
        
        return True
        
    except Exception as e:
        print(f"処理エラー: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

# メイン処理の実行
input_path = r"C:\Users\suyam\Desktop\202501segmentation_with_yakkun\HUdatas\0006_20190712_2"
output_folder = r"C:\Users\suyam\Desktop\202501segmentation_with_yakkun\output"
log_folder = r"C:\Users\suyam\Desktop\202501segmentation_with_yakkun\logs"

try:
    # 出力フォルダとログフォルダの作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # ログ設定を初期化
    setup_logging(log_folder)

    # ケースIDの抽出（パスの最後のフォルダ名を使用）
    case_id = os.path.basename(input_path)

    # DICOMの読み込みと処理実行
    volume_node = load_dicom_volume(input_path)
    if volume_node:
        print("DICOM読み込み完了")
        if process_single_case(volume_node, output_folder, case_id):
            print(f"ケース {case_id} の処理が正常に完了しました")
        else:
            print(f"ケース {case_id} の処理中にエラーが発生しました")
    else:
        print("DICOMの読み込みに失敗しました")

except Exception as e:
    print(f"予期せぬエラーが発生しました: {str(e)}")
    import traceback
    print(traceback.format_exc())


