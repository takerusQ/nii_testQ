# シーンのクリーンアップ
print("シーンのクリーンアップを実行中...")

# すべてのノードを削除
slicer.mrmlScene.Clear(0)

# モジュールをリセット
if hasattr(slicer.modules, 'monaiauto3dseg'):
    widget = slicer.modules.monaiauto3dseg.widgetRepresentation()
    if widget:
        # 入力/出力セレクターをクリア
        selectors = ['inputNodeSelector0', 'outputSegmentationSelector']
        for selector_name in selectors:
            selector = find_widget(widget, slicer.qMRMLNodeComboBox, selector_name)
            if selector:
                selector.setCurrentNode(None)
        
        # プログレスバーをリセット
        progress_bar = find_widget(widget, qt.QProgressBar, 'progressBar')
        if progress_bar:
            progress_bar.reset()

# メモリの解放を促す
gc.collect()

print("クリーンアップ完了")

# ログファイルを閉じる
sys.stdout.log_file.close()
sys.stdout = sys.stdout.terminal





from pydub import AudioSegment
import os

def create_looped_mp3_with_silence(input_file, output_file, num_loops=3, silence_duration=3000):
    """
    Create a new MP3 file by repeating the input file multiple times with silent intervals.
    
    :param input_file: Path to the input MP3 file
    :param output_file: Path to save the output MP3 file
    :param num_loops: Number of times to loop the audio (default: 3)
    :param silence_duration: Duration of silence between loops in milliseconds (default: 3000ms = 3s)
    """
    # Load the audio file
    audio = AudioSegment.from_mp3(input_file)
    
    # Create silent segment
    silence = AudioSegment.silent(duration=silence_duration)
    
    # Create the looped audio with silent intervals
    looped_audio = AudioSegment.empty()
    for i in range(num_loops):
        looped_audio += audio
        if i < num_loops - 1:  # Don't add silence after the last loop
            looped_audio += silence
    
    # Export the final audio
    looped_audio.export(output_file, format="mp3")
    
    # Verify the output file length
    output_audio = AudioSegment.from_mp3(output_file)
    actual_length = len(output_audio)
    expected_length = (len(audio) * num_loops) + (silence_duration * (num_loops - 1))
    
    print(f"Original audio length: {len(audio)} ms")
    print(f"Expected output length: {expected_length} ms")
    print(f"Actual output length: {actual_length} ms")
    
    if abs(actual_length - expected_length) <= 1:  # Allow 1ms tolerance
        print(f"Successfully created looped MP3 with silent intervals: {output_file}")
    else:
        print(f"Warning: Output file length does not match expected length. Difference: {actual_length - expected_length} ms")

# Usage
input_file = "/content/drive/MyDrive/March of Victory.mp3"
output_file = "/content/drive/MyDrive/March of Victory (Looped with Silence).mp3"
create_looped_mp3_with_silence(input_file, output_file)
