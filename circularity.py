from pydub import AudioSegment
import os

def create_looped_mp3(input_file, output_file, num_loops=3, crossfade_duration=3000):
    """
    Create a new MP3 file by looping the input file multiple times with crossfade.
    
    :param input_file: Path to the input MP3 file
    :param output_file: Path to save the output MP3 file
    :param num_loops: Number of times to loop the audio (default: 3)
    :param crossfade_duration: Duration of crossfade in milliseconds (default: 3000ms = 3s)
    """
    # Load the audio file
    audio = AudioSegment.from_mp3(input_file)
    
    # Create the looped audio
    looped_audio = audio * num_loops
    
    # Apply crossfade between loops
    for i in range(1, num_loops):
        position = len(audio) * i - crossfade_duration
        looped_audio = looped_audio.overlay(audio, position=position, gain_during_overlay=0)
    
    # Ensure the final length is exactly num_loops times the original length
    expected_length = len(audio) * num_loops
    if len(looped_audio) > expected_length:
        looped_audio = looped_audio[:expected_length]
    
    # Export the final audio
    looped_audio.export(output_file, format="mp3")
    
    # Verify the output file length
    output_audio = AudioSegment.from_mp3(output_file)
    actual_length = len(output_audio)
    expected_length = len(audio) * num_loops
    
    print(f"Original audio length: {len(audio)} ms")
    print(f"Expected output length: {expected_length} ms")
    print(f"Actual output length: {actual_length} ms")
    
    if abs(actual_length - expected_length) <= 1:  # Allow 1ms tolerance
        print(f"Successfully created looped MP3: {output_file}")
    else:
        print(f"Warning: Output file length does not match expected length. Difference: {actual_length - expected_length} ms")

# Usage
input_file = "/content/drive/MyDrive/March of Victory.mp3"
output_file = "/content/drive/MyDrive/March of Victory (Looped).mp3"
create_looped_mp3(input_file, output_file)