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
    
    # Calculate the duration of the audio minus the crossfade duration
    loop_duration = len(audio) - crossfade_duration
    
    # Create the looped audio
    looped_audio = AudioSegment.empty()
    for i in range(num_loops):
        if i == 0:
            looped_audio += audio
        else:
            looped_audio = looped_audio.overlay(audio, position=i*loop_duration)
    
    # Export the final audio
    looped_audio.export(output_file, format="mp3")
    print(f"Created looped MP3: {output_file}")

# Usage
input_file = "/content/drive/MyDrive/March of Victory.mp3"
output_file = "/content/drive/MyDrive/March of Victory (Looped).mp3"
create_looped_mp3(input_file, output_file)