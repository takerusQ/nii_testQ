from pdf2image import convert_from_path
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip, vfx
import os
import tempfile
import logging
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_pdf_to_images(pdf_path, output_folder):
    """Convert PDF to images"""
    try:
        images = convert_from_path(pdf_path)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f'slide_{i+1}.png')
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
        logging.info(f"Converted PDF to {len(image_paths)} images")
        return image_paths
    except Exception as e:
        logging.error(f"Failed to convert PDF to images: {e}")
        raise

def create_continuous_looping_audio(audio_clip, total_duration, fade_duration=1):
    """Create continuous looping audio with fade in/out and no silent end"""
    looped_clips = []
    current_duration = 0
    
    while current_duration < total_duration:
        remaining_duration = total_duration - current_duration
        if remaining_duration >= audio_clip.duration:
            clip = audio_clip.copy()
        else:
            clip = audio_clip.subclip(0, remaining_duration)
        
        # Apply fade in to the first clip
        if current_duration == 0:
            clip = clip.audio_fadein(fade_duration)
        
        # Apply fade out to the last clip
        if current_duration + clip.duration >= total_duration:
            clip = clip.audio_fadeout(fade_duration)
        
        looped_clips.append(clip)
        current_duration += clip.duration

    return CompositeAudioClip(looped_clips)

def create_video_from_images(image_paths, output_video_path, music_path):
    """Create video from images"""
    try:
        durations = [5 if i < 3 else 10 for i in range(len(image_paths))]
        clips = [ImageClip(img).set_duration(dur) for img, dur in zip(image_paths, durations)]
        
        video = concatenate_videoclips(clips, method="compose")
        
        if os.path.exists(music_path):
            audio_clip = AudioFileClip(music_path)
            total_duration = video.duration
            
            looped_audio = create_continuous_looping_audio(audio_clip, total_duration)
            
            final_video = video.set_audio(looped_audio)
        else:
            logging.warning("Music file not found. Creating video without audio.")
            final_video = video
        
        final_video.write_videofile(output_video_path, fps=24, logger=None)
        logging.info(f"Video created successfully: {output_video_path}")
    
    except Exception as e:
        logging.error(f"Failed to create video: {e}")
        raise

def pdf_to_video(pdf_path, output_video_path, music_path):
    """Convert PDF to video"""
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Convert PDF to images
            image_paths = convert_pdf_to_images(pdf_path, temp_dir)
            
            # Create video from images
            create_video_from_images(image_paths, output_video_path, music_path)
            
        except Exception as e:
            logging.error(f"Failed to convert PDF to video: {e}")
            raise

# Usage example
try:
    name = "ChallengersDraftTraining1"
    pdf_path = "/content/drive/MyDrive/Challengers/"+name+".pdf"
    output_video_path = "/content/drive/MyDrive/Challengers/"+name+".mp4"
    music_path = "/content/drive/MyDrive/March of Victory.mp3"
    pdf_to_video(pdf_path, output_video_path, music_path)
except Exception as e:
    logging.error(f"An error occurred: {e}")