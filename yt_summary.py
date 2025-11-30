import streamlit as st
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCppInvocationLayer
import time
import yt_dlp
import os
from haystack.nodes import PromptNode


st.set_page_config(
    layout="wide"
)

def download_video(youtube_url):
    """
    Downloads the best audio-only stream from a YouTube URL using yt-dlp.
    Returns the path to the downloaded file.
    """
    # Define a consistent filename for the downloaded audio
    output_filename = "downloaded_audio.m4a"

    # Set the options for yt-dlp
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',  # Get the best audio-only stream, prefer m4a format
        'outtmpl': output_filename,       # Save the file with our defined name
        'noplaylist': True,             # Only download a single video, not a whole playlist
        'quiet': True,                  # Don't print a lot of logs
        'overwrites': True              # Overwrite the file if it already exists
    }

    try:
        # The main download command
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        # If download is successful, return the path
        if os.path.exists(output_filename):
            return output_filename
        else:
            return None
            
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None
def initialize_model(full_path):
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCppInvocationLayer,
        use_gpu=False,
        max_length=512
    )

def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

def transcribe_audio(file_path, prompt_node):
    whisper = WhisperTranscriber()
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    output = pipeline.run(file_paths=[file_path])
    return output

def main():

    # Set the title and background color
    st.title("YouTube Video Summarizer 🎥")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Built with the Llama 2 🦙, Haystack, Streamlit and ❤️')
    st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

    # Expander for app details
    with st.expander("About the App"):
        st.write("This app allows you to summarize while watching a YouTube video.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start. This app is built by AI Anytime.")

    # Input box for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL")

    # Submit button
    if st.button("Submit") and youtube_url:
        start_time = time.time()  # Start the timer
        # Download video
        file_path = download_video(youtube_url)
        if file_path is None:
           st.error("Error: Could not download the video audio.")
           st.info("This might be because the video is age-restricted, private, or requires a login.")
           st.stop()
        st.info("Audio downloaded successfully. Initializing model...")
        full_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"
        model = initialize_model(full_path)
        prompt_node = PromptNode(model_name_or_path=model)
        st.info("Transcribing audio... (this may take a moment)")
        output = transcribe_audio(file_path, prompt_node)

        # Initialize model
        full_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"
        model = initialize_model(full_path)
        prompt_node = prompt_node = initialize_prompt_node(model)
        # Transcribe audio
        output = transcribe_audio(file_path, prompt_node)

        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time

        # Display layout with 2 columns
        col1, col2 = st.columns([1,1])

        # Column 1: Video view
        with col1:
            st.video(youtube_url)

        # Column 2: Summary View
        with col2:
            st.header("Summarization of YouTube Video")
            st.write(output)
            st.success(output["results"][0].split("\n\n[INST]")[0])
            st.write(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()