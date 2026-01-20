from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import os
import tensorflow as tf
import warnings

# Suppress TensorFlow deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

app = Flask(__name__)

# Load summarization model (BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Helper function to get transcript from YouTube video
def get_youtube_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([item['text'] for item in transcript])
    return text

# Helper function to split transcript into smaller chunks
def split_text(text, max_length=1024):
    """Split text into chunks that fit within the max token length for the summarizer."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_length = sentence_length
            
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")
        
    return chunks

# Adjust max_length based on input length
def adjust_max_length(input_text, max_length=150):
    input_length = len(input_text.split())
    return min(input_length, max_length)

# Route to serve the HTML page
@app.route("/")
def home():
    return render_template("index.html")  # The HTML file should be placed in a 'templates' folder

# Route to summarize a YouTube transcript
@app.route("/summarize_youtube", methods=["POST"])
def summarize_youtube():
    data = request.json
    video_id = data.get("video_id")
    
    if not video_id:
        return jsonify({"error": "No video ID provided"}), 400
    
    # Get transcript from YouTube
    try:
        transcript = get_youtube_transcript(video_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    # Split the transcript into smaller chunks
    chunks = split_text(transcript)
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        # Adjust max_length based on the input length
        max_len = adjust_max_length(chunk, max_length=150)
        summary = summarizer(chunk, max_length=max_len, min_length=50, do_sample=False, clean_up_tokenization_spaces=True)
        summaries.append(summary[0]['summary_text'])
    
    # Join the summaries
    final_summary = " ".join(summaries)
    
    return jsonify({"summary": final_summary})

# Route to summarize a provided transcript
@app.route("/summarize_transcript", methods=["POST"])
def summarize_transcript():
    data = request.json
    transcript = data.get("transcript")
    
    if not transcript:
        return jsonify({"error": "Transcript not provided"}), 400
    
    # Split the transcript into smaller chunks
    chunks = split_text(transcript)
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        max_len = adjust_max_length(chunk, max_length=150)
        summary = summarizer(chunk, max_length=max_len, min_length=50, do_sample=False, clean_up_tokenization_spaces=True)
        summaries.append(summary[0]['summary_text'])
    
    # Join the summaries
    final_summary = " ".join(summaries)
    
    return jsonify({"summary": final_summary})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
