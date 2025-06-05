import streamlit as st
from st_copy_to_clipboard import st_copy_to_clipboard
import spacy
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import os

nltk.download('punkt')



# Initialize spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    st.error(f"Error loading spaCy model: {str(e)}")
    nlp = None

# Session state init
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'text_input_source_method' not in st.session_state:
    st.session_state.text_input_source_method = None
if 'edited_summary' not in st.session_state:
    st.session_state.edited_summary = None
if 'pasted_text_area_buffer' not in st.session_state:
    st.session_state.pasted_text_area_buffer = ""

# --- Generalized Extractive Summarization ---
def extractive_summary(text, max_sentences=5):
    try:
        if not text.strip():
            return "No valid text found."

        # Use spaCy to split into clean sentences
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not sentences:
            return "No valid sentences found in the text."

        if len(sentences) < max_sentences:
            max_sentences = len(sentences)

        # TF-IDF scoring
        vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            max_df=0.85,
            min_df=1,
            ngram_range=(1, 3)
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

        # Optional: apply position-based weighting
        position_weight = np.linspace(0.8, 1.2, len(sentences))
        sentence_scores *= position_weight

        top_indices = np.argsort(-sentence_scores)[:max_sentences]
        top_indices = sorted(top_indices)
        selected = [sentences[i] for i in top_indices]

        # Clean sentences
        cleaned = []
        for s in selected:
            s = s.strip()
            if not s:
                continue
            if not s.endswith(('.', '!', '?')):
                s += '.'
            s = s[0].upper() + s[1:]  # Capitalize first letter
            cleaned.append(s)

        return "\n\n".join(cleaned)

    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return "Summary generation failed. Please try again by adding more text."

# --- File text extraction ---
def extract_text(file):
    try:
        if file.type == "text/plain":
            text = file.read().decode("utf-8")
        elif file.type == "application/pdf":
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "".join([page.get_text() for page in doc])
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            import docx
            doc = docx.Document(file)
            text = " ".join([para.text for para in doc.paragraphs])
        else:
            st.error("Unsupported file type.")
            return ""
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

# --- UI Layout ---
st.markdown("""
    <div style='background-color: #4B8BBE; padding: 12px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h1 style='text-align: center; color: white; font-size: 38px; font-weight: 600; letter-spacing: 0.5px;'>AI-Powered Text Summarizer</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background-color: #e6f3ff; padding: 6px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>
    <p style='color: #004280;'>Upload a file or paste text to generate a customizable summary.</p>
</div>
""", unsafe_allow_html=True)

# Add spacing
st.markdown("""
<br><br>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("📄 Upload a file", type=["txt", "pdf", "docx"])
if uploaded_file:
    extracted_content = extract_text(uploaded_file)
    if extracted_content:
        st.session_state.text_input = extracted_content
        st.session_state.text_input_source_method = "📄 Upload file"
        st.success("File loaded successfully.")
    else:
        st.error("Failed to extract text from file")

# Paste text manually
text_input_from_area = st.text_area(
    "✍️ Paste your text below:",
    value=st.session_state.pasted_text_area_buffer,
    height=300
)
st.session_state.pasted_text_area_buffer = text_input_from_area

if st.button("Summarize"):
    if text_input_from_area.strip():
        st.session_state.text_input = text_input_from_area.strip()
        st.session_state.text_input_source_method = "✍️ Paste text"
    else:
        st.warning("Please paste some text first.")
        st.session_state.text_input = ""
        st.session_state.text_input_source_method = None

# --- Summary formatting options ---
st.subheader("🔧 Formatting Options")

# Summary length control
max_sentences = st.slider("Number of sentences in summary", min_value=1, max_value=10, value=5)

# Format selection
format_option = st.radio("Choose summary format:", 
                        ["Paragraph", "Bullet Points"],
                        help="Select how you want your summary to be formatted")

# Text styling options
font_size = st.selectbox("Font size", ["Small", "Medium", "Large"],
                         help="Adjust the font size of your summary")
font_css = {"Small": "15px", "Medium": "20px", "Large": "28px"}

# --- Summary output ---
if st.session_state.text_input.strip():
    with st.spinner("Generating summary..."):
        summary = extractive_summary(st.session_state.text_input, max_sentences=max_sentences)
    
    if summary and summary.strip():
        # Display word count comparison
        original_text = st.session_state.text_input.strip()
        summary_text = summary.strip()
        
        original_words = len(original_text.split())
        summary_words = len(summary_text.split())
        
        st.write(f"📊 Original text: {original_words} words")
        st.write(f"📉 Summary: {summary_words} words")
        
        st.subheader("📝 Generated Summary")
        
        # Format the summary based on user preference
        if format_option == "Bullet Points":
            formatted_summary = "\n\n".join([f"• {s}" for s in summary.split('\n\n')])
        else:
            formatted_summary = summary

        # Apply font styling
        st.markdown(f"<div style='font-size:{font_css[font_size]}'>{formatted_summary}</div>", 
                    unsafe_allow_html=True)

        edited_summary = st.text_area("Edit the summary if needed:", 
                                    value=formatted_summary, 
                                    height=200)

        col1, col2 = st.columns(2)
        with col1:
            st_copy_to_clipboard(edited_summary, "Copy Summary")
        with col2:
            if st.button("Save Summary", key="save_summary_btn"):
                # Create a unique filename with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"summary_{timestamp}.txt"
                
                # Create summaries directory if it doesn't exist
                import os
                os.makedirs("summaries", exist_ok=True)
                
                # Save to summaries folder
                filepath = os.path.join("summaries", filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(edited_summary.strip())
                st.success(f"Summary saved to `summaries/{filename}`")
    else:
        st.error("Summary generation failed. Try using more content.")
