# 🎥 YouTube Video Summarizer & Q&A Chatbot

An AI-powered web application built using **Streamlit** that extracts transcripts from YouTube videos, generates concise summaries, and allows users to ask questions based on the video content.

---

## 🚀 Features

- 🔗 **YouTube Transcript Extraction**
  - Supports multiple YouTube URL formats  
  - Fetches captions using `youtube-transcript-api`  

- 🧠 **AI-Based Summarization**
  - Uses **FLAN-T5 (google/flan-t5-base)**  
  - Handles long transcripts using chunking  
  - Generates structured summaries  

- ❓ **Question Answering (Q&A)**
  - Uses **DistilBERT (SQuAD model)**  
  - Extracts answers from transcript context  
  - Selects best answer based on confidence score  

- 💻 **Interactive UI**
  - Built with **Streamlit**  
  - Simple and user-friendly interface  

---

## 🧠 Tech Stack

- **Frontend/UI**: Streamlit  
- **Backend**: Python  
- **NLP Models**:
  - FLAN-T5 (Summarization)
  - DistilBERT (Q&A)
- **Libraries**:
  - Transformers (Hugging Face)
  - PyTorch
  - YouTube Transcript API

---

## 📂 Project Structure

```

├── app.py              # Main Streamlit app
├── youtube.py          # App launcher
├── requirements.txt    # Dependencies
├── .gitignore
├── README.md

````

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/karava22/YouTube-Video-Summarizer-and-Q-A-Chatbot-main
cd YouTube-Video-Summarizer-and-Q-A-Chatbot-main
````

---

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

**OR**

```bash
python youtube.py
```

---

## 🌐 Access the App

Open your browser and go to:

```
http://localhost:8501
```

---

## 🚀 Live Demo

Coming Soon

---

## 🛠️ How It Works

1. User enters a YouTube URL
2. App extracts transcript using API
3. Transcript is split into chunks
4. FLAN-T5 generates summary
5. DistilBERT answers user questions

---

## ⚠️ Limitations

* Works only for videos with available captions
* Long videos may take more processing time
* Performance depends on system hardware

---

## 🔮 Future Improvements

* Chat-style interface
* Multi-language support
* Cloud deployment (Streamlit Cloud / AWS)
* Integration with advanced LLMs

---

## 📌 Conclusion

This project demonstrates a **multi-stage NLP pipeline** combining:

* Text extraction
* Summarization
* Question answering

