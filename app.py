import re
from urllib.parse import parse_qs, urlparse

import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer
from youtube_transcript_api import YouTubeTranscriptApi


@st.cache_resource
def get_text_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


@st.cache_resource
def get_qa_model():
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def generate_text(prompt: str, max_new_tokens: int = 160) -> str:
    tokenizer, model, device = get_text_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def extract_video_id(url: str) -> str | None:
    parsed = urlparse(url)

    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/") or None

    if parsed.netloc in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/")[-1].split("/")[0]

    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[?&/]|$)", url)
    return match.group(1) if match else None


def _join_transcript_items(items) -> str:
    parts = []
    for item in items:
        if hasattr(item, "text"):
            parts.append(item.text)
        elif isinstance(item, dict):
            parts.append(item.get("text", ""))
    return " ".join(parts).strip()


def _fetch_with_old_api(video_id: str) -> str:
    # Old youtube-transcript-api exposes class methods like get_transcript/list_transcripts.
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        text = _join_transcript_items(transcript_list)
        if text:
            return text
    except Exception:
        pass

    transcripts = list(YouTubeTranscriptApi.list_transcripts(video_id))

    for transcript in transcripts:
        try:
            if getattr(transcript, "is_translatable", False):
                text = _join_transcript_items(transcript.translate("en").fetch())
                if text:
                    return text
        except Exception:
            continue

    for transcript in transcripts:
        try:
            text = _join_transcript_items(transcript.fetch())
            if text:
                return text
        except Exception:
            continue

    raise RuntimeError("No usable transcript could be fetched for this video.")


def _fetch_with_new_api(video_id: str) -> str:
    # New youtube-transcript-api uses instance methods like fetch/list.
    api = YouTubeTranscriptApi()

    try:
        text = _join_transcript_items(api.fetch(video_id, languages=["en"]))
        if text:
            return text
    except Exception:
        pass

    try:
        text = _join_transcript_items(api.fetch(video_id))
        if text:
            return text
    except Exception:
        pass

    transcripts = []
    if hasattr(api, "list"):
        transcripts = list(api.list(video_id))

    for transcript in transcripts:
        try:
            if getattr(transcript, "is_translatable", False):
                text = _join_transcript_items(transcript.translate("en").fetch())
                if text:
                    return text
        except Exception:
            continue

    for transcript in transcripts:
        try:
            text = _join_transcript_items(transcript.fetch())
            if text:
                return text
        except Exception:
            continue

    raise RuntimeError("No usable transcript could be fetched for this video.")


def fetch_transcript(video_id: str) -> str:
    # Support both old and new youtube-transcript-api versions.
    if hasattr(YouTubeTranscriptApi, "get_transcript") and hasattr(
        YouTubeTranscriptApi, "list_transcripts"
    ):
        return _fetch_with_old_api(video_id)

    return _fetch_with_new_api(video_id)


def chunk_text(text: str, max_words: int = 450) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


def select_best_answer(question: str, context: str) -> tuple[str, float]:
    tokenizer, model, device = get_qa_model()
    best_answer = ""
    best_score = 0.0

    for chunk in chunk_text(context, max_words=220):
        inputs = tokenizer(
            question,
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=384,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        start_idx = int(torch.argmax(start_logits))
        end_idx = int(torch.argmax(end_logits))

        if end_idx < start_idx:
            continue

        score = torch.softmax(start_logits, dim=0)[start_idx].item() * torch.softmax(end_logits, dim=0)[end_idx].item()
        answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        if score > best_score and answer:
            best_answer = answer
            best_score = score

    return best_answer, best_score


def summarize_text(transcript: str) -> str:
    chunks = chunk_text(transcript, max_words=350)

    summaries = []
    for chunk in chunks:
        prompt = (
            "Summarize the following YouTube transcript chunk in clear bullet-style prose.\n\n"
            f"Transcript:\n{chunk}"
        )
        result = generate_text(prompt, max_new_tokens=170)
        if result:
            summaries.append(result)

    return "\n\n".join(summaries)


def answer_question(context: str, question: str) -> str:
    best_answer, best_score = select_best_answer(question, context)
    if not best_answer or best_score < 0.05:
        return "I could not find that in the video transcript."
    return best_answer


def main() -> None:
    st.set_page_config(page_title="YouTube Summarizer + Q&A", page_icon="🎥", layout="wide")
    st.title("YouTube Video Summarizer and Q&A")
    st.write("Paste a YouTube URL, generate a summary, and ask questions about the video.")

    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""

    video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Generate Summary", type="primary"):
        if not video_url.strip():
            st.error("Please enter a YouTube URL.")
        else:
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Could not extract a valid video ID from this URL.")
            else:
                try:
                    with st.spinner("Fetching transcript..."):
                        transcript = fetch_transcript(video_id)
                    with st.spinner("Generating summary (this can take some time)..."):
                        summary = summarize_text(transcript)

                    st.session_state.transcript = transcript
                    st.session_state.summary = summary
                    st.success("Summary generated.")
                except Exception as exc:
                    st.error(f"Failed to process this video: {exc}")

    if st.session_state.summary:
        st.subheader("Summary")
        st.text_area("Video Summary", st.session_state.summary, height=260)

    st.subheader("Ask a Question")
    question = st.text_input(
        "Question",
        placeholder="What is the main takeaway from this video?",
        key="qa_question",
    )

    if st.button("Get Answer"):
        if not st.session_state.summary or not st.session_state.transcript:
            st.warning("Generate a summary first so the app has video context.")
        elif not question.strip():
            st.warning("Please type a question.")
        else:
            try:
                context = st.session_state.transcript
                with st.spinner("Finding answer..."):
                    answer = answer_question(context, question)
                st.markdown(f"**Answer:** {answer}")
            except Exception as exc:
                st.error(f"Could not generate answer: {exc}")


if __name__ == "__main__":
    main()
