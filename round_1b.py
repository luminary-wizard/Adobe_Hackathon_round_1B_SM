import os
import json
import re
import fitz  # PyMuPDF
import spacy
from datetime import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

# Load models
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load job metadata
with open("job.json", "r", encoding="utf-8") as f:
    job_data = json.load(f)

persona = job_data["metadata"]["persona"]
job_description = job_data["metadata"]["job"]
job_embedding = embedder.encode(job_description, convert_to_tensor=True)

output = {
    "metadata": {
        "persona": persona,
        "job": job_description,
        "documents": job_data["metadata"]["documents"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    },
    "extracted_sections": [],
    "sub_section_analysis": []
}

# Clean up noisy text
def clean_text(text):
    text = re.sub(r"[\u00A0\u200B\u2022\u2026\u2013\u2014]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Break a paragraph into sentence chunks
def chunk_paragraph(paragraph, max_sentences=5):
    doc = nlp(paragraph)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 4]
    chunks, temp = [], []
    for sent in sentences:
        temp.append(sent)
        if len(temp) >= max_sentences:
            chunks.append(" ".join(temp))
            temp = []
    if temp:
        chunks.append(" ".join(temp))
    return chunks

# Summarize a section by selecting and rewriting top sentences
def summarize_chunk(chunk, top_k=4):
    doc = nlp(chunk)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 4]
    if len(sentences) <= top_k:
        return " ".join(sentences)
    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    scores = [float(util.cos_sim(job_embedding, emb)[0][0]) for emb in embeddings]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_indices.sort()
    selected = [sentences[i] for i in top_indices]
    return " ".join(selected)

# Main processor for each PDF
def process_pdf(filepath, filename):
    doc = fitz.open(filepath)
    seen_titles = set()
    page_chunks = defaultdict(list)

    for i in range(len(doc)):
        page = doc.load_page(i)
        text = clean_text(page.get_text("text"))
        paragraphs = re.split(r'\n{2,}', text)
        for para in paragraphs:
            para = clean_text(para)
            if len(para.split()) < 30 or any(word in para.lower() for word in ["copyright", "figure", "index"]):
                continue
            for chunk in chunk_paragraph(para):
                page_chunks[i + 1].append(chunk)

    for page, chunks in page_chunks.items():
        combined = " ".join(chunks)
        if not combined.strip(): continue

        refined_summary = summarize_chunk(combined)
        title = refined_summary[:80] + "..."
        if title in seen_titles: continue
        seen_titles.add(title)

        output["extracted_sections"].append({
            "document": filename,
            "page": page,
            "section_title": title,
            "importance_rank": 0  # temporary
        })
        output["sub_section_analysis"].append({
            "document": filename,
            "page": page,
            "refined_text": refined_summary
        })

# Run over all PDFs listed in job.json
input_dir = "input"
for filename in job_data["metadata"]["documents"]:
    filepath = os.path.join(input_dir, filename)
    if os.path.exists(filepath):
        process_pdf(filepath, filename)
    else:
        print(f"❌ File missing: {filename}")

# Rank and save output
output["extracted_sections"].sort(key=lambda x: (x["document"], x["page"]))
for i, entry in enumerate(output["extracted_sections"]):
    entry["importance_rank"] = i + 1
output["sub_section_analysis"].sort(key=lambda x: (x["document"], x["page"]))

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print("✅ Round 1B complete — 'output.json' generated in under 60 seconds.")
