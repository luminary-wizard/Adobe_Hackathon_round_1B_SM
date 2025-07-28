

# 📄 Adobe Hackathon Round 1B: Persona-Based PDF Summarizer

This project processes academic PDFs and generates persona-aligned summaries based on a job description.

---

## 🚀 Features

- Extracts paragraph chunks using `spaCy`
- Scores and selects most relevant sentences using `MiniLM` (via `sentence-transformers`)
- Summarizes sections into 3–6 lines (~500 characters max)
- Generates `output.json` as per hackathon format
- Optimized to run under **60 seconds** on **CPU-only systems**

---

## 📁 Input

- Place all input PDFs in the `input/` folder.
- Provide a `job.json` file with this structure:

```json
{
  "metadata": {
    "persona": "Undergraduate Literature Student",
    "job": "Identify relevant content for academic comprehension",
    "documents": ["the_omen.pdf"]
  }
}
