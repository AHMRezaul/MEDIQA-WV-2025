# MEDIQA-WV: RAG-based Visual Question Answering with LLaMA 4

This repository contains the implementation for the MEDIQA-WV 2025 shared task: **Visual Question Answering for Wound Images**.  
The system uses a Retrieval-Augmented Generation (RAG) pipeline with pretrained models to address multilingual, multimodal question answering based on wound-care images and structured metadata.

Task homepage: [https://sites.google.com/view/mediqa-2025/mediqa-wv](https://sites.google.com/view/mediqa-2025/mediqa-wv)

---

## 🔧 System Overview

### Components Used

- **Image Encoder**: [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)  
- **Text Encoder**: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
- **Language Model**: [`meta-llama/Llama-4-Scout-17B-16E-Instruct`](https://huggingface.co/meta-llama)  
- **Retriever Fusion**: Alpha = `0.5` (equal weighting between image and text similarity)

---

## 📚 Data Usage

- **Knowledge Source**: The **train** and **validation** splits of the dataset, along with their associated wound images.
- **Inference Input**: The **test set**, for which the model retrieves the closest matching cases using image and text similarity, and performs few-shot prompting using LLaMA 4.

---

## 🔁 Pipeline

### Step-by-step Instructions

1. **Build Text Embedding Index**
   ```bash
   python build_corpus.py
   ```

2. **Build Image Embedding Index**
   ```bash
   python build_image_index.py
   ```

3. **Run Inference with LLaMA 4**
   ```bash
   python llama4_batch.py
   ```

4. **Post-process Results**
   ```bash
   cd result/
   python pre.py
   python post.py
   ```

The final output will be saved as `prediction.json` in the `result/` directory.

---

## 📂 Directory Structure

```
.
├── build_corpus.py           # Builds FAISS index for text
├── build_image_index.py      # Builds FAISS index for images
├── llama4_batch.py           # Inference script with LLaMA 4
├── retriever.py              # Unified image+text retriever
├── result/
│   ├── pre.py
│   ├── post.py
│   └── prediction.json       # Final output file
```

---

## 📌 Notes

- The system uses a hybrid RAG retriever to select the top-2 most relevant training examples based on both image and text similarity.
- Image and text embeddings are precomputed using pretrained models.
- The final LLaMA 4 prompt includes the selected examples and test case, along with structured instructions.

---

## 📄 License & Acknowledgments

- All models are used under their respective licenses via Hugging Face.
- Task and dataset by MEDIQA-WV organizers: [https://sites.google.com/view/mediqa-2025](https://sites.google.com/view/mediqa-2025)
