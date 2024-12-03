
This project extracts, processes, and summarizes text from a PDF document containing a meeting transcript. It generates both extractive and abstractive summaries and provides keywords extracted from the text. The final report is formatted to highlight key points, detailed insights, and the most relevant keywords.

## Requirements

- `pdfplumber`: To extract text from PDF files.
- `spacy`: For natural language processing, tokenization, and preprocessing.
- `sumy`: For extractive summarization using Latent Semantic Analysis (LSA).
- `transformers`: For abstractive summarization using a pre-trained BART model.
- `pytextrank`: For keyword extraction based on TextRank.

You can install the required dependencies using the following command:

```bash
pip install pdfplumber spacy sumy transformers pytextrank
```

Additionally, you'll need to download the spaCy model and tokenizer:

```bash
python -m spacy download en_core_web_sm
```

## Functionality

### 1. Extract Text from PDF

The function `extract_text_from_pdf` extracts the full text from a PDF file using `pdfplumber`.

```python
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text
```

### 2. Text Preprocessing

The `preprocess_text` function uses spaCy to clean the extracted text, removing stopwords and punctuation.

**Alternative Models:**

- **NLTK (Natural Language Toolkit)**: Another popular NLP library that can handle text preprocessing, including tokenization, stopword removal, and punctuation removal.
- **Transformers Tokenizer**: You can use tokenizers from pre-trained models, like the BERT tokenizer, which are highly optimized for handling specific pre-trained models and can handle preprocessing tasks like sentence segmentation and tokenization.

**Why spaCy is Better:**

- **Ease of Use**: spaCy provides a very straightforward API and is designed for fast, efficient text processing.
- **Integrated Pipeline**: spaCy's `en_core_web_sm` model already includes all the necessary preprocessing steps (tokenization, stopword removal, lemmatization), which makes it easy to integrate and highly efficient.
- **Performance**: spaCy is optimized for production-level NLP tasks, providing high-speed processing that scales well with large datasets.

```python
def preprocess_text(text):
    doc = nlp(text)
    cleaned_text = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
    return cleaned_text
```

### 3. Extractive Summarization

The `extractive_summary` function generates a summary of the cleaned text using Latent Semantic Analysis (LSA) summarizer from the `sumy` library.

**Alternative Models:**

- **LuhnSummarizer**: Another extractive summarizer available in the `sumy` library, which uses frequency-based techniques to select sentences for summarization.
- **TextRankSummarizer**: An extractive summarizer that is based on the TextRank algorithm, similar to how Google's PageRank works for ranking web pages.

**Why LSA Summarizer is Better:**

- **Latent Semantic Analysis (LSA)** is a robust technique that can capture deeper semantic meanings in the text, making it more suited for handling complex documents such as meeting transcripts.
- **Efficiency**: LSA is highly efficient at summarizing longer documents by focusing on the latent structures within the data, which often results in better summaries with fewer repetitive sentences.

```python
def extractive_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Adjust the number of sentences as needed
    return ' '.join(str(sentence) for sentence in summary)
```

### 4. Abstractive Summarization

An abstractive summary is generated using the pre-trained BART model from the `transformers` library.

**Alternative Models:**

- **GPT-3 or GPT-4 (OpenAI)**: These models are highly advanced and provide more natural language generation capabilities, but they require API access, which can be expensive.
- **T5 (Text-to-Text Transfer Transformer)**: A transformer model by Google that works well for text-to-text tasks like summarization.
- **BERTSUM**: A variation of BERT specifically fine-tuned for summarization tasks.

**Why BART Model is Better:**

- **Pretrained for Summarization**: The BART model is specifically pre-trained for sequence-to-sequence tasks like summarization, making it a natural fit for generating high-quality abstractive summaries.
- **Balance between Abstraction and Fidelity**: BART produces summaries that strike a good balance between faithfully representing the original content and generating novel, meaningful text.
- **State-of-the-Art**: BART is one of the best models for summarization, often achieving strong performance on multiple benchmarks.

```python
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
abs_summary = summarizer(cleaned_transcript, max_length=500, min_length=40, do_sample=False)
```

### 5. Keyword Extraction using TextRank

Keywords are extracted from the abstractive summary using the `pytextrank` library.

**Alternative Models:**

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: A simple, effective method for keyword extraction based on word frequency.
- **RAKE (Rapid Automatic Keyword Extraction)**: Another keyword extraction technique that is more rule-based compared to TextRank.

**Why TextRank is Better:**

- **Graph-Based**: TextRank, like Google's PageRank algorithm, is graph-based, considering the relationships between words and phrases, leading to more coherent and meaningful keyword extraction.
- **Efficiency**: TextRank automatically extracts relevant keywords and phrases from the entire document, whereas TF-IDF can sometimes miss key concepts that appear less frequently but are still important.

```python
def extract_keywords_text_rank(text):
    doc = nlp(text)
    keywords = [phrase.text for phrase in doc._.phrases]
    return keywords
```

### 6. Generate Final Report

The `generate_final_report` function formats the extractive and abstractive summaries, along with the keywords, into a structured report.

```python
def generate_final_report(extractive_summary, abstractive_summary, keywords):
    formatted_extractive = "\n".join([f"- {point}" for point in extractive_summary.split(". ") if point])
    combined_summary = f"### Key Points from the Meeting:\n{formatted_extractive}\n\n" \
                       f"### Detailed Insight:\n{abstractive_summary}\n\n" \
                       f"### Keywords:\n" + ", ".join(keywords) + "\n\n" \
                       f"### Conclusion:\n" \
                       f"Based on the discussion, the key topics are highlighted above. Further actions can be taken accordingly."
    return combined_summary
```

## Example Usage

To use the program, follow these steps:

1. Replace the `pdf_path` with the path to your own PDF document.
2. Run the script to extract text, preprocess it, generate summaries, and extract keywords.
3. The final report will be printed with key points, detailed insights, and keywords.

### Example Output:

```bash

Final Report:
### Key Points from the Meeting:
- Key point 1
- Key point 2

### Detailed Insight:
This is the detailed summary generated by the BART model.

### Keywords:
Meeting Scheduler, Integration, Feature

### Conclusion:
Based on the discussion, the key topics are highlighted above. Further actions can be taken accordingly.
```

## Conclusion

This script helps in efficiently processing and summarizing meeting transcripts from PDF files. It combines extractive and abstractive summarization techniques with keyword extraction to create a comprehensive report.