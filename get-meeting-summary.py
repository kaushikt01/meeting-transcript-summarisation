import pdfplumber
import spacy


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text

# Replace with the path to your PDF file
pdf_path = "Transcript_ Meeting on Meeting Scheduler Integration Feature.pdf"
transcript = extract_text_from_pdf(pdf_path)

# Print the first 500 characters to verify the extracted text
# print(transcript[:500])


#Step - 1
#---------------------------------------------------------------------------------




# Load spaCy's English model

nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    # Process the text using spaCy
    doc = nlp(text)

    # Tokenize and remove stopwords, punctuation, etc.
    cleaned_text = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])

    return cleaned_text


# Preprocess the transcript
cleaned_transcript = preprocess_text(transcript)

# Print the cleaned transcript
# print(cleaned_transcript[:500])


#Step - 2
#----------------------------------------------------------------



from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer  # You can also use other summarizers like LsaSummarizer, LuhnSummarizer, etc.
from sumy.nlp.tokenizers import Tokenizer

def extractive_summary(text):
    # Parse the text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    # Use LSA Summarizer (Latent Semantic Analysis)
    summarizer = LsaSummarizer()

    # Set the number of sentences you want in the summary (adjust as needed)
    summary = summarizer(parser.document, 3)  # You can change 3 to any number that suits your summary length.

    # Return the summary as a string
    return ' '.join(str(sentence) for sentence in summary)


# Example usage with your cleaned transcript
ext_summary = extractive_summary(cleaned_transcript)

# Print the summary
# print(summary)

#Step - 3
#-----------------------------------------------------------------


import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1
print(device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Generate summary
abs_summary = summarizer(cleaned_transcript, max_length=500, min_length=40, do_sample=False)
print("Abstractive Summary:\n", abs_summary)

#Step - 4
# -------------------------------------------------------------------

import spacy
import pytextrank

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Add the pytextrank component to the spaCy pipeline
nlp.add_pipe("textrank")


def extract_keywords_text_rank(text):
    # Process the text using spaCy
    doc = nlp(text)

    # Extract keywords using TextRank
    keywords = []
    for phrase in doc._.phrases:
        keywords.append(phrase.text)

    return keywords


# Get keywords from TextRank
keywords = extract_keywords_text_rank(abs_summary[0]["summary_text"])

# Print the extracted keywords
print("Extracted Keywords using TextRank:")
for keyword in keywords:
    print(keyword)




#Step - 5
#-------------------------------------------------------------------------




def generate_final_report(extractive_summary, abstractive_summary, keywords):
    # Format the extractive summary to include key points from the meeting
    formatted_extractive = "\n".join([f"- {point}" for point in extractive_summary.split(". ") if point])

    # Combine the extractive and abstractive summaries into the final report
    combined_summary = f"### Key Points from the Meeting:\n{formatted_extractive}\n\n" \
                       f"### Detailed Insight:\n{abstractive_summary}\n\n" \
                       f"### Keywords:\n" + ", ".join(keywords) + "\n\n" \
                       f"### Conclusion:\n" \
                       f"Based on the discussion, the key topics are highlighted above. Further actions can be taken accordingly."

    return combined_summary

# Example usage with your summaries and keywords:
final_report = generate_final_report(ext_summary, abs_summary, keywords)

# Print the final report
print("Final Report:\n", final_report)
