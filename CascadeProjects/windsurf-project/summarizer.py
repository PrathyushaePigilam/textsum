import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation
from typing import Optional

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class TextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(punctuation)
        self.max_sentences = 3  # Fixed number of sentences in summary
        self.min_summary_length = 50  # Minimum length in characters
        self.max_summary_length = 200  # Maximum length in characters

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess the input text.
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters
        text = ''.join([char for char in text if char not in self.punctuation])
        return text.lower()

    def calculate_sentence_scores(self, text: str) -> dict:
        """
        Calculate scores for each sentence based on word frequency.
        """
        # Tokenize sentences and words
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Remove stopwords and get word frequencies
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        word_frequencies = FreqDist(filtered_words)
        
        # Calculate sentence scores
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]
        
        return sentence_scores

    def summarize(self, text: str, num_sentences: Optional[int] = None) -> str:
        """
        Generate a summary using extractive summarization.
        
        Args:
            text (str): Input text to summarize
            num_sentences (int, optional): Number of sentences in the summary
        
        Returns:
            str: Summarized text
        """
        if not text.strip():
            return ""
            
        if num_sentences is not None:
            self.max_sentences = num_sentences
            
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Calculate sentence scores
        sentence_scores = self.calculate_sentence_scores(processed_text)
        
        # Get top sentences
        top_sentences = sorted(
            sentence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_sentences]
        
        # Sort sentences by their original order
        top_sentences.sort(key=lambda x: text.index(x[0]))
        
        # Join sentences to form summary
        summary = ' '.join([sentence for sentence, _ in top_sentences])
        
        # Ensure summary is within length constraints
        if len(summary) < self.min_summary_length:
            # If too short, try to add more sentences
            additional_sentences = sorted(
                sentence_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[self.max_sentences:self.max_sentences+2]
            
            additional_summary = ' '.join([sentence for sentence, _ in additional_sentences])
            summary = summary + ' ' + additional_summary
            
            # If still too short, add more sentences
            if len(summary) < self.min_summary_length:
                additional_sentences = sorted(
                    sentence_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[self.max_sentences+2:self.max_sentences+4]
                additional_summary = ' '.join([sentence for sentence, _ in additional_sentences])
                summary = summary + ' ' + additional_summary
                
        # Trim if too long
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length].rsplit(' ', 1)[0] + '...'
        
        # Format summary consistently
        summary = summary.strip()
        summary = summary[0].upper() + summary[1:]  # Capitalize first letter
        summary = summary + '.' if not summary.endswith('.') else summary  # Add period if missing
        
        return summary
        

# Example usage
if __name__ == "__main__":
    summarizer = TextSummarizer()
    
    # Test with sample text
    sample_text = """
    The United States Declaration of Independence is the statement 
    adopted by the Second Continental Congress meeting at the Pennsylvania State 
    House (now known as Independence Hall) in Philadelphia on July 4, 1776, which 
    announced that the thirteen American colonies, then at war with Great Britain, 
    regarded themselves as thirteen independent sovereign states, no longer under 
    British rule. These states would found a new nation – the United States of America.
    """
    
    summary = summarizer.summarize(sample_text)
    print("\nOriginal Text:")
    print(sample_text)
    print("\nSummary:")
    print(summary)
