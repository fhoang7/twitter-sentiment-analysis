import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    def __init__(self, text: list):
        """
        Expects text to be a List[str] for text preprocessing
        """
        self.stop_words = set(stopwords.words('english'))
        self.text = text

    def remove_special_characters(self):
        # Remove special characters using regex
        processed_text = re.sub(r'[^a-zA-Z0-9\s]', '', self.text)
        processed_text = re.sub(r'https', '', processed_text)
        self.text = processed_text
        print("Removed special characters...")

    def remove_stop_words(self):
        # Tokenize the text
        words = word_tokenize(self.text)
        # Remove stop words and lowercase
        filtered_text = [word.lower() for word in words if word.lower() not in self.stop_words]
        # Join the words back into a string
        processed_text = ' '.join(filtered_text)
        self.text = processed_text
        print("Removed stop words...")
        
    def preprocess(self):
        self.remove_stop_words()
        self.remove_special_characters()
        return self.text
