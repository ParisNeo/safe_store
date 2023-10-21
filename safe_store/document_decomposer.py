import re

   
class DocumentDecomposer:
    @staticmethod
    def clean_text(text):
        # Remove extra returns and leading/trailing spaces
        text = text.replace('\r', '').strip()
        return text

    @staticmethod
    def split_into_paragraphs(text):
        # Split the text into paragraphs using two or more consecutive newlines
        paragraphs = [p+"\n" for p in re.split(r'\n{2,}', text)]
        return paragraphs

    @staticmethod
    def tokenize_sentences(paragraph):
        # Custom sentence tokenizer using simple regex-based approach
        sentences = [s+"." for s in paragraph.split(".")]
        sentences = [sentence for sentence in sentences]
        return sentences

    @staticmethod
    def decompose_document(text, max_chunk_size, overlap_size, tokenize=None, detokenize=None):
        cleaned_text = DocumentDecomposer.clean_text(text)
        paragraphs = DocumentDecomposer.split_into_paragraphs(cleaned_text)

        # List to store the final clean chunks
        clean_chunks = []

        current_chunk = []  # To store the current chunk being built
        l = 0
        for paragraph in paragraphs:
            # Tokenize the paragraph into sentences
            sentences = DocumentDecomposer.tokenize_sentences(paragraph)

            for sentence in sentences:
                if tokenize is not None:
                    tokens = tokenize(sentence)
                else:
                    # Split the text using regular expression to preserve line breaks and multiple spaces
                    tokens = re.split(r'(\s+|\n+)', sentence)
                    # tokens = sentence.split()  # Use words as units

                nb_tokens = len(tokens)
                if nb_tokens > max_chunk_size:
                    while nb_tokens > max_chunk_size:
                        current_chunk += tokens[:max_chunk_size - l - 1]
                        clean_chunks.append(current_chunk)
                        tokens = tokens[max_chunk_size - l - 1 - overlap_size:]
                        nb_tokens -= max_chunk_size - l - 1 - overlap_size
                        l = 0
                        current_chunk = current_chunk[-overlap_size:]
                else:
                    if l + nb_tokens + 1 > max_chunk_size:
                        clean_chunks.append(current_chunk)
                        if overlap_size == 0:
                            current_chunk = []
                        else:
                            current_chunk = current_chunk[-overlap_size:]
                        l = 0

                    # Add the current sentence to the chunk
                    current_chunk += tokens
                    l += nb_tokens

        # Add the remaining chunk from the paragraph to the clean_chunks
        if current_chunk:
            clean_chunks.append(current_chunk)
            current_chunk = ""

        return clean_chunks