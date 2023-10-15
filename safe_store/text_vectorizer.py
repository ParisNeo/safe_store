from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from safe_store.BM25Vectorizer import BM25Vectorizer, split_string  # Import BM25Vectorizer
import numpy as np
from pathlib import Path
import json
from ascii_colors import ASCIIColors, trace_exception
from safe_store.document_decomposer import DocumentDecomposer
from safe_store.tfidf_loader import TFIDFLoader
from safe_store.utils import NumpyEncoderDecoder

from enum import Enum

class VectorizationMethod(Enum):
    MODEL_EMBEDDING = "model_embedding"
    TFIDF_VECTORIZER = "tfidf_vectorizer"
    BM25_VECTORIZER = "bm25_vectorizer"

class VisualizationMethod(Enum):
    PCA = "PCA"
    TSNE = "TSNE"

class TextVectorizer:
    def __init__(
                    self, 
                    vectorization_method:VectorizationMethod|str, # supported "model_embedding" or "tfidf_vectorizer"
                    model=None, #needed in case of using model_embedding
                    database_path=None,
                    save_db=False,
                    data_visualization_method:VisualizationMethod|str=VisualizationMethod.PCA,
                    database_dict=None
                    ):
        if isinstance(vectorization_method, str):
            try:
                vectorization_method = VectorizationMethod(vectorization_method)
            except ValueError:
                raise ValueError("Invalid vectorization_method string. Please use valid enum values or strings.")
        elif not isinstance(vectorization_method, VectorizationMethod):
            raise ValueError("Invalid vectorization_method. Please use VectorizationMethod enum values or strings.")
        
        if isinstance(data_visualization_method, str):
            try:
                data_visualization_method = VisualizationMethod(vectorization_method)
            except ValueError:
                raise ValueError("Invalid vectorization_method string. Please use valid enum values or strings.")
        elif not isinstance(data_visualization_method, VisualizationMethod):
            raise ValueError("Invalid vectorization_method. Please use VisualizationMethod enum values or strings.")
        
        
        
        self.vectorization_method = vectorization_method
        self.save_db = save_db
        self.model = model
        self.database_file = database_path
        
        self.data_visualization_method = data_visualization_method
        
        if database_dict is not None:
            self.chunks =  database_dict["chunks"]
            self.vectorizer = database_dict["vectorizer"]
            self.infos =   database_dict["infos"]
            self.ready = True
        else:
            self.chunks = {}
            self.ready = False
            self.vectorizer = None
        
            if vectorization_method==VectorizationMethod.MODEL_EMBEDDING:
                try:
                    if not self.model or self.model.embed("hi")==None: # test
                        self.vectorization_method=VectorizationMethod.TFIDF_VECTORIZER
                        self.infos={
                            "vectorization_method":VectorizationMethod.TFIDF_VECTORIZER.value
                        }                        
                    else:
                        self.infos={
                            "vectorization_method":VectorizationMethod.MODEL_EMBEDDING.value
                        }
                except Exception as ex:
                    ASCIIColors.error("Couldn't embed the text, so trying to use tfidf instead.")
                    trace_exception(ex)
                    self.infos={
                        "vectorization_method":VectorizationMethod.TFIDF_VECTORIZER.value
                    }
            elif vectorization_method == VectorizationMethod.BM25_VECTORIZER:
                self.infos = {
                    "vectorization_method": VectorizationMethod.BM25_VECTORIZER.value
                }

            else:
                self.infos={
                    "vectorization_method":VectorizationMethod.TFIDF_VECTORIZER.value
                }

        # Load previous state from the JSON file
        if self.save_db:
            if Path(self.database_file).exists():
                ASCIIColors.success(f"Database file found : {self.database_file}")
                try:
                    self.load_from_json()
                except Exception as ex:
                    ASCIIColors.error("Couldn't load vectorized db.\nMoving to safe mode")
                    if self.vectorization_method==VectorizationMethod.TFIDF_VECTORIZER:
                        self.vectorizer = TfidfVectorizer()
                    elif self.vectorization_method == VectorizationMethod.BM25_VECTORIZER:
                        self.vectorizer = BM25Vectorizer()
                self.ready = True
            else:
                ASCIIColors.info(f"No database file found : {self.database_file}")
                if self.vectorization_method==VectorizationMethod.TFIDF_VECTORIZER:
                    self.vectorizer = TfidfVectorizer()
                elif self.vectorization_method == VectorizationMethod.BM25_VECTORIZER:
                    self.vectorizer = BM25Vectorizer()
        else:
            if self.vectorization_method==VectorizationMethod.TFIDF_VECTORIZER:
                self.vectorizer = TfidfVectorizer()
            elif self.vectorization_method == VectorizationMethod.BM25_VECTORIZER:
                self.vectorizer = BM25Vectorizer()

                    
    def show_document(self, query_text=None, save_fig_path=None, show_interactive_form=False, add_hover_detection=False, add_click_detection=False):
        import textwrap
        import seaborn as sns
        import matplotlib.pyplot as plt
        import mplcursors
        from tkinter import Tk, Text, Scrollbar, Frame, Label, TOP, BOTH, RIGHT, LEFT, Y, N, END
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        if save_fig_path is None and show_interactive_form==False:
            raise Exception("[show_document] parameters error. Either select a save_fig_path to save the graph or set show_interactive_form to True")

        if self.vectorization_method==VectorizationMethod.BM25_VECTORIZER:
            ASCIIColors.red("BM25 does not make a representatrion of the data. So we don't show a distribution.")
            return

        if self.data_visualization_method == VisualizationMethod.PCA:
            use_pca = True
        else:
            use_pca = False

        if use_pca:
            print("Showing pca representation:")
        else:
            print("Showing t-sne representation:")

        embeddings = {key: chunk["embeddings"] for key, chunk in self.chunks.items()}
        emb = list(embeddings.values())
        ref = list(embeddings.keys())

        if len(emb) >= 2:
            # Normalize embeddings
            emb = np.vstack(emb)
            norms = np.linalg.norm(emb, axis=1)
            normalized_embeddings = emb / norms[:, np.newaxis]

            # Embed the query text
            if query_text is not None:
                query_embedding = self.embed_query(query_text)
                query_normalized_embedding = query_embedding / np.linalg.norm(query_embedding)

                # Combine the query embeddings with the document embeddings
                combined_embeddings = np.vstack((normalized_embeddings, query_normalized_embedding))
                ref.append("Query")  # Use a unique name for the query
            else:
                # Combine the query embeddings with the document embeddings
                combined_embeddings = normalized_embeddings

            if use_pca:
                # Use PCA for dimensionality reduction
                pca = PCA(n_components=2)
                try:
                    embeddings_2d = pca.fit_transform(combined_embeddings)
                except Exception as ex:
                    embeddings_2d = []
            else:
                # Use t-SNE for dimensionality reduction
                perplexity = min(30, combined_embeddings.shape[0] - 1)
                tsne = TSNE(n_components=2, perplexity=perplexity)
                embeddings_2d = tsne.fit_transform(combined_embeddings)



            trans={tuple(e.tolist()):r for e,r in zip(embeddings_2d,ref)}

            # Create a dictionary to map document paths to unique colors
            document_paths = ["_".join(path.split("_")[:-1]) if "/" in path or "\\" in path else path for path in ref]
            # Define the number of colors you need
            num_colors = len(document_paths)

            # Use Seaborn's "Set1" palette to get a list of distinct colors
            colors = sns.color_palette("Set1", n_colors=num_colors)

            # Now you can use the 'colors' list for your document paths
            document_path_colors = colors
            unique_document_paths = list(set(document_paths))
            legend_labels= []
            embeddings_by_document={doc:[] for doc in unique_document_paths}
            for i,document_name in enumerate(document_paths):
                legend_labels.append(Path(document_name).stem if "/" in document_name or "\\" in document_name else document_name)
                embeddings_by_document[document_name].append(embeddings_2d[i,:][None,:])

            labels=[]
            for i, (document_name, emb) in enumerate(embeddings_by_document.items()):
                label = Path(document_name).stem if "/" in document_name or "\\" in document_name else document_name
                labels.append(label)
                embeddings_2d = np.vstack(emb)
                plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1],  color=document_path_colors[i], label=label, marker='o' if document_name!="Query" else "x")
                # Add labels to the scatter plot
                for j, (x, y) in enumerate(embeddings_2d[:]):
                    if label!="Query":
                        plt.text(x, y, f"{i}_{j}", fontsize=8)
                    else:
                        plt.text(x, y, f"query", fontsize=8)
            # Create the legend
            plt.legend(title='Document Name', loc='upper right')



            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')

            if use_pca:
                plt.title('Embeddings Scatter Plot based on PCA')
            else:
                plt.title('Embeddings Scatter Plot based on t-SNE')

            # Enable mplcursors to show tooltips on hover
            cursor = mplcursors.cursor(hover=True)

            # Define the hover event handler
            if add_hover_detection:
                @cursor.connect("add")
                def on_hover(sel):
                    index = sel.target.index
                    if trans[tuple(sel.target.real.tolist())]!="Query":
                        text = self.chunks[trans[tuple(sel.target.real.tolist())]]["chunk_text"]
                    else:
                        text = query_text
                    wrapped_text = textwrap.fill(text, width=50)  # Wrap the text into multiple lines
                    sel.annotation.set_text(f"Index: {index}\nText:\n{wrapped_text}")

            if add_click_detection:
                # Define the click event handler using matplotlib event handling mechanism
                def on_click(event):
                    if event.xdata is not None and event.ydata is not None:
                        x, y = event.xdata, event.ydata
                        distances = ((embeddings_2d[:, 0] - x) ** 2 + (embeddings_2d[:, 1] - y) ** 2)
                        index = distances.argmin()
                        if distances[index]<0.1:

                            # Check if the click is outside any data point
                            if index < len(self.chunks):
                                text = self.chunks[trans[tuple(embeddings_2d[index,:].tolist())]]["chunk_text"]
                            elif index == len(self.chunks):
                                text = query_text
                            else:
                                # Clicked outside any data point, close the Tkinter window if it's open
                                plt.close('all')
                                return

                            # Open a new Tkinter window with the content of the text
                            root = Tk()
                            root.title(f"Text for Index {index}")
                            frame = Frame(root)
                            frame.pack(fill=BOTH, expand=True)

                            label = Label(frame, text="Text:")
                            label.pack(side=TOP, padx=5, pady=5)

                            text_box = Text(frame)
                            text_box.pack(side=TOP, padx=5, pady=5, fill=BOTH, expand=True)
                            text_box.insert(END, text)

                            scrollbar = Scrollbar(frame)
                            scrollbar.pack(side=RIGHT, fill=Y)
                            scrollbar.config(command=text_box.yview)
                            text_box.config(yscrollcommand=scrollbar.set)

                            text_box.config(state="disabled")

                            root.mainloop()


                # Connect the click event handler to the figure
                plt.gcf().canvas.mpl_connect("button_press_event", on_click)

            if save_fig_path:
                try:
                    plt.savefig(save_fig_path)
                except Exception as ex:
                    trace_exception(ex)

            if show_interactive_form:
                plt.show()

    
    def file_exists(self, document_name:str)->bool:
        # Loop through the list of dictionaries
        for dictionary in self.chunks:
            if 'document_name' in dictionary and dictionary['document_name'] == document_name:
                # If the document_name is found in the current dictionary, set the flag to True and break the loop
                document_name_found = True
                return True
        return False
    
    def remove_document(self, document_name:str):
        for dictionary in self.chunks:
            if 'document_name' in dictionary and dictionary['document_name'] == document_name:
                # If the document_name is found in the current dictionary, set the flag to True and break the loop
                self.chunks.remove(dictionary)
                return True
        return False

    def add_document(self, document_name:Path, text:str, chunk_size: int, overlap_size:int, force_vectorize=False,add_as_a_bloc=False):
        if self.file_exists(document_name) and not force_vectorize:
            print(f"Document {document_name} already exists. Skipping vectorization.")
            return
        if add_as_a_bloc:
            chunks_text = [self.model.tokenize(text)]
            for i, chunk in enumerate(chunks_text):
                chunk_id = f"{document_name}_chunk_{i + 1}"
                chunk_dict = {
                    "document_name": str(document_name),
                    "chunk_index": i+1,
                    "chunk_text":self.model.detokenize(chunk),
                    "chunk_tokens": chunk,
                    "embeddings":[]
                }
                self.chunks[chunk_id] = chunk_dict
        else:
            if self.model:
                chunks_text = DocumentDecomposer.decompose_document(text, chunk_size, overlap_size, self.model.tokenize, self.model.detokenize)
            else:
                chunks_text = DocumentDecomposer.decompose_document(text, chunk_size, overlap_size)

            for i, chunk in enumerate(chunks_text):
                chunk_id = f"{document_name}_chunk_{i + 1}"
                chunk_dict = {
                    "document_name": str(document_name),
                    "chunk_index": i+1,
                    "chunk_text":self.model.detokenize(chunk) if (self.model and self.model.detokenize) else ' '.join(chunk),
                    "chunk_tokens": chunk,
                    "embeddings":[]
                }
                self.chunks[chunk_id] = chunk_dict
        
    def index(self):
        if self.vectorization_method==VectorizationMethod.TFIDF_VECTORIZER:
            #if self.debug:
            #    ASCIIColors.yellow(','.join([len(chunk) for chunk in chunks]))
            data=[]
            for k,chunk in self.chunks.items():
                try:
                    data.append(chunk["chunk_text"]) 
                except Exception as ex:
                    print("oups")
            self.vectorizer.fit(data)
        elif self.vectorization_method==VectorizationMethod.BM25_VECTORIZER:
            #if self.debug:
            #    ASCIIColors.yellow(','.join([len(chunk) for chunk in chunks]))
            data=[]
            for k,chunk in self.chunks.items():
                try:
                    data.append(chunk["chunk_text"]) 
                except Exception as ex:
                    print("oups")
            self.vectorizer.fit(data)

            
        # Generate embeddings for each chunk
        for chunk_id, chunk in self.chunks.items():
            # Store chunk ID, embeddings, and original text
            try:
                if self.vectorization_method==VectorizationMethod.TFIDF_VECTORIZER:
                    chunk["embeddings"] = self.vectorizer.transform([chunk["chunk_text"]]).toarray()
                elif self.vectorization_method==VectorizationMethod.BM25_VECTORIZER:
                    chunk["BM25_data"] = (self.vectorizer.doc_term_freqs, self.vectorizer.doc_lengths) 
                else:
                    chunk["embeddings"] = self.model.embed(chunk["chunk_text"])
            except Exception as ex:
                print("oups")

        if self.save_db:
            self.save_to_json()
            
        self.ready = True


    def embed_query(self, query_text):
        # Generate query embeddings
        if self.vectorization_method == VectorizationMethod.TFIDF_VECTORIZER:
            query_embedding = self.vectorizer.transform([query_text]).toarray()
        elif self.vectorization_method == VectorizationMethod.BM25_VECTORIZER:
            raise Exception("BM25 don't use embedding")
        else:
            query_embedding = self.model.embed(query_text)
            if query_embedding is None:
                ASCIIColors.warning("The model doesn't implement embeddings extraction")
                self.vectorization_method = VectorizationMethod.TFIDF_VECTORIZER
                query_embedding = self.vectorizer.transform([query_text]).toarray()

        return query_embedding

    def __len__(self):
        return len(list(self.chunks.keys()))

    def recover_chunk_by_index(self, index):
        chunk_id = [ch for ch in self.chunks.keys()][index]
        return self.chunks[chunk_id]["chunk_text"]

    def recover_chunk_by_document_name(self, document_name):
        chunks = [ch for ch in self.chunks.values() if ch["document_name"]==document_name]
        return chunks

    def recover_text(self, query, top_k=3):
        if self.vectorization_method==VectorizationMethod.TFIDF_VECTORIZER or self.vectorization_method==VectorizationMethod.MODEL_EMBEDDING:
            similarities = {}
            query_embedding = self.embed_query(query)
            for chunk_id, chunk in self.chunks.items():
                similarity = cosine_similarity(query_embedding, chunk["embeddings"])
                similarities[chunk_id] = similarity

            # Sort the similarities and retrieve the top-k most similar embeddings
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

            # Retrieve the original text associated with the most similar embeddings
            texts = [self.chunks[chunk_id]["chunk_text"] for chunk_id, _ in sorted_similarities]
        elif self.vectorization_method==VectorizationMethod.BM25_VECTORIZER:
            # Use the BM25Vectorizer to compute BM25 scores for the query
            bm25_scores = self.vectorizer.transform(query)

            # Find the top-k documents with the highest BM25 scores
            top_k_indices = np.argsort(bm25_scores)[::-1][:top_k]

            # Retrieve the original text associated with the top-k documents
            chunk_keys = [key for key,_ in self.chunks.items()]
            texts = [self.chunks[chunk_keys[chunk_id]]["chunk_text"] for chunk_id in top_k_indices]   
            sorted_similarities = np.sort(bm25_scores)
        return texts, sorted_similarities

    def toJson(self):
        state = {
            "chunks": self.chunks,
            "infos": self.infos,
            "vectorizer": TFIDFLoader.create_dict_from_vectorizer(self.vectorizer) if self.vectorization_method==VectorizationMethod.TFIDF_VECTORIZER else None
        }
        return state
    
    def setVectorizer(self, vectorizer_dict:dict):
        self.vectorizer=TFIDFLoader.create_vectorizer_from_dict(vectorizer_dict)

    def save_to_json(self):
        state = {
            "chunks": self.chunks,
            "infos": self.infos,
            "vectorizer": TFIDFLoader.create_dict_from_vectorizer(self.vectorizer) if self.vectorization_method==VectorizationMethod.TFIDF_VECTORIZER else None
        }
        with open(self.database_file, "w") as f:
            json.dump(state, f, cls=NumpyEncoderDecoder, indent=4)

    def load_from_json(self):
        ASCIIColors.info("Loading vectorized documents")
        with open(self.database_file, "r") as f:
            database = json.load(f, object_hook=NumpyEncoderDecoder.as_numpy_array)
            self.chunks = database["chunks"]
            self.infos= database["infos"]
            self.ready = True
        if self.vectorization_method==VectorizationMethod.TFIDF_VECTORIZER:
            self.vectorizer = TFIDFLoader.create_vectorizer_from_dict(database["vectorizer"])
        if self.vectorization_method==VectorizationMethod.BM25_VECTORIZER:
            self.vectorizer = BM25Vectorizer()
            data=[]
            for k,chunk in self.chunks.items():
                try:
                    data.append(chunk["chunk_text"]) 
                except Exception as ex:
                    print("oups")
            self.vectorizer.fit(data)

                
                    
                    
    def clear_database(self):
        self.ready = False
        self.vectorizer=None
        self.chunks = {}
        self.infos={}
        if self.save_db:
            self.save_to_json()
            