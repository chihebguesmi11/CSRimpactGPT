import numpy as np
import faiss
import PyPDF2
import os
from typing import Dict, List
import openai
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import pickle

class PDFEmbeddingManager:
    def __init__(self, api_key: str, pdf_directory: str, embedding_file: str = "embeddings.pkl", 
                existing_actions_embedding_file: str = "existing_actions_embeddings.pkl"):
    
        self.api_key = api_key
        openai.api_key = api_key
        self.pdf_directory = pdf_directory
        self.embedding_file = embedding_file
        self.existing_actions_embedding_file = existing_actions_embedding_file
        self.documents = []
        self.embeddings = None
        self.index = None
        self.existing_actions_embeddings = []
        self.chunk_size = 1000

        self.load_pdfs()
        self.load_or_initialize_faiss()

    def load_pdfs(self):
        """Load and process all PDFs from the specified directory"""
        try:
            for filename in os.listdir(self.pdf_directory):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(self.pdf_directory, filename)
                    self.process_pdf(file_path)
            print(f"Successfully loaded {len(self.documents)} documents from PDFs")
        except Exception as e:
            print(f"Error loading PDFs: {str(e)}")

    def process_pdf(self, pdf_path: str):
       
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()

                chunks = textwrap.wrap(text, self.chunk_size, break_long_words=False)
                
                for i, chunk in enumerate(chunks):
                    self.documents.append({
                        'content': chunk,
                        'source': os.path.basename(pdf_path),
                        'chunk_id': i
                    })
                
                print(f"Processed {pdf_path}: {len(chunks)} chunks created")
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")

    def _get_embedding(self, text: str) -> List[float]:
   
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return []

    def initialize_faiss(self):
        """Initialize FAISS index with document embeddings"""
        try:
            if not self.documents:
                print("No documents to process")
                return

            embeddings = []
            for doc in self.documents:
                embedding = self._get_embedding(doc['content'])
                embeddings.append(embedding)

            self.embeddings = np.array(embeddings, dtype='float32')
            
            dimension = len(embeddings[0])
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings)
            print("FAISS index initialized successfully")
        
        except Exception as e:
            print(f"Error initializing FAISS: {str(e)}")

    def load_or_initialize_faiss(self):
        """Load existing embeddings or initialize new ones"""
        try:
            if os.path.exists(self.embedding_file):
                with open(self.embedding_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.embeddings = np.array(data['embeddings'], dtype='float32')
                    
                    # Initialize FAISS index with loaded embeddings
                    dimension = self.embeddings.shape[1]
                    self.index = faiss.IndexFlatL2(dimension)
                    self.index.add(self.embeddings)
                    print("Embeddings loaded from file")
            else:
                self.initialize_faiss()
                self.save_embeddings()

            # Load existing actions embeddings
            if os.path.exists(self.existing_actions_embedding_file):
                with open(self.existing_actions_embedding_file, 'rb') as f:
                    self.existing_actions_embeddings = pickle.load(f)
                    print("Existing actions embeddings loaded")
        
        except Exception as e:
            print(f"Error in load_or_initialize_faiss: {str(e)}")

    def save_embeddings(self):
        """Save current embeddings to file"""
        try:
            with open(self.embedding_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents, 
                    'embeddings': self.embeddings.tolist()
                }, f)
            print("Embeddings saved to file")
        except Exception as e:
            print(f"Error saving embeddings: {str(e)}")

    def save_existing_actions_embeddings(self):
        try:
            with open(self.existing_actions_embedding_file, 'wb') as f:
                pickle.dump(self.existing_actions_embeddings, f)
            print("Existing actions embeddings saved")
        except Exception as e:
            print(f"Error saving existing actions embeddings: {str(e)}")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        
        try:
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                return []

            query_vector = np.array([query_embedding], dtype='float32')
            
            distances, indices = self.index.search(query_vector, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        'content': doc['content'],
                        'source': doc['source'],
                        'chunk_id': doc['chunk_id'],
                        'distance': float(distances[0][i])
                    })
            
            return results
        
        except Exception as e:
            print(f"Error in search: {str(e)}")
            return []

    def process_existing_actions_pdf(self, pdf_path: str):
       
        try:
            print(f"Processing existing actions PDF: {pdf_path}")
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()

                chunks = textwrap.wrap(text, self.chunk_size, break_long_words=False)

                self.existing_actions_embeddings = []
                for chunk in chunks:
                    embedding = self._get_embedding(chunk)
                    self.existing_actions_embeddings.append({
                        "content": chunk,
                        "embedding": embedding
                    })
                
                print(f"Generated {len(self.existing_actions_embeddings)} embeddings from existing actions")
                self.save_existing_actions_embeddings()
        
        except Exception as e:
            print(f"Error processing existing actions PDF: {str(e)}")

    def filter_recommendations_by_existing_actions(self, recommendations: List[Dict], threshold: float = 0.8) -> List[Dict]:
       
        try:
            filtered_recommendations = []
            for rec in recommendations:
                rec_embedding = self._get_embedding(rec['content'])
                is_similar = False

                for action in self.existing_actions_embeddings:
                    similarity = cosine_similarity([rec_embedding], [action["embedding"]])[0][0]
                    if similarity > threshold:
                        is_similar = True
                        break

                if not is_similar:
                    filtered_recommendations.append(rec)
            
            return filtered_recommendations
        
        except Exception as e:
            print(f"Error in filtering recommendations: {str(e)}")
            return recommendations

    def generate_response(self, prompt: str) -> str:
        
        try:
            similar_docs = self.search(prompt)
            filtered_docs = self.filter_recommendations_by_existing_actions(similar_docs)
            
            context = "\n".join([doc['content'] for doc in filtered_docs]) if filtered_docs else ""
            
            messages = [
                {"role": "system", "content": f"Context from knowledge base:\n{context}\n\nUse this context to help generate a response to:\n{prompt}"},
                {"role": "user", "content": prompt}
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )
            
            return response['choices'][0]['message']['content']
        
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating the response. Please try again."
