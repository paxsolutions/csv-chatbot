import os
import pickle
import tempfile
import pandas as pd
import hashlib
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import streamlit as st

os.makedirs('/app/logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/embedder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()

    def createEmbeddingsDir(self):
        """
        Creates a directory to store the embeddings vectors
        """
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def _get_file_hash(self, file_content):
        """
        Generate a hash for file content to enable caching
        """
        return hashlib.md5(file_content).hexdigest()

    def _sample_large_csv(self, df, max_rows=5000):
        """
        Sample large CSV files to reduce processing time while maintaining representativeness
        """
        if len(df) <= max_rows:
            return df

        # Take a stratified sample if possible, otherwise random sample
        try:
            # Try to sample from different parts of the dataset
            step = len(df) // max_rows
            sampled_df = df.iloc[::step][:max_rows]
            st.info(f"Large file detected ({len(df):,} rows). Using {len(sampled_df):,} representative rows for faster processing.")
            return sampled_df
        except:
            # Fallback to random sampling
            sampled_df = df.sample(n=max_rows, random_state=42)
            st.info(f"Large file detected ({len(df):,} rows). Using {len(sampled_df):,} random rows for faster processing.")
            return sampled_df

    def getDocEmbeds(self, file, filename):
        """
        Get document embeddings for a file - wrapper around storeDocEmbeds
        """
        return self.storeDocEmbeds(file, filename)

    def storeDocEmbeds(self, file, filename):
        """
        Stores document embeddings using optimized processing for large files
        """
        # Caching
        file_hash = self._get_file_hash(file)
        cache_filename = f"{filename}_{file_hash}"

        # Check if cached version exists
        if os.path.isfile(f"{self.PATH}/{cache_filename}.pkl"):
            st.info("Using cached embeddings for faster loading...")
            with open(f"{self.PATH}/{cache_filename}.pkl", "rb") as f:
                return pickle.load(f)

        # Write the uploaded file to a temp file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name

        try:
            # Read CSV with pandas and optimize for large files
            df = pd.read_csv(tmp_file_path, encoding="utf-8")
            df = self._sample_large_csv(df)

            if len(df) > 500:
                documents = self._process_large_csv_optimized(df)
            else:
                # Use standard CSVLoader for smaller files
                loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
                documents = loader.load_and_split()

            # Filter out empty documents before processing
            documents = [doc for doc in documents if doc.page_content.strip()]

            if not documents:
                st.error("No valid documents found in CSV file.")
                return None

            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,  # Consistent chunk size
                chunk_overlap=120,  # 10% overlap
                length_function=len,
            )

            # Split documents
            split_docs = []
            for doc in documents:
                if doc.page_content.strip():  # Only process non-empty documents
                    chunks = text_splitter.split_documents([doc])
                    # Ensure all chunks have content
                    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
                    split_docs.extend(valid_chunks)

            if not split_docs:
                st.error("No valid document chunks created.")
                return None

            # Create embeddings
            embeddings = OpenAIEmbeddings()

            # Process embeddings in single batch to avoid dimension issues
            all_vectors = self._create_embeddings_single_batch(split_docs, embeddings)

            # Save cache
            with open(f"{self.PATH}/{cache_filename}.pkl", "wb") as f:
                pickle.dump(all_vectors, f)

            # Also save original filename
            with open(f"{self.PATH}/{filename}.pkl", "wb") as f:
                pickle.dump(all_vectors, f)

            # Return vectors
            return all_vectors

        finally:
            os.remove(tmp_file_path)

    def _process_large_csv_optimized(self, df):
        """
        Optimized processing for large CSV files with better performance
        """
        documents = []
        chunk_size = 200
        columns = df.columns.tolist()

        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            chunk_text = f"Columns: {', '.join(columns)}\n\n"
            chunk_text += chunk_df.to_csv(index=False, header=False)
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": f"csv_chunk_{i//chunk_size + 1}",
                    "rows": f"{i+1}-{min(i+chunk_size, len(df))}",
                    "total_rows": len(df),
                    "columns": columns
                }
            )
            documents.append(doc)

        return documents

    def _create_embeddings_with_retry(self, batch, embeddings, max_retries=3):
        """
        Create embeddings for a batch with retry logic and dimension validation
        """
        for attempt in range(max_retries):
            try:
                valid_batch = [doc for doc in batch if doc.page_content.strip()]
                if not valid_batch:
                    return None

                return FAISS.from_documents(valid_batch, embeddings)
            except Exception as e:
                error_msg = str(e).lower()

                if "permission" in error_msg or "scope" in error_msg or "rate" in error_msg:
                    st.warning(f"API rate limit hit, retrying in {wait_time:.1f}s...")

                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        st.info(f"API rate limit hit, retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        st.warning(f"Max retries reached for batch after {max_retries} attempts")
                        return None
                else:
                    st.warning(f"Batch error: {str(e)}")
                    return None
        return None

    def _create_embeddings_single_batch(self, split_docs, embeddings):
        """
        Create embeddings in a single batch to avoid dimension mismatch issues
        """
        valid_docs = [doc for doc in split_docs if doc.page_content.strip()]

        if not valid_docs:
            st.error("No valid documents to process")
            return None

        max_docs = 100
        if len(valid_docs) > max_docs:
            st.info(f"Processing first {max_docs} documents for stability")

        try:
            # Debug
            st.info("Testing embedding dimensions...")

            # Test embedding
            if valid_docs:
                test_embedding = embeddings.embed_query(valid_docs[0].page_content[:100])
                st.info(f"OpenAI embedding dimension: {len(test_embedding)}")

                # Validate dimension
                expected_dim = 1536
                if len(test_embedding) != expected_dim:
                    st.warning(f"Unexpected embedding dimension: {len(test_embedding)} (expected {expected_dim})")

            # Validate all documents
            import numpy as np

            texts = [doc.page_content for doc in valid_docs]
            embeddings_list = embeddings.embed_documents(texts)

            # Validate dimensions
            dimensions = [len(emb) for emb in embeddings_list]
            unique_dims = set(dimensions)

            if len(unique_dims) > 1:
                st.error(f"Inconsistent embedding dimensions detected: {unique_dims}")
                st.info("Using fallback method due to dimension inconsistency")
                return self._create_embeddings_fallback(valid_docs, embeddings)

            dimension = list(unique_dims)[0]
            st.info(f"All embeddings have consistent dimension: {dimension}")

            with st.spinner(f"Creating embeddings for {len(valid_docs)} documents..."):
                # Create FAISS vectors
                vectors = FAISS.from_documents(valid_docs, embeddings)

                # Debug: Vector store
                if vectors:
                    st.success(f"Successfully created embeddings for {len(valid_docs)} documents!")
                    st.info(f"Vector store created with {len(valid_docs)} documents, dimension: {dimension}")
                return vectors

        except Exception as e:
            error_msg = str(e)
            st.error(f"Failed to create embeddings: {error_msg}")

            # Debug: Log detailed error info
            if "columns" in error_msg.lower() and "dimension" in error_msg.lower():
                st.error("Dimension mismatch detected - this indicates inconsistent embedding sizes")
                st.info("Attempting to validate all document embeddings...")

                # Try to identify problematic docs
                if self._debug_embedding_dimensions(valid_docs, embeddings):
                    st.info("All dimensions consistent, retrying with fallback method...")
                else:
                    st.warning("Inconsistent dimensions found, attempting fallback...")

                # Attempt fallback method
                return self._create_embeddings_fallback(valid_docs, embeddings)

            # For other errors, try fallback
            st.info("Attempting fallback embedding method...")
            return self._create_embeddings_fallback(valid_docs, embeddings)

    def _process_large_csv(self, df):
        """
        Legacy method - kept for backward compatibility
        """

    def _debug_embedding_dimensions(self, documents, embeddings):
        """
        Debug method to validate embedding dimensions and FAISS compatibility for all documents
        """
        try:
            dimensions = []
            problematic_docs = []

            debug_info = []
            debug_info.append("=== EMBEDDING DIMENSION DEBUG ===")
            debug_info.append(f"Total documents to validate: {len(documents)}")

            # Ensure logs directory exists
            os.makedirs('/app/logs', exist_ok=True)

            # Test embedding creation for each document
            for i, doc in enumerate(documents):
                try:
                    content = doc.page_content[:500]
                    embedding = embeddings.embed_query(content)

                    # Log to file
                    with open('/app/logs/debug_dimensions.log', 'a') as f:
                        f.write(f"Document {i}: content_length={len(content)}, embedding_length={len(embedding)}\n")

                    if not embedding:
                        error_msg = f"Document {i}: Empty embedding"
                        debug_info.append(error_msg)
                        logging.error(error_msg)
                        problematic_docs.append(i)
                        continue

                    dim = len(embedding)
                    dimensions.append(dim)

                    # Test numpy conversion
                    np_embedding = np.array(embedding).astype('float32')

                    if np_embedding.shape != (dim,):
                        error_msg = f"Document {i}: Invalid numpy shape {np_embedding.shape}"
                        debug_info.append(error_msg)
                        logging.error(error_msg)
                        problematic_docs.append(i)

                    # Log individual document results
                    debug_info.append(f"Document {i}: dimension={dim}, numpy_shape={np_embedding.shape}")

                except Exception as e:
                    error_msg = f"Document {i}: Embedding failed - {str(e)}"
                    debug_info.append(error_msg)
                    logging.error(error_msg)
                    problematic_docs.append(i)

            # Comprehensive analysis
            unique_dims = set(dimensions)
            debug_info.append(f"=== ANALYSIS RESULTS ===")
            debug_info.append(f"Found dimensions: {sorted(unique_dims)}")
            debug_info.append(f"Documents with issues: {len(problematic_docs)}")
            debug_info.append(f"Problematic document indices: {problematic_docs}")

            # Log to file
            with open('/app/logs/debug_dimensions.log', 'a') as f:
                f.write(f"Analysis complete:\n")
                f.write(f"Unique dimensions: {sorted(unique_dims)}\n")
                f.write(f"Problematic documents: {problematic_docs}\n")

            # Streamlit output
            for info in debug_info:
                logging.info(info)

            # Validate consistency
            if len(unique_dims) > 1:
                error_msg = f"Inconsistent embedding dimensions detected: {sorted(unique_dims)}"
                logging.error(error_msg)

                # Additional detailed logging
                with open('/app/logs/dimension_error.log', 'w') as f:
                    f.write(f"DIMENSION MISMATCH ERROR\n")
                    f.write(f"Expected: 1536\n")
                    f.write(f"Found: {sorted(unique_dims)}\n")
                    f.write(f"Documents: {len(documents)}\n")
                    for i, doc in enumerate(documents[:5]):  # Log first 5 documents
                        f.write(f"Doc {i}: content_length={len(doc.page_content)}\n")

                return False

            # Test FAISS compatibility
            if unique_dims:
                dim = list(unique_dims)[0]
                success_msg = f"All embeddings have consistent dimension: {dim}"
                logging.info(success_msg)

                # Test with FAISS
                try:
                    faiss = dependable_faiss_import()
                    test_vectors = np.random.rand(1, dim).astype('float32')
                    index = faiss.IndexFlatIP(dim)
                    index.add(test_vectors)

                    faiss_success = "FAISS compatibility test passed"
                    logging.info(faiss_success)

                    # Log FAISS test
                    with open('/app/logs/faiss_test.log', 'w') as f:
                        f.write(f"FAISS test successful\n")
                        f.write(f"Dimension: {dim}\n")
                        f.write(f"Test vectors shape: {test_vectors.shape}\n")

                except Exception as e:
                    faiss_error = f"FAISS compatibility test failed: {str(e)}"
                    logging.error(faiss_error)

                    with open('/app/logs/faiss_error.log', 'w') as f:
                        f.write(f"FAISS test failed: {str(e)}\n")

                    return False

            return True

        except Exception as e:
            error_msg = f"Debug validation failed: {str(e)}"
            logging.error(error_msg)

            with open('/app/logs/debug_error.log', 'w') as f:
                f.write(f"Debug validation error: {str(e)}\n")

            return False
