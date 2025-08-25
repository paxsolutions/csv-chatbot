import os
import pickle
import tempfile
import pandas as pd
import hashlib
import time
import random
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import streamlit as st


cwd = os.getcwd()
log_dir = f'{cwd}/logs'

os.makedirs(log_dir, exist_ok=True)

# Create log files if they don't exist
log_files = [
    'faiss_test.log',
    'faiss_error.log',
    'debug_error.log',
    'debug_dimensions.log',
    'debug_dimension_error.log'
]

for log_file in log_files:
    log_path = f'{log_dir}/{log_file}'
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/embedder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def dependable_faiss_import():
    """Import FAISS with fallback handling"""
    try:
        import faiss
        return faiss
    except ImportError:
        st.error("FAISS not installed. Please install with: pip install faiss-cpu")
        raise ImportError("FAISS library not found")


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
        # Log entry point
        logger.info(f"Starting storeDocEmbeds for file: {filename}")
        with open(f'{log_dir}/embedder.log', 'a') as f:
            f.write(f"ENTRY: storeDocEmbeds called for {filename}\n")

        try:
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: About to generate file hash\n")

            # Caching
            file_hash = self._get_file_hash(file)

            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: File hash generated: {file_hash[:10]}...\n")

            cache_filename = f"{filename}_{file_hash}"

            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: Checking for cached directory: {cache_filename}\n")

            # Check if cached version exists (FAISS format)
            cache_path = f"{self.PATH}/{cache_filename}"
            if os.path.isdir(cache_path):
                st.info("Using cached embeddings for faster loading...")
                with open(f'{log_dir}/embedder.log', 'a') as f:
                    f.write(f"STEP: Found cached directory, loading...\n")
                try:
                    embeddings = OpenAIEmbeddings()
                    cached_vectors = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
                    with open(f'{log_dir}/embedder.log', 'a') as f:
                        f.write(f"SUCCESS: Cached vectors loaded successfully\n")
                    return cached_vectors
                except Exception as e:
                    error_msg = f"Cached vectors corrupted: {str(e)}"
                    with open(f'{log_dir}/embedder.log', 'a') as f:
                        f.write(f"ERROR: {error_msg}\n")
                        f.write(f"STEP: Removing corrupted cache and regenerating\n")
                    st.warning("Cached vectors corrupted, regenerating embeddings...")
                    # Remove corrupted cache directory
                    try:
                        import shutil
                        shutil.rmtree(cache_path)
                    except:
                        pass

            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: No cached file found, proceeding with processing\n")

        except Exception as e:
            error_msg = f"Error in storeDocEmbeds caching: {str(e)}"
            logger.error(error_msg)
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"ERROR: {error_msg}\n")
            st.error(error_msg)
            return None

        # Write the uploaded file to a temp file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name

        try:
            # Log progress
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: Reading CSV file {filename}\n")

            # Read CSV with pandas and optimize for large files
            df = pd.read_csv(tmp_file_path, encoding="utf-8")
            df = self._sample_large_csv(df)

            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: CSV loaded, shape: {df.shape}\n")

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
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: Creating OpenAI embeddings instance\n")

            embeddings = OpenAIEmbeddings()

            # Process embeddings in single batch to avoid dimension issues
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: Calling _create_embeddings_single_batch with {len(split_docs)} documents\n")

            all_vectors = self._create_embeddings_single_batch(split_docs, embeddings)

            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: _create_embeddings_single_batch returned: {type(all_vectors)}\n")

            # Save cache using FAISS's built-in save method
            try:
                cache_path = f"{self.PATH}/{cache_filename}"
                all_vectors.save_local(cache_path)

                # Also save with original filename
                original_path = f"{self.PATH}/{filename.replace('.csv', '')}"
                all_vectors.save_local(original_path)

                with open(f'{log_dir}/embedder.log', 'a') as f:
                    f.write(f"SUCCESS: Saved vectors to cache using FAISS save_local\n")
            except Exception as e:
                with open(f'{log_dir}/embedder.log', 'a') as f:
                    f.write(f"WARNING: Failed to save cache: {str(e)}\n")
                # Continue without caching if save fails

            # Return vectors
            return all_vectors

        finally:
            os.remove(tmp_file_path)

    def _process_large_csv_optimized(self, df):
        """
        Optimized processing for large CSV files with better numerical accuracy
        """
        documents = []
        chunk_size = 50  # Smaller chunks for better accuracy
        columns = df.columns.tolist()

        # Create a summary document with key statistics
        summary_text = f"CSV SUMMARY:\nColumns: {', '.join(columns)}\nTotal rows: {len(df)}\n\n"

        # Add numerical column summaries for better accuracy
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                max_val = df[col].max()
                max_idx = df[col].idxmax()
                max_product = df.loc[max_idx, 'Product_name'] if 'Product_name' in df.columns else 'N/A'
                summary_text += f"{col} - Maximum: {max_val} (Product: {max_product})\n"

        summary_doc = Document(
            page_content=summary_text,
            metadata={"source": "csv_summary", "type": "summary"}
        )
        documents.append(summary_doc)

        # Process data in smaller chunks with headers
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            chunk_text = f"DATA CHUNK {i//chunk_size + 1}:\n"
            chunk_text += f"Columns: {', '.join(columns)}\n\n"
            chunk_text += chunk_df.to_csv(index=False, header=True)

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": f"csv_chunk_{i//chunk_size + 1}",
                    "rows": f"{i+1}-{min(i+chunk_size, len(df))}",
                    "total_rows": len(df),
                    "columns": columns,
                    "type": "data_chunk"
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
        try:
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"ENTRY: _create_embeddings_single_batch with {len(split_docs)} docs\n")

            valid_docs = [doc for doc in split_docs if doc.page_content.strip()]

            if not valid_docs:
                st.error("No valid documents to process")
                with open(f'{log_dir}/embedder.log', 'a') as f:
                    f.write(f"ERROR: No valid documents found\n")
                return None
        except Exception as e:
            error_msg = f"Error in _create_embeddings_single_batch entry: {str(e)}"
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"ERROR: {error_msg}\n")
            st.error(error_msg)
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

            # Validate all documents have content
            filtered_docs = []
            for doc in valid_docs:
                if doc.page_content and len(doc.page_content.strip()) > 10:  # Minimum content length
                    filtered_docs.append(doc)

            if not filtered_docs:
                st.error("No valid documents with sufficient content found")
                return None

            st.info(f"Processing {len(filtered_docs)} valid documents")

            # Test with a small batch first to validate dimensions
            test_batch = filtered_docs[:3] if len(filtered_docs) > 3 else filtered_docs
            test_texts = [doc.page_content for doc in test_batch]

            try:
                test_embeddings = embeddings.embed_documents(test_texts)
                test_dimensions = [len(emb) for emb in test_embeddings]

                if len(set(test_dimensions)) > 1:
                    st.error(f"Inconsistent test embedding dimensions: {set(test_dimensions)}")
                    return self._create_embeddings_fallback(filtered_docs, embeddings)

                expected_dim = test_dimensions[0]
                st.info(f"Validated embedding dimension: {expected_dim}")

            except Exception as e:
                st.error(f"Test embedding failed: {str(e)}")
                return self._create_embeddings_fallback(filtered_docs, embeddings)

            with st.spinner(f"Creating embeddings for {len(filtered_docs)} documents..."):
                try:
                    with open(f'{log_dir}/embedder.log', 'a') as f:
                        f.write(f"CRITICAL: About to call FAISS.from_documents with {len(filtered_docs)} docs\n")

                    # Create FAISS vectors with validated documents
                    vectors = FAISS.from_documents(filtered_docs, embeddings)

                    with open(f'{log_dir}/embedder.log', 'a') as f:
                        f.write(f"SUCCESS: FAISS.from_documents completed successfully\n")

                except Exception as e:
                    error_msg = f"FAISS creation failed: {str(e)}"
                    with open(f'{log_dir}/embedder.log', 'a') as f:
                        f.write(f"FAISS ERROR: {error_msg}\n")
                        f.write(f"Exception type: {type(e).__name__}\n")
                        f.write(f"Full traceback: {str(e)}\n")

                    st.error(error_msg)
                    st.info("Attempting fallback method...")
                    return self._create_embeddings_fallback(filtered_docs, embeddings)

                # Debug: Vector store
                if vectors:
                    st.success(f"Successfully created embeddings for {len(filtered_docs)} documents!")
                    st.info(f"Vector store created with {len(filtered_docs)} documents, dimension: {expected_dim}")
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

            # Test embedding creation for each document
            for i, doc in enumerate(documents):
                try:
                    content = doc.page_content[:500]
                    embedding = embeddings.embed_query(content)

                    # Log to file
                    with open(f'{log_dir}/debug_dimensions.log', 'a') as f:
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



            with open(f'{log_dir}/debug_dimensions.log', 'a') as f:
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
                with open(f'{log_dir}/debug_dimension_error.log', 'w') as f:
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
                    with open(f'{log_dir}/faiss_test.log', 'w') as f:
                        f.write(f"FAISS test successful\n")
                        f.write(f"Dimension: {dim}\n")
                        f.write(f"Test vectors shape: {test_vectors.shape}\n")

                except Exception as e:
                    faiss_error = f"FAISS compatibility test failed: {str(e)}"
                    logging.error(faiss_error)

                    with open(f'{log_dir}/faiss_error.log', 'w') as f:
                        f.write(f"FAISS test failed: {str(e)}\n")

                    return False

            return True

        except Exception as e:
            error_msg = f"Debug validation failed: {str(e)}"
            logging.error(error_msg)

            with open(f'{log_dir}/debug_error.log', 'w') as f:
                f.write(f"Debug validation error: {str(e)}\n")

            return False

    def _create_embeddings_fallback(self, documents, embeddings):
        """
        Fallback method for creating embeddings when the main method fails
        """
        try:
            st.info("Using fallback embedding method...")

            # Process documents in smaller batches
            batch_size = 10
            all_batches = []

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                valid_batch = [doc for doc in batch if doc.page_content.strip()]

                if valid_batch:
                    try:
                        # Create embeddings for this batch
                        batch_vectors = FAISS.from_documents(valid_batch, embeddings)
                        all_batches.append(batch_vectors)
                        st.info(f"Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                    except Exception as e:
                        st.warning(f"Skipped batch {i//batch_size + 1} due to error: {str(e)}")
                        continue

            if not all_batches:
                st.error("All batches failed to process")
                return None

            # Merge all successful batches
            if len(all_batches) == 1:
                return all_batches[0]

            # Merge multiple batches
            merged_vectors = all_batches[0]
            for batch_vectors in all_batches[1:]:
                try:
                    merged_vectors.merge_from(batch_vectors)
                except Exception as e:
                    st.warning(f"Failed to merge batch: {str(e)}")
                    continue

            st.success(f"Successfully created embeddings using fallback method with {len(all_batches)} batches")
            return merged_vectors

        except Exception as e:
            st.error(f"Fallback method also failed: {str(e)}")
            return None
