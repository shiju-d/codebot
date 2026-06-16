import os

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import TokenTextSplitter, CodeSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import ServiceConfig

# Extension → tree-sitter language name.
# Extensions absent from this map fall back to _fallback_splitter.
_EXT_LANGUAGE: dict[str, str] = {
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".rb": "ruby",
    ".rake": "ruby",   # Rake DSL is valid Ruby
}

# Token-based splitter used for file types tree-sitter doesn't cover
# (.yml, .json, .erb, etc.) and as a fallback when parsing fails.
_fallback_splitter = TokenTextSplitter(
    chunk_size=600, chunk_overlap=100, separator="\n",
    backup_separators=["class ", "function ", "const ", "export ", "  "],
)

# CodeSplitter instances are expensive to create (they load grammars), so cache by language.
_code_splitters: dict[str, CodeSplitter] = {}


def _get_splitter(file_ext: str):
    lang = _EXT_LANGUAGE.get(file_ext.lower())
    if not lang:
        return _fallback_splitter
    if lang not in _code_splitters:
        _code_splitters[lang] = CodeSplitter(
            language=lang,
            chunk_lines=50,
            chunk_lines_overlap=15,
            max_chars=2000,
        )
    return _code_splitters[lang]


def build_service_index(svc: ServiceConfig) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
    collection_name = f"{svc.name}_codebase"
    collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if collection.count() > 0:
        print(f"[{svc.name}] Loading existing index ({collection.count()} chunks)...")
        return VectorStoreIndex.from_vector_store(vector_store)

    print(f"[{svc.name}] Building index from {len(svc.repos)} repo(s)...")
    all_nodes = []
    for repo_path in svc.repos:
        reader = SimpleDirectoryReader(
            input_dir=repo_path, recursive=True,
            required_exts=svc.file_extensions, exclude_hidden=True,
            exclude=[
                "**/node_modules/**", "**/dist/**", "**/.git/**",
                "**/log/**", "**/tmp/**",
                "**/__tests__/**", "**/*.spec.ts", "**/*.test.ts",
                "**/cypress/**", "**/e2e/**",
                "**/vendor/**", "**/coverage/**",
            ],
        )
        docs = reader.load_data()
        # Strip the /repos/ mount prefix so file paths in responses are repo-relative
        # e.g. /repos/ibe-admin/src/foo.ts → ibe-admin/src/foo.ts
        repos_prefix = os.path.dirname(repo_path).rstrip("/") + "/"
        for doc in docs:
            if "file_path" in doc.metadata:
                doc.metadata["file_path"] = doc.metadata["file_path"].removeprefix(repos_prefix)
            ext = os.path.splitext(doc.metadata.get("file_path", ""))[1]
            splitter = _get_splitter(ext)
            try:
                all_nodes.extend(splitter.get_nodes_from_documents([doc]))
            except Exception as exc:
                fp = doc.metadata.get("file_path", "?")
                print(f"[{svc.name}] CodeSplitter failed for {fp} ({exc}); falling back to token splitter")
                all_nodes.extend(_fallback_splitter.get_nodes_from_documents([doc]))

    index = VectorStoreIndex(all_nodes, storage_context=storage_context)
    print(f"[{svc.name}] Index built ({len(all_nodes)} chunks).")
    return index
