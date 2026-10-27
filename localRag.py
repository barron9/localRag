"""Local RAG CLI: LlamaIndex + ChromaDB + Ollama, with incremental sync and DB browsing."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from rich.box import ROUNDED
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

console = Console()

# ----------------------------- CONFIG ---------------------------------------
DATA_DIR = Path(r"C:\Users\yowov\OneDrive\Masaüstü\chords")

# Absolute path (independent of the working directory). Better to keep this OUTSIDE
# OneDrive, e.g. set env var RAG_DB_PATH=C:\rag\chroma_db
DB_PATH = Path(os.environ.get("RAG_DB_PATH", Path(__file__).resolve().parent / "chroma_db"))

COLLECTION = "local_docs"
LLM_MODEL = "gemma4:12b"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 4
# ----------------------------------------------------------------------------

with console.status("[cyan]Loading libraries and models...", spinner="dots"):
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama
    from llama_index.vector_stores.chroma import ChromaVectorStore

    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class RAG:
    def __init__(self) -> None:
        DB_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(DB_PATH), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._bind()

    def _bind(self) -> None:
        self.collection = self.client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        self.engine = self.index.as_query_engine(similarity_top_k=TOP_K, streaming=True)
        self.retriever = self.index.as_retriever(similarity_top_k=TOP_K)

    # ---------------- inspection ----------------
    def indexed_files(self) -> dict[str, dict]:
        """file_path -> {'hash': str, 'chunks': int}"""
        out: dict[str, dict] = {}
        res = self.collection.get(include=["metadatas"])
        for m in res["metadatas"] or []:
            key = m.get("file_path")
            if not key:
                continue
            e = out.setdefault(key, {"hash": m.get("file_hash"), "chunks": 0})
            e["chunks"] += 1
        return out

    def disk_files(self) -> dict[str, str]:
        files = [
            p for p in DATA_DIR.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.parts)
        ]
        return {str(p.resolve()): md5(p) for p in files}

    # ---------------- sync (add / update / delete) ----------------
    def sync(self) -> None:
        disk = self.disk_files()
        indexed = self.indexed_files()

        new = [k for k in disk if k not in indexed]
        changed = [k for k in disk if k in indexed and indexed[k]["hash"] != disk[k]]
        removed = [k for k in indexed if k not in disk]

        if not (new or changed or removed):
            console.print("[green]✔ Index is up to date.[/]")
            return

        console.print(
            f"[bold]Changes:[/] [green]+{len(new)} new[/]  "
            f"[yellow]~{len(changed)} changed[/]  [red]-{len(removed)} removed[/]"
        )

        for key in removed + changed:  # drop stale chunks
            self.collection.delete(where={"file_path": key})

        todo = new + changed
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding", total=len(todo))
            for key in todo:
                progress.update(task, description=f"Embedding [cyan]{Path(key).name}[/]")
                try:
                    docs = SimpleDirectoryReader(input_files=[key]).load_data()
                    for d in docs:
                        d.metadata["file_path"] = key
                        d.metadata["file_hash"] = disk[key]
                        # keep bookkeeping out of the embedded/LLM text
                        d.excluded_embed_metadata_keys = list(d.metadata.keys())
                        d.excluded_llm_metadata_keys = ["file_hash", "file_path"]
                        self.index.insert(d)
                except Exception as e:  # noqa: BLE001
                    console.print(f"[red]✘ Failed {Path(key).name}: {e}[/]")
                progress.advance(task)
        console.print("[green]✔ Sync complete.[/]")

    def rebuild(self) -> None:
        self.client.delete_collection(COLLECTION)
        self._bind()
        self.sync()

    # ---------------- querying ----------------
    def ask(self, question: str) -> None:
        with console.status("[cyan]Retrieving...", spinner="dots"):
            response = self.engine.query(question)

        text = ""
        start = time.time()
        with Live(console=console, refresh_per_second=12) as live:
            live.update(Panel(Text("…"), title="Answer", border_style="green", box=ROUNDED))
            for token in response.response_gen:
                text += token
                live.update(Panel(Markdown(text), title="Answer", border_style="green", box=ROUNDED))
        console.print(f"[dim]{time.time() - start:.1f}s[/]")

        if response.source_nodes:
            t = Table(title="Sources", box=ROUNDED, border_style="dim", show_lines=True)
            t.add_column("#", justify="right")
            t.add_column("File", style="cyan")
            t.add_column("Score", justify="right")
            t.add_column("Snippet", overflow="fold")
            for i, n in enumerate(response.source_nodes, 1):
                name = Path(n.node.metadata.get("file_path", "?")).name
                snippet = n.node.get_content().strip().replace("\n", " ")[:120]
                t.add_row(str(i), name, f"{n.score:.3f}" if n.score is not None else "-", snippet + "…")
            console.print(t)


# ---------------- UI helpers ----------------
HELP = [
    ("/help", "Show this help"),
    ("/stats", "Database statistics"),
    ("/files", "List indexed files (with numbers)"),
    ("/peek <n|name>", "Show stored chunks of a file"),
    ("/retrieve <text>", "Show raw retrieved chunks, no LLM"),
    ("/sync", "Add new / update changed / remove deleted files"),
    ("/rebuild", "Wipe the collection and re-embed everything"),
    ("/export", "Dump all chunks to chroma_export.json"),
    ("/clear", "Clear the screen"),
    ("/exit", "Quit"),
]


def show_help() -> None:
    t = Table(box=ROUNDED, border_style="dim", show_header=False)
    t.add_column(style="bold cyan")
    t.add_column()
    for c, d in HELP:
        t.add_row(c, d)
    console.print(Panel(t, title="Commands", border_style="cyan"))


def show_stats(rag: RAG) -> None:
    files = rag.indexed_files()
    t = Table(box=ROUNDED, border_style="dim", show_header=False)
    t.add_column(style="bold")
    t.add_column()
    t.add_row("Source folder", str(DATA_DIR))
    t.add_row("DB path", str(DB_PATH))
    t.add_row("DB size on disk", human(dir_size(DB_PATH)))
    t.add_row("Collection", COLLECTION)
    t.add_row("Indexed files", str(len(files)))
    t.add_row("Chunks", str(rag.collection.count()))
    t.add_row("LLM / Embeddings", f"{LLM_MODEL} / {EMBED_MODEL}")
    console.print(Panel(t, title="Stats", border_style="cyan"))


def show_files(rag: RAG, state: dict) -> None:
    files = rag.indexed_files()
    state["files"] = sorted(files)
    t = Table(title=f"{len(files)} indexed files", box=ROUNDED, border_style="dim")
    t.add_column("#", justify="right")
    t.add_column("File", style="cyan")
    t.add_column("Chunks", justify="right")
    t.add_column("Hash", style="dim")
    for i, k in enumerate(state["files"], 1):
        t.add_row(str(i), Path(k).name, str(files[k]["chunks"]), (files[k]["hash"] or "")[:8])
    console.print(t)


def peek(rag: RAG, state: dict, arg: str) -> None:
    if not state.get("files"):
        state["files"] = sorted(rag.indexed_files())
    key = None
    if arg.isdigit() and 1 <= int(arg) <= len(state["files"]):
        key = state["files"][int(arg) - 1]
    else:
        matches = [k for k in state["files"] if arg.lower() in Path(k).name.lower()]
        key = matches[0] if matches else None
    if not key:
        console.print("[red]File not found. Run /files first.[/]")
        return
    res = rag.collection.get(where={"file_path": key}, include=["documents"])
    console.print(f"[bold cyan]{Path(key).name}[/] — {len(res['ids'])} chunks")
    for i, (cid, doc) in enumerate(zip(res["ids"], res["documents"]), 1):
        console.print(Panel(doc, title=f"chunk {i}", subtitle=cid[:8], border_style="dim"))


def retrieve(rag: RAG, q: str) -> None:
    for i, n in enumerate(rag.retriever.retrieve(q), 1):
        console.print(Panel(
            n.node.get_content(),
            title=f"{i}. {Path(n.node.metadata.get('file_path', '?')).name}",
            subtitle=f"score {n.score:.3f}" if n.score is not None else "",
            border_style="magenta",
        ))


def export(rag: RAG) -> None:
    res = rag.collection.get(include=["documents", "metadatas"])
    rows = [{"id": i, "metadata": m, "text": d}
            for i, m, d in zip(res["ids"], res["metadatas"], res["documents"])]
    for r in rows:  # drop huge internal blob
        r["metadata"].pop("_node_content", None)
    out = Path("chroma_export.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]✔ Exported {len(rows)} chunks → {out.resolve()}[/]")


def main() -> None:
    console.print(Panel.fit(
        "[bold cyan]Local RAG Console[/]\n[dim]LlamaIndex • ChromaDB • Ollama[/]",
        border_style="cyan", box=ROUNDED,
    ))
    rag = RAG()
    rag.sync()  # incremental; does nothing if nothing changed
    show_stats(rag)
    console.print("[dim]Type a question, or /help for commands.[/]\n")

    state: dict = {}
    while True:
        try:
            line = Prompt.ask("[bold cyan]❯[/]").strip()
            if not line:
                continue
            if line.startswith("/"):
                cmd, _, arg = line[1:].partition(" ")
                arg = arg.strip()
                if cmd in ("exit", "quit", "q"):
                    break
                elif cmd == "help": show_help()
                elif cmd == "stats": show_stats(rag)
                elif cmd == "files": show_files(rag, state)
                elif cmd == "peek": peek(rag, state, arg)
                elif cmd == "retrieve": retrieve(rag, arg)
                elif cmd == "sync": rag.sync()
                elif cmd == "export": export(rag)
                elif cmd == "clear": console.clear()
                elif cmd == "rebuild":
                    if Confirm.ask("[yellow]Delete everything and re-embed?[/]"):
                        rag.rebuild()
                else:
                    console.print("[red]Unknown command.[/] /help")
                continue
            if rag.collection.count() == 0:
                console.print("[yellow]The index is empty. Put files in the folder and run /sync.[/]")
                continue
            rag.ask(line)
            console.print()
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]Error:[/] {e}")

    console.print("[dim]Goodbye 👋[/]")


if __name__ == "__main__":
    main()
