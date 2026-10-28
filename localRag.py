"""Local RAG CLI: LlamaIndex + ChromaDB + Ollama, with incremental sync, DB browsing,
an agent mode (ReAct / function-calling) and web-surfing tools.

Extra deps:
    pip install -U "llama-index-core>=0.12" ddgs requests beautifulsoup4
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import chromadb
import requests
from bs4 import BeautifulSoup
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
DATA_DIR = Path(r"C:\Users\yowov\Downloads\TXT")

# Absolute path (independent of the working directory). Better to keep this OUTSIDE
# OneDrive, e.g. set env var RAG_DB_PATH=C:\rag\chroma_db
DB_PATH = Path(os.environ.get("RAG_DB_PATH", Path(__file__).resolve().parent / "chroma_db"))

COLLECTION = "local_docs"
LLM_MODEL = "gemma4:12b"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # "BAAI/bge-small-en-v1.5"
TOP_K = 4

# Agent settings
AGENT_TYPE = "react"        # "react" works with any model; "function" needs native tool-calling support
AGENT_MAX_STEPS = 10        # max reasoning/tool iterations per question
WEB_MAX_RESULTS = 5
WEB_FETCH_MAX_CHARS = 6000  # page text handed back to the LLM
WEB_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (compatible; LocalRAG/1.0)"
# ----------------------------------------------------------------------------

with console.status("[cyan]Loading libraries and models...", spinner="dots"):
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.agent.workflow import (
        AgentStream,
        FunctionAgent,
        ReActAgent,
        ToolCall,
        ToolCallResult,
    )
    from llama_index.core.tools import FunctionTool
    from llama_index.core.workflow import Context
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama
    from llama_index.vector_stores.chroma import ChromaVectorStore

    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300.0)
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


# =============================== WEB TOOLS ===================================
def _is_public_url(url: str) -> tuple[bool, str]:
    """Basic SSRF guard: only http(s) and only publicly routable addresses."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False, "Only http/https URLs are allowed."
    try:
        for info in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                return False, "Refusing to fetch a local/private address."
    except socket.gaierror:
        return False, "Could not resolve host."
    return True, ""


def web_search(query: str) -> str:
    """Search the web (DuckDuckGo) and return titles, URLs and snippets.
    Use for current events or anything not found in the local documents."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:  # older package name
            from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=WEB_MAX_RESULTS))
    except Exception as e:  # noqa: BLE001
        return f"Web search failed: {e}"
    if not results:
        return "No results."
    return "\n\n".join(
        f"[{i}] {r.get('title')}\nURL: {r.get('href')}\n{r.get('body', '')}"
        for i, r in enumerate(results, 1)
    )


def web_fetch(url: str) -> str:
    """Download a web page and return its readable text (truncated).
    Use after web_search to read a promising result in full."""
    ok, why = _is_public_url(url)
    if not ok:
        return f"Blocked: {why}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=WEB_TIMEOUT)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"Fetch failed: {e}"
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        return f"Unsupported content type: {ctype}"
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else url
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    if len(text) > WEB_FETCH_MAX_CHARS:
        text = text[:WEB_FETCH_MAX_CHARS] + "\n…[truncated]"
    return f"# {title}\nSource: {url}\n\n{text}"


class RAG:
    def __init__(self) -> None:
        DB_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(DB_PATH), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.loop = asyncio.new_event_loop()  # persistent loop so agent Context survives between turns
        self.agent = None
        self.ctx = None
        self._bind()

    def _bind(self) -> None:
        self.collection = self.client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        self.engine = self.index.as_query_engine(similarity_top_k=TOP_K, streaming=True)
        self.retriever = self.index.as_retriever(similarity_top_k=TOP_K)
        self.agent = None  # tools hold the old retriever -> rebuild lazily
        self.ctx = None

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

    # ---------------- querying (plain RAG) ----------------
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
                snippet = n.node.get_content()  # .strip().replace("\n", " ")[:120]
                t.add_row(str(i), name, f"{n.score:.3f}" if n.score is not None else "-", snippet + "…")
            console.print(t)

    # ---------------- agent ----------------
    def _doc_tools(self) -> list[FunctionTool]:
        retriever = self.retriever
        collection = self.collection

        def search_documents(query: str) -> str:
            """Semantic search over the user's LOCAL indexed documents.
            Returns the most relevant chunks with file names and scores.
            Try this first for any question that may relate to the user's files."""
            nodes = retriever.retrieve(query)
            if not nodes:
                return "No relevant chunks found."
            return "\n\n".join(
                f"[{i}] file={Path(n.node.metadata.get('file_path', '?')).name} "
                f"score={n.score:.3f}\n{n.node.get_content()}"
                if n.score is not None else
                f"[{i}] file={Path(n.node.metadata.get('file_path', '?')).name}\n{n.node.get_content()}"
                for i, n in enumerate(nodes, 1)
            )

        def list_documents() -> str:
            """List the names of all indexed local files."""
            names = sorted({
                Path(m["file_path"]).name
                for m in (collection.get(include=["metadatas"])["metadatas"] or [])
                if m.get("file_path")
            })
            return "\n".join(names) or "The index is empty."

        def read_document(filename: str) -> str:
            """Return the stored text of an indexed file, matched by (part of) its name.
            Output is truncated; use search_documents for targeted lookups."""
            metas = collection.get(include=["metadatas"])["metadatas"] or []
            keys = sorted({m["file_path"] for m in metas if m.get("file_path")})
            matches = [k for k in keys if filename.lower() in Path(k).name.lower()]
            if not matches:
                return "No indexed file matches that name. Call list_documents."
            res = collection.get(where={"file_path": matches[0]}, include=["documents"])
            text = "\n\n".join(res["documents"])
            return f"{Path(matches[0]).name}:\n{text[:8000]}" + ("\n…[truncated]" if len(text) > 8000 else "")

        return [FunctionTool.from_defaults(fn=f) for f in (search_documents, list_documents, read_document)]

    def _build_agent(self):
        tools = self._doc_tools() + [
            FunctionTool.from_defaults(fn=web_search),
            FunctionTool.from_defaults(fn=web_fetch),
        ]
        system = (
            "You are a research assistant with two knowledge sources: the user's local documents "
            "(search_documents, list_documents, read_document) and the web (web_search, web_fetch). "
            "Prefer local documents first; use the web for current or missing information. "
            "After web_search, use web_fetch to read the best result before answering when snippets "
            "are not enough. Never invent facts. Cite local file names and web URLs you relied on. "
            "Answer concisely in the user's language."
        )
        cls = FunctionAgent if AGENT_TYPE == "function" else ReActAgent
        self.agent = cls(tools=tools, llm=Settings.llm, system_prompt=system)
        self.ctx = Context(self.agent)  # keeps conversation memory across questions

    def reset_agent_memory(self) -> None:
        self.agent = None
        self.ctx = None

    async def _run_agent(self, question: str, status) -> str:
        handler = self.agent.run(user_msg=question, ctx=self.ctx, max_iterations=AGENT_MAX_STEPS)
        async for ev in handler.stream_events():
            if isinstance(ev, ToolCall):
                args = json.dumps(ev.tool_kwargs, ensure_ascii=False)
                console.print(f"[magenta]🔧 {ev.tool_name}[/] [dim]{args[:200]}[/]")
                status.update(f"[cyan]Running {ev.tool_name}...")
            elif isinstance(ev, ToolCallResult):
                out = str(ev.tool_output.content).replace("\n", " ")
                console.print(f"[dim]   ↳ {out[:160]}{'…' if len(out) > 160 else ''}[/]")
                status.update("[cyan]Thinking...")
            elif isinstance(ev, AgentStream):
                pass  # raw ReAct tokens contain Thought/Action scaffolding; show only the final answer
        result = await handler
        return str(result)

    def agent_ask(self, question: str) -> None:
        if self.agent is None:
            self._build_agent()
        start = time.time()
        try:
            with console.status("[cyan]Thinking...", spinner="dots") as status:
                answer = self.loop.run_until_complete(self._run_agent(question, status))
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]Agent error:[/] {e}")
            if AGENT_TYPE == "function":
                console.print("[dim]Does this model support tool calling? Try AGENT_TYPE = 'react'.[/]")
            return
        console.print(Panel(Markdown(answer), title=f"Agent answer ({AGENT_TYPE})", border_style="green", box=ROUNDED))
        console.print(f"[dim]{time.time() - start:.1f}s[/]")


# ---------------- UI helpers ----------------
HELP = [
    ("/help", "Show this help"),
    ("/stats", "Database statistics"),
    ("/files", "List indexed files (with numbers)"),
    ("/peek <n|name>", "Show stored chunks of a file"),
    ("/retrieve <text>", "Show raw retrieved chunks, no LLM"),
    ("/agent [on|off]", "Toggle agent mode (docs + web tools, multi-step reasoning)"),
    ("/forget", "Reset the agent's conversation memory"),
    ("/web <query>", "Raw web search, no LLM"),
    ("/fetch <url>", "Fetch a page's readable text, no LLM"),
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


def show_stats(rag: RAG, state: dict | None = None) -> None:
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
    if state is not None:
        t.add_row("Agent mode", f"{'ON' if state.get('agent') else 'off'} ({AGENT_TYPE})")
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
        "[bold cyan]Local RAG Console[/]\n[dim]LlamaIndex • ChromaDB • Ollama • Agent + Web[/]",
        border_style="cyan", box=ROUNDED,
    ))
    rag = RAG()
    rag.sync()  # incremental; does nothing if nothing changed
    state: dict = {"agent": False}
    show_stats(rag, state)
    console.print("[dim]Type a question, or /help for commands. Use /agent to enable web + multi-step mode.[/]\n")

    while True:
        try:
            tag = "[bold magenta]agent ❯[/]" if state["agent"] else "[bold cyan]❯[/]"
            line = Prompt.ask(tag).strip()
            if not line:
                continue
            if line.startswith("/"):
                cmd, _, arg = line[1:].partition(" ")
                arg = arg.strip()
                if cmd in ("exit", "quit", "q"):
                    break
                elif cmd == "help": show_help()
                elif cmd == "stats": show_stats(rag, state)
                elif cmd == "files": show_files(rag, state)
                elif cmd == "peek": peek(rag, state, arg)
                elif cmd == "retrieve": retrieve(rag, arg)
                elif cmd == "sync":
                    rag.sync()
                    rag.reset_agent_memory()
                elif cmd == "export": export(rag)
                elif cmd == "clear": console.clear()
                elif cmd == "agent":
                    if arg in ("on", "off"):
                        state["agent"] = arg == "on"
                    else:
                        state["agent"] = not state["agent"]
                    console.print(f"[magenta]Agent mode {'ON' if state['agent'] else 'off'}[/]")
                elif cmd == "forget":
                    rag.reset_agent_memory()
                    console.print("[green]✔ Agent memory cleared.[/]")
                elif cmd == "web":
                    with console.status("[cyan]Searching the web...", spinner="dots"):
                        out = web_search(arg)
                    console.print(Panel(out, title=f"Web: {arg}", border_style="magenta"))
                elif cmd == "fetch":
                    with console.status("[cyan]Fetching...", spinner="dots"):
                        out = web_fetch(arg)
                    console.print(Panel(out, title="Page", border_style="magenta"))
                elif cmd == "rebuild":
                    if Confirm.ask("[yellow]Delete everything and re-embed?[/]"):
                        rag.rebuild()
                else:
                    console.print("[red]Unknown command.[/] /help")
                continue
            if state["agent"]:
                rag.agent_ask(line)  # works even with an empty index (web only)
                console.print()
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
