~"""
Main CLI for CV AI system.
Commands: ingest, search, match, status, email, send-scheduled, email-stats, insights, evaluate, cluster.
Includes cross‑encoder reranking, skill priority, JD parsing, enhanced fit score,
skill gap detection, min experience filter, diversity, advanced filters,
query expansion, caching, boolean query support, email tracking with scheduling,
retrieval evaluation metrics, parallel ingestion, Redis caching, and data intelligence clustering.
"""
from __future__ import annotations

import sys
import time
import logging
import json
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Set

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from tqdm import tqdm
from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import config, setup_logging
    from app.retrieval.retriever import HybridCVRetriever
    from app.storage.vector_store import vector_store
    from app.ingest.extractor import extract_structured_data
    from app.ingest.indexer import index_cv
    from app.jd_parser import parse_job_description
    from app.email.tracker import EmailTracker
    from app.evaluation.golden import GoldenSet
    from app.evaluation import metrics
except Exception as e:
    raise RuntimeError(f"❌ Import failure — check project structure: {e}") from e

setup_logging()
logger = logging.getLogger(__name__)

app = typer.Typer(help="CV AI — Hybrid Retrieval System")
console = Console()

_retriever: Optional[HybridCVRetriever] = None
_email_tracker: Optional[EmailTracker] = None

def get_retriever() -> HybridCVRetriever:
    global _retriever
    if _retriever is None:
        logger.info("Initialising HybridCVRetriever singleton")
        cross_encoder = None
        try:
            cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
            logger.info(f"✅ Loaded cross-encoder: {config.CROSS_ENCODER_MODEL}")
        except Exception as e:
            logger.warning(f"Cross-encoder not available, proceeding without reranking: {e}")

        # Attempt to connect to Redis only if host is not 'localhost' or if explicitly enabled
        redis_client = None
        if config.REDIS_HOST and config.REDIS_HOST != "localhost" and config.REDIS_HOST != "redis":
            try:
                import redis
                # Use a short connection timeout to avoid hanging
                redis_client = redis.Redis(
                    host=config.REDIS_HOST,
                    port=config.REDIS_PORT,
                    db=config.REDIS_DB,
                    password=config.REDIS_PASSWORD,
                    socket_connect_timeout=2,   # seconds
                    socket_timeout=2,
                    decode_responses=False,
                )
                redis_client.ping()
                logger.info("✅ Connected to Redis")
            except Exception as e:
                logger.warning(f"Redis not available, using in‑memory cache: {e}")
                redis_client = None
        else:
            logger.info("Redis not configured, using in‑memory cache.")

        _retriever = HybridCVRetriever(
            collection=vector_store.collection,
            embedder=vector_store.embedder,
            bm25_index_path=config.BM25_INDEX_PATH,
            cross_encoder=cross_encoder,
            redis_client=redis_client,
        )
        try:
            _retriever.build_bm25_index()
        except Exception as e:
            logger.warning(f"BM25 warmup failed: {e}")
    return _retriever

def get_email_tracker() -> EmailTracker:
    global _email_tracker
    if _email_tracker is None:
        _email_tracker = EmailTracker(config.EMAIL_TRACKER_DB)
    return _email_tracker

def parse_priority_skills(priority_str: Optional[str]) -> Optional[Dict[str, float]]:
    if not priority_str:
        return None
    result = {}
    parts = priority_str.split(',')
    for part in parts:
        part = part.strip()
        if ':' in part:
            skill, weight = part.split(':', 1)
            try:
                weight = float(weight)
            except ValueError:
                weight = 1.0
        else:
            skill = part
            weight = 1.0
        result[skill.strip().lower()] = weight
    return result

def display_results(results: List[Dict], query: str, alpha: Optional[float] = None):
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    table = Table(title=f"🔍 Results for: '{query}'" + (f" (α={alpha})" if alpha is not None else ""))
    table.add_column("#", style="dim")
    table.add_column("Name")
    table.add_column("Score", justify="right")
    table.add_column("Experience", justify="right")
    table.add_column("Skills (first 3)")
    for i, r in enumerate(results):
        meta = r.get("metadata", {}) or {}
        name = meta.get("name", "N/A")
        score = r.get("score", 0.0)
        exp = meta.get("years_experience", 0)
        skills_raw = meta.get("skills", "")
        skills_list = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else []
        skills_display = ", ".join(skills_list[:3])
        table.add_row(str(i+1), name, f"{score:.3f}", str(exp), skills_display)
    console.print(table)

def display_match_results(results: List[Dict], jd_summary: Dict):
    if not results:
        console.print("[yellow]No candidates found.[/yellow]")
        return

    console.print(Panel(f"[bold]Job Description Analysis[/bold]\n"
                        f"Skills: {', '.join(jd_summary['skills'])}\n"
                        f"Min Experience: {jd_summary['min_experience']} years\n"
                        f"Education: {jd_summary['education'] or 'Not specified'}", title="JD Summary"))

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {}) or {}
        name = meta.get("name", "N/A")
        score = r.get("score", 0.0)
        exp = meta.get("years_experience", 0)
        edu = meta.get("education_level", "unknown")
        seniority = meta.get("seniority", "unknown")
        skills_raw = meta.get("skills", "")
        skills_list = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else []

        breakdown = r.get("fit_breakdown", {})
        missing = r.get("missing_skills", [])

        content = Text()
        content.append(f"Name: {name}\n", style="bold cyan")
        content.append(f"Overall Score: {score:.3f}\n", style="bold yellow")
        content.append(f"Experience: {exp} years\n")
        content.append(f"Education: {edu}\n")
        content.append(f"Seniority: {seniority}\n")
        content.append(f"Skills: {', '.join(skills_list[:5])}\n")
        content.append("\n[Fit Breakdown]\n", style="underline")
        for comp, val in breakdown.items():
            content.append(f"  {comp}: {val:.3f}\n")
        if missing:
            content.append(f"\n[Missing Skills]\n", style="bold red")
            content.append(f"  {', '.join(missing)}\n")
        else:
            content.append(f"\n[Missing Skills]\n", style="bold green")
            content.append("  None\n")

        panel = Panel(content, title=f"#{i}", border_style="blue")
        console.print(panel)

# ======================================================================
# SEARCH
# ======================================================================
@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", min=1, max=100),
    alpha: float = typer.Option(0.5, "--alpha", help="0=vector, 1=BM25"),
    min_exp: Optional[float] = typer.Option(None, "--min-exp", help="Minimum years of experience (hard filter)"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable cross-encoder reranking"),
    priority_skills: Optional[str] = typer.Option(None, "--priority-skills", help="Comma-separated skill:weight pairs, e.g. 'python:2.0,docker:1.5'"),
    diversity: Optional[float] = typer.Option(None, "--diversity", help="Enable diversity with lambda (0-1); lower = more diverse"),
    filter_skills: Optional[str] = typer.Option(None, "--filter-skills", help="Comma-separated skills (must have ALL)"),
    filter_min_exp: Optional[float] = typer.Option(None, "--filter-min-exp", help="Minimum experience"),
    filter_max_exp: Optional[float] = typer.Option(None, "--filter-max-exp", help="Maximum experience"),
    filter_edu: Optional[str] = typer.Option(None, "--filter-edu", help="Required education (bachelor/master/phd)"),
    filter_seniority: Optional[str] = typer.Option(None, "--filter-seniority", help="Required seniority (junior/mid/senior/lead/principal)"),
    no_expand: bool = typer.Option(False, "--no-expand", help="Disable query expansion with synonyms"),
    boolean: Optional[str] = typer.Option(None, "--boolean", help="Boolean expression over skills, e.g. 'python AND docker OR (java NOT kotlin)'"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable query caching"),
):
    """Hybrid search with all optional features."""
    if not query or query.isspace():
        console.print("[red]❌ Query cannot be empty[/red]")
        raise typer.Exit(code=1)
    if len(query) > config.MAX_QUERY_LENGTH:
        console.print("[red]❌ Query too long[/red]")
        raise typer.Exit(code=1)
    if not (0.0 <= alpha <= 1.0):
        console.print("[red]❌ alpha must be between 0 and 1[/red]")
        raise typer.Exit(code=1)
    if diversity is not None and not (0.0 <= diversity <= 1.0):
        console.print("[red]❌ diversity lambda must be between 0 and 1[/red]")
        raise typer.Exit(code=1)

    filter_skills_set = None
    if filter_skills:
        filter_skills_set = {s.strip() for s in filter_skills.split(',') if s.strip()}

    retriever = get_retriever()
    skill_priority = parse_priority_skills(priority_skills)

    start = time.perf_counter()
    try:
        results = retriever.hybrid_search(
            query,
            top_k=limit,
            alpha=alpha,
            min_experience=min_exp,
            skill_priority=skill_priority,
            use_rerank=not no_rerank,
            diversity_lambda=diversity,
            filter_skills=filter_skills_set,
            filter_min_exp=filter_min_exp,
            filter_max_exp=filter_max_exp,
            filter_edu=filter_edu,
            filter_seniority=filter_seniority,
            expand_query=not no_expand,
            boolean_query=boolean,
            use_cache=not no_cache,
        )
    except ValueError as e:
        console.print(f"[red]❌ Boolean query error: {e}[/red]")
        raise typer.Exit(code=1)
    elapsed = time.perf_counter() - start

    logger.info("search query='%s' results=%d time=%.3fs", query, len(results), elapsed)
    display_results(results, query, alpha)

# ======================================================================
# MATCH
# ======================================================================
@app.command()
def match(
    jd_file: Optional[Path] = typer.Option(None, "--file", "-f", help="Path to job description text file"),
    jd_text: Optional[str] = typer.Option(None, "--text", "-t", help="Job description text (if not using file)"),
    limit: int = typer.Option(10, "--limit", "-l", min=1, max=100),
    min_exp: Optional[float] = typer.Option(None, "--min-exp", help="Override minimum experience (hard filter)"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable cross-encoder reranking"),
    priority_skills: Optional[str] = typer.Option(None, "--priority-skills", help="Override: comma-separated skill:weight pairs"),
    diversity: Optional[float] = typer.Option(None, "--diversity", help="Enable diversity with lambda (0-1); lower = more diverse"),
    filter_skills: Optional[str] = typer.Option(None, "--filter-skills", help="Comma-separated skills (must have ALL)"),
    filter_min_exp: Optional[float] = typer.Option(None, "--filter-min-exp", help="Minimum experience"),
    filter_max_exp: Optional[float] = typer.Option(None, "--filter-max-exp", help="Maximum experience"),
    filter_edu: Optional[str] = typer.Option(None, "--filter-edu", help="Required education (bachelor/master/phd)"),
    filter_seniority: Optional[str] = typer.Option(None, "--filter-seniority", help="Required seniority (junior/mid/senior/lead/principal)"),
    no_expand: bool = typer.Option(False, "--no-expand", help="Disable query expansion with synonyms"),
    boolean: Optional[str] = typer.Option(None, "--boolean", help="Boolean expression over skills, e.g. 'python AND docker OR (java NOT kotlin)'"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable query caching"),
):
    """Match candidates to a job description with full feature set."""
    if jd_file and jd_text:
        console.print("[red]❌ Provide either --file or --text, not both[/red]")
        raise typer.Exit(code=1)
    if not jd_file and not jd_text:
        console.print("[red]❌ Must provide either --file or --text[/red]")
        raise typer.Exit(code=1)

    if jd_file:
        if not jd_file.exists():
            console.print(f"[red]❌ File not found: {jd_file}[/red]")
            raise typer.Exit(code=1)
        try:
            jd_text = jd_file.read_text(encoding="utf-8")
        except Exception as e:
            console.print(f"[red]❌ Failed to read file: {e}[/red]")
            raise typer.Exit(code=1)

    parsed = parse_job_description(jd_text)

    skill_priority = None
    if priority_skills:
        skill_priority = parse_priority_skills(priority_skills)
    else:
        skill_priority = {skill: 1.0 for skill in parsed['skills']} if parsed['skills'] else None

    effective_min_exp = min_exp if min_exp is not None else parsed['min_experience']

    filter_skills_set = None
    if filter_skills:
        filter_skills_set = {s.strip() for s in filter_skills.split(',') if s.strip()}

    retriever = get_retriever()
    start = time.perf_counter()
    try:
        results = retriever.hybrid_search(
            query=jd_text,
            top_k=limit,
            alpha=0.3,
            min_experience=effective_min_exp,
            desired_experience=parsed['min_experience'],
            query_skills=parsed['skills'],
            skill_priority=skill_priority,
            required_education=parsed['education'],
            required_seniority=None,
            use_rerank=not no_rerank,
            diversity_lambda=diversity,
            filter_skills=filter_skills_set,
            filter_min_exp=filter_min_exp,
            filter_max_exp=filter_max_exp,
            filter_edu=filter_edu,
            filter_seniority=filter_seniority,
            expand_query=not no_expand,
            boolean_query=boolean,
            use_cache=not no_cache,
        )
    except ValueError as e:
        console.print(f"[red]❌ Boolean query error: {e}[/red]")
        raise typer.Exit(code=1)
    elapsed = time.perf_counter() - start

    logger.info("match with JD results=%d time=%.3fs", len(results), elapsed)
    display_match_results(results, parsed)

# ======================================================================
# INGEST (parallel processing)
# ======================================================================
@app.command()
def ingest(
    path: str = typer.Argument(..., help="Folder containing CV PDFs"),
    max_workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
):
    """Ingest PDFs into the vector database (parallel processing)."""
    folder = Path(path)
    if not folder.exists():
        console.print(f"[red]❌ Path not found: {folder}[/red]")
        raise typer.Exit(code=1)

    pdfs = list(folder.glob("*.pdf"))
    if not pdfs:
        console.print("[red]❌ No PDFs found.[/red]")
        raise typer.Exit(code=1)

    console.print(f"📄 Found {len(pdfs)} PDF(s). Processing with {max_workers} workers...")

    # Function to process a single PDF
    def process_one(pdf_path: Path) -> tuple[str, bool, str]:
        try:
            data = extract_structured_data(str(pdf_path))
            index_cv(pdf_path.stem, data)
            return pdf_path.name, True, ""
        except Exception as e:
            logger.exception("Failed processing %s", pdf_path.name)
            return pdf_path.name, False, str(e)

    # Process in parallel with progress bar
    successful = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, pdf): pdf for pdf in pdfs}
        with tqdm(total=len(pdfs), desc="Ingesting", unit="pdf") as pbar:
            for future in concurrent.futures.as_completed(futures):
                name, ok, error = future.result()
                if ok:
                    console.print(f"✅ Indexed [bold]{name}[/bold]")
                    successful += 1
                else:
                    console.print(f"[red]❌ Failed: {name} — {error}[/red]")
                    failed += 1
                pbar.update(1)

    console.print(f"\n[bold]Ingestion complete: {successful} succeeded, {failed} failed.[/bold]")

    # Rebuild BM25 after ingestion
    try:
        get_retriever().build_bm25_index(force_rebuild=True)
    except Exception as e:
        logger.warning(f"BM25 rebuild skipped: {e}")

# ======================================================================
# STATUS
# ======================================================================
@app.command()
def status():
    """Show system status."""
    console.print("[bold]📊 System Status[/bold]")
    try:
        vector_count = vector_store.count()
    except Exception:
        vector_count = "ERROR"

    console.print(f"  • Raw CVs: {len(list(config.RAW_CVS_DIR.glob('*')))}")
    console.print(f"  • Processed CVs: {len(list(config.PROCESSED_DIR.glob('*.json')))}")
    console.print(f"  • Vector DB entries: {vector_count}")
    console.print(f"  • API Key: {'✅' if config.GOOGLE_API_KEY else '❌'}")
    console.print(f"  • BM25 index exists: {config.BM25_INDEX_PATH.exists()}")

# ======================================================================
# EMAIL (Enhanced with tracking, scheduling, template variants)
# ======================================================================
@app.command()
def email(
    query: str = typer.Argument(..., help="Search query to find candidates"),
    limit: int = typer.Option(3, "--limit", "-l", min=1, max=10),
    subject: str = typer.Option(config.EMAIL_SUBJECT, "--subject", help="Email subject line"),
    template_variant: Optional[str] = typer.Option(None, "--template-variant", help="Template variant name (e.g., 'followup' -> email_template_followup.txt)"),
    schedule: Optional[str] = typer.Option(None, "--schedule", help="Schedule email for future (ISO format, e.g., '2026-03-13T10:30:00')"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview emails without sending/scheduling"),
    no_cooldown: bool = typer.Option(False, "--no-cooldown", help="Ignore duplicate prevention"),
):
    """Search for top candidates and send/schedule emails with tracking."""
    # Determine template path
    if template_variant:
        template_path = config.EMAIL_TEMPLATE_PATH.with_stem(f"email_template_{template_variant}")
    else:
        template_path = config.EMAIL_TEMPLATE_PATH

    if not template_path.exists():
        console.print(f"[red]❌ Template file not found: {template_path}[/red]")
        raise typer.Exit(code=1)

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template_text = f.read()
    except Exception as e:
        console.print(f"[red]❌ Failed to read template: {e}[/red]")
        raise typer.Exit(code=1)

    # Parse schedule if provided
    scheduled_time = None
    if schedule:
        try:
            scheduled_time = datetime.fromisoformat(schedule)
            if scheduled_time <= datetime.now():
                console.print("[red]❌ Scheduled time must be in the future[/red]")
                raise typer.Exit(code=1)
        except ValueError:
            console.print("[red]❌ Invalid datetime format. Use ISO format (e.g., 2026-03-13T10:30:00)[/red]")
            raise typer.Exit(code=1)

    # Search for candidates
    retriever = get_retriever()
    results = retriever.hybrid_search(query, top_k=limit, alpha=config.HYBRID_ALPHA_DEFAULT)

    if not results:
        console.print("[yellow]No candidates found for this query.[/yellow]")
        return

    # Filter candidates with email
    candidates_with_email = []
    for r in results:
        meta = r.get("metadata", {})
        email_addr = meta.get("email", "").strip()
        if email_addr:
            candidates_with_email.append({
                "name": meta.get("name", "Candidate"),
                "email": email_addr,
                "id": r.get("id", ""),
                "score": r.get("score", 0.0),
                "metadata": meta
            })

    if not candidates_with_email:
        console.print("[yellow]No candidates have email addresses in their metadata.[/yellow]")
        return

    # Apply duplicate prevention (unless disabled)
    tracker = get_email_tracker()
    if not no_cooldown:
        filtered = []
        for cand in candidates_with_email:
            if tracker.was_emailed_recently(cand["email"], config.EMAIL_COOLDOWN_DAYS):
                console.print(f"[yellow]⏭️ Skipping {cand['name']} – emailed within last {config.EMAIL_COOLDOWN_DAYS} days[/yellow]")
            else:
                filtered.append(cand)
        candidates_with_email = filtered

    if not candidates_with_email:
        console.print("[yellow]No new candidates to email after duplicate check.[/yellow]")
        return

    # Show candidates
    console.print(f"\n[bold]Top {len(candidates_with_email)} candidates:[/bold]")
    for i, cand in enumerate(candidates_with_email, 1):
        console.print(f"{i}. {cand['name']} <{cand['email']}> (score: {cand['score']:.3f})")

    if dry_run:
        if scheduled_time:
            console.print(f"\n[yellow]Dry run – would schedule {len(candidates_with_email)} emails for {scheduled_time}.[/yellow]")
        else:
            console.print(f"\n[yellow]Dry run – would send {len(candidates_with_email)} emails now.[/yellow]")
        return

    # If scheduling, store in DB and exit
    if scheduled_time:
        for cand in candidates_with_email:
            tracker.schedule_email(
                candidate_id=cand["id"],
                candidate_email=cand["email"],
                candidate_name=cand["name"],
                subject=subject,
                template_variant=template_variant or "default",
                scheduled_time=scheduled_time,
            )
        console.print(f"[green]✅ Scheduled {len(candidates_with_email)} emails for {scheduled_time}[/green]")
        return

    # Otherwise, send now
    confirm = typer.confirm(f"\nSend emails to these {len(candidates_with_email)} candidates?")
    if not confirm:
        console.print("[yellow]Aborted.[/yellow]")
        return

    from app.email.sender import get_sender
    sender = get_sender(config.CREDENTIALS_PATH, config.TOKEN_PATH)

    try:
        sender.authenticate()
    except Exception as e:
        console.print(f"[red]❌ Gmail authentication failed: {e}[/red]")
        raise typer.Exit(code=1)

    success_count = 0
    for cand in candidates_with_email:
        body = template_text.replace("{name}", cand["name"])
        msg_id = sender.send_message(cand["email"], subject, body)
        if msg_id:
            tracker.log_sent(
                candidate_id=cand["id"],
                candidate_email=cand["email"],
                candidate_name=cand["name"],
                subject=subject,
                template_variant=template_variant or "default",
                message_id=msg_id,
                thread_id=None,
            )
            success_count += 1
            console.print(f"✅ Email sent to {cand['name']} ({cand['email']})")
        else:
            console.print(f"[red]❌ Failed to send to {cand['name']} ({cand['email']})[/red]")

    console.print(f"\n[bold]Emails sent: {success_count}/{len(candidates_with_email)}[/bold]")

# ======================================================================
# SEND SCHEDULED EMAILS
# ======================================================================
@app.command()
def send_scheduled(
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of scheduled emails to process"),
):
    """Process and send scheduled emails that are due."""
    tracker = get_email_tracker()
    pending = tracker.get_pending_scheduled(limit=limit)

    if not pending:
        console.print("[yellow]No pending scheduled emails found.[/yellow]")
        return

    console.print(f"[bold]Processing {len(pending)} scheduled emails...[/bold]")

    from app.email.sender import get_sender
    sender = get_sender(config.CREDENTIALS_PATH, config.TOKEN_PATH)
    try:
        sender.authenticate()
    except Exception as e:
        console.print(f"[red]❌ Gmail authentication failed: {e}[/red]")
        raise typer.Exit(code=1)

    success_count = 0
    for sched in pending:
        # Load appropriate template
        variant = sched["template_variant"]
        if variant == "default":
            template_path = config.EMAIL_TEMPLATE_PATH
        else:
            template_path = config.EMAIL_TEMPLATE_PATH.with_stem(f"email_template_{variant}")

        if not template_path.exists():
            console.print(f"[red]❌ Template file not found: {template_path} – skipping[/red]")
            tracker.mark_scheduled_failed(sched["id"])
            continue

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template_text = f.read()
        except Exception as e:
            console.print(f"[red]❌ Failed to read template: {e} – skipping[/red]")
            tracker.mark_scheduled_failed(sched["id"])
            continue

        body = template_text.replace("{name}", sched["candidate_name"])
        msg_id = sender.send_message(sched["candidate_email"], sched["subject"], body)
        if msg_id:
            tracker.mark_scheduled_sent(
                scheduled_id=sched["id"],
                message_id=msg_id,
                thread_id=None,
            )
            success_count += 1
            console.print(f"✅ Email sent to {sched['candidate_name']} ({sched['candidate_email']})")
        else:
            tracker.mark_scheduled_failed(sched["id"])
            console.print(f"[red]❌ Failed to send to {sched['candidate_name']}[/red]")

    console.print(f"\n[bold]Scheduled emails sent: {success_count}/{len(pending)}[/bold]")

# ======================================================================
# EMAIL STATISTICS
# ======================================================================
@app.command()
def email_stats():
    """Show email tracking statistics."""
    tracker = get_email_tracker()
    stats = tracker.get_stats()
    console.print("[bold]📧 Email Statistics[/bold]")
    console.print(f"  • Sent emails: {stats['sent']}")
    console.print(f"  • Scheduled pending: {stats['scheduled']}")

# ======================================================================
# INSIGHTS
# ======================================================================
@app.command()
def insights(
    candidate_id: Optional[str] = typer.Option(None, "--id", help="Candidate document ID from Chroma"),
    query: Optional[str] = typer.Option(None, "--query", help="Search query to select top candidate"),
    jd_file: Optional[Path] = typer.Option(None, "--jd-file", help="Job description file (for gap analysis)"),
    jd_text: Optional[str] = typer.Option(None, "--jd-text", help="Job description text (for gap analysis)"),
):
    """
    Generate AI insights for a candidate: summary, strengths, weaknesses, interview questions.
    Provide either a candidate ID or a search query (picks top result).
    Optionally provide a job description for targeted gap analysis.
    """
    if not candidate_id and not query:
        console.print("[red]❌ Must provide either --id or --query[/red]")
        raise typer.Exit(code=1)
    if candidate_id and query:
        console.print("[red]❌ Provide only one of --id or --query[/red]")
        raise typer.Exit(code=1)

    # Get job description if provided
    job_desc = None
    if jd_file and jd_text:
        console.print("[red]❌ Provide either --jd-file or --jd-text, not both[/red]")
        raise typer.Exit(code=1)
    if jd_file:
        if not jd_file.exists():
            console.print(f"[red]❌ File not found: {jd_file}[/red]")
            raise typer.Exit(code=1)
        job_desc = jd_file.read_text(encoding="utf-8")
    elif jd_text:
        job_desc = jd_text

    # Retrieve candidate data
    retriever = get_retriever()
    cv_data = None

    if candidate_id:
        # Fetch by ID
        try:
            doc = retriever.collection.get(ids=[candidate_id], include=["metadatas", "documents"])
            if doc["ids"]:
                cv_data = doc["metadatas"][0]
                cv_data["document"] = doc["documents"][0] if doc["documents"] else ""
            else:
                console.print(f"[red]❌ Candidate ID '{candidate_id}' not found.[/red]")
                raise typer.Exit(code=1)
        except Exception as e:
            console.print(f"[red]❌ Error fetching candidate: {e}[/red]")
            raise typer.Exit(code=1)
    else:
        # Search and take top result
        results = retriever.hybrid_search(query, top_k=1, alpha=0.5, use_rerank=True)
        if not results:
            console.print("[yellow]No candidates found for query.[/yellow]")
            raise typer.Exit(code=1)
        top = results[0]
        cv_data = top["metadata"]
        cv_data["document"] = top.get("document", "")

    from app.insights.generator import generate_insights

    with console.status("[bold green]Generating AI insights..."):
        insights = generate_insights(cv_data, job_desc)

    # Display
    console.print(Panel(f"[bold cyan]Candidate: {cv_data.get('name', 'Unknown')}[/bold cyan]", title="Insights"))

    if insights["summary"]:
        console.print(Panel(insights["summary"], title="Summary", border_style="green"))

    if insights["strengths"]:
        strengths = "\n".join(f"• {s}" for s in insights["strengths"])
        console.print(Panel(strengths, title="Strengths", border_style="blue"))

    if insights["weaknesses"]:
        weaknesses = "\n".join(f"• {w}" for w in insights["weaknesses"])
        console.print(Panel(weaknesses, title="Weaknesses / Gaps", border_style="yellow"))

    if insights["interview_questions"]:
        questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(insights["interview_questions"]))
        console.print(Panel(questions, title="Interview Questions", border_style="magenta"))

# ======================================================================
# EVALUATE
# ======================================================================
@app.command()
def evaluate(
    golden_path: Path = typer.Argument(..., help="Path to golden set JSON file"),
    k_values: str = typer.Option("1,3,5,10", help="Comma-separated K values for Precision/Recall"),
):
    """
    Evaluate retrieval performance using a golden dataset.
    The golden JSON should contain a list of queries with 'id', 'text', and 'relevant' (list of candidate IDs).
    """
    # Parse K values
    ks = [int(k.strip()) for k in k_values.split(",") if k.strip()]
    if not ks:
        console.print("[red]❌ No valid K values provided[/red]")
        raise typer.Exit(code=1)

    # Load golden set
    try:
        golden = GoldenSet(golden_path)
    except Exception as e:
        console.print(f"[red]❌ Failed to load golden set: {e}[/red]")
        raise typer.Exit(code=1)

    retriever = get_retriever()
    results = []

    with console.status("[bold green]Running evaluation..."):
        for q in golden.get_all_queries():
            qid = q["id"]
            text = q["text"]
            relevant = golden.get_relevant(qid)

            # Perform search (use hybrid with default settings)
            retrieved = retriever.hybrid_search(
                query=text,
                top_k=max(ks),  # fetch enough for the largest K
                use_rerank=True,
                expand_query=True,
            )
            retrieved_ids = [r["id"] for r in retrieved]

            results.append({
                "query_id": qid,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": list(relevant),
            })

    # Compute metrics
    precisions = {k: [] for k in ks}
    recalls = {k: [] for k in ks}
    rrs = []

    for res in results:
        retrieved = res["retrieved_ids"]
        relevant = set(res["relevant_ids"])
        for k in ks:
            precisions[k].append(metrics.precision_at_k(retrieved, relevant, k))
            recalls[k].append(metrics.recall_at_k(retrieved, relevant, k))
        rrs.append(metrics.reciprocal_rank(retrieved, relevant))

    # Aggregate
    mean_precisions = {k: sum(v)/len(v) for k, v in precisions.items()}
    mean_recalls = {k: sum(v)/len(v) for k, v in recalls.items()}
    mrr_value = sum(rrs) / len(rrs) if rrs else 0.0

    # Display results
    console.print("[bold green]✅ Evaluation Complete[/bold green]\n")
    console.print("[bold]Mean Reciprocal Rank (MRR):[/bold]", f"{mrr_value:.4f}\n")
    console.print("[bold]Precision@K & Recall@K:[/bold]")
    table = Table(title="Metrics")
    table.add_column("K", style="dim")
    table.add_column("Precision@K", justify="right")
    table.add_column("Recall@K", justify="right")
    for k in ks:
        table.add_row(str(k), f"{mean_precisions[k]:.4f}", f"{mean_recalls[k]:.4f}")
    console.print(table)

    # Optionally save detailed results
    output_path = Path("evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "mrr": mrr_value,
            "precisions": mean_precisions,
            "recalls": mean_recalls,
            "detailed": results,
        }, f, indent=2)
    console.print(f"\n[dim]Detailed results saved to {output_path}[/dim]")

# ======================================================================
# CLUSTER (Data Intelligence)
# ======================================================================
@app.command()
def cluster(
    force: bool = typer.Option(False, "--force", help="Force recompute clusters even if model exists")
):
    """
    Compute skill clusters for all candidates and update metadata.
    Uses K‑means on document embeddings to group similar candidates.
    Cluster labels are stored in the 'tech_cluster' metadata field.
    """
    try:
        from app.clustering.skill_cluster import run_clustering
    except ImportError as e:
        console.print("[red]❌ Clustering module not available. Install scikit-learn and ensure app.clustering exists.[/red]")
        raise typer.Exit(code=1)
    with console.status("[bold green]Clustering candidates..."):
        try:
            run_clustering(force_recompute=force)
            console.print("[green]✅ Clustering completed and metadata updated.[/green]")
        except Exception as e:
            console.print(f"[red]❌ Clustering failed: {e}[/red]")
            raise typer.Exit(code=1)

# ======================================================================
# ENTRYPOINT
# ======================================================================
if __name__ == "__main__":
    app()