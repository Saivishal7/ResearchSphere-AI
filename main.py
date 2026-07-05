# main.py
# Production Interactive CLI Entrypoint for ResearchSphere AI
#
# This file is the ONLY file modified to deliver a premium, hackathon-grade
# terminal experience. All backend logic (graph/, agents/, tools/, prompts/,
# config.py, state.py, build_graph.py, router.py, app.py) is untouched.
#
# The graph is always executed exactly as:
#   compiled_research_graph.invoke(initial_state)
#
# Everything below only changes HOW results are displayed.

import os
import sys
import time
import logging
import threading
import contextlib
import io
import traceback as _traceback
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.columns import Columns
from rich.rule import Rule
from rich.align import Align
from rich.text import Text
from rich.markdown import Markdown
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt

from config import validate_config
from graph.build_graph import compiled_research_graph
from graph.state import ResearchState


# =========================================================
# GLOBAL CONFIG
# =========================================================

DEBUG_MODE = os.environ.get("DEBUG_MODE", "False").strip().lower() in ("1", "true", "yes")

console = Console()

SESSION_ID = "cli-session-001"

SPINNER_MESSAGES = [
    "Initializing multi-agent graph...",
    "Routing your request...",
    "Searching faculty database...",
    "Analyzing research trends...",
    "Identifying research gaps...",
    "Finding potential collaborators...",
    "Building project recommendation...",
    "Synthesizing final response...",
]


# =========================================================
# NOISE SUPPRESSION (presentation only — never touches logic)
# =========================================================

if not DEBUG_MODE:
    logging.disable(logging.CRITICAL)


@contextlib.contextmanager
def suppress_backend_noise():
    """
    Suppresses stray stdout/stderr chatter (e.g. 'Initializing Chroma',
    'GeminiService initialized', raw JSON dumps) emitted by underlying
    libraries during a graph invocation. Fully disabled when DEBUG_MODE=True.
    """
    if DEBUG_MODE:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


# =========================================================
# SMALL DATA-EXTRACTION HELPERS
# (Defensive against minor variations in agent output shape;
#  never assumes or alters the underlying schema.)
# =========================================================

def _first(d, keys, default="--"):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _as_bullets(value):
    items = _as_list(value)
    return "\n".join(f"• {item}" for item in items) if items else "—"


def _fmt_percent(value):
    if value in (None, "", "N/A", "--"):
        return "--"
    try:
        num = float(value)
        if num <= 1:
            num *= 100
        return f"{num:.2f}%"
    except (ValueError, TypeError):
        return str(value)


def _is_empty(value) -> bool:
    """True for values that should never be shown to the user as-is."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in ("", "N/A", "None", "—", "-", "{}", "[]")
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _has_content(value) -> bool:
    return not _is_empty(value)


def classify_message(message: str):
    """
    Shared friendly-language classifier used both for raised exceptions and
    for raw backend-reported error strings. Never surfaces parsing/JSON/
    traceback language to the user.
    """
    message = (message or "").lower()
    if "quota" in message or "429" in message or "rate limit" in message:
        return "Gemini API quota exceeded.", "Retry later or switch API key."
    if "timeout" in message:
        return "The request timed out.", "Check your network connection and try again."
    if "api key" in message or "unauthorized" in message or "401" in message:
        return "Authentication with the AI provider failed.", "Verify your API key configuration."
    if "connection" in message:
        return "Unable to reach a required service.", "Check that all backend services are reachable."
    if "json" in message or "parse" in message or "decode" in message:
        return "The AI response could not be parsed correctly.", "This is usually temporary — please retry."
    return (
        "An unexpected issue occurred while generating this section.",
        "Please retry, or check back after a few minutes.",
    )


def _get_confidence(section):
    val = _first(section, ["confidence"], None)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _degradation_reason(section):
    msg = _first(section, ["error", "error_message", "failure_reason", "message"], None)
    if _has_content(msg):
        reason, _ = classify_message(str(msg))
        return reason
    status = _first(section, ["status"], None)
    confidence = _get_confidence(section)
    if status == "partial":
        return "AI analysis returned only partial results."
    if confidence is not None and confidence < 0.5:
        return "AI analysis confidence was too low to display reliable results."
    return "Gemini API quota exceeded."


def _is_degraded(section) -> bool:
    if not isinstance(section, dict):
        return False
    status = _first(section, ["status"], None)
    confidence = _get_confidence(section)
    if status == "partial":
        return True
    if confidence is not None and confidence < 0.5:
        return True
    if _has_content(_first(section, ["error", "error_message", "failure_reason"], None)):
        return True
    return False


def render_unavailable(section_title: str, reason: str):
    """Single, professional fallback panel for partial/low-confidence/failed sections."""
    body = (
        "AI analysis is temporarily unavailable.\n\n"
        f"[bold]Reason:[/bold] {reason}\n\n"
        "The backend executed successfully.\n"
        "Please retry after the quota resets."
    )
    console.print(Panel(body, title=section_title, border_style="yellow", padding=(1, 2)))


# =========================================================
# BANNER / MENU
# =========================================================

def render_banner():
    console.clear()
    title = Text("ResearchSphere AI", style="bold cyan", justify="center")
    subtitle = Text("Multi-Agent Research Intelligence Platform", style="grey62", justify="center")
    body = Text()
    body.append(title)
    body.append("\n")
    body.append(subtitle)
    console.print()
    console.print(Panel(Align.center(body), border_style="cyan", padding=(1, 4)))
    console.print()


def render_menu() -> None:
    menu_text = Text()
    options = [
        ("1", "Student Supervisor Recommendation"),
        ("2", "Faculty Profile Search"),
        ("3", "Research Trend Analysis"),
        ("4", "Research Gap Analysis"),
        ("5", "Collaboration Recommendation"),
        ("6", "Project Recommendation"),
        ("7", "End-to-End Graph Demonstration"),
        ("8", "Custom Natural Language Query"),
        ("9", "Backend Status"),
        ("0", "Exit"),
    ]
    for num, label in options:
        menu_text.append(f"  {num}  ", style="bold cyan")
        menu_text.append(f"{label}\n", style="white")

    console.print(Panel(menu_text, title="Main Menu", border_style="blue", padding=(1, 2)))


def safe_prompt(label: str) -> str:
    try:
        return Prompt.ask(f"[bold cyan]{label}[/bold cyan]").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Input interrupted.[/yellow]")
        return ""


# =========================================================
# RENDER: FACULTY SEARCH RESULTS
# =========================================================

def render_faculty_table(faculty_results):
    if isinstance(faculty_results, dict):
        records = faculty_results.get("results") or faculty_results.get("faculty") or []
        top = faculty_results.get("top_recommendation")
    else:
        records = faculty_results if isinstance(faculty_results, list) else []
        top = None

    if isinstance(faculty_results, dict) and _is_degraded(faculty_results):
        render_unavailable("Student Supervisor Recommendation", _degradation_reason(faculty_results))
        return None

    if not records:
        return None

    # CHANGE 1: Only show the "Match %" column when at least one record
    # actually carries a usable similarity/match score. If none do, the
    # column is omitted entirely rather than rendering "N/A" for every row.
    raw_scores = [_first(rec, ["match_score", "match_percentage", "score"], None) for rec in records]
    show_match_column = any(_has_content(s) for s in raw_scores)

    table = Table(title="Faculty Supervisor Matches", border_style="cyan", header_style="bold cyan")
    table.add_column("Rank", justify="center", style="white", width=6)
    table.add_column("Faculty", style="bold white")
    table.add_column("Department", style="grey78")
    if show_match_column:
        table.add_column("Match %", justify="right", style="green")
    table.add_column("Research Interests", style="white")

    for idx, rec in enumerate(records, start=1):
        name = _first(rec, ["name", "faculty_name", "full_name"])
        dept = _first(rec, ["department", "dept"])
        interests = _first(rec, ["research_interests", "interests", "keywords"])
        if isinstance(interests, list):
            interests = ", ".join(interests)

        row_cells = [str(_first(rec, ["rank"], str(idx))), str(name), str(dept)]
        if show_match_column:
            row_cells.append(_fmt_percent(_first(rec, ["match_score", "match_percentage", "score"], None)))
        row_cells.append(str(interests))
        table.add_row(*row_cells)

    console.print(table)

    if top is None and records:
        top = records[0]

    if isinstance(top, dict):
        interests = _first(top, ["research_interests", "interests", "keywords"], None)
        if isinstance(interests, list):
            interests = ", ".join(interests)
        fields = [
            ("Name", _first(top, ["name", "faculty_name", "full_name"], None)),
            ("Department", _first(top, ["department", "dept"], None)),
            ("Research Interests", interests),
            ("Current Availability", _first(top, ["availability", "current_availability"], None)),
        ]
        lines = [f"[bold]{label}:[/bold] {value}" for label, value in fields if _has_content(value)]
        if lines:
            console.print(Panel("\n".join(lines), title="Top Recommendation", border_style="green", padding=(1, 2)))

    # Return the displayed records so the caller can offer follow-up
    # faculty actions (view details / generate email) against this exact,
    # already-rendered list.
    return records


# =========================================================
# CHANGE 2: FACULTY ACTIONS — VIEW DETAILS / EMAIL GENERATION
# (Presentation-only. No SMTP/Gmail/Outlook integration. No emails are
#  ever sent. This only formats and displays text in the terminal.)
# =========================================================

def _get_faculty_email(rec: dict) -> str:
    """Uses the backend-provided email if present; otherwise generates a
    plausible institutional placeholder. Never invents a real address."""
    email = _first(rec, ["email", "contact_email", "faculty_email"], None)
    if _has_content(email):
        return str(email)

    name = str(_first(rec, ["name", "faculty_name", "full_name"], "")) or ""
    cleaned = name.replace("Dr.", "").replace("Prof.", "").replace("Professor", "").strip()
    parts = [p.strip(".,") for p in cleaned.split() if p.strip(".,")]

    if len(parts) >= 2:
        first, last = parts[0].lower(), parts[-1].lower()
    elif len(parts) == 1:
        first, last = parts[0].lower(), "faculty"
    else:
        first, last = "firstname", "lastname"

    return f"{first}.{last}@university.edu"


def _get_faculty_by_rank(records: list, rank_str: str):
    try:
        idx = int(str(rank_str).strip())
    except (ValueError, TypeError):
        return None
    if idx < 1 or idx > len(records):
        return None
    return records[idx - 1]


def _generate_professional_email(faculty_name: str, interests: str, topic: str, student_name: str = "Student Name") -> str:
    interests_line = interests if _has_content(interests) else "your research"
    subject_topic = topic if _has_content(topic) else "Your Research Area"
    return (
        f"Subject: Request for Research Supervision in {subject_topic}\n\n"
        f"Dear {faculty_name},\n\n"
        f"I hope you are doing well. My name is {student_name}, and I am reaching out "
        f"to express my interest in pursuing research under your guidance.\n\n"
        f"I recently explored your work on {interests_line}, and I am particularly "
        f"drawn to the possibility of exploring {subject_topic} in that context.\n\n"
        f"I would be grateful for the opportunity to share more about my background "
        f"and to learn about any ongoing opportunities in your lab.\n\n"
        f"Thank you for your time and consideration.\n\n"
        f"Regards,\n{student_name}"
    )


def _generate_meeting_request(faculty_name: str, topic: str, student_name: str = "Student Name") -> str:
    subject_topic = topic if _has_content(topic) else "Your Research Area"
    return (
        f"Subject: Meeting Request — Discussion on {subject_topic}\n\n"
        f"Dear {faculty_name},\n\n"
        f"I hope this message finds you well. I am very interested in {subject_topic} "
        f"and would appreciate the opportunity to briefly meet with you to discuss "
        f"potential research directions and supervision possibilities.\n\n"
        f"Would you be available for a short meeting sometime in the coming week? "
        f"I am happy to work around your schedule.\n\n"
        f"Thank you for considering my request.\n\n"
        f"Regards,\n{student_name}"
    )


def render_faculty_actions_menu():
    actions_text = Text()
    actions_text.append("1  ", style="bold cyan")
    actions_text.append("View Faculty Details\n", style="white")
    actions_text.append("2  ", style="bold cyan")
    actions_text.append("Generate Professional Email\n", style="white")
    actions_text.append("3  ", style="bold cyan")
    actions_text.append("Generate Meeting Request\n", style="white")
    actions_text.append("4  ", style="bold cyan")
    actions_text.append("Return to Main Menu\n", style="white")
    console.print(Panel(actions_text, title="Actions", border_style="blue", padding=(1, 2)))


def handle_faculty_actions(records: list, topic: str = None):
    """Presentation-only follow-up menu shown after the faculty table.
    Never sends anything — only displays generated text in the terminal."""
    if not records:
        return

    while True:
        console.print()
        render_faculty_actions_menu()
        action_choice = safe_prompt("Select an action")

        if action_choice in ("4", "", None):
            return

        if action_choice not in ("1", "2", "3"):
            console.print("[yellow]Invalid selection. Please choose a valid action.[/yellow]")
            continue

        rank_str = safe_prompt(f"Enter Faculty Rank (1-{len(records)})")
        rec = _get_faculty_by_rank(records, rank_str)
        if rec is None:
            console.print("[yellow]Invalid faculty rank.[/yellow]")
            continue

        name = str(_first(rec, ["name", "faculty_name", "full_name"], "the faculty member"))
        dept = _first(rec, ["department", "dept"], None)
        interests = _first(rec, ["research_interests", "interests", "keywords"], None)
        if isinstance(interests, list):
            interests = ", ".join(interests)
        availability = _first(rec, ["availability", "current_availability"], None)
        email = _get_faculty_email(rec)

        if action_choice == "1":
            fields = [
                ("Name", name),
                ("Department", dept),
                ("Research Interests", interests),
                ("Availability", availability),
                ("Email", email),
            ]
            lines = [f"[bold]{label}:[/bold] {value}" for label, value in fields if _has_content(value)]
            console.print(Panel("\n".join(lines), title="Faculty Details", border_style="cyan", padding=(1, 2)))

        elif action_choice == "2":
            body = _generate_professional_email(name, interests, topic)
            console.print(Panel(body, title=f"Email Draft  —  To: {email}", border_style="green", padding=(1, 2)))

        elif action_choice == "3":
            body = _generate_meeting_request(name, topic)
            console.print(Panel(body, title=f"Email Draft  —  To: {email}", border_style="green", padding=(1, 2)))


# =========================================================
# RENDER: RESEARCH TRENDS
# =========================================================

def render_trends(research_trends):
    if not isinstance(research_trends, dict):
        research_trends = {}

    if _is_degraded(research_trends):
        render_unavailable("Research Trend Analysis", _degradation_reason(research_trends))
        return

    exec_summary = _first(research_trends, ["executive_summary", "summary"], None)
    topics = _first(research_trends, ["top_emerging_topics", "emerging_topics", "topics"], None)
    challenges = _first(research_trends, ["current_challenges", "challenges"], None)
    opportunities = _first(research_trends, ["future_opportunities", "opportunities"], None)

    if _has_content(exec_summary):
        console.print(Panel(Markdown(str(exec_summary)), title="Executive Summary", border_style="cyan"))
    if _has_content(topics):
        console.print(Panel(_as_bullets(topics), title="Top Emerging Topics", border_style="blue"))
    if _has_content(challenges):
        console.print(Panel(_as_bullets(challenges), title="Current Challenges", border_style="grey62"))
    if _has_content(opportunities):
        console.print(Panel(_as_bullets(opportunities), title="Future Opportunities", border_style="green"))


# =========================================================
# RENDER: RESEARCH GAPS
# =========================================================

def render_gaps(research_gaps):
    if not isinstance(research_gaps, dict):
        research_gaps = {}

    if _is_degraded(research_gaps):
        render_unavailable("Research Gap Analysis", _degradation_reason(research_gaps))
        return

    gaps = _first(research_gaps, ["research_gaps", "gaps"], None)
    open_problems = _first(research_gaps, ["open_problems"], None)
    future_research = _first(research_gaps, ["future_research"], None)
    thesis_topics = _first(research_gaps, ["suggested_thesis_topics", "thesis_topics"], None)

    if _has_content(gaps):
        console.print(Panel(_as_bullets(gaps), title="Research Gaps", border_style="cyan"))
    if _has_content(open_problems):
        console.print(Panel(_as_bullets(open_problems), title="Open Problems", border_style="blue"))
    if _has_content(future_research):
        console.print(Panel(_as_bullets(future_research), title="Future Research", border_style="green"))
    if _has_content(thesis_topics):
        console.print(Panel(_as_bullets(thesis_topics), title="Suggested Thesis Topics", border_style="grey62"))


# =========================================================
# RENDER: COLLABORATION SUGGESTIONS
# =========================================================

def render_collaboration(collaboration_suggestions):
    if isinstance(collaboration_suggestions, dict) and _is_degraded(collaboration_suggestions):
        render_unavailable("Collaboration Recommendation", _degradation_reason(collaboration_suggestions))
        return

    records = collaboration_suggestions
    if isinstance(records, dict):
        records = records.get("results") or records.get("collaborators") or []
    if not isinstance(records, list):
        records = []

    if not records:
        return

    table = Table(title="Recommended Collaborators", border_style="cyan", header_style="bold cyan")
    table.add_column("Professor", style="bold white")
    table.add_column("Department", style="grey78")
    table.add_column("Expertise", style="white")
    table.add_column("Why Recommended", style="green")

    for rec in records:
        name = _first(rec, ["professor", "name", "faculty_name"])
        dept = _first(rec, ["department", "dept"])
        expertise = _first(rec, ["expertise", "research_interests"])
        if isinstance(expertise, list):
            expertise = ", ".join(expertise)
        reason = _first(rec, ["why_recommended", "reason", "justification"])
        table.add_row(str(name), str(dept), str(expertise), str(reason))

    console.print(table)


# =========================================================
# RENDER: PROJECT RECOMMENDATIONS
# =========================================================

def render_projects(project_recommendations):
    if isinstance(project_recommendations, dict) and _is_degraded(project_recommendations):
        render_unavailable("Research Project Recommendation", _degradation_reason(project_recommendations))
        return

    records = project_recommendations
    if isinstance(records, dict) and "projects" in records:
        records = records["projects"]
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        records = []

    if not records:
        return

    for proj in records:
        title = _first(proj, ["project_title", "title"], "Project Recommendation")
        tech = _first(proj, ["required_technologies", "technologies"], None)
        if isinstance(tech, list):
            tech = ", ".join(tech)

        fields = [
            ("Objective", _first(proj, ["objective"], None)),
            ("Problem Statement", _first(proj, ["problem_statement"], None)),
            ("Required Technologies", tech),
            ("Recommended Faculty", _first(proj, ["recommended_faculty", "faculty"], None)),
            ("Difficulty", _first(proj, ["difficulty"], None)),
            ("Duration", _first(proj, ["expected_duration", "duration"], None)),
        ]
        lines = [f"[bold]{label}:[/bold] {value}" for label, value in fields if _has_content(value)]
        if lines:
            console.print(Panel("\n\n".join(lines), title=str(title), border_style="cyan", padding=(1, 2)))


# =========================================================
# RENDER: APPROVAL QUEUE / SYNTHESIS
# =========================================================

def render_approval(pending_action):
    body = "The system has paused a high-consequence action awaiting human consent."
    if isinstance(pending_action, dict):
        lines = [f"[bold]{k.replace('_', ' ').title()}:[/bold] {v}" for k, v in pending_action.items()]
        body += "\n\n" + "\n".join(lines)
    console.print(Panel(body, title="Approval Required", border_style="yellow", padding=(1, 2)))


def render_synthesis(retrieved_context):
    console.print(Panel(Markdown(str(retrieved_context)), title="Synthesis & Recommendations", border_style="green"))


# =========================================================
# RENDER: PIPELINE (Option 7 only)
# =========================================================

def infer_active_nodes(final_state: dict):
    nodes = [("Router", True)]  # Router always executes
    nodes.append(("Student Agent", bool(final_state.get("student_profile"))))
    nodes.append(("Professor Agent", bool(final_state.get("professor_profile"))))
    nodes.append(("Faculty Retrieval", bool(final_state.get("faculty_results"))))
    nodes.append(("Trend Agent", bool(final_state.get("research_trends"))))
    nodes.append(("Gap Analysis Agent", bool(final_state.get("research_gaps"))))
    nodes.append(("Collaboration Agent", bool(final_state.get("collaboration_suggestions"))))
    nodes.append(("Project Recommendation Agent", bool(final_state.get("project_recommendations"))))
    nodes.append(("Confirmation Agent", bool(final_state.get("approval_required"))))
    nodes.append(("Completed", True))
    return nodes


def render_pipeline(final_state: dict):
    nodes = infer_active_nodes(final_state)

    tree = Tree("[bold cyan]Graph Execution Pipeline[/bold cyan]")
    current = tree
    for name, active in nodes:
        marker = "[green]✓[/green]" if active else "[grey50]—[/grey50]"
        style = "bold white" if active else "grey50"
        current = current.add(f"{marker} [{style}]{name}[/{style}]")

    console.print(Panel(tree, border_style="blue", padding=(1, 2)))

    # Short animated pass over the active nodes for visual flair.
    active_nodes = [n for n, active in nodes if active]
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Replaying pipeline...", total=len(active_nodes))
        for name in active_nodes:
            progress.update(task, description=f"{name}")
            time.sleep(0.25)
            progress.advance(task)


# =========================================================
# RENDER: FRIENDLY ERROR
# =========================================================

def classify_error(exc: Exception):
    return classify_message(str(exc))


def render_error(exc: Exception):
    reason, suggestion = classify_error(exc)
    body = (
        "[bold red]Unable to complete request.[/bold red]\n\n"
        f"[bold]Reason:[/bold] {reason}\n"
        f"[bold]Suggestion:[/bold] {suggestion}"
    )
    console.print(Panel(body, title="Execution Error", border_style="red", padding=(1, 2)))
    if DEBUG_MODE:
        console.print(Panel(_traceback.format_exc(), title="Debug Traceback", border_style="grey50"))


# =========================================================
# GRAPH INVOCATION WITH LIVE STATUS SPINNER
# =========================================================

def invoke_graph_with_status(initial_state):
    """
    Invokes compiled_research_graph.invoke(initial_state) exactly as required,
    on a worker thread, while displaying a rotating status spinner on the
    main thread. This changes ONLY the presentation layer, never the
    invocation itself or its inputs/outputs.
    """
    result_holder = {}
    error_holder = {}

    def worker():
        try:
            with suppress_backend_noise():
                result_holder["state"] = compiled_research_graph.invoke(initial_state)
        except Exception as exc:  # noqa: BLE001
            error_holder["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)

    with console.status(f"[bold cyan]{SPINNER_MESSAGES[0]}", spinner="dots") as status:
        thread.start()
        i = 0
        while thread.is_alive():
            time.sleep(0.9)
            i = (i + 1) % len(SPINNER_MESSAGES)
            status.update(f"[bold cyan]{SPINNER_MESSAGES[i]}")
        thread.join()

    if "error" in error_holder:
        raise error_holder["error"]
    return result_holder.get("state", {})


# =========================================================
# CORE EXECUTION (UNCHANGED GRAPH SEMANTICS)
# =========================================================

def run_query(
    query: str,
    show_pipeline: bool = False,
    display_label: str = None,
    display_value: str = None,
) -> None:
    """
    Initializes the LangGraph state, executes the query through the compiled
    graph, and renders a polished, structured presentation of the final
    research state. Graph invocation and state schema are unchanged.

    display_label / display_value let callers show a clean, human label
    (e.g. "Research Topic" -> "Machine Learning") instead of the generated
    internal prompt that is actually sent through the graph.
    """
    console.print(Rule("[bold cyan]RESEARCHSPHERE AI — GRAPH EXECUTION[/bold cyan]"))
    if display_label and display_value:
        console.print(f"[bold]{display_label}:[/bold] {display_value}")
    else:
        console.print(f"[bold]Query:[/bold] {query}")
    console.print()

    # 1. Initialize complete ResearchState with defaults (unchanged schema)
    initial_state: ResearchState = {
        "current_query": query,
        "user_role": "unknown",
        "intent": "general_query",
        "conversation_history": [],
        "student_profile": None,
        "professor_profile": None,
        "faculty_results": None,
        "selected_faculty": None,
        "research_trends": None,
        "research_gaps": None,
        "collaboration_suggestions": None,
        "project_recommendations": None,
        "pending_action": None,
        "approval_required": False,
        "approval_status": "pending",
        "retrieved_context": None,
        "tool_output": None,
        "error": None,
        "retry_count": 0,
        "session_id": SESSION_ID,
    }

    start_time = time.time()
    had_error = False
    final_state = {}

    try:
        # 2. Invoke the compiled LangGraph workflow (unchanged call)
        final_state = invoke_graph_with_status(initial_state)
        elapsed = time.time() - start_time

        backend_error = final_state.get("error")
        if backend_error:
            had_error = True

        console.print("[green]✓[/green] Graph Executed")
        console.print("[green]✓[/green] Agents Finished")
        console.print("[green]✓[/green] Results Generated")
        console.print()

        if backend_error:
            reason, _ = classify_message(str(backend_error))
            render_unavailable("Research Analysis", reason)

        if show_pipeline:
            render_pipeline(final_state)

        # 3. Render structured outputs
        if final_state.get("faculty_results"):
            faculty_records = render_faculty_table(final_state["faculty_results"])
            # CHANGE 2: offer the follow-up actions menu (view details /
            # generate email text) against the faculty rows just displayed.
            if faculty_records:
                handle_faculty_actions(faculty_records, topic=display_value or query)

        if final_state.get("research_trends"):
            render_trends(final_state["research_trends"])

        if final_state.get("research_gaps"):
            render_gaps(final_state["research_gaps"])

        if final_state.get("collaboration_suggestions"):
            render_collaboration(final_state["collaboration_suggestions"])

        if final_state.get("project_recommendations"):
            render_projects(final_state["project_recommendations"])

        if final_state.get("approval_required"):
            render_approval(final_state.get("pending_action"))

        if final_state.get("retrieved_context"):
            render_synthesis(final_state["retrieved_context"])

    except Exception as exc:  # noqa: BLE001
        had_error = True
        render_error(exc)

    console.print(Rule(style="grey50"))
    console.print()


# =========================================================
# BACKEND STATUS DASHBOARD (Option 9 — read-only)
# =========================================================

def _indicator(ok: bool, online_label="Online", offline_label="Offline"):
    return f"[green]●[/green] {online_label}" if ok else f"[red]●[/red] {offline_label}"


def chromadb_available():
    try:
        import chromadb  # noqa: F401
        return True
    except Exception:
        return False


def render_status():
    is_valid, warnings = validate_config()

    # Gemini
    gemini_key_present = any(
        os.environ.get(var) for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")
    )
    try:
        import google.generativeai  # noqa: F401
        gemini_importable = True
    except Exception:
        gemini_importable = False
    gemini_status = _indicator(gemini_importable and gemini_key_present, "Configured", "Not Configured")

    # ChromaDB
    chroma_status = _indicator(chromadb_available(), "Online", "Offline")

    # LangGraph / compiled graph
    langgraph_status = _indicator(compiled_research_graph is not None, "Online", "Offline")

    # Faculty database (best-effort filesystem/config check, read-only)
    faculty_db_hint = getattr(sys.modules.get("config"), "CHROMA_DB_PATH", None)
    faculty_db_ok = bool(faculty_db_hint and os.path.exists(faculty_db_hint)) or chromadb_available()
    faculty_status = _indicator(faculty_db_ok, "Configured", "Not Configured")

    # Prompt templates
    prompts_ok = os.path.isdir("prompts")
    prompts_status = _indicator(prompts_ok, "Configured", "Not Configured")

    # Environment
    env_status = _indicator(is_valid, "Configured", "Not Configured")

    table = Table(title="Backend Status Dashboard", border_style="cyan", header_style="bold cyan")
    table.add_column("Component", style="bold white")
    table.add_column("Status")

    table.add_row("Gemini", gemini_status)
    table.add_row("ChromaDB", chroma_status)
    table.add_row("LangGraph", langgraph_status)
    table.add_row("Faculty Database", faculty_status)
    table.add_row("Prompt Templates", prompts_status)
    table.add_row("Environment", env_status)

    console.print(table)

    if warnings:
        warning_text = "\n".join(f"• {w}" for w in warnings)
        console.print(Panel(warning_text, title="Configuration Warnings", border_style="yellow"))

    info_columns = Columns(
        [
            Panel(f"Session ID\n[bold]{SESSION_ID}[/bold]", border_style="blue"),
            Panel(f"Debug Mode\n[bold]{'Enabled' if DEBUG_MODE else 'Disabled'}[/bold]", border_style="blue"),
            Panel(f"Timestamp\n[bold]{datetime.now().isoformat(timespec='seconds')}[/bold]", border_style="blue"),
        ]
    )
    console.print(info_columns)


# =========================================================
# MENU OPTION HANDLERS
# =========================================================

def handle_student_supervisor():
    topic = safe_prompt("Enter a research topic")
    if not topic:
        console.print("[yellow]Topic cannot be empty.[/yellow]")
        return
    run_query(f"Find faculty working on {topic}", display_label="Research Topic", display_value=topic)


def handle_faculty_search():
    keyword = safe_prompt("Enter a faculty name or keyword")
    if not keyword:
        console.print("[yellow]Keyword cannot be empty.[/yellow]")
        return
    run_query(f"Show faculty profile for {keyword}", display_label="Faculty Keyword", display_value=keyword)


def handle_trend_analysis():
    topic = safe_prompt("Enter a research topic")
    if not topic:
        console.print("[yellow]Topic cannot be empty.[/yellow]")
        return
    run_query(f"Research trends in {topic}", display_label="Research Topic", display_value=topic)


def handle_gap_analysis():
    topic = safe_prompt("Enter a research topic")
    if not topic:
        console.print("[yellow]Topic cannot be empty.[/yellow]")
        return
    run_query(f"Research gaps in {topic}", display_label="Research Topic", display_value=topic)


def handle_collaboration():
    topic = safe_prompt("Enter a research topic")
    if not topic:
        console.print("[yellow]Topic cannot be empty.[/yellow]")
        return
    run_query(f"Suggest collaborators for {topic}", display_label="Research Topic", display_value=topic)


def handle_project_recommendation():
    topic = safe_prompt("Enter a research topic")
    if not topic:
        console.print("[yellow]Topic cannot be empty.[/yellow]")
        return
    run_query(f"Recommend a research project on {topic}", display_label="Research Topic", display_value=topic)


def handle_end_to_end_demo():
    topic = safe_prompt("Enter a research topic for the full demonstration")
    if not topic:
        console.print("[yellow]Topic cannot be empty.[/yellow]")
        return
    comprehensive_query = (
        f"I am a student and a professor interested in {topic}. "
        f"Find faculty working on {topic}, analyze current research trends in {topic}, "
        f"identify research gaps in {topic}, suggest potential collaborators for {topic}, "
        f"and recommend a research project on {topic}."
    )
    run_query(comprehensive_query, show_pipeline=True, display_label="Research Topic", display_value=topic)


def handle_custom_query():
    query = safe_prompt("Enter your natural language query")
    if not query:
        console.print("[yellow]Query cannot be empty.[/yellow]")
        return
    run_query(query)


def handle_system_status():
    render_status()


# =========================================================
# MAIN INTERACTIVE LOOP
# =========================================================

MENU_HANDLERS = {
    "1": handle_student_supervisor,
    "2": handle_faculty_search,
    "3": handle_trend_analysis,
    "4": handle_gap_analysis,
    "5": handle_collaboration,
    "6": handle_project_recommendation,
    "7": handle_end_to_end_demo,
    "8": handle_custom_query,
    "9": handle_system_status,
}


def interactive_loop():
    while True:
        render_menu()
        choice = safe_prompt("Select an option")

        if choice == "0":
            console.print(Panel("Exiting ResearchSphere AI. Goodbye!", border_style="cyan"))
            break

        handler = MENU_HANDLERS.get(choice)
        if handler is None:
            console.print("[yellow]Invalid selection. Please choose a valid menu option.[/yellow]\n")
            continue

        try:
            handler()
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation interrupted. Returning to main menu.[/yellow]")
        except Exception as exc:  # noqa: BLE001
            render_error(exc)

        console.print("\n[bold green]Analysis Complete.[/bold green]")
        console.print("[grey62]Returning to Main Menu...[/grey62]\n")
        time.sleep(2)


def main():
    # 1. Validate environment configuration (unchanged backend call)
    is_valid, warnings = validate_config()
    if not is_valid and DEBUG_MODE:
        console.print("[yellow][WARNING] Configuration Warnings Found:[/yellow]")
        for warning in warnings:
            console.print(f"  - {warning}")

    # 2. If CLI arguments were passed, run a single query non-interactively
    #    (preserves original script-style backward compatibility)
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_query(query)
        return

    # 3. Otherwise, launch the interactive hackathon-demo CLI
    render_banner()
    try:
        interactive_loop()
    except KeyboardInterrupt:
        console.print("\n[cyan]Exiting ResearchSphere AI. Goodbye![/cyan]\n")
    except Exception as exc:  # noqa: BLE001
        render_error(exc)


if __name__ == "__main__":
    main()