import json
from datetime import datetime
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from grader import evaluate_intern_submission

DATABASE_URL = "sqlite:///./tracker.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Intern(Base):
    __tablename__ = "interns"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    domain = Column(String, nullable=False)
    custom_tool = Column(String, nullable=False)
    buddy_pair = Column(String, nullable=False)


class DailyTask(Base):
    __tablename__ = "daily_tasks"
    id = Column(Integer, primary_key=True, index=True)
    day_number = Column(Integer, unique=True, nullable=False)
    title = Column(String, nullable=False)
    rubric = Column(Text, nullable=False)


class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True, index=True)
    intern_id = Column(Integer, ForeignKey("interns.id"), nullable=False)
    day_number = Column(Integer, nullable=False)
    repo_url = Column(String, nullable=False)
    notes = Column(Text, default="")
    passed = Column(Boolean, default=False)
    score = Column(Integer, default=0)
    missing_json = Column(Text, default="[]")
    warnings_json = Column(Text, default="[]")
    feedback = Column(Text, default="")
    submitted_at = Column(DateTime, default=datetime.utcnow)

    intern = relationship("Intern")


Base.metadata.create_all(bind=engine)


def seed_database():
    db = SessionLocal()
    try:
        if db.query(Intern).count() == 0:
            interns = [
                Intern(name="Shumyze Najam", domain="E-Commerce Support Bot", custom_tool="get_order_status(order_id)", buddy_pair="Pair A"),
                Intern(name="Alina Khalid", domain="IT Helpdesk Assistant", custom_tool="create_ticket(issue, priority)", buddy_pair="Pair A"),
                Intern(name="Abdul Razzaq", domain="University Course Advisor", custom_tool="check_prerequisites(code)", buddy_pair="Pair B"),
                Intern(name="Sidra", domain="Restaurant Concierge Bot", custom_tool="check_table(date, guests)", buddy_pair="Pair B"),
                Intern(name="Aiman", domain="Fitness & Gym Coach Bot", custom_tool="book_session(trainer, day)", buddy_pair="Pair C"),
                Intern(name="Mahnoor", domain="Travel & Flight Assistant", custom_tool="check_flight(flight_no)", buddy_pair="Pair C"),
            ]
            db.add_all(interns)

        if db.query(DailyTask).count() == 0:
            tasks = [
                DailyTask(
                    day_number=1,
                    title="Repo Setup & Basic Domain LLM Script",
                    rubric=(
                        "1. Must have .gitignore (ignoring .env and venv/), requirements.txt, and .env.example.\n"
                        "2. Must contain a Python script calling OpenAI/Groq or Gemini API with a system prompt tailored to their assigned domain.\n"
                        "3. Must NOT hardcode any API keys."
                    ),
                ),
                DailyTask(
                    day_number=2,
                    title="FastAPI WebSocket & Token Streaming",
                    rubric=(
                        "1. Must have a FastAPI application with a @app.websocket('/ws/chat') endpoint.\n"
                        "2. Must maintain multi-turn conversation history in a messages list per session.\n"
                        "3. Must use stream=True (or async streaming) to send chunks incrementally over the WebSocket.\n"
                        "4. Should include a test_client.py or client test script."
                    ),
                ),
                DailyTask(
                    day_number=3,
                    title="Function Calling (1 Custom Tool)",
                    rubric=(
                        "1. Must implement their assigned Python tool function with mock data.\n"
                        "2. Must define a valid tool/function JSON schema passed to the LLM.\n"
                        "3. Must handle tool_calls in the WebSocket loop, execute the Python function, append the tool result, and generate the final answer."
                    ),
                ),
                DailyTask(
                    day_number=4,
                    title="Mini Knowledge Base (RAG)",
                    rubric=(
                        "1. Must include a domain FAQ file (e.g., faqs.json or CSV) with at least 10-15 Q&A pairs.\n"
                        "2. Must embed and index the FAQs into a vector store (ChromaDB, FAISS, or pgvector).\n"
                        "3. Must expose a search_faq tool or retrieval step so the bot answers domain policy questions from the dataset."
                    ),
                ),
                DailyTask(
                    day_number=5,
                    title="Chat UI & Final README",
                    rubric=(
                        "1. Must include a working frontend UI (Streamlit app or index.html with WebSocket JS) connected to the backend.\n"
                        "2. Must have a complete README.md explaining how to install dependencies, configure .env, and run the bot.\n"
                        "3. All previous features (streaming, memory, tool calling, RAG) must remain intact."
                    ),
                ),
            ]
            db.add_all(tasks)

        db.commit()
    finally:
        db.close()


seed_database()

app = FastAPI(title="Zigron Intern AI Progress Tracker")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    db = SessionLocal()
    try:
        interns = db.query(Intern).all()
        tasks = db.query(DailyTask).order_by(DailyTask.day_number).all()
        submissions = db.query(Submission).order_by(Submission.submitted_at.desc()).all()

        matrix = {i.id: {} for i in interns}
        for sub in submissions:
            if sub.day_number not in matrix[sub.intern_id]:
                matrix[sub.intern_id][sub.day_number] = {
                    "passed": sub.passed,
                    "score": sub.score,
                    "repo_url": sub.repo_url,
                    "feedback": sub.feedback,
                    "missing": json.loads(sub.missing_json or "[]"),
                    "warnings": json.loads(sub.warnings_json or "[]"),
                    "submitted_at": sub.submitted_at.strftime("%b %d, %H:%M"),
                }

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "interns": interns,
                "tasks": tasks,
                "matrix": matrix,
                "recent_submissions": submissions[:15],
            },
        )
    finally:
        db.close()


@app.post("/submit")
async def submit_task(
    intern_id: int = Form(...),
    day_number: int = Form(...),
    repo_url: str = Form(...),
    notes: str = Form(""),
):
    db = SessionLocal()
    try:
        intern = db.query(Intern).filter(Intern.id == intern_id).first()
        task = db.query(DailyTask).filter(DailyTask.day_number == day_number).first()
        if not intern or not task:
            raise HTTPException(status_code=400, detail="Invalid intern or day selected.")

        try:
            eval_result = await evaluate_intern_submission(
                intern_name=intern.name,
                assigned_domain=f"{intern.domain} (Tool: {intern.custom_tool})",
                day_number=task.day_number,
                day_title=task.title,
                rubric=task.rubric,
                repo_url=repo_url,
            )
            passed = eval_result.passed
            score = eval_result.score
            missing = eval_result.missing_requirements
            warnings = eval_result.security_warnings
            feedback = eval_result.feedback
        except Exception as e:
            passed = False
            score = 0
            missing = [f"Repo fetch/evaluation error: {str(e)}"]
            warnings = []
            feedback = "Could not inspect the GitHub repository. Verify the URL is public and pushed to main."

        submission = Submission(
            intern_id=intern.id,
            day_number=day_number,
            repo_url=repo_url.strip(),
            notes=notes.strip(),
            passed=passed,
            score=score,
            missing_json=json.dumps(missing),
            warnings_json=json.dumps(warnings),
            feedback=feedback,
        )
        db.add(submission)
        db.commit()
        return RedirectResponse(url="/", status_code=303)
    finally:
        db.close()