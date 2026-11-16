import os
import ast
import re
from urllib.parse import urlparse
import httpx
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()


class TaskEvaluation(BaseModel):
    passed: bool = Field(
        description="True ONLY if the core requirements for the selected day are genuinely implemented."
    )
    score: int = Field(ge=0, le=100, description="Completion score from 0 to 100.")
    missing_requirements: list[str] = Field(
        description="Specific missing features, files, or logic gaps."
    )
    security_warnings: list[str] = Field(
        description="Exposed API keys, missing .gitignore, or committed venv/.env files."
    )
    feedback: str = Field(
        description="Clear, encouraging 2-3 sentence engineering review telling the intern what worked and what to fix."
    )


def parse_github_url(repo_url: str) -> tuple[str, str]:
    cleaned = repo_url.strip().rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    parsed = urlparse(cleaned)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL. Expected format: https://github.com/username/repo-name")
    return parts[0], parts[1]


async def fetch_repo_contents(repo_url: str) -> dict[str, str]:
    owner, repo = parse_github_url(repo_url)
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    async with httpx.AsyncClient(timeout=20.0) as http:
        branch = "main"
        tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        res = await http.get(tree_url, headers=headers)
        if res.status_code == 404:
            branch = "master"
            tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
            res = await http.get(tree_url, headers=headers)

        if res.status_code == 404:
            raise ValueError("Repository not found or is private (check GITHUB_TOKEN in .env).")
        res.raise_for_status()

        files_content = {}
        allowed_exts = (".py", ".txt", ".json", ".md", ".html", ".yaml", ".yml")

        for item in res.json().get("tree", []):
            path = item["path"]
            if item["type"] != "blob":
                continue
            if path == ".env" or path.startswith("venv/") or "/venv/" in path:
                files_content[f"[COMMITTED_BAD_FILE] {path}"] = "Committed sensitive/environment directory."
                continue

            if path.endswith(allowed_exts) or path in (".gitignore", ".env.example", "Dockerfile"):
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
                f_res = await http.get(raw_url, headers=headers)
                if f_res.status_code == 200:
                    files_content[path] = f_res.text[:15000]

        return files_content


def run_static_checks(repo_files: dict[str, str]) -> list[str]:
    warnings = []
    if ".gitignore" not in repo_files:
        warnings.append("Missing .gitignore file.")
    else:
        gi = repo_files[".gitignore"]
        if ".env" not in gi:
            warnings.append(".gitignore does not exclude .env.")
        if "venv" not in gi:
            warnings.append(".gitignore does not exclude venv/.")

    secret_pattern = re.compile(r"(sk-[a-zA-Z0-9]{20,}|AIza[0-9A-Za-z\-_]{30,}|gsk_[a-zA-Z0-9]{30,})")

    for path, content in repo_files.items():
        if path.startswith("[COMMITTED_BAD_FILE]"):
            warnings.append(f"Bad commit detected: {path.replace('[COMMITTED_BAD_FILE] ', '')} should not be in Git.")
            continue
        if secret_pattern.search(content):
            warnings.append(f"CRITICAL: Possible hardcoded API key found in {path}!")
        if path.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as e:
                warnings.append(f"Python SyntaxError in {path} (line {e.lineno}): {e.msg}")

    return warnings


async def evaluate_intern_submission(
    intern_name: str,
    assigned_domain: str,
    day_number: int,
    day_title: str,
    rubric: str,
    repo_url: str,
) -> TaskEvaluation:
    repo_files = await fetch_repo_contents(repo_url)
    if not repo_files:
        return TaskEvaluation(
            passed=False,
            score=0,
            missing_requirements=["Repository appears empty—no code files found."],
            security_warnings=[],
            feedback="Your repository is empty. Make sure you committed and pushed your files to the main branch.",
        )

    static_warnings = run_static_checks(repo_files)

    code_dump = "\n\n".join(
        f"=== FILE: {fname} ===\n{code}"
        for fname, code in repo_files.items()
        if not fname.startswith("[COMMITTED_BAD_FILE]")
    )

    system_prompt = (
        "You are an AI Engineering Team Lead at Zigron reviewing an intern's daily chatbot task. "
        "Verify whether the intern genuinely implemented the required code for their assigned domain. "
        "Do not pass empty boilerplate, copy-pasted unrelated domains, or fake hardcoded answers. "
        "Be constructive, fair, and strict on API key security."
    )

    user_prompt = f"""
Intern Name: {intern_name}
Assigned Domain & Tool: {assigned_domain}
Milestone: Day {day_number} — {day_title}

--- DAY {day_number} GRADING RUBRIC ---
{rubric}

--- AUTOMATED STATIC ANALYSIS ---
{chr(10).join(static_warnings) if static_warnings else "No static syntax or secret leaks detected."}

--- REPOSITORY FILE TREE ---
{", ".join(repo_files.keys())}

--- FILE CONTENTS ---
{code_dump}
"""

    response = await client.beta.chat.completions.parse(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=TaskEvaluation,
        temperature=0.1,
    )

    result = response.choices[0].message.parsed
    for w in static_warnings:
        if w not in result.security_warnings:
            result.security_warnings.append(w)
    return result