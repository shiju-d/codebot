import asyncio
import httpx
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llms import local_llm, bedrock_llm, JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN
import engine
from message import parse_message
from jira import (
    parse_rca_input, extract_adf_text, build_rca_message,
    md_to_jira, fetch_jira_issue, post_jira_comment,
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class RcaRequest(BaseModel):
    input: str
    session_id: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(engine.init_all_services)
    yield


app = FastAPI(title="codebot", lifespan=lifespan)


async def _chat(request: ChatRequest, llm, llm_key: str):
    if not engine.services:
        raise HTTPException(status_code=503, detail="RAG engine is initializing")

    try:
        service_name, message = parse_message(request.message)
    except ValueError:
        valid = ", ".join(engine.services.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Missing service prefix. Format: <service>: <message>. Valid services: {valid}",
        )

    if service_name not in engine.services:
        valid = ", ".join(engine.services.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown service '{service_name}'. Valid services: {valid}. Format: <service>: <message>",
        )

    try:
        eng = engine.get_engine(request.session_id, service_name, llm, llm_key)
        response = await asyncio.to_thread(eng.chat, message)
        sources = list({
            node.metadata.get("file_path", "unknown")
            for node in response.source_nodes
        })
        return {"response": response.response, "sources": sources}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/services")
def list_services():
    return {"services": list(engine.services.keys())}


@app.get("/project/{project_key}")
def resolve_project(project_key: str):
    upper = project_key.upper()
    for name, svc in engine.services.items():
        if svc.get("jira_project_key") == upper:
            return {"service": name, "project_key": upper}
    raise HTTPException(status_code=404, detail=f"No service mapped to Jira project '{project_key}'")


@app.post("/chat")
async def chat_local(request: ChatRequest):
    return await _chat(request, local_llm, "local")


@app.post("/chat/bedrock")
async def chat_bedrock(request: ChatRequest):
    if not bedrock_llm:
        raise HTTPException(status_code=503, detail="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY not configured")
    return await _chat(request, bedrock_llm, "bedrock")


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    for svc in engine.services.values():
        for llm_sessions in svc["sessions"].values():
            llm_sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.post("/rca")
async def rca(request: RcaRequest):
    if not bedrock_llm:
        raise HTTPException(status_code=503, detail="AWS credentials not configured")
    if not all([JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN]):
        raise HTTPException(
            status_code=503,
            detail="Jira credentials not configured — set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN",
        )
    if not engine.services:
        raise HTTPException(status_code=503, detail="RAG engine is initializing")

    try:
        service_name, issue_key, additional_context = parse_rca_input(request.input)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if service_name not in engine.services:
        valid = ", ".join(engine.services.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown service '{service_name}'. Valid services: {valid}",
        )

    try:
        issue = await fetch_jira_issue(JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, issue_key)
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        detail = f"Jira API error fetching {issue_key}: {e.response.status_code}" if isinstance(e, httpx.HTTPStatusError) else f"Jira connection error: {e}"
        raise HTTPException(status_code=502, detail=detail)

    summary = issue["fields"].get("summary", "(no summary)")
    desc_adf = issue["fields"].get("description")
    description = extract_adf_text(desc_adf).strip() if desc_adf else "No description provided."

    session_id = request.session_id or f"jira-{issue_key}"
    message = build_rca_message(service_name, issue_key, summary, description, additional_context)

    try:
        eng = engine.get_engine(session_id, service_name, bedrock_llm, "bedrock")
        response = await asyncio.to_thread(eng.chat, message)
        rca_text = response.response
        sources = list({
            node.metadata.get("file_path", "unknown")
            for node in response.source_nodes
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    jira_body = "\n".join([
        "h2. \U0001f916 AI-Generated Root Cause Analysis",
        "",
        md_to_jira(rca_text),
        "",
        "----",
        "_Generated automatically by codebot. Please review before acting on it._",
    ])

    try:
        await post_jira_comment(JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, issue_key, jira_body)
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        detail = f"Failed to post Jira comment on {issue_key}: {e.response.status_code}" if isinstance(e, httpx.HTTPStatusError) else f"Jira connection error posting comment: {e}"
        raise HTTPException(status_code=502, detail=detail)

    return {
        "response": rca_text,
        "sources": sources,
        "issue_key": issue_key,
        "comment_posted": True,
        "output": f"✅ RCA posted to {issue_key}.\n\n{rca_text}",
    }


@app.post("/reindex")
async def reindex_all():
    service_names = await asyncio.to_thread(engine.reindex_all_services)
    return {"status": "reindexed", "services": service_names}


@app.post("/reindex/{service_name}")
async def reindex_service(service_name: str):
    try:
        await asyncio.to_thread(engine.reindex_one_service, service_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "reindexed", "service": service_name}
