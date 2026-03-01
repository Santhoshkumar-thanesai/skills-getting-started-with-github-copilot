"""
High School Management System API

A super simple FastAPI application that allows students to view and sign up
for extracurricular activities at Mergington High School.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Literal, Optional
import httpx
import json
import os
from datetime import date, datetime
from pathlib import Path

app = FastAPI(title="Mergington High School API",
              description="API for viewing and signing up for extracurricular activities")

# Mount the static files directory
current_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=os.path.join(Path(__file__).parent,
          "static")), name="static")

# In-memory activity database
activities = {
    "Chess Club": {
        "description": "Learn strategies and compete in chess tournaments",
        "schedule": "Fridays, 3:30 PM - 5:00 PM",
        "max_participants": 12,
        "participants": ["michael@mergington.edu", "daniel@mergington.edu"]
    },
    "Programming Class": {
        "description": "Learn programming fundamentals and build software projects",
        "schedule": "Tuesdays and Thursdays, 3:30 PM - 4:30 PM",
        "max_participants": 20,
        "participants": ["emma@mergington.edu", "sophia@mergington.edu"]
    },
    "Gym Class": {
        "description": "Physical education and sports activities",
        "schedule": "Mondays, Wednesdays, Fridays, 2:00 PM - 3:00 PM",
        "max_participants": 30,
        "participants": ["john@mergington.edu", "olivia@mergington.edu"]
    }
}


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/activities")
def get_activities():
    return activities


@app.post("/activities/{activity_name}/signup")
def signup_for_activity(activity_name: str, email: str):
    """Sign up a student for an activity"""
    # Validate activity exists
    if activity_name not in activities:
        raise HTTPException(status_code=404, detail="Activity not found")

    # Get the specific activity
    activity = activities[activity_name]

    # Add student
    activity["participants"].append(email)
    return {"message": f"Signed up {email} for {activity_name}"}


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

CategoryCode = Literal["SAR", "KBY", "KGL", "MDR"]


class CaptionRequest(BaseModel):
    model_id: str
    image_url: str
    extra_context: Optional[str] = None


class CaptionResponse(BaseModel):
    model_id: str
    caption: str
    hashtags: List[str]
    category_code: CategoryCode


class ModelInfo(BaseModel):
    model_id: str
    category_code: CategoryCode
    date: date


def parse_model_id(model_id: str) -> ModelInfo:
    """Parse IDs like SAR60213 into category and date."""
    if len(model_id) < 8:
        raise ValueError("Model ID must be at least 8 characters, e.g. SAR60213")

    prefix = model_id[:3].upper()
    if prefix not in {"SAR", "KBY", "KGL", "MDR"}:
        raise ValueError("Unknown category code. Use SAR, KBY, KGL, or MDR.")

    suffix = model_id[3:]
    if len(suffix) != 5 or not suffix.isdigit():
        raise ValueError("Suffix must be 5 digits in the format YMMDD, e.g. 60213.")

    year_digit = int(suffix[0])
    month = int(suffix[1:3])
    day = int(suffix[3:])

    current_year = datetime.utcnow().year
    base_decade = current_year - (current_year % 10)
    year = base_decade + year_digit

    parsed_date = date(year, month, day)

    return ModelInfo(model_id=model_id, category_code=prefix, date=parsed_date)


@app.post("/ai/caption", response_model=CaptionResponse)
async def generate_caption(payload: CaptionRequest):
    """Generate caption, hashtags, and category code for a product image."""
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server.",
        )

    prompt_parts = [
        "You are an expert fashion social media copywriter for an Indian ethnic wear boutique.",
        "Generate:",
        "1) A 2–3 line Instagram caption in Hinglish (Indian audience).",
        "2) 15–20 relevant hashtags for Indian ethnic / kids wear.",
        "3) A category code from: SAR (Saree), KBY (Kids-Boy), KGL (Kids-Girl), MDR (Modern Dress).",
        "",
        f"Model ID: {payload.model_id}",
        f"Image URL: {payload.image_url}",
    ]

    if payload.extra_context:
        prompt_parts.append(f"Extra context: {payload.extra_context}")

    prompt_parts.append(
        'Reply in strict JSON with keys: "caption" (string), '
        '"hashtags" (array of strings), "category_code" (one of SAR,KBY,KGL,MDR). '
        "No extra commentary.",
    )

    user_content = "\n".join(prompt_parts)

    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You reply only with valid JSON."},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Unexpected response from AI provider.",
        ) from exc

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {exc}") from exc

    hashtags = parsed.get("hashtags") or []
    if isinstance(hashtags, str):
        hashtags = [h.strip() for h in hashtags.split() if h.strip()]

    category_code = (parsed.get("category_code") or "").upper()
    if category_code not in {"SAR", "KBY", "KGL", "MDR"}:
        try:
            model_info = parse_model_id(payload.model_id)
            category_code = model_info.category_code
        except Exception:
            category_code = "SAR"

    return CaptionResponse(
        model_id=payload.model_id,
        caption=parsed.get("caption", ""),
        hashtags=hashtags,
        category_code=category_code,  # type: ignore[arg-type]
    )


@app.get("/models/{model_id}", response_model=ModelInfo)
def get_model_info(model_id: str):
    """Decode a model ID like SAR60213 into structured information."""
    try:
        return parse_model_id(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
