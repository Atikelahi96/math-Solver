import os
import io
import base64
import sys
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
from PIL import Image
import requests
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Math Bot and Message Classification API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable is not set.")
    print("Please create a .env file with your API key:")
    print("GEMINI_API_KEY=your_api_key_here")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

# Initialize models for math solving
text_model = genai.GenerativeModel('gemini-2.5-pro')

# Initialize the classification model with deterministic settings
classification_generation_config = genai.types.GenerationConfig(
    temperature=0.1,  # Low temperature for consistent outputs
    max_output_tokens=500,  # Adjust based on expected response length
)

classification_model = genai.GenerativeModel(
    'gemini-2.5-pro',
    generation_config=classification_generation_config
)

def process_math_problem(prompt: str, image_data=None):
    """
    Process a math problem using Gemini API with natural output formatting
    """
    try:
        if image_data:
            # For vision tasks, we need to use the correct model
            vision_model = genai.GenerativeModel('gemini-2.5-pro')
            response = vision_model.generate_content([prompt, image_data])
        else:
            # Process text with Gemini Pro
            response = text_model.generate_content(prompt)
        
        # Return the raw response without excessive formatting
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing with Gemini: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Serve the index.html file
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback HTML if file doesn't exist
        return """
        <html>
            <head>
                <title>Math Bot & Message Classifier</title>
                <meta http-equiv="refresh" content="0; URL='/static/index.html'" />
            </head>
            <body>
                <p>Redirecting to <a href="/static/index.html">the application</a>...</p>
            </body>
        </html>
        """

@app.post("/solve/text")
async def solve_text_problem(problem: str = Form(...)):
    """
    Solve a math problem from text input
    """
    prompt = f"""
    Solve the following math problem. Provide a clear, step-by-step solution in natural language.
    Use LaTeX formatting for mathematical expressions (enclose in $ for inline and $$ for display equations).
    Avoid markdown formatting except for LaTeX.
    
    Problem: {problem}
    
    Format your response with clear steps and a final answer.
    """
    
    try:
        solution = process_math_problem(prompt)
        return JSONResponse(content={"problem": problem, "solution": solution})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve/image")
async def solve_image_problem(file: UploadFile = File(...)):
    """
    Solve a math problem from an image
    """
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_data = await file.read()
        
        # Create prompt for the vision model
        prompt = """
        Extract and solve the math problem from this image. Provide a clear, step-by-step solution in natural language.
        Use LaTeX formatting for mathematical expressions (enclose in $ for inline and $$ for display equations).
        Avoid markdown formatting except for LaTeX.
        
        Format your response with clear steps and a final answer.
        """
        
        # Process the image
        img = Image.open(io.BytesIO(image_data))
        solution = process_math_problem(prompt, img)
        
        return JSONResponse(content={"solution": solution})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve/image-with-prompt")
async def solve_image_with_prompt(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Solve a math problem from an image with a custom user prompt
    """
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_data = await file.read()
        
        # Process the image
        img = Image.open(io.BytesIO(image_data))
        solution = process_math_problem(prompt, img)
        
        return JSONResponse(content={"solution": solution, "user_prompt": prompt})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve/url")
async def solve_image_url(url: str = Form(...)):
    """
    Solve a math problem from an image URL
    """
    try:
        # Download image from URL
        response = requests.get(url)
        response.raise_for_status()
        
        # Open image
        image_data = Image.open(io.BytesIO(response.content))
        
        # Create prompt for the vision model
        prompt = """
        Extract and solve the math problem from this image. Provide a clear, step-by-step solution in natural language.
        Use LaTeX formatting for mathematical expressions (enclose in $ for inline and $$ for display equations).
        Avoid markdown formatting except for LaTeX.
        
        Format your response with clear steps and a final answer.
        """
        
        # Process the image
        solution = process_math_problem(prompt, image_data)
        
        return JSONResponse(content={"solution": solution})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image URL: {str(e)}")

@app.post("/check-solution")
async def check_solution(
    problem: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Check if the uploaded solution matches the correct solution for a given problem
    """
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # First, get the correct solution
        solution_prompt = f"""
        Solve the following math problem. Provide only the final answer in its simplest form.
        Use LaTeX formatting if appropriate.
        Do not include any explanations or steps.
        
        Problem: {problem}
        
        Final answer only:
        """
        
        correct_solution = process_math_problem(solution_prompt)
        
        # Read image file
        image_data = await file.read()
        
        # Create prompt for checking the solution
        check_prompt = f"""
        Extract the final answer from this solution image. 
        Compare it with the correct answer: {correct_solution}
        
        Return only:
        - "CORRECT" if the answers match (consider different formats like fractions, decimals, etc.)
        - "INCORRECT" if they don't match
        - "UNCLEAR" if you can't determine
        
        Do not include any explanations.
        """
        
        # Process the image
        img = Image.open(io.BytesIO(image_data))
        result = process_math_problem(check_prompt, img)
        
        return JSONResponse(content={
            "problem": problem,
            "correct_solution": correct_solution,
            "result": result.strip()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
async def classify_message(message: str = Form(...)):
    """
    Classify a message to detect inappropriate content
    Returns 1 if content is problematic, 0 if normal
    """
    classification_prompt = f"""
    Analyze the following message and classify it. Return only a single digit (0 or 1) with no additional text.
    
    Return 1 if the message contains any of the following:
    - Bullying or harassment
    - Slang or inappropriate language
    - Dangerous links (malware, viruses, etc.)
    - Phishing attempts
    - Any other harmful content
    
    Return 0 if the message is normal, safe conversation.
    
    Message: {message}
    
    Classification:
    """
    
    try:
        response = classification_model.generate_content(classification_prompt)
        
        # Extract just the classification digit
        classification = re.search(r'\d', response.text)
        
        if classification:
            result = int(classification.group(0))
            return JSONResponse(content={"message": message, "classification": result})
        else:
            # If no digit found, default to safe (0)
            return JSONResponse(content={"message": message, "classification": 0})
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying message: {str(e)}")

MAX_QUESTIONS = 20  # safety limit

@app.post("/generate-question")
async def generate_math_question(
    grade: str = Form(...),
    subject: str = Form(...),
    count: int = Form(1)  # default is 1 if not provided
):
    """
    Generate one or more math questions based on grade level and subject/topic.
    Uses multiple strategies to reliably return `count` question/answer pairs.
    """
    try:
        # sanitize count
        try:
            count = int(count)
        except Exception:
            count = 1
        if count < 1:
            count = 1
        if count > MAX_QUESTIONS:
            count = MAX_QUESTIONS

        # Primary prompt: ask for strict JSON only
        json_prompt = f"""
You are a math teacher. Generate {count} unique math questions for a student in grade {grade}
on the topic of {subject}. Each question should be age-appropriate, clear, and solvable.

Return **only** valid JSON. The JSON must be an array of objects with exactly these keys:
[
  {{
    "question": "question text here",
    "answer": "answer text here"
  }},
  ...
]

Do NOT include any additional text outside the JSON array. Make sure there are exactly {count} objects.
"""
        response = text_model.generate_content(json_prompt)
        text = response.text.strip()

        questions = []

        # 1) Try to extract JSON array from response
        json_match = re.search(r'(\[.*\])', text, re.DOTALL)
        if json_match:
            try:
                arr = json.loads(json_match.group(1))
                # normalize and add
                for item in arr:
                    q = item.get("question") if isinstance(item, dict) else None
                    a = item.get("answer") if isinstance(item, dict) else None
                    if q and a:
                        questions.append({"question": q.strip(), "answer": a.strip()})
            except Exception:
                # JSON parse failed, fall through to regex parsing
                pass

        # 2) If JSON didn't yield enough, try to parse using Q/A regex
        if len(questions) < count:
            # Find all Question ... Answer ... pairs using robust regex
            qa_pairs = re.findall(
                r"(?:Question\s*\d*[:：]\s*)(.*?)(?:\r?\n\s*Answer\s*\d*[:：]\s*)(.*?)(?=(?:\r?\n\s*Question\s*\d*[:：])|$)",
                text,
                re.DOTALL | re.IGNORECASE
            )
            for q, a in qa_pairs:
                if len(questions) >= count:
                    break
                questions.append({"question": q.strip(), "answer": a.strip()})

        # 3) If still short, request additional individual questions until we reach `count`
        attempt = 0
        while len(questions) < count and attempt < (count * 2):
            attempt += 1
            remaining = count - len(questions)
            single_prompt = f"""
Generate 1 unique math question for grade {grade} on the topic {subject}.
Return as:
Question: ...
Answer: ...
Do not repeat previous questions.
"""
            resp = text_model.generate_content(single_prompt)
            text_single = resp.text.strip()

            # try to parse single pair
            m = re.search(r"Question\s*\d*[:：]\s*(.*?)(?:\r?\n\s*Answer\s*\d*[:：]\s*(.*))?$",
                          text_single, re.DOTALL | re.IGNORECASE)
            if m:
                q = m.group(1).strip()
                a = m.group(2).strip() if m.group(2) else ""
                if q and a:
                    # Avoid duplicates (basic check)
                    if not any(q == existing["question"] for existing in questions):
                        questions.append({"question": q, "answer": a})
                        continue

            # fallback: try to split by lines if model returned short text
            lines = [ln.strip() for ln in text_single.splitlines() if ln.strip()]
            if len(lines) >= 2:
                q = lines[0]
                a = lines[1]
                if not any(q == e["question"] for e in questions):
                    questions.append({"question": q, "answer": a})

            # if nothing added in this loop, continue and retry (up to attempt limit)

        # final safety: if still short, pad with placeholders
        while len(questions) < count:
            questions.append({
                "question": "Unable to generate question — please retry.",
                "answer": ""
            })

        # normalize numbering and return exactly `count` items
        result = []
        for i in range(count):
            qitem = questions[i]
            result.append({
                "number": i + 1,
                "question": qitem["question"],
                "answer": qitem["answer"]
            })

        return JSONResponse(content={
            "grade": grade,
            "subject": subject,
            "count": count,
            "questions": result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)