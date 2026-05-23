"""
AIRAJ AI Service — FastAPI (port 8000)
Resume analysis, job matching, ATS scoring, skill extraction
"""

import os
import time
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    import joblib
    classifier = joblib.load("models/best_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    print("[AI] Classifier and label encoder loaded from models/")
except FileNotFoundError:
    classifier = None
    label_encoder = None
    print("[AI] WARNING: models/best_model.pkl not found. Using rule-based category prediction.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Load Embedding Model ──────────────────────────────────────
logger.info("[AI] Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("[AI] Model loaded successfully")

app = FastAPI(title="AIRAJ AI Service", version="1.0.0", description="Resume analysis and job matching API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Skill List ────────────────────────────────────────────────
SKILL_LIST = [
    "python", "javascript", "typescript", "react", "vue", "angular", "node.js", "express",
    "django", "flask", "fastapi", "mongodb", "postgresql", "mysql", "redis", "docker",
    "kubernetes", "aws", "gcp", "azure", "git", "ci/cd", "machine learning", "deep learning",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "sql", "html", "css",
    "tailwind", "bootstrap", "rest api", "graphql", "microservices", "linux", "bash",
    "java", "c++", "c#", "php", "ruby", "swift", "kotlin", "flutter", "react native",
    "tableau", "power bi", "excel", "spark", "hadoop", "nlp", "computer vision",
    "data analysis", "data science", "xgboost", "opencv", "selenium", "jest", "pytest",
]

# ── Resource Map ──────────────────────────────────────────────
RESOURCE_MAP = {
    "python": {"reason": "Most in-demand AI/backend language", "course": "Python for Everybody - Coursera", "url": "https://coursera.org/specializations/python", "time": "4-6 weeks"},
    "javascript": {"reason": "Essential for web development", "course": "JavaScript.info", "url": "https://javascript.info", "time": "4-8 weeks"},
    "typescript": {"reason": "Industry standard for large JS projects", "course": "TypeScript Handbook", "url": "https://typescriptlang.org/docs/handbook/intro.html", "time": "2-3 weeks"},
    "react": {"reason": "Top frontend framework for web apps", "course": "React Official Docs", "url": "https://react.dev/learn", "time": "3-4 weeks"},
    "vue": {"reason": "Popular frontend framework, gentle learning curve", "course": "Vue.js Official Guide", "url": "https://vuejs.org/guide/introduction", "time": "2-3 weeks"},
    "angular": {"reason": "Enterprise-grade frontend framework", "course": "Angular Official Tutorial", "url": "https://angular.io/tutorial", "time": "4-6 weeks"},
    "node.js": {"reason": "JavaScript runtime for backend development", "course": "Node.js Official Docs", "url": "https://nodejs.org/en/learn/getting-started/introduction-to-nodejs", "time": "3-4 weeks"},
    "express": {"reason": "Most popular Node.js web framework", "course": "Express.js Guide", "url": "https://expressjs.com/en/guide/routing.html", "time": "1-2 weeks"},
    "django": {"reason": "Batteries-included Python web framework", "course": "Django Official Tutorial", "url": "https://djangoproject.com/start/", "time": "3-4 weeks"},
    "flask": {"reason": "Lightweight Python web framework", "course": "Flask Official Documentation", "url": "https://flask.palletsprojects.com/en/3.0.x/tutorial/", "time": "2-3 weeks"},
    "fastapi": {"reason": "Modern high-performance Python API framework", "course": "FastAPI Tutorial", "url": "https://fastapi.tiangolo.com/tutorial/", "time": "1-2 weeks"},
    "mongodb": {"reason": "Most popular NoSQL database", "course": "MongoDB University", "url": "https://learn.mongodb.com", "time": "2-3 weeks"},
    "postgresql": {"reason": "Advanced open-source relational database", "course": "PostgreSQL Tutorial", "url": "https://postgresqltutorial.com", "time": "2-3 weeks"},
    "mysql": {"reason": "World's most used open-source database", "course": "MySQL Tutorial", "url": "https://mysqltutorial.org", "time": "2-3 weeks"},
    "redis": {"reason": "Fast in-memory data store for caching", "course": "Redis University", "url": "https://university.redis.com", "time": "1-2 weeks"},
    "docker": {"reason": "Essential containerization platform", "course": "Docker Official Get Started", "url": "https://docs.docker.com/get-started/", "time": "2-3 weeks"},
    "kubernetes": {"reason": "Industry-standard container orchestration", "course": "Kubernetes Official Tutorial", "url": "https://kubernetes.io/docs/tutorials/kubernetes-basics/", "time": "4-6 weeks"},
    "aws": {"reason": "Most widely used cloud platform", "course": "AWS Cloud Practitioner - Coursera", "url": "https://coursera.org/learn/aws-cloud-practitioner-essentials", "time": "4-6 weeks"},
    "gcp": {"reason": "Google Cloud Platform - strong in AI/ML", "course": "Google Cloud Skills Boost", "url": "https://cloudskillsboost.google", "time": "4-6 weeks"},
    "azure": {"reason": "Microsoft's enterprise cloud platform", "course": "Microsoft Learn - Azure Fundamentals", "url": "https://learn.microsoft.com/en-us/training/paths/azure-fundamentals-describe-azure-architecture-services/", "time": "4-6 weeks"},
    "git": {"reason": "Essential version control for all developers", "course": "Pro Git Book (free)", "url": "https://git-scm.com/book/en/v2", "time": "1-2 weeks"},
    "ci/cd": {"reason": "Automate testing and deployment pipelines", "course": "GitHub Actions Quickstart", "url": "https://docs.github.com/en/actions/quickstart", "time": "2-3 weeks"},
    "machine learning": {"reason": "Core ML skills are highly sought-after", "course": "ML Specialization - Andrew Ng", "url": "https://coursera.org/specializations/machine-learning-introduction", "time": "3 months"},
    "deep learning": {"reason": "Powers modern AI applications", "course": "Deep Learning Specialization - deeplearning.ai", "url": "https://coursera.org/specializations/deep-learning", "time": "3-4 months"},
    "tensorflow": {"reason": "Google's production-grade ML framework", "course": "TensorFlow Official Tutorials", "url": "https://tensorflow.org/tutorials", "time": "4-6 weeks"},
    "pytorch": {"reason": "Research-to-production ML framework", "course": "PyTorch Official Tutorials", "url": "https://pytorch.org/tutorials/beginner/basics/intro.html", "time": "4-6 weeks"},
    "scikit-learn": {"reason": "Standard ML library for Python", "course": "scikit-learn User Guide", "url": "https://scikit-learn.org/stable/user_guide.html", "time": "2-3 weeks"},
    "pandas": {"reason": "Essential data manipulation library", "course": "Pandas Official Getting Started", "url": "https://pandas.pydata.org/docs/getting_started/index.html", "time": "2-3 weeks"},
    "numpy": {"reason": "Foundation of scientific computing in Python", "course": "NumPy Official Tutorial", "url": "https://numpy.org/doc/stable/user/quickstart.html", "time": "1-2 weeks"},
    "sql": {"reason": "Critical skill for data roles", "course": "SQLZoo Interactive SQL", "url": "https://sqlzoo.net", "time": "2-3 weeks"},
    "html": {"reason": "Fundamental building block of the web", "course": "MDN HTML Basics", "url": "https://developer.mozilla.org/en-US/docs/Learn/HTML/Introduction_to_HTML", "time": "1-2 weeks"},
    "css": {"reason": "Essential for styling web applications", "course": "CSS-Tricks Complete Guide", "url": "https://css-tricks.com/where-do-you-learn-html-css-in-2020", "time": "2-3 weeks"},
    "tailwind": {"reason": "Utility-first CSS framework gaining rapid adoption", "course": "Tailwind CSS Official Docs", "url": "https://tailwindcss.com/docs/installation", "time": "1-2 weeks"},
    "bootstrap": {"reason": "Most popular CSS framework", "course": "Bootstrap Official Docs", "url": "https://getbootstrap.com/docs/5.3/getting-started/introduction/", "time": "1-2 weeks"},
    "rest api": {"reason": "Standard for modern web service communication", "course": "REST API Tutorial", "url": "https://restfulapi.net", "time": "1-2 weeks"},
    "graphql": {"reason": "Modern API query language replacing REST", "course": "GraphQL Official Introduction", "url": "https://graphql.org/learn/", "time": "2-3 weeks"},
    "microservices": {"reason": "Architecture pattern used by leading tech companies", "course": "Microservices.io Patterns", "url": "https://microservices.io/patterns/microservices.html", "time": "3-4 weeks"},
    "linux": {"reason": "Essential for backend and DevOps roles", "course": "Linux Foundation Training", "url": "https://training.linuxfoundation.org/training/introduction-to-linux/", "time": "3-4 weeks"},
    "bash": {"reason": "Required for scripting and automation", "course": "Bash Scripting Tutorial", "url": "https://linuxconfig.org/bash-scripting-tutorial-for-beginners", "time": "1-2 weeks"},
    "java": {"reason": "Enterprise-grade language, still widely used", "course": "Java Programming MOOC", "url": "https://java-programming.mooc.fi", "time": "2-3 months"},
    "c++": {"reason": "High-performance systems and game development", "course": "learncpp.com", "url": "https://learncpp.com", "time": "2-3 months"},
    "c#": {"reason": "Microsoft ecosystem and Unity game development", "course": "Microsoft C# Fundamentals", "url": "https://learn.microsoft.com/en-us/dotnet/csharp/tour-of-csharp/", "time": "4-6 weeks"},
    "php": {"reason": "Powers ~75% of the web (WordPress, Laravel)", "course": "PHP Official Manual", "url": "https://php.net/manual/en/getting-started.php", "time": "3-4 weeks"},
    "ruby": {"reason": "Elegant scripting and Rails web framework", "course": "The Odin Project - Ruby", "url": "https://theodinproject.com/paths/full-stack-ruby-on-rails/courses/ruby", "time": "4-6 weeks"},
    "swift": {"reason": "Required for iOS/macOS development", "course": "Apple Swift Tour", "url": "https://docs.swift.org/swift-book/documentation/the-swift-programming-language/guidedtour/", "time": "4-6 weeks"},
    "kotlin": {"reason": "Modern Android development language", "course": "Kotlin Official Documentation", "url": "https://kotlinlang.org/docs/getting-started.html", "time": "3-4 weeks"},
    "flutter": {"reason": "Cross-platform mobile development framework", "course": "Flutter Official Codelabs", "url": "https://docs.flutter.dev/get-started/codelab", "time": "4-6 weeks"},
    "react native": {"reason": "Mobile apps using React knowledge", "course": "React Native Official Tutorial", "url": "https://reactnative.dev/docs/tutorial", "time": "3-4 weeks"},
    "tableau": {"reason": "Leading data visualization and BI tool", "course": "Tableau Free Training Videos", "url": "https://www.tableau.com/learn/training", "time": "2-3 weeks"},
    "power bi": {"reason": "Microsoft BI tool, widely used in enterprises", "course": "Microsoft Power BI Learning", "url": "https://learn.microsoft.com/en-us/power-bi/fundamentals/service-get-started", "time": "2-3 weeks"},
    "excel": {"reason": "Universal tool for data analysis", "course": "Microsoft Excel Training", "url": "https://support.microsoft.com/en-us/office/excel-for-windows-training-9bc05390-e94c-46af-a5b3-d7c22f6990bb", "time": "2-3 weeks"},
    "spark": {"reason": "Distributed computing for big data", "course": "Databricks Learning - Apache Spark", "url": "https://academy.databricks.com/award/completion/a0B8Y000001QaEi", "time": "4-6 weeks"},
    "hadoop": {"reason": "Big data processing framework", "course": "Hadoop Tutorial - Simplilearn", "url": "https://simplilearn.com/tutorials/hadoop-tutorial/what-is-hadoop", "time": "3-4 weeks"},
    "nlp": {"reason": "Critical for AI and text processing roles", "course": "NLP with Python - NLTK Book", "url": "https://nltk.org/book/", "time": "4-6 weeks"},
    "computer vision": {"reason": "High-demand AI specialization", "course": "CS231n: CNN for Visual Recognition - Stanford", "url": "https://cs231n.github.io", "time": "2-3 months"},
    "data analysis": {"reason": "Foundational skill for data roles", "course": "Data Analysis with Python - freeCodeCamp", "url": "https://freecodecamp.org/learn/data-analysis-with-python/", "time": "4-6 weeks"},
    "data science": {"reason": "Combines statistics, coding, and domain expertise", "course": "Data Science Specialization - JHU Coursera", "url": "https://coursera.org/specializations/jhu-data-science", "time": "4-6 months"},
    "xgboost": {"reason": "Winning algorithm in many ML competitions", "course": "XGBoost Documentation", "url": "https://xgboost.readthedocs.io/en/stable/tutorials/model.html", "time": "1-2 weeks"},
    "opencv": {"reason": "Essential library for computer vision tasks", "course": "OpenCV Python Tutorials", "url": "https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html", "time": "2-3 weeks"},
    "selenium": {"reason": "Automated browser testing framework", "course": "Selenium with Python", "url": "https://selenium-python.readthedocs.io", "time": "1-2 weeks"},
    "jest": {"reason": "Most popular JavaScript testing framework", "course": "Jest Official Docs", "url": "https://jestjs.io/docs/getting-started", "time": "1-2 weeks"},
    "pytest": {"reason": "Standard Python testing framework", "course": "pytest Documentation", "url": "https://docs.pytest.org/en/stable/getting-started.html", "time": "1-2 weeks"},
}

# ── Category Prediction Fallback ──────────────────────────────
CATEGORY_RULES = {
    "Data Science": ["data science", "machine learning", "pandas", "numpy", "sklearn", "scikit-learn", "tensorflow", "pytorch", "xgboost", "jupyter"],
    "Web Developer": ["react", "vue", "angular", "html", "css", "javascript", "typescript", "frontend", "web developer"],
    "Python Developer": ["python", "django", "flask", "fastapi"],
    "Java Developer": ["java", "spring", "hibernate", "maven"],
    "DevOps Engineer": ["docker", "kubernetes", "ci/cd", "linux", "bash", "ansible", "terraform", "jenkins"],
    "Data Analyst": ["sql", "excel", "tableau", "power bi", "data analysis"],
    "Mobile Developer": ["flutter", "react native", "swift", "kotlin", "android", "ios"],
    "Backend Developer": ["node.js", "express", "postgresql", "mongodb", "rest api", "microservices"],
}

def predict_category_by_skills(skills: list) -> str:
    skill_lower = [s.lower() for s in skills]
    best_cat = "Software Developer"
    best_count = 0
    for cat, keywords in CATEGORY_RULES.items():
        count = sum(1 for kw in keywords if any(kw in s for s in skill_lower))
        if count > best_count:
            best_count = count
            best_cat = cat
    return best_cat

# ── Helper Functions ──────────────────────────────────────────
def extract_skills(text: str) -> list:
    lower = text.lower()
    return [s for s in SKILL_LIST if s in lower]

def build_skill_gaps(missing: list) -> list:
    gaps = []
    for skill in missing:
        if skill in RESOURCE_MAP:
            gaps.append({**RESOURCE_MAP[skill], "skill": skill})
    return gaps

# ── Pydantic Models ───────────────────────────────────────────
class ExtractSkillsRequest(BaseModel):
    text: str

class AnalyzeRequest(BaseModel):
    resume_text: str
    job_description: Optional[str] = None

class BulkJobItem(BaseModel):
    id: str
    title: str
    description: str
    skills: List[str] = []

class BulkAnalyzeRequest(BaseModel):
    resume_text: str
    jobs: List[BulkJobItem]

class ATSAnalyzeRequest(BaseModel):
    resume_text: str
    job_description: str

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "AIRAJ AI Service is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classifier_loaded": classifier is not None,
    }

@app.post("/extract-skills")
def extract_skills_endpoint(req: ExtractSkillsRequest):
    skills = extract_skills(req.text)
    return {"skills": skills, "count": len(skills)}

@app.post("/analyze")
def analyze_resume(req: AnalyzeRequest):
    start = time.time()
    resume_skills = extract_skills(req.resume_text)

    # Predict category
    resume_embedding = model.encode([req.resume_text])[0]
    if classifier is not None and label_encoder is not None:
        try:
            predicted_category = label_encoder.inverse_transform(
                classifier.predict([resume_embedding])
            )[0]
        except Exception:
            predicted_category = predict_category_by_skills(resume_skills)
    else:
        predicted_category = predict_category_by_skills(resume_skills)

    result = {
        "skills": resume_skills,
        "skill_count": len(resume_skills),
        "predicted_category": predicted_category,
        "processing_time_ms": round((time.time() - start) * 1000, 2),
    }

    # If job description provided, compute match
    if req.job_description:
        job_skills = extract_skills(req.job_description)
        job_embedding = model.encode([req.job_description])[0]

        semantic_raw = cosine_similarity([resume_embedding], [job_embedding])[0][0]
        semantic_score = float(semantic_raw) * 100

        matched = list(set(resume_skills) & set(job_skills))
        missing = list(set(job_skills) - set(resume_skills))
        keyword_score = (len(matched) / len(job_skills) * 100) if job_skills else 0
        match_score = round(semantic_score * 0.6 + keyword_score * 0.4, 2)
        skill_gaps = build_skill_gaps(missing)

        result.update({
            "match_score": match_score,
            "semantic_score": round(semantic_score, 2),
            "keyword_score": round(keyword_score, 2),
            "matched_skills": matched,
            "missing_skills": missing,
            "skill_gaps": skill_gaps,
            "job_skills": job_skills,
        })

    logger.info(f"[AI] /analyze completed in {result['processing_time_ms']}ms, category: {predicted_category}")
    return result

@app.post("/bulk-analyze")
def bulk_analyze(req: BulkAnalyzeRequest):
    start = time.time()

    if not req.resume_text or not req.jobs:
        raise HTTPException(status_code=400, detail="resume_text and jobs are required")

    resume_embedding = model.encode([req.resume_text])[0]
    resume_skills = extract_skills(req.resume_text)

    recommendations = []
    for job in req.jobs:
        job_text = f"{job.title} {job.description} {' '.join(job.skills)}"
        job_skills = extract_skills(job_text)
        job_embedding = model.encode([job_text])[0]

        semantic_raw = cosine_similarity([resume_embedding], [job_embedding])[0][0]
        semantic_score = float(semantic_raw) * 100

        matched = list(set(resume_skills) & set(job_skills))
        missing = list(set(job_skills) - set(resume_skills))
        keyword_score = (len(matched) / len(job_skills) * 100) if job_skills else 0
        match_score = round(semantic_score * 0.6 + keyword_score * 0.4, 2)
        skill_gaps = build_skill_gaps(missing)

        recommendations.append({
            "job_id": job.id,
            "match_score": match_score,
            "semantic_score": round(semantic_score, 2),
            "keyword_score": round(keyword_score, 2),
            "matched_skills": matched,
            "missing_skills": missing,
            "skill_gaps": skill_gaps,
        })

    recommendations.sort(key=lambda x: x["match_score"], reverse=True)

    total_ms = round((time.time() - start) * 1000, 2)
    logger.info(f"[AI] /bulk-analyze: {len(req.jobs)} jobs in {total_ms}ms")

    return {"recommendations": recommendations, "total_jobs": len(req.jobs), "processing_time_ms": total_ms}

@app.post("/ats-analyze")
def ats_analyze(req: ATSAnalyzeRequest):
    start = time.time()
    lower_resume = req.resume_text.lower()

    # Section detection
    sections = {
        "Summary": any(w in lower_resume for w in ["summary", "objective", "profile", "about me"]),
        "Skills": any(w in lower_resume for w in ["skills", "technologies", "tools", "technical"]),
        "Experience": any(w in lower_resume for w in ["experience", "work history", "employment", "professional"]),
        "Education": any(w in lower_resume for w in ["education", "degree", "university", "college", "bachelor", "master"]),
    }
    section_score = (sum(sections.values()) / 4) * 100

    # Format issues detection
    format_issues = []
    if "|" in req.resume_text:
        format_issues.append({"issue": "Tables detected", "fix": "Replace pipe-separated tables with plain text lists"})
    if req.resume_text.count("\t") > 5:
        format_issues.append({"issue": "Tab-based columns detected", "fix": "Use simple line breaks instead of tab alignment"})
    if any(c in req.resume_text for c in ["★", "●", "◆", "►", "✦"]):
        format_issues.append({"issue": "Special/decorative characters found", "fix": "Replace with standard bullet points (-) or remove"})
    if len(req.resume_text) < 300:
        format_issues.append({"issue": "Resume seems very short", "fix": "Expand your experience and skills sections"})

    format_score = max(0, 100 - len(format_issues) * 20)

    # Keyword scoring
    job_skills = extract_skills(req.job_description)
    resume_skills = extract_skills(req.resume_text)
    matched_skills = list(set(resume_skills) & set(job_skills))
    missing_skills = list(set(job_skills) - set(resume_skills))
    keyword_score = (len(matched_skills) / len(job_skills) * 100) if job_skills else 0

    # ATS Score
    ats_score = round(keyword_score * 0.4 + section_score * 0.3 + format_score * 0.3, 2)

    # Bullet rewriting
    lines = req.resume_text.split("\n")
    rewritten_bullets = []
    weak_starters = ["did", "helped", "worked", "assisted", "responsible for", "involved in", "part of", "contributed to"]
    action_rewrites = {
        "did": "Executed",
        "helped": "Collaborated to",
        "worked": "Developed",
        "assisted": "Supported and accelerated",
        "responsible for": "Spearheaded",
        "involved in": "Contributed directly to",
        "part of": "Integral team member delivering",
        "contributed to": "Directly drove",
    }
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 20:
            lower_line = stripped.lower()
            for weak in weak_starters:
                if lower_line.startswith(weak):
                    action = action_rewrites.get(weak, "Led")
                    rest = stripped[len(weak):].strip()
                    rewritten = f"{action} {rest}, resulting in measurable impact on team productivity and project delivery."
                    rewritten_bullets.append({"original": stripped, "rewritten": rewritten})
                    break

    # Feedback
    if ats_score >= 80:
        overall_feedback = "Excellent! Your resume is well-optimized for ATS systems."
        suggestions = ["Consider adding more specific metrics to quantify your achievements", "Tailor keywords to each specific job posting", "Keep formatting clean and consistent"]
    elif ats_score >= 60:
        overall_feedback = "Good resume but needs keyword improvements to pass more ATS filters."
        suggestions = ["Add more industry-specific keywords from the job description", "Ensure all major sections are clearly labeled", "Quantify your achievements with numbers and percentages"]
    else:
        overall_feedback = "Significant improvements needed for ATS compatibility."
        suggestions = ["Add missing keywords from the job description", "Structure resume with clear section headers", "Remove tables, graphics, and special characters", "Expand on your skills and experience sections"]

    skill_gaps = build_skill_gaps(missing_skills)

    total_ms = round((time.time() - start) * 1000, 2)
    logger.info(f"[AI] /ats-analyze completed in {total_ms}ms, ATS score: {ats_score}")

    return {
        "ats_score": ats_score,
        "keyword_score": round(keyword_score, 2),
        "section_score": round(section_score, 2),
        "format_score": round(format_score, 2),
        "sections": sections,
        "format_issues": format_issues,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_gaps": skill_gaps,
        "rewritten_bullets": rewritten_bullets,
        "overall_feedback": overall_feedback,
        "suggestions": suggestions,
        "processing_time_ms": total_ms,
    }

@app.get("/skill-resources")
def skill_resources():
    return {"resources": RESOURCE_MAP, "count": len(RESOURCE_MAP)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
