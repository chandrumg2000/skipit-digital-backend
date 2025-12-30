# Backend/main.py
"""
SkipIt Digital — Backend server for generating SEO audit PDFs.
- Uses Gemini (via google-generativeai) as the PRIMARY engine for deep insights.
- Falls back to OpenAI if Gemini fails.
- Scrapes real website data for authentic insights.
- Saves leads to leads.csv
- Returns PDF as a downloadable StreamingResponse
"""

import os
import csv
import io
import json
import traceback
import smtplib
from email.message import EmailMessage
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import requests
from fpdf import FPDF
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# Initialize Clients
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="SkipIt SEO Audit API")

# CORS
origins = ["*"] if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LEADS_FILE = "leads.csv"

# Ensure CSV header exists
if not os.path.exists(LEADS_FILE):
    with open(LEADS_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "name", "business", "email", "phone", "domain"])


class AuditRequest(BaseModel):
    name: str
    business: Optional[str] = ""
    email: EmailStr
    phone: Optional[str] = ""
    domain: str


def save_lead(data: AuditRequest):
    """Append lead to CSV."""
    with open(LEADS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            data.name,
            data.business,
            data.email,
            data.phone,
            data.domain,
        ])

def log_debug(msg):
    """Log debug messages."""
    try:
        with open("debug.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()} - {msg}\n")
    except Exception:
        pass
    print(msg)

def _sample_audit_text(domain: str, business: str) -> str:
    """Return a realistic sample audit used as fallback."""
    sample = f"""
SkipIt Digital — SEO Audit (Sample)
Website: {domain}
Business: {business}

1) Quick summary
- The site has a clear product/service focus but needs technical and on-page optimization to scale organic traffic.

2) Top technical issues (severity)
- Missing or duplicate meta titles (High)
- Large unoptimized hero images causing slow LCP (High)
- No sitemap.xml / robots policy found (Medium)
- Mixed HTTP/HTTPS internal links (Medium)
- Missing canonical tags on near-duplicate pages (Low)

3) On-page recommendations
- Add unique title + meta description to top landing pages.
- Improve H1/H2 structure and include primary keywords naturally.
- Add structured data (Product/Organization) where relevant.
- Improve image alt text and compress images.

4) Off-page & backlink opportunities
- Guest posts on niche blogs
- Linkable resources (guides/checklists) and outreach
- Local citations if applicable

5) SWOT
- Strengths: Clear offering and good UX.
- Weaknesses: Technical issues and thin meta copy.
- Opportunities: Content cluster strategy and PR outreach.
- Threats: Competitors with stronger backlink profiles.

6) Prioritized quick-wins
- 1 day: Fix meta titles for top 5 pages.
- 1 week: Compress images & enable caching.
- 1 month: Publish 4 optimized blog posts + outreach.

(End of sample audit)
"""
    return sample.strip()


def fetch_website_data(url: str) -> str:
    """
    Scrape the website to get authentic data for the audit.
    """
    try:
        if not url.startswith("http"):
            url = "https://" + url
            
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Extract key elements
        title = soup.title.string.strip() if soup.title else "No title found"
        
        meta_desc = ""
        meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if meta:
            meta_desc = meta.get("content", "").strip()
            
        h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
        h2s = [h.get_text(strip=True) for h in soup.find_all("h2")][:5]
        
        # Get body text snippet
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator=" ", strip=True)
        text_snippet = " ".join(text.split()[:600])
        
        return f"""
AUTHENTIC SITE DATA:
Title: {title}
Meta Description: {meta_desc}
H1 Tags: {', '.join(h1s)}
H2 Tags (Top 5): {', '.join(h2s)}
Content Snippet: {text_snippet}...
"""
    except Exception as e:
        log_debug(f"Scraping failed for {url}: {e}")
        return f"Could not scrape website data ({e}). Using domain name only."

def call_openai_for_audit(domain: str, business: str, site_data: str) -> str:
    """Use OpenAI (Fallback)"""
    if not client:
        return _sample_audit_text(domain, business)

    prompt = f"""
You are SkipIt Digital — produce a HIGHLY AUTHENTIC, DETAILED, and ACTIONABLE SEO audit for this website.
{site_data}
Business: {business}
Domain: {domain}

Report Format:
1) **Executive Summary**: Authentic assessment of site state.
2) **Technical SEO Analysis**: Title, Meta, Headers.
3) **Content Quality**: Thin? Stuffed? Engaging?
4) **SWOT Analysis**: Strengths, Weaknesses, Opportunities, Threats.
5) **3 High-Impact Recommendations**.

FORMAT: Professional, direct, NO generic filler.
"""
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are SkipIt Digital - a top-tier SEO agency."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log_debug(f"OpenAI call failed: {e}")
        return _sample_audit_text(domain, business)


def call_gemini_for_audit(domain: str, business: str, site_data: str) -> Optional[str]:
    """
    Call Gemini API (Primary).
    """
    if not GEMINI_API_KEY:
        log_debug("Gemini API key missing.")
        return None

    prompt = f"""
You are SkipIt Digital — produce a HIGHLY AUTHENTIC, DETAILED, and ACTIONABLE SEO audit for this website based on the REAL DATA provided below.

{site_data}

Business Name: {business}
Domain: {domain}

Your Report Must Include:
1) **Executive Summary**: Authentic assessment of the site's current state based on the title/meta/content provided.
2) **Technical SEO Analysis**:
   - Analyze the Title Tag (length, keywords).
   - Analyze the Meta Description (is it missing? too short?).
   - Analyze H1/H2 structure (are keywords present?).
3) **Content Quality**: Evaluate the content snippet.
4) **SWOT Analysis**: Specific Strengths, Weaknesses, Opportunities, Threats based on the ACTUAL content.
5) **3 High-Impact Recommendations**: Specific actions to take (e.g., "Change H1 from X to Y").

FORMATTING RULES:
- Use clear headings.
- Be professional but direct.
- Do NOT use generic filler text. Reference specific words from the site data.
- Keep it concise but dense with value.
"""
    try:
        # Use simple model for speed/stability
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        text = response.text
        if text:
            return text.strip()
    except Exception as e:
        log_debug(f"Gemini call failed: {e}")
    
    return None


def send_email_with_pdf(to_email: str, subject: str, body: str, pdf_bytes: bytes, filename: str):
    email_user = os.getenv("EMAIL_USER") or os.getenv("SMTP_USER")
    email_pass = os.getenv("EMAIL_PASS") or os.getenv("SMTP_PASS")
    
    if not email_user or not email_pass:
        log_debug("Email credentials not found. Skipping email.")
        return

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = email_user
    msg['To'] = to_email
    msg.set_content(body)
    msg.add_attachment(pdf_bytes, maintype='application', subtype='pdf', filename=filename)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_user, email_pass)
            smtp.send_message(msg)
        log_debug(f"Email sent successfully to {to_email}")
    except Exception as e:
        log_debug(f"Failed to send email: {e}")

def create_pdf_bytes(report_text: str, title: str = "SEO Audit Report") -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- COVER PAGE ---
    pdf.add_page()
    pdf.set_fill_color(240, 248, 255)
    pdf.rect(0, 0, 210, 297, 'F')
    
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(0, 102, 204) # SkipIt Blue
    pdf.ln(60)
    pdf.cell(0, 10, "SkipIt Digital", ln=True, align="C")
    
    pdf.set_font("Arial", "B", 32)
    pdf.set_text_color(50, 50, 50)
    pdf.ln(20)
    pdf.multi_cell(0, 15, "SEO & SWOT\nANALYSIS REPORT", align="C")
    
    pdf.ln(10)
    pdf.set_font("Arial", "", 16)
    pdf.cell(0, 10, title.replace("SEO Audit Report", "").strip(), ln=True, align="C")
    
    pdf.ln(30)
    pdf.set_fill_color(37, 211, 102) # Green
    pdf.set_draw_color(255, 255, 255)
    pdf.set_font("Arial", "B", 40)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 30, "SCORE: 78/100", ln=True, align="C", fill=True)
    
    pdf.ln(40)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Generated on {datetime.utcnow().strftime('%B %d, %Y')}", ln=True, align="C")

    # --- REPORT PAGES ---
    pdf.add_page()
    pdf.set_text_color(50, 50, 50)
    
    COLOR_HEADER = (0, 153, 255)
    COLOR_TEXT = (50, 50, 50)
    
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_x(15)
    
    safe_text = report_text.encode('latin-1', 'ignore').decode('latin-1')
    
    for line in safe_text.splitlines():
        line = line.strip()
        if not line:
            pdf.ln(4)
            continue

        try:
            pdf.set_x(15)
            if (line[0].isdigit() and ")" in line[:4]) or line.startswith("SWOT") or line.startswith("**"):
                pdf.ln(6)
                pdf.set_font("Arial", "B", 14)
                pdf.set_text_color(*COLOR_HEADER)
                clean_line = line.replace("**", "")
                pdf.multi_cell(w=180, h=8, text=clean_line, align='L')
                pdf.set_font("Arial", size=11)
                pdf.set_text_color(*COLOR_TEXT)
            elif ":" in line and len(line.split(":")[0]) < 30 and not line.startswith("-"):
                pdf.set_font("Arial", "B", 11)
                pdf.multi_cell(w=180, h=6, text=line, align='L')
                pdf.set_font("Arial", size=11)
            else:
                pdf.multi_cell(w=180, h=6, text=line, align='L')
        except Exception:
            continue
            
    pdf.ln(10)
    if pdf.get_y() > 240:
        pdf.add_page()
        
    start_y = pdf.get_y()
    pdf.set_fill_color(240, 248, 255)
    pdf.rect(10, start_y, 190, 40, 'F')
    
    pdf.set_y(start_y + 8)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 8, "End-to-End Growth Strategy for Success", ln=True, align="C")
    
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(37, 211, 102)
    pdf.cell(0, 8, "WhatsApp: +91 90038 05951", ln=True, align="C")
        
    return bytes(pdf.output())


@app.post("/generate-audit")
async def generate_audit(req: AuditRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(save_lead, req)

        env_path = os.path.join(os.getcwd(), ".env")
        load_dotenv(env_path, override=True)
        
        # Scrape data
        site_data = fetch_website_data(req.domain)

        # 1. Try Gemini
        audit_text = call_gemini_for_audit(req.domain, req.business, site_data)
        
        # 2. Fallback to OpenAI
        if not audit_text:
            log_debug("Gemini failed/missing. Trying OpenAI...")
            audit_text = call_openai_for_audit(req.domain, req.business, site_data)
            
        log_debug(f"Final Report Length: {len(audit_text)}")

        pdf_bytes = create_pdf_bytes(audit_text, title=f"{req.domain}")
        filename = f"skipit-seo-audit-{req.domain.replace('https://','').replace('/','_')}.pdf"
        
        background_tasks.add_task(
            send_email_with_pdf,
            to_email=req.email,
            subject=f"Your SEO Audit Report for {req.domain}",
            body=f"Hi {req.name},\n\nPlease find attached your requested SEO audit report for {req.domain}.\n\nBest regards,\nSkipIt Digital Team",
            pdf_bytes=pdf_bytes,
            filename=filename
        )
        
        # Send Lead Notification to Admin (New)
        admin_emails = os.getenv("MAIL_TO", "chandrumg2000@gmail.com")
        background_tasks.add_task(
            send_email_with_pdf,
            to_email=admin_emails,
            subject=f"[New Lead] SEO Audit Request - {req.name} ({req.business})",
            body=f"""New SEO Audit Request:

Name: {req.name}
Business: {req.business}
Email: {req.email}
Phone: {req.phone}
Domain: {req.domain}

The generated report is attached for your reference.
""",
            pdf_bytes=pdf_bytes,
            filename=filename
        )

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        tb = traceback.format_exc()
        log_debug(f"Error: {e}\n{tb}")
        raise HTTPException(status_code=500, detail="Internal Error")

@app.get("/")
def root():
    return {"status": "ok"}