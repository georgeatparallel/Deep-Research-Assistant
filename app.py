import json
import time
import re
import os
import asyncio
import uuid
import aiohttp
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from llama_cpp import Llama
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def section_for_format(fmt: str) -> list[str]:
    fmt = (fmt or "").strip().lower()
    if fmt == "executive":
        return ["EXECUTIVE SUMMARY"]
    if fmt == "detailed":
        return [
            "INTRODUCTION",
            "DETAILED ANALYSIS",
            "CURRENT TRENDS AND DEVELOPMENTS",
            "IMPLICATIONS AND RECCOMENDATIONS",
            "CONCLUSION",
        ]
    if fmt == "academic":
        return [
            "ABSTRACT",
            "INTRODUCTION",
            "METHODOLOGY",
            "FINDINGS",
            "DISCUSSIONS",
            "CONCLUSION",
        ]
    if fmt == "presentation":
        return [
            "OVERVIEW",
            "KEY INSIGHTS",
            "RECOMMENDATIONS",
            "NEXT STEPS",
            "CONCLUSION",
        ]
    return ["INTRODUCTION", "DETAILED ANALYSIS", "CONCLUSION"]


def extract_final_block(text: str) -> str:
    m = re.search(r"<final>([\s\S]*?)</final>", text, flags=re.IGNORECASE)
    cleaned_text = m.group(1).strip() if m else text

    preamble_patterns = [
        r"^(?:note:|okay,|hmm,|internal|let me|i (?:will|i'll)|as an ai|thinking|plan:|here is your report|the following is|i have prepared|i am presnting|based on the provided information|below is the report|i hope this satisfies your requirements|this report outlines|this is the final report).*?$",
        r"^(?:Here is the report|I have compiled the report|The report is provided below|This is the requested report).*?$",
        r"^(?:Please find the report below|Here's the report).*?$",
    ]
    for pattern in preamble_patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE | re.MULTILINE).strip()

    cleaned_text = re.sub(r"(?m)^\s*[-*.]\s+", "", cleaned_text)
    cleaned_text = re.sub(r"[#`*_]{1,3}", "", cleaned_text)

    headers = [
        "EXECUTIVE SUMMARY", "INTRODUCTION", "DETAILED ANALYSIS",
        "CURRENT TRENDS AND DEVELOPMENTS", "IMPLICATIONS AND RECCOMENDATIONS",
        "CONCLUSION", "ABSTRACT", "METHODOLOGY", "FINDINGS", "DISCUSSION",
        "OVERVIEW", "KEY INSIGHTS", "RECOMMENDATIONS", "NEXT STEPS",
    ]
    sorted_headers = sorted(headers, key=len, reverse=True)
    first_pos = -1
    for h in sorted_headers:
        match = re.search(r"\b" + re.escape(h) + r"\b", cleaned_text, flags=re.IGNORECASE)
        if match:
            if first_pos == -1 or match.start() < first_pos:
                first_pos = match.start()
    if first_pos >= 0:
        cleaned_text = cleaned_text[first_pos:].strip()
    return cleaned_text


@dataclass
class ResearchConfig:
    model_path: str = "C:/Users/lenovo/OneDrive/Desktop/OneDrive/DeepSearch AgenticAI Project/models/Jan-v1-4B-Q4_K_M.gguf"
    max_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: float = 20
    context_length: int = 4096
    search_api_key: str = os.getenv("SERPER_API_KEY", "")
    search_engine: str = os.getenv("SEARCH_ENGINE", "serper").strip().lower()


class DeepResearchAssistant:
    def __init__(self, config=ResearchConfig):
        self.config = config
        self.llm = None
        self.demo_mode = False
        self.parallel_session_id = uuid.uuid4().hex

    def load_model(self):
        try:
            if not os.path.exists(self.config.model_path):
                print(f"Model file not found: {self.config.model_path}")
                return False
            file_size_gb = os.path.getsize(self.config.model_path) / (1024 ** 3)
            if file_size_gb < 1.0:
                print(f"Model file too small ({file_size_gb:.1f} GB)")
                return False
            self.llm = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.context_length,
                verbose=False,
                n_threads=max(1, min(4, os.cpu_count() // 2)),
                n_gpu_layers=0,
                use_mmap=True,
                use_mlock=False,
                n_batch=128,
                f16_kv=True,
            )
            test = self.llm("Hi", max_tokens=3, temperature=0.1, echo=False)
            ok = bool(test and "choices" in test)
            print("Model loaded" if ok else "Model loaded but generation failed")
            return ok
        except Exception as e:
            print(f"Model loading failed: {e}")
            return False

    def generate_response(self, prompt: str, max_tokens: int = None, extra_stops: Optional[List[str]] = None) -> str:
        if not self.llm:
            return "Model not loaded."
        stops = ["</s>", "<|im_end|>", "<|endoftext|>"]
        if extra_stops:
            stops.extend(extra_stops)
        mt = max_tokens or self.config.max_tokens
        try:
            resp = self.llm(
                prompt,
                max_tokens=mt,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                stop=stops,
                echo=False,
            )
            return resp["choices"][0]["text"].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def search_web(self, query: str, num_results: int = 10) -> List[Dict]:
        if self.config.search_engine == "parallel":
            from parallel_search import search_parallel
            return await search_parallel(query, num_results, self.parallel_session_id)
        if self.config.search_engine != "serper":
            raise ValueError("SEARCH_ENGINE must be 'serper' or 'parallel'.")
        if self.config.search_api_key:
            return await self.search_serper(query, num_results)
        return []

    async def search_serper(self, query: str, num_results: int) -> List[Dict]:
        url = "https://google.serper.dev/search"
        payload = {"q": query, "num": num_results}
        headers = {"X-API-KEY": self.config.search_api_key, "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                data = await response.json()
                results = []
                for item in data.get("organic", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "web",
                    })
                return results

    def generate_search_queries(self, topic: str, focus_area: str, depth: str) -> List[str]:
        counts = {"surface": 5, "moderate": 8, "deep": 15, "comprehensive": 25}
        n = counts.get(depth, 8)
        base = [
            f"{topic} overview",
            f"{topic} recent developments",
            f"{topic} academic studies",
            f"{topic} case studies",
            f"{topic} policy and regulation",
            f"{topic} technical approaches",
            f"{topic} market analysis",
            f"{topic} statistics and data",
        ]
        return base[:n]

    def synthesize_search(self, topic: str, search_results: List[Dict], focus_area: str, report_format: str) -> str:
        context_lines = []
        for i, result in enumerate(search_results[:20]):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            context_lines.append(f"Source {i+1} Title: {title}\n Source {i+1} Summary: {snippet}")
        context = "\n".join(context_lines)
        sections = section_for_format(report_format)
        sections_text = "\n".join(sections)
        synthesis_prompt = f"""
You are an expert research analyst. Write the final, published report on: "{topic}" for a professional, real world audience.
***CRITICAL INSTRUCTIONS:***
- Your entire response MUST be the final report, wrapped **EXACTLY** inside <final> and </final> tags.
- DO NOT output any text, thoughts, or commentary BEFORE the <final> tag or AFTER the </final> tag.
- DO NOT include any conversational filler, internal thoughts, or commentary about the generation process.
- DO NOT use markdown formatting (e.g., #, ##, *, -).
- DO NOT use bullet points or lists.
- Maintain a formal, academic/professional tone throughout.
- Ensure the report is complete and self-contained.
- Include the following section headers, in this order, and no others:
{sections_text}
Guidance:
- Base your writing strictly on the Research Notes provided below.
Research Notes:
{context}
Now begin the report immediately.

<final>
Introduction
"""
        raw = self.generate_response(synthesis_prompt, max_tokens=1800, extra_stops=["</final>"])
        print("------ RAW MODEL OUTPUT ------")
        print(raw)
        print("--------------------------------")
        final_report = extract_final_block(raw)
        final_report = re.sub(r"(?m)^\s*[-*•]\s+", "", final_report)
        final_report = re.sub(r"[#`*_]{1,3}", "", final_report)
        first = next((h for h in sections if h in final_report), None)
        if first:
            final_report = final_report[final_report.find(first):].strip()
        return final_report



_config = ResearchConfig()
_assistant = DeepResearchAssistant(_config)
_model_loaded = _assistant.load_model()



def run_research(topic, depth, focus, timeframe, report_format, progress=gr.Progress(track_tqdm=False)):
    if not _model_loaded:
        return (
            "❌ Model not loaded. Check terminal logs for details.",
            "",
            None,
            None,
            gr.update(visible=False),
        )
    if not topic.strip():
        return (
            "❌ Please enter a research topic.",
            "",
            None,
            None,
            gr.update(visible=False),
        )

    try:
        progress(0.10, desc="Generating search queries…")
        queries = _assistant.generate_search_queries(topic, focus, depth)

        progress(0.30, desc="Searching sources…")
        all_results = []
        for i, query in enumerate(queries):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(_assistant.search_web(query, 5))
            all_results.extend(results)
            loop.close()
            frac = 0.30 + ((i + 1) / max(1, len(queries))) * 0.35
            progress(frac, desc=f"Searched {i+1}/{len(queries)} queries…")
            time.sleep(0.05)

        progress(0.75, desc="Synthesising report…")
        report = _assistant.synthesize_search(topic, all_results, focus, report_format)

        progress(1.0, desc="Done!")

        # ── Build sources HTML ──
        sources_html = ""
        for src in all_results[:12]:
            sources_html += f"""
<div class="source-card">
  <h4>{src['title']}</h4>
  <p>{src['snippet']}</p>
  <a href="{src['url']}" target="_blank">{src['url']}</a>
</div>"""

        # ── Plain-text export ──
        report_text = f"Research Report: {topic}\n\nGenerated: {datetime.now()}\n\n{report}"
        txt_path = save_txt(report_text)

        # ── JSON export ──
        export_data = {
            "topic": topic,
            "report": report,
            "sources": all_results,
            "queries": queries,
            "config": {"depth": depth, "focus": focus, "timeframe": timeframe, "format": report_format},
            "timestamp": str(datetime.now()),
        }
        json_str = json.dumps(export_data, default=str, indent=2)
        json_path = save_json(json_str)

        return (
            report,
            sources_html,
            txt_path,
            json_path,
            gr.update(visible=True),
        )

    except Exception as e:
        return (
            f"❌ Research failed: {str(e)}",
            "",
            None,
            None,
            gr.update(visible=False),
        )


def save_txt(report_text: str) -> str:
    import tempfile
    tmp_dir = tempfile.gettempdir()
    fname = f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path = os.path.join(tmp_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return path


def save_json(json_str: str) -> str:
    import tempfile
    tmp_dir = tempfile.gettempdir()
    fname = f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(tmp_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)
    return path


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

body, .gradio-container {
    background: #0a0118 !important;
    background-image:
        radial-gradient(at 0% 0%, rgba(255,0,255,0.15) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(138,43,226,0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(255,20,147,0.15) 0px, transparent 50%),
        radial-gradient(at 0% 100%, rgba(75,0,130,0.15) 0px, transparent 50%) !important;
    font-family: 'Inter', sans-serif !important;
    color: #d4c5e8 !important;
}

/* Grid overlay */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        linear-gradient(90deg, rgba(255,0,255,0.03) 1px, transparent 1px),
        linear-gradient(0deg,  rgba(255,0,255,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* ── Header ── */
.main-header {
    background: rgba(20,10,30,0.6);
    backdrop-filter: blur(20px) saturate(180%);
    padding: 2.5rem;
    border-radius: 28px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 0 80px rgba(255,0,255,0.2), 0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.1);
    border: 1px solid rgba(255,0,255,0.2);
    overflow: hidden;
    position: relative;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,0,255,0.1), transparent);
    animation: shimmer 3s infinite;
}
@keyframes shimmer { 0%{left:-100%} 100%{left:200%} }

.main-header h1 {
    background: linear-gradient(135deg, #ff0080 0%, #ff00ff 25%, #8000ff 50%, #00ffff 75%, #ff0080 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    animation: gradient 3s ease infinite;
    filter: drop-shadow(0 0 30px rgba(255,0,255,0.5));
    font-family: 'Space Grotesk', sans-serif;
}
@keyframes gradient { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
.main-header p { color: #e0b3ff !important; font-size: 1.1rem; font-weight: 500; }

/* ── Section titles ── */
h2, h3, .gr-form > label, .label-wrap span {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #ff80ff !important;
}

/* ── Inputs / Textareas / Dropdowns ── */
textarea, input[type="text"], .gr-box, select, .gr-dropdown {
    background: rgba(10,5,20,0.6) !important;
    border: 2px solid rgba(255,0,255,0.2) !important;
    border-radius: 16px !important;
    color: #ffffff !important;
    padding: 1rem 1.25rem !important;
    font-family: 'Inter', sans-serif !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #ff00ff !important;
    box-shadow: 0 0 0 4px rgba(255,0,255,0.1), 0 0 30px rgba(255,0,255,0.3) !important;
    transform: translateY(-2px);
}

/* ── Primary button ── */
button.primary, #run-btn {
    width: 100%;
    padding: 1.25rem 2rem !important;
    background: linear-gradient(135deg, #ff0080 0%, #ff00ff 50%, #8000ff 100%) !important;
    background-size: 200% auto !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 0 40px rgba(255,0,255,0.5), 0 4px 20px rgba(0,0,0,0.5) !important;
    transition: all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    cursor: pointer;
}
button.primary:hover, #run-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 0 60px rgba(255,0,255,0.8), 0 10px 30px rgba(0,0,0,0.6) !important;
}

/* ── Report output ── */
.result-card {
    background: linear-gradient(135deg, rgba(255,0,255,0.05) 0%, rgba(138,43,226,0.05) 100%);
    padding: 2rem;
    border-radius: 20px;
    border: 2px solid rgba(255,0,255,0.2);
    margin-bottom: 2rem;
    box-shadow: 0 0 30px rgba(255,0,255,0.1);
    white-space: pre-wrap;
    color: #d4c5e8;
    font-family: 'Inter', sans-serif;
    line-height: 1.8;
}

/* ── Source cards ── */
.source-card {
    background: rgba(20,10,30,0.6);
    backdrop-filter: blur(10px);
    padding: 1.25rem;
    border-radius: 16px;
    border: 1px solid rgba(255,0,255,0.15);
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.source-card:hover {
    border-color: #ff00ff;
    box-shadow: 0 0 30px rgba(255,0,255,0.2);
    transform: translateX(5px);
}
.source-card h4 { color: #ff80ff; margin: 0 0 0.5rem 0; }
.source-card p  { color: #b39ddb; margin: 0 0 0.5rem 0; }
.source-card a  { color: #00ffff; text-decoration: none; }

/* ── Download buttons ── */
.gr-download button, button.secondary {
    background: rgba(255,0,255,0.1) !important;
    color: #ff80ff !important;
    border: 2px solid #ff00ff !important;
    border-radius: 14px !important;
    padding: 0.9rem 1.5rem !important;
    font-weight: 600 !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease !important;
    cursor: pointer;
}
.gr-download button:hover, button.secondary:hover {
    background: linear-gradient(135deg, #ff00ff 0%, #8000ff 100%) !important;
    color: white !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 0 25px rgba(255,0,255,0.5) !important;
}

/* ── Progress bar ── */
.progress-bar-wrap .progress-bar { background: linear-gradient(90deg,#ff0080,#ff00ff,#8000ff) !important; }

/* ── Accordion ── */
.gr-accordion { border: 1px solid rgba(255,0,255,0.15) !important; border-radius: 16px !important; }
.gr-accordion > .label-wrap { color: #ff80ff !important; font-weight: 600 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: rgba(10,5,20,0.5); }
::-webkit-scrollbar-thumb { background: linear-gradient(135deg,#ff00ff,#8000ff); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: linear-gradient(135deg,#ff0080,#ff00ff); }
"""

HEADER_HTML = """
<div class="main-header">
    <h1>🔍 Deep Research Assistant</h1>
    <p>AI-powered research made simple and beautiful</p>
</div>
"""

DEPTH_CHOICES = ["surface", "moderate", "deep", "comprehensive"]
DEPTH_LABELS  = {
    "surface": "Surface (5–8 sources)",
    "moderate": "Moderate (10–15 sources)",
    "deep": "Deep Dive (20–30 sources)",
    "comprehensive": "Comprehensive (40+ sources)",
}
FOCUS_CHOICES = ["general", "academic", "business", "technical", "policy"]
TIME_CHOICES  = ["current", "recent", "comprehensive"]
TIME_LABELS   = {"current": "Current (≤6 months)", "recent": "Recent (≤2 years)", "comprehensive": "All time"}
FMT_CHOICES   = ["executive", "detailed", "academic", "presentation"]
FMT_LABELS    = {
    "executive": "Executive Summary",
    "detailed": "Detailed Analysis",
    "academic": "Academic Style",
    "presentation": "Presentation Format",
}

with gr.Blocks(css=CUSTOM_CSS, title="Deep Research Assistant") as demo:

    gr.HTML(HEADER_HTML)

    if _assistant.config.search_engine == "serper" and not _assistant.config.search_api_key:
        gr.HTML('<div style="background:linear-gradient(135deg,#ff6b00,#ff0080);color:white;padding:1rem 1.5rem;border-radius:14px;margin-bottom:1rem;">⚠️ Web search is disabled. Set SERPER_API_KEY or enable Parallel with SEARCH_ENGINE=parallel and its optional dependency.</div>')

    # ── Config section ──
    gr.Markdown("## ⚙️ Research Configuration")

    research_topic = gr.Textbox(
        label="Research Topic",
        placeholder="e.g., Impact of artificial intelligence on healthcare efficiency and patient outcomes",
        lines=4,
    )

    with gr.Row():
        with gr.Column():
            research_depth = gr.Dropdown(
                choices=[(DEPTH_LABELS[c], c) for c in DEPTH_CHOICES],
                value="moderate",
                label="Research Depth",
            )
            focus_area = gr.Dropdown(
                choices=[(c.title(), c) for c in FOCUS_CHOICES],
                value="general",
                label="Focus Area",
            )
        with gr.Column():
            time_frame = gr.Dropdown(
                choices=[(TIME_LABELS[c], c) for c in TIME_CHOICES],
                value="recent",
                label="Time Frame",
            )
            report_format = gr.Dropdown(
                choices=[(FMT_LABELS[c], c) for c in FMT_CHOICES],
                value="detailed",
                label="Report Format",
            )

    run_btn = gr.Button("✨ Start Deep Research", variant="primary", elem_id="run-btn")

    # ── Output section ──
    gr.Markdown("## 📊 Research Report")

    report_out = gr.HTML(label="Report")

    with gr.Accordion("📚 Sources", open=False):
        sources_out = gr.HTML()

    with gr.Row(visible=False) as export_row:
        gr.Markdown("### 💾 Export")

    with gr.Row(visible=False) as dl_row:
        dl_txt  = gr.DownloadButton("📄 Download Text",  value=None, visible=True)
        dl_json = gr.DownloadButton("📋 Download JSON", value=None, visible=True)

    # ── Wire up ──
    def _run(topic, depth, focus, timeframe, fmt, progress=gr.Progress(track_tqdm=False)):
        report, sources_html, txt_path, json_path, _ = run_research(
            topic, depth, focus, timeframe, fmt, progress
        )
        report_html = (
            f'<div class="result-card">{report}</div>'
            if report and not report.startswith("❌")
            else f'<p style="color:#ff6b6b">{report}</p>'
        )
        show = gr.update(visible=txt_path is not None)
        return (
            report_html,
            sources_html,
            gr.update(value=txt_path),   # sets DownloadButton file
            gr.update(value=json_path),  # sets DownloadButton file
            show,                        # export_row visibility
            show,                        # dl_row visibility
        )

    run_btn.click(
        fn=_run,
        inputs=[research_topic, research_depth, focus_area, time_frame, report_format],
        outputs=[report_out, sources_out, dl_txt, dl_json, export_row, dl_row],
    )


if __name__ == "__main__":
    demo.launch(share=False)
