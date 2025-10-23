# agents/pdf_report_generator.py
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import json
import re

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    Flowable, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

# ---- Open LLM (OpenAI SDK v1) ----
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

from state_schema import PipelineState


# ===================== 유틸 =====================

def _safe_get(d: dict, path: List, default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def _fmt_float(x: Optional[float], digits: int = 2):
    if x is None:
        return "-"
    try:
        return f"{x:.{digits}f}"
    except Exception:
        return "-"

def _nn(x, default="-"):
    return x if (x is not None and x != "") else default

def _wrap(s: str, width: int = 110):
    # 강제 줄바꿈은 ReportLab Paragraph가 처리(CJK wrap)
    return s or ""

def _grade_color(grade: str) -> colors.Color:
    g = (grade or "").upper()
    if g in ("A+", "A"):
        return colors.HexColor("#27ae60")
    if g == "B":
        return colors.HexColor("#f39c12")
    if g in ("C", "D"):
        return colors.HexColor("#e74c3c")
    return colors.HexColor("#7f8c8d")

def _risk_color(level: str) -> str:
    m = {"CRITICAL": "#e74c3c", "HIGH": "#e67e22", "MEDIUM": "#f1c40f", "LOW": "#2ecc71", "UNKNOWN": "#7f8c8d"}
    return m.get((level or "").upper(), "#7f8c8d")

def _severity_emoji(sev: str) -> str:
    m = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    return m.get((sev or "").lower(), "⚪")


class HRLine(Flowable):
    """수평 라인 구분자"""
    def __init__(self, width=450, thickness=0.5, color=colors.HexColor("#bdc3c7")):
        Flowable.__init__(self)
        self.width = width
        self.thickness = thickness
        self.color = color
    def draw(self):
        self.canv.saveState()
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)
        self.canv.restoreState()


# ======== LLM 출력 정리 ========

def _sanitize_markdown_headers(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = t.replace("—", "-").replace("–", "-").replace("|", "")
    return t.strip()

def _split_to_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    lines = text.splitlines()
    paras: List[str] = []
    buf: List[str] = []
    bullet_re = re.compile(r"^\s*(?:[-•]|\d+\.)\s+")
    for ln in lines:
        ln = ln.rstrip()
        if not ln:
            if buf:
                paras.append(" ".join(buf).strip())
                buf = []
            continue
        if bullet_re.match(ln):
            if buf:
                paras.append(" ".join(buf).strip())
                buf = []
            paras.append(ln.strip())
        else:
            buf.append(ln.strip())
    if buf:
        paras.append(" ".join(buf).strip())
    return paras

def _render_llm_text(text: str, styles: Dict) -> List[Flowable]:
    flows: List[Flowable] = []
    text = _sanitize_markdown_headers(text)
    paras = _split_to_paragraphs(text)
    bullet_re = re.compile(r"^\s*(?:[-•]|\d+\.)\s+(.*)$")
    for p in paras:
        m = bullet_re.match(p)
        if m:
            flows.append(Paragraph(f"• {m.group(1)}", styles["KBullet"]))
        else:
            flows.append(Paragraph(_wrap(p, 120), styles["KBody"]))
    return flows


# ===================== 메인 클래스 =====================

class PDFReportGenerator:
    """
    - CJK 줄바꿈 적용(문장 중간 잘림 방지)
    - TOC 항목 글자 크게(KTOCItem)
    - Appendix 5.3 '평가 기준 출처' (페이지 정보 포함)
    - Appendix 5.4 'RAG 트레이스(검색 쿼리 & 근거 페이지)' 추가
      · state["criteria_rag_traces"] (구조화) 또는 state["criteria_rag_log_text"] (원문 로그) 사용
    """

    def __init__(self, model_name: Optional[str] = None):
        self.korean_font = "Helvetica"  # fallback
        self._setup_korean_fonts()
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.client = None
        if _OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.client = OpenAI()
            except Exception:
                self.client = None

    # ---------- Font ----------
    def _setup_korean_fonts(self):
        try:
            font_paths = [
                "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
                "C:\\Windows\\Fonts\\malgun.ttf",
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            ]
            for fp in font_paths:
                if os.path.exists(fp):
                    pdfmetrics.registerFont(TTFont("KoreanFont", fp))
                    self.korean_font = "KoreanFont"
                    print(f"✅ 한글 폰트 로드 완료: {fp}")
                    return
            print("⚠️ 한글 폰트를 찾을 수 없습니다. Helvetica 사용")
        except Exception as e:
            print(f"⚠️ 폰트 로드 실패: {e}")

    # ---------- Styles ----------
    def _create_styles(self) -> Dict:
        styles = getSampleStyleSheet()
        font = self.korean_font
        styles.add(ParagraphStyle(
            name="KTitle",
            parent=styles["Heading1"],
            fontName=font,
            fontSize=22,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=14,
            alignment=TA_LEFT,
            wordWrap='CJK',
        ))
        styles.add(ParagraphStyle(
            name="KSubTitle",
            parent=styles["Heading2"],
            fontName=font,
            fontSize=15,
            textColor=colors.HexColor("#34495e"),
            spaceBefore=12,
            spaceAfter=10,
            wordWrap='CJK',
        ))
        styles.add(ParagraphStyle(
            name="KHead3",
            parent=styles["Heading3"],
            fontName=font,
            fontSize=12.5,
            textColor=colors.HexColor("#2c3e50"),
            spaceBefore=8,
            spaceAfter=6,
            wordWrap='CJK',
        ))
        styles.add(ParagraphStyle(
            name="KBody",
            parent=styles["BodyText"],
            fontName=font,
            fontSize=10.3,
            leading=15.6,
            alignment=TA_JUSTIFY,
            spaceAfter=4,
            wordWrap='CJK',
        ))
        styles.add(ParagraphStyle(
            name="KBullet",
            parent=styles["BodyText"],
            fontName=font,
            fontSize=10.3,
            leading=14.2,
            leftIndent=14,
            spaceAfter=2,
            wordWrap='CJK',
        ))
        styles.add(ParagraphStyle(
            name="KMeta",
            parent=styles["BodyText"],
            fontName=font,
            fontSize=9.3,
            textColor=colors.HexColor("#7f8c8d"),
            leading=13.5,
            spaceAfter=3,
            wordWrap='CJK',
        ))
        styles.add(ParagraphStyle(  # 표 셀용
            name="KCell",
            parent=styles["BodyText"],
            fontName=font,
            fontSize=9.5,
            leading=13.0,
            alignment=TA_LEFT,
            wordWrap='CJK',
        ))
        styles.add(ParagraphStyle(  # TOC 항목 크게
            name="KTOCItem",
            parent=styles["BodyText"],
            fontName=font,
            fontSize=13.5,
            leading=18.0,
            alignment=TA_LEFT,
            wordWrap='CJK',
        ))
        return styles

    # ---------- Header / Footer ----------
    def _header_footer(self, canvas, doc, company_name, domain, created_str):
        canvas.saveState()
        w, h = A4
        canvas.setFont(self.korean_font, 9)
        canvas.setFillColor(colors.HexColor("#7f8c8d"))
        canvas.drawString(20*mm, h - 12*mm, f"{company_name} · {domain} · 생성일 {created_str}")
        page_num = canvas.getPageNumber()
        canvas.setFillColor(colors.HexColor("#95a5a6"))
        canvas.drawRightString(w - 20*mm, 12*mm, f"{page_num}")
        canvas.restoreState()

    # ---------- LLM ----------
    def _compose_llm_context(self, state: PipelineState, derived_risk: str) -> str:
        payload = {
            "company_name": state.get("company_name"),
            "domain": state.get("domain"),
            "ethics_score": state.get("ethics_score", {}),
            "ethics_evaluation": {
                k: {
                    "score": _safe_get(state, ["ethics_evaluation", k, "score"], 0),
                    "confidence": _safe_get(state, ["ethics_evaluation", k, "confidence"], 0.0),
                    "strengths": _safe_get(state, ["ethics_evaluation", k, "strengths"], []),
                    "issues": _safe_get(state, ["ethics_evaluation", k, "issues"], []),
                    "evidence": [
                        {
                            "finding": e.get("finding"),
                            "source": e.get("source"),
                            "tier": e.get("tier"),
                            "reliability": e.get("reliability"),
                            "type": e.get("evidence_type"),
                            "weight": e.get("weight"),
                            "url": e.get("url")
                        } for e in _safe_get(state, ["ethics_evaluation", k, "evidence"], [])[:6]
                    ]
                } for k in ["transparency", "human_oversight", "data_governance", "accuracy_validation", "accountability"]
            },
            "risk_level_resolved": derived_risk,
            "critical_issues": [
                {
                    "severity": x.get("severity"),
                    "category": x.get("category"),
                    "description": x.get("description"),
                    "recommendation": x.get("recommendation"),
                    "eu_ai_act_article": x.get("eu_ai_act_article"),
                    "evidence_quality": x.get("evidence_quality")
                } for x in state.get("critical_issues", [])[:8]
            ],
            "analysis_result": {
                "summary": _safe_get(state, ["analysis_result", "summary"], ""),
                "business_model": _safe_get(state, ["analysis_result", "business_model"], ""),
                "target_users": _safe_get(state, ["analysis_result", "target_users"], [])[:6],
                "technology_stack": _safe_get(state, ["analysis_result", "technology_stack"], [])[:10],
            }
        }
        return json.dumps(payload, ensure_ascii=False)

    def _llm_generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
        if not self.client:
            return ""
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.25,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"⚠️ LLM 호출 실패: {e}")
            return ""

    def _resolve_risk_level(self, state: PipelineState) -> str:
        current = _safe_get(state, ["final_scores", "risk_level"], "UNKNOWN") or "UNKNOWN"
        if current and current.upper() != "UNKNOWN":
            return current
        if not self.client:
            return "UNKNOWN"
        sys_p = (
            "역할: EU AI Act 맥락에서 시스템/도메인의 위험군을 추정하는 분석가.\n"
            "규칙: 입력 JSON의 도메인/요약/대상유저/기술만 사용. 외부 지식 추가 금지. "
            "출력은 CRITICAL/HIGH/MEDIUM/LOW 중 하나의 대문자 토큰만.\n"
            "민감 분야(의료·교육·고용·공공안전 등)는 보수적으로 HIGH 이상."
        )
        user_p = json.dumps({
            "domain": state.get("domain"),
            "analysis_hint": {
                "summary": _safe_get(state, ["analysis_result", "summary"], ""),
                "business_model": _safe_get(state, ["analysis_result", "business_model"], ""),
                "target_users": _safe_get(state, ["analysis_result", "target_users"], [])[:6],
            }
        }, ensure_ascii=False)
        out = self._llm_generate(sys_p, user_p, max_tokens=5).upper().strip()
        return out if out in {"CRITICAL", "HIGH", "MEDIUM", "LOW"} else "UNKNOWN"

    # ---------- Public ----------
    def execute(self, state: PipelineState) -> PipelineState:
        print("\n" + "="*70)
        print("📄 [PDF 보고서 생성 Agent] 시작 (TOC 확대 & 출처 + RAG 트레이스 포함)")
        print("="*70)

        try:
            filename = f"ethics_report_{state['company_name']}_{state['domain']}_{datetime.now().strftime('%Y%m%d')}.pdf"
            print(f"📝 PDF 생성 중: {filename}")
            self._generate_pdf(state, filename)
            state["report_path"] = filename
            print("\n" + "="*70)
            print("✅ PDF 보고서 생성 완료!")
            print("="*70 + f"\n📁 파일: {filename}")
        except Exception as e:
            import traceback
            print(f"❌ PDF 생성 실패: {e}")
            state["errors"].append({
                "stage": "pdf_generation",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "recovered": False,
                "traceback": traceback.format_exc(),
                "impact_on_reliability": "medium"
            })
        return state

    # ---------- Build ----------
    def _generate_pdf(self, state: PipelineState, filename: str):
        company = _nn(state.get("company_name"))
        domain = _nn(state.get("domain"))
        created_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        doc = SimpleDocTemplate(
            filename, pagesize=A4,
            rightMargin=20*mm, leftMargin=20*mm,
            topMargin=22*mm, bottomMargin=18*mm
        )
        styles = self._create_styles()
        story: List = []

        # 위험등급 LLM 보정
        derived_risk = self._resolve_risk_level(state)

        # 1) Cover
        story.extend(self._cover_page(state, styles))
        story.append(PageBreak())

        # 2) TOC (항목 폰트 크게)
        story.extend(self._toc(styles))
        story.append(PageBreak())

        # 3) Executive Summary
        story.extend(self._executive_summary_with_llm(state, styles, derived_risk))
        story.append(PageBreak())

        # 4) Ethics
        story.extend(self._ethics_section_with_llm(state, styles, derived_risk))
        story.append(PageBreak())

        # 5) Final Score & Reco
        story.extend(self._final_scores_and_llm_reco(state, styles, derived_risk))
        story.append(PageBreak())

        # 6) References
        story.extend(self._references(state, styles))
        story.append(PageBreak())

        # 7) Appendix
        story.extend(self._appendix(state, styles))

        doc.build(
            story,
            onFirstPage=lambda c, d: self._header_footer(c, d, company, domain, created_str),
            onLaterPages=lambda c, d: self._header_footer(c, d, company, domain, created_str)
        )
        print(f"✅ PDF 파일 생성 완료: {filename}")

    # ---------- Sections ----------
    def _cover_page(self, state: PipelineState, styles: Dict) -> List:
        s: List = []
        s.append(Spacer(1, 25*mm))
        s.append(Paragraph("EU AI Act", ParagraphStyle(
            name="BigTitleLine1", parent=styles["KTitle"], fontSize=26, alignment=TA_CENTER, spaceAfter=10*mm, wordWrap='CJK'
        )))
        s.append(Paragraph("윤리 위험성 평가 보고서", ParagraphStyle(
            name="BigTitleLine2", parent=styles["KTitle"], fontSize=22, alignment=TA_CENTER, spaceAfter=12*mm, wordWrap='CJK'
        )))

        info = [
            ["대상 기업", _nn(state.get("company_name"))],
            ["도메인", _nn(state.get("domain"))],
            ["평가 일시", datetime.now().strftime("%Y년 %m월 %d일 %H:%M")],
            ["보고서 ID", _nn(state.get("run_id"), "-")[:16]],
        ]
        s.append(self._table(info, [55*mm, 100*mm], head=False))
        s.append(Spacer(1, 8*mm))

        es = state.get("ethics_score", {})
        total = es.get("total", 0)
        grade = es.get("grade", "N/A")
        score_data = [
            ["항목", "값"],
            ["종합 점수(윤리)", f"{total}/100"],
            ["최종 등급(윤리)", grade],
        ]
        s.append(self._table(score_data, [55*mm, 100*mm], head=True,
                             emphasize=[(1,1)], emphasize_color=_grade_color(grade)))
        return s

    def _toc(self, styles: Dict) -> List:
        s: List = []
        s.append(Paragraph("목차", ParagraphStyle(
            name="TOCTitle", parent=styles["KTitle"], alignment=TA_LEFT, wordWrap='CJK'
        )))
        s.append(Spacer(1, 6*mm))
        items = [
            "1. Executive Summary",
            "2. Ethics Evaluation (EU AI Act)",
            "3. Final Score & Recommendations",
            "4. References",
            "5. Appendix",
        ]
        for it in items:
            s.append(Paragraph(it, styles["KTOCItem"]))  # 크게
        return s

    # ====== Executive Summary ======
    def _executive_summary_with_llm(self, state: PipelineState, styles: Dict, derived_risk: str) -> List:
        s: List = []
        s.append(Paragraph("1. Executive Summary", styles["KTitle"]))
        s.append(HRLine()); s.append(Spacer(1, 3*mm))

        web_cnt = _safe_get(state, ["web_collection", "count"], 0) or 0
        sp_cnt = _safe_get(state, ["specialized_collection", "count"], 0) or 0
        s.append(Paragraph(
            f"본 평가는 수집된 문서(Web {web_cnt}건, Specialized {sp_cnt}건)를 바탕으로 "
            "간접 지표를 추정하여 EU AI Act 준수 수준을 평가했습니다.",
            styles["KMeta"]
        ))
        s.append(Spacer(1, 2*mm))

        es = state.get("ethics_score", {})
        rows = [
            ["최종 종합 점수(윤리)", f"{es.get('total', 0)}/100"],
            ["최종 등급(윤리 기준)", _nn(es.get("grade", "N/A"))],
            ["위험 등급(도메인/RAG)", _nn(derived_risk)],
        ]
        summary_tbl = self._table([["항목", "값"], *rows], [65*mm, 90*mm], head=True)
        summary_tbl.splitByRow = 1
        s.append(KeepTogether([summary_tbl]))
        s.append(Spacer(1, 4*mm))

        context = self._compose_llm_context(state, derived_risk)
        system_prompt = (
            "역할: EU AI Act 윤리평가 Executive Summary 작성자.\n"
            "제약: 입력 JSON(윤리 점수/카테고리 증거/위험등급)만 사용. 외부 지식 금지.\n"
            "스타일: 한국어, 명확한 줄바꿈, 각 불릿 1문장, 총 120~220단어 내.\n"
            "품질: 근거를 명시적으로 언급(예: 직접증거/Tier, 점수/이슈)."
        )
        user_prompt = (
            "다음 STATE_JSON을 근거로 작성:\n"
            "① 첫 문단: 대상/도메인 + 윤리 종합 점수/등급 핵심 결론(2~3문장)\n"
            "② 근거 기반 하이라이트 3~5개(각 1문장, 증거 속성 괄호 표기)\n"
            "③ 부족한 점/리스크 3~5개(각 1문장, 관련 카테고리 명시)\n"
            "④ 마지막 문장: RAG 도메인 위험등급의 의미 1문장\n"
            f"\n# STATE_JSON\n{context}"
        )
        llm_text = self._llm_generate(system_prompt, user_prompt, max_tokens=700)
        for flow in _render_llm_text(llm_text, styles):
            s.append(flow)
        return s

    # ====== 윤리평가(+ LLM 코멘트, 근거 스택형) ======
    def _ethics_section_with_llm(self, state: PipelineState, styles: Dict, derived_risk: str) -> List:
        s: List = []
        s.append(Paragraph("2. Ethics Evaluation (EU AI Act)", styles["KTitle"]))
        s.append(HRLine()); s.append(Spacer(1, 3*mm))

        es = state.get("ethics_score", {})
        ev = state.get("ethics_evaluation", {})

        cats = ["transparency", "human_oversight", "data_governance", "accuracy_validation", "accountability"]
        cat_names_en = ["Transparency", "Oversight", "Data", "Accuracy", "Accountability"]
        cat_scores = [int(es.get(c, 0)) for c in cats]

        s.append(Paragraph("Category Scores", styles["KHead3"]))
        s.append(self._bar_chart(cat_names_en, cat_scores, height=160))
        s.append(Spacer(1, 2*mm))

        context = self._compose_llm_context(state, derived_risk)
        system_prompt = (
            "역할: EU AI Act 윤리 카테고리 해설자.\n"
            "제약: 입력 JSON만 사용, 외부 지식 금지. 한국어. 헤더문법/표 금지. 각 불릿=1문장. 총 150~260단어.\n"
            "요구: 강점/부족/개선 우선순위를 카테고리별로 제시하고, 증거 타입(직접/간접)과 tier를 괄호로 표시."
        )
        user_prompt = (
            "STATE_JSON을 바탕으로 카테고리별로 작성:\n"
            "• 투명성: 강점 2~3, 부족 2~3, 개선 1~3(High/Med/Low)\n"
            "• 인간감독: 동일\n"
            "• 데이터거버넌스: 동일\n"
            "• 정확도검증: 동일\n"
            "• 책임성: 동일\n"
            f"\n# STATE_JSON\n{context}"
        )
        llm_cmt = self._llm_generate(system_prompt, user_prompt, max_tokens=900)
        for flow in _render_llm_text(llm_cmt, styles):
            s.append(flow)

        for idx, key in enumerate(cats):
            cat = ev.get(key, {})
            if not cat:
                continue
            header = Paragraph(f"{idx+1}. {cat_names_en[idx]}", styles["KSubTitle"])
            rows1 = [
                ["점수", str(cat.get("score", 0))],
                ["레벨", str(cat.get("level", 0))],
            ]
            tbl1 = self._table([["항목","값"], *rows1], [65*mm, 90*mm], head=True)
            s.append(KeepTogether([header, tbl1, Spacer(1, 2*mm)]))

            rows2 = [
                ["근거 수(직접/Tier1)", f"{cat.get('evidence_count',0)} / {cat.get('direct_evidence_count',0)} / {cat.get('tier1_evidence_count',0)}"],
                ["정보 가용성", _nn(cat.get("information_availability","none"))],
            ]
            tbl2 = self._table([["항목","값"], *rows2], [65*mm, 90*mm], head=True)
            s.append(KeepTogether([tbl2, Spacer(1, 2*mm)]))

            if cat.get("strengths"):
                s.append(Paragraph("Strengths", styles["KHead3"]))
                for st in cat["strengths"][:4]:
                    s.append(Paragraph(_wrap(f"• {st}", 120), styles["KBullet"]))
            if cat.get("issues"):
                s.append(Paragraph("Issues", styles["KHead3"]))
                for it in cat["issues"][:4]:
                    s.append(Paragraph(_wrap(f"• {it}", 120), styles["KBullet"]))

            evid_list = cat.get("evidence") or []
            if evid_list:
                s.append(Paragraph("Key Evidence", styles["KHead3"]))
                for evi in evid_list[:4]:
                    finding = _wrap(_nn(evi.get("finding")), 110)
                    src = _nn(evi.get("source"))
                    tier = _nn(evi.get("tier"))
                    etype = _nn(evi.get("evidence_type"))
                    rel = _nn(evi.get("reliability"))
                    wgt = _fmt_float(evi.get("weight"))
                    meta = f"출처: {src} · Tier: {tier} · 타입: {etype} · 신뢰성: {rel} · 가중치: {wgt}"
                    block = [Paragraph(f"• {finding}", styles["KBody"]),
                             Paragraph(meta, styles["KMeta"]),
                             Spacer(1, 1*mm)]
                    s.append(KeepTogether(block))
            s.append(Spacer(1, 3*mm))
        return s

    # ====== 최종 점수 & 권고 ======
    def _final_scores_and_llm_reco(self, state: PipelineState, styles: Dict, derived_risk: str) -> List:
        s: List = []
        s.append(Paragraph("3. Final Score & Recommendations", styles["KTitle"]))
        s.append(HRLine()); s.append(Spacer(1, 3*mm))

        es = state.get("ethics_score", {})
        risk_color = _risk_color(derived_risk)
        rows = [
            ["종합 점수(윤리)", f"{es.get('total', 0)}/100"],
            ["최종 등급(윤리 기준)", _nn(es.get("grade", "N/A"))],
            ["위험 등급(도메인/RAG)", _nn(derived_risk)],
        ]
        s.append(self._table([["항목","값"], *rows], [65*mm, 90*mm], head=True))
        s.append(Spacer(1, 2*mm))
        s.append(Paragraph(
            f'<font color="{risk_color}">※ 위험 등급은 도메인 특성과 용도를 RAG/LLM으로 해석해 추정한 값이며, 윤리 점수와는 별개 축입니다.</font>',
            styles["KBody"]
        ))

        context = self._compose_llm_context(state, derived_risk)
        system_prompt = (
            "역할: EU AI Act 준수/윤리 리스크 완화를 위한 권고안 작성자.\n"
            "제약: 입력 JSON만 사용. 한국어. 각 항목 1~2문장. 총 6~10개 권고. "
            "각 항목에 [우선순위]와 (담당부서) 표기, 마지막 줄에 '근거:' 1줄."
        )
        user_prompt = (
            "STATE_JSON을 바탕으로 다음 형식으로 작성:\n"
            "• [High] 데이터 거버넌스 표준화(담당: 데이터 거버넌스팀) – 1~2문장 설명\n"
            "  근거: 관련 카테고리/핵심 증거 요약 1줄\n"
            "• [Medium] …\n"
            f"\n# STATE_JSON\n{context}"
        )
        llm_reco = self._llm_generate(system_prompt, user_prompt, max_tokens=900)
        for flow in _render_llm_text(llm_reco, styles):
            s.append(flow)
        return s

    # ====== References: 스택형 항목 ======
    def _references(self, state: PipelineState, styles: Dict) -> List:
        s: List = []
        s.append(Paragraph("4. References", styles["KTitle"]))
        s.append(HRLine()); s.append(Spacer(1, 3*mm))

        all_docs = []
        all_docs += state.get("merged_documents", [])
        all_docs += state.get("web_collection", {}).get("documents", [])
        all_docs += state.get("specialized_collection", {}).get("documents", [])

        by_tier: Dict[str, List[dict]] = {"tier1": [], "tier2": [], "tier3": []}
        for d in all_docs:
            tier = d.get("evidence_tier") or d.get("tier") or "tier3"
            by_tier.setdefault(tier, [])
            by_tier[tier].append(d)

        tier_title = {
            "tier1": "Tier 1 (최고 신뢰도)",
            "tier2": "Tier 2 (높은 신뢰도)",
            "tier3": "Tier 3 (보통 신뢰도)"
        }

        for tier_name in ["tier1", "tier2", "tier3"]:
            docs = by_tier.get(tier_name, [])
            if not docs:
                continue
            s.append(Paragraph(tier_title[tier_name], styles["KSubTitle"]))
            for d in docs[:30]:
                title = _wrap(_nn(d.get("title")), 105)
                meta = f"발행처: {_nn(d.get('publisher','-'))} · 카테고리: {_nn(d.get('source_category','-'))} · 날짜: {_nn(d.get('date','-'))}"
                url = _wrap(_nn(d.get("url")), 105)
                block = [
                    Paragraph(f"• {title}", styles["KBody"]),
                    Paragraph(meta, styles["KMeta"]),
                    Paragraph(url, styles["KMeta"]),
                    Spacer(1, 1*mm)
                ]
                s.append(KeepTogether(block))
            s.append(Spacer(1, 2*mm))
        return s

    # ====== Appendix (요약 표 + 평가 기준 상세 + 출처 + RAG 트레이스) ======
    def _appendix(self, state: PipelineState, styles: Dict) -> List:
        def _cell(text) -> Paragraph:
            return Paragraph(_nn(text), styles["KCell"])

        s: List = []
        s.append(Paragraph("5. Appendix", styles["KTitle"]))
        s.append(HRLine()); s.append(Spacer(1, 3*mm))

        crit = state.get("ethics_evaluation_criteria", {})
        categories = [
            ("transparency", "Transparency"),
            ("human_oversight", "Human Oversight"),
            ("data_governance", "Data Governance"),
            ("accuracy_validation", "Accuracy & Validation"),
            ("accountability", "Accountability"),
        ]

        # --- 5.1 Summary Table ---
        header = [_cell("분류"), _cell("기준수"), _cell("평균가중"), _cell("예시")]
        rows = [header]
        for key, name in categories:
            c = _safe_get(crit, ["categories", key], {})
            items = c.get("criteria", []) if isinstance(c, dict) else []
            cnt = len(items)
            if cnt > 0:
                weights = [float(it.get("weight", 0) or 0) for it in items]
                avg_w = sum(weights) / len(weights) if weights else 0.0
                ex_titles = [str(it.get("title") or "-") for it in items[:2]]
                examples = ", ".join(ex_titles)
            else:
                avg_w = 0.0
                examples = "-"
            rows.append([
                _cell(name),
                _cell(str(cnt)),
                _cell(_fmt_float(avg_w, 2)),
                _cell(examples),
            ])

        summary_tbl = self._table(rows, [40*mm, 18*mm, 22*mm, 80*mm], head=True)
        summary_tbl.splitByRow = 1
        s.append(KeepTogether([Paragraph("5.1 Criteria Summary", styles["KSubTitle"]),
                               summary_tbl,
                               Spacer(1, 3*mm)]))

        # --- 5.2 상세 항목 ---
        if crit:
            s.append(Paragraph("5.2 평가 기준 상세", styles["KSubTitle"]))
            for key, name in categories:
                c = _safe_get(crit, ["categories", key], {})
                if not c:
                    continue
                s.append(Paragraph(name, styles["KHead3"]))
                for i, it in enumerate(c.get("criteria", [])[:6], 1):
                    block = f"""
<b>{i}. {_nn(it.get('title'))}</b><br/>
{_wrap(_nn(it.get('description')), 110)}<br/>
<b>측정</b>: {_wrap(_nn(it.get('measurement')), 110)}<br/>
<b>가중치</b>: {_fmt_float(it.get('weight'), 2)}
"""
                    s.append(Paragraph(block, styles["KBody"]))
                    s.append(Spacer(1, 1*mm))

        # --- 5.3 평가 기준 출처(페이지 표기 포함) ---
        sources = state.get("criteria_sources") or []
        if not sources:
            sources = [{
                "title": "Vara - Crunchbase Company Profile & Funding",
                "tier": "tier2",
                "type": "indirect",
                "reliability": "medium",
                "weight": None,
                "page": None,
                "url": None,
                "publisher": "Crunchbase"
            }]
        s.append(Paragraph("5.3 평가 기준 출처", styles["KSubTitle"]))
        for src in sources[:40]:
            title = _nn(src.get("title"))
            tier = _nn(src.get("tier"))
            etype = _nn(src.get("type"))
            rel = _nn(src.get("reliability"))
            wgt = _fmt_float(src.get("weight"), 2) if src.get("weight") is not None else "-"
            page = src.get("page")
            page_str = f"{page}쪽 참고" if page else "-"
            pub = _nn(src.get("publisher", "-"))
            meta = f"출처: {title} · 발행처: {pub} · Tier: {tier} · 타입: {etype} · 신뢰성: {rel} · 가중치: {wgt} · 페이지: {page_str}"
            s.append(Paragraph(f"• {meta}", styles["KBody"]))
            url = src.get("url")
            if url:
                s.append(Paragraph(_nn(url), styles["KMeta"]))
            s.append(Spacer(1, 1*mm))

        # --- 5.4 RAG 트레이스(검색 쿼리 & 근거 페이지) ---
        traces = state.get("criteria_rag_traces")
        if not traces:
            log_text = state.get("criteria_rag_log_text") or ""
            traces = self._parse_rag_logs(log_text) if log_text else []

        if traces:
            s.append(Paragraph("5.4 RAG 트레이스 (검색 쿼리 & 근거 페이지)", styles["KSubTitle"]))
            for t in traces[:10]:  # 과도한 길이 방지
                cat = _nn(t.get("category"))
                q = _nn(t.get("query"))
                s.append(Paragraph(f"<b>카테고리:</b> {cat}", styles["KBody"]))
                s.append(Paragraph(f"<b>검색 쿼리:</b> {q}", styles["KMeta"]))

                matches = t.get("matches") or []
                if matches:
                    s.append(Paragraph("근거 페이지(상위):", styles["KHead3"]))
                    # 표로 보여주되 셀은 Paragraph로 감싸 CJK 줄바꿈
                    rows = [[Paragraph("페이지", styles["KCell"]), Paragraph("발췌", styles["KCell"])]]
                    for m in matches[:5]:
                        page = _nn(m.get("page"))
                        snippet = _wrap(_nn(m.get("snippet")), 90)
                        rows.append([Paragraph(str(page), styles["KCell"]), Paragraph(snippet, styles["KCell"])])
                    tbl = self._table(rows, [18*mm, 122*mm], head=True)
                    s.append(KeepTogether([tbl, Spacer(1, 2*mm)]))

                criteria = t.get("generated_criteria") or []
                if criteria:
                    s.append(Paragraph("생성된 기준:", styles["KHead3"]))
                    for i, c in enumerate(criteria[:8], 1):
                        s.append(Paragraph(f"• {c}", styles["KBullet"]))
                s.append(Spacer(1, 3*mm))

        return s

    # ---------- RAG 로그 파서 ----------
    def _parse_rag_logs(self, text: str) -> List[Dict]:
        """
        원문 로그 텍스트 예시(사용자 제공 포맷)를 파싱해 구조화:
        🔍 검색 쿼리: ...
        📜 검색된 조항: 10개
           [1] Page 16: ...
           [2] Page 18: ...
        ✅ 정확도 검증 평가 기준:
           생성된 기준: 6개
           1. ...
           2. ...
        (카테고리별 반복)
        """
        traces: List[Dict] = []
        # 블록을 '======================================================================' 로 나눠도 되지만
        # 안전하게 '🔍 검색 쿼리:'를 기준으로 split
        blocks = re.split(r"\n(?=🔍\s*검색 쿼리:)", text)
        for blk in blocks:
            blk = blk.strip()
            if not blk:
                continue
            m_q = re.search(r"🔍\s*검색\s*쿼리:\s*(.+)", blk)
            query = m_q.group(1).strip() if m_q else ""

            # 카테고리명은 '✅ XXX 평가 기준' 라인에서 캡처
            m_cat = re.search(r"✅\s*(.+?)\s*평가\s*기준", blk)
            category = m_cat.group(1).strip() if m_cat else ""

            # 검색된 조항 리스트
            matches = []
            for m in re.finditer(r"\[\d+\]\s*Page\s*(\d+):\s*(.+)", blk):
                page = int(m.group(1))
                snippet = m.group(2).strip()
                matches.append({"page": page, "snippet": snippet})

            # 생성된 기준 목록
            criteria = []
            # '생성된 기준' 이후의 numbered list 추출
            sect = re.search(r"생성된\s*기준[:：]\s*\d+개?(.*)", blk, re.S)
            tail = sect.group(1) if sect else blk
            for m in re.finditer(r"^\s*\d+\.\s*(.+)$", tail, re.M):
                criteria.append(m.group(1).strip())

            traces.append({
                "category": category,
                "query": query,
                "matches": matches,
                "generated_criteria": criteria
            })
        return traces

    # ---------- Widgets ----------
    def _table(
        self,
        data: List[List[str]],
        col_widths: List[float],
        head: bool = False,
        emphasize: Optional[List[Tuple[int, int]]] = None,
        emphasize_color: colors.Color = colors.HexColor("#27ae60"),
    ):
        t = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1 if head else 0)
        t.splitByRow = 1
        base = [
            ("FONTNAME", (0, 0), (-1, -1), self.korean_font),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bdc3c7")),
            ("ROWBACKGROUNDS", (0, 1 if head else 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ]
        if head:
            base += [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, 0), 10.5),
            ]
        if emphasize:
            for (r, c) in emphasize:
                base += [
                    ("TEXTCOLOR", (c, r), (c, r), emphasize_color),
                    ("FONTSIZE", (c, r), (c, r), 13),
                    ("FONTNAME", (c, r), (c, r), self.korean_font),
                ]
        t.setStyle(TableStyle(base))
        return t

    def _bar_chart(self, labels: List[str], values: List[float], height: int = 140):
        width = 420
        d = Drawing(width, height)
        bc = VerticalBarChart()
        bc.x = 40; bc.y = 20
        bc.height = height - 40; bc.width = width - 80
        bc.data = [values]
        bc.categoryAxis.categoryNames = labels
        bc.categoryAxis.labels.boxAnchor = "ne"; bc.categoryAxis.labels.angle = 20
        bc.barWidth = 15
        bc.valueAxis.valueMin = 0; bc.valueAxis.valueMax = 100; bc.valueAxis.valueStep = 20
        bc.groupSpacing = 12
        bc.bars[0].fillColor = colors.HexColor("#3498db")
        d.add(bc)
        return d
