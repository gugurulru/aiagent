# agents/web_collection.py

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import re
import json
from urllib.parse import urlparse

from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class WebCollectionAgent:
    """
    Web collection agent with self-gating loop.
    Searches via Tavily, classifies, scores, and retries to fill missing slots.
    """

    # ------- Gate policy (thresholds) ------- 💰 비용 최적화 설정
    DEFAULT_MIN_DOCS = 50              
    DEFAULT_MIN_PRIMARY = 2
    DEFAULT_RECENT_WINDOW_DAYS = 180
    DEFAULT_MIN_RECENT = 3
    DEFAULT_AVG_REL_MIN = 0.6
    DEFAULT_LOW_RATIO_MAX = 0.3
    DEFAULT_DISTINCT_DOMAINS_MIN = 4
    DEFAULT_CATEGORY_DIVERSITY_MIN = 2
    DEFAULT_MAX_ATTEMPTS = 2           
    DOMAIN_CAP_PER_HOST = 15           

    def __init__(self, tavily_api_key: str, openai_api_key: str):
        self.tavily = TavilyClient(api_key=tavily_api_key)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0
        )

    # -------------------- Public entry --------------------
    def execute(self, state: Dict) -> Dict:
        """Main run with self-gating; accumulates docs and retries."""
        print("\n" + "=" * 70)
        print("🌐 [웹 정보 수집 Agent] 시작 (자가 게이트/루프)")
        print("=" * 70)

        if "errors" not in state or not isinstance(state["errors"], list):
            state["errors"] = []

        company_name = state.get("company_name", "").strip()
        domain = state.get("domain", "").strip()

        threshold_profile = {
            "min_docs": self.DEFAULT_MIN_DOCS,
            "min_primary": self.DEFAULT_MIN_PRIMARY,
            "recent_window_days": self.DEFAULT_RECENT_WINDOW_DAYS,
            "min_recent": self.DEFAULT_MIN_RECENT,
            "avg_rel_min": self.DEFAULT_AVG_REL_MIN,
            "low_ratio_max": self.DEFAULT_LOW_RATIO_MAX,
            "distinct_domains_min": self.DEFAULT_DISTINCT_DOMAINS_MIN,
            "category_diversity_min": self.DEFAULT_CATEGORY_DIVERSITY_MIN,
            "max_attempts": self.DEFAULT_MAX_ATTEMPTS,
        }

        state["web_collection"] = {
            "status": "insufficient",     # completed | insufficient | partial | degraded
            "documents": [],
            "count": 0,
            "sources": [],
            "sources_breakdown": {},
            "primary_sources_count": 0,
            "secondary_sources_count": 0,
            "newest_date": None,
            "oldest_date": None,
            "recent_documents_count": 0,
            "high_reliability_count": 0,
            "medium_reliability_count": 0,
            "low_reliability_count": 0,
            "average_reliability": 0.0,
            "gate": {
                "passed": False,
                "missing_slots": [],
                "reasons": []
            },
            "attempts": {
                "current": 0,
                "max": threshold_profile["max_attempts"]
            },
            "decision_log": [],
            "threshold_profile": threshold_profile,
            "error": None
        }

        try:
            cumulative_docs: List[Dict] = []
            seen_urls = set()

            while state["web_collection"]["attempts"]["current"] < state["web_collection"]["attempts"]["max"]:
                state["web_collection"]["attempts"]["current"] += 1
                attempt_no = state["web_collection"]["attempts"]["current"]
                print(f"\n🔁 수집 시도 {attempt_no}/{state['web_collection']['attempts']['max']}")

                # 1) Query planning
                queries = self._build_queries(company_name, domain, state)
                # 2) Search
                tavily_results = self._run_tavily_queries(queries)
                # 3) Analyze & classify (also produce two-line summary)
                analyzed_docs = self._analyze_documents(tavily_results, company_name)

                # 4) Deduplicate and merge
                new_docs = []
                for d in analyzed_docs:
                    url = d.get("url", "")
                    if not url:
                        continue
                    norm = self._normalize_url(url)
                    if norm in seen_urls:
                        continue
                    seen_urls.add(norm)
                    new_docs.append(d)

                cumulative_docs.extend(new_docs)

                # 5) Reliability and recency
                cumulative_docs = self._assess_reliability(cumulative_docs)

                # 6) Update stats
                stats = self._calculate_statistics(cumulative_docs)

                # 7) Gate evaluation
                gate_passed, missing_slots, reasons = self._evaluate_gate(
                    documents=cumulative_docs,
                    stats=stats,
                    policy=threshold_profile
                )

                # 8) State update
                state["web_collection"].update({
                    "documents": cumulative_docs,
                    "count": len(cumulative_docs),
                    "sources": list(set(doc.get("source_category") for doc in cumulative_docs)),
                    **stats,
                    "gate": {
                        "passed": gate_passed,
                        "missing_slots": missing_slots,
                        "reasons": reasons
                    },
                    "status": "completed" if gate_passed else "insufficient",
                    "error": None
                })

                # 9) Decision log
                state["web_collection"]["decision_log"].append({
                    "attempt": attempt_no,
                    "queries": queries,
                    "new_docs": len(new_docs),
                    "total_docs": len(cumulative_docs),
                    "gate_passed": gate_passed,
                    "missing_slots": missing_slots,
                    "reasons": reasons,
                    "timestamp": datetime.now().isoformat()
                })

                # 10) Exit or retry
                if gate_passed:
                    print(f"✅ 게이트 통과. 총 문서 {len(cumulative_docs)}개")
                    break
                else:
                    print(f"⚠️ 게이트 미통과 → 부족 슬롯: {missing_slots or ['(없음)']}")
                    if attempt_no >= state["web_collection"]["attempts"]["max"]:
                        state["web_collection"]["status"] = "partial"
                        print("⛔ 최대 시도 도달. partial로 종료합니다.")
                        break

            print(f"\n📦 최종 상태: {state['web_collection']['status']} / 문서 {state['web_collection']['count']}개")

        except Exception as e:
            print(f"❌ 웹 수집 실패(런타임): {str(e)}")
            state["web_collection"]["status"] = "degraded"
            state["web_collection"]["error"] = str(e)
            state["errors"].append({
                "stage": "web_collection",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "recovered": False,
                "traceback": None,
                "impact_on_reliability": "high"
            })

        return state

    # -------------------- Internals --------------------
    def _build_queries(self, company_name: str, domain: str, state: Dict) -> List[str]:
        """Build search queries; first broad, later target missing slots."""
        wc = state["web_collection"]
        attempt = wc["attempts"]["current"]
        missing = wc["gate"]["missing_slots"]

        # Broad/baseline queries (keep many to reach 100 docs)
        base = [
            f"{company_name} {domain} company overview",
            f"{company_name} product features technology",
            f"{company_name} news press release",
            f"{company_name} case study",
            f"{company_name} whitepaper",
            f"{company_name} interview",
            f"{company_name} site:medium.com",
            f"{company_name} site:github.com",
            f"{company_name} site:linkedin.com",
            f"{company_name} funding announcement",
        ]

        # ✅ 수정: 첫 시도에서 모든 base 쿼리 반환
        if attempt == 1:
            return base  # 10개 모두 반환

        # 이후 시도에서는 missing slots에 따라 추가 쿼리
        if not missing:
            return base

        q = []
        recent_cut = (datetime.utcnow() - timedelta(days=wc["threshold_profile"]["recent_window_days"])).date().isoformat()

        if "primary" in missing:
            q += [
                f'{company_name} site:{self._company_domain_hint(company_name)} (press OR newsroom OR investors)',
                f'{company_name} press release site:{self._company_domain_hint(company_name)}',
                f'{company_name} site:{self._company_domain_hint(company_name)} blog',
            ]
        if "recent" in missing:
            q += [
                f'{company_name} ("launch" OR "announced" OR "funding") after:{recent_cut}',
                f'{company_name} press release after:{recent_cut}',
            ]
        if "docs_count" in missing:
            q += [
                f"{company_name} {domain} case study pdf",
                f"{company_name} {domain} whitepaper pdf",
                f"{company_name} architecture overview",
            ]
        if "reliability" in missing:
            q += [
                f'{company_name} site:sec.gov OR site:europa.eu OR site:ec.europa.eu',
                f'{company_name} site:iso.org OR site:ieee.org standard',
                f'{company_name} analyst report site:gartner.com OR site:forrester.com',
            ]
        if "diversity" in missing:
            q += [
                f"{company_name} review technology analysis",
                f"{company_name} {domain} industry report",
                f"{company_name} competitors comparison",
            ]

        return base + q

    def _run_tavily_queries(self, queries: List[str]) -> List[Dict]:
        all_results = []

        def _search_one(query: str, depth: str, max_results: int, include_raw: bool) -> List[Dict]:
            try:
                resp = self.tavily.search(
                    query=query,
                    search_depth=depth,
                    max_results=max_results,
                    include_raw_content=include_raw
                )
                results = resp.get("results", []) or []
                print(f"  ✓ '{query[:70]}' [{depth}] → {len(results)}개")
                return results
            except Exception as e:
                print(f"  ✗ '{query[:70]}' 검색 실패: {str(e)}")
                return []

        # ✅ 비용 최적화
        print(f"🔍 Tavily 검색 중 (총 {len(queries)}개 쿼리)...")
        for q in queries:
            all_results.extend(_search_one(q, "basic", 5, False))
        print(f"📊 1차 수집(중복 포함): {len(all_results)}개")

        # ✅ 비용 최적화: 2차 검색은 100개 미달 시에만
        if len(all_results) < 100:
            print("📈 추가 심층 검색 실행 중...")
            for q in queries[:min(5, len(queries))]:  # 상위 5개 쿼리만
                all_results.extend(_search_one(q, "advanced", 8, True))
        print(f"📊 2차 수집(중복 포함 누계): {len(all_results)}개")

        # 도메인 캡
        capped, per_host = [], {}
        for r in all_results:
            host = ""
            try:
                host = urlparse(r.get("url","")).netloc.lower()
            except Exception:
                pass
            # 캡 적용
            cnt = per_host.get(host, 0)
            if cnt < self.DOMAIN_CAP_PER_HOST:
                capped.append(r)
                per_host[host] = cnt + 1

        print(f"🌈 도메인 캡 후: {len(capped)}개  (host≤{self.DOMAIN_CAP_PER_HOST}, 호스트수={len(per_host)})")
        return capped


    def _analyze_documents(self, tavily_results: List[Dict], company_name: str) -> List[Dict]:
        """Classify doc (primary/secondary, category, publisher/date) and add two-line summary."""
        print("🤖 LLM으로 문서 분석 중...")
        analyzed_docs = []

        classify_prompt = ChatPromptTemplate.from_messages([
            ("system", "Classify source and date. Return compact JSON only."),
            ("user", "Company: {company_name}\nTitle: {title}\nURL: {url}\nSnippet:\n{content}\nJSON:")
        ])
        classify_chain = classify_prompt | self.llm

        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize in exactly two lines. No bullets, max ~35 words total."),
            ("user", "URL: {url}\nText:\n{content}\nTwo-line summary:")
        ])
        summary_chain = summary_prompt | self.llm

        for idx, result in enumerate(tavily_results):
            try:
                raw = result.get("raw_content") or result.get("content") or ""
                snippet_for_classify = (result.get("content", "") or raw)[:500]
                summary_basis = raw if raw else (result.get("content", "") or "")[:2000]

                # 1) classification
                cls_resp = classify_chain.invoke({
                    "company_name": company_name,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": snippet_for_classify
                })
                analysis = self._parse_llm_response(cls_resp.content)

                # 2) two-line summary
                try:
                    sum_resp = summary_chain.invoke({
                        "url": result.get("url", ""),
                        "content": summary_basis[:6000]  # keep token budget sane
                    })
                    two_line = self._to_two_lines(sum_resp.content)
                except Exception:
                    two_line = None

                doc = {
                    "id": f"web_{uuid.uuid4().hex[:8]}",
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": raw,
                    "excerpt": (result.get("content", "") or "")[:200],
                    "two_line_summary": two_line,
                    "source_type": analysis.get("source_type", "secondary"),
                    "source_category": analysis.get("source_category", "news"),
                    "publisher": analysis.get("publisher"),
                    "author": None,
                    "date": analysis.get("date"),
                    "age_days": 0,
                    "is_recent": False,
                    "reliability": "medium",
                    "reliability_score": 0.5,
                    "reliability_reasons": [],
                    "evidence_tier": "tier2",
                    "is_verified": False,
                    "verified_by": [],
                    "collected_by": "web_collection",
                    "collected_at": datetime.now().isoformat()
                }
                analyzed_docs.append(doc)

            except Exception as e:
                print(f"  ✗ 문서 {idx+1} 분석 실패: {str(e)}")
                continue

        print(f"✅ {len(analyzed_docs)}개 문서 분석 완료")
        return analyzed_docs

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse JSON from LLM; fallback to defaults on failure."""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(response)
        except Exception:
            return {
                "source_type": "secondary",
                "source_category": "news",
                "publisher": None,
                "date": None
            }

    def _to_two_lines(self, text: str) -> str:
        """Normalize summary to at most two lines of plain text."""
        if not text:
            return None
        # remove code fences and markdown
        txt = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
        # collapse whitespace
        txt = re.sub(r"\s+", " ", txt).strip()
        # heuristic split into two lines
        # try to keep around ~18 words per line
        words = txt.split()
        if len(words) <= 18:
            return txt
        # split roughly in half
        mid = len(words) // 2
        line1 = " ".join(words[:mid])
        line2 = " ".join(words[mid:])
        return f"{line1}\n{line2}"

    def _assess_reliability(self, documents: List[Dict]) -> List[Dict]:
        """Assess reliability for each doc (date+source); mark is_recent."""
        for doc in documents:
            date_str = doc.get("date")
            doc_date = self._parse_date(date_str)
            if doc_date:
                now = datetime.utcnow()
                age_days = (now - doc_date).days
                doc["age_days"] = max(0, age_days)
                doc["is_recent"] = age_days <= self.DEFAULT_RECENT_WINDOW_DAYS
                doc["date"] = doc_date.date().isoformat()
            else:
                doc["age_days"] = 9999
                doc["is_recent"] = False
                doc["date"] = None

            # reliability
            reliability = self._calculate_reliability(doc)
            doc.update(reliability)

        return documents

    def _calculate_reliability(self, doc: Dict) -> Dict:
        """Score reliability via source type, category, and recency."""
        score = 0.5
        reasons = []

        if doc.get("source_type") == "primary":
            score += 0.2
            reasons.append("primary source")

        category_bonus = {
            "company": 0.1,
            "news": 0.0,
            "blog": -0.1
        }
        score += category_bonus.get(doc.get("source_category"), 0.0)

        if doc.get("is_recent"):
            score += 0.1
            reasons.append("recent within policy window")

        score = max(0.0, min(1.0, score))

        if score >= 0.7:
            reliability = "high"
            tier = "tier1"
        elif score >= 0.4:
            reliability = "medium"
            tier = "tier2"
        else:
            reliability = "low"
            tier = "tier3"

        return {
            "reliability": reliability,
            "reliability_score": round(score, 2),
            "reliability_reasons": reasons,
            "evidence_tier": tier
        }

    def _calculate_statistics(self, documents: List[Dict]) -> Dict:
        """Aggregate counts, date range, and average reliability."""
        stats = {
            "sources_breakdown": {},
            "primary_sources_count": 0,
            "secondary_sources_count": 0,
            "newest_date": None,
            "oldest_date": None,
            "recent_documents_count": 0,
            "high_reliability_count": 0,
            "medium_reliability_count": 0,
            "low_reliability_count": 0,
            "average_reliability": 0.0
        }
        if not documents:
            return stats

        for doc in documents:
            cat = doc.get("source_category", "news")
            stats["sources_breakdown"][cat] = stats["sources_breakdown"].get(cat, 0) + 1

            if doc.get("source_type") == "primary":
                stats["primary_sources_count"] += 1
            else:
                stats["secondary_sources_count"] += 1

            if doc.get("is_recent"):
                stats["recent_documents_count"] += 1

            rel = doc.get("reliability")
            if rel == "high":
                stats["high_reliability_count"] += 1
            elif rel == "medium":
                stats["medium_reliability_count"] += 1
            else:
                stats["low_reliability_count"] += 1

        dates = [doc.get("date") for doc in documents if doc.get("date")]
        if dates:
            stats["newest_date"] = max(dates)
            stats["oldest_date"] = min(dates)

        if len(documents) > 0:
            avg_rel = sum(doc.get("reliability_score", 0.5) for doc in documents) / len(documents)
            stats["average_reliability"] = round(avg_rel, 2)

        return stats

    # -------------------- Gate check --------------------
    def _evaluate_gate(self, documents: List[Dict], stats: Dict, policy: Dict) -> Tuple[bool, List[str], List[str]]:
        """Check sufficiency across volume, recency, reliability, and diversity."""
        reasons = []
        missing = []

        total = len(documents)
        if total < policy["min_docs"]:
            missing.append("docs_count")
            reasons.append(f"문서수 부족 {total}/{policy['min_docs']}")

        if stats.get("primary_sources_count", 0) < policy["min_primary"]:
            missing.append("primary")
            reasons.append(f"1차 출처 부족 {stats.get('primary_sources_count',0)}/{policy['min_primary']}")

        if stats.get("recent_documents_count", 0) < policy["min_recent"]:
            missing.append("recent")
            reasons.append(f"최신 문서 부족 {stats.get('recent_documents_count',0)}/{policy['min_recent']}")

        avg_rel = stats.get("average_reliability", 0.0)
        if avg_rel < policy["avg_rel_min"]:
            missing.append("reliability")
            reasons.append(f"평균 신뢰도 낮음 {avg_rel:.2f} < {policy['avg_rel_min']:.2f}")

        low_cnt = stats.get("low_reliability_count", 0)
        low_ratio = (low_cnt / total) if total > 0 else 1.0
        if low_ratio > policy["low_ratio_max"]:
            if "reliability" not in missing:
                missing.append("reliability")
            reasons.append(f"저신뢰 비율 높음 {low_ratio:.2f} > {policy['low_ratio_max']:.2f}")

        distinct_domains = len(self._distinct_domains(documents))
        if distinct_domains < policy["distinct_domains_min"]:
            missing.append("diversity")
            reasons.append(f"도메인 다양성 부족 {distinct_domains}/{policy['distinct_domains_min']}")

        category_div = len([k for k, v in stats.get("sources_breakdown", {}).items() if v > 0])
        if category_div < policy["category_diversity_min"]:
            if "diversity" not in missing:
                missing.append("diversity")
            reasons.append(f"카테고리 다양성 부족 {category_div}/{policy['category_diversity_min']}")

        passed = (len(missing) == 0)
        return passed, sorted(set(missing)), reasons

    # -------------------- Helpers --------------------
    def _parse_date(self, s: Optional[str]) -> Optional[datetime]:
        """Parse several date formats; return None if not parseable."""
        if not s:
            return None
        s = s.strip()
        try:
            return datetime.fromisoformat(s)
        except Exception:
            pass
        m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
        if m:
            y, mo, d = m.groups()
            try:
                return datetime(int(y), int(mo), int(d))
            except Exception:
                return None
        try:
            from calendar import month_abbr, month_name
            mon_map = {m.lower(): i for i, m in enumerate(month_abbr) if m}
            mon_map.update({m.lower(): i for i in month_name if m})
            m2 = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", s)
            if m2:
                mon_s, d, y = m2.groups()
                mi = mon_map.get(mon_s.lower())
                if mi:
                    return datetime(int(y), int(mi), int(d))
        except Exception:
            return None
        return None

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for de-duplication (drop query/fragment)."""
        try:
            u = urlparse(url)
            return f"{u.scheme}://{(u.netloc or '').lower()}{u.path}"
        except Exception:
            return url

    def _distinct_domains(self, documents: List[Dict]) -> set:
        """Collect distinct hostnames for diversity metric."""
        ds = set()
        for d in documents:
            u = d.get("url", "")
            try:
                host = urlparse(u).netloc.lower()
                if host:
                    ds.add(host)
            except Exception:
                continue
        return ds

    def _company_domain_hint(self, company_name: str) -> str:
        """Loose hint for company domain; prefer real profile extraction."""
        hint = re.sub(r"[^a-z0-9]", "", company_name.lower())
        return f"{hint}.com"