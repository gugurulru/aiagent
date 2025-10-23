# agents/specialized_collection.py

from typing import Dict, List
from datetime import datetime
import uuid
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from state_schema import PipelineState, Document


class SpecializedCollectionAgent:
    """
    전문 자료 수집 에이전트
    - 학술 논문 (Google Scholar)
    - 규제 문서 (FDA, EU DB)
    - 임상시험 데이터
    """
    
    def __init__(self, tavily_api_key: str, openai_api_key: str):
        self.tavily = TavilyClient(api_key=tavily_api_key)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0
        )
    
    def execute(self, state: PipelineState) -> PipelineState:
        """메인 실행 함수"""
        print("\n" + "="*70)
        print("🎓 [전문 자료 수집 Agent] 시작")
        print("="*70)
        
        company_name = state["company_name"]
        domain = state["domain"]
        
        try:
            documents = []
            
            # 1. 학술 논문 검색
            academic_docs = self._search_academic(company_name, domain)
            documents.extend(academic_docs)
            
            # 2. 규제 문서 검색
            regulatory_docs = self._search_regulatory(company_name, domain)
            documents.extend(regulatory_docs)
            
            # 3. 임상시험 데이터 검색 (추가)
            clinical_docs = self._search_clinical_trials(company_name, domain)
            documents.extend(clinical_docs)
            
            # 4. 신뢰도 평가
            final_docs = self._assess_reliability(documents)
            
            # 5. 통계 계산
            stats = self._calculate_statistics(final_docs)
            
            # 6. State 업데이트
            state["specialized_collection"] = {
                "status": "completed",
                "documents": final_docs,
                "count": len(final_docs),
                "sources": list(set(doc["source_category"] for doc in final_docs)),
                **stats,
                "error": None
            }
            
            print(f"✅ 전문 자료 수집 완료: {len(final_docs)}개 문서")
            print(f"  📚 학술 논문: {sum(1 for d in final_docs if d['source_category']=='academic')}개")
            print(f"  📋 규제 문서: {sum(1 for d in final_docs if d['source_category']=='regulatory')}개")
            print(f"  🧪 임상시험: {sum(1 for d in final_docs if d['source_category']=='clinical_trial')}개")
            
        except Exception as e:
            print(f"❌ 전문 자료 수집 실패: {str(e)}")
            state["specialized_collection"] = {
                "status": "failed",
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
                "error": str(e)
            }
            
            state["errors"].append({
                "stage": "specialized_collection",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "recovered": False,
                "traceback": None,
                "impact_on_reliability": "medium"
            })
        
        return state
    
    def _search_academic(self, company_name: str, domain: str) -> List[Dict]:
        """학술 논문 검색 (Tavily로 Scholar 검색)"""
        print(f"🔬 학술 논문 검색 중: {company_name}")
        
        # ✅ 쿼리 증가: 2개 → 8개
        queries = [
            f"{company_name} AI {domain} clinical validation study",
            f"{company_name} algorithm accuracy peer reviewed",
            f"{company_name} {domain} machine learning research",
            f"{company_name} diagnostic accuracy study",
            f"{company_name} {domain} clinical trial results",
            f"{company_name} AI medical imaging validation",
            f"{company_name} {domain} retrospective study",
            f"{company_name} algorithm performance evaluation",
        ]
        
        all_docs = []
        
        for query in queries:
            try:
                # Tavily로 Scholar 검색 (도메인 제한)
                response = self.tavily.search(
                    query=query,
                    search_depth="advanced",
                    max_results=5,  # ✅ 3 → 5로 증가
                    include_domains=["scholar.google.com", "pubmed.ncbi.nlm.nih.gov", "arxiv.org"]
                )
                
                for result in response.get("results", []):
                    doc = {
                        "id": f"spec_{uuid.uuid4().hex[:8]}",
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "content": result.get("raw_content", result.get("content", "")),
                        "excerpt": result.get("content", "")[:200],
                        "source_type": "primary",  # 학술 논문은 1차 출처
                        "source_category": "academic",
                        "publisher": self._extract_publisher(result.get("url", "")),
                        "author": None,
                        "date": None,  # 추출 필요
                        "age_days": 0,
                        "is_recent": False,
                        "reliability": "high",  # 학술 논문은 기본 high
                        "reliability_score": 0.9,
                        "reliability_reasons": ["학술 논문", "동료 검토"],
                        "evidence_tier": "tier1",
                        "is_verified": False,
                        "verified_by": [],
                        "collected_by": "specialized_collection",
                        "collected_at": datetime.now().isoformat()
                    }
                    all_docs.append(doc)
                
                print(f"  ✓ '{query[:50]}...' → {len(response.get('results', []))}개")
                
            except Exception as e:
                print(f"  ✗ '{query[:50]}...' 실패: {str(e)}")
                continue
        
        print(f"📚 학술 논문 총 {len(all_docs)}개 수집")
        return all_docs
    
    def _search_regulatory(self, company_name: str, domain: str) -> List[Dict]:
        """규제 문서 검색"""
        print(f"📋 규제 문서 검색 중: {company_name}")
        
        # ✅ 쿼리 증가: 2개 → 6개
        queries = [
            f"{company_name} FDA approval CE mark",
            f"{company_name} EU MDR medical device",
            f"{company_name} FDA 510k clearance",
            f"{company_name} CE marking certificate",
            f"{company_name} medical device regulation approval",
            f"{company_name} regulatory submission documentation",
        ]
        
        all_docs = []
        
        for query in queries:
            try:
                response = self.tavily.search(
                    query=query,
                    search_depth="advanced",
                    max_results=10,  # ✅ 3 → 5로 증가
                    include_domains=["fda.gov", "ec.europa.eu", "ema.europa.eu"]
                )
                
                for result in response.get("results", []):
                    doc = {
                        "id": f"spec_{uuid.uuid4().hex[:8]}",
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "content": result.get("raw_content", result.get("content", "")),
                        "excerpt": result.get("content", "")[:200],
                        "source_type": "primary",
                        "source_category": "regulatory",
                        "publisher": "FDA" if "fda.gov" in result.get("url", "") else "EU",
                        "author": None,
                        "date": None,
                        "age_days": 0,
                        "is_recent": False,
                        "reliability": "high",
                        "reliability_score": 0.95,
                        "reliability_reasons": ["규제 기관 공식 문서", "1차 출처"],
                        "evidence_tier": "tier1",
                        "is_verified": False,
                        "verified_by": [],
                        "collected_by": "specialized_collection",
                        "collected_at": datetime.now().isoformat()
                    }
                    all_docs.append(doc)
                
                print(f"  ✓ '{query[:50]}...' → {len(response.get('results', []))}개")
                
            except Exception as e:
                print(f"  ✗ '{query[:50]}...' 실패: {str(e)}")
                continue
        
        print(f"📋 규제 문서 총 {len(all_docs)}개 수집")
        return all_docs
    
    def _search_clinical_trials(self, company_name: str, domain: str) -> List[Dict]:
        """임상시험 데이터 검색 (추가)"""
        print(f"🧪 임상시험 데이터 검색 중: {company_name}")
        
        queries = [
            f"{company_name} clinical trial {domain}",
            f"{company_name} clinicaltrials.gov study",
            f"{company_name} {domain} trial results",
            f"{company_name} multicenter trial",
        ]
        
        all_docs = []
        
        for query in queries:
            try:
                response = self.tavily.search(
                    query=query,
                    search_depth="advanced",
                    max_results=5,
                    include_domains=["clinicaltrials.gov", "who.int", "nih.gov"]
                )
                
                for result in response.get("results", []):
                    doc = {
                        "id": f"spec_{uuid.uuid4().hex[:8]}",
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "content": result.get("raw_content", result.get("content", "")),
                        "excerpt": result.get("content", "")[:200],
                        "source_type": "primary",
                        "source_category": "clinical_trial",
                        "publisher": "ClinicalTrials.gov" if "clinicaltrials.gov" in result.get("url", "") else "WHO",
                        "author": None,
                        "date": None,
                        "age_days": 0,
                        "is_recent": False,
                        "reliability": "high",
                        "reliability_score": 0.92,
                        "reliability_reasons": ["임상시험 등록 데이터", "1차 출처"],
                        "evidence_tier": "tier1",
                        "is_verified": False,
                        "verified_by": [],
                        "collected_by": "specialized_collection",
                        "collected_at": datetime.now().isoformat()
                    }
                    all_docs.append(doc)
                
                print(f"  ✓ '{query[:50]}...' → {len(response.get('results', []))}개")
                
            except Exception as e:
                print(f"  ✗ '{query[:50]}...' 실패: {str(e)}")
                continue
        
        print(f"🧪 임상시험 데이터 총 {len(all_docs)}개 수집")
        return all_docs
    
    def _extract_publisher(self, url: str) -> str:
        """URL에서 발행처 추출"""
        if "pubmed" in url:
            return "PubMed"
        elif "scholar.google" in url:
            return "Google Scholar"
        elif "arxiv" in url:
            return "arXiv"
        elif "nature.com" in url:
            return "Nature"
        elif "clinicaltrials.gov" in url:
            return "ClinicalTrials.gov"
        else:
            return "Academic"
    
    def _assess_reliability(self, documents: List[Dict]) -> List[Document]:
        """신뢰도 재평가"""
        # 전문 자료는 이미 높은 신뢰도지만 최신성 체크
        for doc in documents:
            if doc["date"]:
                try:
                    doc_date = datetime.fromisoformat(doc["date"])
                    age_days = (datetime.now() - doc_date).days
                    doc["age_days"] = age_days
                    doc["is_recent"] = age_days <= 180
                except:
                    doc["age_days"] = 999
                    doc["is_recent"] = False
        
        return documents
    
    def _calculate_statistics(self, documents: List[Document]) -> Dict:
        """통계 계산 (web_collection과 동일)"""
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
            cat = doc["source_category"]
            stats["sources_breakdown"][cat] = stats["sources_breakdown"].get(cat, 0) + 1
            
            if doc["source_type"] == "primary":
                stats["primary_sources_count"] += 1
            else:
                stats["secondary_sources_count"] += 1
            
            if doc["is_recent"]:
                stats["recent_documents_count"] += 1
            
            if doc["reliability"] == "high":
                stats["high_reliability_count"] += 1
            elif doc["reliability"] == "medium":
                stats["medium_reliability_count"] += 1
            else:
                stats["low_reliability_count"] += 1
        
        dates = [doc["date"] for doc in documents if doc["date"]]
        if dates:
            stats["newest_date"] = max(dates)
            stats["oldest_date"] = min(dates)
        
        if len(documents) > 0:
            avg_rel = sum(doc["reliability_score"] for doc in documents) / len(documents)
            stats["average_reliability"] = round(avg_rel, 2)
        
        return stats