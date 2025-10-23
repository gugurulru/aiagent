"""
LangGraph State Schema (신뢰성 & 출처 추적 강화 버전)
모든 에이전트가 공유하는 State 정의만 포함
"""

from typing import TypedDict, List, Dict, Literal, Annotated, Optional
from datetime import datetime
import operator


# ========================================
# 서브 스키마 (Document, Evidence 등)
# ========================================

class Document(TypedDict):
    """개별 문서 구조"""
    id: str
    url: str
    title: str
    content: str
    excerpt: str
    
    # 출처 분류
    source_type: Literal["primary", "secondary"]
    source_category: Literal["company", "news", "academic", "regulatory", "blog", "social_media"]
    publisher: Optional[str]
    author: Optional[str]
    
    # 최신성
    date: Optional[str]
    age_days: int
    is_recent: bool
    
    # 신뢰성
    reliability: Literal["high", "medium", "low"]
    reliability_score: float
    reliability_reasons: List[str]
    evidence_tier: Literal["tier1", "tier2", "tier3"]
    
    # 검증
    is_verified: bool
    verified_by: List[str]
    
    # 메타
    collected_by: str
    collected_at: str


class CollectionResult(TypedDict):
    """수집 결과 구조"""
    status: Literal["completed", "failed", "partial"]
    documents: List[Document]
    count: int
    sources: List[str]
    
    # 출처 상세 추적
    sources_breakdown: Dict[str, int]
    primary_sources_count: int
    secondary_sources_count: int
    
    # 최신성 추적
    newest_date: Optional[str]
    oldest_date: Optional[str]
    recent_documents_count: int
    
    # 신뢰성 추적
    high_reliability_count: int
    medium_reliability_count: int
    low_reliability_count: int
    average_reliability: float
    
    error: Optional[str]


class QualityScoreBreakdown(TypedDict):
    """품질 점수 상세"""
    primary_sources: List[str]
    secondary_sources: List[str]
    sources_by_category: Dict[str, List[str]]
    topics_covered: List[str]
    missing_topics: List[str]
    
    # 최신성 상세
    recent_documents: List[str]
    outdated_documents: List[str]
    recency_threshold_days: int
    
    # 신뢰성 상세
    high_reliability_docs: List[str]
    low_reliability_docs: List[str]
    verified_documents: List[str]


class QualityScore(TypedDict):
    """데이터 품질 점수"""
    diversity: int
    primary_ratio: int
    recency: int
    total: int
    breakdown: QualityScoreBreakdown
    overall_reliability: float
    confidence: float


class KeyFact(TypedDict):
    """핵심 사실"""
    fact: str
    source: str
    confidence: float
    evidence: List[str]
    
    # 출처 추적 강화
    source_documents: List[str]
    evidence_tier: Literal["tier1", "tier2", "tier3"]
    is_cross_validated: bool
    validation_sources: List[str]


class Risk(TypedDict):
    """리스크"""
    type: Literal["bias", "accuracy", "privacy", "safety", "transparency", "accountability", "other"]
    severity: Literal["critical", "high", "medium", "low"]
    description: str
    evidence: List[str]
    source_documents: List[str]
    identified_by: str


class AnalysisResult(TypedDict):
    """분석 결과"""
    summary: str
    key_facts: List[KeyFact]
    risks: List[Risk]
    business_model: str
    target_users: List[str]
    technology_stack: List[str]
    sources_used: List[str]
    primary_sources_used: List[str]
    total_sources_count: int


class AnalysisScore(TypedDict):
    """분석 신뢰도 점수"""
    evidence_strength: int
    cross_validation: int
    total: int
    confidence: float
    confidence_factors: Dict[str, float]


class EthicsEvidence(TypedDict):
    """윤리 평가 근거"""
    finding: str
    source: str
    url: str
    tier: Literal["tier1", "tier2", "tier3"]
    weight: float
    source_document_id: str
    evidence_type: Literal["direct", "indirect", "inferred"]
    reliability: Literal["high", "medium", "low"]


class EthicsCategoryEvaluation(TypedDict):
    """윤리 평가 항목별 상세"""
    score: int
    level: int
    confidence: float
    evidence: List[EthicsEvidence]
    issues: List[str]
    strengths: List[str]
    evidence_count: int
    tier1_evidence_count: int
    direct_evidence_count: int
    information_availability: Literal["abundant", "sufficient", "limited", "none"]
    limitations: List[str]


class EthicsEvaluation(TypedDict):
    """윤리 평가 전체"""
    transparency: EthicsCategoryEvaluation
    human_oversight: EthicsCategoryEvaluation
    data_governance: EthicsCategoryEvaluation
    accuracy_validation: EthicsCategoryEvaluation
    accountability: EthicsCategoryEvaluation
    all_sources_used: List[str]
    total_evidence_count: int


class EthicsScore(TypedDict):
    """윤리 평가 점수"""
    transparency: int
    human_oversight: int
    data_governance: int
    accuracy_validation: int
    accountability: int
    total: int
    grade: str
    overall_confidence: float
    confidence_by_category: Dict[str, float]
    average_evidence_quality: float


class CriticalIssue(TypedDict):
    """치명적 이슈"""
    severity: Literal["critical", "high", "medium", "low"]
    category: str
    description: str
    evidence: List[str]
    recommendation: str
    eu_ai_act_article: Optional[str]
    source_documents: List[str]
    evidence_quality: Literal["strong", "moderate", "weak"]
    confidence: float


class MediaSentiment(TypedDict):
    """언론 감성 분석"""
    score: float
    positive_count: int
    negative_count: int
    neutral_count: int
    key_articles: List[Dict]
    analyzed_articles: List[str]
    high_impact_articles: List[str]
    sentiment_confidence: float


class ExpertOpinion(TypedDict):
    """전문가 의견"""
    expert: str
    stance: Literal["positive", "negative", "neutral"]
    quote: str
    source: str
    source_document_id: str
    expert_credibility: Literal["high", "medium", "low"]
    verification_status: Literal["verified", "unverified"]


class Controversy(TypedDict):
    """논란"""
    title: str
    description: str
    severity: Literal["high", "medium", "low"]
    sources: List[str]
    source_documents: List[str]
    corroboration_level: Literal["multiple_sources", "single_source", "unverified"]


class SocialAnalysis(TypedDict):
    """사회 반응 분석"""
    media_sentiment: MediaSentiment
    expert_opinions: List[ExpertOpinion]
    controversies: List[Controversy]
    regulatory_actions: List[Dict]
    all_sources_analyzed: List[str]


class SocialScore(TypedDict):
    """사회 신뢰도 점수"""
    media_reputation: int
    expert_trust: int
    controversy_level: int
    total: int
    analysis_confidence: float
    data_recency: Literal["recent", "outdated", "mixed"]


class FinalScores(TypedDict):
    """최종 종합 점수"""
    data_quality: int
    analysis_reliability: int
    ethics: int
    social_trust: int
    total: int
    grade: str
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    overall_confidence: float
    confidence_breakdown: Dict[str, float]
    total_sources_used: int
    primary_sources_ratio: float
    average_source_reliability: float
    oldest_source_age_days: int
    newest_source_age_days: int


class SourceSummary(TypedDict):
    """출처 종합 요약"""
    all_document_ids: List[str]
    documents_by_stage: Dict[str, List[str]]
    citations: List[Dict]


class ReportContent(TypedDict):
    """보고서 내용"""
    executive_summary: str
    detailed_findings: Dict
    comparison_benchmark: Optional[Dict]
    recommendations: List[str]
    limitations: List[str]
    sources_section: SourceSummary
    reliability_statement: str
    confidence_caveats: List[str]


class ErrorLog(TypedDict):
    """에러 로그"""
    stage: str
    error: str
    timestamp: str
    recovered: bool
    traceback: Optional[str]
    impact_on_reliability: Literal["none", "low", "medium", "high"]


class ExecutionLog(TypedDict):
    """실행 로그"""
    timestamp: str
    stage: str
    agent: str
    action: str
    duration_sec: float
    documents_processed: List[str]
    sources_added: List[str]


# ========================================
# 메인 State
# ========================================

class PipelineState(TypedDict):
    """
    LangGraph 파이프라인 전체 State
    """
    
    # 메타 정보
    run_id: str
    company_name: str
    domain: str
    start_time: str
    current_stage: str
    
    # Phase 1: 병렬 수집
    web_collection: CollectionResult
    specialized_collection: CollectionResult
    
    # Phase 2: 데이터 통합
    merged_documents: List[Document]
    quality_score: QualityScore
    is_data_sufficient: bool
    retry_collection: int
    collection_focus: List[str]
    
    # Phase 3: 분석
    analysis_result: AnalysisResult
    analysis_score: AnalysisScore
    is_analysis_sufficient: bool
    retry_analysis: int

    # Phase 3.5: 평가 기준 생성
    ethics_evaluation_criteria: Dict 

    # Phase 4: 윤리 평가
    ethics_evaluation: EthicsEvaluation
    ethics_score: EthicsScore
    critical_issues: List[CriticalIssue]
    
    # Phase 5: 사회 반응
    social_analysis: SocialAnalysis
    social_score: SocialScore
    
    # Phase 6: 최종 보고서
    final_scores: FinalScores
    report_content: ReportContent
    report_path: Optional[str]
    
    # 출처 추적 (전역)
    source_summary: SourceSummary
    
    # 메타: 경고 & 에러
    warnings: Annotated[List[str], operator.add]
    limitations: Annotated[List[str], operator.add]
    errors: Annotated[List[ErrorLog], operator.add]
    execution_log: Annotated[List[ExecutionLog], operator.add]


# ========================================
# 초기 State 생성
# ========================================

def create_initial_state(company_name: str, domain: str = "medical") -> PipelineState:
    """초기 State 생성"""
    import uuid
    
    return {
        "run_id": str(uuid.uuid4()),
        "company_name": company_name,
        "domain": domain,
        "start_time": datetime.now().isoformat(),
        "current_stage": "init",
        
        "web_collection": {
            "status": "pending",
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
            "error": None
        },
        
        "specialized_collection": {
            "status": "pending",
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
            "error": None
        },
        
        "merged_documents": [],
        
        "quality_score": {
            "diversity": 0,
            "primary_ratio": 0,
            "recency": 0,
            "total": 0,
            "breakdown": {
                "primary_sources": [],
                "secondary_sources": [],
                "sources_by_category": {},
                "topics_covered": [],
                "missing_topics": [],
                "recent_documents": [],
                "outdated_documents": [],
                "recency_threshold_days": 180,
                "high_reliability_docs": [],
                "low_reliability_docs": [],
                "verified_documents": []
            },
            "overall_reliability": 0.0,
            "confidence": 0.0
        },
        
        "is_data_sufficient": False,
        "retry_collection": 0,
        "collection_focus": [],
        
        "analysis_result": {
            "summary": "",
            "key_facts": [],
            "risks": [],
            "business_model": "",
            "target_users": [],
            "technology_stack": [],
            "sources_used": [],
            "primary_sources_used": [],
            "total_sources_count": 0
        },
        
        "analysis_score": {
            "evidence_strength": 0,
            "cross_validation": 0,
            "total": 0,
            "confidence": 0.0,
            "confidence_factors": {}
        },
        
        "is_analysis_sufficient": False,
        "retry_analysis": 0,
        
        # Phase 3.5: 평가 기준 생성
        "ethics_evaluation_criteria": {},  # ← 이거 추가!
        
        "ethics_evaluation": {
            "transparency": _create_empty_ethics_category(),
            "human_oversight": _create_empty_ethics_category(),
            "data_governance": _create_empty_ethics_category(),
            "accuracy_validation": _create_empty_ethics_category(),
            "accountability": _create_empty_ethics_category(),
            "all_sources_used": [],
            "total_evidence_count": 0
        },
        
        "ethics_score": {
            "transparency": 0,
            "human_oversight": 0,
            "data_governance": 0,
            "accuracy_validation": 0,
            "accountability": 0,
            "total": 0,
            "grade": "N/A",
            "overall_confidence": 0.0,
            "confidence_by_category": {},
            "average_evidence_quality": 0.0
        },
        
        "critical_issues": [],
        
        "social_analysis": {
            "media_sentiment": {
                "score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "key_articles": [],
                "analyzed_articles": [],
                "high_impact_articles": [],
                "sentiment_confidence": 0.0
            },
            "expert_opinions": [],
            "controversies": [],
            "regulatory_actions": [],
            "all_sources_analyzed": []
        },
        
        "social_score": {
            "media_reputation": 0,
            "expert_trust": 0,
            "controversy_level": 0,
            "total": 0,
            "analysis_confidence": 0.0,
            "data_recency": "mixed"
        },
        
        "final_scores": {
            "data_quality": 0,
            "analysis_reliability": 0,
            "ethics": 0,
            "social_trust": 0,
            "total": 0,
            "grade": "N/A",
            "risk_level": "UNKNOWN",
            "overall_confidence": 0.0,
            "confidence_breakdown": {},
            "total_sources_used": 0,
            "primary_sources_ratio": 0.0,
            "average_source_reliability": 0.0,
            "oldest_source_age_days": 0,
            "newest_source_age_days": 0
        },
        
        "report_content": {
            "executive_summary": "",
            "detailed_findings": {},
            "comparison_benchmark": None,
            "recommendations": [],
            "limitations": [],
            "sources_section": {
                "all_document_ids": [],
                "documents_by_stage": {},
                "citations": []
            },
            "reliability_statement": "",
            "confidence_caveats": []
        },
        
        "report_path": None,
        
        "source_summary": {
            "all_document_ids": [],
            "documents_by_stage": {
                "collection": [],
                "analysis": [],
                "ethics": [],
                "social": []
            },
            "citations": []
        },
        
        "warnings": [],
        "limitations": [],
        "errors": [],
        "execution_log": []
    }


def _create_empty_ethics_category() -> EthicsCategoryEvaluation:
    """윤리 평가 카테고리 초기값"""
    return {
        "score": 0,
        "level": 0,
        "confidence": 0.0,
        "evidence": [],
        "issues": [],
        "strengths": [],
        "evidence_count": 0,
        "tier1_evidence_count": 0,
        "direct_evidence_count": 0,
        "information_availability": "none",
        "limitations": []
    }