# agents/ethics_criteria_generator.py

from typing import Dict, List
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from state_schema import PipelineState


class EthicsCriteriaGenerator:
    """
    Domain 기반 윤리 평가 지표 생성 Agent
    
    프로세스:
    1. Domain 기반 쿼리 생성 (예: "medical AI transparency requirements")
    2. BM25로 관련 EU AI Act 조항 10개 검색
    3. 조항 분석하여 5개 카테고리별 평가 기준 생성
    4. 각 기준에 대한 상세 근거와 참고 조항 명시
    """
    
    # 평가 카테고리
    CATEGORIES = {
        "transparency": "투명성",
        "human_oversight": "인간 감독", 
        "data_governance": "데이터 거버넌스",
        "accuracy_validation": "정확도 검증",
        "accountability": "책임성"
    }
    
    def __init__(
        self, 
        openai_api_key: str,
        chroma_path: str = "./chroma/ethics",
        collection_name: str = "EU_ai_act"
    ):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0
        )
        
        # ChromaDB 초기화
        embed_fn = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embed_fn,
            persist_directory=chroma_path
        )
        
        # 모든 문서 로드
        print("📚 EU AI Act 문서 로딩 중...")
        self.all_docs = self._fetch_all_docs()
        print(f"✅ {len(self.all_docs)}개 조항 로드 완료")
        
        # BM25 Retriever 초기화
        self.bm25 = BM25Retriever.from_documents(self.all_docs)
        self.bm25.k = 10  # 카테고리당 10개 조항
    
    def execute(self, state: PipelineState) -> PipelineState:
        """메인 실행: 평가 지표 생성"""
        print("\n" + "="*70)
        print("📋 [윤리 평가 지표 생성 Agent] 시작")
        print("="*70)
        
        domain = state["domain"]
        print(f"🏥 Domain: {domain}")
        
        try:
            # 각 카테고리별 평가 지표 생성
            all_criteria = {}
            
            for category_key, category_name in self.CATEGORIES.items():
                print(f"\n{'='*70}")
                print(f"📋 {category_name} ({category_key}) 평가 지표 생성 중...")
                print(f"{'='*70}")
                
                # 1) Domain + Category 기반 쿼리 생성
                query = self._generate_query(category_key, domain)
                print(f"🔍 검색 쿼리: {query}")
                
                # 2) BM25로 관련 조항 10개 검색
                relevant_articles = self._retrieve_articles(query)
                print(f"📜 검색된 조항: {len(relevant_articles)}개")
                
                # 조항 미리보기
                for i, article in enumerate(relevant_articles[:3], 1):
                    md = article.metadata or {}
                    print(f"   [{i}] Page {md.get('page', 'N/A')}: {article.page_content[:100]}...")
                
                # 3) 조항 기반 평가 기준 생성
                criteria = self._generate_criteria(
                    category_key=category_key,
                    category_name=category_name,
                    domain=domain,
                    articles=relevant_articles
                )
                
                all_criteria[category_key] = criteria
                
                print(f"\n✅ {category_name} 평가 기준:")
                print(f"   생성된 기준: {len(criteria['criteria'])}개")
                for i, crit in enumerate(criteria['criteria'], 1):
                    print(f"   {i}. {crit['title']}")
            
            # State 업데이트
            state["ethics_evaluation_criteria"] = {
                "domain": domain,
                "categories": all_criteria,
                "generated_at": datetime.now().isoformat(),
                "total_criteria_count": sum(
                    len(cat['criteria']) for cat in all_criteria.values()
                )
            }
            
            print(f"\n{'='*70}")
            print(f"🎉 평가 지표 생성 완료!")
            print(f"{'='*70}")
            print(f"📊 통계:")
            print(f"   - 카테고리: {len(all_criteria)}개")
            print(f"   - 총 평가 기준: {state['ethics_evaluation_criteria']['total_criteria_count']}개")
            print(f"   - Domain: {domain}")
            
        except Exception as e:
            print(f"❌ 평가 지표 생성 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            
            state["errors"].append({
                "stage": "ethics_criteria_generation",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "recovered": False,
                "traceback": traceback.format_exc(),
                "impact_on_reliability": "high"
            })
        
        return state
    
    def _fetch_all_docs(self) -> List[Document]:
        """ChromaDB에서 모든 문서 가져오기"""
        col = self.vectordb._collection
        out_docs = []
        total = col.count()
        offset = 0
        batch_size = 1000
        
        while True:
            batch = col.get(
                where={},
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"]
            )
            ids = batch.get("ids") or []
            docs = batch.get("documents") or []
            metas = batch.get("metadatas") or []
            
            if not ids:
                break
            
            for txt, md in zip(docs, metas):
                out_docs.append(Document(page_content=txt or "", metadata=md or {}))
            
            offset += len(ids)
            if offset >= total:
                break
        
        return out_docs
    
    def _generate_query(self, category: str, domain: str) -> str:
        """Domain + Category 기반 검색 쿼리 생성"""
        
        # Domain별 키워드
        domain_keywords = {
            "medical": "healthcare medical diagnosis clinical patient safety",
            "finance": "financial credit risk discrimination fair lending",
            "recruitment": "employment hiring discrimination bias",
            "law_enforcement": "criminal justice predictive policing",
            "education": "student assessment learning profiling"
        }
        
        # Category별 키워드
        category_keywords = {
            "transparency": "transparency explainability disclosure information",
            "human_oversight": "human oversight supervision intervention control",
            "data_governance": "data quality bias training dataset privacy",
            "accuracy_validation": "accuracy performance validation testing evaluation",
            "accountability": "accountability liability responsibility redress"
        }
        
        domain_kw = domain_keywords.get(domain, "AI system")
        category_kw = category_keywords.get(category, "requirements")
        
        # 쿼리 생성
        query = f"EU AI Act {category_kw} requirements for {domain_kw} high-risk AI systems"
        
        return query
    
    def _retrieve_articles(self, query: str) -> List[Document]:
        """BM25로 관련 조항 검색"""
        results = self.bm25.get_relevant_documents(query)
        return results[:10]  # 최대 10개
    
    def _generate_criteria(
        self,
        category_key: str,
        category_name: str,
        domain: str,
        articles: List[Document]
    ) -> Dict:
        """조항 기반 평가 기준 생성"""
        
        # 조항 텍스트 준비
        article_texts = []
        for i, article in enumerate(articles, 1):
            md = article.metadata or {}
            article_texts.append(
                f"[조항 {i}]\n"
                f"출처: {md.get('source', 'EU AI Act')}\n"
                f"페이지: {md.get('page', 'N/A')}\n"
                f"내용:\n{article.page_content}\n"
            )
        
        # 평가 기준 생성 프롬프트
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 EU AI Act 전문가입니다. 
제공된 EU AI Act 조항들을 분석하여 {domain} 도메인의 AI 시스템을 평가하기 위한 
구체적인 평가 기준(criteria)을 생성하세요.

중요 지침:
1. 제공된 조항에서만 평가 기준을 도출하세요
2. 각 기준은 측정 가능하고 구체적이어야 합니다
3. 5-7개의 핵심 기준을 생성하세요
4. 각 기준에 대해:
   - 왜 이 기준이 중요한지
   - 어떤 조항에서 도출되었는지
   - 구체적으로 무엇을 평가해야 하는지
   상세히 설명하세요

JSON 형식으로 반환:
{{
    "category": "{category_name}",
    "domain": "{domain}",
    "criteria": [
        {{
            "id": "unique_id",
            "title": "평가 기준 제목 (간결하게)",
            "description": "이 기준이 무엇을 평가하는지 설명 (100자 이내)",
            "importance": "왜 이 기준이 {domain} AI에 중요한지 설명 (200자)",
            "measurement": "구체적으로 무엇을 확인해야 하는지 (예: 문서 존재 여부, 프로세스 유무 등)",
            "referenced_articles": [
                {{
                    "article_number": 1,
                    "excerpt": "해당 조항의 핵심 문구 (100자 이내)",
                    "relevance": "이 조항이 왜 이 기준의 근거가 되는지 설명 (150자)"
                }}
            ],
            "weight": 0.0-1.0,
            "pass_threshold": "이 기준을 통과하기 위한 최소 요구사항"
        }}
    ],
    "overall_rationale": "이 카테고리의 평가 기준들이 왜 {domain} AI에 중요한지 종합 설명 (300자)"
}}"""),
            ("user", """카테고리: {category_name} ({category_key})
도메인: {domain}

=== EU AI ACT 조항들 ===
{articles}

위 조항들을 분석하여 평가 기준을 생성하세요:""")
        ])
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "category_key": category_key,
            "category_name": category_name,
            "domain": domain,
            "articles": "\n\n".join(article_texts)
        })
        
        # JSON 파싱
        try:
            import json
            import re
            
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = json.loads(response.content)
            
            # 메타데이터 추가
            result["source_articles_count"] = len(articles)
            result["generated_at"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            print(f"    ⚠️ 기준 파싱 실패: {str(e)}")
            return {
                "category": category_name,
                "domain": domain,
                "criteria": [],
                "overall_rationale": "생성 실패",
                "source_articles_count": 0
            }