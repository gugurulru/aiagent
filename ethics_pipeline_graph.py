# ethics_pipeline_graph.py

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator

from state_schema import PipelineState, create_initial_state
from agents import WebCollectionAgent, SpecializedCollectionAgent
from agents.ethics_criteria_generator import EthicsCriteriaGenerator
from agents.ethics_evaluator import EthicsEvaluator
from agents.report_generator import PDFReportGenerator


class EthicsPipelineGraph:
    """
    LangGraph 기반 윤리 평가 파이프라인
    
    노드:
    1. web_collection: 웹 정보 수집
    2. specialized_collection: 전문 자료 수집
    3. criteria_generation: 평가 기준 생성
    4. evaluation: 윤리 평가 수행
    5. report_generation: PDF 보고서 생성
    """
    
    def __init__(
        self,
        tavily_api_key: str,
        openai_api_key: str
    ):
        self.tavily_api_key = tavily_api_key
        self.openai_api_key = openai_api_key
        
        # Agent 초기화
        self.web_agent = WebCollectionAgent(
            tavily_api_key=tavily_api_key,
            openai_api_key=openai_api_key
        )
        
        self.spec_agent = SpecializedCollectionAgent(
            tavily_api_key=tavily_api_key,
            openai_api_key=openai_api_key
        )
        
        self.criteria_generator = EthicsCriteriaGenerator(
            openai_api_key=openai_api_key
        )
        
        self.evaluator = EthicsEvaluator(
            openai_api_key=openai_api_key
        )
        
        # PDF 보고서 생성 (report_generator가 PDF까지 생성)
        self.report_generator = PDFReportGenerator()
        
        # 그래프 생성
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """LangGraph 생성"""
        
        # StateGraph 초기화
        workflow = StateGraph(PipelineState)
        
        # 노드 추가
        workflow.add_node("web_collection", self._web_collection_node)
        workflow.add_node("specialized_collection", self._specialized_collection_node)
        workflow.add_node("criteria_generation", self._criteria_generation_node)
        workflow.add_node("evaluation", self._evaluation_node)
        workflow.add_node("check_recollection", self._check_recollection_node)
        workflow.add_node("report_generation", self._report_generation_node)
        
        # 엣지 정의
        workflow.set_entry_point("web_collection")
        
        # web_collection -> specialized_collection
        workflow.add_edge("web_collection", "specialized_collection")
        
        # specialized_collection -> criteria_generation
        workflow.add_edge("specialized_collection", "criteria_generation")
        
        # criteria_generation -> evaluation
        workflow.add_edge("criteria_generation", "evaluation")
        
        # evaluation -> check_recollection (재수집 필요 여부 확인)
        workflow.add_edge("evaluation", "check_recollection")
        
        # check_recollection -> 조건부 분기
        workflow.add_conditional_edges(
            "check_recollection",
            self._should_recollect,
            {
                "recollect": "web_collection",     # 재수집
                "continue": "report_generation"    # PDF 보고서 생성
            }
        )
        
        # report_generation -> END
        workflow.add_edge("report_generation", END)
        
        # 컴파일
        return workflow.compile()
    
    # ===== Worker Nodes =====
    
    def _web_collection_node(self, state: PipelineState) -> PipelineState:
        """웹 정보 수집 노드"""
        print("\n" + "="*70)
        print("🌐 [Node 1/6] 웹 정보 수집")
        print("="*70)
        
        state = self.web_agent.execute(state)
        return state
    
    def _specialized_collection_node(self, state: PipelineState) -> PipelineState:
        """전문 자료 수집 노드"""
        print("\n" + "="*70)
        print("🎓 [Node 2/6] 전문 자료 수집")
        print("="*70)
        
        state = self.spec_agent.execute(state)
        return state
    
    def _criteria_generation_node(self, state: PipelineState) -> PipelineState:
        """평가 기준 생성 노드"""
        print("\n" + "="*70)
        print("📋 [Node 3/6] 평가 기준 생성")
        print("="*70)
        
        state = self.criteria_generator.execute(state)
        return state
    
    def _evaluation_node(self, state: PipelineState) -> PipelineState:
        """윤리 평가 수행 노드"""
        print("\n" + "="*70)
        print("⚖️ [Node 4/6] 윤리 평가 수행")
        print("="*70)
        
        state = self.evaluator.execute(state)
        return state
    
    def _check_recollection_node(self, state: PipelineState) -> PipelineState:
        """재수집 필요 여부 확인 노드"""
        print("\n" + "="*70)
        print("🔍 재수집 필요 여부 확인")
        print("="*70)
        
        if not state.get('is_data_sufficient', True):
            print(f"⚠️ 재수집 필요 (시도 횟수: {state.get('retry_collection', 0)})")
            print(f"집중 영역: {state.get('collection_focus', [])}")
        else:
            print("✅ 데이터 충분 - 보고서 생성 진행")
        
        return state
    
    def _report_generation_node(self, state: PipelineState) -> PipelineState:
        """PDF 보고서 생성 노드"""
        print("\n" + "="*70)
        print("📄 [Node 5/5] PDF 보고서 생성")
        print("="*70)
        
        state = self.report_generator.execute(state)
        return state
    
    # ===== Conditional Edges =====
    
    def _should_recollect(self, state: PipelineState) -> str:
        """재수집 필요 여부 판단"""
        
        # 재시도 횟수 제한 (최대 2회)
        max_retries = 2
        retry_count = state.get('retry_collection', 0)
        
        if retry_count >= max_retries:
            print(f"⚠️ 최대 재시도 횟수 도달 ({retry_count}회) - 보고서 생성 진행")
            state['is_data_sufficient'] = True
            return "continue"
        
        # 데이터 충분 여부
        if state.get('is_data_sufficient', True):
            return "continue"
        else:
            return "recollect"
    
    # ===== Main Execution =====
    
    def run(
        self,
        company_name: str,
        domain: str,
        verbose: bool = True
    ) -> PipelineState:
        """파이프라인 실행"""
        
        print("\n" + "="*70)
        print("🚀 윤리 평가 파이프라인 시작")
        print("="*70)
        print(f"회사: {company_name}")
        print(f"도메인: {domain}")
        print("="*70)
        
        # 초기 State 생성
        initial_state = create_initial_state(company_name, domain)
        
        # 그래프 실행
        final_state = self.graph.invoke(initial_state)
        
        # 최종 결과 출력
        self._print_final_results(final_state)
        
        return final_state
    
    def _print_final_results(self, state: PipelineState):
        """최종 결과 출력"""
        
        print("\n" + "="*70)
        print("🎉 파이프라인 완료!")
        print("="*70)
        
        print(f"\n📊 수집 통계:")
        print(f"   웹 문서: {state['web_collection']['count']}개")
        print(f"   전문 자료: {state['specialized_collection']['count']}개")
        print(f"   총: {state['web_collection']['count'] + state['specialized_collection']['count']}개")
        
        print(f"\n📋 평가:")
        print(f"   평가 기준: {state['ethics_evaluation_criteria']['total_criteria_count']}개")
        print(f"   발견 근거: {state['ethics_evaluation']['total_evidence_count']}개")
        
        ethics_score = state['ethics_score']
        print(f"\n⚖️ 윤리 점수:")
        print(f"   투명성: {ethics_score['transparency']}/100")
        print(f"   인간 감독: {ethics_score['human_oversight']}/100")
        print(f"   데이터 거버넌스: {ethics_score['data_governance']}/100")
        print(f"   정확도 검증: {ethics_score['accuracy_validation']}/100")
        print(f"   책임성: {ethics_score['accountability']}/100")
        print(f"   종합: {ethics_score['total']}/100 (등급: {ethics_score['grade']})")
        print(f"   신뢰도: {ethics_score['overall_confidence']:.2f}")
        
        print(f"\n⚠️ Critical Issues: {len(state['critical_issues'])}개")
        
        if 'pdf_report_path' in state:
            print(f"\n📄 생성된 파일:")
            print(f"   PDF: {state['pdf_report_path']}")
        
        print(f"\n🆔 Run ID: {state['run_id']}")
        print("="*70)
    
    def visualize(self, output_path: str = "pipeline_graph.png"):
        """그래프 시각화"""
        try:
            from IPython.display import Image, display
            
            # Mermaid 다이어그램 생성
            graph_image = self.graph.get_graph().draw_mermaid_png()
            
            with open(output_path, 'wb') as f:
                f.write(graph_image)
            
            print(f"✅ 그래프 시각화 저장: {output_path}")
            
            # Jupyter에서 실행 중이면 표시
            try:
                display(Image(graph_image))
            except:
                pass
                
        except Exception as e:
            print(f"⚠️ 그래프 시각화 실패: {e}")
            print("graphviz가 설치되어 있는지 확인하세요.")


# ===== 편의 함수 =====

def run_ethics_pipeline(
    company_name: str,
    domain: str,
    tavily_api_key: str,
    openai_api_key: str,
    visualize: bool = False
) -> PipelineState:
    """
    윤리 평가 파이프라인 실행 (간단한 인터페이스)
    
    Args:
        company_name: 평가 대상 회사명
        domain: 도메인 (medical, finance, recruitment 등)
        tavily_api_key: Tavily API 키
        openai_api_key: OpenAI API 키
        visualize: 그래프 시각화 여부
    
    Returns:
        최종 State
    """
    
    # 파이프라인 생성
    pipeline = EthicsPipelineGraph(
        tavily_api_key=tavily_api_key,
        openai_api_key=openai_api_key
    )
    
    # 시각화
    if visualize:
        pipeline.visualize()
    
    # 실행
    final_state = pipeline.run(company_name, domain)
    
    return final_state