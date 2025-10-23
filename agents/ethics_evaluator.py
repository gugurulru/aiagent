# agents/ethics_evaluator.py

from typing import Dict, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from state_schema import PipelineState, EthicsEvidence, EthicsCategoryEvaluation


class EthicsEvaluator:
    """
    생성된 평가 기준을 바탕으로 실제 평가 수행 Agent
    
    프로세스:
    1. State에서 평가 기준 로드 (ethics_evaluation_criteria)
    2. State에서 수집된 문서 로드 (web_collection + specialized_collection)
    3. 각 기준별로 문서 분석 및 점수 부여
    4. 상세한 근거와 함께 평가 결과 저장
    """
    
    # 평가 카테고리
    CATEGORIES = [
        "transparency",
        "human_oversight", 
        "data_governance",
        "accuracy_validation",
        "accountability"
    ]
    
    # 재수집 임계값
    MIN_CONFIDENCE_THRESHOLD = 0.6
    MIN_EVIDENCE_COUNT = 3
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0
        )
    
    def execute(self, state: PipelineState) -> PipelineState:
        """메인 실행: 평가 기준 기반 실제 평가"""
        print("\n" + "="*70)
        print("⚖️ [윤리 평가 실행 Agent] 시작")
        print("="*70)
        
        company_name = state["company_name"]
        domain = state["domain"]
        
        # 1. 평가 기준 확인
        if "ethics_evaluation_criteria" not in state:
            print("❌ 평가 기준이 없습니다. 먼저 EthicsCriteriaGenerator를 실행하세요.")
            state["errors"].append({
                "stage": "ethics_evaluation",
                "error": "평가 기준 없음",
                "timestamp": datetime.now().isoformat(),
                "recovered": False,
                "traceback": None,
                "impact_on_reliability": "high"
            })
            return state
        
        criteria = state["ethics_evaluation_criteria"]
        print(f"✅ 평가 기준 로드: {criteria['total_criteria_count']}개")
        
        # 2. 수집된 문서 확인
        web_docs = state.get("web_collection", {}).get("documents", [])
        spec_docs = state.get("specialized_collection", {}).get("documents", [])
        all_docs = web_docs + spec_docs
        
        print(f"📚 평가 대상 문서:")
        print(f"   - 웹 수집: {len(web_docs)}개")
        print(f"   - 전문 자료: {len(spec_docs)}개")
        print(f"   - 총: {len(all_docs)}개")
        
        if len(all_docs) == 0:
            print("❌ 평가할 문서가 없습니다.")
            state["errors"].append({
                "stage": "ethics_evaluation",
                "error": "평가 문서 없음",
                "timestamp": datetime.now().isoformat(),
                "recovered": False,
                "traceback": None,
                "impact_on_reliability": "high"
            })
            return state
        
        try:
            # 3. 카테고리별 평가 수행
            evaluations = {}
            all_evidence = []
            
            for category in self.CATEGORIES:
                print(f"\n{'='*70}")
                print(f"📋 {category.upper()} 평가 중...")
                print(f"{'='*70}")
                
                # 해당 카테고리의 평가 기준 가져오기
                category_criteria = criteria['categories'].get(category, {})
                
                if not category_criteria or not category_criteria.get('criteria'):
                    print(f"⚠️ {category} 평가 기준이 없습니다. 스킵합니다.")
                    continue
                
                print(f"📝 평가 기준: {len(category_criteria['criteria'])}개")
                
                # 평가 수행
                evaluation = self._evaluate_category(
                    category=category,
                    company_name=company_name,
                    documents=all_docs,
                    criteria=category_criteria
                )
                
                evaluations[category] = evaluation
                all_evidence.extend(evaluation["evidence"])
                
                print(f"\n📊 평가 결과:")
                print(f"   점수: {evaluation['score']}/100")
                print(f"   신뢰도: {evaluation['confidence']:.2f}")
                print(f"   근거: {len(evaluation['evidence'])}개")
                print(f"   이슈: {len(evaluation['issues'])}개")
                print(f"   강점: {len(evaluation['strengths'])}개")
            
            # 4. 점수 계산
            ethics_score = self._calculate_scores(evaluations)
            
            # 5. Critical Issues 추출
            critical_issues = self._identify_critical_issues(evaluations)
            
            # 6. State 업데이트
            state["ethics_evaluation"] = {
                "transparency": evaluations.get("transparency", self._get_empty_evaluation()),
                "human_oversight": evaluations.get("human_oversight", self._get_empty_evaluation()),
                "data_governance": evaluations.get("data_governance", self._get_empty_evaluation()),
                "accuracy_validation": evaluations.get("accuracy_validation", self._get_empty_evaluation()),
                "accountability": evaluations.get("accountability", self._get_empty_evaluation()),
                "all_sources_used": list(set(e["source_document_id"] for e in all_evidence)),
                "total_evidence_count": len(all_evidence)
            }
            
            state["ethics_score"] = ethics_score
            state["critical_issues"] = critical_issues
            
            # 7. 재수집 필요 여부 판단
            needs_recollection = self._check_recollection_needed(
                ethics_score=ethics_score,
                evaluations=evaluations
            )
            
            if needs_recollection:
                print("\n⚠️ 정보 부족으로 재수집 필요")
                state["is_data_sufficient"] = False
                state["retry_collection"] = state.get("retry_collection", 0) + 1
                state["collection_focus"] = self._suggest_collection_focus(evaluations)
            else:
                print("\n✅ 윤리 평가 완료")
                state["is_data_sufficient"] = True
            
            print(f"\n{'='*70}")
            print(f"🎉 윤리 평가 완료!")
            print(f"{'='*70}")
            print(f"📊 최종 점수: {ethics_score['total']}/100 (등급: {ethics_score['grade']})")
            print(f"💡 전체 신뢰도: {ethics_score['overall_confidence']:.2f}")
            print(f"⚠️ Critical Issues: {len(critical_issues)}개")
            print(f"📝 총 근거: {len(all_evidence)}개")
            
        except Exception as e:
            print(f"❌ 윤리 평가 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            
            state["errors"].append({
                "stage": "ethics_evaluation",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "recovered": False,
                "traceback": traceback.format_exc(),
                "impact_on_reliability": "high"
            })
        
        return state
    
    def _evaluate_category(
        self,
        category: str,
        company_name: str,
        documents: List[Dict],
        criteria: Dict
    ) -> EthicsCategoryEvaluation:
        """카테고리별 실제 평가 수행"""
        
        # 문서 요약 준비
        doc_summaries = []
        for i, doc in enumerate(documents[:50], 1):  # 최대 50개
            summary = (
                f"[문서 {i}]\n"
                f"ID: {doc['id']}\n"
                f"제목: {doc['title']}\n"
                f"출처: {doc['source_category']}\n"
                f"발행처: {doc.get('publisher', 'N/A')}\n"
                f"신뢰도: {doc['reliability']} ({doc['reliability_score']})\n"
                f"내용: {doc.get('excerpt', doc.get('content', ''))[:300]}\n"
            )
            doc_summaries.append(summary)
        
        # 평가 기준 정리
        criteria_list = []
        for i, crit in enumerate(criteria['criteria'], 1):
            criteria_text = (
                f"[기준 {i}] {crit['title']}\n"
                f"ID: {crit['id']}\n"
                f"설명: {crit['description']}\n"
                f"측정 방법: {crit['measurement']}\n"
                f"가중치: {crit['weight']}\n"
                f"통과 기준: {crit['pass_threshold']}\n"
                f"중요성: {crit['importance']}\n"
            )
            criteria_list.append(criteria_text)
        
        # 평가 프롬프트
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 EU AI Act 준수 평가자입니다.

주어진 평가 기준과 회사 문서를 바탕으로 {company_name}의 {category} 준수도를 평가하세요.

평가 지침:
1. 각 평가 기준을 개별적으로 검토하세요
2. 회사 문서에서 각 기준과 관련된 증거를 찾으세요
3. 증거의 질과 양에 따라 점수를 부여하세요
4. 점수 산정:
   - 기준의 가중치를 고려하세요
   - 직접 증거(명시적 언급): 100% 반영
   - 간접 증거(암시적 내용): 60% 반영
   - 증거 없음: 0%
5. 매우 구체적으로 근거를 제시하세요

점수 계산 방법:
- 각 기준별로 0-100점 부여
- 가중치를 적용하여 종합 점수 계산
- 예: 기준1(가중치0.9, 80점) + 기준2(가중치0.7, 60점) + ...

JSON 형식으로 반환:
{{
    "score": 0-100,
    "level": 1-5,
    "confidence": 0.0-1.0,
    "evidence": [
        {{
            "finding": "매우 구체적인 발견사항 - 어떤 문서에서 어떤 내용을 발견했는지",
            "source": "문서 제목",
            "url": "문서 URL (있으면)",
            "tier": "tier1/tier2/tier3",
            "weight": 0.0-1.0,
            "source_document_id": "문서 ID",
            "evidence_type": "direct/indirect/inferred",
            "reliability": "high/medium/low",
            "criterion_id": "평가 기준 ID",
            "criterion_title": "평가 기준 제목",
            "score_contribution": "이 근거가 전체 점수에 기여한 정도 (0-100)",
            "detailed_explanation": "이 근거가 왜 중요한지, 어떻게 평가 기준을 충족/미충족하는지 300자 이상 상세 설명"
        }}
    ],
    "issues": ["구체적 문제점 - 어떤 기준을 충족하지 못했는지"],
    "strengths": ["구체적 강점 - 어떤 기준을 잘 충족했는지"],
    "information_availability": "abundant/sufficient/limited/none",
    "limitations": ["부족한 정보 - 어떤 기준을 평가하기에 정보가 부족한지"],
    "criteria_scores": {{
        "criterion_id": {{
            "score": 0-100,
            "status": "pass/fail/partial",
            "evidence_count": 숫자
        }}
    }}
}}"""),
            ("user", """회사: {company_name}
카테고리: {category}

=== 평가 기준 ===
{criteria}

=== 회사 문서 ===
{documents}

위 평가 기준에 따라 회사를 평가하세요:""")
        ])
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "company_name": company_name,
            "category": category,
            "criteria": "\n\n".join(criteria_list),
            "documents": "\n\n".join(doc_summaries)
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
            
            # 통계 계산
            result["evidence_count"] = len(result.get("evidence", []))
            result["tier1_evidence_count"] = sum(
                1 for e in result.get("evidence", []) 
                if e.get("tier") == "tier1"
            )
            result["direct_evidence_count"] = sum(
                1 for e in result.get("evidence", []) 
                if e.get("evidence_type") == "direct"
            )
            
            return result
            
        except Exception as e:
            print(f"    ⚠️ 평가 파싱 실패: {str(e)}")
            return self._get_empty_evaluation()
    
    def _calculate_scores(self, evaluations: Dict) -> Dict:
        """점수 계산"""
        scores = {}
        
        for cat in self.CATEGORIES:
            if cat in evaluations:
                scores[cat] = evaluations[cat]["score"]
            else:
                scores[cat] = 0
        
        # 전체 평균
        valid_scores = [s for s in scores.values() if s > 0]
        total = sum(valid_scores) // len(valid_scores) if valid_scores else 0
        
        # 등급
        if total >= 90:
            grade = "A+"
        elif total >= 80:
            grade = "A"
        elif total >= 70:
            grade = "B"
        elif total >= 60:
            grade = "C"
        else:
            grade = "D"
        
        # 신뢰도
        confidences = {}
        for cat in self.CATEGORIES:
            if cat in evaluations:
                confidences[cat] = evaluations[cat]["confidence"]
            else:
                confidences[cat] = 0.0
        
        valid_confs = [c for c in confidences.values() if c > 0]
        overall_confidence = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0
        
        # 근거 품질
        total_evidence = sum(
            evaluations[cat]["evidence_count"] 
            for cat in self.CATEGORIES 
            if cat in evaluations
        )
        tier1_evidence = sum(
            evaluations[cat]["tier1_evidence_count"] 
            for cat in self.CATEGORIES 
            if cat in evaluations
        )
        avg_evidence_quality = tier1_evidence / total_evidence if total_evidence > 0 else 0
        
        return {
            **scores,
            "total": total,
            "grade": grade,
            "overall_confidence": round(overall_confidence, 2),
            "confidence_by_category": confidences,
            "average_evidence_quality": round(avg_evidence_quality, 2)
        }
    
    def _identify_critical_issues(self, evaluations: Dict) -> List[Dict]:
        """Critical Issues 추출"""
        critical_issues = []
        
        for category, evaluation in evaluations.items():
            if evaluation["score"] < 50:
                for issue in evaluation.get("issues", []):
                    critical_issues.append({
                        "severity": "critical" if evaluation["score"] < 30 else "high",
                        "category": category,
                        "description": issue,
                        "evidence": [e["finding"] for e in evaluation["evidence"][:3]],
                        "recommendation": f"{category} 개선 필요",
                        "eu_ai_act_article": None,
                        "source_documents": [e["source_document_id"] for e in evaluation["evidence"]],
                        "evidence_quality": "strong" if evaluation["confidence"] > 0.7 else "moderate",
                        "confidence": evaluation["confidence"]
                    })
        
        return critical_issues
    
    def _check_recollection_needed(self, ethics_score: Dict, evaluations: Dict) -> bool:
        """재수집 필요 여부"""
        if ethics_score["overall_confidence"] < self.MIN_CONFIDENCE_THRESHOLD:
            print(f"  → 전체 신뢰도 부족: {ethics_score['overall_confidence']:.2f} < {self.MIN_CONFIDENCE_THRESHOLD}")
            return True
        
        low_evidence_count = sum(
            1 for cat in self.CATEGORIES
            if cat in evaluations and evaluations[cat]["evidence_count"] < self.MIN_EVIDENCE_COUNT
        )
        
        if low_evidence_count >= 2:
            print(f"  → 근거 부족 카테고리: {low_evidence_count}개")
            return True
        
        return False
    
    def _suggest_collection_focus(self, evaluations: Dict) -> List[str]:
        """재수집 집중 영역"""
        focus_areas = []
        
        for category, evaluation in evaluations.items():
            if evaluation["confidence"] < 0.5 or evaluation["evidence_count"] < 3:
                for limitation in evaluation.get("limitations", []):
                    focus_areas.append(f"{category}: {limitation}")
        
        return focus_areas[:5]
    
    def _get_empty_evaluation(self) -> EthicsCategoryEvaluation:
        """빈 평가 결과"""
        return {
            "score": 0,
            "level": 0,
            "confidence": 0.0,
            "evidence": [],
            "issues": ["평가 실패"],
            "strengths": [],
            "evidence_count": 0,
            "tier1_evidence_count": 0,
            "direct_evidence_count": 0,
            "information_availability": "none",
            "limitations": ["평가 데이터 부족"],
            "criteria_scores": {}
        }