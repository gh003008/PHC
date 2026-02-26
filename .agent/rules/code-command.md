---
trigger: always_on
---

# PHC Project Rules

## 1. General Rules
- 모든 명령어 실행 전에는 반드시 'conda activate phc'를 실행하여 환경을 활성화합니다.

## 2. Documentation Rules
성격에 맞는 폴더에 문서를 작성하며 파일명은 'YYMMDD_topic_name.extension' 형식을 따릅니다. 모든 문서는 한글로 작성합니다.

- 01_research_docs: 각 학습(실험) 세팅, 구현 방식, 학습 전략 분석 및 정리.
- 02_research_dev: 학습 결과(Result) 및 상세 결과 분석.
- 03_궁금한 것들 정리: 개발 및 연구 과정에서의 질문과 답변 정리.

텍스트(.txt) 파일 작성 시 주의사항:
- 가독성을 위해 '###', '**', '####' 등 마크다운 언어 형식을 사용하지 않습니다.
- 문장 중심의 정갈하고 읽기 편한 텍스트 형식을 유지합니다.
- 작성된 문서 내용을 본 규칙 파일에 요약하지 않습니다.

## 3. VIC (Variable Impedance Control) Implementation
- VIC (Variable Impedance Control) 구현 시, 기존 PHC 스크립트를 직접 수정하지 않고 복사본을 만들어 `_vic` 접미사를 붙여 사용한다 (예: `script_name_vic.py`).
- 단, Low-level 유틸리티 스크립트는 예외로 하되, 원본 수정 시 반드시 주석으로 `VIC` 관련 수정임을 명시한다.