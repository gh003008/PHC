---
trigger: always_on
---

명령어를 실행하기 전에는 항상 conda activate phc 를 실행해서 conda 환경을 활성화 한 상태에서 진행한다.

분석 요청이 들어오면 `01_research_docs` 폴더에 `YYMMDD_topic_analysis.md` 형식으로 문서를 작성한다.
문서의 내용은 항상 한로 작성한다.
작성된 문서의 내용은 `code-command.md`에 요약하여 추가하지 않는다. (규칙만 추가함)

VIC (Variable Impedance Control) 구현 시, 기존 PHC 스크립트를 직접 수정하지 않고 복사본을 만들어 `_vic` 접미사를 붙여 사용한다 (예: `script_name_vic.py`).
단, Low-level 유틸리티 스크립트는 예외로 하되, 원본 수정 시 반드시 주석으로 `VIC` 관련 수정임을 명시한다.