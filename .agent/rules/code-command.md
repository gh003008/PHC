---
trigger: always_on
---

# PHC Project Rules

## 1. General Rules
- 모든 명령어 실행 전에는 반드시 'conda activate phc'를 실행하여 환경을 활성화합니다.
- 명령어가 헷갈릴 경우 반드시 `00_basic/commands` 파일을 먼저 확인한다. 실제 실행에 사용된 명령어들이 기록되어 있다.
- VIC 학습 명령어 기본 형식 (run.py 사용):
  ```
  python phc/run.py --task HumanoidImVIC --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml --cfg_train phc/data/cfg/learning/im_walk_vic.yaml --headless --num_envs 512
  ```
- VIC 평가 명령어 기본 형식:
  ```
  python phc/run.py --task HumanoidImVIC --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml --cfg_train phc/data/cfg/learning/im_walk_vic.yaml --num_envs 1 --test --epoch -1 --no_virtual_display
  ```

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

## 4. Evaluation Result Analysis

학습 평가(test 모드)가 완료된 후, 반드시 `02_research_dev/` 폴더에 결과 분석 문서를 작성한다.
- 파일명 형식: `YYMMDD_forward_walk_vicXX_result_analysis.md`
- 예시: `02_research_dev/260304_forward_walk_vic07_result_analysis.md`
- 내용: 실험 설정 요약, 정량적 평가 결과(avg reward/steps, 이전 실험과 비교 표), 시각화 관찰, 원인 분석, 다음 실험 방향
- 사용자가 별도로 요청하지 않아도 평가 결과가 나오면 자동으로 작성한다.

## 5. Experiment Config Backup
- 학습 실행 전, 해당 실험의 설정 파일들을 `exp_config/forward_walking/YYMMDD_실험이름/` 폴더에 백업한다.
- 백업 대상: 환경 설정(env yaml), 학습 설정(learning yaml), 태스크 코드(humanoid_im_vic.py 등) 실제 학습에 사용되는 파일들.
- 예를 들어 env_im_walk_vic.yaml, im_walk.yaml, humnanoid_im_vic.py, smpl_humanoid.xml 등.. 수정된 파일 대상으로 스냅샷 필수
- 학습에 사용되는 원본 파일은 계속 수정하며 사용하므로, 각 실험 시점의 스냅샷을 보존하기 위한 목적이다.
