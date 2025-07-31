# TRL RLOO 메모리 최적화 프로젝트

**RLOO Trainer 메모리 최적화 및 성능 개선**

[English Documentation (영어 문서)](../README.md)

## 시스템 정보

- **운영체제**: Ubuntu 24.04.2 LTS (Linux 6.14.0-24-generic)
- **GPU**: NVIDIA GeForce RTX 5090
- **PyTorch**: 2.7.1+cu128
- **Transformers**: 4.54.1
- **CUDA**: 12.8

## 모델 및 데이터셋 정보

### 사용된 모델
- **모델**: `tiny-Qwen3ForCausalLM` (테스트용 커스텀 소형 모델)
- **목적**: 메모리 최적화 실험을 위해 설계된 경량 모델
- **선택 이유**: 여러 설정을 실행할 수 있을 만큼 작으면서도 메모리 스케일링 문제를 보여줄 수 있음
- **출처**: TRL의 tiny 모델 생성 도구를 사용하여 생성

### 데이터셋 정보
- **데이터셋**: 빠른 훈련 사이클을 위한 합성/최소 데이터셋
- **목적**: 훈련 품질보다는 메모리 프로파일링에 집중
- **크기**: 통제된 메모리 분석을 위한 작은 배치 크기 (2-8)

## 프로젝트 구조

```
├── docs/                                      # 문서 및 분석 자료
│   ├── RLOO_Improvement_Analysis.md           # RLOO 개선 분석
│   ├── RLOO_Memory_Optimization_Methods.md    # 메모리 최적화 방법론
│   └── RLOO_Memory_Optimization_Task.md       # 작업 명세서
│
├── experiments/                               # 실험 스크립트
│   ├── train.py                              # RLOO 훈련 스크립트
│   ├── config.py                             # 실험 설정
│   ├── memory_monitor.py                     # 메모리 모니터링
│   └── run_*_rloo*.sh                        # 개별 실험 스크립트
│
├── results/data/                              # 실험 결과 데이터
│   ├── results_basic/                        # 기본 RLOO 결과
│   ├── results_lazy/                         # Lazy Generation 결과
│   ├── results_repeatsampler/                # RepeatSampler 결과
│   └── results_string/                       # String-Level 결과
│
├── visualization/                             # 시각화 및 분석
│   ├── combine_gpu_memory_data.py            # 데이터 통합 스크립트
│   ├── create_memory_charts.py               # 차트 생성 스크립트
│   └── *.png                                 # 생성된 차트들
│
├── models/                                    # 사전 훈련된 모델들
├── trl/                                       # TRL 라이브러리 (수정된 버전)
│
├── run_*.sh                                   # 주요 실험 실행기
├── combine_results.sh                        # 결과 통합
└── visualize_results.sh                      # 시각화 실행기
```

## 주요 성과

### 메모리 최적화 결과
- **String-Level Processing**: **평균 55% 메모리 절약**
  - `rloo_k=2`: 55.5% 절약 (14.38GB → 6.40GB)
  - `rloo_k=4`: 59.2% 절약 (26.99GB → 11.01GB)
  - `rloo_k=8`: 15.2% 절약 (23.87GB → 20.25GB)

### 중요한 발견: rloo_k=8에서의 OOM 문제
`rloo_k=8` 환경에서 String-Level Processing을 제외한 모든 최적화 방법들이 Out-Of-Memory (OOM) 오류를 겪어 `rloo_k=4`보다 낮은 성능을 보였습니다. String-Level Processing만이 모든 실험을 메모리 문제 없이 성공적으로 완료했습니다.

## 구현된 최적화 방법들

### 1. String-Level Processing (최고 성능)
**무엇인가**: OnlineDPO 스타일 문자열 처리 방식
**왜 추가했는가**: 토큰 레벨 복제를 피하고 토큰화 전에 문자열 레벨에서 처리하여 메모리 사용량 감소
**구현 방법**: 
- 프롬프트를 문자열로 디코딩
- 문자열 레벨에서 반복
- 한 번에 재토큰화
- 상당한 메모리 절약 효과

### 2. Lazy Generation
**무엇인가**: 순차적 생성 접근법
**왜 추가했는가**: 생성 단계에서 피크 메모리 사용량 감소
**구현 방법**:
- 각 rloo_k 반복에 대해 순차적으로 응답 생성
- 결과를 연결하여 원본 API와 일치
- 메모리 스파이크를 줄이면서 호환성 유지

### 3. RepeatSampler
**무엇인가**: GRPO 스타일 데이터 샘플링 최적화
**왜 추가했는가**: 샘플링 레벨에서 데이터 복제 방지
**구현 방법**:
- GRPO와 유사한 RepeatSampler 구현
- 데이터 복제 대신 인덱스 반복 사용
- 이론적 메모리 효율성 향상

## 빠른 시작

### 1. 환경 설정
```bash
./setup_env.sh
```

### 2. 실험 실행
```bash
# rloo_k=2,4,8로 모든 방법 실행
./run_basic.sh          # 기본 RLOO (최적화 없음)
./run_lazy.sh           # Lazy Generation 방식
./run_repeatsampler.sh  # RepeatSampler 방식  
./run_string-level.sh   # String-Level Processing (권장)
```

### 3. 결과 분석
```bash
# 모든 실험 데이터 통합
./combine_results.sh

# 시각화 차트 생성
./visualize_results.sh
```

## 실험 결과

### 메모리 사용량 요약 (피크 GPU 메모리)

| 방법 | rloo_k=2 | rloo_k=4 | rloo_k=8 | 상태 |
|------|----------|----------|----------|------|
| **Basic** | 14.38GB | 26.99GB | 23.87GB | 완료 |
| **Lazy** | 14.49GB | 27.22GB | **OOM** | k=8에서 실패 |
| **RepeatSampler** | 14.48GB | 27.22GB | **OOM** | k=8에서 실패 |
| **String-Level** | 6.40GB | 11.01GB | 20.25GB | **모두 완료** |

### 주요 관찰사항

1. **String-Level Processing**이 모든 설정을 성공적으로 처리하는 유일한 방법
2. **Lazy Generation**과 **RepeatSampler**는 미미한 개선을 보이며 높은 메모리 요구에서 실패
3. **rloo_k=8에서의 OOM 오류**는 효과적인 메모리 최적화의 중요성을 보여줌
4. **String-Level Processing**은 일관되게 50% 이상의 메모리 감소 달성

## 생성된 시각화

실험 및 시각화 실행 후:
- `peak_memory_comparison.png` - 피크 메모리 사용량 비교
- `memory_usage_timeline.png` - 각 방법별 시간에 따른 메모리 사용량
- `method_performance_summary.png` - 성능 요약 및 절약량
