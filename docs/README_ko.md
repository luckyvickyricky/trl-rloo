# TRL RLOO 메모리 최적화 프로젝트

## 빠른 시작

이 프로젝트는 TRL RLOO Trainer에서 평균 55%의 메모리 사용량을 줄이는 string-level 처리 최적화를 보여줍니다. 결과를 재현하려면:

```bash
# 1. 환경 설정
./setup_env.sh

# 2. 기준 RLOO 실행 (원본 구현)
./run_baseline-rloo.sh

# 3. 최적화된 RLOO 실행 (string-level 처리)
./run_improve-rloo.sh

# 4. 비교 차트 생성
./visualize_results.sh
```

## 시스템 요구사항

**테스트 환경:**
- 운영체제: Ubuntu 24.04.2 LTS
- GPU: NVIDIA GeForce RTX 5090 (32GB VRAM)
- PyTorch: 2.7.1+cu128
- Transformers: 4.54.1
- CUDA: 12.8

**RTX 5090 외 사용자 참고:** setup_env.sh 스크립트는 CUDA 12.8용 PyTorch nightly를 설치합니다. 다른 GPU를 사용하는 경우 setup_env.sh에서 PyTorch 설치 명령을 해당 CUDA 버전에 맞게 수정해야 합니다.

## 테스트 구성

**모델:** `trl-internal-testing/tiny-Qwen3ForCausalLM`

**테스트 매개변수:**
- rloo_k 값: 2, 4, 8
- 시퀀스 길이: 3.1K 토큰
- 통계적 유효성을 위한 여러 번 실행

## 메모리 최적화 결과

string-level 처리 방식은 상당한 메모리 개선을 보여줍니다:

| 설정 | 기준 메모리 | 최적화 메모리 | 감소율 |
|------|------------|-------------|-------|
| rloo_k=2 | 14.38GB | 6.40GB | 55.5% |
| rloo_k=4 | 26.99GB | 11.01GB | 59.2% |
| rloo_k=8 | OOM 오류 | 20.25GB | N/A |

주요 발견: rloo_k=8에서 기준 구현은 OOM 오류가 발생하지만, 최적화된 버전은 20.25GB 피크 메모리로 성공적으로 완료됩니다.

## 시각화 결과

### 피크 메모리 비교
![피크 메모리 비교](results/peak_memory_comparison.png)

### 메모리 사용 패턴
<table>
<tr>
<td width="50%">

![피크 보존 스무딩](results/rloo_individual_peak_preserving.png)
*피크 보존 스무딩 (정확한 피크 값)*

</td>
<td width="50%">

![초 스무스 시각화](results/rloo_individual_ultra_smooth.png)
*초 스무스 시각화 (스무딩으로 인해 피크 값 부정확)*

</td>
</tr>
</table>

참고: 초 스무스 시각화는 트렌드 분석을 위해 설계되었습니다. 스무딩 알고리즘으로 인해 y축에 표시된 피크 메모리 값은 정확하지 않습니다.

## 구현 세부사항

핵심 최적화는 비효율적인 토큰 레벨 복제를 대체합니다:

```python
# 원본 구현 (비효율적)
queries = queries.repeat(args.rloo_k, 1)  # 메모리 사용량 × rloo_k

# 최적화된 구현 (효율적)
repeated_prompts = prompts_text * rloo_k
queries = processing_class(repeated_prompts, ...)["input_ids"]
```

OnlineDPO의 string-level 처리에서 영감을 받은 이 방식은 동일한 기능을 유지하면서 GPU 메모리에서 물리적 토큰 복제를 피합니다.

## 프로젝트 구조

```
├── trl/                      # 수정된 TRL 서브모듈
├── experiments/              # 훈련 스크립트 및 설정
│   ├── train.py             # 메인 훈련 스크립트
│   ├── config.py            # 실험 설정
│   ├── memory_monitor.py    # 메모리 사용량 추적
│   └── run_*_rloo*.sh       # 개별 테스트 스크립트
├── visualization/           # 분석 및 플롯
│   ├── combine_memory_data.py
│   ├── create_charts.py
│   └── create_basic_string_charts.py
├── results/                 # 실험 데이터 및 차트
└── docs/                    # 문서
```

## 재현 방법

### 환경 설정
설정 스크립트는 필요한 의존성을 가진 conda 환경을 생성합니다:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### 실험 실행
간섭을 피하기 위해 실험을 순차적으로 실행합니다:
```bash
# 기준 (큰 입력에서 rloo_k=8일 때 OOM 발생 가능)
./run_baseline-rloo.sh

# 최적화된 구현
./run_improve-rloo.sh
```

각 스크립트는 메모리 모니터링과 함께 세 가지 설정(rloo_k=2,4,8)을 실행합니다.

### 결과 생성
```bash
./visualize_results.sh
```

이는 메모리 사용 패턴과 피크 소비량을 보여주는 비교 차트를 results/ 디렉토리에 생성합니다.

## 주요 기여사항

1. **메모리 효율성**: GPU 메모리 사용량 평균 55% 감소
2. **안정성**: 높은 rloo_k 값에서 OOM 오류 제거
3. **호환성**: 기존 코드와 완전한 API 호환성 유지
4. **성능**: 훈련 수렴성이나 품질 저하 없음

이 최적화는 RLOO trainer를 더 큰 모델과 높은 rloo_k 값에서 실용적으로 만들어, 프로덕션 환경에서의 적용 가능성을 확장합니다.

## 오픈소스 기여

이 구현은 TRL 라이브러리에 기여하기 위해 설계되었습니다. 수정된 RLOO trainer는 상당한 메모리 개선을 제공하면서 이전 버전과의 호환성을 유지합니다. 모든 변경사항은 핵심 생성 로직에 격리되어 통합이 간단합니다.

실험 프레임워크는 기여 제안을 지원하기 위한 포괄적인 검증 데이터를 제공하여 최적화의 효과성과 신뢰성을 모두 보여줍니다.