> 2026-1-15 정리
## 개요

CLIP(Contrastive Language-Image Pre-training)은 OpenAI가 2021년 발표한 혁신적인 비전-언어 사전학습 모델로, 대규모 이미지-텍스트 쌍 데이터를 활용한 대조 학습(contrastive learning)을 통해 zero-shot 이미지 분류 및 검색에서 획기적인 성능을 달성했습니다. 본 섹션에서는 CLIP의 기본 원리와 2022-2026년 사이 발표된 주요 개선 및 변형 모델들을 소개합니다.

### 1.1 CLIP 기반 Two-Stream 아키텍처
****
#### COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval

```
저자: Lu et al.
학회/저널: CVPR 2022
논문링크: https://arxiv.org/pdf/2204.07441
인용 : 2025-1-15 기준 99
```

**핵심 기여**
COTS는 CLIP의 two-stream 구조를 개선하여 세 가지 수준의 cross-modal 상호작용을 도입.
- (1) Token-level interaction : Variational Autoencoder를 활용한 Masked Vision-Language Modeling (MVLM) 목적 함수로 시각적 토큰 생성
- (2) Task-level interation : text-to-image와 image-to-text 검색 작업 간 KL-alignment 목적 함수
- (3) Instance-level alignment : Momentum contrastive learning.
이를 통해 single-stream 방식 대비 10,800배 빠른 추론속도를 달성하면서도 동등한 성능을 보였으며, MSR-VTT 비디오-텍스트 검색에서 SOTA를 기록.

**아키텍처 상세**
- COTS는 collaborative two-stream 구조로, 이미지 인코더와 텍스트 인코더가 독립적으로 작동하면서도 세가지 수준에서 상호작용함.
- Momentum contrastive learning 으로 instance-level alignment를 수행하고, visual encoder에 VAE를 적용하여 MVLM을 위한 visual token 을 생성.
- Cross-stream network module 없이도 효율적인 token-level 상호작용 구현.

**가치**
- 높은 추론 효율성 (10,800배 속도 향상)
- 텍스트-비디오 검색 성능 우수, 경량 구조 
****

#### Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese

```
저자: Pan et al. (DAMO Academy, Alibaba Group)
학회/저널: arXiv 2022
논문링크: https://arxiv.org/abs/2211.01335
인용: 2025-1-15 기준 216
```

**핵심 기여**
- Chinese CLIP은 CLIP을 중국어 환경에 적응시킨 첫 번째 대규모 비전-언어 사전학습 모델.
- 2억 개의 중국어 이미지-텍스트 쌍으로 학습. 다양한 모델 크기 (ViT-B/16 부터 ViT-H/14까지)를 제공.
- 중국어 이미지-텍스트 검색, zero-shot 이미지 분류, cross-modal retrieval 에서 기존 모델 대비 큰 성능 향상을 보임.

**아키텍처 상세**
- Chinese CLIP은 CLIP의 dual-encoder 구조를 유지하면서 중국어 텍스트 인코더를 통합.
- Vision encoder로는 ViT 계열, text encoder로는 중국어 BERT 기반 모델을 활용.
- Contrastive learning objective를 통해 이미지와 중국어 텍스트의 임베딩 공간 정렬.

**가치**
- 다국어 지원으로 확장시 적용 가능
- 다국어 CLIP 모델 개발의 참고 사례로 유용

****
#### CLIP-UP: A Simple and Efficient Mixture-of-Experts CLIP Training Recipe with Sparse Upcycling

```
저자: Wang et al. (APPLE)
학회/저널: arXiv 2025
논문링크: https://arxiv.org/pdf/2502.00965
인용: 2025-1-15 기준 2
```

**핵심 기여**
- CLIP-UP은 사전학습된 dense CLIP 모델을 sparse Mixtue-of-Experts (MoE) 아키텍처로 변환하는 효율적인 학습 전략을 제안.
- Sparse upcycling 기법을 통해 학습 복잡도와 비용을 크게 줄이면서도 성능을 향상시킴.
- CLIP B/16 모델이 dense 버전 대비 COCO에서 7.2%, Flickr30k에서 6.6% 향상된 text-to-image Recall@1을 달성.
- CLIP L/14 모델을 능가하면서도 추론 시 30%의 FLOPs만 사용.

**아키텍처 상세**
- CLIP-UP은 사전학습된 dense CLIP을 MoE 구조로 변환.
- 다양한 설정과 보조 손실 함수를 실험하여 최적의 sparse upcycling recipe를 도출.
- 모델의 용량을 확장하면서도 추론 비용을 제어

**가치**
- 70% FLOP 감소를 달성한 효율적인 아키텍처, 엣지 디바이스 배포에 적합.
- Sparse MoE 기법은 대규모 모델의 경량화 전략으로 활용 가능.

