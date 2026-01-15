https://scispace.com/share/f7ff62d2-2eb2-4cee-ae0b-b59b4434b5ae
### 주요 연구 영역

본 가이드는 다음 7개 핵심 연구 영역을 다룹니다:

1. CLIP 기초 및 최신 발전: CLIP의 기본 원리부터 최신 변형 모델까지
2. VLM 아키텍처: 대규모 비전-언어 모델의 구조적 혁신
3. 멀티모달 검색 시스템: 이미지-텍스트, 비디오-텍스트 검색 기술
4. 비디오-언어 이해: 시간적 모델링 및 효율적 비디오 처리
5. 엣지-클라우드 통합: 경량화 및 분산 배포 전략
6. 감시 및 보안 응용: 실시간 모니터링 및 보안 시스템
7. 시스템 최적화: 대규모 학습 및 추론 최적화

### 가이드 활용 방법

초급 연구원 (0-6개월)

- Part 1의 CLIP 기초 논문부터 시작 (⭐ 표시 논문 우선)
- 각 논문의 "핵심 기여" 섹션을 먼저 읽고 전체 논문 학습
- 학습 로드맵의 "기초 단계" 순서를 따라 진행

중급 연구원 (6-12개월)

- Part 2-4의 VLM 아키텍처 및 비디오 이해 논문 집중 학습
- 🎯 표시 논문을 우선적으로 학습하여 Hanwha Vision 응용 분야 이해
- 오픈소스 구현체를 활용한 실습 병행

고급 연구원 (12개월+)

- Part 5-7의 시스템 최적화 및 배포 전략 논문 학습
- 최신 연구 동향 파악 및 자체 연구 방향 설정
- 실제 프로덕션 시스템 설계 및 구현

기호 설명

- ⭐ Must-Read: 해당 분야의 필수 기초 논문
- 🎯 Hanwha Vision 관련성 높음: 감시, 보안, 산업 모니터링에 직접 적용 가능한 논문

---

## 목차 (Table of Contents)

1. [Executive Summary (개요)](https://scispace.com/chat/f7ff62d2-2eb2-4cee-ae0b-b59b4434b5ae#)
2. [Part 1: CLIP 기초 및 최신 발전](https://scispace.com/chat/f7ff62d2-2eb2-4cee-ae0b-b59b4434b5ae#)
3. [Part 2: Vision-Language Models (VLM) 아키텍처](https://scispace.com/chat/f7ff62d2-2eb2-4cee-ae0b-b59b4434b5ae#)
4. [Part 3: 멀티모달 검색 시스템](https://scispace.com/chat/f7ff62d2-2eb2-4cee-ae0b-b59b4434b5ae#)
5. [Part 4: 비디오-언어 이해](https://scispace.com/chat/f7ff62d2-2eb2-4cee-ae0b-b59b4434b5ae#)
6. [Part 5: 엣지-클라우드 통합 및 배포](https://scispace.com/chat/f7ff62d2-2eb2-4cee-ae0b-b59b4434b5ae#)
7. [Part 6: 감시, 보안 및 산업 모니터링 응용](https://scispace.com/chat/f7ff62d2-2eb2-4cee-ae0b-b59b4434b5ae#)
8. [Part 7: 시스템 최적화 및 확장성](https://scispace.com/chat/f7ff62d2-2eb2-4cee-ae0b-b59b4434b5ae#)
9. [Reading Roadmap (학습 로드맵)](https://scispace.com/chat/f7ff62d2-2eb2-4cee-ae0b-b59b4434b5ae#)
10. [Additional Resources (추가 자료)](https://scispace.com/chat/f7ff62d2-2eb2-4cee-ae0b-b59b4434b5ae#)

---

## Part 1: CLIP 기초 및 최신 발전

### 개요

CLIP(Contrastive Language-Image Pre-training)은 OpenAI가 2021년 발표한 혁신적인 비전-언어 사전학습 모델로, 대규모 이미지-텍스트 쌍 데이터를 활용한 대조 학습(contrastive learning)을 통해 zero-shot 이미지 분류 및 검색에서 획기적인 성능을 달성했습니다. 본 섹션에서는 CLIP의 기본 원리와 2022-2026년 사이 발표된 주요 개선 및 변형 모델들을 소개합니다.

### 1.1 CLIP 기반 Two-Stream 아키텍처

#### ⭐ COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval

저자: Lu et al.  
학회/저널: CVPR 2022  
DOI: [10.1109/cvpr52688.2022.01524](https://doi.org/10.1109/cvpr52688.2022.01524)

핵심 기여  
COTS는 CLIP의 two-stream 구조를 개선하여 세 가지 수준의 cross-modal 상호작용을 도입했습니다: (1) Token-level interaction - Variational Autoencoder를 활용한 Masked Vision-Language Modeling (MVLM) 목적 함수로 시각적 토큰 생성, (2) Task-level interaction - text-to-image와 image-to-text 검색 작업 간 KL-alignment 목적 함수, (3) Instance-level alignment - Momentum contrastive learning. 이를 통해 single-stream 방식 대비 10,800배 빠른 추론 속도를 달성하면서도 동등한 성능을 보였으며, MSR-VTT 비디오-텍스트 검색에서 SOTA를 기록했습니다 [1].

아키텍처 상세  
COTS는 collaborative two-stream 구조로, 이미지 인코더와 텍스트 인코더가 독립적으로 작동하면서도 세 가지 수준에서 상호작용합니다. Momentum contrastive learning으로 instance-level alignment를 수행하고, visual encoder에 VAE를 적용하여 MVLM을 위한 visual token을 생성합니다. Cross-stream network module 없이도 효율적인 token-level 상호작용을 구현했습니다 [1].

Hanwha Vision 적용 가치  
COTS의 높은 추론 효율성(10,800배 속도 향상)은 대규모 비디오 데이터베이스 검색 시스템에 이상적입니다. 텍스트-비디오 검색 성능이 우수하여 보안 및 산업 모니터링 환경에서 자연어 쿼리를 통한 효율적인 콘텐츠 발견 및 모니터링이 가능합니다. 엣지 디바이스 배포에도 적합한 경량 구조입니다 [1].

---

#### Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese

저자: Pan et al.  
학회/저널: arXiv 2022  
DOI: [10.48550/arXiv.2211.01335](https://doi.org/10.48550/arXiv.2211.01335)

핵심 기여  
Chinese CLIP은 CLIP을 중국어 환경에 적응시킨 첫 번째 대규모 비전-언어 사전학습 모델입니다. 2억 개의 중국어 이미지-텍스트 쌍으로 학습되었으며, 다양한 모델 크기(ViT-B/16부터 ViT-H/14까지)를 제공합니다. 중국어 이미지-텍스트 검색, zero-shot 이미지 분류, cross-modal retrieval에서 기존 모델 대비 큰 성능 향상을 보였습니다 [13].

아키텍처 상세  
Chinese CLIP은 CLIP의 dual-encoder 구조를 유지하면서 중국어 텍스트 인코더를 통합했습니다. Vision encoder로는 ViT(Vision Transformer) 계열을 사용하고, text encoder로는 중국어 BERT 기반 모델을 활용합니다. Contrastive learning objective를 통해 이미지와 중국어 텍스트의 임베딩 공간을 정렬합니다 [13].

Hanwha Vision 적용 가치  
다국어 지원이 필요한 글로벌 감시 시스템에 적용 가능합니다. 중국어 시장 진출 시 현지화된 비전-언어 검색 시스템 구축에 활용할 수 있으며, 다국어 CLIP 모델 개발의 참고 사례로 유용합니다 [13].

---

#### ⭐ CLIP-UP: A Simple and Efficient Mixture-of-Experts CLIP Training Recipe with Sparse Upcycling

저자: Wang et al.  
학회/저널: 2025  
핵심 기여  
CLIP-UP은 사전학습된 dense CLIP 모델을 sparse Mixture-of-Experts (MoE) 아키텍처로 변환하는 효율적인 학습 전략을 제안합니다. Sparse upcycling 기법을 통해 학습 복잡도와 비용을 크게 줄이면서도 성능을 향상시켰습니다. CLIP B/16 모델이 dense 버전 대비 COCO에서 7.2%, Flickr30k에서 6.6% 향상된 text-to-image Recall@1을 달성했으며, CLIP L/14 모델을 능가하면서도 추론 시 30%의 FLOPs만 사용합니다 [29].

아키텍처 상세  
CLIP-UP은 사전학습된 dense CLIP을 MoE 구조로 변환합니다. 다양한 설정과 보조 손실 함수(auxiliary losses)를 실험하여 최적의 sparse upcycling recipe를 도출했습니다. 이를 통해 모델 용량을 확장하면서도 추론 비용을 제어할 수 있습니다 [29].

Hanwha Vision 적용 가치  
70% FLOP 감소를 달성한 효율적인 아키텍처로, 엣지 디바이스 배포에 매우 적합합니다. 제한된 컴퓨팅 자원에서도 높은 성능을 유지할 수 있어 실시간 감시 시스템에 이상적입니다. Sparse MoE 기법은 대규모 모델의 경량화 전략으로 활용 가능합니다 [29].

---

### 1.2 도메인 특화 CLIP 변형

#### RemoteCLIP: A Vision Language Foundation Model for Remote Sensing

저자: 2023  
DOI: [10.48550/arxiv.2306.11029](https://doi.org/10.48550/arxiv.2306.11029)

핵심 기여  
RemoteCLIP은 원격 감지(remote sensing) 도메인에 특화된 CLIP 변형 모델입니다. 대규모 원격 감지 이미지-텍스트 데이터로 사전학습되어 위성 이미지 및 항공 사진의 cross-modal retrieval에서 우수한 성능을 보입니다. 일반 CLIP 대비 도메인 특화 데이터로 학습하여 원격 감지 작업에서 큰 성능 향상을 달성했습니다 [27].

아키텍처 상세  
RemoteCLIP은 CLIP의 기본 dual-encoder 구조를 유지하면서 원격 감지 도메인에 최적화된 사전학습 데이터와 학습 전략을 사용합니다. Vision encoder는 고해상도 위성 이미지 처리에 적합하도록 조정되었습니다 [27].

Hanwha Vision 적용 가치  
산업 모니터링 및 광역 감시 시스템에 적용 가능합니다. 드론 기반 감시, 대규모 시설 모니터링, 환경 감시 등 넓은 영역을 커버하는 비전 시스템에서 활용할 수 있습니다. 도메인 특화 CLIP 개발의 참고 사례로 유용합니다 [27].

---

#### LowCLIP: Adapting the CLIP Model Architecture for Low-Resource Languages in Multimodal Image Retrieval Task

저자: Asgarov et al.  
학회/저널: 2024

핵심 기여  
LowCLIP은 저자원 언어(low-resource languages)를 위한 CLIP 적응 모델입니다. Multilingual BERT와 경량 백본을 결합하여 제한된 데이터로도 효과적인 이미지-텍스트 검색을 가능하게 합니다. 저자원 언어 환경에서 CLIP의 성능을 크게 향상시켰습니다 [26].

아키텍처 상세  
LowCLIP은 multilingual BERT를 텍스트 인코더로 사용하고, 경량 vision backbone을 채택하여 효율성을 높였습니다. 저자원 언어 데이터의 특성을 고려한 학습 전략을 적용했습니다 [26].

Hanwha Vision 적용 가치  
다국어 지원이 필요한 글로벌 감시 시스템에 적용 가능합니다. 특히 데이터가 부족한 언어 환경에서도 효과적인 비전-언어 검색 시스템을 구축할 수 있습니다. 경량 백본 사용으로 엣지 디바이스 배포에도 적합합니다 [26].

---

### 1.3 효율성 개선 및 최적화

#### Contrastive vision-language pre-training with limited resources

저자: Liu et al.  
학회/저널: ECCV 2022  
DOI: [10.1007/978-3-031-20059-5_14](https://doi.org/10.1007/978-3-031-20059-5_14)

핵심 기여  
이 논문은 제한된 컴퓨팅 자원으로 CLIP 스타일의 contrastive pre-training을 수행하는 방법을 제시합니다. 효율성 향상 기법들을 도입하여 1억 개의 웹 샘플로 강력한 이미지-텍스트 검색 성능을 달성했습니다. 대규모 컴퓨팅 자원 없이도 효과적인 VLM 학습이 가능함을 입증했습니다 [24].

아키텍처 상세  
CLIP 스타일의 dual-encoder 구조를 유지하면서 효율적인 학습 전략을 적용합니다. Batch size 최적화, gradient accumulation, mixed precision training 등의 기법을 활용하여 제한된 자원에서도 효과적인 학습을 가능하게 합니다 [24].

Hanwha Vision 적용 가치  
제한된 컴퓨팅 자원으로 자체 VLM 모델을 학습할 수 있는 실용적인 방법을 제공합니다. 사내 데이터로 도메인 특화 모델을 효율적으로 개발할 수 있으며, 학습 비용을 크게 절감할 수 있습니다 [24].

---

#### Overcoming the Pitfalls of Vision-Language Model for Image-Text Retrieval

저자: Zhang et al.  
학회/저널: 2024  
DOI: [10.1145/3664647.3680591](https://doi.org/10.1145/3664647.3680591)

핵심 기여  
이 논문은 CLIP 기반 이미지-텍스트 검색의 주요 문제점들을 분석하고 해결 방안을 제시합니다. 다양한 CLIP 변형 모델과 대규모 사전학습의 효과를 평가하고, 효율성 향상 기법들을 제안합니다. 실용적인 검색 시스템 구축을 위한 best practices를 제공합니다 [28].

아키텍처 상세  
여러 CLIP 변형 모델들을 체계적으로 비교 분석하고, 각 모델의 장단점을 평가합니다. 효율성과 성능의 trade-off를 고려한 최적화 전략을 제시합니다 [28].

Hanwha Vision 적용 가치  
실제 프로덕션 환경에서 CLIP 기반 검색 시스템을 구축할 때 발생하는 문제점들과 해결 방안을 제공합니다. 모델 선택, 최적화 전략, 배포 시 고려사항 등 실용적인 가이드라인을 얻을 수 있습니다 [28].

---

## Part 2: Vision-Language Models (VLM) 아키텍처

### 개요

Vision-Language Models(VLM)은 시각과 언어 정보를 통합하여 이해하고 처리하는 대규모 멀티모달 모델입니다. CLIP을 넘어서 더 복잡한 비전-언어 작업(VQA, image captioning, visual reasoning 등)을 수행할 수 있는 통합 모델들이 발전하고 있습니다. 본 섹션에서는 최신 VLM 아키텍처와 이론적 기반을 다룹니다.

### 2.1 통합 VLM 아키텍처

#### ⭐ X2-VLM: All-In-One Pre-trained Model For Vision-Language Tasks

저자: Zeng et al.  
학회/저널: IEEE TPAMI 2023  
DOI: [10.1109/tpami.2023.3339661](https://doi.org/10.1109/tpami.2023.3339661)

핵심 기여  
X2-VLM은 이미지-언어 작업을 위한 통합 사전학습 모델입니다. Multi-grained alignment를 통해 image-text retrieval, VQA, image captioning 등 다양한 downstream 작업에서 우수한 성능을 달성했습니다. Unified architecture로 여러 작업을 효율적으로 처리할 수 있으며, 대규모 사전학습을 통해 강력한 zero-shot 및 few-shot 성능을 보입니다 [14].

아키텍처 상세  
X2-VLM은 vision encoder, text encoder, cross-modal encoder로 구성된 통합 아키텍처입니다. Multi-grained alignment 메커니즘을 통해 region-level, object-level, image-level에서 시각-언어 정렬을 수행합니다. Masked language modeling, masked region modeling, image-text matching 등 다양한 사전학습 목적 함수를 사용합니다 [14].

Hanwha Vision 적용 가치  
단일 모델로 다양한 비전-언어 작업을 처리할 수 있어 시스템 복잡도를 줄일 수 있습니다. 감시 영상에서 자연어 쿼리 기반 검색, 이벤트 설명 생성, 시각적 질의응답 등 다양한 기능을 통합 모델로 구현 가능합니다 [14].

---

#### ⭐ OmniVL: One Foundation Model for Image-Language and Video-Language Tasks

저자: Wang et al.  
학회/저널: NeurIPS 2022  
DOI: [10.48550/arXiv.2209.07526](https://doi.org/10.48550/arXiv.2209.07526)

핵심 기여  
OmniVL은 이미지-언어와 비디오-언어 작업을 모두 처리하는 통합 foundation model입니다. Unified transformer encoder와 UniVLC contrastive loss를 사용하여 CLIP 스타일 모델을 확장했습니다. 대규모 이미지/비디오-텍스트 데이터로 사전학습되어 image/video-text retrieval, video-text matching 등에서 우수한 성능을 보입니다 [25].

아키텍처 상세  
OmniVL은 unified transformer encoder를 사용하여 이미지와 비디오를 동일한 방식으로 처리합니다. UniVLC (Unified Vision-Language Contrastive) loss를 통해 이미지-텍스트와 비디오-텍스트 쌍을 동시에 학습합니다. Temporal modeling을 위한 추가 모듈을 포함하여 비디오의 시간적 정보를 효과적으로 처리합니다 [25].

Hanwha Vision 적용 가치  
이미지와 비디오를 통합된 프레임워크로 처리할 수 있어 감시 시스템의 아키텍처를 단순화할 수 있습니다. 정지 이미지 검색과 비디오 검색을 단일 모델로 처리하여 시스템 효율성을 높일 수 있습니다 [25].

---

#### EVA: Exploring the Limits of Masked Visual Representation Learning at Scale

저자: Fang et al.  
학회/저널: arXiv 2022  
DOI: [10.48550/arXiv.2211.07636](https://doi.org/10.48550/arXiv.2211.07636)

핵심 기여  
EVA는 대규모 masked visual representation learning의 한계를 탐구한 연구입니다. 10억 개 이상의 파라미터를 가진 vision transformer를 masked image modeling으로 학습하여 다양한 비전 작업에서 SOTA 성능을 달성했습니다. CLIP과 결합하여 강력한 vision-language 모델로 확장 가능함을 보였습니다 [19].

아키텍처 상세  
EVA는 대규모 Vision Transformer (ViT) 기반 모델로, masked image modeling을 사전학습 목적 함수로 사용합니다. CLIP의 vision encoder로 초기화하고 추가 학습을 통해 성능을 향상시킵니다. Multi-scale feature extraction과 효율적인 attention mechanism을 적용했습니다 [19].

Hanwha Vision 적용 가치  
강력한 visual representation을 제공하여 다양한 downstream 작업의 성능을 향상시킬 수 있습니다. 감시 영상의 고품질 feature extraction에 활용하여 객체 탐지, 추적, 이벤트 인식 등의 성능을 개선할 수 있습니다 [19].

---

### 2.2 효율적인 VLM 설계

#### VLAB: Enhancing Video Language Pre-training by Feature Adapting and Blending

저자: He et al.  
학회/저널: arXiv 2023  
DOI: [10.48550/arXiv.2305.13167](https://doi.org/10.48550/arXiv.2305.13167)

핵심 기여  
VLAB은 feature adapting과 blending을 통해 비디오-언어 사전학습을 향상시키는 방법을 제안합니다. 사전학습된 이미지-언어 모델을 효율적으로 비디오 도메인으로 전이하며, 최소한의 추가 파라미터로 강력한 비디오-텍스트 이해 성능을 달성합니다 [20].

아키텍처 상세  
VLAB은 사전학습된 CLIP을 기반으로 feature adapter와 blending module을 추가합니다. Adapter는 이미지 feature를 비디오에 적합하게 변환하고, blending module은 temporal information을 효과적으로 통합합니다. 파라미터 효율적인 설계로 빠른 학습과 배포가 가능합니다 [20].

Hanwha Vision 적용 가치  
기존 이미지 모델을 비디오 도메인으로 효율적으로 확장할 수 있어 개발 비용을 절감할 수 있습니다. 최소한의 추가 학습으로 비디오 이해 기능을 추가할 수 있어 빠른 프로토타이핑과 배포가 가능합니다 [20].

---

#### TVLT: Textless Vision-Language Transformer

저자: Tang et al.  
학회/저널: NeurIPS 2022  
DOI: [10.48550/arXiv.2209.14156](https://doi.org/10.48550/arXiv.2209.14156)

핵심 기여  
TVLT는 텍스트 없이 vision과 audio만으로 학습하는 새로운 접근법을 제시합니다. Contrastive vision-audio learning을 통해 효율적인 추론과 대규모 사전학습을 달성했습니다. 텍스트 의존성을 제거하여 더 넓은 범위의 멀티모달 데이터를 활용할 수 있습니다 [23].

아키텍처 상세  
TVLT는 vision과 audio를 입력으로 받는 transformer 기반 모델입니다. Contrastive learning을 통해 vision-audio alignment를 학습하며, masked modeling을 통해 각 modality의 representation을 강화합니다. 텍스트 없이도 강력한 멀티모달 이해 능력을 보입니다 [23].

Hanwha Vision 적용 가치  
감시 시스템에서 시각과 음향 정보를 통합하여 이벤트를 더 정확하게 탐지할 수 있습니다. 텍스트 주석이 없는 대규모 감시 영상 데이터를 활용하여 자체 모델을 학습할 수 있습니다 [23].

---

## Part 3: 멀티모달 검색 시스템

### 개요

멀티모달 검색 시스템은 텍스트 쿼리로 이미지나 비디오를 검색하거나, 이미지로 유사한 이미지를 찾는 등 서로 다른 modality 간의 검색을 가능하게 합니다. 본 섹션에서는 효율적이고 확장 가능한 cross-modal retrieval 아키텍처와 기법들을 다룹니다.

### 3.1 이미지-텍스트 검색

#### ⭐ COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval

_[Part 1에서 상세히 다룸 - 참조: Section 1.1]_

COTS는 이미지-텍스트 검색에서 two-stream 방식의 효율성과 single-stream 방식의 성능을 모두 달성한 대표적인 모델입니다. 10,800배 빠른 추론 속도로 대규모 검색 시스템에 이상적입니다 [1].

---

### 3.2 비디오-텍스트 검색

#### ⭐ 🎯 CenterCLIP: Token Clustering for Efficient Text-Video Retrieval

저자: Zhao et al.  
학회/저널: SIGIR 2022  
DOI: [10.1145/3477495.3531950](https://doi.org/10.1145/3477495.3531950)

핵심 기여  
CenterCLIP은 token clustering을 통해 효율적인 텍스트-비디오 검색을 실현합니다. 중복되는 visual token들을 클러스터링하여 계산량을 크게 줄이면서도 검색 정확도를 유지합니다. CLIP 기반 비디오 검색의 효율성을 획기적으로 개선하여 실시간 대규모 비디오 검색을 가능하게 합니다 [4].

아키텍처 상세  
CenterCLIP은 CLIP의 vision encoder를 확장하여 비디오 프레임을 처리합니다. Token clustering module을 도입하여 유사한 visual token들을 클러스터로 그룹화하고, 각 클러스터의 center token만을 사용하여 계산량을 줄입니다. Cross-modal attention을 통해 텍스트와 비디오 간의 정렬을 수행합니다 [4].

Hanwha Vision 적용 가치  
대규모 감시 영상 데이터베이스에서 실시간 텍스트 쿼리 검색을 가능하게 합니다. Token clustering으로 계산량을 줄여 엣지 디바이스나 제한된 서버 자원에서도 효율적인 비디오 검색이 가능합니다. 보안 이벤트 검색, 용의자 추적 등에 직접 적용 가능합니다 [4].

---

#### ⭐ X-CLIP: End-to-End Multi-Grained Contrastive Learning for Video-Text Retrieval

저자: Ma et al.  
학회/저널: ACM MM 2022  
DOI: [10.1145/3503161.3547910](https://doi.org/10.1145/3503161.3547910)

핵심 기여  
X-CLIP은 multi-grained contrastive learning을 통해 비디오-텍스트 검색 성능을 향상시킵니다. Frame-level, clip-level, video-level에서 contrastive learning을 수행하여 세밀한 시각-언어 정렬을 달성합니다. End-to-end 학습이 가능하며, 다양한 비디오-텍스트 검색 벤치마크에서 SOTA 성능을 기록했습니다 [18].

아키텍처 상세  
X-CLIP은 CLIP을 기반으로 multi-grained contrastive learning을 적용합니다. Video encoder는 여러 temporal scale에서 feature를 추출하고, 각 scale에서 text embedding과의 contrastive loss를 계산합니다. Hierarchical attention mechanism을 통해 중요한 프레임과 클립을 강조합니다 [18].

Hanwha Vision 적용 가치  
다양한 시간 스케일의 이벤트를 효과적으로 검색할 수 있습니다. 짧은 순간적 이벤트부터 긴 시간에 걸친 활동까지 정확하게 검색 가능하여 감시 시스템의 유연성을 높입니다. 세밀한 이벤트 분석과 검색이 필요한 보안 응용에 적합합니다 [18].

---

#### Text-Video Retrieval with Global-Local Semantic Consistent Learning

저자: Zhang et al.  
학회/저널: arXiv 2024  
DOI: [10.48550/arxiv.2405.12710](https://doi.org/10.48550/arxiv.2405.12710)

핵심 기여  
이 논문은 global-local semantic consistency를 통해 텍스트-비디오 검색을 개선합니다. Global video representation과 local frame/region representation을 모두 활용하여 더 정확한 semantic matching을 수행합니다. Consistency learning을 통해 서로 다른 granularity의 representation들이 일관성을 유지하도록 합니다 [5].

아키텍처 상세  
Global encoder와 local encoder를 결합한 dual-path 아키텍처를 사용합니다. Global path는 전체 비디오의 semantic을 포착하고, local path는 개별 프레임과 region의 세부 정보를 추출합니다. Consistency loss를 통해 두 path의 representation이 서로 보완하도록 학습합니다 [5].

Hanwha Vision 적용 가치  
전체적인 장면 이해와 세부적인 객체/이벤트 탐지를 동시에 수행할 수 있습니다. 감시 영상에서 "주차장에서 차량이 충돌하는 장면"과 같이 global context와 local detail이 모두 중요한 쿼리를 효과적으로 처리할 수 있습니다 [5].

---

#### GHAN: Graph-Based Hierarchical Aggregation Network for Text-Video Retrieval

저자: Yu et al.  
학회/저널: EMNLP 2022  
DOI: [10.18653/v1/2022.emnlp-main.374](https://doi.org/10.18653/v1/2022.emnlp-main.374)

핵심 기여  
GHAN은 graph-based hierarchical aggregation을 통해 텍스트-비디오 검색을 수행합니다. 비디오의 프레임들을 graph로 모델링하고, hierarchical aggregation을 통해 temporal relationship을 효과적으로 포착합니다. Graph neural network를 활용하여 프레임 간의 복잡한 관계를 학습합니다 [22].

아키텍처 상세  
GHAN은 비디오 프레임을 graph의 node로 표현하고, 프레임 간의 관계를 edge로 모델링합니다. Graph convolutional network를 통해 node feature를 업데이트하고, hierarchical pooling을 통해 video-level representation을 생성합니다. Text encoder와의 cross-modal matching을 수행합니다 [22].

Hanwha Vision 적용 가치  
복잡한 시간적 관계를 가진 이벤트를 효과적으로 모델링할 수 있습니다. 여러 객체 간의 상호작용이나 연속적인 행동 패턴을 이해해야 하는 감시 시나리오에 적합합니다. 이상 행동 탐지나 복잡한 이벤트 검색에 활용 가능합니다 [22].

---

#### SHE-Net: Syntax-Hierarchy-Enhanced Text-Video Retrieval

저자: Yu et al.  
학회/저널: arXiv 2024  
DOI: [10.48550/arxiv.2404.14066](https://doi.org/10.48550/arxiv.2404.14066)

핵심 기여  
SHE-Net은 텍스트의 syntax hierarchy를 활용하여 텍스트-비디오 검색을 향상시킵니다. 문장의 구문 구조를 분석하여 hierarchical text representation을 생성하고, 이를 비디오의 hierarchical structure와 정렬합니다. 언어의 구조적 정보를 활용하여 더 정확한 semantic matching을 달성합니다 [17].

아키텍처 상세  
SHE-Net은 syntax parser를 사용하여 텍스트의 구문 트리를 생성하고, tree-structured encoder를 통해 hierarchical text representation을 학습합니다. 비디오 측에서도 hierarchical temporal structure를 모델링하여 텍스트와 대응시킵니다. Cross-modal attention을 통해 서로 다른 hierarchy level에서 alignment를 수행합니다 [17].

Hanwha Vision 적용 가치  
복잡한 자연어 쿼리를 더 정확하게 이해하고 처리할 수 있습니다. "빨간 차가 주차장에 진입한 후 사람이 내리는 장면"과 같이 구조적으로 복잡한 쿼리를 효과적으로 처리하여 검색 정확도를 높일 수 있습니다 [17].

---

### 3.3 확장 가능한 검색 시스템

#### 🎯 Frame as Video Clip: Proposal-Free Moment Retrieval by Semantic Aligned Frames

저자: Shi et al.  
학회/저널: IEEE Transactions on Industrial Informatics 2024  
DOI: [10.1109/tii.2024.3431097](https://doi.org/10.1109/tii.2024.3431097)

핵심 기여  
"Frame as Video Clip"은 sparse frame sampling과 사전학습된 vision-language 모델을 통합하여 효율적인 moment retrieval을 실현합니다. Proposal-free 전략으로 vanilla transformer를 사용하며, 입력 비디오 프레임 수를 25배 이상, 최대 100배까지 줄여 계산 부담을 크게 감소시킵니다. 보안 및 감시 시스템에서 긴 비디오의 특정 순간을 빠르게 식별하는 데 특화되어 있습니다 [2].

아키텍처 상세  
Sparse frame sampling을 통해 필수 프레임만 선택하고, 사전학습된 vision-language model (CLIP 등)을 활용하여 각 프레임을 인코딩합니다. Vanilla transformer를 사용한 proposal-free 방식으로 moment를 직접 예측하며, semantic alignment를 통해 텍스트 쿼리와 비디오 프레임 간의 정렬을 최적화합니다 [2].

Hanwha Vision 적용 가치  
감시 시스템에서 긴 영상의 특정 이벤트를 빠르게 찾는 데 매우 유용합니다. 25-100배의 프레임 감소로 실시간 처리가 가능하며, 제한된 컴퓨팅 자원에서도 효율적으로 작동합니다. 보안 이벤트 검색, 사건 조사, 영상 분석 등에 직접 적용 가능합니다 [2].

---

## Part 4: 비디오-언어 이해

### 개요

비디오-언어 이해는 정지 이미지를 넘어 시간적 정보를 포함한 비디오 콘텐츠를 언어와 연결하는 기술입니다. 감시 시스템에서는 연속적인 이벤트 이해, 행동 인식, 시간적 추론이 핵심적입니다. 본 섹션에서는 효율적인 비디오 처리와 temporal modeling 기법들을 다룹니다.

### 4.1 CLIP의 비디오 도메인 적응

#### ⭐ CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Representation Alignment

저자: Xue et al.  
학회/저널: arXiv 2022  
DOI: [10.48550/arxiv.2209.06430](https://doi.org/10.48550/arxiv.2209.06430)

핵심 기여  
CLIP-ViP는 사전학습된 CLIP 이미지-텍스트 모델을 비디오-언어 representation alignment로 효율적으로 적응시키는 방법을 제안합니다. Temporal modeling module을 추가하여 프레임 간의 시간적 관계를 학습하며, 최소한의 추가 파라미터로 강력한 비디오 이해 성능을 달성합니다. 비디오-텍스트 검색, action recognition 등 다양한 작업에서 우수한 성능을 보입니다 [3], [6], [8].

아키텍처 상세  
CLIP-ViP는 frozen CLIP encoder를 기반으로 temporal modeling module을 추가합니다. 각 프레임을 CLIP vision encoder로 인코딩한 후, temporal transformer를 통해 프레임 간의 시간적 관계를 모델링합니다. Parameter-efficient design으로 빠른 학습과 배포가 가능합니다 [3], [6], [8].

Hanwha Vision 적용 가치  
기존 CLIP 모델을 비디오 도메인으로 효율적으로 확장할 수 있어 개발 비용을 절감할 수 있습니다. 감시 영상의 시간적 패턴 이해, 연속적인 행동 인식, 이벤트 시퀀스 분석 등에 활용 가능합니다 [3], [6], [8].

---

#### Revisiting Temporal Modeling for CLIP-based Image-to-Video Knowledge Transferring

저자: 2023  
DOI: [10.48550/arxiv.2301.11116](https://doi.org/10.48550/arxiv.2301.11116)

핵심 기여  
이 논문은 CLIP 기반 이미지-비디오 지식 전이에서 temporal modeling의 역할을 재검토합니다. 다양한 temporal modeling 기법들을 체계적으로 비교 분석하고, 효율성과 성능의 trade-off를 평가합니다. 실용적인 temporal modeling 전략을 제시하여 CLIP의 비디오 적응을 최적화합니다 [21].

아키텍처 상세  
여러 temporal modeling 방법들(temporal attention, temporal convolution, temporal pooling 등)을 비교 분석합니다. CLIP의 spatial feature와 temporal feature를 효과적으로 결합하는 방법을 제시하며, 파라미터 효율성과 계산 효율성을 고려한 설계 가이드라인을 제공합니다 [21].

Hanwha Vision 적용 가치  
CLIP 기반 비디오 시스템을 구축할 때 최적의 temporal modeling 전략을 선택할 수 있는 가이드를 제공합니다. 성능과 효율성의 균형을 고려하여 실제 배포 환경에 적합한 아키텍처를 설계할 수 있습니다 [21].

---

#### ⭐ Temporal Modeling With Frozen Vision–Language Foundation Models for Parameter-Efficient Text–Video Retrieval

저자: Shen et al.  
학회/저널: IEEE Transactions on Neural Networks and Learning Systems 2025  
DOI: [10.1109/tnnls.2025.3605657](https://doi.org/10.1109/tnnls.2025.3605657)

핵심 기여  
이 논문은 frozen vision-language foundation model을 활용한 parameter-efficient 텍스트-비디오 검색 방법을 제안합니다. 사전학습된 모델을 고정하고 경량 temporal adapter만 학습하여 효율적인 비디오 이해를 달성합니다. 최소한의 학습 가능 파라미터로 강력한 성능을 보이며, 빠른 학습과 배포가 가능합니다 [7].

아키텍처 상세  
Frozen CLIP 또는 다른 VLM을 backbone으로 사용하고, lightweight temporal adapter를 추가합니다. Adapter는 프레임 간의 시간적 관계를 모델링하며, 전체 모델의 1% 미만의 파라미터만 학습합니다. Efficient attention mechanism과 temporal pooling을 결합하여 계산 효율성을 높입니다 [7].

Hanwha Vision 적용 가치  
대규모 사전학습 모델을 효율적으로 활용하여 빠른 프로토타이핑과 배포가 가능합니다. 제한된 학습 데이터와 컴퓨팅 자원으로도 강력한 비디오 검색 시스템을 구축할 수 있습니다. 엣지 디바이스 배포에도 적합한 경량 설계입니다 [7].

---

### 4.2 효율적인 비디오 처리

#### ⭐ Prompting Visual-Language Models for Efficient Video Understanding

저자: Ju et al.  
학회/저널: ECCV 2022 (Lecture Notes in Computer Science)  
DOI: [10.1007/978-3-031-19833-5_7](https://doi.org/10.1007/978-3-031-19833-5_7)

핵심 기여  
이 논문은 prompting 기법을 통해 visual-language 모델의 효율적인 비디오 이해를 가능하게 합니다. Learnable prompt를 도입하여 사전학습된 VLM을 비디오 작업에 적응시키며, 전체 모델을 fine-tuning하지 않고도 강력한 성능을 달성합니다. Parameter-efficient하고 빠른 학습이 가능합니다 [12].

아키텍처 상세  
사전학습된 VLM에 learnable visual prompt와 text prompt를 추가합니다. Prompt는 입력 이미지/비디오와 텍스트에 concatenate되어 모델의 행동을 조정합니다. Temporal prompt를 통해 비디오의 시간적 정보를 효과적으로 인코딩하며, 전체 모델은 frozen 상태로 유지합니다 [12].

Hanwha Vision 적용 가치  
최소한의 학습으로 다양한 비디오 이해 작업에 VLM을 적응시킬 수 있습니다. 새로운 감시 시나리오나 이벤트 타입에 빠르게 적응 가능하며, 모델 전체를 재학습할 필요가 없어 운영 효율성이 높습니다 [12].

---

#### Prompt Switch: Efficient CLIP Adaptation for Text-Video Retrieval

저자: Deng et al.  
학회/저널: arXiv 2023  
DOI: [10.48550/arxiv.2308.07648](https://doi.org/10.48550/arxiv.2308.07648)

핵심 기여  
Prompt Switch는 효율적인 CLIP 적응을 위한 새로운 prompting 전략을 제안합니다. 다양한 prompt를 동적으로 선택하는 switching mechanism을 도입하여 서로 다른 비디오 콘텐츠에 적응적으로 대응합니다. 텍스트-비디오 검색에서 높은 효율성과 성능을 동시에 달성합니다 [10].

아키텍처 상세  
Prompt pool과 switching network로 구성됩니다. Prompt pool은 다양한 시나리오에 특화된 learnable prompt들을 포함하고, switching network는 입력 비디오/텍스트에 따라 적절한 prompt를 선택합니다. CLIP은 frozen 상태로 유지되며, prompt와 switching network만 학습합니다 [10].

Hanwha Vision 적용 가치  
다양한 감시 환경(실내/실외, 주간/야간, 다양한 날씨 조건 등)에 적응적으로 대응할 수 있습니다. 각 환경에 특화된 prompt를 학습하여 전반적인 검색 성능을 향상시킬 수 있습니다 [10].

---

#### CLIPping: Distilling CLIP-based Models with a Student Base for Video-Language Retrieval

저자: Pei et al.  
학회/저널: CVPR 2023  
DOI: [10.1109/cvpr52729.2023.01820](https://doi.org/10.1109/cvpr52729.2023.01820)

핵심 기여  
CLIPping은 CLIP 기반 모델을 student base로 distillation하여 비디오-언어 검색을 효율화합니다. Knowledge distillation을 통해 대규모 teacher 모델의 지식을 경량 student 모델로 전이하며, 성능 저하를 최소화하면서 추론 속도를 크게 향상시킵니다 [11].

아키텍처 상세  
Large CLIP 기반 teacher 모델과 compact student 모델로 구성됩니다. Teacher 모델은 비디오-텍스트 검색에 fine-tuning되고, student 모델은 distillation loss를 통해 teacher의 출력을 모방하도록 학습합니다. Feature-level과 prediction-level distillation을 결합하여 효과적인 지식 전이를 달성합니다 [11].

Hanwha Vision 적용 가치  
고성능 모델의 지식을 경량 모델로 전이하여 엣지 디바이스 배포를 가능하게 합니다. 제한된 컴퓨팅 자원에서도 높은 검색 성능을 유지할 수 있어 실시간 감시 시스템에 적합합니다 [11].

---

### 4.3 대규모 비디오-언어 사전학습

#### Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations

저자: Jin et al.  
학회/저널: NeurIPS 2022  
DOI: [10.48550/arXiv.2211.11427](https://doi.org/10.48550/arXiv.2211.11427)

핵심 기여  
이 논문은 Expectation-Maximization (EM) 알고리즘을 contrastive learning에 적용하여 compact video-language representation을 학습합니다. EM 프레임워크를 통해 더 효율적이고 robust한 representation을 생성하며, 비디오-텍스트 검색 및 이해 작업에서 우수한 성능을 보입니다 [15].

아키텍처 상세  
EM 알고리즘의 E-step에서는 현재 representation을 기반으로 positive/negative pair의 확률을 추정하고, M-step에서는 이 확률을 사용하여 contrastive loss를 최적화합니다. 이를 통해 hard negative mining과 유사한 효과를 얻으면서도 더 안정적인 학습이 가능합니다 [15].

Hanwha Vision 적용 가치  
더 compact하고 효율적인 비디오 representation을 학습하여 저장 공간과 계산 비용을 절감할 수 있습니다. 대규모 감시 영상 데이터베이스에서 효율적인 검색과 분석이 가능합니다 [15].

---

#### Align and Prompt: Video-and-Language Pre-training with Entity Prompts

저자: 2022  
학회/저널: CVPR 2022  
DOI: [10.1109/cvpr52688.2022.00490](https://doi.org/10.1109/cvpr52688.2022.00490)

핵심 기여  
이 논문은 entity prompt를 활용한 비디오-언어 사전학습 방법을 제안합니다. 비디오와 텍스트에서 entity를 추출하고, 이를 prompt로 사용하여 더 정확한 cross-modal alignment를 달성합니다. Entity-level alignment를 통해 세밀한 semantic understanding이 가능합니다 [16].

아키텍처 상세  
Entity detection module을 통해 비디오와 텍스트에서 entity를 추출합니다. 추출된 entity는 prompt로 변환되어 모델의 입력에 추가되며, entity-level contrastive learning을 통해 정렬됩니다. Global video-text alignment와 entity-level alignment를 결합하여 multi-level understanding을 달성합니다 [16].

Hanwha Vision 적용 가치  
감시 영상에서 중요한 객체(사람, 차량, 물체 등)를 중심으로 한 검색과 분석이 가능합니다. "빨간 차량", "검은 옷을 입은 사람" 등 entity 기반 쿼리를 효과적으로 처리할 수 있습니다 [16].

---

## Part 5: 엣지-클라우드 통합 및 배포

### 개요

실제 감시 및 보안 시스템은 엣지 디바이스(카메라, IoT 센서)와 클라우드 서버가 협력하는 분산 아키텍처를 필요로 합니다. 본 섹션에서는 모델 경량화, 효율적인 추론, 엣지-클라우드 협업 전략을 다룹니다.

### 5.1 경량 모델 아키텍처

#### ⭐ CLIP-UP: A Simple and Efficient Mixture-of-Experts CLIP Training Recipe with Sparse Upcycling

_[Part 1에서 상세히 다룸 - 참조: Section 1.3]_

CLIP-UP의 sparse MoE 아키텍처는 70% FLOP 감소를 달성하여 엣지 디바이스 배포에 이상적입니다. 제한된 컴퓨팅 자원에서도 높은 성능을 유지할 수 있습니다 [29].

---

#### LowCLIP: Adapting the CLIP Model Architecture for Low-Resource Languages in Multimodal Image Retrieval Task

_[Part 1에서 상세히 다룸 - 참조: Section 1.2]_

LowCLIP의 경량 백본 설계는 엣지 디바이스 배포에 적합하며, 제한된 자원에서도 효과적인 이미지-텍스트 검색을 가능하게 합니다 [26].

---

### 5.2 효율적인 추론 최적화

#### 🎯 CenterCLIP: Token Clustering for Efficient Text-Video Retrieval

_[Part 3에서 상세히 다룸 - 참조: Section 3.2]_

CenterCLIP의 token clustering 기법은 계산량을 크게 줄여 실시간 비디오 검색을 가능하게 합니다. 엣지 디바이스나 제한된 서버 자원에서도 효율적으로 작동합니다 [4].

---

#### 🎯 Frame as Video Clip: Proposal-Free Moment Retrieval by Semantic Aligned Frames

_[Part 3에서 상세히 다룸 - 참조: Section 3.3]_

25-100배의 프레임 감소로 실시간 처리가 가능하며, 제한된 컴퓨팅 자원에서도 효율적으로 작동합니다. 감시 시스템의 실시간 이벤트 탐지에 직접 적용 가능합니다 [2].

---

#### CLIPping: Distilling CLIP-based Models with a Student Base for Video-Language Retrieval

_[Part 4에서 상세히 다룸 - 참조: Section 4.2]_

Knowledge distillation을 통해 경량 모델로 지식을 전이하여 엣지 디바이스 배포를 가능하게 합니다. 제한된 컴퓨팅 자원에서도 높은 검색 성능을 유지합니다 [11].

---

### 5.3 파라미터 효율적 학습

#### Temporal Modeling With Frozen Vision–Language Foundation Models for Parameter-Efficient Text–Video Retrieval

_[Part 4에서 상세히 다룸 - 참조: Section 4.1]_

Frozen foundation model과 경량 temporal adapter를 사용하여 최소한의 파라미터로 강력한 성능을 달성합니다. 빠른 학습과 배포가 가능하여 실용적입니다 [7].

---

#### Prompting Visual-Language Models for Efficient Video Understanding

_[Part 4에서 상세히 다룸 - 참조: Section 4.2]_

Learnable prompt를 통해 전체 모델을 fine-tuning하지 않고도 강력한 성능을 달성합니다. 새로운 작업에 빠르게 적응 가능하여 운영 효율성이 높습니다 [12].

---

#### Prompt Switch: Efficient CLIP Adaptation for Text-Video Retrieval

_[Part 4에서 상세히 다룸 - 참조: Section 4.2]_

Dynamic prompt switching을 통해 다양한 환경에 적응적으로 대응하면서도 효율성을 유지합니다. CLIP을 frozen 상태로 유지하여 메모리 효율성이 높습니다 [10].

---

### 5.4 분산 시스템 및 확장성

#### Contrastive vision-language pre-training with limited resources

_[Part 1에서 상세히 다룸 - 참조: Section 1.3]_

제한된 컴퓨팅 자원으로 효과적인 VLM 학습이 가능함을 입증합니다. Batch size 최적화, gradient accumulation 등의 기법을 활용하여 분산 학습 효율성을 높입니다 [24].

---

## Part 6: 감시, 보안 및 산업 모니터링 응용

### 개요

Hanwha Vision의 핵심 비즈니스 영역인 감시, 보안, 산업 모니터링에 직접 적용 가능한 연구들을 다룹니다. 실시간 이벤트 탐지, 이상 행동 인식, 객체 추적, 장면 이해 등 실용적인 응용 분야를 중심으로 정리했습니다.

### 6.1 감시 시스템 응용

#### 🎯 Frame as Video Clip: Proposal-Free Moment Retrieval by Semantic Aligned Frames

_[Part 3에서 상세히 다룸 - 참조: Section 3.3]_

감시 시스템 특화 가치  
이 논문은 보안 및 감시 시스템을 명시적인 응용 대상으로 설정하고 있습니다. 긴 감시 영상에서 특정 이벤트를 빠르게 식별하는 것이 핵심 목표이며, 25-100배의 프레임 감소로 실시간 처리가 가능합니다. 자연어 쿼리를 통한 이벤트 검색으로 보안 요원의 작업 효율성을 크게 향상시킬 수 있습니다 [2].

실제 배포 시나리오

- 24시간 감시 영상에서 "차량 충돌 사고" 검색
- 대규모 시설의 "무단 침입" 이벤트 탐지
- 주차장에서 "차량 도난" 순간 식별
- 공항/역에서 "의심스러운 행동" 패턴 발견

---

#### 🎯 CenterCLIP: Token Clustering for Efficient Text-Video Retrieval

_[Part 3에서 상세히 다룸 - 참조: Section 3.2]_

감시 시스템 특화 가치  
실시간 대규모 비디오 검색을 가능하게 하는 효율적인 아키텍처로, 다수의 카메라에서 동시에 수집되는 영상을 실시간으로 분석하고 검색할 수 있습니다. Token clustering으로 계산량을 줄여 엣지 디바이스나 제한된 서버 자원에서도 효율적으로 작동합니다 [4].

실제 배포 시나리오

- 다중 카메라 네트워크에서 실시간 이벤트 검색
- 엣지 디바이스에서의 로컬 비디오 분석
- 클라우드 서버에서의 대규모 영상 데이터베이스 검색
- 보안 이벤트 발생 시 유사 패턴 즉시 검색

---

#### CLIP-ReIdent: Contrastive Training for Player Re-Identification

저자: Habel et al.  
학회/저널: 2023  
DOI: [10.1145/3552437.3555698](https://doi.org/10.1145/3552437.3555698)

핵심 기여  
CLIP-ReIdent는 CLIP을 person re-identification에 적응시킨 연구입니다. Contrastive image-to-image training을 통해 동일 인물을 서로 다른 카메라 뷰나 시간대에서 재식별합니다. InfoNCE loss를 사용하며, class-agnostic 방식으로 대규모 사전학습의 이점을 활용합니다. Fine-tuned CLIP ViT-L/14로 98.44% mAP를 달성했습니다 [30].

아키텍처 상세  
CLIP의 vision encoder를 기반으로 image-to-image contrastive learning을 수행합니다. 두 이미지가 같은 사람인지 판단하기 위해 similarity score를 계산하며, Score-CAM을 통해 중요한 이미지 영역(예: 옷 번호)을 시각화합니다. Zero-shot OCR 능력을 활용하여 추가 fine-tuning 없이도 유용한 feature를 식별합니다 [30].

Hanwha Vision 적용 가치  
다중 카메라 감시 시스템에서 person re-identification은 핵심 기능입니다. 용의자 추적, 출입 관리, 행동 패턴 분석 등에 직접 적용 가능합니다. CLIP의 강력한 사전학습 지식을 활용하여 제한된 학습 데이터로도 높은 성능을 달성할 수 있습니다 [30].

실제 배포 시나리오

- 대형 쇼핑몰/공항에서 용의자 추적
- 출입 통제 구역의 인원 모니터링
- 장시간에 걸친 개인 행동 패턴 분석
- 다중 카메라 네트워크에서 동일 인물 자동 연결

---

### 6.2 산업 모니터링 응용

#### 🎯 Frame as Video Clip: Proposal-Free Moment Retrieval by Semantic Aligned Frames

_[감시 시스템 응용에서 다룸 - 참조: Section 6.1]_

산업 모니터링 특화 가치  
IEEE Transactions on Industrial Informatics에 게재된 이 논문은 산업 응용을 명시적으로 다룹니다. 제조 공정 모니터링, 품질 관리, 안전 사고 탐지 등에 활용 가능하며, 긴 공정 영상에서 특정 이벤트를 빠르게 찾을 수 있습니다 [2].

실제 배포 시나리오

- 제조 라인에서 불량품 발생 순간 식별
- 작업자 안전 규정 위반 행동 탐지
- 설비 이상 동작 패턴 검색
- 공정 최적화를 위한 특정 작업 단계 분석

---

#### RemoteCLIP: A Vision Language Foundation Model for Remote Sensing

_[Part 1에서 상세히 다룸 - 참조: Section 1.2]_

산업 모니터링 특화 가치  
광역 산업 시설 모니터링, 환경 감시, 대규모 인프라 관리 등에 적용 가능합니다. 드론이나 위성을 통한 원격 모니터링 시스템에 활용할 수 있습니다 [27].

실제 배포 시나리오

- 대규모 산업 단지의 드론 기반 모니터링
- 송전선, 파이프라인 등 인프라 점검
- 환경 오염 모니터링 및 탐지
- 건설 현장 진행 상황 추적

---

### 6.3 실시간 이벤트 탐지

#### X-CLIP: End-to-End Multi-Grained Contrastive Learning for Video-Text Retrieval

_[Part 3에서 상세히 다룸 - 참조: Section 3.2]_

실시간 탐지 특화 가치  
Multi-grained contrastive learning을 통해 다양한 시간 스케일의 이벤트를 효과적으로 탐지할 수 있습니다. 짧은 순간적 이벤트(예: 낙상)부터 긴 시간에 걸친 활동(예: 배회)까지 정확하게 인식 가능합니다 [18].

실제 배포 시나리오

- 실시간 이상 행동 탐지 (폭력, 낙상, 침입 등)
- 교통 사고 자동 탐지 및 알림
- 군중 이상 행동 모니터링
- 긴급 상황 자동 인식 및 대응

---

#### GHAN: Graph-Based Hierarchical Aggregation Network for Text-Video Retrieval

_[Part 3에서 상세히 다룸 - 참조: Section 3.2]_

실시간 탐지 특화 가치  
Graph-based modeling을 통해 복잡한 시간적 관계를 가진 이벤트를 효과적으로 모델링합니다. 여러 객체 간의 상호작용이나 연속적인 행동 패턴을 이해해야 하는 복잡한 이벤트 탐지에 적합합니다 [22].

실제 배포 시나리오

- 복잡한 상호작용 이벤트 탐지 (예: 싸움, 절도)
- 다중 객체 추적 및 행동 분석
- 연속적인 행동 패턴 인식 (예: 배회 후 침입)
- 이상 행동 시퀀스 탐지

---

### 6.4 보안 특화 기능

#### Text-Video Retrieval with Global-Local Semantic Consistent Learning

_[Part 3에서 상세히 다룸 - 참조: Section 3.2]_

보안 특화 가치  
Global context와 local detail을 모두 고려하여 정확한 이벤트 검색이 가능합니다. "주차장에서 차량이 충돌하는 장면"과 같이 장소(global)와 행동(local)이 모두 중요한 보안 쿼리를 효과적으로 처리합니다 [5].

실제 배포 시나리오

- 특정 장소에서의 특정 행동 검색
- 용의자의 세부 특징(옷 색상, 소지품 등) 기반 검색
- 차량 번호판과 차량 행동 동시 검색
- 복합적인 보안 이벤트 분석

---

#### SHE-Net: Syntax-Hierarchy-Enhanced Text-Video Retrieval

_[Part 3에서 상세히 다룸 - 참조: Section 3.2]_

보안 특화 가치  
복잡한 자연어 쿼리를 정확하게 이해하여 보안 요원의 검색 효율성을 높입니다. 구조적으로 복잡한 쿼리("빨간 차가 주차장에 진입한 후 사람이 내리는 장면")를 효과적으로 처리합니다 [17].

실제 배포 시나리오

- 복잡한 시나리오 기반 영상 검색
- 시간적 순서가 중요한 이벤트 검색
- 다단계 행동 패턴 분석
- 상세한 사건 재구성을 위한 영상 검색

---

## Part 7: 시스템 최적화 및 확장성

### 개요

대규모 감시 시스템은 수백, 수천 대의 카메라에서 생성되는 방대한 영상 데이터를 실시간으로 처리하고 저장해야 합니다. 본 섹션에서는 대규모 학습, 효율적인 추론, 분산 시스템 설계 등 시스템 레벨의 최적화 기법들을 다룹니다.

### 7.1 대규모 사전학습

#### EVA: Exploring the Limits of Masked Visual Representation Learning at Scale

_[Part 2에서 상세히 다룸 - 참조: Section 2.1]_

확장성 관점  
10억 개 이상의 파라미터를 가진 대규모 모델의 효율적인 학습 방법을 제시합니다. Masked image modeling을 통해 대규모 unlabeled 데이터를 활용할 수 있어 감시 영상과 같은 레이블이 없는 데이터로도 강력한 모델을 학습할 수 있습니다 [19].

---

#### OmniVL: One Foundation Model for Image-Language and Video-Language Tasks

_[Part 2에서 상세히 다룸 - 참조: Section 2.1]_

확장성 관점  
대규모 이미지/비디오-텍스트 데이터로 사전학습된 통합 foundation model로, 다양한 downstream 작업에 효율적으로 전이 가능합니다. 단일 모델로 여러 작업을 처리하여 시스템 복잡도를 줄이고 유지보수 비용을 절감할 수 있습니다 [25].

---

#### COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval

_[Part 1에서 상세히 다룸 - 참조: Section 1.1]_

확장성 관점  
대규모 사전학습을 통해 강력한 cross-modal retrieval 성능을 달성하면서도 10,800배 빠른 추론 속도를 제공합니다. 대규모 비디오 데이터베이스에서 실시간 검색이 가능한 확장 가능한 아키텍처입니다 [1].

---

### 7.2 추론 최적화

#### CLIP-UP: A Simple and Efficient Mixture-of-Experts CLIP Training Recipe with Sparse Upcycling

_[Part 1에서 상세히 다룸 - 참조: Section 1.3]_

추론 최적화 관점  
Sparse MoE를 통해 추론 시 30%의 FLOPs만 사용하면서도 더 큰 dense 모델을 능가하는 성능을 달성합니다. 대규모 배포 시 서버 비용을 크게 절감할 수 있습니다 [29].

---

#### CenterCLIP: Token Clustering for Efficient Text-Video Retrieval

_[Part 3에서 상세히 다룸 - 참조: Section 3.2]_

추론 최적화 관점  
Token clustering을 통해 계산량을 크게 줄여 실시간 비디오 검색을 가능하게 합니다. 다수의 카메라 스트림을 동시에 처리해야 하는 대규모 감시 시스템에 이상적입니다 [4].

---

#### Frame as Video Clip: Proposal-Free Moment Retrieval by Semantic Aligned Frames

_[Part 3에서 상세히 다룸 - 참조: Section 3.3]_

추론 최적화 관점  
25-100배의 프레임 감소로 메모리 사용량과 계산량을 크게 줄입니다. 긴 감시 영상을 실시간으로 처리할 수 있어 대규모 시스템의 처리량을 크게 향상시킵니다 [2].

---

### 7.3 분산 학습 및 배포

#### Contrastive vision-language pre-training with limited resources

_[Part 1에서 상세히 다룸 - 참조: Section 1.3]_

분산 시스템 관점  
제한된 컴퓨팅 자원으로 효과적인 분산 학습을 수행하는 방법을 제시합니다. Batch size 최적화, gradient accumulation, mixed precision training 등의 기법을 활용하여 효율적인 분산 학습이 가능합니다 [24].

---

#### X2-VLM: All-In-One Pre-trained Model For Vision-Language Tasks

_[Part 2에서 상세히 다룸 - 참조: Section 2.1]_

분산 시스템 관점  
통합 아키텍처로 여러 작업을 단일 모델로 처리하여 시스템 복잡도를 줄입니다. 모델 배포, 버전 관리, 유지보수가 간소화되어 대규모 시스템 운영 효율성이 향상됩니다 [14].

---

### 7.4 엣지-클라우드 협업

#### CLIPping: Distilling CLIP-based Models with a Student Base for Video-Language Retrieval

_[Part 4에서 상세히 다룸 - 참조: Section 4.2]_

엣지-클라우드 협업 관점  
Knowledge distillation을 통해 경량 student 모델을 엣지에, 강력한 teacher 모델을 클라우드에 배포하는 계층적 아키텍처를 구현할 수 있습니다. 엣지에서 1차 필터링, 클라우드에서 정밀 분석을 수행하여 전체 시스템 효율성을 최적화합니다 [11].

---

#### Temporal Modeling With Frozen Vision–Language Foundation Models for Parameter-Efficient Text–Video Retrieval

_[Part 4에서 상세히 다룸 - 참조: Section 4.1]_

엣지-클라우드 협업 관점  
Frozen foundation model과 경량 adapter를 분리하여 배포할 수 있습니다. Foundation model은 클라우드에서 공유하고, task-specific adapter만 엣지에 배포하여 메모리와 대역폭을 절약할 수 있습니다 [7].

---

#### Prompt Switch: Efficient CLIP Adaptation for Text-Video Retrieval

_[Part 4에서 상세히 다룸 - 참조: Section 4.2]_

엣지-클라우드 협업 관점  
Dynamic prompt switching을 통해 다양한 엣지 환경에 적응적으로 대응할 수 있습니다. 각 엣지 디바이스의 특성(실내/실외, 조명 조건 등)에 맞는 prompt를 선택하여 전체 시스템 성능을 최적화합니다 [10].

---

## Reading Roadmap (학습 로드맵)

### 개요

본 로드맵은 Hanwha Vision 딥러닝 연구팀의 신입 연구원이 체계적으로 CLIP, VLM, 멀티모달 검색 시스템을 학습할 수 있도록 설계되었습니다. 기초부터 고급까지 단계별로 구성되어 있으며, 각 단계별 예상 학습 시간과 실습 방향을 제시합니다.

---

### 단계 1: 기초 (Foundation) - 2-3주

목표: CLIP의 기본 원리와 contrastive learning 이해

필수 논문 (⭐)

1. COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval [1]
    
    - 학습 시간: 3-4일
    - 핵심 개념: Two-stream architecture, contrastive learning, cross-modal interaction
    - 실습: CLIP 기반 이미지-텍스트 검색 구현
2. Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese [13]
    
    - 학습 시간: 2-3일
    - 핵심 개념: CLIP의 다국어 적응, dual-encoder 구조
    - 실습: 사전학습된 CLIP 모델 fine-tuning
3. Contrastive vision-language pre-training with limited resources [24]
    
    - 학습 시간: 2-3일
    - 핵심 개념: 효율적인 학습 전략, 제한된 자원 활용
    - 실습: 소규모 데이터셋으로 CLIP 학습

학습 방법

- 각 논문의 Introduction과 Method 섹션을 정독
- 핵심 수식과 알고리즘을 직접 구현
- OpenAI CLIP, OpenCLIP 등 오픈소스 구현체 분석
- 간단한 이미지-텍스트 검색 데모 구축

체크포인트

- [ ]  Contrastive learning의 원리를 설명할 수 있다
- [ ]  CLIP의 two-stream architecture를 이해하고 구현할 수 있다
- [ ]  InfoNCE loss의 수식을 이해하고 코드로 작성할 수 있다
- [ ]  사전학습된 CLIP으로 zero-shot 이미지 분류를 수행할 수 있다

---

### 단계 2: 중급 (Intermediate) - 3-4주

목표: VLM 아키텍처와 비디오-언어 이해 기술 습득

필수 논문 (⭐)

4. X2-VLM: All-In-One Pre-trained Model For Vision-Language Tasks [14]
    
    - 학습 시간: 4-5일
    - 핵심 개념: Unified VLM architecture, multi-grained alignment
    - 실습: Multi-task VLM 구현
5. CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Representation Alignment [3], [6], [8]
    
    - 학습 시간: 4-5일
    - 핵심 개념: Temporal modeling, video-language alignment
    - 실습: CLIP을 비디오 도메인으로 확장
6. X-CLIP: End-to-End Multi-Grained Contrastive Learning for Video-Text Retrieval [18]
    
    - 학습 시간: 3-4일
    - 핵심 개념: Multi-grained contrastive learning, hierarchical attention
    - 실습: 비디오-텍스트 검색 시스템 구축
7. OmniVL: One Foundation Model for Image-Language and Video-Language Tasks [25]
    
    - 학습 시간: 4-5일
    - 핵심 개념: Unified foundation model, image/video 통합 처리
    - 실습: 통합 멀티모달 모델 구현

추가 권장 논문

8. EVA: Exploring the Limits of Masked Visual Representation Learning at Scale [19]
    
    - 학습 시간: 3-4일
    - 핵심 개념: Masked image modeling, large-scale pre-training
9. Prompting Visual-Language Models for Efficient Video Understanding [12]
    
    - 학습 시간: 2-3일
    - 핵심 개념: Visual prompting, parameter-efficient learning

학습 방법

- 각 논문의 실험 결과와 ablation study 분석
- 다양한 VLM 아키텍처 비교 및 장단점 정리
- 비디오 데이터셋(MSR-VTT, ActivityNet 등)으로 실습
- 간단한 비디오-텍스트 검색 시스템 프로토타입 개발

체크포인트

- [ ]  VLM의 주요 아키텍처 패턴을 이해하고 비교할 수 있다
- [ ]  Temporal modeling의 다양한 방법을 설명할 수 있다
- [ ]  비디오-텍스트 검색 시스템을 구현할 수 있다
- [ ]  Multi-grained alignment의 개념을 이해하고 적용할 수 있다

---

### 단계 3: 고급 (Advanced) - 4-5주

목표: 효율적인 시스템 설계 및 실제 배포 전략 습득

필수 논문 (⭐ & 🎯)

10. CenterCLIP: Token Clustering for Efficient Text-Video Retrieval [4] 🎯
    
    - 학습 시간: 3-4일
    - 핵심 개념: Token clustering, 효율적인 추론
    - 실습: 대규모 비디오 검색 시스템 최적화
11. Frame as Video Clip: Proposal-Free Moment Retrieval by Semantic Aligned Frames [2] 🎯
    
    - 학습 시간: 4-5일
    - 핵심 개념: Sparse sampling, proposal-free retrieval, 감시 시스템 응용
    - 실습: 감시 영상 이벤트 검색 시스템 구축
12. CLIP-UP: A Simple and Efficient Mixture-of-Experts CLIP Training Recipe with Sparse Upcycling [29]
    
    - 학습 시간: 4-5일
    - 핵심 개념: Sparse MoE, model efficiency
    - 실습: 경량 CLIP 모델 개발
13. Temporal Modeling With Frozen Vision–Language Foundation Models for Parameter-Efficient Text–Video Retrieval [7]
    
    - 학습 시간: 3-4일
    - 핵심 개념: Parameter-efficient learning, frozen models
    - 실습: 효율적인 비디오 모델 fine-tuning

추가 권장 논문

14. CLIPping: Distilling CLIP-based Models with a Student Base for Video-Language Retrieval [11]
    
    - 학습 시간: 3-4일
    - 핵심 개념: Knowledge distillation, model compression
15. Prompt Switch: Efficient CLIP Adaptation for Text-Video Retrieval [10]
    
    - 학습 시간: 2-3일
    - 핵심 개념: Dynamic prompting, adaptive systems
16. CLIP-ReIdent: Contrastive Training for Player Re-Identification [30]
    
    - 학습 시간: 2-3일
    - 핵심 개념: Person re-identification, 감시 시스템 응용

학습 방법

- 실제 프로덕션 환경을 고려한 시스템 설계
- 성능-효율성 trade-off 분석 및 최적화
- 엣지-클라우드 협업 아키텍처 설계
- 실제 감시 영상 데이터로 end-to-end 시스템 구축

체크포인트

- [ ]  대규모 비디오 검색 시스템의 아키텍처를 설계할 수 있다
- [ ]  모델 경량화 및 최적화 기법을 적용할 수 있다
- [ ]  엣지-클라우드 협업 전략을 수립할 수 있다
- [ ]  실제 감시 시스템에 적용 가능한 프로토타입을 개발할 수 있다

---

### 단계 4: 전문가 (Expert) - 지속적 학습

목표: 최신 연구 동향 파악 및 독자적 연구 수행

심화 논문

17. Text-Video Retrieval with Global-Local Semantic Consistent Learning [5]
18. GHAN: Graph-Based Hierarchical Aggregation Network for Text-Video Retrieval [22]
19. SHE-Net: Syntax-Hierarchy-Enhanced Text-Video Retrieval [17]
20. VLAB: Enhancing Video Language Pre-training by Feature Adapting and Blending [20]
21. Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations [15]
22. Align and Prompt: Video-and-Language Pre-training with Entity Prompts [16]
23. TVLT: Textless Vision-Language Transformer [23]
24. RemoteCLIP: A Vision Language Foundation Model for Remote Sensing [27]
25. LowCLIP: Adapting the CLIP Model Architecture for Low-Resource Languages [26]
26. Overcoming the Pitfalls of Vision-Language Model for Image-Text Retrieval [28]
27. Revisiting Temporal Modeling for CLIP-based Image-to-Video Knowledge Transferring [21]

학습 방법

- 최신 논문 정기적으로 리뷰 (주 1-2편)
- 연구 세미나 및 논문 발표
- 독자적인 연구 아이디어 개발 및 실험
- 실제 프로덕션 시스템 개선 및 최적화

체크포인트

- [ ]  최신 연구 동향을 파악하고 평가할 수 있다
- [ ]  독자적인 연구 문제를 정의하고 해결할 수 있다
- [ ]  연구 결과를 논문으로 작성하고 발표할 수 있다
- [ ]  실제 프로덕션 시스템에 최신 기술을 적용할 수 있다

---

### 주제별 우선순위 논문

#### CLIP 기초 및 변형

우선순위 1 (필수)

- COTS [1]
- Chinese CLIP [13]
- CLIP-UP [29]

우선순위 2 (권장)

- Contrastive vision-language pre-training with limited resources [24]
- Overcoming the Pitfalls of Vision-Language Model [28]

#### VLM 아키텍처

우선순위 1 (필수)

- X2-VLM [14]
- OmniVL [25]
- EVA [19]

우선순위 2 (권장)

- VLAB [20]
- TVLT [23]

#### 비디오-언어 이해

우선순위 1 (필수)

- CLIP-ViP [3], [6], [8]
- X-CLIP [18]
- Temporal Modeling With Frozen VLMs [7]

우선순위 2 (권장)

- Prompting Visual-Language Models [12]
- Prompt Switch [10]
- CLIPping [11]
- Expectation-Maximization Contrastive Learning [15]
- Align and Prompt [16]
- Revisiting Temporal Modeling [21]

#### 멀티모달 검색

우선순위 1 (필수, Hanwha Vision 관련성 높음)

- CenterCLIP [4] 🎯
- Frame as Video Clip [2] 🎯
- X-CLIP [18]

우선순위 2 (권장)

- Text-Video Retrieval with Global-Local Semantic Consistent Learning [5]
- GHAN [22]
- SHE-Net [17]

#### 감시 및 보안 응용

우선순위 1 (필수, Hanwha Vision 핵심)

- Frame as Video Clip [2] 🎯
- CenterCLIP [4] 🎯
- CLIP-ReIdent [30]

우선순위 2 (권장)

- X-CLIP [18]
- GHAN [22]

#### 효율성 및 배포

우선순위 1 (필수)

- CLIP-UP [29]
- CenterCLIP [4] 🎯
- Frame as Video Clip [2] 🎯
- Temporal Modeling With Frozen VLMs [7]

우선순위 2 (권장)

- CLIPping [11]
- Prompt Switch [10]
- Prompting Visual-Language Models [12]
- Contrastive vision-language pre-training with limited resources [24]

---

### 예상 학습 시간 투자

전체 로드맵 완료: 약 3-4개월 (풀타임 기준)

- 단계 1 (기초): 2-3주
    
    - 논문 읽기: 1-2주
    - 실습 및 구현: 1주
- 단계 2 (중급): 3-4주
    
    - 논문 읽기: 2주
    - 실습 및 구현: 1-2주
- 단계 3 (고급): 4-5주
    
    - 논문 읽기: 2-3주
    - 실습 및 시스템 구축: 2주
- 단계 4 (전문가): 지속적 (주 5-10시간)
    
    - 최신 논문 리뷰: 주 2-3시간
    - 연구 및 실험: 주 3-7시간

파트타임 학습 시 조정

- 주 10-15시간 투자 시: 6-8개월
- 주 5-10시간 투자 시: 10-12개월

---

### 이론과 실무 연결 전략

#### 1. 프로토타입 개발 (단계 1-2)

- 목표: 기본 개념을 코드로 구현
- 프로젝트:
    - CLIP 기반 이미지 검색 시스템
    - 간단한 비디오-텍스트 검색 데모
- 데이터셋: COCO, Flickr30k, MSR-VTT
- 도구: PyTorch, Hugging Face Transformers

#### 2. 감시 시스템 적용 (단계 2-3)

- 목표: Hanwha Vision 응용 분야에 기술 적용
- 프로젝트:
    - 감시 영상 이벤트 검색 시스템
    - Person re-identification 시스템
    - 실시간 이상 행동 탐지
- 데이터셋: 사내 감시 영상 데이터, ActivityNet, Charades
- 도구: OpenCV, TensorRT, ONNX

#### 3. 시스템 최적화 (단계 3-4)

- 목표: 프로덕션 레벨 시스템 구축
- 프로젝트:
    - 대규모 비디오 검색 시스템
    - 엣지-클라우드 협업 아키텍처
    - 실시간 멀티카메라 분석 시스템
- 최적화: Model quantization, pruning, distillation
- 배포: Docker, Kubernetes, edge devices

#### 4. 연구 개발 (단계 4)

- 목표: 독자적인 연구 수행 및 논문 발표
- 방향:
    - Hanwha Vision 특화 VLM 개발
    - 감시 시스템 최적화 기법 연구
    - 새로운 멀티모달 검색 알고리즘 개발
- 발표: CVPR, ICCV, ECCV 등 최상위 학회

---

### 학습 팁

1. 논문 읽기 전략
    
    - 첫 읽기: Abstract, Introduction, Conclusion (30분)
    - 두 번째: Method, Experiments (1-2시간)
    - 세 번째: 상세 분석 및 코드 구현 (2-4시간)
2. 실습 우선
    
    - 이론만 읽지 말고 반드시 코드로 구현
    - 오픈소스 구현체를 먼저 실행해보고 분석
    - 작은 데이터셋으로 빠르게 실험
3. 동료 학습
    
    - 주간 논문 세미나 참여
    - 팀원들과 논문 리뷰 및 토론
    - 구현 코드 공유 및 리뷰
4. 문서화
    
    - 학습 내용을 정리하여 내부 위키에 기록
    - 구현 코드에 상세한 주석 작성
    - 실험 결과를 체계적으로 기록
5. 최신 동향 파악
    
    - arXiv daily digest 구독
    - Twitter/X에서 주요 연구자 팔로우
    - 학회 발표 영상 시청 (CVPR, ICCV 등)

---

## Additional Resources (추가 자료)

### 주요 연구자 및 연구실

#### 국제 연구자

- Kaiming He (Meta AI / MIT) - Vision foundation models, masked autoencoders
- Alec Radford (OpenAI) - CLIP 원저자
- Jianfeng Gao (Microsoft Research) - Vision-language models
- Trevor Darrell (UC Berkeley) - Multimodal learning
- Cordelia Schmid (Google Research) - Video understanding
- Yann LeCun (Meta AI / NYU) - Deep learning foundations

#### 주요 연구실

- OpenAI - CLIP, GPT-Vision 등 foundation models
- Meta AI (FAIR) - EVA, MAE 등 self-supervised learning
- Microsoft Research - X-VLM, Florence 등 large-scale VLMs
- Google Research - ViT, ALIGN 등 vision-language models
- UC Berkeley BAIR - Multimodal learning and robotics
- Stanford Vision Lab - Video understanding and action recognition

#### 한국 연구자 및 기관

- KAIST AI - Multimodal learning, video understanding
- 서울대 AI연구원 - Vision-language models
- NAVER AI Lab - Large-scale pre-training, efficient models
- Kakao Brain - Multimodal AI, Korean language models

---

### 관련 학회 및 워크샵

#### 최상위 학회 (Tier 1)

- CVPR (Conference on Computer Vision and Pattern Recognition) - 매년 6월
    
    - Vision-language learning, video understanding 트랙
    - 논문 수: 약 2,000-3,000편
    - Acceptance rate: 약 25-30%
- ICCV (International Conference on Computer Vision) - 격년 10월
    
    - Multimodal learning, large-scale vision 트랙
    - 논문 수: 약 1,500-2,000편
    - Acceptance rate: 약 25-30%
- ECCV (European Conference on Computer Vision) - 격년 9-10월
    
    - Video analysis, efficient models 트랙
    - 논문 수: 약 1,500-2,000편
    - Acceptance rate: 약 25-30%
- NeurIPS (Neural Information Processing Systems) - 매년 12월
    
    - Multimodal learning, foundation models 트랙
    - 논문 수: 약 2,500-3,000편
    - Acceptance rate: 약 20-25%
- ICML (International Conference on Machine Learning) - 매년 7월
    
    - Representation learning, efficient learning 트랙
    - 논문 수: 약 2,000-2,500편
    - Acceptance rate: 약 20-25%
- ICLR (International Conference on Learning Representations) - 매년 4-5월
    
    - Self-supervised learning, vision-language models 트랙
    - 논문 수: 약 1,500-2,000편
    - Acceptance rate: 약 25-30%

#### 주요 저널

- IEEE TPAMI (Transactions on Pattern Analysis and Machine Intelligence)
    
    - Impact Factor: ~20
    - 최상위 vision 저널
- IJCV (International Journal of Computer Vision)
    
    - Impact Factor: ~15
    - 최상위 vision 저널
- IEEE TIP (Transactions on Image Processing)
    
    - Impact Factor: ~10
    - 이미지 처리 및 분석
- IEEE TNNLS (Transactions on Neural Networks and Learning Systems)
    
    - Impact Factor: ~10
    - 딥러닝 및 신경망
- IEEE TII (Transactions on Industrial Informatics)
    
    - Impact Factor: ~10
    - 산업 응용 (본 가이드의 [2]번 논문 게재)

#### 관련 워크샵

- CVPR Workshop on Vision and Language
- ICCV Workshop on Multi-Modal Video Analysis
- NeurIPS Workshop on Self-Supervised Learning
- ECCV Workshop on Efficient Deep Learning
- CVPR Workshop on Large Scale Computer Vision

---

### 오픈소스 구현체

#### CLIP 관련

1. OpenAI CLIP (공식)
    
    - GitHub: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
    - 언어: Python (PyTorch)
    - 특징: 원본 CLIP 구현, 사전학습 모델 제공
2. OpenCLIP
    
    - GitHub: [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
    - 언어: Python (PyTorch)
    - 특징: 다양한 CLIP 변형, 대규모 학습 지원
3. Chinese-CLIP
    
    - GitHub: [https://github.com/OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)
    - 언어: Python (PyTorch)
    - 특징: 중국어 CLIP, 다국어 확장 참고
4. CLIP-ViP
    
    - GitHub: [https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP)
    - 언어: Python (PyTorch)
    - 특징: 비디오 도메인 적응

#### VLM 관련

5. X-VLM / X2-VLM
    
    - GitHub: [https://github.com/zengyan-97/X-VLM](https://github.com/zengyan-97/X-VLM)
    - 언어: Python (PyTorch)
    - 특징: 통합 VLM 아키텍처
6. OmniVL
    
    - GitHub: [https://github.com/om-ai-lab/OmniVL](https://github.com/om-ai-lab/OmniVL)
    - 언어: Python (PyTorch)
    - 특징: 이미지/비디오 통합 모델
7. EVA
    
    - GitHub: [https://github.com/baaivision/EVA](https://github.com/baaivision/EVA)
    - 언어: Python (PyTorch)
    - 특징: 대규모 masked visual representation

#### 비디오-언어 모델

8. X-CLIP
    
    - GitHub: [https://github.com/xuguohai/X-CLIP](https://github.com/xuguohai/X-CLIP)
    - 언어: Python (PyTorch)
    - 특징: Multi-grained video-text retrieval
9. CLIP4Clip
    
    - GitHub: [https://github.com/ArrowLuo/CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)
    - 언어: Python (PyTorch)
    - 특징: CLIP 기반 비디오 검색
10. Frozen in Time
    
    - GitHub: [https://github.com/m-bain/frozen-in-time](https://github.com/m-bain/frozen-in-time)
    - 언어: Python (PyTorch)
    - 특징: Efficient video-text learning

#### 효율성 및 배포

11. TensorRT
    
    - GitHub: [https://github.com/NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)
    - 언어: C++, Python
    - 특징: GPU 추론 최적화
12. ONNX Runtime
    
    - GitHub: [https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
    - 언어: C++, Python
    - 특징: 크로스 플랫폼 추론
13. OpenVINO
    
    - GitHub: [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)
    - 언어: C++, Python
    - 특징: Intel 하드웨어 최적화

#### 유틸리티

14. Hugging Face Transformers
    
    - GitHub: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
    - 언어: Python (PyTorch, TensorFlow)
    - 특징: 다양한 사전학습 모델, 쉬운 사용
15. MMAction2 (OpenMMLab)
    
    - GitHub: [https://github.com/open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2)
    - 언어: Python (PyTorch)
    - 특징: 비디오 이해 툴킷
16. PyTorch Video (Meta)
    
    - GitHub: [https://github.com/facebookresearch/pytorchvideo](https://github.com/facebookresearch/pytorchvideo)
    - 언어: Python (PyTorch)
    - 특징: 비디오 모델 라이브러리

---

### 실험용 데이터셋

#### 이미지-텍스트 데이터셋

1. COCO Captions
    
    - 크기: 123K images, 5 captions per image
    - 용도: Image-text retrieval, captioning
    - 다운로드: [https://cocodataset.org/](https://cocodataset.org/)
2. Flickr30k
    
    - 크기: 31K images, 5 captions per image
    - 용도: Image-text retrieval
    - 다운로드: [http://shannon.cs.illinois.edu/DenotationGraph/](http://shannon.cs.illinois.edu/DenotationGraph/)
3. Visual Genome
    
    - 크기: 108K images, dense annotations
    - 용도: Visual reasoning, scene understanding
    - 다운로드: [https://visualgenome.org/](https://visualgenome.org/)

#### 비디오-텍스트 데이터셋

4. MSR-VTT
    
    - 크기: 10K videos, 200K captions
    - 용도: Video-text retrieval, captioning
    - 다운로드: [https://www.microsoft.com/en-us/research/publication/msr-vtt/](https://www.microsoft.com/en-us/research/publication/msr-vtt/)
5. ActivityNet Captions
    
    - 크기: 20K videos, 100K descriptions
    - 용도: Dense video captioning, moment retrieval
    - 다운로드: [http://activity-net.org/](http://activity-net.org/)
6. Charades-STA
    
    - 크기: 9.8K videos, 16K temporal annotations
    - 용도: Temporal action localization, moment retrieval
    - 다운로드: [https://prior.allenai.org/projects/charades](https://prior.allenai.org/projects/charades)
7. LSMDC (Large Scale Movie Description Challenge)
    
    - 크기: 118K video clips from movies
    - 용도: Video description, retrieval
    - 다운로드: [https://sites.google.com/site/describingmovies/](https://sites.google.com/site/describingmovies/)

#### 행동 인식 데이터셋

8. Kinetics-400/600/700
    
    - 크기: 400K-650K videos
    - 용도: Action recognition, video understanding
    - 다운로드: [https://deepmind.com/research/open-source/kinetics](https://deepmind.com/research/open-source/kinetics)
9. UCF-101
    
    - 크기: 13K videos, 101 action classes
    - 용도: Action recognition
    - 다운로드: [https://www.crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php)
10. HMDB-51
    
    - 크기: 7K videos, 51 action classes
    - 용도: Action recognition
    - 다운로드: [https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)

#### 감시 및 보안 관련

11. AVA (Atomic Visual Actions)
    
    - 크기: 430 videos, 1.6M action labels
    - 용도: Spatio-temporal action detection
    - 다운로드: [https://research.google.com/ava/](https://research.google.com/ava/)
12. UCF-Crime
    
    - 크기: 1,900 videos, 13 anomaly types
    - 용도: Anomaly detection in surveillance
    - 다운로드: [https://www.crcv.ucf.edu/projects/real-world/](https://www.crcv.ucf.edu/projects/real-world/)
13. VIRAT
    
    - 크기: 8.5 hours of videos
    - 용도: Event recognition in surveillance
    - 다운로드: [https://viratdata.org/](https://viratdata.org/)

#### Person Re-Identification

14. Market-1501
    
    - 크기: 32K images, 1,501 identities
    - 용도: Person re-identification
    - 다운로드: [https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
15. DukeMTMC-reID
    
    - 크기: 36K images, 1,404 identities
    - 용도: Person re-identification
    - 다운로드: [https://github.com/layumi/DukeMTMC-reID_evaluation](https://github.com/layumi/DukeMTMC-reID_evaluation)

---

### 학습 플랫폼 및 도구

#### 온라인 강의

1. Stanford CS231n - Convolutional Neural Networks for Visual Recognition
    
    - URL: [http://cs231n.stanford.edu/](http://cs231n.stanford.edu/)
    - 무료, 비디오 강의 및 과제 제공
2. Deep Learning Specialization (Coursera - Andrew Ng)
    
    - URL: [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning)
    - 유료, 딥러닝 기초부터 고급까지
3. Fast.ai Practical Deep Learning
    
    - URL: [https://course.fast.ai/](https://course.fast.ai/)
    - 무료, 실용적인 딥러닝 접근

#### 논문 검색 및 관리

4. Google Scholar
    
    - URL: [https://scholar.google.com/](https://scholar.google.com/)
    - 논문 검색, 인용 추적
5. Semantic Scholar
    
    - URL: [https://www.semanticscholar.org/](https://www.semanticscholar.org/)
    - AI 기반 논문 추천
6. Papers with Code
    
    - URL: [https://paperswithcode.com/](https://paperswithcode.com/)
    - 논문 + 코드 + 벤치마크
7. arXiv
    
    - URL: [https://arxiv.org/](https://arxiv.org/)
    - 최신 preprint 논문
8. Connected Papers
    
    - URL: [https://www.connectedpapers.com/](https://www.connectedpapers.com/)
    - 논문 관계 시각화

#### 실험 관리

9. Weights & Biases
    
    - URL: [https://wandb.ai/](https://wandb.ai/)
    - 실험 추적, 시각화
10. TensorBoard
    
    - URL: [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)
    - 학습 모니터링
11. MLflow
    
    - URL: [https://mlflow.org/](https://mlflow.org/)
    - ML 실험 관리

#### 협업 도구

12. GitHub
    
    - URL: [https://github.com/](https://github.com/)
    - 코드 버전 관리, 협업
13. Notion
    
    - URL: [https://www.notion.so/](https://www.notion.so/)
    - 문서화, 지식 관리
14. Slack / Microsoft Teams
    
    - 팀 커뮤니케이션

---

### 추천 블로그 및 뉴스레터

#### 기술 블로그

1. OpenAI Blog - [https://openai.com/blog/](https://openai.com/blog/)
2. Meta AI Blog - [https://ai.facebook.com/blog/](https://ai.facebook.com/blog/)
3. Google AI Blog - [https://ai.googleblog.com/](https://ai.googleblog.com/)
4. Microsoft Research Blog - [https://www.microsoft.com/en-us/research/blog/](https://www.microsoft.com/en-us/research/blog/)
5. Distill.pub - [https://distill.pub/](https://distill.pub/) (시각적 설명)

#### 개인 블로그

6. Lil'Log (Lilian Weng) - [https://lilianweng.github.io/](https://lilianweng.github.io/)
7. Jay Alammar - [https://jalammar.github.io/](https://jalammar.github.io/)
8. Sebastian Ruder - [https://ruder.io/](https://ruder.io/)

#### 뉴스레터

9. The Batch (DeepLearning.AI) - [https://www.deeplearning.ai/the-batch/](https://www.deeplearning.ai/the-batch/)
10. Import AI (Jack Clark) - [https://jack-clark.net/](https://jack-clark.net/)
11. Papers with Code Newsletter - [https://paperswithcode.com/newsletter](https://paperswithcode.com/newsletter)

#### 한국어 자료

12. AI Hub - [https://www.aihub.or.kr/](https://www.aihub.or.kr/)
13. TensorFlow Korea - [https://www.facebook.com/groups/TensorFlowKR/](https://www.facebook.com/groups/TensorFlowKR/)
14. PyTorch Korea - [https://www.facebook.com/groups/PyTorchKR/](https://www.facebook.com/groups/PyTorchKR/)

---

### 컨퍼런스 참가 및 네트워킹

#### 참가 전략

1. 논문 읽기: 컨퍼런스 전에 관심 논문 미리 읽기
2. 포스터 세션: 저자와 직접 대화, 질문 준비
3. 워크샵: 특정 주제 심화 학습
4. 네트워킹: 동료 연구자, 기업 관계자와 교류
5. 튜토리얼: 새로운 기술 빠르게 학습

#### 온라인 참가

- 대부분의 주요 컨퍼런스는 온라인 참가 옵션 제공
- 발표 영상은 보통 YouTube에 공개
- Virtual poster session 및 Q&A 참여 가능

---

### Hanwha Vision 특화 학습 경로

#### 1단계: 기초 기술 습득 (1-2개월)

- CLIP 기초 이해 및 구현
- 이미지-텍스트 검색 시스템 프로토타입
- 사내 이미지 데이터로 실험

#### 2단계: 비디오 기술 확장 (1-2개월)

- 비디오-언어 모델 학습
- 감시 영상 데이터로 실험
- 간단한 이벤트 검색 시스템 구축

#### 3단계: 실용화 및 최적화 (2-3개월)

- 대규모 시스템 설계
- 엣지-클라우드 협업 아키텍처
- 실시간 처리 최적화
- Person re-identification 통합

#### 4단계: 프로덕션 배포 (2-3개월)

- 실제 감시 시스템에 통합
- 성능 모니터링 및 개선
- 사용자 피드백 반영
- 지속적인 모델 업데이트

#### 5단계: 연구 개발 (지속적)

- Hanwha Vision 특화 기술 개발
- 논문 발표 및 특허 출원
- 최신 기술 지속적 적용

---

## 결론

본 가이드는 Hanwha Vision 딥러닝 연구팀의 신입 연구원이 CLIP, VLM, 멀티모달 검색 시스템 분야에서 전문성을 갖추는 데 필요한 핵심 논문 30편과 체계적인 학습 로드맵을 제공합니다.

### 핵심 요약

1. CLIP 기초: Contrastive learning과 two-stream architecture를 이해하는 것이 모든 후속 연구의 기반입니다.
    
2. VLM 아키텍처: 통합된 vision-language 모델은 다양한 작업을 효율적으로 처리할 수 있습니다.
    
3. 비디오-언어 이해: Temporal modeling과 parameter-efficient learning이 실용적인 비디오 시스템의 핵심입니다.
    
4. 효율성: Token clustering, sparse MoE, knowledge distillation 등의 기법으로 실시간 처리가 가능합니다.
    
5. 감시 응용: Frame sampling, moment retrieval, person re-identification 등이 Hanwha Vision의 핵심 응용 분야입니다.
    

### 실천 가이드

- ⭐ 표시 논문: 반드시 읽고 구현해야 할 필수 논문
- 🎯 표시 논문: Hanwha Vision 응용에 직접 관련된 고우선순위 논문
- 학습 로드맵: 단계별로 체계적으로 학습하여 3-4개월 내에 전문성 확보
- 실습 중심: 모든 이론은 반드시 코드로 구현하고 실험
- 지속적 학습: 최신 연구 동향을 파악하고 독자적인 연구 수행

### 다음 단계

1. 즉시 시작: Part 1의 COTS 논문부터 읽기 시작
2. 실습 환경 구축: PyTorch, CUDA, 필요한 라이브러리 설치
3. 데이터 준비: COCO, MSR-VTT 등 기본 데이터셋 다운로드
4. 팀 협업: 동료들과 논문 세미나 및 코드 리뷰 시작
5. 프로젝트 계획: 첫 번째 프로토타입 목표 설정

본 가이드를 따라 체계적으로 학습하면, Hanwha Vision의 차세대 비전-언어 검색 시스템 개발에 핵심적인 역할을 수행할 수 있는 전문성을 갖추게 될 것입니다.

연구의 성공을 기원합니다!

---

## References

[1] Lu, H., et al. (2022). COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval. _2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_. [https://doi.org/10.1109/cvpr52688.2022.01524](https://doi.org/10.1109/cvpr52688.2022.01524)

[2] Shi, Y., et al. (2024). Frame as Video Clip: Proposal-Free Moment Retrieval by Semantic Aligned Frames. _IEEE Transactions on Industrial Informatics_. [https://doi.org/10.1109/tii.2024.3431097](https://doi.org/10.1109/tii.2024.3431097)

[3] Xue, H., et al. (2022). CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Representation Alignment. _arXiv preprint_. [https://doi.org/10.48550/arxiv.2209.06430](https://doi.org/10.48550/arxiv.2209.06430)

[4] Zhao, S., et al. (2022). CenterCLIP: Token Clustering for Efficient Text-Video Retrieval. _Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval_. [https://doi.org/10.1145/3477495.3531950](https://doi.org/10.1145/3477495.3531950)

[5] Zhang, Y., et al. (2024). Text-Video Retrieval with Global-Local Semantic Consistent Learning. _arXiv preprint_. [https://doi.org/10.48550/arxiv.2405.12710](https://doi.org/10.48550/arxiv.2405.12710)

[6] Xue, H., et al. (2022). CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Representation Alignment. _arXiv.org_. [https://doi.org/10.48550/arXiv.2209.06430](https://doi.org/10.48550/arXiv.2209.06430)

[7] Shen, X., et al. (2025). Temporal Modeling With Frozen Vision–Language Foundation Models for Parameter-Efficient Text–Video Retrieval. _IEEE Transactions on Neural Networks and Learning Systems_. [https://doi.org/10.1109/tnnls.2025.3605657](https://doi.org/10.1109/tnnls.2025.3605657)

[8] Xue, H., et al. (2022). CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Representation Alignment.

[9] Lu, H., et al. (2022). COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval. _Computer Vision and Pattern Recognition_. [https://doi.org/10.1109/CVPR52688.2022.01524](https://doi.org/10.1109/CVPR52688.2022.01524)

[10] Deng, C., et al. (2023). Prompt Switch: Efficient CLIP Adaptation for Text-Video Retrieval. _arXiv.org_. [https://doi.org/10.48550/arxiv.2308.07648](https://doi.org/10.48550/arxiv.2308.07648)

[11] Pei, W., et al. (2023). CLIPping: Distilling CLIP-based Models with a Student Base for Video-Language Retrieval. _CVPR 2023_. [https://doi.org/10.1109/cvpr52729.2023.01820](https://doi.org/10.1109/cvpr52729.2023.01820)

[12] Ju, C., et al. (2022). Prompting Visual-Language Models for Efficient Video Understanding. _Lecture Notes in Computer Science_. [https://doi.org/10.1007/978-3-031-19833-5_7](https://doi.org/10.1007/978-3-031-19833-5_7)

[13] Pan, Y., et al. (2022). Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese. _arXiv.org_. [https://doi.org/10.48550/arXiv.2211.01335](https://doi.org/10.48550/arXiv.2211.01335)

[14] Zeng, Y., et al. (2023). X2-VLM: All-In-One Pre-trained Model For Vision-Language Tasks. _IEEE Transactions on Pattern Analysis and Machine Intelligence_. [https://doi.org/10.1109/tpami.2023.3339661](https://doi.org/10.1109/tpami.2023.3339661)

[15] Jin, P., et al. (2022). Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations. _Neural Information Processing Systems_. [https://doi.org/10.48550/arXiv.2211.11427](https://doi.org/10.48550/arXiv.2211.11427)

[16] Wang, Y., et al. (2022). Align and Prompt: Video-and-Language Pre-training with Entity Prompts. _2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_. [https://doi.org/10.1109/cvpr52688.2022.00490](https://doi.org/10.1109/cvpr52688.2022.00490)

[17] Yu, J., et al. (2024). SHE-Net: Syntax-Hierarchy-Enhanced Text-Video Retrieval. _arXiv.org_. [https://doi.org/10.48550/arxiv.2404.14066](https://doi.org/10.48550/arxiv.2404.14066)

[18] Ma, Y., et al. (2022). X-CLIP: End-to-End Multi-Grained Contrastive Learning for Video-Text Retrieval. _Proceedings of the 30th ACM International Conference on Multimedia_. [https://doi.org/10.1145/3503161.3547910](https://doi.org/10.1145/3503161.3547910)

[19] Fang, Y., et al. (2022). EVA: Exploring the Limits of Masked Visual Representation Learning at Scale. _arXiv.org_. [https://doi.org/10.48550/arXiv.2211.07636](https://doi.org/10.48550/arXiv.2211.07636)

[20] He, S., et al. (2023). VLAB: Enhancing Video Language Pre-training by Feature Adapting and Blending. _arXiv.org_. [https://doi.org/10.48550/arXiv.2305.13167](https://doi.org/10.48550/arXiv.2305.13167)

[21] (2023). Revisiting Temporal Modeling for CLIP-based Image-to-Video Knowledge Transferring. _arXiv preprint_. [https://doi.org/10.48550/arxiv.2301.11116](https://doi.org/10.48550/arxiv.2301.11116)

[22] Yu, J., et al. (2022). GHAN: Graph-Based Hierarchical Aggregation Network for Text-Video Retrieval. _EMNLP 2022_. [https://doi.org/10.18653/v1/2022.emnlp-main.374](https://doi.org/10.18653/v1/2022.emnlp-main.374)

[23] Tang, Z., et al. (2022). TVLT: Textless Vision-Language Transformer. _Neural Information Processing Systems_. [https://doi.org/10.48550/arXiv.2209.14156](https://doi.org/10.48550/arXiv.2209.14156)

[24] Liu, Y., et al. (2022). Contrastive Vision-Language Pre-training with Limited Resources. _ECCV 2022_. [https://doi.org/10.1007/978-3-031-20059-5_14](https://doi.org/10.1007/978-3-031-20059-5_14)

[25] Wang, J., et al. (2022). OmniVL: One Foundation Model for Image-Language and Video-Language Tasks. _Neural Information Processing Systems_. [https://doi.org/10.48550/arXiv.2209.07526](https://doi.org/10.48550/arXiv.2209.07526)

[26] Asgarov, E., et al. (2024). LowCLIP: Adapting the CLIP Model Architecture for Low-Resource Languages in Multimodal Image Retrieval Task.

[27] (2023). RemoteCLIP: A Vision Language Foundation Model for Remote Sensing. _arXiv preprint_. [https://doi.org/10.48550/arxiv.2306.11029](https://doi.org/10.48550/arxiv.2306.11029)

[28] Zhang, X., et al. (2024). Overcoming the Pitfalls of Vision-Language Model for Image-Text Retrieval. [https://doi.org/10.1145/3664647.3680591](https://doi.org/10.1145/3664647.3680591)

[29] Wang, Z., et al. (2025). CLIP-UP: A Simple and Efficient Mixture-of-Experts CLIP Training Recipe with Sparse Upcycling.

[30] Habel, R., et al. (2023). CLIP-ReIdent: Contrastive Training for Player Re-Identification. [https://doi.org/10.1145/3552437.3555698](https://doi.org/10.1145/3552437.3555698)