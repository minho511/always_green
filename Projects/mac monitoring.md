
```zsh
pip install asitop
sudo asitop
```

1. 지표별 의미

- P-CPU (Performance Core) : 고성능 코어. 모델 학습 시 데이터 전처리(Data Loading), 복잡한 연산 제어 등 담당.
- E-CPU (Efficiency Core) : 저전력 코어. 시스템의 배경 작업이나 가벼운 프로세스를 처리하여 전력 소모를 줄임.
- GPU (Graphics Unit) : 핵심 연산 장치. MPS(Metal)을 활용한 행렬 연산과 가중치 업데이트가 여기서 일어남.
- ANE (Apple Neural Engine) : AI 전용 가속기. 주로 추론(Inference)이나 CoreML 모델 실행에 최적화되어 있음.

1. 모델 학습 시 중점적으로 볼것.
- GPU 사용률 (가장 중요)
	- 학습 중 GPU 사용률이 **90-100%** 꾸준히 유지하는지 확인.
	- 만약 사용률이 낮고 출렁이면, GPU 가 계산을 하는 시간보다 데이터를 기다리는 시간(병목) 이 더길다는 뜻.
- P-CPU 사용률 (데이터 병목 확인)
	- GPU 사용률이 낮은데 P-CPU 가 풀가동 중이라면 CPU 에서 데이터 전처리 (Augmentation, Shuffle) 하는 속도가 GPU 의 연산 속도를 못 따라가고 있는것.
		-> `DataLoader`의 `num_workers` 값을 높이거나 전처리 로직을 최적화.
- ANE (Neural Engine) 사용량
	- `PyTorch` `Tensorflow`로 일반적인 학습을 할 때는 거의 0%로 표시되는게 정상.
	- 학습은 주로 GPU(MPS)를 사용. ANE는 학습이 끝난 모델을 `CoreML` 로 변환하여 실행할 때 주로 활성화됨.
- 메모리 (Unified Memory) 점유율
	- 체크포인트: `Swap` 사용량이 늘어나는지 확인.
	- 위험 -> 모델의 배치 사이즈(Batch Size) 가 너무 커서 물리 RAM 을 넘어서면 SSD를 메모리처럼 쓰는 스왑이 발생. 학습 속도가 수십배 느려짐.
- 