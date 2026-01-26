
# Technical Architecture & Logic Flow

딥러닝의 예측 능력(Deep Learning)과 강화학습의 자금 관리(RL Allocation), 그리고 전통적 퀀트의 리스크 제어(Rule-based Control)가 결합된 하이브리드 아키텍처

****

1. System Architecture Diagram
   데이터 수집 부터 주문 집행까지의 End-to-End 파이프라인

![[Untitled diagram-2026-01-26-021303.png]]

```
graph TD

%% 스타일 정의

classDef data fill:#2d3436,stroke:#dfe6e9,stroke-width:2px,color:#fff;

classDef model fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;

classDef rl fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;

classDef exec fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff;

classDef risk fill:#d63031,stroke:#ff7675,stroke-width:2px,color:#fff;

classDef logic fill:#e17055,stroke:#fab1a0,stroke-width:2px,color:#fff;

  

%% ---------------------------------------------------------

%% 1. Data Ingestion Layer

%% ---------------------------------------------------------

subgraph Data_Layer [Market Data Ingestion]

API[Upbit API Rest/WebSocket]:::data

Raw[OHLCV Data<br/>1m, 5m, 15m, 60m]:::data

Tech[Technical Indicators<br/>RSI, MA, Volatility, Bollinger]:::data

Tensor[Tensor Normalization<br/>Window Size: 60]:::data

API --> Raw --> Tech --> Tensor

end

  

%% ---------------------------------------------------------

%% 2. Deep Learning Core (PrecognNet)

%% ---------------------------------------------------------

subgraph DL_Core [PrecognNet: Predictive Engine]

direction TB

Input_Emb[Input Embedding]:::model

Context[Learnable Context Prompt]:::model

subgraph TTT_Process ["Test-Time Training (TTT)"]

MIM_Loss{MIM Reconstruction Loss}:::model

Grad_Update["Gradient Descent<br/>Update Prompt ONLY"]:::model

Adapt["Adapted Prompt<br/>Instance-Specific"]:::model

end

Backbone[Transformer Encoder Backbone]:::model

Head_Policy["Policy Head<br/>Buy/Sell/Hold Prob"]:::model

Head_Value["Value Head<br/>Expected Return"]:::model

Head_Attn["Attention Map<br/>Market Focus"]:::model

  

Tensor --> Input_Emb

Input_Emb --> Backbone

Context --> MIM_Loss

Backbone --> MIM_Loss

MIM_Loss --> Grad_Update --> Adapt

Adapt --> Backbone

Backbone --> Head_Policy & Head_Value & Head_Attn

end

  

%% ---------------------------------------------------------

%% 3. RL Manager Layer

%% ---------------------------------------------------------

subgraph RL_Layer [RL Meta-Controller]

State_Vec["State Vector<br/>Vol, WinRate, PnL, Prob"]:::rl

PPO_Agent["PPO Agent<br/>Multi-Head Actor-Critic"]:::rl

Action_Mode["Strategy Mode<br/>Conservative / Neutral / Aggressive"]:::rl

Action_Conf["Confidence Factor<br/>Bet Sizing Ratio"]:::rl

  

Head_Policy --> State_Vec

Tech --> State_Vec

State_Vec --> PPO_Agent

PPO_Agent --> Action_Mode & Action_Conf

end

  

%% ---------------------------------------------------------

%% 4. Strategic Logic & Filters

%% ---------------------------------------------------------

subgraph Logic_Core [Decision Logic Gate]

Prob_Filter{"Prob > 15%?"}:::logic

RSI_Filter{"RSI < 70?"}:::logic

Spread_Filter{"Spread < 3%?"}:::logic

Cool_Filter{"Cooldown Active?<br/>60m Lock"}:::logic

Filter_Pass[Signal Validated]:::logic

Filter_Fail[Signal Rejected]:::risk

end

  

%% ---------------------------------------------------------

%% 5. Execution Engine

%% ---------------------------------------------------------

subgraph Execution [Smart Execution Engine]

RR["Round-Robin Scheduler<br/>1 Ticker / 1 Sec"]:::exec

Order_Limit["Limit Order<br/>Orderbook Top 1"]:::exec

Monitor["Monitor Filling<br/>5 Sec Timeout"]:::exec

Chase["Market Chase<br/>If Slippage Safe"]:::exec

Cancel[Cancel Order]:::exec

Log[CSV Logging & UI Update]:::exec

end

  

%% Flow Connections

Head_Policy --> Prob_Filter

Action_Mode --> Prob_Filter

Prob_Filter -- Yes --> RSI_Filter

RSI_Filter -- Yes --> Cool_Filter

Cool_Filter -- Yes --> Spread_Filter

Spread_Filter -- Yes --> Filter_Pass

Prob_Filter -- No --> Filter_Fail

RSI_Filter -- No --> Filter_Fail

Cool_Filter -- Yes --> Filter_Fail

Spread_Filter -- No --> Filter_Fail

  

Filter_Pass --> RR

RR --> Order_Limit

Order_Limit --> Monitor

Monitor -- Filled --> Log

Monitor -- Timeout --> Chase

Chase -- Success --> Log

Chase -- Slippage High --> Cancel

  

Log --> |Feedback Loop| State_Vec
```

****

2. 핵심 알고리즘 상세 설명

#### **A. TTT (Test-Time Training) 기반 적응형 모델**

- **개념:** 기존 AI 모델은 학습된 데이터(과거)에 고정되어 있어 급변하는 시장(Non-stationary Market)에 취약합니다.
    
- **PrecognNet의 혁신:** 추론(Inference) 시점마다 입력 데이터에 맞춰 **모델의 일부(Context Prompt)를 실시간으로 미세 조정(Fine-tuning)**합니다.
    
- **효과:** "어제의 지식"으로 오늘의 시장을 보는 것이 아니라, **"방금 본 시장 데이터"에 적응하여 판단**합니다.
    

#### **B. RL(강화학습) 기반 자금 관리 (Meta-Controller)**

- **역할:** 모델이 "사라(Buy)"고 해도, 얼마를 살지(Bet Sizing)와 얼마나 공격적으로 할지(Mode)는 RL 에이전트가 결정합니다.
    
- **입력 상태(State):** 시장 변동성, 최근 승률, 계좌 잔고 추이, 모델의 확신도.
    
- **출력 행동(Action):**
    
    - **Strategy:** 시장이 불안하면 '보수적 모드(Conservative)'로 전환하여 손절 라인을 타이트하게 잡습니다.
        
    - **Confidence:** 확신이 쌀 때는 베팅 금액을 줄이고, 확실할 때는 비중을 늘립니다.
        

#### **C. 3중 방어 필터 (Triple Safety Layer)**

무분별한 매매로 인한 손실을 막기 위해 3단계의 **Hard-coded Rule**이 작동합니다.

1. **RSI Filter:** `RSI(14) ≥ 70` 구간(과매수)에서는 AI가 매수 신호를 보내도 강제로 차단합니다. (고점 판독기)
    
2. **Spread Protection:** 매수/매도 호가 차이가 `3%` 이상 벌어진 종목은 진입하지 않습니다. (슬리피지 방지)
    
3. **Penalty Cooldown:** 매도(익절/손절)가 발생한 종목은 **60분간 블랙리스트**에 올려 재진입을 원천 봉쇄합니다. (뇌동매매 방지)
    

#### **D. 라운드 로빈 스케줄러 (Round-Robin Scheduler)**

- **문제 해결:** 다수의 종목을 동시에 감시할 때 발생하는 API 과부하(Rate Limit) 및 UI 멈춤 현상을 해결했습니다.
    
- **작동 방식:** 1초에 하나의 종목만 정밀 타격(Deep Scan)하고 다음 종목으로 넘어가는 순차 처리 방식을 통해 시스템 안정성을 극대화했습니다.

****
3. 전문가 코멘트 (Conclusion)

"이 시스템은 단순히 '언제 살까'를 고민하는 수준을 넘어섰습니다. **TTT를 통한 시장 적응(Adaptation)**, **RL을 통한 리스크 관리(Risk Management)**, 그리고 **엄격한 퀀트 로직(Execution Logic)**이 유기적으로 결합되어 있습니다. 특히 최근 적용된 **쿨타임 및 RSI 필터**는 시스템의 손익비(Profit Factor)를 극적으로 개선할 핵심적인 'Alpha(알파)' 요소입니다."



# 3. Data Engineering & Labeling Methodology

****
본 시스템은 Raw Market Data를 딥러닝 모델이 해석 가능한 Feature Space로 변환하고, 강화학습 및 지도학습을 위한 Optimal Signal 을 생성하는 고도화된 파이프라인 구축했음.

## 3.1 Data Processing Pipeline(Overview)

데이터는
수집(Ingestion) -> 가공(Feature Eng) -> 윈도우 슬라이싱(Windowing) -> 텐서 변환(Tensorization)의 4단계를 거침.

![[Upbit OHLCV RSI MA Pipeline-2026-01-26-022208.png]]

```
graph TD
    %% 스타일 정의
    classDef raw fill:#2d3436,stroke:#636e72,color:#fff
    classDef feat fill:#0984e3,stroke:#74b9ff,color:#fff
    classDef norm fill:#6c5ce7,stroke:#a29bfe,color:#fff
    classDef tensor fill:#d63031,stroke:#ff7675,color:#fff

    subgraph Step1 ["1. Ingestion"]
        API[Upbit API]:::raw --> OHLCV["Raw OHLCV<br/>(Open, High, Low, Close, Vol)"]:::raw
    end

    subgraph Step2 ["2. Feature Engineering"]
        OHLCV --> Tech["Indicators<br/>RSI, MA, Bollinger, Volatility"]:::feat
        OHLCV --> LogRet["Log Returns<br/>ln(Pt / Pt-1)"]:::feat
    end

    subgraph Step3 ["3. Normalization (Robust Scaling)"]
        Tech --> Zscore["Z-Score Norm<br/>(x - μ) / σ"]:::norm
        LogRet --> Clip["Outlier Clipping"]:::norm
    end

    subgraph Step4 ["4. Tensor Construction"]
        Zscore & Clip --> Concat["Feature Vector"]:::tensor
        Concat --> Sliding["Sliding Window<br/>Seq Len: 60"]:::tensor
        Sliding --> Tensor["Input Tensor<br/>(B, 60, 10)"]:::tensor
    end
```


## 3.2 Model Input Structure (Detailed)

모델은 과거의 데이터(Histroy)를 보고 미래를 예측해야함. 이를 위해 슬라이딩 윈도우(Sliding Window) 기법을 사용하여 시계열 데이터를 [Batch Size, Sequence Length, Feature Dim] 형태의 3차원 텐서로 변환함.

- `Sequence Length` ($T$): 60 (최근 60분/60개의 캔들)
- `Feature Dimension` ($D$): 10 (가격 등락률, 거래량 변화율, RSI, 변동성 지표 등)

### Sliding Window Mechanism

다음은 연속된 시계열 데이터가 어떻게 모델의 입력(Input)과 정답(Target) 쌍으로 잘리는지 보여줌.
![[Upbit OHLCV RSI MA Pipeline-2026-01-26-022741.png]]
```
gantt

title Sliding Window Process (Sequence Length = 60)

dateFormat X

axisFormat %s

section Time Series

Raw Data Stream (t=0 to t=100) : 0, 100

section Sample 1

Input (t=0 ~ t=59) :active, a1, 0, 59

Target (Predict t+k) :crit, t1, 60, 65

section Sample 2

Input (t=1 ~ t=60) :active, a2, 1, 60

Target (Predict t+k+1) :crit, t2, 61, 66

section Sample 3

Input (t=2 ~ t=61) :active, a3, 2, 61

Target (Predict t+k+2) :crit, t3, 62, 67
```

- 입력 (Input $X_t$): 현재 시점 $t$ 를 기준으로 과거 60개의 데이터 포인트.	$$X_t = \{f_{t-59}, f_{t-58}, ..., f_t\}$$
- 특징 벡터 (Features $f_t$): 각 시점마다 아래와 같은 10개의 핵심 지표가 포함됨.
  
	1. `Log Return` (로그 수익률)
        
    2. `Volume Change` (거래량 변화율)
        
    3. `High-Low Spread` (고가-저가 변동폭)
        
    4. `Close-Open Spread` (종가-시가 변동폭)
        
    5. `RSI(14)` (상대강도지수)
        
    6. `MACD`
        
    7. `MACD Signal`
        
    8. `Bollinger Upper` (위치값)
        
    9. `Bollinger Lower` (위치값)
        
    10. `CCI` (Commodity Channel Index)

## 3.3 Ground Truth (GT) Labeling Logic

학습을 위해서는 "이 시점에 샀어야 했나, 팔았어야 했나"에 대한 정답지(GT, Label) 가 필요.
미래 수익률(Future Retrun) 을 기반으로 3-Class Classification 라벨을 생성함.

### GT Generation Algorithm
- Prediction Horizon ($k$): 미래 5분 뒤의 가격($P_{t+5}$)을 예측 목표로 함.
- Threshold ($\alpha$): 횡보장(Noise)을 걸러내기 위한 최소 이익 구간(예: 0.5%)

### **레이블링 로직:**

1. 미래 수익률 계산: $R_{future} = \frac{P_{t+k}-P_t}{P_t}$ 
2. 분류 (Classification):
	- $R_{future} > +\alpha$ (수수료 떼고도 남을 만큼 오름) -> `BUY` (1)
	- $R_{future} < -\alpha$ (손실 위험이 큼) -> `SELL` (2)
	- $-\alpha \le R_{future} \le +\alpha (별 변화 없음)$ -> `HOLD` (0)

![[Upbit OHLCV RSI MA Pipeline-2026-01-26-022640.png]]
```
graph TD

%% 스타일

classDef logic fill:#fff,stroke:#333,stroke-width:2px;

classDef buy fill:#00b894,stroke:#00b894,color:#fff;

classDef sell fill:#d63031,stroke:#d63031,color:#fff;

classDef hold fill:#b2bec3,stroke:#b2bec3,color:#fff;

  

Start((Start)) --> Calc["Calculate Future Return<br/>(Price_t+5 - Price_t) / Price_t"]

Calc --> CheckBuy{"Return > +0.5%?"}

CheckBuy -- Yes --> LabelBuy["Label: BUY (1)"]:::buy

CheckBuy -- No --> CheckSell{"Return < -0.5%?"}

CheckSell -- Yes --> LabelSell["Label: SELL (2)"]:::sell

CheckSell -- No --> LabelHold["Label: HOLD (0)"]:::hold
```

## 3.4 Dataset & Loader Code Snippet

실제 연구에서 사용된 `Pytorch` `Dataset` 클래스 핵심 구조.

```python
class CryptoDataset(Dataset):
    def __init__(self, ohlcv_df, seq_len=60, pred_len=5, threshold=0.005):
        """
        Args:
            ohlcv_df: 정규화된 OHLCV 데이터프레임
            seq_len: 입력 시퀀스 길이 (60분)
            pred_len: 예측할 미래 시점 (5분 뒤)
            threshold: 매수/매도 판단 기준 (0.5%)
        """
        self.data = ohlcv_df.values
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.threshold = threshold

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        # 1. Input Tensor Construction
        # 과거 60개의 데이터 슬라이싱
        x = self.data[idx : idx + self.seq_len]  # Shape: (60, 10)
        
        # 2. GT Generation
        current_price = self.data[idx + self.seq_len - 1, 3] # Close price
        future_price = self.data[idx + self.seq_len + self.pred_len - 1, 3]
        
        # 미래 수익률 계산
        ret = (future_price - current_price) / current_price
        
        # 3. Labeling (0: Hold, 1: Buy, 2: Sell)
        label = 0
        if ret > self.threshold: label = 1
        elif ret < -self.threshold: label = 2
            
        return torch.FloatTensor(x), torch.LongTensor([label]), torch.FloatTensor([ret])
```

## **3.5 Expert Insight (Gemini Summary)**

> "많은 퀀트 시스템이 실패하는 이유는 모델이 나빠서가 아니라 **'데이터(Input)'와 '정답(GT)'의 정의가 느슨하기 때문**입니다.
> 
> 본 연구에서는:
> 
> 1. **Stationarity:** 가격 절대값이 아닌 **'로그 수익률'과 'Z-Score'**를 사용하여 데이터의 시계열적 안정성을 확보했습니다.
>     
> 2. **Look-ahead Bias:** 슬라이딩 윈도우 방식에서 미래 데이터를 참조하는 오류(Look-ahead Bias)를 철저히 배제했습니다.
>     
> 3. **Noise Filtering:** GT 생성 시 단순 등락이 아닌 **`Threshold`($\alpha$)를 적용**하여, 수수료를 극복할 수 있는 유의미한 변동성만을 학습하도록 설계했습니다."


