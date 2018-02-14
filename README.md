# gym-tictactoe
![tictactoe](./img/Tic_Tac_Toe.gif)
----------------------------
-틱택토 예시-

### 비개발자의 강화학습 도전기!
## <알파고 제로 따라하기: 틱택토> 

AlphaGo Fan, AlphaGo Zero, Alpha Zero 논문(https://github.com/kekmodel/RL_Study/tree/master/AlphaGo) 을

수십번 읽고 (너무 어려워요ㅠ) 셋 중 가장 심플하고 강력한 Alpha Zero의 방법을 중심으로 따라가고 있습니다.

몬테카를로-트리서치(MCTS) 부분은 Fan 버전, 스스로 학습하는 방법에 대해선 Zero 버전을 주로 참고했습니다.

아직 <**미완성**>이고 직장인이라 여가시간에 틈틈히 공부하면서 개발중입니다. (아래 진행 상황 참고)

#### 주의! 혼자 이해한 걸로 진행한 거라 오개념이 있을 수 있습니다.ㅎㅎ (피드백 주세요!) 



 
직접 돌려보시려면 python 3, numpy, gym을 설치하시면 됩니다. (아래 요구사항 참조)

신경망 버전을 돌리시려면 PyTorch를 설치하시길 바랍니다.

대국할 때 그래픽 모드를 선택하면 약간의 허접한 렌더링도 지원합니다. ㅋㅋㅋ

gym폴더와 gym-tictactoe폴더가 같은 곳에 있으면 안전합니다.


    
### 저는..?!

알파고를 사랑하는 호기심 많은 보통사람 

    코딩 수준: 코딩의 코자도 모르는 코린이 (파이썬을 공부한지 2주 정도에서 시작함. 코드 눈갱 주의ㅠ)
    전공: 학부 기억을 상실한 산업공학 02학번 (뭐?)
    직업: 수학 강사
    강화 학습 수준: 파이썬과 케라스로 배우는 강화학습 1독, 모두의 RL 강의 1독
    영어 수준: 구글 번역 성애자 (노벨상 줘야함)

사정이 이렇다보니 --> 삽질의 삽질의 삽질의 연속...(파도 파도 끝이 없는..ㅠㅠ) --> 그렇지만 재밌어서 계속 삽질... (여가시간 타임머신) 

--> 오기로 계속 하다보니 어느 정도 성과가 나오기 시작!!(오? 재능 발견?ㅋ) 

--> 생각한 것이 실제로 돌아가니까 너무 재밌음! 시간만 나면 알파고 덕질 중!! 함께 해요 ㅎㅎ 

#### RL관련 편하게 대화나누실 분은 언제든지 페이스북 메시지 주세요 ~ :P (https://www.facebook.com/kekmodel) 

  

### 프로젝트 진행 요약

      1. OpenAI Gym 기반 틱택토 환경 만들기 (완료)
      2. MCTS 구현 (완료)
      3. 데이터 저장 및 불러오기 기능 구현 (완료)
      4. 에이전트 vs 에이전트 테스트 환경 구현 (완료)
      5. 사람 vs 에이전트 테스트 환경 구현 (완료)
      6. 정책 + 가치 신경망 구현 (완료)
      7. 신경망 학습 (**진행 중**)
      8. 사람 vs 에이전트 실제 플레이 환경 구현
      9. 자바스크립트 공부해서 웹에 올려보기


  

## 요구 사항
    
      python3
      git
      numpy
      pytorch
      gym     : 강화학습 API
      dill    : dict 저장용
      xxhash  : dict key로 사용


### gym 설치

      git clone https://github.com/openai/gym.git
      cd gym
      pip install -e .


### numpy or Anaconda5 설치 (https://www.anaconda.com/download)


### PyTorch 설치 (http://pytorch.org)


## gym-tictactoe 설치 (제가 만든 거)
    
        git clone https://github.com/kekmodel/gym-tictactoe.git


    
## 파일 구성

text editor or IDE 로 build 

단, human_interface.py 는 interpreter기반 환경에서만 됨 (e.g. console, IPython)

\* jupyter notebook으로 사용가능

\* 렌더링 오류 시: pip install pyglet==1.2.4  
     
        tictactoe_env.py                강화학습 환경 제공 (gym 기반)
        mcts_zero.py                    신경망이 학습할 최초 데이터 생성(PUCT-MCTS 알고리즘)
        neural_network_cpu.py           정책 + 가치망 cpu버전 (ResNet 5 block)
        neural_network_gpu.py           정책 + 가치망 gpu버전 (ResNet 5 block)
        human_interface.py              사람과 대결하는 인터페이스 제공
        evaluate.py                     에이전트 vs 에이전트 테스트 환경
        data_viewer.ipynb               저장 데이터 분석용 jupyter notebook
        data/state_memory.npy           모든 step의 state가 저장됨 (mcts_zero.py)
        data/edge_memory.npy            모든 step의 edge(action 정보)가 저장됨 (mcts_zero.py)
        data/state|edge_memory_30k.npy  30000 에피소드에 대한 state, edge 데이터
        data/zero_data_30k.pkl          위 데이터를 (state, pi, z)로 묶은 train용 데이터
        data/model_SGD_res5_ch128.pkl   training 완료한 신경망의 파라미터 
    


### AI와 한판 붙고 싶다면? (현재 순수 강화학습만 진행한 버전: 신경망 미적용)

        cd gym-tictactoe
        python human_interface.py


default: text or graphic 선택, 5판 승부, 선공 사람, 착수: 1 ~ 9번 (콘솔창에 치면 됨;;)

        [1][2][3]
        [4][5][6]
        [7][8][9] 



# 프로젝트 진행 기록 (ing..!)

## 1. OpenAI Gym기반 틱택토 환경만들기 (완료)
- git, text editor, jupyter 등 개발환경 설정
- 파이썬 기초 공부
    - 기본 문법
    - 객체지향 개념
    - Numpy 공부
- gym 구조 파악
- 강화학습 기초 공부
    - 파이썬과 케라스로 배우는 강화학습 (강아지 책!)

<2017년>

11월 21일: 코딩 시작

12월 3일: 틱택토 환경 프로토타입 구현 (렌더링 포함)

<공부하며 휴식기간> 

12월 13일: 버그 수정, gym 형식에 맞게 수정

12월 18일: state 타입 수정 (알파고 제로 방식)

12월 19일: **<U>틱택토 환경 정식버전 완성</U>** (tictactoe_env.py)



## 2. 에이전트 만들기 (진행중)
- 파이썬 중급 공부
- 강화학습 공부
    - Marcov Decision Process
    - 가치함수, 정책함수, Q함수
    - Dynamic Programming
    - DQN
    - Actor-Critic 기초
    - PPO 기초
- 알파고 논문 공부
    - AlphaGo Fan
    - AlphaGo Zero
    - Alpha Zero
- 딥러닝 공부
    - 머신러닝 기초
    - CNN
    - ResNet
    - 선형대수학 기초
- PyTorch 기초 공부
- 평가방법 공부
    - ELO Rating
    - Bayes Elo Rating
    - WHR
- ing...

12월 19일: 코딩 시작

12월 25일: **PyTorch로 간단한 ResNet 구현** (neural_network_cpu.py)

12월 31일: state를 hash로 바꿔 다뤄봄

             - dict의 key로 쓸려고
             - 검색 속도 빨라짐
             - 뭘 기준으로 바뀌는지는 공부 필요

<2018년>

1월 1일: PUCT-MCTS 프로토타입 구현
           
             - dict로 접근
             - {state: edge} -> pi계산

1월 3일: (state, edge) set을 hdf5로 저장 구현 (data/) (현재 npy형식으로 바꿈)

1월 6일: **PUCT-MCTS 정식버전 구현** (mcts_zero.py)

             - 알고리즘 오류 수정
             - 첫번째 state에 Dirichlet 노이즈 설정(e-greedy) 
                - 알파 제로 방식
             - 20,000 episode 데이터 저장
                - 약 15만 step
             - 완료시 Slack에 메시지 보내는 기능 추가
                - 개인용

1월 10일: **RL Agent 프로토타입 구현** (agent_rl.py -> human_interface.py에 통합)

              - 딥러닝 없이 순수 강화학습만 활용한 버전
              - 데이터로 트리 재구현 
              - edge의 방문횟수를 softmax를 사용해 확률로 전환
              - 정책함수 구현
              - self Play 전적 1600전 1600무

1월 11일: Human interface 프로토 타입 구현 

              - 사람과 처음으로 게임을 함: 지인들에게 테스트 
                  - 자신이 경험해본 상황에선 강하지만 경험이 부족한 상황에선 최적 판단 못함
                      - 탐험부족으로 생기는 상황
                          - expand 과정을 추가하여 탐험률을 높임 (현재 삭제) 
                              - 100회이상 방문한 edge는 노이즈 주기
                              - 알파고 Fan 방식 단순화
                          - random seed 고정값 제거
              - 방문횟수가 0인데 확률은 0이 아닌 경우가 있어서 반칙패 생김
                  - 정책함수의 확률 계산 방법을 softmax에서 일반평균법으로 바꿈
                  - softmax temperature 추가, 1 hot 인코딩 추가 (현재 삭제)
                      - 경험이 부족한 state에선 여전히 이상행동
                      - 신경망으로 학습할 필요성
              - 저장 형식을 hdf5 -> npy 교체. 로딩 속도 확실히 빨라짐 (강추)

1월 14일: hyperparameter 최적화 후 다시 sample 생성
              
              - 탐험률이 가장 높은 hyperparameter 로 튜닝
              - hyperparameter 추가
                  - 보상에 감가율(decay) 적용해봄 (현재 삭제)
                      - 더 빨리 승부를 내는 것에 가중치를 주기 위해
              - human_interface.py 업데이트
                  - 모드 선택 추가: text & graphic

1월 15일: 에이전트끼리 플레이하는 평가 프로토 타입 구현 (evaluate.py)

              - 평가를 통해 hyperparameter 최적화 
                  - decay 삭제, c_puct: 1, alpha: 1, expand_count: 100 (현재 수정) 
              - 평가 방법 
                  - 각 hyperparameter 조합으로 3000 에피소드를 돌려서 sample 생성
                  - sample로 만든 에이전트끼리 3000 에피소드씩 대결
                  - 5번 시행해서 승률 높은 쪽 선택
              - 해당 에이전트가 생성한 데이터를 신경망 학습용으로 생성

1월 16일: **Human interface 정식 버전 완성** (human_interface.py)

              - AI의 첫번째 수는 확률이 최댓값인 곳만 선택, 그 이후엔 확률에 따라 선택
                  - softmax -> 일반 확률법으로 확정

1월 26일: state 변경 -> 각 플레이어의 최근 4-history 저장

              - 플레이어 3x3 array 4장, 상대 3x3 array 4장, 선공 구별 1장
              - 9x3x3 numpy array -> flatten() 하여 저장

2월 1일: 신경망 학습 시작

              - batch size: 32, epoch: 100, learnig rate: 0.2, momentum:0.9, c: 0.0001
              - loss = MSE(z, v) + CrossEntropy(pi, p) + c * L2 Regularization
              - 최적화: SGD-Momentum 사용
              - lr decay (0.2 -> 0.02 -> 0.002 -> 0.0002)

ing...



