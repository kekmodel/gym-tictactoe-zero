# gym-tictactoe
![tictactoe](./img/Tic_Tac_Toe.gif)
----------------------------
-틱택토 예시-

### 비개발자의 강화학습 도전기!
## <알파고 제로 따라하기: 틱택토> 

AlphaGo Fan, AlphaGo Zero, Alpha Zero 논문을 수십번 읽고 (너무 어려워요ㅠ) 셋 중 가장 심플하고 강력한 Alpha Zero의 방법을 중심으로 따라가고 있습니다.

몬테카를로-트리서치(MCTS) 부분은 Fan 버전, 스스로 학습하는 방법에 대해선 Zero 버전을 주로 참고했습니다.

직장인이라 여가시간에 틈틈히 공부하면서 개발하였습니다!
 


 
직접 돌려보시려면 python3, numpy, gym, xxhash, PyTorch를 설치하시길 바랍니다. (요구사항 참조)

사람과 대결할 땐 약간의 허접한 렌더링도 지원합니다. ㅋㅋㅋ



    
### 저는..?!

알파고를 사랑하는 호기심 많은 보통사람 

    코딩 수준: 코딩의 코자도 모르는 코린이 (파이썬을 공부한지 2주 정도에서 시작함. 코드 눈갱 주의ㅠ)
    전공: 학부 기억을 상실한 산업공학 02학번 (뭐?)
    직업: 수학 강사
    강화 학습 수준: 파이썬과 케라스로 배우는 강화학습 1독, 모두의 RL 강의 1독
    영어 수준: 구글 번역 성애자 (노벨상 줘야함)

사정이 이렇다보니 --> 삽질의 삽질의 삽질의 연속...(파도 파도 끝이 없는..ㅠㅠ) --> 그렇지만 재밌어서 계속 삽질... (여가시간 타임머신) --> 오기로 계속 하다보니 어느 정도 성과가 나오기 시작!!(오? 재능 발견? ㅎㅎ) --> 생각한 것이 실제로 돌아가니 너무 재밌음! 시간만 나면 알파고 덕질 중!! 모두 함께 해요 ㅎㅎㅎ 

#### 알파고 관련 편하게 대화나누실 분은 언제든지 페이스북 메시지 주세요 ~ :P (https://www.facebook.com/kekmodel) 

  

### 프로젝트 진행 상황 요약

      1. OpenAI Gym 기반 틱택토 환경 만들기 (완료)
      2. MCTS 일반버전 구현 (완료)
      3. 데이터 저장 및 불러오기 기능 구현 (완료)
      4. 에이전트 vs 에이전트 테스트 환경 구현 (완료)
      5. 사람 vs 에이전트 테스트 환경 구현 (완료)
      6. 정책 + 가치 신경망 구현 (완료)
      7. MCTS 제로버전 구현 (완료)
      8. 신경망 학습 (완료)
      9. 사람 vs 에이전트 실제 플레이 환경 구현(완료)
      10. 자바스크립트 공부해서 웹에 올려보기 (잠정 보류)


  

## 요구 사항
    
      python3  : 홈페이지 참조
      git      : 홈페이지 참조 
      numpy    : pip install numpy (아나콘다 추천 mkl까지 돼서 더 빠름)
      gym      : 강화학습 API (아래 참조)
      xxhash   : pip install xxhash (현재 가장 빠른 비암호화 hash)
      pytorch  : 홈페이지 참조
      slackweb : pip install slackweb (결과를 메시지로 보내기: 개인용)   


### gym 설치

    git clone https://github.com/openai/gym.git
    cd gym
    pip install -e .


### numpy or Anaconda5 설치 (https://www.anaconda.com/download)

### PyTorch 설치 (http://pytorch.org)


## gym-tictactoe 설치 (제가 만든 거)
    
    git clone https://github.com/kekmodel/gym-tictactoe-zerp.git


    
## 파일 구성

human_interface.py 는 interpreter기반 환경에서만 됨 (e.g. console, IPython, pycharm)

\* jupyter notebook으로 사용가능

\* 렌더링 오류 시: pip install pyglet==1.2.4  
     
    tictactoe_env.py                강화학습 환경 제공 (gym 기반)
    tictactoe_env_simul.py          시뮬레이션 돌릴 환경    
    selfplay_***.py                 셀프플레이 -> 데이터 저장 (시뮬레이션 내용 확인 가능)
    mcts_simple.py                  위 파일을 간단하게 보완한 최종 버전. (gpu는 모델, 텐서에 .cuda()만 붙여주세요)    
    opimization_***.py              저장된 셀프플레이 데이터로 신경망 학습
    evaluator_***.py                에이전트 vs 에이전트 테스트 환경
    neural_net_***.py               정책 + 가치망 (ResNet 5~40 block)
    human_play_***.py               사람과 대결하는 테스트 환경
    data/data_viewer.ipynb          저장 데이터 분석용 jupyter notebook
    data/                           train dataset, 학습한 모델 저장
    
    


### AI와 한판 붙고 싶다면?

    cd gym-tictactoe
    python human_play_cpu.py(랜덤 파라미터) or gpu(1회 학습 파라미터 800게임)


default: 5판 승부, 선공 사람, 착수: 1 ~ 9번 (콘솔창에 치면 됨;;)

              [1][2][3]
              [4][5][6]
              [7][8][9] 



## 프로젝트 진행 기록 (완료)

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



## 2. 에이전트 만들기 완료
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

12월 25일: **PyTorch로 간단한 ResNet 구현** (neural_net_cpu.py)

12월 31일: state를 hash로 바꿔 다뤄봄

     - dict의 key로 쓸려고
     - 검색 속도 빨라짐
     - 뭘 기준으로 바뀌는지는 공부 필요

<2018년>

1월 1일: PUCT-MCTS 프로토타입 구현
           
     - dict로 접근
     - {state: edge} -> pi계산

1월 3일: (state, edge) set을 hdf5로 저장 구현 (data/) (현재 pickle으로 바꿈)

1월 6일: **PUCT-MCTS 정식버전 구현** (mcts.py) (현재 mcts_simple.py로 교체)

     - 알고리즘 오류 수정
     - 첫번째 state에 Dirichlet 노이즈 설정(e-greedy) 
        - 알파 제로 방식
     - 20,000 episode 데이터 저장
        - 약 15만 step
     - 완료시 Slack에 메시지 보내는 기능 추가
        - 개인용

1월 10일: **RL Agent 프로토타입 구현** (agent_rl.py -> human_play.py에 통합)

    - 딥러닝 없이 순수 강화학습만 활용한 버전
    - 데이터로 트리 재구현 
    - edge의 방문횟수를 softmax를 사용해 확률로 전환
    - 정책함수 구현
    - self Play 전적 1600전 1600무

1월 11일: Human interface 프로토 타입 구현 (현재 human_play.py)

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
      - 저장 형식을 hdf5 -> pkl 교체. 로딩 속도 확실히 빨라짐 (강추)

1월 14일: hyperparameter 최적화 후 다시 sample 생성
              
    - 탐험률이 가장 높은 hyperparameter 로 튜닝
    - hyperparameter 추가
        - 보상에 감가율(decay) 적용해봄 (현재 삭제)
            - 더 빨리 승부를 내는 것에 가중치를 주기 위해
    - human_interface.py 업데이트
        - 모드 선택 추가: text & graphic

1월 15일: 에이전트끼리 플레이하는 평가 프로토 타입 구현 (evaluator.py)

    - 평가를 통해 hyperparameter 최적화 
        - decay 삭제, c_puct: 1, alpha: 1, expand_count: 100 (현재 수정) 
    - 평가 방법 
        - 각 hyperparameter 조합으로 3000 에피소드를 돌려서 sample 생성
        - sample로 만든 에이전트끼리 3000 에피소드씩 대결
        - 5번 시행해서 승률 높은 쪽 선택
    - 해당 에이전트가 생성한 데이터를 신경망 학습용으로 생성

1월 16일: **Human interface 정식 버전 완성** (human_play.py)

    - AI의 첫번째 수는 확률이 최댓값인 곳만 선택, 그 이후엔 확률에 따라 선택
        - softmax -> 일반 확률법으로 확정

1월 26일: state 변경 -> 각 플레이어의 최근 4-history 저장

    - 플레이어 3x3 array 4장, 상대 3x3 array 4장, 선공 구별 1장
    - 9x3x3 numpy array -> flatten() 하여 저장
    - local minimum에 빠지던 현상 현저히 줄어서 더 강해짐

2월 1일: 신경망 학습 시작

    - batch size: 32, epoch: 100, learnig rate: 0.2, momentum:0.9, c: 0.0001
    - loss = MSE(z, v) + CrossEntropy(pi, p) + c * L2 Regularization
    - 최적화: SGD-Momentum 사용
    - lr decay (0.2 -> 0.02 -> 0.002 -> 0.0002)

2월 14일: **MCTS 알고리즘 검색 속도 혁신적으로 개선**

    - 코딩 실력이 늘면서 기존 코드의 개선점이 보였음 ㅋㅋ
    - xxhash 적용 및 코드 최적화를 통해 검색 속도 6배 개선
        - 100만 에피소드가 80분정도 걸림! (i5 린필드;;)
    - pickle로 dict가 저장안됐던 문제 해결

2월 15일: **사람 실력을 뛰어 넘음**

    - 코드 개선된 참에 100만 episode 돌려봄
        - 하이퍼파라미터: [c_puct: 5, epsilon: 0.25, alpha: 0.7]
        - 약 864만 step의 policy iteration
        - 약 13만 경우의 수 학습
    - 첫번째 수만 stochastic (확률로 착수) 그 후엔 deterministic (최댓값만 착수)
        - 사람과 대결해서 현재 무패


2월 24일: **알파고 제로 셀프 플레이 프로토타입 구현**

    - 논문 리뷰후 제대로 된 알고리즘 탑재 (thanks..알메)

3월 1일:  **프로젝트 완료**

    <종료! 알파고 제로 따라하기: 틱택토>
    드디어 3개월간 진행 한 나의 첫프로젝트 <알파고 제로 따라하기: 틱택토>가 끝이 났다.😂 이 다음에 뭘 할지는.... 비밀? ㅋㅋㅋ
    
    <구현해본 것들>
    0. 코딩을 막 시작한 코린이가 코를 딱으면서 gym의 cartpole 코드를 보며 틱택토 환경을 겨우 겨우 구현함. 허접한 렌더링은 덤 ㅋㅋㅋ. (지금 보면 너무 쉬운 코드다 ㅋㅋㅋ)
    1. (젤 뿌듯) 알파고 제로의 MCTS 방법으로 셀프플레이하는 것 구현!
    셀프플레이 결과로 train dataset: (state, pi, z) 생성하여 파일로 저장하는 것 까지!
    2. dataset 파일을 불러와서 policy / value 통합 네트웍을 학습시키기 (Resnet 128채널 5block 구조)
    (더 강해지려면 학습된 네트웍으로 1번 부터 다시 반복해야 함!)
    3. 강해진 것을 확인하기 위해 AI vs AI 환경 만듦. 하이퍼파라미터 별 성능차이를 비교해보는데 유용했음. 논문 수준으로 리뷰하려면 ELO Rating이 필요할 것 같음.
    4. 사람과 실력차를 보려고 AI vs Human 환경 만들어 봄. 고작 1번만 학습했지만 인간 고수 정도의 기량은 나옴! 흡조오옥!
    5. 멀티 프로세싱, 병렬로 트리 탐색하기 등은 앞으로의 숙제로 남겨 놨음. 그리고 강화학습을 제대로 이해하려면 베이즈 확률론을 심도 있게 공부해야 한다는 느낌이 강하게 듦.
    6. 틱택토를 제외하고 배낀 코드 하나 없이 순수 발코딩 and 삽질로 구현함. ㅋㅋㅋ 
    삽질 = 실력 = 셀프플레이 = 강화학습 의 철학!

    <느낀 점>
    1. 파이썬은 진짜 짱이다. 코딩을 갓 시작한 나도 구글링, github을 통해 위와 같은 것들을 해낼 수 있었다. 머릿속의 알고리즘만 명확하다면 구현하는 건 예전 보다 훨씬 쉬워졌다. C로 시작하는 거로 하려고 했으면 이미 접었을 듯......
    2. 알파고 제로가 바둑의 방대한 경우의 수 문제를 어떻게 인간 지식없이 해결했는지 핵심 아이디어를 이해하게 됐다. 딥마인드가 강화학습에 집착하는 이유를 알 것도 같다. 앞으로 쉽진 않겠지만 보드 게임 외에 다른 도메인에 응용해보는 것도 재밌을 것 같다. 그래도 역시 난 게임이 젤 재밌다.ㅋㅋ
    3. 좋은 사람들을 얻었다. 나도 처음에 시작할 때 설마 이게 되겠어? 하고 별 기대 없이 직장생활하면서 취미로 덕질(?)하던 것인데. 많은 사람들이 관심가져 주셨고, 모르는 것을 물어보면 잘 대답해주셨다. 이런 도움들이 프로젝트를 잘 마무리하는데 무엇보다 큰 힘이 됐다. IT계열 분들은 참 쿨한 것 같다. (아직 세상은 살만하구나.ㅠㅠ ) 도움주신 분들 정말 감사합니다!ㅎㅎㅎ

    마지막으로, 알파고 논문 세편을 수십번 읽고 3달간 직접 구현하고 나서 들은 생각을 한마디로 정리하자면 "데이터는 '양'보다 '질'이고, 질 좋은 데이터를 확보하는 것은 결국 알고리즘을 설계하는 아이디어에 달려있다."라고 감히 말하고 싶다.



