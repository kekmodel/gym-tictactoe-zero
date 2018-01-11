# gym-tictactoe
![tictactoe](./img/Tic_Tac_Toe.gif)
----------------------------
-틱택토 예시-

## <비전공자가 찔러보는 강화학습>
gym을 설치한 후 실행하면 돌아갑니다. 파이썬 3.5로 만들었습니다.

약간의 허접한 렌더링도 지원합니다.ㅋ

gym폴더와 gym-tictactoe폴더가 같은 곳에 있으면 안전합니다.

### 저는..?!

알파고를 사랑하는 호기심 많은 보통사람 

    코딩 수준: 코딩의 코자도 모르는 코린이 (파이썬을 공부한지 2주 정도에서 시작)
    전공: 기억을 상실한 산업공학 학사 02학번(?)
    직업: 겨우 밥벌어 먹는 수학 강사
    강화 학습 수준: 반도의 흔한 알파고 빠돌이
    영어 수준: 구글 번역 성애자

당.연.히 --> 삽질의 삽질의 삽질의 연속... (파도 파도 끝이 없는..)

--> 그렇지만 재밌어서 계속 삽질... (여가와 주말 자진 반납)

--> 오기로 하다보니 어느 정도 성과가 나기 시작!!(오? 재능 발견?)

--> 여러분도 함께 해요~~~ :P (페이스북 친추걸고 같이 스터디해요! https://www.facebook.com/kekmodel)


## python 3 설치

### Mac OS X

xcode 컴파일러 설치

        xcode-select --install

brew 설치

        ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

PATH 설정

        export PATH=/usr/local/bin:/usr/local/sbin:$PATH

파이썬 설치

        brew install python3

### Windows

download python 3 (https://www.python.org/ftp/python/3.6.4/python-3.6.4.exe)

path 지정

        C:\Python36\;C:\Python36\Scripts\

pip 설치

        python -m pip install -U pip

### Linux

        sudo apt-get update
        sudo apt-get install python3.6

### 모든 OS (다 귀찮으면..)

Anaconda 설치 (https://www.anaconda.com/download)


## git 설치 (https://git-scm.com/downloads)


## gym 설치 

        git clone https://github.com/openai/gym.git
        cd gym
        pip install -e .


## gym-tictactoe 설치 (제가 만든 거)
    
        git clone https://github.com/kekmodel/gym-tictactoe.git


## 실행방법

    text editor or IDE 로 build (단, human_interface.py 는 interpreter 환경만 됨)

### AI와 한판 붙고 싶다면? (허접함 주의)

        cd gym-tictactoe
        python human_interface.py

default: 10판 승부, 선공 랜덤, 착수: 1 ~ 9번 (콘솔창에;;) 

    [1][2][3]
    [4][5][6]
    [7][8][9] 


## 프로젝트 진행 상황

### 1. OpenAI Gym기반 틱택토 환경만들기 (완료)
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

12월 19일: 틱택토 환경 정식버전 완성 (tictactoe_env.py)

### 2. 에이전트 만들기 (진행중)
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

12월 25일: PyTorch로 간단한 ResNet 구현 (neural_network_cpu.py)

12월 31일: state를 hash로 바꿔 다뤄봄

             - dict의 key로 쓸려고
             - 검색 속도 빨라짐
             - 뭘 기준으로 바뀌는지는 공부 필요

<2018년>

1월 1일: PUCT-MCTS 프로토타입 구현
           
             - dict로 접근해보았음

1월 3일: (state, edge) set을 hdf5로 저장 구현 (data/)

1월 6일: PUCT-MCTS 정식버전 구현 (mcts_zero.py)

             - 알고리즘 오류 수정
             - 첫번째 state에 Dirichlet 노이즈 설정(e-greedy) 
             - 20,000 episode 데이터 저장
             - 완료시 Slack에 메시지 보내는 기능 추가

1월 10일: RL Agent 프로토타입 구현 (agent_rl.py)

              - 딥러닝 없이 순수 강화학습만 활용한 버전
              - 데이터로 트리 재구현 
              - edge의 방문횟수를 softmax를 사용해 확률로 전환
              - 정책함수 구현
              - self Play 전적 400전 400무

1월 11일: Human interface 구현

              - 사람과 처음으로 게임을 함: 고수에겐 약함
              - AI가 선일 때는 무적이나 상대가 선일 땐 특정 상태에서 탐험부족으로 최적 판단 못함
                  -> expand 과정을 추가하여 탐험확률 높임, episode 회수 2배 증가시킴 
              - 방문횟수가 0인데 확률은 0이 아닌 경우가 있어서 반칙패 생김
                  -> 정책함수의 확률 계산 방법을 softmax에서 일반평균법으로 바꿈

ing...



