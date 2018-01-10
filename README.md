# gym-tictactoe
![tictactoe](./img/Tic_Tac_Toe.gif)
----------------------------
-틱택토 예시-

## <비전공자가 찔러보는 강화학습>
gym을 설치한 후 실행하면 돌아감. 파이썬 3.5로 만들었음.

약간의 허접한 렌더링도 지원함.ㅋㅋㅋ

gym폴더가 gym-tictactoe폴더가 있는 곳에 있어야 함.

https://github.com/openai/gym


### 나는..?!

코딩 수준: 코딩의 코자도 모르는 코린이 (파이썬과 공부한 순수시간 24시간(?)에서 시작)

전공: 통계에 장애가 있는 산업공학과 학사 02학번(?)

강화 학습 수준: 반도의 흔한 알파고 빠돌이

영어 수준: 구글 번역 성애자

--> 삽질의 삽질의 삽질의 연속...

--> 그렇지만 재밌어서 계속 삽질...

--> 하다보니 어느 정도 성과가 나기 시작!!(재능 발견?)

--> 여러분도 함께 해요~~~ :P

### 1. OpenAI Gym기반 틱택토 환경만들기
- git, text editor, jupyter 등 개발환경 설정
- 파이썬 기초 공부
    - 기본 문법
    - 객체지향 개념
    - Numpy 공부
- gym 구조 파악

<2017년>

11월 21일: 코딩 시작

12월 3일: 틱택토 환경 프로토타입 구현 (렌더링 포함)

<공부하며 휴식기간> 

12월 13일: 버그 수정, gym 형식에 맞게 수정

12월 18일: state 타입 수정 (알파고 제로 방식)

12월 19일: 틱택토 환경 정식버전 완성

### 2. 에이전트 만들기
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

12월 19일: 코딩 시작

12월 25일: 간단한 ResNet 구현 (PyTorch)

12월 31일: state를 hash로 바꿔 다뤄봄

<2018년>

1월 1일: PUCT-MCTS 프로토타입 구현

1월 3일: (state, edge) set을 hdf5로 저장 구현

1월 6일: PUCT-MCTS 정식버전 구현

             - 알고리즘 오류 수정, overflow 잡기
             - 20,000 episode 데이터 저장
             - 완료시 Slack에 메시지 보내는 기능 추가

1월 10일: Agent 프로토타입 구현(RL)

              - 데이터로 트리 재구현               
              - edge의 방문횟수를 softmax를 사용해 확률로 전환
              - 정책함수 구현

ing...



