import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


def check_win(state):  # state 승패체크 함수
    current_state = state
    loc_mark_O = np.zeros(9)  # 승패체크 전처리용 배열: O용
    loc_mark_X = np.zeros(9)  # X용
    if current_state[0] == 1:
        print(current_state)
        # 현재 보드에서 1이 표시된 인덱스를 받아와서 loc_mark_O의 같은 자리에 1을 넣기 나머진 0
        loc_mark_O[np.where(current_state[1] == 1)] = 1
        batch_state = [current_state[0], loc_mark_O]  # 승패를 필터링할 상태로 전처리
        print(batch_state, '\n')
    else:
        print(current_state)  # 현재 상태 확인
        # 현재 보드에서 2가 표시된 인덱스를 받아와서 loc_mark_X의 같은 자리에 1을 넣기 나머진 0
        loc_mark_X[np.where(current_state[1] == 2)] = 1
        batch_state = [current_state[0], loc_mark_X]  # 전처리 완료
        print(batch_state, '\n')  # 전처리 상태 확인
    # 승리패턴 8가지 구성
    win_pattern = np.array([np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]),
                            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0]),
                            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                            np.array([1, 0, 0, 1, 0, 0, 1, 0, 0]),
                            np.array([1, 0, 0, 1, 0, 0, 1, 0, 0]),
                            np.array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
                            np.array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
                            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])])
    # 전처리한 보드와 승리패턴 8가지와 비교하여 해당 정보 리턴
    for i in range(8):
        if np.array_equal(batch_state[1], win_pattern[i]):  # 패턴과 일치: 승리
            mark_type = batch_state[0]
            match_result = 1   # 승부남
            return [mark_type, match_result]  # 마크타입, 경기결과, 진행 리턴
        elif np.count_nonzero(current_state[1]) == 9:  # 보드가 꽉찼는데도 승부가 안남: 비김
            mark_type = batch_state[0]
            match_result = 0  # 비김
            return [mark_type, match_result]
        else:
            mark_type = batch_state[0]
            match_result = 2  # 승부안남 게임 계속
            return [mark_type, match_result]


class TicTacToeEnv(gym.Env):
    """gym.Env를 상속하여 틱택토 게임 환경 클래스 정의
        gym.Env: OpenAI Gym의 주요 클래스, 환경 뒤에서 이루어지는 변화들을 캡슐화(gym/core.py 참조)
    """
    metadata = {'render.modes': ['human']}  # _render()의 리턴 타입 구분: 사람 눈으로 볼거임
    reward_range = (-1, 1)  # 보상의 범위: 패배:-1, 무승부:0, 승리:1
    mark_O = 1  # O 표시의 대응값
    mark_X = 2  # X 표시의 대응값
    player_pool = np.array([mark_O, mark_X])  # 플레이어 랜덤 선택용

    def __init__(self):
        self.board_size = 3  # 보드크기 지정: 한변의 길이
        self._seed()  # 랜덤 시드 설정하는 함수 호출
        # 관찰공간인 3x3배열 정의에 쓸 튜플
        self.board_shape = (self.board_size, self.board_size)
        # 관찰 공간 정의: 3x3배열, 빈공간~꽉찬공간
        self.observation_space = spaces.Box(
            np.zeros(self.board_shape), np.ones(self.board_shape))
        self.action_space = spaces.Discrete(
            self.board_size**2)  # 행동 공간 정의: 3^2개, 0~8의 정수
        self._reset()  # 리셋 함수 호출

    def _seed(self, seed=None):  # 랜덤 시드 설정 함수: 한 에피소드 동안 유지하기 위함
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):  # 상태 초기화 함수
        self.turn = TicTacToeEnv.mark_O
        self.player = np.random.choice(
            TicTacToeEnv.player_pool, 1)  # 플레이어 마크 랜덤선택
        self.board = np.zeros(self.board_size**2)  # 보드 초기화 : 0이 9개인 배열
        # 상태 초기화: [턴, 보드상태]의 리스트. 첫턴:mark_O
        self.state = [self.turn, self.board]
        self.reward = 1
        self.info = {}
        self.done = False  # 게임의 진행 상태: 안 끝남
        return self.state  # 상태 리스트 리턴

    def _step(self, action):
        """한번의 행동에 환경의 변수들이 어떻게 변하는지 정하는 함수
            승부가 나면 _reset()을 호출하여 환경을 초기화 해야 함
            action을 받아서 (state, reward, done, info)인 튜플 리턴

        Args:
             int: action

        Return:
            list: state
             int: reward
            bool: done
            dict: info
        """
        self.action = action  # 액션 받아오기
        # 들어온 액션의 주체가 상태의 턴과 같고, 액션자리가 비어 있다면
        if self.action[0] == self.state[0] and self.state[1][self.action[1]] == 0:
                # 액션 요청 자리에 해당 턴의 인덱스를 넣는다
            self.state[1][self.action[1]] = self.state[0]
            check_state = check_win(self.state)  # 승부 체크 [액션 주체, 결과]
            # 액션 주체가 플레이어이고 승부가 났다면 플레이어 승리
            if check_state[0] == self.player and check_state[1] == 1:
                print('You Win')   # 승리 메시지 출력
                self.reward = 1  # 보상 +1
                self.done = True   # 게임 끝
                return self.state, self.reward, self.done, self.info
            elif check_state[1] == 1:  # 액션 주체가 플레이어가 아닌데 승부가 났으면 패배
                print('You Lose')  # 너 짐
                self.reward = -1  # 보상 -1
                self.done = True  # 게임 끝
                return self.state, self.reward, self.done, self.info
            elif check_state[1] == 0:  # 비겼다는 정보를 줄 경우
                print('Draw')  # 비김
                self.reward = 0  # 보상 0
                self.done = True  # 게임 끝
                return self.state, self.reward, self.done, self.info
            else:  # 승부도 안나고 비기지도 않으면?
                self.state[0] = TicTacToeEnv.mark_X  # 상태의 턴을 바꾸고
                self.done = False  # 게임 계속
                return self.state, self.reward, self.done, self.info
        # 들어온 액션의 주체와 상태의 턴이 다르면
        elif self.action[0] != self.state[0]:
            print('Not Your Turn')
            self.done = True  # 게임 중단
            return self.state, self.reward, self.done, self.info
        else:  # 액션 자리가 이미 차있으면
            print('No Mark')  # 안돼
            self.done = False  # 다시해
            self.info = {'NO'}  # 착수금지에 놓음
            return self.state, self.reward, self.done, self.info

    def _render(self, mode='human', close=False):  # 현재 상태를 그려주는 함수
        if close:
            return


# 모니터링 용
tEnv = TicTacToeEnv()
while not tEnv.done:
    tEnv.step([tEnv.player, tEnv.action_space.sample()])
tEnv.reset()
