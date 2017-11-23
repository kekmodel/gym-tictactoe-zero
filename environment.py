import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class TicTacToeEnv(gym.Env):
    """gym.Env를 상속하여 틱택토 게임 환경 클래스 정의
        gym.Env: OpenAI Gym의 주요 클래스, 환경 뒤에서 이루어지는 변화들을 캡슐화 해놓음
    """
    metadata = {'render.modes': ['human']}  # _render()의 리턴 타입 구분: 사람 눈으로 볼거임
    reward_range = (-1, 1)  # 보상의 범위: 패배:-1, 무승부:0, 승리:1
    mark_O = 1  # O 표시의 대응값
    mark_X = 2  # X 표시의 대응값
    player_pool = np.array([mark_O, mark_X])  # 플레이어 랜덤 선택용

    def __init__(self):
        self.set_start = TicTacToeEnv.mark_O  # O표시가 선
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
        self.turn = self.set_start  # 상태 정의를 이쁘게 하려고 이름 바꾸기
        self.player = np.random.choice(
            TicTacToeEnv.player_pool, 1)  # 에이전트 선공 랜덤선택
        self.board = np.zeros(self.board_size**2)  # 보드 초기화 : 0이 9개인 배열
        self.state = [self.turn, self.board]  # 상태 초기화: [턴, 보드상태]의 리스트
        self.done = False  # 게임의 진행 상태: 안 끝남
        return self.state  # 상태 리스트 리턴

    def _step(self, action):
        """한번의 행동에 환경의 변수들이 어떻게 변하는지 정하는 함수
            승부가 나면 _reset()을 호출하여 환경을 초기화 해야 함
            action을 받아서 (state, reward, don, info)인 튜플 리턴

        Args:
             int: action

        Return:
            list: state
             int: reward
            bool: done
            dict: info
        """
        # 객체의 정해진 차례에만 액션하도록 수정 필요
        self.action = action
        if not self.done:
            if self.state[0] == TicTacToeEnv.mark_O:
                if self.state[1][self.action] == 0:
                    self.state[1][self.action] = TicTacToeEnv.mark_O
                    self.state[0] = TicTacToeEnv.mark_X
                else:
                    print('nope.')
                    return
            else:
                if self.state[1][self.action] == 0:
                    self.state[1][self.action] = TicTacToeEnv.mark_X
                    self.state[0] = TicTacToeEnv.mark_O
                else:
                    print('nope.')
                    return
        else:
            self._reset()
        return

    def chek_win(self, state):  # 승무패 체크 함수 최적화 알고리즘 생각해보기
        ...

    def _render(self, mode='human', close=False):
        if close:
            return


# 모니터링 용
sEnv = TicTacToeEnv()
print(sEnv.player_pool)
print(sEnv.state)
for action in range(14):
    sEnv._step(sEnv.action_space.sample())
    print(sEnv.state)
