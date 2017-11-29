import logging
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

logger = logging.getLogger(__name__)


def check_win(state):  # state 승패체크 함수
    current_state = state
    loc_mark_O = np.zeros(9, 'int32')  # 승패체크 전처리용 배열: O용
    loc_mark_X = np.zeros(9, 'int32')  # X용
    if current_state[0] == 0:
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
    # _render()의 리턴 타입 구분: 사람 눈으로 볼거고 rgb값 60프레임으로
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 60}
    reward_range = (-1, 0, 1)  # 보상의 범위: 패배:-1, 무승부:0, 승리:1

    def __init__(self):
        self.mark_O = 0  # O 표시의 대응값
        self.mark_X = 1  # X 표시의 대응값
        self.board_size = 9  # 3x3 보드 사이즈
        # 관찰 공간 정의: [순서, 보드]: [1~2, 0~8]
        self.observation_space = np.array(
            [spaces.Discrete(2), spaces.Discrete(9)])
        self.action_space = self.observation_space.copy()  # 액션 공간 == 관찰 공간
        self.viewer = None  # 뷰어 초기화
        self.state = None  # 상태 초기화
        self.done = False
        self.first_turn = self.mark_O  # 첫턴은 O
        self._seed()  # 랜덤 시드 설정하는 함수 호출
        self.render_loc = {0: (50, 250), 1: (150, 250), 2: (250, 250),
                           3: (50, 150), 4: (150, 150), 5: (250, 150),
                           6: (50, 50), 7: (150, 50), 8: (250, 50)}

    def _seed(self, seed=None):  # 랜덤 시드 설정 함수: 한 에피소드 동안 유지하기 위함
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):  # 상태 리셋 함수
        self.done = False
        self.player = np.random.choice(
            [self.mark_O, self.mark_X])  # 플레이어 랜덤 선택
        # 상태 리셋: [턴, [보드 상태:9개 배열]] 리스트
        self.state = [self.first_turn, np.zeros(self.board_size, 'int32')]
        print('state reset')
        return self.state  # 상태 리스트 리턴

    def _step(self, action):
        """한번의 행동에 환경의 변수들이 어떻게 변하는지 정하는 함수
            승부가 나면 _reset()을 호출하여 환경을 초기화 해야 함
            action을 받아서 (state, reward, done, info)인 튜플 리턴

        Args:
             list: action

        Return:
            list: state
             int: reward
            bool: done
            dict: info
        """
        state = self.state
        action_mark = action[0]  # 액션 주체 인덱스
        action_target = action[1]  # 액션 타겟 좌표
        state_turn = state[0]
        state_loc = state[1][action_target]
        # 들어온 액션의 주체가 상태의 턴과 같고, 상태 보드가 비어있는 경우
        if action_mark == state_turn and state_loc == 0:
            # 액션 요청 자리에 해당 턴의 인덱스+1를 넣는다
            state[1][action[1]] = action_mark + 1

            check_state = check_win(state)  # 승부 체크 [액션 주체, 결과]

            # 액션 주체가 플레이어이고 승부가 났다면 플레이어 승리
            if check_state[0] == self.player and check_state[1] == 1:
                print('You Win')   # 승리 메시지 출력
                reward = 1  # 보상 +1
                self.done = True   # 게임 끝
                return self.state, reward, self.done, {}
            elif check_state[1] == 1:  # 액션 주체가 플레이어가 아닌데 승부가 났으면 패배
                print('You Lose')  # 너 짐
                reward = -1  # 보상 -1
                self.done = True  # 게임 끝
                return self.state, reward, self.done, {}
            elif check_state[1] == 0:  # 비겼다는 정보를 줄 경우
                print('Draw')  # 비김
                reward = 0  # 보상 0
                self.done = True  # 게임 끝
                return self.state, reward, self.done, {}
            else:  # 승부도 안나고 비기지도 않으면?
                state[0] = self.mark_X  # 상태의 턴을 바꾸고
                reward = 0
                self.done = False  # 게임 계속
                return self.state, reward, self.done, {}
        # 들어온 액션의 주체와 상태의 턴이 다르면
        elif action[0] != state[0]:
            print('turn error')
            self.done = True  # 게임 중단
            reward = None
            return self.state, reward, self.done, {'turn error'}
        else:  # 액션 자리가 이미 차있으면
            print('overlap error')  # 안돼
            self.done = False  # 안끝남

    def _render(self, mode='human', close=False):  # 현재 상태를 그려주는 함수
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 300
        screen_height = 300

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.line_1 = rendering.Line((0, 100), (300, 100))
            self.line_1.set_color(0, 0, 0)

            self.line_2 = rendering.Line((0, 200), (300, 200))
            self.line_2.set_color(0, 0, 0)

            self.line_a = rendering.Line((100, 0), (100, 300))
            self.line_a.set_color(0, 0, 0)

            self.line_b = rendering.Line((200, 0), (200, 300))
            self.line_b.set_color(0, 0, 0)

            self.image_O = rendering.Image("img/O.png", 96, 96)
            self.trans_O = rendering.Transform(self.render_loc[action[1]])
            self.image_O.add_attr(self.trans_O)

            self.image_X = rendering.Image("img/X.png", 96, 96)
            self.trans_X = rendering.Transform(self.render_loc[action[1]])
            self.image_X.add_attr(self.trans_X)

            self.viewer.add_geom(self.line_1)
            self.viewer.add_geom(self.line_2)
            self.viewer.add_geom(self.line_a)
            self.viewer.add_geom(self.line_b)

            self.viewer.add_geom(self.image_O)
            self.viewer.add_geom(self.image_X)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


# 모니터링 용
env = TicTacToeEnv()
env.seed()
env.reset()
for _ in range(60):
    action = [0, 4]
    state, reward, done, info = env.step(action)
    env.render()
