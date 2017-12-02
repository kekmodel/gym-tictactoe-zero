import logging
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np  # c의 array 지원

logger = logging.getLogger(__name__)


def check_win(state):  # state 승패체크 함수
    current_state = state
    loc_mark_O = np.zeros(9, 'int32')  # 승패체크 전처리용 배열: O용
    loc_mark_X = np.zeros(9, 'int32')  # X용
    # 승리패턴 8가지 구성
    win_pattern = np.array([np.array([1, 1, 1, 3, 3, 3, 3, 3, 3]),
                            np.array([3, 3, 3, 1, 1, 1, 3, 3, 3]),
                            np.array([3, 3, 3, 3, 3, 3, 1, 1, 1]),
                            np.array([1, 3, 3, 1, 3, 3, 1, 3, 3]),
                            np.array([1, 3, 3, 1, 3, 3, 1, 3, 3]),
                            np.array([3, 1, 3, 3, 1, 3, 3, 1, 3]),
                            np.array([3, 3, 1, 3, 3, 1, 3, 3, 1]),
                            np.array([1, 3, 3, 3, 1, 3, 3, 3, 1])])
    if current_state[0] == 0:
        print(current_state)
        # 현재 보드에서 1이 표시된 인덱스를 받아와서 loc_mark_O의 같은 자리에 1을 넣기 나머진 0
        loc_mark_O[np.where(current_state[1] == 1)] = 1
        batch_state = [current_state[0], loc_mark_O]  # 승패를 필터링할 상태로 전처리
    else:
        print(current_state)  # 현재 상태 확인
        # 현재 보드에서 2가 표시된 인덱스를 받아와서 loc_mark_X의 같은 자리에 1을 넣기 나머진 0
        loc_mark_X[np.where(current_state[1] == 2)] = 1
        batch_state = [current_state[0], loc_mark_X]  # 전처리 완료

    # 전처리한 보드와 승리패턴 8가지와 비교하여 해당 정보 리턴
    judge = np.zeros((8, 9), 'int32')  # 비교 결과를 넣은 8*9 배열 초기화
    for i in range(8):   # 각각의 값을 비교해서
        for k in range(9):
            if batch_state[1][k] == win_pattern[i][k]:   # 일치하는 자리엔 1을 넣고
                judge[i][k] = 1
            else:
                judge[i][k] = 0   # 아닌 것은 0을 넣어
        if judge[i].sum() == 3:   # 그 배열을 더해서 총합이 3이면 승부난 거임
            mark_type = batch_state[0]  # 그럼 현재턴을 저장하고
            match_result = 1   # 승부났음을 저장하고
            return [mark_type, match_result]  # 리턴해라
        # 승부안났는데 현재 상태 꽉참
        elif np.count_nonzero(current_state[1]) == 9:
            mark_type = batch_state[0]  # 의미 없지만 리턴 타입 오류를 막자
            match_result = 0  # 비김
            return [mark_type, match_result]
    # 승부도 안났고 비기지도 않았으면 계속 진행해라
    mark_type = batch_state[0]
    mark_result = 2   # 0,1이 아닌 아무 수
    return [mark_type, mark_result]


class TicTacToeEnv(gym.Env):
    """gym.Env를 상속하여 틱택토 게임 환경 클래스 정의
        gym.Env: OpenAI Gym의 주요 클래스, 환경 뒤에서 이루어지는 변화들을 캡슐화(gym/core.py 참조)
    """
    # _render()의 리턴 타입 구분: 사람 눈으로 볼거고 rgb값 60프레임으로
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 60}
    reward_range = (-1, 0, 1)  # 보상의 범위 참고: 패배:-1, 무승부:0, 승리:1

    def __init__(self):
        self.mark_O = 0  # O 표시의 대응값
        self.mark_X = 1  # X 표시의 대응값
        self.mark_dict = {0: 'Mark O', 1: 'Mark X'}
        self.player = 0  # player의 타입을 설정하는 멤버
        self.board_size = 9  # 3x3 보드 사이즈
        # 관찰 공간 정의: [순서, 보드]: [1~2, 0~8]
        self.observation_space = np.array(
            [spaces.Discrete(2), spaces.Discrete(9)])
        self.action_space = self.observation_space  # 액션 공간 == 관찰 공간
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
        action_mark = action[0]  # 액션 주체 인덱스
        action_target = action[1]  # 액션 타겟 좌표
        state_turn = self.state[0]
        state_loc = self.state[1][action_target]
        # 들어온 액션의 주체가 상태의 턴과 같고, 상태 보드가 비어있는 경우
        if action_mark == state_turn and state_loc == 0:
            # 액션 요청 자리에 해당 턴의 인덱스+1를 넣는다
            self.state[1][action_target] = action_mark + 1

            check_state = check_win(self.state)  # 승부 체크 리턴:[액션 주체, 결과]

            # 액션 주체가 플레이어인데 승부가 났다면 플레이어 승리
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
                if self.state[0] == self.mark_O:  # 상태의 턴을 바꾸고
                    self.state[0] = self.mark_X
                else:
                    self.state[0] = self.mark_O
                reward = 0  # 보상 0
                self.done = False  # 게임 계속
                return self.state, reward, self.done, {}
        # 들어온 액션의 주체와 상태의 턴이 다르면
        elif action[0] != self.state[0]:
            print('turn error')
            reward = 0
            self.done = True  # 게임 중단
            return self.state, reward, self.done, {}

        else:  # 액션 자리가 이미 차있으면
            print('Overlab Lose')  # 반칙패
            reward = -1  # 보상 -1
            self.done = True
            return self.state, reward, self.done, {}

    def _render(self, mode='human', close=False):  # 현재 상태를 그려주는 함수
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(300, 300)

            self.line_1 = rendering.Line((0, 100), (300, 100))
            self.line_1.set_color(0, 0, 0)
            self.line_2 = rendering.Line((0, 200), (300, 200))
            self.line_2.set_color(0, 0, 0)
            self.line_a = rendering.Line((100, 0), (100, 300))
            self.line_a.set_color(0, 0, 0)
            self.line_b = rendering.Line((200, 0), (200, 300))
            self.line_b.set_color(0, 0, 0)

            self.viewer.add_geom(self.line_1)
            self.viewer.add_geom(self.line_2)
            self.viewer.add_geom(self.line_a)
            self.viewer.add_geom(self.line_b)

            self.image_O1 = rendering.Image("img/O.png", 96, 96)
            self.trans_O1 = rendering.Transform(self.render_loc[0])
            self.image_O1.add_attr(self.trans_O1)

            self.image_O2 = rendering.Image("img/O.png", 96, 96)
            self.trans_O2 = rendering.Transform(self.render_loc[1])
            self.image_O2.add_attr(self.trans_O2)

            self.image_O3 = rendering.Image("img/O.png", 96, 96)
            self.trans_O3 = rendering.Transform(self.render_loc[2])

            self.image_O4 = rendering.Image("img/O.png", 96, 96)
            self.trans_O4 = rendering.Transform(self.render_loc[3])
            self.image_O4.add_attr(self.trans_O4)

            self.image_O5 = rendering.Image("img/O.png", 96, 96)
            self.trans_O5 = rendering.Transform(self.render_loc[4])
            self.image_O5.add_attr(self.trans_O5)

            self.image_O6 = rendering.Image("img/O.png", 96, 96)
            self.trans_O6 = rendering.Transform(self.render_loc[5])
            self.image_O6.add_attr(self.trans_O6)

            self.image_O7 = rendering.Image("img/O.png", 96, 96)
            self.trans_O7 = rendering.Transform(self.render_loc[6])
            self.image_O7.add_attr(self.trans_O7)

            self.image_O8 = rendering.Image("img/O.png", 96, 96)
            self.trans_O8 = rendering.Transform(self.render_loc[7])
            self.image_O8.add_attr(self.trans_O8)

            self.image_O9 = rendering.Image("img/O.png", 96, 96)
            self.trans_O9 = rendering.Transform(self.render_loc[8])
            self.image_O9.add_attr(self.trans_O9)

            self.image_X1 = rendering.Image("img/X.png", 96, 96)
            self.trans_X1 = rendering.Transform(self.render_loc[0])
            self.image_X1.add_attr(self.trans_X1)

            self.image_X2 = rendering.Image("img/X.png", 96, 96)
            self.trans_X2 = rendering.Transform(self.render_loc[1])
            self.image_X2.add_attr(self.trans_X2)

            self.image_X3 = rendering.Image("img/X.png", 96, 96)
            self.trans_X3 = rendering.Transform(self.render_loc[2])
            self.image_X3.add_attr(self.trans_X3)

            self.image_X4 = rendering.Image("img/X.png", 96, 96)
            self.trans_X4 = rendering.Transform(self.render_loc[3])
            self.image_X4.add_attr(self.trans_X4)

            self.image_X5 = rendering.Image("img/X.png", 96, 96)
            self.trans_X5 = rendering.Transform(self.render_loc[4])
            self.image_X5.add_attr(self.trans_X5)

            self.image_X6 = rendering.Image("img/X.png", 96, 96)
            self.trans_X6 = rendering.Transform(self.render_loc[5])
            self.image_X6.add_attr(self.trans_X6)

            self.image_X7 = rendering.Image("img/X.png", 96, 96)
            self.trans_X7 = rendering.Transform(self.render_loc[6])
            self.image_X7.add_attr(self.trans_X7)

            self.image_X8 = rendering.Image("img/X.png", 96, 96)
            self.trans_X8 = rendering.Transform(self.render_loc[7])
            self.image_X8.add_attr(self.trans_X8)

            self.image_X9 = rendering.Image("img/X.png", 96, 96)
            self.trans_X9 = rendering.Transform(self.render_loc[8])
            self.image_X9.add_attr(self.trans_X9)

            state = self.state.copy()
            map_dict = dict(zip([0, 1, 2, 3, 4, 5, 6, 7, 8], state[1]))

            if 0 in map_dict.keys():
                if map_dict[0] == 1:
                    self.viewer.add_geom(self.image_O1)
                elif map_dict[0] == 2:
                    self.viewer.add_geom(self.image_X1)

            if 1 in map_dict.keys():
                if map_dict[1] == 1:
                    self.viewer.add_geom(self.image_O2)
                elif map_dict[1] == 2:
                    self.viewer.add_geom(self.image_X2)

            if 2 in map_dict.keys():
                if map_dict[2] == 1:
                    self.viewer.add_geom(self.image_O3)
                elif map_dict[2] == 2:
                    self.viewer.add_geom(self.image_X3)

            if 3 in map_dict.keys():
                if map_dict[3] == 1:
                    self.viewer.add_geom(self.image_O4)
                elif map_dict[3] == 2:
                    self.viewer.add_geom(self.image_X4)

            if 4 in map_dict.keys():
                if map_dict[4] == 1:
                    self.viewer.add_geom(self.image_O5)
                elif map_dict[4] == 2:
                    self.viewer.add_geom(self.image_X5)

            if 5 in map_dict.keys():
                if map_dict[5] == 1:
                    self.viewer.add_geom(self.image_O6)
                elif map_dict[5] == 2:
                    self.viewer.add_geom(self.image_X6)

            if 6 in map_dict.keys():
                if map_dict[6] == 1:
                    self.viewer.add_geom(self.image_O7)
                elif map_dict[6] == 2:
                    self.viewer.add_geom(self.image_X7)

            if 7 in map_dict.keys():
                if map_dict[7] == 1:
                    self.viewer.add_geom(self.image_O8)
                elif map_dict[7] == 2:
                    self.viewer.add_geom(self.image_X8)

            if 8 in map_dict.keys():
                if map_dict[8] == 1:
                    self.viewer.add_geom(self.image_O9)
                elif map_dict[8] == 2:
                    self.viewer.add_geom(self.image_X9)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


# 테스트 용
env = TicTacToeEnv()
env.seed()
env.reset()
env.player = 0  # 나는 동그라미!

for i in range(1000):
    action = [i % 2, env.action_space[1].sample()]
    observation, reward, done, info = env.step(action)
    print(reward)
    if done:
        env.reset()
