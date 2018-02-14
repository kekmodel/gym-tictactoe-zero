# -*- coding: utf-8 -*-
import tictactoe_env
import neural_network

import time
import hashlib
from collections import deque, defaultdict

import torch
from torch.autograd import Variable
# from torch.optim import lr_scheduler

import slackweb
import dill as pickle
import numpy as np
np.set_printoptions(suppress=True)

PLAYER = 0
OPPONENT = 1
MARK_O = 0
MARK_X = 1
N, W, Q, P = 0, 1, 2, 3

EPISODE = 1
SAVE_CYCLE = 1600

NUM_CHANNEL = 128
BATCH_SIZE = 32

PLANE = np.zeros((3, 3), 'int').flatten()


class MCTS(object):
    """몬테카를로 트리 탐색 클래스

    시뮬레이션을 통해 train 데이터 생성 (state, edge 저장)

    state
    ------
    각 주체당 4수까지 저장해서 state_new 로 만듦

        9x3x3 numpy array -> 1x81 tuple

    edge
    -----
    현재 state에서 착수 가능한 모든 action자리에 4개의 정보 저장

    type: 3x3x4 numpy array

        9개 좌표에 4개의 정보 N, W, Q, P 매칭
        N: edge 방문횟수, W: 보상누적값, Q: 보상평균값(W/N), P: 선택 확률 추정치
        edge[좌표행][좌표열][번호]로 접근

    """

    def __init__(self):
        # ROM
        self.state_memory = deque(maxlen=9 * EPISODE)
        self.tree_memory = defaultdict(lambda: np.zeros((3, 3, 4), 'float'))

        # model
        self.pv_net = neural_network.PolicyValueNet(NUM_CHANNEL)

        # hyperparameter
        self.c_puct = 5
        self.epsilon = 0.25
        self.alpha = 0.7

        # reset_step member
        self.edge = None
        self.node = None
        self.puct = None
        self.total_visit = None
        self.empty_loc = None
        self.state = None
        self.state_new = None
        self.state_tensor = None
        self.state_variable = None
        self.p_theta = None
        self.value = None
        self.pi = None

        # reset_episode member
        self.my_history = None
        self.your_history = None
        self.node_memory = None
        self.edge_memory = None
        self.action_memory = None
        self.value_memory = None
        self.pi_memory = None
        self.action_count = None
        self.board = None
        self.first_turn = None
        self.user_type = None

        # member init
        self._reset_step()
        self._reset_episode()

    def _reset_step(self):
        self.edge = np.zeros((3, 3, 4), 'float')
        self.node = None
        self.puct = np.zeros((3, 3), 'float')
        self.total_visit = 0
        self.empty_loc = None
        self.state = None
        self.state_new = None
        self.state_tensor = None
        self.state_variable = None
        self.p_theta = None
        self.value = None
        self.pi = None

    def _reset_episode(self):
        self.my_history = deque([PLANE] * 4, maxlen=4)
        self.your_history = deque([PLANE] * 4, maxlen=4)
        self.node_memory = deque(maxlen=9)
        self.edge_memory = deque(maxlen=9)
        self.action_memory = deque(maxlen=9)
        self.value_memory = deque(maxlen=9)
        self.pi_memory = deque(maxlen=9)
        self.action_count = 0
        self.board = None
        self.first_turn = None
        self.user_type = None

    def select_action(self, state):
        """raw state를 받아 변환 및 저장 후 action을 리턴하는 외부 메소드.

        state 변환
        ---------
        state_new -> node & state_variable

            state_new: 9x3x3 numpy array.
                유저별 최근 4-histroy 저장하여 재구성. (저장용)

            state_variable: 1x9x3x3 torch.autograd.Variable.
                신경망의 인수로 넣을 수 있게 조정. (학습용)

            node: str. (hashlib.md5)
                state_new를 string으로 바꾼 후 hash 생성. (탐색용)

        action 선택
        -----------
        puct 값이 가장 높은 곳을 선택함, 동점이면 랜덤 선택.

            action: 1x3 tuple.
            action = (피아식별, 보드의 x좌표, 보드의 y좌표)

        """

        # 턴 관리
        self.action_count += 1

        # 호출될 때마다 첫턴 기준 교대로 행동주체 바꿈, 최종 action에 붙여줌
        # PLAYER or OPPONENT
        self.user_type = (self.first_turn + self.action_count - 1) % 2

        # state 변환
        self.state = state
        self.state_new = self._convert_state(state)

        # new state 저장
        self.state_memory.appendleft(self.state_new)

        # new state에 Variable 씌움
        self.state_tensor = torch.from_numpy(self.state_new)
        self.state_variable = Variable(
            self.state_tensor.view(1, 9, 3, 3).float(), requires_grad=True)

        # 신경망에 인풋으로 넣고 아웃풋 받기 (p, v)
        self.p_theta, self.value = self.pv_net(self.state_variable)
        print('[NN_output]\n', self.p_theta.data, self.value.data[0])

        # state -> 문자열 -> hash로 변환 후 저장 (new state 대신 dict의 key로 사용)
        self.node = hashlib.md5(self.state_new.tostring()).hexdigest()
        self.node_memory.appendleft(self.node)

        # edge 세팅: tree 탐색 -> edge 생성 or 세팅 -> PUCT 점수 계산
        self._set_edge()

        # PUCT 점수 출력
        print('***  PUCT Score  ***')
        print(self.puct.round(decimals=2))
        print('')

        # 빈자리가 아닌 곳은 PUCT값으로 -9999를 넣어 빈자리가 최댓값이 되는 것 방지
        puct = self.puct.tolist()
        for i, v in enumerate(puct):
            for k, s in enumerate(v):
                if [i, k] not in self.empty_loc.tolist():
                    puct[i][k] = -9999

        # PUCT가 최댓값인 곳 찾기
        self.puct = np.array(puct)
        puct_max = np.argwhere(self.puct == self.puct.max()).tolist()

        # 최댓값 동점인 곳 처리
        move_target = puct_max[np.random.choice(len(puct_max))]

        # 최종 action 구성 (행동주체 + 좌표) 접붙히기
        action = np.r_[self.user_type, move_target]

        # action 저장 및 step member 초기화
        self.action_memory.appendleft(action)
        self._reset_step()

        # tuple로 action 리턴
        return tuple(action)

    def _convert_state(self, state):
        """state변환 메소드: action 주체별 최대 4수까지 history를 저장하여 새로운 state로 변환."""

        if abs(self.user_type - 1) == PLAYER:
            self.my_history.appendleft(state[PLAYER].flatten())
        else:
            self.your_history.appendleft(state[OPPONENT].flatten())
        state_new = np.r_[np.array(self.my_history).flatten(),
                          np.array(self.your_history).flatten(),
                          self.state[2].flatten()]
        return state_new

    def _set_edge(self):
        """확장할 edge의 초기화 하는 메소드.

        dict{node: edge}인 MCTS Tree 구성
        Q, P를 계산하여
        9개의 좌표에 PUCT값을 계산하여 매칭하는 메소드.

        """

        # tree에서 현재 node를 검색하여 해당 edge의 누적정보 가져오기
        self.edge = self.tree_memory[self.node]

        # root node면 확률에 노이즈
        if self.action_count == 1:
            self.pr = (1 - self.epsilon) * self.p_theta + self.epsilon * \
                np.random.dirichlet(
                    self.alpha * np.ones(self.legal_move_n))

        # 현재 보드에서 착수가능한 수 저장
        self.board = self.state[PLAYER] + self.state[OPPONENT]
        self.empty_loc = np.argwhere(self.board == 0)

        # edge의 총 방문횟수 계산 및 출력
        for i in range(3):
            for j in range(3):
                self.total_visit += self.edge[i][j][N]
        print('(visit count: %d)\n' % (self.total_visit + 1))

        for i in range(self.legal_move_n):
            self.edge[tuple(self.empty_loc[i])][P] = self.pr[i]

        # Q값 계산 후 배치
        for c in range(3):
            for r in range(3):
                if self.edge[c][r][N] != 0:
                    self.edge[c][r][Q] = self.edge[c][r][W] / \
                        self.edge[c][r][N]

                # 각자리의 PUCT 계산
                self.puct[c][r] = self.edge[c][r][Q] + \
                    self.c_puct * \
                    self.edge[c][r][P] * \
                    np.sqrt(self.total_visit) / (1 + self.edge[c][r][N])

        # Q, P값을 배치한 edge 에피소드 동안 저장
        self.edge_memory.appendleft(self.edge)

    def backup(self, reward):
        """에피소드가 끝나면 지나온 edge의 N과 W를 업데이트."""

        steps = self.action_count
        for i in range(steps):
            # W 배치
            # 내가 지나온 edge는 reward 로
            if self.action_memory[i][0] == PLAYER:
                self.edge_memory[i][tuple(
                    self.action_memory[i][1:])][W] += reward
            # 상대가 지나온 edge는 -reward 로
            else:
                self.edge_memory[i][tuple(
                    self.action_memory[i][1:])][W] -= reward
            # N 배치
            self.edge_memory[i][tuple(self.action_memory[i][1:])][N] += 1
            # N, W, Q, P 가 계산된 edge들을 Tree에 최종 업데이트
            self.tree_memory[self.node_memory[i]] = self.edge_memory[i]

        self._reset_episode()


if __name__ == "__main__":
    # 시작 시간 측정
    start = time.time()

    # 환경 생성
    env = tictactoe_env.TicTacToeEnv()

    # 셀프 플레이 인스턴스 생성
    zero_play = MCTS()

    # 통계용
    result = {1: 0, 0: 0, -1: 0}
    win_mark_O = 0

    # 시뮬레이션 시작
    for e in range(EPISODE):
        # raw state 생성
        state = env.reset()

        # 에피소드 출력
        print('=' * 65, '\nEpisode: %d' % (e + 1))

        # 선공 정하고 교대로 하기 -> 환경에 알림
        zero_play.first_turn = (PLAYER + e) % 2
        env.player_color = zero_play.first_turn

        # 진행 변수 초기화
        done = False

        while not done:
            # 보드 상황 출력: 내 착수:1, 상대 착수:2
            print("---- BOARD ----")
            print(state[PLAYER] + state[OPPONENT] * 2.0)

            # action 선택하기
            action = zero_play.select_action(state)

            # step 진행
            state, reward, done, info = env.step(action)

        if done:
            # 승부난 보드 보기
            print("- FINAL BOARD -")
            print(state[PLAYER] + state[OPPONENT] * 2.0)
            print("")

            # 보상을 지나온 edge에 백업
            zero_play.backup(reward)

            # 승부 결과 체크
            result[reward] += 1

            # 선공으로 이긴 경우 체크 (밸런스 확인)
            if reward == 1:
                if env.player_color == 0:
                    win_mark_O += 1

        # SAVE_CYCLE 마다 Data 저장
        if (e + 1) % SAVE_CYCLE == 0:
            with open('data/state_memory_e{}.pkl'.format(e + 1), 'wb') as f:
                pickle.dump(zero_play.state_memory, f, pickle.HIGHEST_PROTOCOL)
            with open('data/tree_memory_e{}.pkl'.format(e + 1), 'wb') as f:
                pickle.dump(zero_play.tree_memory, f, pickle.HIGHEST_PROTOCOL)

            # 저장 알림
            print('[{} Episode Data Saved]'.format(e + 1))

            # 종료 시간 측정
            finish = round(float(time.time() - start))

            # 에피소드 통계 문자열 생성
            statics = ('\nWin: {}  Lose: {}  Draw: {}  Winrate: {:0.1f}%  \
WinMarkO: {}'.format(result[1], result[-1], result[0],
                     1 / (1 + np.exp(result[-1] / EPISODE) /
                          np.exp(result[1] / EPISODE)) * 100,
                     win_mark_O))

            # 통계 출력
            print('=' * 65, statics)

            # 슬랙에 보고 메시지 보내기
            slack = slackweb.Slack(
                url="https://hooks.slack.com/services/T8P0E384U/B8PR44F1C/\
4gVy7zhZ9teBUoAFSse8iynn")
            slack.notify(
                text="Finished: {} episode in {}s".format(e + 1, finish))
            slack.notify(text=statics)
