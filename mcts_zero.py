# -*- coding: utf-8 -*-
import tictactoe_env

import time
from collections import deque, defaultdict

import slackweb
import numpy as np
np.set_printoptions(suppress=True)

PLAYER = 0
OPPONENT = 1
MARK_O = 0
MARK_X = 1
N, W, Q, P = 0, 1, 2, 3
EPISODE = 800
SAVE_CYCLE = 800


class MCTS(object):
    """몬테카를로 트리 탐색 클래스

    시뮬레이션을 통해 train 데이터 생성 (state, edge 저장)

    state
    ------
    각 주체당 4수까지 저장해서 state_new로 만듦

        9x3x3 numpy array -> 1x81 tuple (저장용)

    edge
    -----
    현재 state에서 착수 가능한 모든 action자리에 4개의 정보 저장

    type: 3x3x4 numpy array

        9개 좌표에 4개의 정보 N, W, Q, P 매칭
        N: edge 방문횟수, W: 보상누적값, Q: 보상평균(W/N), P: edge 선택 사전확률
        edge[좌표행][좌표열][번호]로 접근

    """

    def __init__(self):
        # memories
        self.state_memory = deque(maxlen=9 * EPISODE)
        self.tree_memory = defaultdict(lambda: np.zeros((3, 3, 4), 'float'))

        # hyperparameter
        self.c_puct = 5
        self.epsilon = 0.25
        self.alpha = 0.7

        # reset_step member
        self.pr = None
        self.puct = None
        self.edge = None
        self.empty_loc = None
        self.legal_move_n = None
        self.total_visit = None
        self.first_turn = None
        self.user_type = None
        self.state = None
        self.state_new = None
        self.node = None

        # reset_episode member
        self.node_memory = None
        self.edge_memory = None
        self.action_memory = None
        self.action_count = None
        self.my_history = None
        self.your_history = None
        self.board = None

        # member 초기화
        self._reset_step()
        self._reset_episode()

    def _reset_step(self):
        self.edge = np.zeros((3, 3, 4), 'float')
        self.node = None
        self.puct = np.zeros((3, 3), 'float')
        self.total_visit = 0
        self.empty_loc = None
        self.legal_move_n = 0
        self.pr = 0.
        self.state = None
        self.state_new = None

    def _reset_episode(self):
        plane = np.zeros((3, 3)).flatten()
        self.my_history = deque([plane, plane, plane, plane], maxlen=4)
        self.your_history = deque([plane, plane, plane, plane], maxlen=4)
        self.node_memory = deque(maxlen=9)
        self.edge_memory = deque(maxlen=9)
        self.action_memory = deque(maxlen=9)
        self.action_count = 0
        self.board = None
        self.first_turn = None
        self.user_type = None

    def select_action(self, state):
        """raw state를 받아 변환 및 저장 후 action을 리턴하는 외부 메소드.

        state 변환
        ----------
        state -> state_new -> node

            state_new: 9x3x3 numpy array.
                유저별 최근 4-histroy 저장하여 재구성. (저장용)

            node: str. (hash)
                state_new를 string으로 바꾼 후 hash 생성. (탐색용)
                node로 부름.

        action 선택
        -----------
        puct 값이 가장 높은 곳을 선택함, 동점이면 랜덤 선택.

            action: 1x3 tuple.
                action = (주체 인덱스, 보드의 x좌표, 보드의 y좌표)

        """
        # 턴 관리
        self.action_count += 1
        # 호출될 때마다 첫턴 기준 교대로 행동주체 바꿈, 최종 action에 붙여줌
        self.user_type = (self.first_turn + self.action_count - 1) % 2

        # state 변환 및 저장
        self.state = state
        self.state_new = self._convert_state(state)
        # 새로운 state 저장
        self.state_memory.appendleft(self.state_new)
        # state를 문자열 -> hash로 변환 (dict의 key로 쓰려고)
        self.node = hash(self.state_new.tostring())
        self.node_memory.appendleft(self.node)

        # Tree를 호출하여 PUCT 값 계산
        self._cal_puct()

        # 점수 확인
        print("* PUCT Score *")
        print(self.puct.round(decimals=2))
        print("")

        # 값이 음수가 나올 수 있어서 빈자리가 아닌 곳은 -9999를 넣어 최댓값 방지
        puct = self.puct.tolist()
        for i, v in enumerate(puct):
            for k, s in enumerate(v):
                if [i, k] not in self.empty_loc.tolist():
                    puct[i][k] = -9999

        # PUCT가 최댓값인 곳 찾기
        self.puct = np.asarray(puct)
        puct_max = np.argwhere(self.puct == self.puct.max()).tolist()

        # 동점 처리
        move_target = puct_max[np.random.choice(len(puct_max))]

        # 최종 action 구성
        # 배열 접붙히기
        action = np.r_[self.user_type, move_target]

        # action 저장 및 step 초기화
        self.action_memory.appendleft(action)
        self._reset_step()
        return tuple(action)

    def _convert_state(self, state):
        """ state변환 메소드: action 주체별 최대 4수까지 history를 저장하여 새로운 state로 구성"""
        if abs(self.user_type - 1) == PLAYER:
            self.my_history.appendleft(state[PLAYER].flatten())
        else:
            self.your_history.appendleft(state[OPPONENT].flatten())
        state_new = np.r_[np.array(self.my_history).flatten(),
                          np.array(self.your_history).flatten(),
                          self.state[2].flatten()]
        return state_new

    def _cal_puct(self):
        """9개의 좌표에 PUCT값을 계산하여 매칭하는 메소드.

        dict{node: edge}인 Tree 구성
        """
        # Tree에서 현재 node를 검색하여 해당 edge의 누적정보 가져오기
        self.edge = self.tree_memory[self.node]

        # 현재 보드의 착수가능 좌표와 개수를 알아낸 후 랜덤일 확률을 계산
        self.board = self.state[PLAYER] + self.state[OPPONENT]
        self.empty_loc = np.argwhere(self.board == 0)
        self.legal_move_n = self.empty_loc.shape[0]
        prob = 1 / self.legal_move_n
        # root node면 확률에 일정부분 노이즈
        if self.action_count == 1:
            self.pr = (1 - self.epsilon) * prob + self.epsilon * \
                np.random.dirichlet(
                    self.alpha * np.ones(self.legal_move_n))
        else:  # 아니면 n분의 1
            self.pr = prob * np.ones(self.legal_move_n)

        # edge의 총 방문횟수 계산
        for i in range(3):
            for j in range(3):
                self.total_visit += self.edge[i][j][N]
        # 방문횟수 출력
        print('visit count: %d\n' % (self.total_visit + 1))

        for i in range(self.legal_move_n):
            self.edge[tuple(self.empty_loc[i])][P] = self.pr[i]

        # Q값 계산 후 배치
        for c in range(3):
            for r in range(3):
                if self.edge[c][r][N] != 0:
                    self.edge[c][r][Q] = self.edge[c][r][W] / \
                        self.edge[c][r][N]
                # 각자리의 PUCT 계산!
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
    start = time.time()
    # 환경 생성
    env = tictactoe_env.TicTacToeEnv()
    # 셀프 플레이 인스턴스 생성
    zero_play = MCTS()
    # 통계용
    result = {1: 0, 0: 0, -1: 0}
    win_mark_O = 0
    # 초기 train data 생성 루프
    for e in range(EPISODE):
        # state 생성
        state = env.reset()
        print('-' * 30, '\nepisode: %d' % (e + 1))
        # 선공 정하고 교대로 하기
        zero_play.first_turn = (PLAYER + e) % 2
        env.player_color = zero_play.first_turn
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

            # 보상을 edge에 백업
            zero_play.backup(reward)
            # 결과 체크
            result[reward] += 1
            # 선공으로 이긴 경우 체크
            if reward == 1:
                if env.player_color == 0:
                    win_mark_O += 1
        # 데이터 저장
        if (e + 1) % SAVE_CYCLE == 0:
            finish = round(float(time.time() - start))
            print('%d episode data saved' % (e + 1))
            np.save('data/state_memory_e{}_c{}a{}.npy'.format(
                (e + 1),
                zero_play.c_puct,
                zero_play.alpha),
                zero_play.state_memory)

            # 에피소드 통계
            statics = ('\nWin: %d  Lose: %d  Draw: %d  Winrate: %0.1f%%  \
WinMarkO: %d' % (result[1], result[-1], result[0],
                 1 / (1 + np.exp(result[-1] / EPISODE) /
                      np.exp(result[1] / EPISODE)) * 100,
                 win_mark_O))
            print('-' * 22, statics)

            slack = slackweb.Slack(
                url="https://hooks.slack.com/services/T8P0E384U/B8PR44F1C/\
4gVy7zhZ9teBUoAFSse8iynn")
            slack.notify(
                text="Finished: {} episode in {}s".format(e + 1, finish))
            slack.notify(text=statics)
