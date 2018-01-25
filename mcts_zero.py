# -*- coding: utf-8 -*-
from tictactoe_env import TicTacToeEnv
import numpy as np
from collections import deque, defaultdict


PLAYER = 0
OPPONENT = 1
MARK_O = 2
N, W, Q, P = 0, 1, 2, 3
EPISODE = 2400


# 몬테카를로 트리 탐색 클래스 (최초 train 데이터 생성 용)
# edge는 현재 state에서 착수 가능한 모든 action
# edge 구성: (3*3*4)array: 9개 좌표에 4개의 정보 매칭
# 4개의 정보: (N, W, Q, P) N: edge 방문횟수, W: 보상누적값, Q: 보상평균(W/N), P: edge 선택확률
# edge[좌표행][좌표열][번호]로 접근
class MCTS(object):
    def __init__(self):
        # memories
        self.state_memory = deque(maxlen=9 * EPISODE)
        self.node_memory = deque(maxlen=9 * EPISODE)
        self.edge_memory = deque(maxlen=9 * EPISODE)

        # reset_step member
        self.tree_memory = None
        self.pr = None
        self.puct = None
        self.edge = None
        self.legal_move_n = None
        self.empty_loc = None
        self.total_visit = None
        self.first_turn = None

        # reset_episode member
        self.action_memory = None
        self.action_count = None
        self.board = None
        self.state = None

        # hyperparameter
        self.c_puct = 1
        self.epsilon = 0.25
        self.alpha = 0.5
        self.expand_count = 100

        # member 초기화
        self._reset_step()
        self._reset_episode()

    def _reset_step(self):
        self.edge = np.zeros((3, 3, 4))
        self.puct = np.zeros((3, 3))
        self.total_visit = 0
        self.legal_move_n = 0
        self.empty_loc = None
        self.state_hash = None
        self.pr = 0
        self.tree_memory = defaultdict(lambda: 0)

    def _reset_episode(self):
        self.action_memory = deque(maxlen=9)
        self.action_count = -1
        self.board = None
        self.state = None

    def select_action(self, state):
        self.action_count += 1
        self.state = state
        # save state (flatten 안하고 넣으면 값이 바뀜 버그인듯?)
        self.state_memory.appendleft(state.flatten())
        # state를 문자열 -> hash로 변환 (dict의 key로 쓰려고)
        self.state_hash = hash(self.state.tostring())
        # 변환한 state를 node로 부르자. 저장!
        self.node_memory.appendleft(self.state_hash)
        # 호출될 때마다 첫턴 기준 교대로 행동주체 바꿈, 최종 action에 붙여줌
        user_type = (self.first_turn + self.action_count) % 2
        self.init_edge()
        self._cal_puct()
        print("* PUCT Score *")
        print(self.puct.round(decimals=2))  # 점수 확인용
        # 빈자리가 아닌 곳은 -9999로 최댓값 방지
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
        # 두 배열을 붙여서 최종 action 구성
        action = np.r_[user_type, move_target]
        self.action_memory.appendleft(action)
        self._reset_step()
        return action

    def init_edge(self, pr=0):
        '''들어온 상태에서 가능한 action 자리의 엣지를 초기화 (P값 배치)
           빈자리를 검색하여 규칙위반 방지 및 랜덤 확률 생성
           root node, expand node 확인 후 노이즈 줌 (e-greedy)
        '''
        # 들어 온 사전확률이 없으면
        if pr == 0:
            # 빈 자리의 좌표와 개수를 저장하고 이를 이용해 동일 확률을 계산
            self.board = self.state[PLAYER] + self.state[OPPONENT]
            self.empty_loc = np.argwhere(self.board == 0)
            self.legal_move_n = self.empty_loc.shape[0]
            prob = 1 / self.legal_move_n
            count = self.node_memory.count(hash(self.state.tostring()))
            # root node or expand node 이면
            if self.action_count == 0 or count >= self.expand_count:
                self.pr = (1 - self.epsilon) * prob + self.epsilon * \
                    np.random.dirichlet(
                        self.alpha * np.ones(self.legal_move_n))
            else:  # 아니면 랜덤 확률로 n분의 1
                self.pr = prob * np.ones(self.legal_move_n)

            # 빈자리의 엣지에 넣기
            for i in range(self.legal_move_n):
                self.edge[self.empty_loc[i][0]
                          ][self.empty_loc[i][1]][P] = self.pr[i]
        else:  # 사전확률 값이 들어오면 그걸로 넣기
            self.pr = pr
            for i in range(3):
                for k in range(3):
                    self.edge[i][k][P] = self.pr[i][k]
        # edge 메모리에 저장
        self.edge_memory.appendleft(self.edge)

    def _cal_puct(self):
        '''9개의 좌표에 PUCT값을 계산하여 매칭'''
        # 지금까지의 액션을 반영한 트리 구성 하기. dict{node: edge}
        memory = list(zip(self.node_memory, self.edge_memory))
        # 지금까지의 동일한 state에 대한 edge의 N,W 누적
        # Q,P는 덧셈이라 손상되므로 보정함
        for v in memory:
            key = v[0]
            value = v[1]
            self.tree_memory[key] += value
        if self.node_memory[0] in self.tree_memory:
            edge = self.tree_memory[self.node_memory[0]]
            for i in range(3):
                for k in range(3):
                    self.total_visit += edge[i][k][N]
            for c in range(3):
                for r in range(3):
                    if edge[c][r][N] != 0:
                        # Q 보정
                        edge[c][r][Q] = edge[c][r][W] / edge[c][r][N]
                    # P 보정
                    edge[c][r][P] = self.edge[c][r][P]
                    # PUCT 계산!
                    self.puct[c][r] = edge[c][r][Q] + \
                        self.c_puct * edge[c][r][P] * \
                        np.sqrt(self.total_visit) / \
                        (1 + edge[c][r][N])
            # 보정한 edge를 최종 트리에 업데이트
            self.tree_memory[self.node_memory[0]] = edge

    def backup(self, reward, info):
        '''에피소드가 끝나면 지나 온 edge의 N과 W를 업데이트 함'''
        steps = info['steps']
        for i in range(steps):
            if self.action_memory[i][0] == PLAYER:
                self.edge_memory[i][self.action_memory[i][1]
                                    ][self.action_memory[i][2]
                                      ][W] += reward
            else:
                self.edge_memory[i][self.action_memory[i][1]
                                    ][self.action_memory[i][2]
                                      ][W] -= reward
            self.edge_memory[i][self.action_memory[i][1]
                                ][self.action_memory[i][2]][N] += 1
        self._reset_episode()


if __name__ == "__main__":
    # 환경 생성 및 시드 설정
    env = TicTacToeEnv()
    np.random.seed(1)
    # 셀프 플레이 인스턴스 생성
    zero_play = MCTS()
    # 통계용
    result = {1: 0, 0: 0, -1: 0}
    play_mark_O = 0
    win_mark_O = 0
    # train data 생성
    for e in range(EPISODE):
        state = env.reset()
        print('-' * 22, '\nepisode: %d' % (e + 1))
        # 선공 정하고 교대로 하기
        zero_play.first_turn = ((OPPONENT + e) % 2)
        # 첫턴인 경우 기록
        if zero_play.first_turn == PLAYER:
            play_mark_O += 1
        done = False
        while not done:
            # 보드 상황 출력: 내 착수:1, 상대 착수:2
            print("---- BOARD ----")
            print(state[PLAYER] + state[OPPONENT] * 2)
            # action 선택하기
            action = zero_play.select_action(state)
            # action 진행
            state, reward, done, info = env.step(action)
        if done:
            # 승부난 보드 보기
            print("- FINAL BOARD -")
            print(state[PLAYER] + state[OPPONENT] * 2)
            # 보상을 edge에 백업
            zero_play.backup(reward, info)
            # 결과 dict에 기록
            result[reward] += 1
            if reward == 1:
                if env.mark_O == PLAYER:
                    win_mark_O += 1
    # 에피소드 통계
    print('-' * 22, '\nWin: %d   Lose: %d   Draw: %d   Winrate: %0.1f%%   \
PlayMarkO: %d   WinMarkO: %d' %
          (result[1], result[-1], result[0], result[1] / EPISODE * 100,
           play_mark_O, win_mark_O))
    # data save
    print("data saved")
    np.save('data/state_memory.npy', zero_play.state_memory)
    np.save('data/edge_memory.npy', zero_play.edge_memory)