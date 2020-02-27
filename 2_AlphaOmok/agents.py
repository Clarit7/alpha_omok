import sys
import time
import threading

import numpy as np
import torch

import utils


PRINT_MCTS = True
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

"""
인공지능 에이전트, 휴먼 에이전트
"""

class Agent(object):
    def __init__(self, board_size):

        self.policy = np.zeros(board_size**2, 'float')  # 착수위치에 따른 승리확률
        self.visit = np.zeros(board_size**2, 'float')   # 이미 방문한 위치 표시
        self.message = 'Hello'

    def get_policy(self):
        return self.policy

    def get_visit(self):
        return self.visit

    def get_name(self):
        return type(self).__name__

    def get_message(self):
        return self.message

    def get_pv(self, root_id):
        return None, None


class ZeroAgent(Agent):
    def __init__(self, board_size, num_mcts, inplanes, noise=True):
        super(ZeroAgent, self).__init__(board_size)  # agent.Agent() 호출
        self.board_size = board_size  # 보드사이즈 설정
        self.num_mcts = num_mcts      # MCTS 반복 횟수 설정
        self.inplanes = inplanes      # inplanes 설정
        # tictactoe and omok
        self.win_mark = 3 if board_size == 3 else 5  # 틱택토와 오목에서의 승리 기준 설정
        self.alpha = 10 / self.board_size**2  # 알파값 설정
        self.c_puct = 5
        self.noise = noise
        self.root_id = None # 루트 아이디 초기화
        self.model = None
        self.tree = {}
        self.is_real_root = True

    def reset(self):
        self.root_id = None
        self.tree.clear()
        self.is_real_root = True

    # MCTS를 통해 각 위치의 방문횟수를 알아내고 이를 통해 pi를 반환
    # eval_main.get_action()으로부터 호출
    def get_pi(self, root_id, tau):
        # MCTS
        self._init_mcts(root_id)
        self._mcts(self.root_id)

        # 초기화
        visit = np.zeros(self.board_size**2, 'float')
        policy = np.zeros(self.board_size**2, 'float')

        for action_index in self.tree[self.root_id]['child']:
            child_id = self.root_id + (action_index,)
            visit[action_index] = self.tree[child_id]['n']  # MCTS에서 각 자식 노드의 방문 횟수
            policy[action_index] = self.tree[child_id]['p'] # 정책, 승리 가능성이 높은 위치가 높은 값을 가짐

        self.visit = visit
        self.policy = policy

        pi = visit / visit.sum()
        if tau == 0:
            pi, _ = utils.argmax_onehot(pi) # 최대값을 가진 위치 중 하나만 onehot인코딩된다

        return pi

    # MCTS 초기화
    # 해당 클래스의 get_pi()로부터 호출
    def _init_mcts(self, root_id):
        self.root_id = root_id
        if self.root_id not in self.tree:   # 아직 트리에 없는 root_id라면 추가
            self.is_real_root = True    # 시뮬레이션 루트가 아닌 실제 플레이가 이루어진 루트
            # init root node
            self.tree[self.root_id] = {'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.,
                                       'p': 0.}
        # 트리에 이미 있는 root_id
        # p에 랜덤한 값의 노이즈를 추가함
        else:
            self.is_real_root = False   # 시뮬레이션 루트
            if self.noise:  # default = false
                children = self.tree[self.root_id]['child']
                noise_probs = np.random.dirichlet(
                    self.alpha * np.ones(len(children)))

                for i, action_index in enumerate(children):
                    child_id = self.root_id + (action_index,)
                    self.tree[child_id]['p'] = 0.75 * \
                        self.tree[child_id]['p'] + 0.25 * noise_probs[i]

    # MCTS
    # 해당 클래스의 get_pi()로부터 호출
    def _mcts(self, root_id):
        start = time.time() # 시작 시간
        if self.is_real_root:
            # do not count first expansion of the root node
            num_mcts = self.num_mcts + 1
        else:
            num_mcts = self.num_mcts

        for i in range(num_mcts):   # 지정한 횟수만큼 반복

            if PRINT_MCTS:
                sys.stdout.write('simulation: {}\r'.format(i + 1))
                sys.stdout.flush()

            self.message = 'simulation: {}\r'.format(i + 1)

            # selection
            leaf_id, win_index = self._selection(root_id)

            # expansion and evaluation
            value, reward = self._expansion_evaluation(leaf_id, win_index)

            # backup
            self._backup(leaf_id, value, reward)

        finish = time.time() - start    # 종료 시간
        if PRINT_MCTS:
            print("{} simulations end ({:0.0f}s)".format(i + 1, finish))

    # 가장 승산 있는 노드를 고르는 selection
    def _selection(self, root_id):
        node_id = root_id

        # 한 번이라도 시뮬레이션이 이루어진 노드에서만 선택
        while self.tree[node_id]['n'] > 0:
            # node_id로부터 보드의 현 상태를 생성하고 승패를 판단함
            board = utils.get_board(node_id, self.board_size)
            win_index = utils.check_win(board, self.win_mark)

            if win_index != 0:
                return node_id, win_index   # 해당 노드에서 승패가 결정나면 node_id 와 win_index(승패결과) 반환

            qu = {}     # q + u
            ids = []    # key에 child_id와 value에 qu가 입력될 dict
            total_n = 0 # 해당 부모노드에 포함된 자식노드들 시뮬레이션 수의 합

            # 모든 자식노드들의 n값을 더해 total_n을 구함
            for action_idx in self.tree[node_id]['child']:
                edge_id = node_id + (action_idx,)
                n = self.tree[edge_id]['n']
                total_n += n

            # 모든 자식노드들의 q+u 값을 구함
            for i, action_index in enumerate(self.tree[node_id]['child']):
                child_id = node_id + (action_index,)
                n = self.tree[child_id]['n']
                q = self.tree[child_id]['q']
                p = self.tree[child_id]['p']
                u = self.c_puct * p * np.sqrt(total_n) / (n + 1)
                qu[child_id] = q + u

            max_value = max(qu.values())    # qu중 최대값을 구함
            ids = [key for key, value in qu.items() if value == max_value]  # qu최대값에 해당하는 child_id와 value를 dict에 입력
            node_id = ids[np.random.choice(len(ids))]   # 최대값 중 하나에 해당하는 노드를 선택

        # node_id로부터 보드의 현 상태를 생성하고 승패를 판단함
        board = utils.get_board(node_id, self.board_size)
        win_index = utils.check_win(board, self.win_mark)

        return node_id, win_index

    # selection으로부터 선택된 노드에서의 무작위 확장과 승리 가능성 평가
    def _expansion_evaluation(self, leaf_id, win_index):
        # 최근 몇턴간의 one hot 인코딩된 흑돌의 위치와 one hot 인코딩된 백돌의 위치, 그리고 색깔 정보
        leaf_state = utils.get_state_pt(
            leaf_id, self.board_size, self.inplanes)
        self.model.eval()   # 드롭아웃 및 배치 정규화를 평가 모드로 설정
        with torch.no_grad():   # Tensor로 부터의 기록 추적과 메모리 사용 방지
            state_input = torch.tensor([leaf_state]).to(device).float() # 지정한 디바이스에 새로운 Tensor 인스턴스 생성
            policy, value = self.model(state_input) # 모델에 Tensor 적용
            policy = policy.cpu().numpy()[0]    # policy : 승리가능성이 높을수록 높게 책정된다
            value = value.cpu().numpy()[0]      # value : (-1 ~ 1) 마지막 턴 플레이어의 승리 가능성이 높으면 낮은 값을 반환

        if win_index == 0:  # 승패가 결정되지 않은 경우
            # expansion
            actions = utils.legal_actions(leaf_id, self.board_size) # 잎 노드의 보드 상황에서 모든 가능한 착수 위치
            prior_prob = np.zeros(self.board_size**2)   # policy의 정규화 값이 저장될 array

            # re-nomalization
            for action_index in actions:
                prior_prob[action_index] = policy[action_index]

            prior_prob /= prior_prob.sum()

            if self.noise:  # 노이즈 생성
                # root node noise
                if leaf_id == self.root_id:
                    noise_probs = np.random.dirichlet(
                        self.alpha * np.ones(len(actions)))

            # 잎 노드에서 착수 가능한 위치에 해당하는 자식 노드 생성
            for i, action_index in enumerate(actions):
                child_id = leaf_id + (action_index,)

                prior_p = prior_prob[action_index]

                if self.noise:
                    if leaf_id == self.root_id:
                        prior_p = 0.75 * prior_p + 0.25 * noise_probs[i]

                # 트리에 자식 노드 추가
                self.tree[child_id] = {'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.,
                                       'p': prior_p}

                self.tree[leaf_id]['child'].append(action_index)    # 잎 노드의 child 밸류 수정
            # return value
            reward = False
            return value, reward
        else:   # 게임의 승패가 결정됐을 때
            # terminal node
            # return reward
            reward = 1.
            value = False
            return value, reward

    # 역전파
    def _backup(self, leaf_id, value, reward):
        node_id = leaf_id
        count = 0
        while node_id != self.root_id[:-1]: # root 노드에 도달할 때 까지 역전파
            self.tree[node_id]['n'] += 1    # 시뮬레이션 횟수 1회 추가

            if not reward:  # expansion 과정에서 승패가 결정되지 않은 노드라면
                # 자신의 턴이 마지막인 경우엔 낮은 value, 상대의 턴이 마지막인 경우엔 높은 value일수록 w가 큰 값이 됨
                self.tree[node_id]['w'] += (-value) * (-1)**(count)
                count += 1
            else:   # expansion 과정에서 승패가 결정된 노드라면
                self.tree[node_id]['w'] += reward * (-1)**(count)   # w는 내 턴에서 1로 고정, 상대 턴에선 -1로 고정
                count += 1

            # 잎 노드의 q 수정, q = w / n
            self.tree[node_id]['q'] = (self.tree[node_id]['w'] /
                                       self.tree[node_id]['n'])
            parent_id = node_id[:-1]    # 잎 노드를 지우고 부모 노드로 올라감
            node_id = parent_id         # 현재노드를 부모 노드로 수정

    # 지금까지 착수 순서를 나타내는 root_id보다 짧은 tree를 전부 삭제함
    # eval_main.main()으로부터 호출
    def del_parents(self, root_id):
        max_len = 0
        if self.tree:
            for key in list(self.tree.keys()):  # tree : 착수 순서를 나타내는 key와 해당 수의 child, n, w, q, p
                if len(key) > max_len:
                    max_len = len(key)
                if len(key) < len(root_id): # root_id보다 짧은 tree를 전부 삭제
                    del self.tree[key]
        print('tree size:', len(self.tree))
        print('tree depth:', 0 if max_len <= 0 else max_len - 1)

    # policy(각 착수위치의 승리 가능성)와 value반환
    def get_pv(self, root_id):
        # state
        # s[t = (0, 1, ... , inplanes - 2)] = (inplanes - 2 - t)턴 전에 착수한 플레이어의 모든 돌 위치가 one hot 인코딩된 array
        # s[inplanes-1] = color feature (마지막 착수가 흑이었으면 all 0, 백이면 all 1)
        state = utils.get_state_pt(root_id, self.board_size, self.inplanes)
        self.model.eval()      # 드롭아웃 및 배치 정규화를 평가 모드로 설정
        with torch.no_grad():      # Tensor로 부터의 기록 추적과 메모리 사용 방지
            state_input = torch.tensor([state]).to(device).float()  # 지정한 디바이스에 새로운 Tensor 인스턴스 생성
            policy, value = self.model(state_input) # 모델에 Tensor 적용
            p = policy.data.cpu().numpy()[0]    # policy : 각 착수 위치의 승리 가능성이 높을수록 높게 책정된다
            v = value.data.cpu().numpy()[0]     # value : (-1 ~ 1) 마지막 턴 플레이어의 승리 가능성이 높으면 낮은 값을 반환

        return p, v


class PUCTAgent(Agent):                                                        
    def __init__(self, board_size, num_mcts):
        super(PUCTAgent, self).__init__(board_size)
        self.board_size = board_size
        self.num_mcts = num_mcts
        # tictactoe and omok
        self.win_mark = 3 if board_size == 3 else 5
        self.c_puct = 5
        self.root_id = None
        self.board = None
        self.turn = None
        self.tree = {}
        self.is_real_root = True

    def reset(self):
        self.is_real_root = True
        self.root_id = None
        self.turn = None
        self.tree.clear()

    def get_pi(self, root_id, board, turn, tau):
        self._init_mcts(root_id, board, turn)
        self._mcts(self.root_id)

        visit = np.zeros(self.board_size**2, 'float')
        pi = np.zeros(self.board_size**2, 'float')

        for action_index in self.tree[self.root_id]['child']:
            child_id = self.root_id + (action_index,)
            visit[action_index] = self.tree[child_id]['n']

        max_idx = np.argwhere(visit == visit.max())
        pi[max_idx[np.random.choice(len(max_idx))]] = 1

        return pi

    def _init_mcts(self, root_id, board, turn):
        self.root_id = root_id
        self.board = board
        self.turn = turn
        self.tree[self.root_id] = {'board': self.board,
                                   'player': self.turn,
                                   'parent': None,
                                   'child': [],
                                   'n': 0.,
                                   'w': 0.,
                                   'q': 0.,
                                   'p': 0.}

    def _mcts(self, root_id):
        start = time.time()

        for i in range(self.num_mcts + 1):
            sys.stdout.write('simulation: {}\r'.format(i + 1))
            sys.stdout.flush()
            leaf_id, win_index = self._selection(root_id)
            reward = self._expansion_simulation(leaf_id, win_index)
            self._backup(leaf_id, reward)

        finish = time.time() - start
        print("{} simulations end ({:0.0f}s)".format(self.num_mcts, finish))

    def _selection(self, root_id):
        node_id = root_id

        while self.tree[node_id]['n'] > 0:
            win_index = utils.check_win(
                self.tree[node_id]['board'], self.win_mark)

            if win_index != 0:
                return node_id, win_index

            qu = {}
            ids = []
            total_n = 0

            for action_idx in self.tree[node_id]['child']:
                edge_id = node_id + (action_idx,)
                n = self.tree[edge_id]['n']
                total_n += n

            for action_index in self.tree[node_id]['child']:
                child_id = node_id + (action_index,)
                n = self.tree[child_id]['n']
                q = self.tree[child_id]['q']
                p = self.tree[child_id]['p']
                u = self.c_puct * p * np.sqrt(total_n) / (n + 1)
                qu[child_id] = q + u

            max_value = max(qu.values())
            ids = [key for key, value in qu.items()
                   if value == max_value]
            node_id = ids[np.random.choice(len(ids))]

        win_index = utils.check_win(self.tree[node_id]['board'],
                                    self.win_mark)
        return node_id, win_index

    def _expansion_simulation(self, leaf_id, win_index):
        leaf_board = self.tree[leaf_id]['board']
        current_player = self.tree[leaf_id]['player']

        if win_index == 0:
            actions = utils.valid_actions(leaf_board)
            prior_prob = 1 / len(actions)

            for i, action in enumerate(actions):
                action_index = action[1]
                child_id = leaf_id + (action_index,)
                child_board = utils.get_board(child_id, self.board_size)
                next_turn = utils.get_turn(child_id)

                self.tree[child_id] = {'board': child_board,
                                       'player': next_turn,
                                       'parent': leaf_id,
                                       'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.,
                                       'p': prior_prob}

                self.tree[leaf_id]['child'].append(action_index)

            if self.tree[leaf_id]['parent']:
                board_sim = leaf_board.copy()
                turn_sim = current_player

                while True:
                    actions_sim = utils.valid_actions(board_sim)
                    action_sim = actions_sim[
                        np.random.choice(len(actions_sim))]
                    coord_sim = action_sim[0]

                    if turn_sim == 0:
                        board_sim[coord_sim] = 1
                    else:
                        board_sim[coord_sim] = -1

                    win_idx_sim = utils.check_win(board_sim, self.win_mark)

                    if win_idx_sim == 0:
                        turn_sim = abs(turn_sim - 1)

                    else:
                        reward = utils.get_reward(win_idx_sim, leaf_id)
                        return reward
            else:
                # root node don't simulation
                reward = 0.
                return reward
        else:
            # terminal node don't expansion
            reward = 1.
            return reward

    def _backup(self, leaf_id, reward):
        node_id = leaf_id
        count = 0

        while node_id is not None:
            self.tree[node_id]['n'] += 1
            self.tree[node_id]['w'] += reward * (-1)**(count)
            self.tree[node_id]['q'] = (self.tree[node_id]['w'] /
                                       self.tree[node_id]['n'])
            parent_id = self.tree[node_id]['parent']
            node_id = parent_id
            count += 1

    def del_parents(self, root_id):
        max_len = 0
        if self.tree:
            for key in list(self.tree.keys()):
                if len(key) > max_len:
                    max_len = len(key)
                if len(key) < len(root_id):
                    del self.tree[key]
        print('tree size:', len(self.tree))
        print('tree depth:', 0 if max_len <= 0 else max_len - 1)


class UCTAgent(Agent):
    def __init__(self, board_size, num_mcts):
        super(UCTAgent, self).__init__(board_size)
        self.board_size = board_size
        self.num_mcts = num_mcts
        # tictactoe and omok
        self.win_mark = 3 if board_size == 3 else 5
        self.root_id = None
        self.board = None
        self.turn = None
        self.tree = {}
        self.is_real_root = True

    def reset(self):
        self.is_real_root = True
        self.root_id = None
        self.board = None
        self.turn = None
        self.tree.clear()

    def get_pi(self, root_id, board, turn, tau):
        self._init_mcts(root_id, board, turn)
        self._mcts(self.root_id)

        root_node = self.tree[self.root_id]
        q = np.ones(self.board_size**2, 'float') * -np.inf
        pi = np.zeros(self.board_size**2, 'float')

        for action_index in root_node['child']:
            child_id = self.root_id + (action_index,)
            q[action_index] = self.tree[child_id]['q']

        max_idx = np.argwhere(q == q.max())
        pi[max_idx[np.random.choice(len(max_idx))]] = 1
        return pi

    def _init_mcts(self, root_id, board, turn):
        self.root_id = root_id
        self.board = board
        self.turn = turn
        self.is_real_root = True
        # init root node
        self.tree[self.root_id] = {'board': self.board,
                                   'player': self.turn,
                                   'parent': None,
                                   'child': [],
                                   'n': 0.,
                                   'w': 0.,
                                   'q': 0.}

    def _mcts(self, root_id):
        start = time.time()

        if self.is_real_root:
            num_mcts = self.num_mcts + 1
        else:
            num_mcts = self.num_mcts

        for i in range(num_mcts):

            mcts_count = i

            sys.stdout.write('simulation: {}\r'.format(i + 1))
            sys.stdout.flush()
            leaf_id, win_index = self._selection(root_id)
            reward = self._expansion_simulation(leaf_id, win_index)
            self._backup(leaf_id, reward)

        finish = time.time() - start
        print("{} simulations end ({:0.0f}s)".format(self.num_mcts, finish))

    def _selection(self, root_id):
        node_id = root_id

        while self.tree[node_id]['n'] > 0:
            win_index = utils.check_win(
                self.tree[node_id]['board'], self.win_mark)

            if win_index != 0:
                return node_id, win_index

            qu = {}
            ids = []
            total_n = 0

            for action_idx in self.tree[node_id]['child']:
                edge_id = node_id + (action_idx,)
                n = self.tree[edge_id]['n']
                total_n += n

            for action_index in self.tree[node_id]['child']:
                child_id = node_id + (action_index,)
                n = self.tree[child_id]['n']
                q = self.tree[child_id]['q']

                if n == 0:
                    u = np.inf
                else:
                    u = np.sqrt(2 * np.log(total_n) / n)

                qu[child_id] = q + u

            max_value = max(qu.values())
            ids = [key for key, value in qu.items()
                   if value == max_value]
            node_id = ids[np.random.choice(len(ids))]

        win_index = utils.check_win(self.tree[node_id]['board'],
                                    self.win_mark)
        return node_id, win_index

    def _expansion_simulation(self, leaf_id, win_index):
        leaf_board = self.tree[leaf_id]['board']
        current_player = self.tree[leaf_id]['player']

        if win_index == 0:
            # expansion
            actions = utils.valid_actions(leaf_board)

            for action in actions:
                action_index = action[1]
                child_id = leaf_id + (action_index,)
                child_board = utils.get_board(child_id, self.board_size)
                next_turn = utils.get_turn(child_id)

                self.tree[child_id] = {'board': child_board,
                                       'player': next_turn,
                                       'parent': leaf_id,
                                       'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.}

                self.tree[leaf_id]['child'].append(action_index)

            if self.tree[leaf_id]['parent']:
                # simulation
                board_sim = leaf_board.copy()
                turn_sim = current_player

                while True:
                    actions_sim = utils.valid_actions(board_sim)
                    action_sim = actions_sim[
                        np.random.choice(len(actions_sim))]
                    coord_sim = action_sim[0]

                    if turn_sim == 0:
                        board_sim[coord_sim] = 1
                    else:
                        board_sim[coord_sim] = -1

                    win_idx_sim = utils.check_win(board_sim, self.win_mark)

                    if win_idx_sim == 0:
                        turn_sim = abs(turn_sim - 1)

                    else:
                        reward = utils.get_reward(win_idx_sim, leaf_id)
                        return reward
            else:
                # root node don't simulation
                reward = 0.
                return reward
        else:
            # terminal node don't expansion
            reward = 1.
            return reward

    def _backup(self, leaf_id, reward):

        node_id = leaf_id
        count = 0

        while node_id is not None:
            self.tree[node_id]['n'] += 1
            self.tree[node_id]['w'] += reward * (-1)**(count)
            self.tree[node_id]['q'] = (self.tree[node_id]['w'] /
                                       self.tree[node_id]['n'])
            parent_id = self.tree[node_id]['parent']
            node_id = parent_id
            count += 1

    def del_parents(self, root_id):
        max_len = 0
        if self.tree:
            for key in list(self.tree.keys()):
                if len(key) > max_len:
                    max_len = len(key)
                if len(key) < len(root_id):
                    del self.tree[key]
        print('tree size:', len(self.tree))
        print('tree depth:', 0 if max_len <= 0 else max_len - 1)


class RandomAgent(Agent):
    def __init__(self, board_size):
        super(RandomAgent, self).__init__(board_size)
        self.board_size = board_size

    def get_pi(self, root_id, board, turn, tau):
        self.root_id = root_id
        action = utils.valid_actions(board)
        prob = 1 / len(action)
        pi = np.zeros(self.board_size**2, 'float')

        for loc, idx in action:
            pi[idx] = prob

        return pi

    def reset(self):
        self.root_id = None

    def del_parents(self, root_id):
        return


# 플레이어가 human일 때 에이전트 설정
# eval_main.Evaluator().set_agents() 로부터 호출
class HumanAgent(Agent):
    COLUMN = {"a": 0, "b": 1, "c": 2,
              "d": 3, "e": 4, "f": 5,
              "g": 6, "h": 7, "i": 8,
              "j": 9, "k": 10, "l": 11,
              "m": 12, "n": 13, "o": 14}

    def __init__(self, board_size, env):
        super(HumanAgent, self).__init__(board_size)
        self.board_size = board_size
        self._init_board_label()
        self.root_id = (0,)
        self.env = env

    # 플레이어의 착수위치를 onehot인코딩해 반환한다
    # eval_main.get_action()으로부터 호출
    def get_pi(self, root_id, board, turn, tau):
        self.root_id = root_id

        # 플레이어가 빈 착수위치를 클릭할때까지 대기하고, 유효한 위치를 클릭하면 pi를 반환한다
        while True:
            action = 0

            _, check_valid_pos, _, _, action_index = self.env.step(
                action)

            if check_valid_pos is True:
                pi = np.zeros(self.board_size**2, 'float')
                pi[action_index] = 1
                break

        return pi

    def _init_board_label(self):
        self.last_label = str(self.board_size)

        for k, v in self.COLUMN.items():
            if v == self.board_size - 1:
                self.last_label += k
                break

    def input_action(self, last_label):
        action_coord = input('1a ~ {}: '.format(last_label)).rstrip().lower()
        row = int(action_coord[0]) - 1
        col = self.COLUMN[action_coord[1]]
        action_index = row * self.board_size + col
        return action_index

    def reset(self):
        self.root_id = None

    def del_parents(self, root_id):
        return


class WebAgent(Agent):

    def __init__(self, board_size):
        super(WebAgent, self).__init__(board_size)
        self.board_size = board_size
        self.root_id = (0,)
        self.wait_action_idx = -1
        self.cv = threading.Condition()

    def get_pi(self, root_id, board, turn, tau):
        self.root_id = root_id

        self.cv.acquire()
        while self.wait_action_idx == -1:
            self.cv.wait()

        action_index = self.wait_action_idx
        self.wait_action_idx = -1

        self.cv.release()

        pi = np.zeros(self.board_size**2, 'float')
        pi[action_index] = 1

        return pi

    def put_action(self, action_idx):

        if action_idx < 0 and action_idx >= self.board_size**2:
            return

        self.cv.acquire()
        self.wait_action_idx = action_idx
        self.cv.notifyAll()
        self.cv.release()

    def reset(self):
        self.root_id = None

    def del_parents(self, root_id):
        return
