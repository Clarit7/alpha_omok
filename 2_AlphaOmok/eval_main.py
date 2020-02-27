import numpy as np
import torch

import agents
import model
import utils

# env_small: 9x9, env_regular: 15x15
from env import env_small as game

# Web API
import logging
import threading
import flask
from webapi import web_api
from webapi import game_info
from webapi import player_agent_info
from webapi import enemy_agent_info
from info.agent_info import AgentInfo
from info.game_info import GameInfo

BOARD_SIZE = game.Return_BoardParams()[0]

# 딥러닝 과정에서 연결된 블록 수
N_BLOCKS_PLAYER = 10
N_BLOCKS_ENEMY = 10

IN_PLANES_PLAYER = 5  # history(착수 위치 기록 순서) * 2 + 1
IN_PLANES_ENEMY = 5

# 각 블록에서의 출력 채널의 수
OUT_PLANES_PLAYER = 128
OUT_PLANES_ENEMY = 128

# MCTS의 반복횟수
N_MCTS_PLAYER = 80
N_MCTS_ENEMY = 80
N_MCTS_MONITOR = 40

N_MATCH = 2  # 매치 수

use_cuda = torch.cuda.is_available()    # GPU를 사용 가능한가?
device = torch.device('cuda' if use_cuda else 'cpu')    # GPU사용 가능시 GPU사용, 아니면 CPU사용

# ==================== input model path ================= #
#       'human': human play       'puct': PUCB MCTS       #
#       'uct': UCB MCTS           'random': random        #
#       'web': web play                                   #
# ======================================================= #
# example)

# 플레이어, 적 플레이어, 웹에서 알파제로와의 플레이 비교를 위한 모니터 모델 설정
player_model_path = 'human'
enemy_model_path = './data/180927_9400_297233_step_model.pickle'
monitor_model_path = './data/180927_9400_297233_step_model.pickle'

# 플레이어 모델 설정과 동작관련 명령을 처리
class Evaluator(object):
    # 초기화
    def __init__(self):
        self.player = None
        self.enemy = None
        self.monitor = None
        pass

    # 에이전트 설정
    # main()으로부터 가장 먼저 호출됨
    def set_agents(self, model_path_a, model_path_b, model_path_m):

        # 플레이어 중 human이 있으면 pygame창에서 게임 실행, 아니면 텍스트만 출력
        if model_path_a == 'human' or model_path_b == 'human':
            game_mode = 'pygame'
        else:
            game_mode = 'text'

        # env파일의 gamemode 설정
        self.env = game.GameState(game_mode)

        # 플레이어의 모델 설정 (human)
        if model_path_a == 'random':
            print('load player model:', model_path_a)
            self.player = agents.RandomAgent(BOARD_SIZE)
        elif model_path_a == 'puct':
            print('load player model:', model_path_a)
            self.player = agents.PUCTAgent(BOARD_SIZE, N_MCTS_PLAYER)
        elif model_path_a == 'uct':
            print('load player model:', model_path_a)
            self.player = agents.UCTAgent(BOARD_SIZE, N_MCTS_PLAYER)
        elif model_path_a == 'human':
            print('load player model:', model_path_a)
            self.player = agents.HumanAgent(BOARD_SIZE, self.env)
        elif model_path_a == 'web':
            print('load player model:', model_path_a)
            self.player = agents.WebAgent(BOARD_SIZE)
        else:
            print('load player model:', model_path_a)
            self.player = agents.ZeroAgent(BOARD_SIZE,
                                           N_MCTS_PLAYER,
                                           IN_PLANES_PLAYER,
                                           noise=False)
            self.player.model = model.PVNet(N_BLOCKS_PLAYER,
                                            IN_PLANES_PLAYER,
                                            OUT_PLANES_PLAYER,
                                            BOARD_SIZE).to(device)
            state_a = self.player.model.state_dict()
            my_state_a = torch.load(
                model_path_a, map_location='cuda:0' if use_cuda else 'cpu')
            for k, v in my_state_a.items():
                if k in state_a:
                    state_a[k] = v
            self.player.model.load_state_dict(state_a)

        # 적 플레이어의 모델 설정 (
        if model_path_b == 'random':
            print('load enemy model:', model_path_b)
            self.enemy = agents.RandomAgent(BOARD_SIZE)
        elif model_path_b == 'puct':
            print('load enemy model:', model_path_b)
            self.enemy = agents.PUCTAgent(BOARD_SIZE, N_MCTS_ENEMY)
        elif model_path_b == 'uct':
            print('load enemy model:', model_path_b)
            self.enemy = agents.UCTAgent(BOARD_SIZE, N_MCTS_ENEMY)
        elif model_path_b == 'human':
            print('load enemy model:', model_path_b)
            self.enemy = agents.HumanAgent(BOARD_SIZE, self.env)
        elif model_path_b == 'web':
            print('load enemy model:', model_path_b)
            self.enemy = agents.WebAgent(BOARD_SIZE)
        else:   # 이미 만들어진 데이터를 사용할땐 이 부분이 실행됨
            print('load enemy model:', model_path_b)
            # 적 에이전트 설정
            self.enemy = agents.ZeroAgent(BOARD_SIZE,
                                          N_MCTS_ENEMY,
                                          IN_PLANES_ENEMY,
                                          noise=False)
            # 적 신경망 모델 설정 및 device(GPU)로 불러와 agents.ZeroAgent().model에 저장
            self.enemy.model = model.PVNet(N_BLOCKS_ENEMY,
                                           IN_PLANES_ENEMY,
                                           OUT_PLANES_ENEMY,
                                           BOARD_SIZE).to(device)
            state_b = self.enemy.model.state_dict() # dict형식의 신경망 파라미터의 텐서
            my_state_b = torch.load(
                model_path_b, map_location='cuda:0' if use_cuda else 'cpu') # 저장한 파라미터 파일을 불러옴
            # state_b에는 키 값으로 여러 레이어의 weight, bias 등과 그에 해당하는 value들이 저장됨
            for k, v in my_state_b.items():
                if k in state_b:
                    state_b[k] = v
            self.enemy.model.load_state_dict(state_b)   # 딥러닝 모델에 파라미터 설정

        # monitor agent 위와 동일
        self.monitor = agents.ZeroAgent(BOARD_SIZE,
                                        N_MCTS_MONITOR,
                                        IN_PLANES_ENEMY,
                                        noise=False)
        self.monitor.model = model.PVNet(N_BLOCKS_ENEMY,
                                         IN_PLANES_ENEMY,
                                         OUT_PLANES_ENEMY,
                                         BOARD_SIZE).to(device)
        state_b = self.monitor.model.state_dict()
        my_state_b = torch.load(
            model_path_m, map_location='cuda:0' if use_cuda else 'cpu')
        for k, v in my_state_b.items():
            if k in state_b:
                state_b[k] = v
        self.monitor.model.load_state_dict(state_b)

    # 착수위치(boardsize**2 크기의 array)와 착수위치의 index 반환
    def get_action(self, root_id, board, turn, enemy_turn):
        if turn != enemy_turn:
            if isinstance(self.player, agents.ZeroAgent):   # isinstance() : 내장함수, 첫번째 파라미터의 객체가 두번째 파라미터의 클래스에 해당하는지 확인한다
                pi = self.player.get_pi(root_id, tau=0)
            else:
                pi = self.player.get_pi(root_id, board, turn, tau=0)

                # for monitor
                self.monitor.get_pi(root_id, tau=0)
        else:
            if isinstance(self.enemy, agents.ZeroAgent):
                pi = self.enemy.get_pi(root_id, tau=0)
            else:
                pi = self.enemy.get_pi(root_id, board, turn, tau=0)

        action, action_index = utils.argmax_onehot(pi)

        return action, action_index

    # env.env_small.GameState() 반환
    def return_env(self):
        return self.env

    def reset(self):
        self.player.reset()
        self.enemy.reset()

    def put_action(self, action_idx, turn, enemy_turn):

        print(self.player)

        if turn != enemy_turn:
            if type(self.player) is agents.WebAgent:
                self.player.put_action(action_idx)
        else:
            if type(self.enemy) is agents.WebAgent:
                self.enemy.put_action(action_idx)


# 게임 결과에 따라 플레이어와 적의 레이팅 조정
def elo(player_elo, enemy_elo, p_winscore, e_winscore):
    elo_diff = enemy_elo - player_elo
    ex_pw = 1 / (1 + 10**(elo_diff / 400))
    ex_ew = 1 / (1 + 10**(-elo_diff / 400))
    player_elo += 32 * (p_winscore - ex_pw)
    enemy_elo += 32 * (e_winscore - ex_ew)

    return player_elo, enemy_elo


evaluator = Evaluator()


def main():
    # 에이전트 설정
    evaluator.set_agents(
        player_model_path, enemy_model_path, monitor_model_path)

    # 웹 서버에 각 agent들의 정보 전달
    player_agent_info.agent = evaluator.player
    enemy_agent_info.agent = evaluator.enemy

    env = evaluator.return_env()    # env.env_small.GameState()

    result = {'Player': 0, 'Enemy': 0, 'Draw': 0}   # 승, 패, 무승부
    turn = 0        # 플레이어 턴
    enemy_turn = 1  # 적 턴
    player_elo = 1500   # 플레이어 레이팅
    enemy_elo = 1500    # 적 레이팅

    # 웹 서버에 적 턴 변수와 게임 상태 전달
    game_info.enemy_turn = enemy_turn
    game_info.game_status = 0

    # 플레이어와 적의 레이팅 출력
    print('Player ELO: {:.0f}, Enemy ELO: {:.0f}'.format(
        player_elo, enemy_elo))

    # N_MATCH 번의 매치 실행
    for i in range(N_MATCH):
        board = np.zeros([BOARD_SIZE, BOARD_SIZE])  # 가로, 세로 BOARD_SIZE크기인 2차원 array형태의 판 생성
        root_id = (0,)          # 지금까지의 착수 위치들이 기록됨
        win_index = 0           # 승패 결과 (0: 플레이 중, 1: 흑 승, 2: 백 승, 3: 무승부)
        action_index = None     # 현재 턴의 착수 위치

        # 웹 서버에 게임판의 정보 전달
        game_info.game_board = board

        # 한 게임마다 선공을 바꿈
        if i % 2 == 0:
            print('Player Color: Black')
        else:
            print('Player Color: White')
        # 0:Running 1:Player Win, 2: Enemy Win 3: Draw
        game_info.game_status = 0

        # 승패가 결정날때 까지 실행되는 게임의 메인 루프
        while win_index == 0:
            utils.render_str(board, BOARD_SIZE, action_index)   # 게임판을 콘솔창에 텍스트 형식으로 출력

            # agents.ZeroAgent().get_pv() 호출
            # policy(각 착수위치의 승리 가능성)와 value(이번 턴의 플레이어의 승리 가능성이 높으면 높은 값)를 받음
            p, v = evaluator.monitor.get_pv(root_id)

            # 착수위치를 입력받음
            # action : (boradsize**2)크기의 1차원 array에 착수위치가 입력됨
            # action_index : 착수위치의 index
            action, action_index = evaluator.get_action(root_id,
                                                        board,
                                                        turn,
                                                        enemy_turn)

            if turn != enemy_turn:
                # player turn
                root_id = evaluator.player.root_id + (action_index,)
            else:
                # enemy turn
                root_id = evaluator.enemy.root_id + (action_index,)

            board, check_valid_pos, win_index, turn, _ = env.step(action)

            game_info.game_board = board
            game_info.action_index = int(action_index)
            game_info.win_index = win_index
            game_info.curr_turn = turn  # 0 black 1 white

            move = np.count_nonzero(board)

            if turn == enemy_turn:

                if isinstance(evaluator.player, agents.HumanAgent) or \
                        isinstance(evaluator.player, agents.WebAgent):
                    player_agent_info.visit = evaluator.monitor.get_visit()
                    player_agent_info.p = evaluator.monitor.get_policy()
                else:
                    player_agent_info.visit = evaluator.player.get_visit()
                    player_agent_info.p = evaluator.player.get_policy()

                player_agent_info.add_value(move, v)
                evaluator.enemy.del_parents(root_id)

            else:
                enemy_agent_info.visit = evaluator.enemy.get_visit()
                enemy_agent_info.p = evaluator.enemy.get_policy()
                enemy_agent_info.add_value(move, v)
                evaluator.player.del_parents(root_id)

            if win_index != 0:
                player_agent_info.clear_values()
                enemy_agent_info.clear_values()
                # 0:Running 1:Player Win, 2: Enemy Win 3: Draw
                game_info.game_status = win_index

                if turn == enemy_turn:
                    if win_index == 3:
                        result['Draw'] += 1
                        print('\nDraw!')
                        player_elo, enemy_elo = elo(
                            player_elo, enemy_elo, 0.5, 0.5)
                    else:
                        result['Player'] += 1
                        print('\nPlayer Win!')
                        player_elo, enemy_elo = elo(
                            player_elo, enemy_elo, 1, 0)
                else:
                    if win_index == 3:
                        result['Draw'] += 1
                        print('\nDraw!')
                        player_elo, enemy_elo = elo(
                            player_elo, enemy_elo, 0.5, 0.5)
                    else:
                        result['Enemy'] += 1
                        print('\nEnemy Win!')
                        player_elo, enemy_elo = elo(
                            player_elo, enemy_elo, 0, 1)

                utils.render_str(board, BOARD_SIZE, action_index)
                # Change turn
                enemy_turn = abs(enemy_turn - 1)
                turn = 0

                game_info.enemy_turn = enemy_turn
                game_info.curr_turn = turn

                pw, ew, dr = result['Player'], result['Enemy'], result['Draw']
                winrate = (pw + 0.5 * dr) / (pw + ew + dr) * 100
                print('')
                print('=' * 20, " {}  Game End  ".format(i + 1), '=' * 20)
                print('Player Win: {}'
                      '  Enemy Win: {}'
                      '  Draw: {}'
                      '  Winrate: {:.2f}%'.format(
                          pw, ew, dr, winrate))
                print('Player ELO: {:.0f}, Enemy ELO: {:.0f}'.format(
                    player_elo, enemy_elo))
                evaluator.reset()


# Web API
app = flask.Flask(__name__)
app.register_blueprint(web_api)
log = logging.getLogger('werkzeug')
log.disabled = True


@app.route('/action')
def action():

    action_idx = int(flask.request.args.get("action_idx"))
    data = {"success": False}
    evaluator.put_action(action_idx, game_info.curr_turn, game_info.enemy_turn)
    data["success"] = True

    return flask.jsonify(data)


if __name__ == '__main__':
    print('cuda:', use_cuda)
    np.set_printoptions(suppress=True)
    np.random.seed(0)
    torch.manual_seed(0)
    if use_cuda:
        torch.cuda.manual_seed_all(0)

    # Web API
    print("Activate Web API...")
    app_th = threading.Thread(target=app.run,
                              kwargs={"host": "0.0.0.0", "port": 5000})
    app_th.start()
    main()
