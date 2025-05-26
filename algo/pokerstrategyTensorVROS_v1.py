from pokertrees import *
import random
import numpy as np
import torch
from collections import defaultdict
import os

def create_default_list():
    return [0,0,0]

def projected_gradient(valid_tensor, grad):    
    # 选择有效的梯度项并计算均值
    valid_grads = grad[valid_tensor]
    grad_mean = valid_grads.mean() # BUG!?
    # 计算投影梯度：有效位置减去均值，无效位置为0
    proj_grad = torch.where(valid_tensor, grad - grad_mean, torch.zeros_like(grad))
    return proj_grad

def projected_gradient_norm(valid_tensor, grad):    
    # 选择有效的梯度项并计算均值
    valid_grads = grad[valid_tensor]
    grad_mean = valid_grads.mean()
    # 计算投影梯度：有效位置减去均值，无效位置为0
    proj_grad = torch.where(valid_tensor, grad - grad_mean, torch.zeros_like(grad))
    # 计算投影梯度的范数
    proj_grad_norm = proj_grad.norm()
    return proj_grad_norm

def L2proj_simplex(v):
    """
    project v to the probability simplex using L2 norm
    """
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    rho = np.where(u * np.arange(1, len(v) + 1) > (sv - 1))[0][-1]
    theta = (sv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w

def legal_L2proj_simplex(probs, legal_actions):
    res = [0,0,0]
    legal_probs = [probs[a] for a in range(3) if a in legal_actions]
    legal_probs = L2proj_simplex(legal_probs)
    j = 0
    for a in range(3):
        if a in legal_actions:
            res[a] = legal_probs[j]
            j += 1
    return res

def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

class StrategyTensor(object):
    def __init__(self, player, filename=None):
        self.player = player
        self.policy = {}
        if filename is not None:
            self.load_from_file(filename)

    def build_default(self, gametree):
        for key in gametree.information_sets:
            infoset = gametree.information_sets[key]
            test_node = infoset[0]
            if test_node.player == self.player:
                for node in infoset:
                    prob = 1.0 / float(len(node.children))
                    probs = [0,0,0]
                    for action in range(3):
                        if node.valid(action):
                            probs[action] = prob
                    if type(node.player_view) is tuple:
                        for pview in node.player_view:
                            self.policy[pview] = torch.tensor(probs, requires_grad=True) # 注意避免多个infoset共享一个tensor
                    else:
                        self.policy[node.player_view] = torch.tensor(probs, requires_grad=True)

    def build_random(self, gametree, seed=0):
        random.seed(seed)
        for key in gametree.information_sets:
            infoset = gametree.information_sets[key]
            test_node = infoset[0]
            if test_node.player == self.player:
                for node in infoset:
                    probs = [0 for _ in range(3)]
                    total = 0
                    for action in range(3):
                        if node.valid(action):
                            probs[action] = random.random()
                            total += probs[action]
                    if type(node.player_view) is tuple:
                        for pview in node.player_view:
                            self.policy[pview] = torch.tensor([x / total for x in probs], requires_grad=True)
                    else:
                        self.policy[node.player_view] = torch.tensor([x / total for x in probs], requires_grad=True)

    def probs(self, infoset):
        assert(infoset in self.policy)
        return self.policy[infoset]

    def sample_action(self, infoset):
        assert(infoset in self.policy)
        probs = self.policy[infoset]
        val = random.random()
        total = 0
        for i,p in enumerate(probs):
            total += p
            if p > 0 and val <= total:
                return i
        raise Exception('Invalid probability distribution. Infoset: {0} Probs: {1}'.format(infoset, probs))
    
    def load_from_file(self, filename):
        self.policy = {}
        f = open(filename, 'r')
        for line in f:
            line = line.strip()
            if line == "" or line.startswith('#'):
                continue
            tokens = line.split(' ')
            assert(len(tokens) == 4)
            key = tokens[0]
            probs = [float(x) for x in reversed(tokens[1:])]
            self.policy[key] = torch.tensor(probs, requires_grad=True)

    def save_to_file(self, filename):
        # 先检查filename的父路径存在与否，不存在则创建
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        f = open(filename, 'w')
        for key in sorted(self.policy.keys()):
            val = self.policy[key]
            f.write("{0} {1:.9f} {2:.9f} {3:.9f}\n".format(key, val[2], val[1], val[0]))
        f.flush()
        f.close()


class StrategyProfileTensor(object):
    def __init__(self, rules, strategies, teams=None, tao=0.01):
        assert(rules.players == len(strategies))
        self.rules = rules
        self.strategies = strategies
        self.teams = teams
        self.gametree = None
        self.publictree = None
        # self.tao = torch.tensor(tao, dtype=torch.float32, requires_grad=True)
        # self.proj_grad_norms = {p: {infoset: None for infoset in self.strategies[p].policy.keys()} for p in range(rules.players)}

    def expected_value(self):
        """
        Calculates the expected value of each strategy in the profile.
        Returns an array of scalars corresponding to the expected payoffs.
        """
        if self.gametree is None:
            self.gametree = PublicTree(self.rules, self.teams)
        if self.gametree.root is None:
            self.gametree.build()
        # 此处没有用tensor，是为了让holdcard_node直接用torch.tensor，而不需要用clone
        # expected_values = self.ev_helper(self.gametree.root, [{(): torch.tensor(1.0, dtype=torch.float32, requires_grad=True)} for _ in range(self.rules.players)])
        expected_values = self.ev_helper(self.gametree.root, [{(): 1} for _ in range(self.rules.players)])
        for ev in expected_values:
            assert(len(ev) == 1)
        return tuple(list(ev.values())[0] for ev in expected_values) # pull the EV from the dict returned

    def ev_helper(self, root, reachprobs):
        if type(root) is TerminalNode:
            return self.ev_terminal_node(root, reachprobs)
        if type(root) is HolecardChanceNode:
            return self.ev_holecard_node(root, reachprobs)
        if type(root) is BoardcardChanceNode:
            return self.ev_boardcard_node(root, reachprobs)
        return self.ev_action_node(root, reachprobs)

    def ev_terminal_node(self, root, reachprobs, counterfactual=True):
        payoffs = [None for _ in range(self.rules.players)]
        if counterfactual:
            # q_i,pi(s) = [Σ_z∈S\ eta_pi_-i(h) * u_i(z)] / |S|
            for player in range(self.rules.players):
                player_payoffs = {hc: torch.tensor(0.0, dtype=torch.float32, requires_grad=True) for hc in root.holecards[player]}
                counts = {hc: torch.tensor(0.0, dtype=torch.float32, requires_grad=True) for hc in root.holecards[player]}
                for hands,winnings in root.payoffs.items():
                    prob = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
                    player_hc = None
                    for opp,hc in enumerate(hands):
                        if opp == player:
                            player_hc = hc
                        else:
                            prob = prob * reachprobs[opp][hc] # 不能用*=原地修改
                    player_payoffs[player_hc] = player_payoffs[player_hc] + prob * torch.tensor(winnings[player], dtype=torch.float32, requires_grad=True)
                    counts[player_hc] = counts[player_hc] + 1
                for hc,count in counts.items():
                    if count > 0:
                        player_payoffs[hc] = player_payoffs[hc] / count
                payoffs[player] = player_payoffs
        else:
            raise NotImplementedError("Non-counterfactual EV not implemented.")
        return payoffs

    def ev_holecard_node(self, root, reachprobs):
        assert(len(root.children) == 1)
        prevlen = len(list(reachprobs[0].keys())[0])
        possible_deals = float(choose(len(root.deck) - prevlen,root.todeal))
        next_reachprobs = [{ hc: torch.tensor(reachprobs[player][hc[0:prevlen]] / possible_deals) for hc in root.children[0].holecards[player] } for player in range(self.rules.players)]
        subpayoffs = self.ev_helper(root.children[0], next_reachprobs)
        payoffs = [{ hc: 0 for hc in root.holecards[player] } for player in range(self.rules.players)]
        for player, subpayoff in enumerate(subpayoffs):
            for hand,winnings in subpayoff.items():
                hc = hand[0:prevlen]
                payoffs[player][hc] += winnings
        return payoffs

    def ev_boardcard_node(self, root, reachprobs):
        prevlen = len(list(reachprobs[0].keys())[0])
        possible_deals = float(choose(len(root.deck) - prevlen,root.todeal))
        payoffs = [{ hc: 0 for hc in root.holecards[player] } for player in range(self.rules.players)]
        for bc in root.children:
            next_reachprobs = [{ hc: reachprobs[player][hc] / possible_deals for hc in bc.holecards[player] } for player in range(self.rules.players)]
            subpayoffs = self.ev_helper(bc, next_reachprobs)
            for player,subpayoff in enumerate(subpayoffs):
                for hand,winnings in subpayoff.items():
                    payoffs[player][hand] += winnings
        return payoffs

    def ev_action_node(self, root, reachprobs):
        strategy = self.strategies[root.player]
        next_reachprobs = [{hc: reachprobs[player][hc].clone() for hc in root.children[0].holecards[player]} for player in range(self.rules.players)]
        action_probs = { hc: strategy.probs(self.rules.infoset_format(root.player, hc, root.board, root.bet_history)) for hc in root.holecards[root.player] }
        action_payoffs = [None, None, None]
        if root.fold_action:
            next_reachprobs[root.player] = { hc: action_probs[hc][FOLD] * reachprobs[root.player][hc] for hc in root.holecards[root.player] }
            action_payoffs[FOLD] = self.ev_helper(root.fold_action, next_reachprobs)
        if root.call_action:
            next_reachprobs[root.player] = { hc: action_probs[hc][CALL] * reachprobs[root.player][hc] for hc in root.holecards[root.player] }
            action_payoffs[CALL] = self.ev_helper(root.call_action, next_reachprobs)
        if root.raise_action:
            next_reachprobs[root.player] = { hc: action_probs[hc][RAISE] * reachprobs[root.player][hc] for hc in root.holecards[root.player] }
            action_payoffs[RAISE] = self.ev_helper(root.raise_action, next_reachprobs)
        payoffs = []
        for player in range(self.rules.players):
            player_payoffs = { hc: torch.tensor(0.0, dtype=torch.float32, requires_grad=True) for hc in root.holecards[player] }
            for action,subpayoff in enumerate(action_payoffs):
                if subpayoff is None:
                    continue
                if root.player == player:
                    for hc,winnings in subpayoff[player].items():
                        player_payoffs[hc] = player_payoffs[hc] + winnings * action_probs[hc][action] # v_i_pi(s) = Σ_a∈A\ q_i_pi(s,a) * pi_i(a|s) 
                else:
                    for hc,winnings in subpayoff[player].items():
                        player_payoffs[hc] = player_payoffs[hc] + winnings
            payoffs.append(player_payoffs)
        valid = torch.tensor([True if root.valid(action) else False for action in range(3)], dtype=torch.bool)
        # k = list(action_probs.keys())
        for hc, winnings in payoffs[root.player].items():
            # 计算反事实价值对动作的梯度
            grad = torch.autograd.grad(winnings, action_probs[hc], create_graph=True)[0]
            # 计算动作熵的梯度
            # ent_grad = torch.where(valid, - torch.log(action_probs[hc]) - 1, torch.zeros_like(grad)) # nan仍然影响计算图
            valid_ent_grad = - torch.log(torch.clamp(torch.masked_select(action_probs[hc], valid), 1e-5, 1)) - 1 # clamp避免log(0)
            ent_grad = torch.zeros_like(grad)
            ent_grad.masked_scatter_(valid, valid_ent_grad)
            # 计算总梯度的投影的范数，作为损失函数
            proj_grad_norm = projected_gradient_norm(valid, grad + self.tao * ent_grad)
            self.proj_grad_norms[root.player][self.rules.infoset_format(root.player, hc, root.board, root.bet_history)] = proj_grad_norm
        # if root.bet_history == '/r':
        #     loss = sum([0 if value==None else value for value in self.proj_grad_norms[root.player].values()])
        #     print(loss)
        #     loss.backward()
        #     print(action_probs[k[0]].grad)
        return payoffs

    def best_response(self, br_players=[], ga_players=[], ga_lr=1):
        """
        Calculates the best response OR gradient ascent response for each player in the strategy profile.
        Returns a list of tuples of the best response strategy and its expected value for each player.

        if a player P is not in br_players nor ga_players, 
        then the corresponding returned br is empty and returned ev is the expected value of P's current strategy. 
        """
        # br_players = [x for x in list(range(self.rules.players)) if x not in ga_players]
        if self.publictree is None:
            self.publictree = PublicTree(self.rules, self.teams)
        if self.publictree.root is None:
            self.publictree.build()
        responses = [StrategyTensor(player) for player in range(self.rules.players)]
        expected_values = self.br_helper(self.publictree.root, [{(): 1} for _ in range(self.rules.players)], responses, br_players, ga_players, ga_lr)
        for ev in expected_values:
            assert(len(ev) == 1)
        expected_values = tuple(list(ev.values())[0] for ev in expected_values) # pull the EV from the dict returned
        return (StrategyProfileTensor(self.rules, responses, self.teams), expected_values)

    def br_helper(self, root, reachprobs, responses, br_players, ga_players, ga_lr):
        if type(root) is TerminalNode:
            return self.ev_terminal_node(root, reachprobs)
        if type(root) is HolecardChanceNode:
            return self.br_holecard_node(root, reachprobs, responses, br_players, ga_players, ga_lr)
        if type(root) is BoardcardChanceNode:
            return self.br_boardcard_node(root, reachprobs, responses, br_players, ga_players, ga_lr)
        return self.br_action_node(root, reachprobs, responses, br_players, ga_players, ga_lr)

    def br_holecard_node(self, root, reachprobs, responses, br_players, ga_players, ga_lr):
        assert(len(root.children) == 1)
        prevlen = len(list(reachprobs[0].keys())[0])
        possible_deals = float(choose(len(root.deck) - prevlen,root.todeal))
        next_reachprobs = [{ hc: torch.tensor(reachprobs[player][hc[0:prevlen]] / possible_deals) for hc in root.children[0].holecards[player] } for player in range(self.rules.players)]
        subpayoffs = self.br_helper(root.children[0], next_reachprobs, responses, br_players, ga_players, ga_lr)
        payoffs = [{ hc: 0 for hc in root.holecards[player] } for player in range(self.rules.players)]
        for player, subpayoff in enumerate(subpayoffs):
            for hand,winnings in subpayoff.items():
                hc = hand[0:prevlen]
                payoffs[player][hc] += winnings
        return payoffs

    def br_boardcard_node(self, root, reachprobs, responses, br_players, ga_players, ga_lr):
        prevlen = len(list(reachprobs[0].keys())[0])
        possible_deals = float(choose(len(root.deck) - prevlen,root.todeal))
        payoffs = [{ hc: 0 for hc in root.holecards[player] } for player in range(self.rules.players)]
        for bc in root.children:
            next_reachprobs = [{ hc: reachprobs[player][hc] / possible_deals for hc in bc.holecards[player] } for player in range(self.rules.players)]
            subpayoffs = self.br_helper(bc, next_reachprobs, responses, br_players, ga_players, ga_lr)
            for player,subpayoff in enumerate(subpayoffs):
                for hand,winnings in subpayoff.items():
                    payoffs[player][hand] += winnings
        return payoffs

    def br_action_node(self, root, reachprobs, responses, br_players, ga_players, ga_lr):
        strategy = self.strategies[root.player]
        next_reachprobs = [{hc: reachprobs[player][hc].clone() for hc in root.children[0].holecards[player]} for player in range(self.rules.players)]
        action_probs = { hc: strategy.probs(self.rules.infoset_format(root.player, hc, root.board, root.bet_history)) for hc in root.holecards[root.player] }
        action_payoffs = [None, None, None]
        if root.fold_action:
            next_reachprobs[root.player] = { hc: action_probs[hc][FOLD] * reachprobs[root.player][hc] for hc in root.holecards[root.player] }
            action_payoffs[FOLD] = self.br_helper(root.fold_action, next_reachprobs, responses, br_players, ga_players, ga_lr)
        if root.call_action:
            next_reachprobs[root.player] = { hc: action_probs[hc][CALL] * reachprobs[root.player][hc] for hc in root.holecards[root.player] }
            action_payoffs[CALL] = self.br_helper(root.call_action, next_reachprobs, responses, br_players, ga_players, ga_lr)
        if root.raise_action:
            next_reachprobs[root.player] = { hc: action_probs[hc][RAISE] * reachprobs[root.player][hc] for hc in root.holecards[root.player] }
            action_payoffs[RAISE] = self.br_helper(root.raise_action, next_reachprobs, responses, br_players, ga_players, ga_lr)
        payoffs = []
        for player in range(self.rules.players):
            if player is root.player:
                if player in ga_players: # gradient ascent
                    if self.gradient_only: # compute gradient only, without updating strategy
                        payoffs.append(self.compute_gradient(root, responses, action_payoffs))
                    else: # compute gradient and update strategy
                        payoffs.append(self.ga_response_action(root, responses, action_payoffs, ga_lr))
                elif player in br_players: # best response
                    payoffs.append(self.br_response_action(root, responses, action_payoffs))
                else:
                    player_payoffs = { hc: 0 for hc in root.holecards[player] }
                    for action, subpayoff in enumerate(action_payoffs):
                        if subpayoff is None:
                            continue
                        for hc,winnings in subpayoff[player].items():
                            player_payoffs[hc] += winnings * action_probs[hc][action]
                    payoffs.append(player_payoffs)
            else:
                player_payoffs = { hc: 0 for hc in root.holecards[player] }
                for subpayoff in action_payoffs:
                    if subpayoff is None:
                        continue
                    for hc,winnings in subpayoff[player].items():
                        player_payoffs[hc] += winnings
                payoffs.append(player_payoffs)
        return payoffs

    def br_response_action(self, root, responses, action_payoffs):
        """
        compute best response strategy (parameter: repsonses) and return the corresponding payoffs
        """
        player_payoffs = { }
        max_strategy = responses[root.player]
        for hc in root.holecards[root.player]:
            max_action = None
            if action_payoffs[FOLD]:
                max_action = [FOLD]
                max_value = action_payoffs[FOLD][root.player][hc]
            if action_payoffs[CALL]:
                value = action_payoffs[CALL][root.player][hc]
                if max_action is None or value > max_value:
                    max_action = [CALL]
                    max_value = value
                elif max_value == value:
                    max_action.append(CALL)
            if action_payoffs[RAISE]:
                value = action_payoffs[RAISE][root.player][hc]
                if max_action is None or value > max_value:
                    max_action = [RAISE]
                    max_value = value
                elif max_value == value:
                    max_action.append(RAISE)
            probs = [0,0,0]
            for action in max_action:
                probs[action] = 1.0 / float(len(max_action))
            infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
            max_strategy.policy[infoset] = probs
            player_payoffs[hc] = max_value
        return player_payoffs

    def ga_response_action(self, root, responses, action_payoffs, ga_lr):
        """
        compute gradient ascent strategy (parameter: repsonses) and return the corresponding payoffs
        """
        player_payoffs = { hc: 0 for hc in root.holecards[root.player] }
        max_strategy = responses[root.player]
        legal_actions = [action for action in range(3) if action_payoffs[action]]
        for hc in root.holecards[root.player]:
            infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
            probs = self.strategies[root.player].policy[infoset]
            if action_payoffs[FOLD]:
                advantage = action_payoffs[FOLD][root.player][hc]
                probs[FOLD] = probs[FOLD] + advantage * ga_lr
            if action_payoffs[CALL]:
                advantage = action_payoffs[CALL][root.player][hc]
                probs[CALL] = probs[CALL] + advantage * ga_lr
            if action_payoffs[RAISE]:
                advantage = action_payoffs[RAISE][root.player][hc]
                probs[RAISE] = probs[RAISE] + advantage * ga_lr
            max_strategy.policy[infoset] = legal_L2proj_simplex(probs, legal_actions)
            for action,subpayoff in enumerate(action_payoffs):
                if subpayoff is None:
                    continue
                player_payoffs[hc] += subpayoff[root.player][hc] * probs[action]
        return player_payoffs

    def compute_gradient(self, root, responses, action_payoffs):
        """
        compute strategy gradient (para: repsonses) and return the corresponding payoffs
        """
        player_payoffs = { hc: 0 for hc in root.holecards[root.player] }
        gradients = responses[root.player]
        for hc in root.holecards[root.player]:
            infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
            probs = self.strategies[root.player].policy[infoset]
            gradient = [None, None, None]
            if action_payoffs[FOLD]:
                cf_value = action_payoffs[FOLD][root.player][hc]
                gradient[FOLD] = cf_value
            if action_payoffs[CALL]:
                cf_value = action_payoffs[CALL][root.player][hc]
                gradient[CALL] = cf_value
            if action_payoffs[RAISE]:
                cf_value = action_payoffs[RAISE][root.player][hc]
                gradient[RAISE] = cf_value
            gradients.policy[infoset] = gradient
            for action,subpayoff in enumerate(action_payoffs):
                if subpayoff is None:
                    continue
                player_payoffs[hc] += subpayoff[root.player][hc] * probs[action]
        return player_payoffs


class VROutcomeSamplingSolver(object):
    def __init__(self, rules, teams, exploration=0.1, lr=0.01, tao=0.01, sample_traj_num=10, 
                 baseline_type="infoset", baseline_weighting="exponential", baseline_reset=False, 
                 save_path=None, debug=False):
        self.rules = rules
        self.profile = StrategyProfileTensor(rules, [StrategyTensor(i) for i in range(rules.players)], tao=tao)
        self.iteration = 0
        self.action_reachprobs = []
        self.exploration = exploration
        self.tree = PublicTree(rules, teams=teams)
        self.tree.build()
        # print('Information sets: {0}'.format(len(self.tree.information_sets)))
        for s in self.profile.strategies:
            s.build_default(self.tree)
            self.action_reachprobs.append({ infoset: [0,0,0] for infoset in s.policy })
        self.tao = tao
        self.sample_traj_num = sample_traj_num
        self.proj_grad_dict = {} # 储存信息集-投影梯度
        self.norm_dict = defaultdict(float) # 为成对出现的信息集计算投影梯度范数的估计值
        self.num_infoset_visited = 0 # TODO：目前的方式是统计到达的，而非凑够两个更新的
        self.num_infoset_updated = 0
        self.results = []
        self.save_path = save_path
        self.debug = debug
        
        # self.valid_dict = {}
        # for key in self.tree.information_sets:
        #     infoset = self.tree.information_sets[key]
        #     test_node = infoset[0]
        #     self.valid_dict[key] = [True if test_node.valid(action) is not None else False for action in range(3)]

        self.params = {}
        for p in range(rules.players):
            for infoset, param in self.profile.strategies[p].policy.items():
                self.params[infoset] = param
        self.optimizer = torch.optim.SGD(list(self.params.values()), lr=lr)
        
        # FOR DEBUG
        if self.debug:
            self.cfv = {}
            self.cfv_cnt = {}

        self.norm_cnt = {}
        for p in range(rules.players):
            for infoset, param in self.profile.strategies[p].policy.items():
                self.norm_cnt[infoset] = 0
                if self.debug:
                    self.cfv_cnt[infoset] = 0
                    self.cfv[infoset] = 0
        
        # variance-reduction baseline
        self.baseline_type = baseline_type # "na" for MCCFR, "infoset" for VR-MCCFR, "history"
        if baseline_type == "na":
            pass
        elif baseline_type == "infoset" or baseline_type == "history":
            self.baseline = {p : defaultdict(create_default_list) for p in range(rules.players)} # b(I,a) or b(h,a), 注意对于不是自己的信息集也维护baseline
        else:
            raise NotImplementedError("Baseline type not implemented.")
        self.baseline_weighting = baseline_weighting # "linear" or "exponential"
        if baseline_weighting == "exponential":
            self.bl_decay_weight = 0.5 # 与low-variance论文相同
        elif baseline_weighting == "linear":
            self.history_cnt = {p: defaultdict(create_default_list) for p in range(rules.players)} # cnt_of_(I,a) or cnt_of_(h,a)
        else:
            raise NotImplementedError("Baseline weighting method not implemented.")
        self.baseline_reset = baseline_reset # only work for "infoset" or "history"
        
    def run(self, num_iterations, eval_freq=100):
        for iteration in range(num_iterations):
            if iteration % eval_freq == 0:
                _, ev_br = self.profile.best_response(br_players=list(range(self.rules.players)))
                nashconv = sum(ev_br).item()
                print(f"iter {iteration}: num_infoset_visited={self.num_infoset_visited}, num_infoset_updated={self.num_infoset_updated}, nashconv={nashconv}")
                self.results.append((iteration, self.num_infoset_visited, self.num_infoset_updated, nashconv))
                np.save(self.save_path + "/results.npy", np.array(self.results))
            self.update()
            self.iteration += 1
            
    def update(self):
        for _ in range(self.sample_traj_num // 2): # TODO：或者改为norm_dict中达到一定数量？
            # Sample all cards to be used
            holecards_per_player = sum([x.holecards for x in self.rules.roundinfo])
            boardcards_per_hand = sum([x.boardcards for x in self.rules.roundinfo])
            todeal = random.sample(self.rules.deck, boardcards_per_hand + holecards_per_player * self.rules.players)
            # Deal holecards
            self.holecards = [tuple(todeal[p*holecards_per_player:(p+1)*holecards_per_player]) for p in range(self.rules.players)]
            # print("sampled holecards: ", self.holecards)
            self.board = tuple(todeal[-boardcards_per_hand:] if boardcards_per_hand > 0 else [])
            # Set the top card of the deck
            self.top_card = len(todeal) - boardcards_per_hand
            # Run the standard algorithm
            # 对同样的手牌进行两次采样，增大信息集重复的概率，避免无法独立采到两次相同的信息集
            # 把reachprob设为tensor，方便clone。不需要requires_grad，因为不是需要更新的参数
            self.ev_helper(self.tree.root, [torch.tensor(1.0) for _ in range(self.rules.players)], 1.0) # TODO：为什么不需要考虑chance的sampleprob？传入6.0则和CFR结果一致。
            self.ev_helper(self.tree.root, [torch.tensor(1.0) for _ in range(self.rules.players)], 1.0)

        self.num_infoset_updated += sum(self.norm_cnt.values())

        for infoset in self.norm_cnt:
            if self.norm_cnt[infoset] > 0:
                self.norm_dict[infoset] = self.norm_dict[infoset] / self.norm_cnt[infoset]
            # FOR DEBUG
            if self.debug:
                if self.cfv_cnt[infoset] > 0:
                    self.cfv[infoset] = self.cfv[infoset] / self.cfv_cnt[infoset]
        
        loss = sum(self.norm_dict.values())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.optimizer.param_groups[0]['params'], clip_value=0.99) # 限制更新步长为0.99，避免梯度爆炸，同时不影响更新（因为正常更新步长即使大于1也会被裁剪到1；如果设为1，当policy刚好为1时就会得到0，导致归一化出错）
        self.optimizer.step()

        # print(f"before softmax: param_list[6]={param_list[6]}, grad={param_list[6].grad}")
        for p in range(self.rules.players):
            for infoset, policy in self.profile.strategies[p].policy.items():
                with torch.no_grad():
                    # # 加和归一化：
                    # policy.div_(torch.sum(policy))
                    # # 限制在[0,1]范围内
                    # policy.clamp_(0, 1)
                    policy.clamp_(min=0)
                    policy.div_(torch.sum(policy))
           
        # clear infomation
        self.norm_dict = defaultdict(float)
        self.norm_cnt = defaultdict(int)
        self.proj_grad_dict = {}

        # FOR DEBUG
        # self.cfv = defaultdict(float)
        # self.cfv_cnt = defaultdict(int)

        # reset baseline
        if self.baseline_reset:
            if self.baseline_type == "infoset" or self.baseline_type == "history":
                self.baseline = {p : defaultdict(create_default_list) for p in range(self.rules.players)}
                if self.baseline_weighting == "linear":
                    self.history_cnt = {p: defaultdict(create_default_list) for p in range(self.rules.players)}
            
    def ev_helper(self, root, reachprobs, sampleprobs):
        if type(root) is TerminalNode:
            return self.ev_terminal_node(root)
        if type(root) is HolecardChanceNode:
            return self.ev_holecard_node(root, reachprobs, sampleprobs)
        if type(root) is BoardcardChanceNode:
            return self.ev_boardcard_node(root, reachprobs, sampleprobs)
        return self.ev_action_node(root, reachprobs, sampleprobs)

    def ev_terminal_node(self, root):
        payoffs = [0 for _ in range(self.rules.players)]
        for hands,winnings in root.payoffs.items():
            if not self.terminal_match(hands):
                continue
            return [winnings[player] for player in range(self.rules.players)]
        
    def terminal_match(self, hands):
        for p in range(self.rules.players):
            if not self.hcmatch(hands[p], p):
                return False
        return True

    def hcmatch(self, hc, player):
        # Checks if this hand is isomorphic to the sampled hand
        sampled = self.holecards[player][:len(hc)]
        for c in hc:
            if c not in sampled:
                return False
        return True
        
    def ev_holecard_node(self, root, reachprobs, sampleprobs):
        assert(len(root.children) == 1)
        return self.ev_helper(root.children[0], reachprobs, sampleprobs)
        
    def ev_boardcard_node(self, root, reachprobs, sampleprobs):
        # Number of community cards dealt this round
        num_dealt = len(root.children[0].board) - len(root.board)
        # Find the child that matches the sampled board card(s)
        for bc in root.children:
            if self.boardmatch(num_dealt, bc):
                results = self.ev_helper(bc, reachprobs, sampleprobs)
                return results
        raise Exception('Sampling from impossible board card')

    def boardmatch(self, num_dealt, node):
        # Checks if this node is a match for the sampled board card(s)
        for next_card in range(0, len(node.board)):
            if self.board[next_card] not in node.board:
                return False
        return True
    
    def ev_action_node(self, root, reachprobs, sampleprobs):
        strategy = self.profile.strategies[root.player]
        hc = self.holecards[root.player][0:len(root.holecards[root.player])]
        infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
        action_probs = strategy.probs(infoset)
        if random.random() < self.exploration:
            action = self.random_action(root)
        else:
            action = strategy.sample_action(infoset)
        csp = self.exploration * (1.0 / len(root.children)) + (1.0 - self.exploration) * action_probs[action]
        
        next_reachprobs = [rp.clone() for rp in reachprobs]
        next_reachprobs[root.player] *= action_probs[action]
        payoffs = self.ev_helper(root.get_child(action), next_reachprobs, sampleprobs * csp) # 此处payoff是u(h,a)

        valid = torch.tensor([True if root.valid(action) else False for action in range(3)], dtype=torch.bool)

        # 计算基线
        if self.baseline_type == "na": # MCCFR
            for player in range(self.rules.players):
                # u(h,a) with baseline for all action a
                u_act_bl = torch.tensor([0,0,0], dtype=torch.float32)
                u_act_bl[action] += payoffs[player] / csp.detach()
                payoffs[player] = torch.dot(u_act_bl, action_probs.detach()) # u(h) # 递归往上传

                if player == root.player:
                    # 为当前玩家计算cfv
                    opp_rp = 1.0 # opponent reach prob
                    for opp in range(self.rules.players):
                        if opp != root.player:
                            opp_rp = opp_rp * reachprobs[opp]
                    cfq = u_act_bl * opp_rp / sampleprobs # v(I,a) (or v(h,a)) for all action a
                    cfv = torch.dot(cfq.detach(), action_probs) # v(I) # cfq仅估计值，不传播梯度
                    # FOR DEBUG
                    if self.debug:
                        self.cfv[infoset] += cfv
                        self.cfv_cnt[infoset] += 1

        elif self.baseline_type == "infoset": # VR-MCCFR
            # 为所有玩家计算u(h)并递归上传，同时更新baseline（即u(h)的历史平均）
            for player in range(self.rules.players):
                # u(h,a) with baseline for all action a
                u_act_bl = torch.where(valid, torch.tensor([self.baseline[player][infoset][act] for act in range(3)], dtype=torch.float32), 0) 
                u_act_bl[action] += (payoffs[player] - self.baseline[player][infoset][action]) / csp.detach()
                
                if player == root.player:
                    # 为当前玩家计算cfv
                    opp_rp = 1.0 # opponent reach prob
                    for opp in range(self.rules.players):
                        if opp != root.player:
                            opp_rp = opp_rp * reachprobs[opp]
                    cfq = u_act_bl * opp_rp / sampleprobs # v(I,a) (or v(h,a)) for all action a
                    cfv = torch.dot(cfq.detach(), action_probs) # v(I) # cfq仅估计值，不传播梯度
                    # FOR DEBUG
                    if self.debug:
                        self.cfv[infoset] += cfv
                        self.cfv_cnt[infoset] += 1

                # update baseline
                if self.baseline_weighting == "exponential":
                    # self.baseline[player][infoset][action] = (1-self.bl_decay_weight) * self.baseline[player][infoset][action] + self.bl_decay_weight * u_act_bl[action].item() # u_act_bl[action]已经被baseline修正过了，用它更新会导致自举，
                    self.baseline[player][infoset][action] = (1-self.bl_decay_weight) * self.baseline[player][infoset][action] + self.bl_decay_weight * payoffs[player]
                elif self.baseline_weighting == "linear":
                    self.history_cnt[player][infoset][action] += 1
                    # self.baseline[player][infoset][action] += (u_act_bl[action].item() - self.baseline[player][infoset][action]) / self.history_cnt[player][infoset][action]
                    self.baseline[player][infoset][action] += (payoffs[player] - self.baseline[player][infoset][action]) / self.history_cnt[player][infoset][action]
                else:
                    raise NotImplementedError("Baseline weighting not implemented.")
                
                payoffs[player] = torch.dot(u_act_bl, action_probs.detach()).item() # u(h) # 递归往上传

        elif self.baseline_type == "history":
            if len(self.board) == 0: # kuhn history format: holecards_per_player/bet_history, e.g. AsKs/r
                history = ''.join([str(self.holecards[p][0]) for p in range(self.rules.players)]) + root.bet_history
            else: # leduc history format: holecards_per_player/boardcard/bet_history, e.g. AsKs/Ah/rr
                history = ''.join([str(self.holecards[p][0]) for p in range(self.rules.players)]) + '/' + str(self.board[0]) + root.bet_history

            for player in range(self.rules.players):    
                # u(h,a) with baseline for all action a
                u_act_bl = torch.where(valid, torch.tensor([self.baseline[player][history][act] for act in range(3)], dtype=torch.float32), 0) 
                # u_act_bl[action] += (payoffs[player] - self.baseline[root.player][history][action]) / csp.detach() # BUG：错了反而有效果！？
                u_act_bl[action] += (payoffs[player] - self.baseline[player][history][action]) / csp.detach()

                if player == root.player:
                    # 为当前玩家计算cfv
                    opp_rp = 1.0 # opponent reach prob
                    for opp in range(self.rules.players):
                        if opp != root.player:
                            opp_rp = opp_rp * reachprobs[opp]
                    cfq = u_act_bl * opp_rp / sampleprobs # v(I,a) (or v(h,a)) for all action a
                    cfv = torch.dot(cfq.detach(), action_probs) # v(I) # cfq仅估计值，不传播梯度
                    # FOR DEBUG
                    if self.debug:
                        self.cfv[infoset] += cfv
                        self.cfv_cnt[infoset] += 1

                # update baseline
                if self.baseline_weighting == "exponential":
                    self.baseline[player][history][action] = (1-self.bl_decay_weight) * self.baseline[player][history][action] + self.bl_decay_weight * payoffs[player]
                elif self.baseline_weighting == "linear":
                    self.history_cnt[player][infoset][action] += 1
                    self.baseline[player][history][action] += (payoffs[player] - self.baseline[player][history][action]) / self.history_cnt[player][history][action]
                else:
                    raise NotImplementedError("Baseline weighting not implemented.")
                
                payoffs[player] = torch.dot(u_act_bl, action_probs.detach()).item() # u(h) # 递归往上传
        else:
            raise NotImplementedError("Baseline type not implemented.")

        # 计算反事实价值对动作的梯度
        # BUG：pytorch对于常数不会追踪计算图，因此grad没有grad_fn；ent_grad与1向量平行，投影后为0，所以单独哪个都不会传播梯度。但加在一起投影就有些微妙的反应。
        # 原论文对玩家k的梯度范数是反传到玩家-k，所以这里用行为策略I的梯度范数反传到I，是不对的。
        grad = torch.autograd.grad(cfv, action_probs, create_graph=True)[0]
        
        # 计算动作熵的梯度
        valid_ent_grad = - torch.log(torch.clamp(torch.masked_select(action_probs, valid), 1e-5, 1)) - 1 # clamp避免log(0)
        ent_grad = torch.zeros_like(grad)
        ent_grad.masked_scatter_(valid, valid_ent_grad)
        # 计算总梯度的投影的范数，作为损失函数
        proj_grad = projected_gradient(valid, grad + self.tao * ent_grad)
        
        # DEBUG
        # proj_grad = projected_gradient(valid, self.tao * ent_grad)
        
        # self.profile.proj_grad_norms[root.player][infoset] = proj_grad_norm
        old_proj_grad = self.proj_grad_dict.get(infoset)
        if old_proj_grad is not None:
            self.norm_dict[infoset] = self.norm_dict[infoset] + torch.dot(proj_grad, old_proj_grad)
            del self.proj_grad_dict[infoset]
            self.norm_cnt[infoset] += 1
        else:
            self.proj_grad_dict[infoset] = proj_grad

        self.num_infoset_visited += 1

        return payoffs
    

    def random_action(self, root):
        options = []
        if root.fold_action:
            options.append(FOLD)
        if root.call_action:
            options.append(CALL)
        if root.raise_action:
            options.append(RAISE)
        return random.choice(options)
    