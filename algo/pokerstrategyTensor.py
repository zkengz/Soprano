from pokertrees import *
import random
import os
import numpy as np
import torch
from pokerstrategy import Strategy, StrategyProfile

def projected_gradient_norm_square(valid_tensor, grad):    
    # 选择有效的梯度项并计算均值
    valid_grads = grad[valid_tensor]
    grad_mean = valid_grads.mean()
    # 计算投影梯度：有效位置减去均值，无效位置为0
    proj_grad = torch.where(valid_tensor, grad - grad_mean, torch.zeros_like(grad))
    # 计算投影梯度的范数
    proj_grad_norm = torch.square(proj_grad.norm())
    return proj_grad_norm

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
                    probs = torch.zeros(3, dtype=torch.float32)
                    for action in range(3):
                        if node.valid(action):
                            probs[action] = prob
                    probs.requires_grad = True
                    if type(node.player_view) is tuple:
                        for pview in node.player_view:
                            self.policy[pview] = probs
                    else:
                        self.policy[node.player_view] = probs

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
                    probs = torch.tensor([x / total for x in probs], dtype=torch.float32,requires_grad=True)
                    if type(node.player_view) is tuple:
                        for pview in node.player_view:
                            self.policy[pview] = probs
                    else:
                        self.policy[node.player_view] = probs

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
            self.policy[key] = torch.tensor(probs, dtype=torch.float32, requires_grad=True)

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


class StrategyProfileTensor(object):
    def __init__(self, rules, strategies, teams=None, gradient_only=False, tao=0.01):
        assert(rules.players == len(strategies))
        self.rules = rules
        self.strategies = strategies
        self.gametree = None
        self.publictree = None
        self.teams = teams
        self.tao = torch.tensor(tao, dtype=torch.float32, requires_grad=True)
        self.gradient_only = gradient_only # whether to update strategy in gradient ascent
        self.proj_grad_norms = {p: {infoset: None for infoset in self.strategies[p].policy.keys()} for p in range(rules.players)}
        self.cfv = {}
        for p in range(rules.players):
            for infoset in self.strategies[p].policy.keys():
                self.cfv[infoset] = 0 # debug

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
            # 等价于torch.autograd.grad(- torch.sum(valid_probs * torch.log(valid_probs)), action_probs[hc], create_graph=True)[0]
            # ent_grad = torch.where(valid, - torch.log(action_probs[hc]) - 1, torch.zeros_like(grad)) # nan仍然影响计算图
            valid_ent_grad = - torch.log(torch.clamp(torch.masked_select(action_probs[hc], valid), 1e-5, 1)) - 1 # clamp避免log(0)
            ent_grad = torch.zeros_like(grad)
            ent_grad.masked_scatter_(valid, valid_ent_grad)
            # 计算总梯度的投影的范数，作为损失函数
            total_grad = grad + self.tao * ent_grad

            # 另一种方案：概率在边缘的动作梯度不参与计算
            # total_grad[valid] = total_grad[valid] - torch.mean(total_grad[valid])
            # action_probs[hc].data += total_grad * 1e-3 * 10
            # total_grad = total_grad[0.499 > abs(action_probs[hc] - 0.5)]
            # proj_grad_norm = total_grad.square().sum()
            # # total_grad = torch.where((action_probs[hc] > 1e-3) & (action_probs[hc] < 1 - 1e-3), total_grad, torch.zeros_like(total_grad))

            proj_grad_norm = projected_gradient_norm_square(valid, total_grad)

            # proj_grad_norm = projected_gradient_norm(valid, total_grad)

            self.proj_grad_norms[root.player][self.rules.infoset_format(root.player, hc, root.board, root.bet_history)] = proj_grad_norm
        
            self.cfv[self.rules.infoset_format(root.player, hc, root.board, root.bet_history)] = winnings
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
