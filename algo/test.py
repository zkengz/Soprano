import argparse
from pokertrees import *
from pokergames import *
from pokerstrategy import *
from pokerstrategyTensor import *
from pokercfr import *
from card import Card
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# vscode单独打开项目文件夹，所以用从algo导入；如果vscode打开整个用户文件夹，可以用from stoch_optim.algo导入
# 如果无法导入，需要在当前终端内输入export PYTHONPATH=$PYTHONPATH:/home/zengzekeng/stoch_optim
from algo.pokerstrategyTensorSample import OutcomeSamplingSolver
from algo.pokerstrategyTensorVROS import VROutcomeSamplingSolver
from algo.pokerstrategyTensorESCHER import ESCHEROutcomeSamplingSolver


def test_compute_graph():
    """计算图clone测试"""
    a = torch.tensor([3], dtype=torch.float32, requires_grad=True)
    b = torch.tensor([2], dtype=torch.float32, requires_grad=True)

    c = a * b
    c.retain_grad()

    d = c.clone()
    d.retain_grad()

    pred = c + d
    pred.retain_grad()

    target = torch.tensor([10], dtype=torch.float32)
    loss = (pred - target) ** 2
    loss.backward()

    print(pred.grad)
    print(c.grad)
    print(d.grad)
    print(a.grad)
    print(b.grad)

    debug = 1

    """计算图二阶导测试"""
    a = torch.tensor([0.3,0.7,0.0], dtype=torch.float32, requires_grad=True)

    c = a[0] * torch.tensor([5], dtype=torch.float32, requires_grad=True)
    grad = torch.autograd.grad(c, a, create_graph=True)[0] # 这一步操作，要求c的构建过程中所有节点都要requires_grad=True
    # c.backward(retain_graph=True)

    norm = projected_gradient_norm([True, True, False], grad)
    # norm = projected_gradient_norm([True, True, False], a.grad)
    norm.backward()
    print(a.grad)

    debug = 1
    

# def projected_gradient(grad): # grad: [None, 0.1, 0.2] -> [None, -0.05, 0.05]
#     mean_val = np.mean([g for g in grad if g != None])
#     grad_return = [None if g == None else g - mean_val for g in grad]
#     return grad_return

"""
不对，损失函数是梯度，我需要对梯度求梯度，需要更改底层的compute_gradient函数
"""
# def policy_proj_grad_descent(strategy, grad, lr):
#     for infoset in grad.keys():
#         # project gradient
#         grad_mean = np.mean([g for g in grad[infoset] if g != None])
#         grad[infoset] = [None if g == None else g - grad_mean for g in grad[infoset]]

#         new_update = float('inf')
#         for a, g in enumerate(grad[infoset]):
#             if g != None:
#                 if g > 0:
#                     new_update = min(strategy.policy[infoset][a], lr * g, new_update)
#                 if g < 0:
#                     new_update = min(1 - strategy.policy[infoset][a], -lr * g, new_update)
#         for a, g in enumerate(grad[infoset]):
#             g = grad[infoset][a]
#             if g != None:
#                 if g > 0:
#                     strategy.policy[infoset][a] -= new_update
#                 if g < 0:
#                     strategy.policy[infoset][a] += new_update


def projected_gradient_norm(valid, grad):    
    # 将 valid 转换为布尔张量以便索引
    valid_tensor = torch.tensor(valid, dtype=torch.bool)
    # 选择有效的梯度项并计算均值
    valid_grads = grad[valid_tensor]
    grad_mean = valid_grads.mean()
    # 计算投影梯度：有效位置为0，无效位置减去均值
    proj_grad = torch.where(valid_tensor, grad - grad_mean, torch.zeros_like(grad))
    # 计算投影梯度的范数
    proj_grad_norm = proj_grad.norm()
    return proj_grad_norm


def test_stoch_optim(random_seed=0, n_iter=1000, lr=0.01, tao=0.01):

    gamerules = half_street_kuhn_rules()
    gametree = half_street_kuhn_gametree()
    players = 2
    teams = None
    """1000次 结果是nashconv=0.054"""

    # gamerules = kuhn_rules()
    # gametree = kuhn_gametree()
    # players = 2
    # teams = None
    """1000次 结果是nashconv=0.068"""
    """如果用范数而非范数平方，先降后升，最小0.02，最终0.09，为何？ Adam. SGD就不会有先降后升，lr=0.01和lr=0.1，到0.04"""

    # 3-p Kuhn
    # players = 3
    # deck = [Card(14,1),Card(13,1),Card(12,1),Card(11,1)]
    # ante = 1
    # blinds = None
    # rounds = [RoundInfo(holecards=1,boardcards=0,betsize=1,maxbets=[1,1,1])]
    # gamerules = GameRules(players, deck, rounds, ante, blinds, handeval=kuhn_eval, infoset_format=leduc_format)
    # gametree = GameTree(gamerules)
    # gametree.build()
    # teams = [[0,2], [1]]
    """1000次 结果是nashconv=0.072"""

    # 4-p Kuhn
    # players = 4
    # deck = [Card(14,1),Card(13,1),Card(12,1),Card(11,1),Card(10,1)]
    # ante = 1
    # blinds = None
    # rounds = [RoundInfo(holecards=1,boardcards=0,betsize=1,maxbets=[1,1,1,1])]
    # gamerules = GameRules(players, deck, rounds, ante, blinds, handeval=kuhn_eval, infoset_format=leduc_format)
    # gametree = GameTree(gamerules)
    # gametree.build()
    # teams = [[0,2], [1,3]]
    """1000次 结果是nashconv=0.037"""


    valid_dict = {}
    for key in gametree.information_sets:
        infoset = gametree.information_sets[key]
        test_node = infoset[0]
        valid_dict[key] = [True if test_node.valid(action) is not None else False for action in range(3)]

    s = []
    for p in range(players):
        s.append(StrategyTensor(p))
        s[p].build_default(gametree)
        # s[p].build_random(gametree, random_seed)

    # s[0].load_from_file("/home/zengzekeng/stoch_optim/strategies/kuhn_p2r3_p0_ne.strat")
    # s[1].load_from_file("/home/zengzekeng/stoch_optim/strategies/kuhn_p2r3_p1_ne.strat")
    
    profile = StrategyProfileTensor(gamerules, s, teams=teams, gradient_only=True, tao=tao)

    param_list = []
    for p in range(players):
        for param in profile.strategies[p].policy.values():
            param_list.append(param)
    optimizer = torch.optim.SGD(param_list, lr=lr)

    loss = None

    for i in range(n_iter):
        if i % 10 == 0:
            _, ev_br = profile.best_response(br_players=list(range(players)))
            nashconv = sum(ev_br)
            print(f"iter {i}: loss={loss}, nashconv={nashconv}")

        ev = profile.expected_value()
        loss = sum(sum(profile.proj_grad_norms[p].values()) for p in range(players))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"before softmax: param_list[6]={param_list[6]}, grad={param_list[6].grad}")
        for p in range(players):
            for infoset, policy in profile.strategies[p].policy.items():
                with torch.no_grad():
                    # softmax归一化：不知为何更新会卡住
                    # updated_policy = torch.zeros_like(policy)
                    # updated_policy[valid_dict[infoset]] = F.softmax(policy[valid_dict[infoset]], dim=0)
                    # policy.copy_(updated_policy)

                    # # 加和归一化：
                    # prob_sum = torch.sum(policy, dtype=torch.float32)
                    # policy.div_(prob_sum) # bug：用policy = policy / prob_sum不知为何会导致policy总和不为1
                    # # 限制在[0,1]范围内
                    # policy.clamp_(0, 1)

                    policy.clamp_(min=0)
                    policy.div_(torch.sum(policy))
        
        # print(f"iter {i}: param_list[6]={param_list[6]}, grad={param_list[6].grad}")

    for p in range(players):
        for infoset, policy in profile.strategies[p].policy.items():
            print(f"p={p}, infoset={infoset}, policy={policy}")

    debug = 1


def test_cfr():
    # hskuhn = half_street_kuhn_rules()
    # leduc = leduc_rules()
    players = 3
    rank = 4
    maxbets = 1
    deck = [Card(14-r,1) for r in range(rank)]
    betsize = 1 
    rounds = [RoundInfo(holecards=1,boardcards=0,betsize=betsize,maxbets=[maxbets,maxbets,maxbets])]
    ante = 1 
    blinds = None 
    gamerules = GameRules(players, deck, rounds, ante, blinds, handeval=kuhn_eval, infoset_format=leduc_format) 
    
    # cfr = CounterfactualRegretMinimizer(leduc)
    # cfr = ChanceSamplingCFR(hskuhn)
    cfr = OutcomeSamplingCFR(gamerules)

    iter_per_block = 10000
    blocks = 100
    for block in range(blocks):
        print(f"iterations: {block * iter_per_block}")
        cfr.run(iter_per_block)
        result = cfr.profile.best_response(br_players=list(range(players)))
        print(f"best response EV: {result[1]}")
        print(f"nashconv: {sum(result[1])}")
    print(cfr.profile.strategies[0].policy)
    print(cfr.profile.strategies[1].policy)
    print(cfr.counterfactual_regret)

    debug = 1

    return sum(result[1])

def test_sample_stoch_optim():
    gamerules = half_street_kuhn_rules()
    # gamerules = kuhn_rules()
    solver = OutcomeSamplingSolver(gamerules, lr=0.1, tao=0.01)
    solver.run(num_iterations=500)

def test_VROS_stoch_optim():
    # gamerules = half_street_kuhn_rules()
    # gamerules = kuhn_rules()
    # gamerules = leduc_rules()
    players = 3
    rank = 4
    maxbets = 1
    deck = [Card(14-r,1) for r in range(rank)]
    betsize = 1 
    rounds = [RoundInfo(holecards=1,boardcards=0,betsize=betsize,maxbets=[maxbets,maxbets,maxbets])]
    ante = 1 
    blinds = None 
    gamerules = GameRules(players, deck, rounds, ante, blinds, handeval=kuhn_eval, infoset_format=leduc_format) 

    bl_type = "na"
    # bl_type = "infoset"
    # bl_type = "history"

    bl_weighting = "exponential"
    # bl_weighting = "linear"

    bl_reset = False
    # bl_reset = True

    random_seed = 0

    save_path="/home/zengzekeng/stoch_optim/results/test_results"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    solver = VROutcomeSamplingSolver(gamerules, teams=None, lr=5e-3, exploration=0.1, tao=0.5, sample_traj_num=10, 
                                     baseline_type=bl_type, baseline_weighting=bl_weighting, baseline_reset=bl_reset, 
                                     save_path=save_path, seed=None,
                                     debug=False)
    solver.run(num_iterations=50000, eval_freq=100)
    
    debug = 1
    return solver.results[-1][-1]

def test_ESCHER_stoch_optim():
    # gamerules = half_street_kuhn_rules()
    gamerules = kuhn_rules()
    # gamerules = leduc_rules()

    bl_type = "na"
    # bl_type = "infoset"
    # bl_type = "history"

    bl_weighting = "exponential"
    # bl_weighting = "linear"

    # bl_reset = False
    bl_reset = True

    solver = ESCHEROutcomeSamplingSolver(gamerules, lr=1e-1, tao=0.01, baseline_type=bl_type, baseline_weighting=bl_weighting, baseline_reset=bl_reset)
    solver.run(num_iterations=1000)
    """

    """

if __name__ == "__main__":
    seed = 0
    random.seed(seed)

    cfr_nashconv = test_cfr()

    # test_stoch_optim()

    # test_sample_stoch_optim()

    soprano_nashconv = test_VROS_stoch_optim()

    # test_ESCHER_stoch_optim()

    print(f"cfr_nashconv: {cfr_nashconv}, soprano_nashconv: {soprano_nashconv}")