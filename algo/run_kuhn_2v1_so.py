import argparse
from pokertrees import *
from pokergames import *
from pokerstrategy import *
from pokerstrategyTensor import *
from card import Card
import numpy as np
import torch.nn.functional as F
import pickle
import os

# vscode单独打开项目文件夹，所以用从algo导入；如果vscode打开整个用户文件夹，可以用from stoch_optim.algo导入
# 如果无法导入，需要在当前终端内输入export PYTHONPATH=$PYTHONPATH:/home/zengzekeng/stoch_optim
from algo.pokerstrategyTensorSample import OutcomeSamplingSolver
from algo.pokerstrategyTensorVROS import VROutcomeSamplingSolver
from algo.pokerstrategyTensorESCHER import ESCHEROutcomeSamplingSolver


def parse_list(input_str):
    return input_str.split(',')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def even_int(value):
    ivalue = int(value)
    if ivalue % 2 != 0:
        raise argparse.ArgumentTypeError(f"{value} 必须是偶数。")
    return ivalue

if __name__ == "__main__":
    """
    3-player Kuhn Poker ([0,2] vs [1])
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=4, help="rank of the deck")
    parser.add_argument("--maxbets", type=int, default=1, help="max number of bets")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed of initial strategy")
    parser.add_argument("--n_iter", type=int, default=50000, help="number of iterations")
    parser.add_argument("--n_traj", type=even_int, default=10, help="number of sampled trajectories per iteration, must be even")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--expl", type=float, default=0.1, help="exploration rate of sample policy")
    parser.add_argument("--tau", type=float, default=0.01, help="coefficient of entropy regularization")
    # parser.add_argument("--algo", type=parse_list, default=["MCCFR", "MCCFR-infoset", "MCCFR-history"], help="algorithms to run")
    parser.add_argument("--algo", type=str, default="MCCFR", help="algorithm to run")
    parser.add_argument("--bl_weight", type=str, default="exponential", help="baseline weighting")
    parser.add_argument("--bl_reset", type=str2bool, default="False", help="reset baseline")
    parser.add_argument("--save_path_prefix", type=str, default="/home/zengzekeng/stoch_optim/results/mnt_results/2v1Kuhn", help="save path prefix")
    args = parser.parse_args()
    
    players = 3
    rank = args.rank
    maxbets = args.maxbets
    random_seed = args.random_seed
    algo = args.algo

    global_n_iter = args.n_iter
    num_traj_per_iter = args.n_traj

    lr = args.lr
    expl = args.expl
    tau = args.tau

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    assert rank > 3 and rank < 14
    deck = [Card(14-r,1) for r in range(rank)]
    betsize = 1 
    rounds = [RoundInfo(holecards=1,boardcards=0,betsize=betsize,maxbets=[maxbets,maxbets,maxbets])]
    ante = 1 
    blinds = None 
    gamerules = GameRules(players, deck, rounds, ante, blinds, handeval=kuhn_eval, infoset_format=leduc_format) 
    teams=[[0,2],[1]]

    if algo == "MCCFR":
        bl_type = "na"
    elif algo == "MCCFR-infoset":
        bl_type = "infoset"
    elif algo == "MCCFR-history":
        bl_type = "history"
    else:
        raise ValueError(f"unknown algorithm: {algo}")
    bl_weighting = args.bl_weight
    bl_reset = args.bl_reset

    save_path = args.save_path_prefix + f"/r{rank}m{maxbets}/seed{random_seed}/{bl_type}_traj{num_traj_per_iter}_tau{tau}_lr{lr}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    solver = VROutcomeSamplingSolver(gamerules, teams, lr=lr, tao=tau, sample_traj_num=num_traj_per_iter, 
                                     baseline_type=bl_type, baseline_weighting=bl_weighting, baseline_reset=bl_reset, 
                                     save_path=save_path, seed=random_seed, 
                                     )

    depth = 0
    for s in solver.tree.information_sets:
        depth = max(depth, len(s))
    depth -= 3
    
    print("Begin training.")
    print("scenario: 2v1Kuhn_r{}m{}: #terminal={}, #infoset={}, depth={}".format(rank, maxbets, solver.tree.num_terminal_nodes, len(solver.tree.information_sets), depth))
    print("random_seed: {}".format(random_seed))
    print("algorithm: {}. bl_type={}, bl_weighting={}, bl_reset={}".format(algo, bl_type, bl_weighting, bl_reset))
    print("hyperparameters: n_iter={}, n_traj_per_iter={}, lr={}, expl={}, tau={}".format(global_n_iter, num_traj_per_iter, lr, expl, tau))
    print("save_path: {}".format(save_path))

    eval_freq = 100
    solver.run(num_iterations=global_n_iter, eval_freq=eval_freq)
    _, ev_br = solver.profile.best_response(br_players=list(range(solver.rules.players)))
    nashconv = sum(ev_br)
    print(f"iter {global_n_iter}: num_infoset_visited={solver.num_infoset_visited}, nashconv={nashconv}")
    np.save(save_path + "/results.npy", np.array(solver.results))

    print("Finish training.")
    print("scenario: 2v1Kuhn_r{}m{}: #terminal={}, #infoset={}, depth={}".format(rank, maxbets, solver.tree.num_terminal_nodes, len(solver.tree.information_sets), depth))
    print("random_seed: {}".format(random_seed))
    print("algorithm: {}. bl_type={}, bl_weighting={}, bl_reset={}".format(algo, bl_type, bl_weighting, bl_reset))
    print("hyperparameters: n_iter={}, n_traj_per_iter={}, lr={}, expl={}, tau={}".format(global_n_iter, num_traj_per_iter, lr, expl, tau))
    print("save_path: {}".format(save_path))
