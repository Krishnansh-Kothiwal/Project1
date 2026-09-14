import argparse
import os,json, sys
import numpy as np
# single gpu    

try:
    if os.system('nvidia-smi -q -d Memory > tmp.txt 2> nul') == 0 and os.path.exists('tmp.txt'):
        lines = [x for x in open('tmp.txt', 'r').readlines() if 'Free' in x]
        if lines:
            memory_gpu = [int(x.split()[2]) for x in lines]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
        if os.path.exists('tmp.txt'):
            os.remove('tmp.txt')
except Exception:
    if os.path.exists('tmp.txt'):
        os.remove('tmp.txt')

import torch
import utils

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    utils.print_logo(subtitle="Maintained by Research Center for Applied Mathematics and Machine Intelligence, Zhejiang Lab")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SimpleDoorKey", help="SimpleDoorKey, KeyInBox, RandomBoxKey, ColoredDoorKey") 
    parser.add_argument("--save_name", type=str, required=True, help="path to folder containing policy and run details")
    parser.add_argument("--logdir", type=str, default="./log/")          # Where to log diagnostics to
    parser.add_argument("--record", default=False, action='store_true') 
    parser.add_argument("--seed", default=None)
    parser.add_argument("--ask_lambda", type=float, default=0.01, help="weight on communication penalty term")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
    parser.add_argument("--n_itr", type=int, default=1000, help="Number of iterations of the learning algorithm")
    parser.add_argument("--policy",   type=str, default='ppo')
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--traj_per_itr", type=int, default=10)
    parser.add_argument("--show", default=False, action='store_true')
    parser.add_argument("--test_num", type=int, default=100)
    
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--run_seed_list", type=int, nargs="*", default=[0])
    parser.add_argument("--resume", default=False, action="store_true", help="resume from latest checkpoint")

    if sys.argv[1] == 'eval':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
 
        output_dir = os.path.join(args.logdir, args.policy, args.task, args.save_name)
        
        policy = torch.load(output_dir + "/acmodel.pt")
        policy.eval()
        eval = utils.Eval(args,policy)
        eval.eval_policy(args.test_num)
    elif sys.argv[1] == 'eval_RL':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        output_dir = os.path.join(args.logdir, args.policy, args.task, args.save_name)
   
        policy = torch.load(output_dir + "/acmodel.pt")
        policy.eval()
        eval = utils.Eval(args,policy)
        eval.eval_RL_policy(args.test_num)

    elif sys.argv[1] == 'train':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        from env.Game import Game

        base_save_name = args.save_name
        for i in args.run_seed_list:
            setup_seed(i)
            if len(args.run_seed_list) > 1:
                args.save_name = base_save_name + str(i)
            else:
                args.save_name = base_save_name
            game = Game(args, run_seed=i)
            game.reset()
            game.train()
    elif sys.argv[1] == 'train_RL':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        from env.Game_RL import Game_RL

        base_save_name = args.save_name
        for i in args.run_seed_list:
            setup_seed(i)
            if len(args.run_seed_list) > 1:
                args.save_name = base_save_name + str(i)
            else:
                args.save_name = base_save_name
            game = Game_RL(args)
            game.reset()
            game.train()
    elif sys.argv[1] == 'baseline':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        eval = utils.Eval(args)
        eval.eval_baseline(args.test_num)
    elif sys.argv[1] == 'random':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        eval = utils.Eval(args)
        eval.eval_policy(args.test_num)
    elif sys.argv[1] == 'always':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        eval = utils.Eval(args)
        eval.eval_always_ask(args.test_num)
    else:
        print("Invalid option '{}'".format(sys.argv[1]))

