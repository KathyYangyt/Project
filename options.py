import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--global_epochs', type=int, default=1)
    parser.add_argument('--local_epochs', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--fraction', type=float, default=0.1)    
    parser.add_argument('--local_batch', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)    
    parser.add_argument('--dataset', type=str, default='Cmnist')
    parser.add_argument('--iid', type=str, default='non_iid_unequal')
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0,help='set 0 for gpu,set -1 for cpu')

    args = parser.parse_args()
    return args