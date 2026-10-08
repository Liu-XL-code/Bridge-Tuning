import os
import warnings

import torch
import torch.multiprocessing as mp
import multiprocessing
import sys

# 设置matplotlib使用非GUI后端，避免多进程冲突
import matplotlib
matplotlib.use('Agg')

from lib.utils import set_seed, dist_setup, get_conf
import lib.trainers as trainers

from setproctitle import setproctitle
setproctitle('main')

# 默认配置文件
if len(sys.argv) == 1:
    sys.argv = ['', 'configs/stage2_bridge_tuning.yaml']

# 提取备注（配置文件之后的所有参数）
remarks = sys.argv[2:] if len(sys.argv) > 2 else []

# 移除备注，只保留配置文件
if len(sys.argv) > 2:
    sys.argv = sys.argv[:2]

def main():
    saved_remarks = remarks

    args = get_conf()

    # Record optional run remarks.
    args.run_remarks = '_'.join(saved_remarks) if saved_remarks else None
    if args.run_remarks:
        print(f"Run remarks: {args.run_remarks}")

    args.test = False

    # Prefer the top-level seed, then fall back to experiment.seed.
    seed = getattr(args, 'seed', None)
    if seed is None and hasattr(args, 'experiment') and args.experiment:
        seed = args.experiment.get('seed')
    set_seed(seed)

    if not args.multiprocessing_distributed and args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                nprocs=ngpus_per_node,
                args=(args,))
    else:
        print("single process")
        main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    ngpus_per_node = args.ngpus_per_node
    dist_setup(ngpus_per_node, args)

    # init trainer
    trainer_class = getattr(trainers, f'{args.trainer_name}', None)#导入训练模型
    assert trainer_class is not None, f"Trainer class {args.trainer_name} is not defined"
    trainer = trainer_class(args)



    # create model
    trainer.build_model()
    # create optimizer
    trainer.build_optimizer()
    # resume training
    if args.resume:
        trainer.resume()
    #因为是多个中心的微调所以不在此处构建数据加载器

    trainer.run()



if __name__ == '__main__':
    multiprocessing.set_start_method("fork")
    main()
