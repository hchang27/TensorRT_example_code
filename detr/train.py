import IPython
import numpy as np
import torch
from copy import deepcopy
from params_proto import ParamsProto, Proto
from tqdm import tqdm

from detr.policy import get_n_act_policy
from detr.utils import set_seed, compute_dict_mean, detach_dict, load_data_combined

e = IPython.embed


class TrainArgs(ParamsProto):
    datasets: str = Proto(env="$LUCIDSIM_DATASETS")
    datasets = "/data"
    dataset_prefix = [
        # "lucidsim/lucidsim/corl/baseline_datasets/depth_v1/extensions_gaps_many_v3/datasets/dagger_0",
        # "lucidsim/lucidsim/corl/baseline_datasets/depth_v1/extensions_gaps_many_act_no_scandots_v1/datasets/dagger_1",
        # "lucidsim/lucidsim/corl/baseline_datasets/depth_v1/extensions_gaps_many_act_no_scandots_v1/datasets/dagger_2",
        "haoran/local_datasets/test_hdf5"
    ]

    # ckpt params
    load_checkpoint = "/home/exx/Downloads/ckpts/debug_100_no_scandots_v1/dagger_1_cont/policy_last.ckpt"
    load_checkpoint = None
    # if false, load from logger
    local_load = True
    img_memory_size = 10

    checkpoint_interval = 100

    # train params
    batch_size_train = 128
    batch_size_val = 128

    num_epochs = 2000
    seed = 100


def main(_deps=None, **deps):
    from ml_logger import logger
    from lucidsim_experiments import RUN

    TrainArgs._update(_deps, **deps)

    try:
        RUN._update(_deps)
        logger.configure(RUN.prefix)
    except KeyError:
        pass

    dataset_dirs = [f"{TrainArgs.datasets}/{prefix}" for prefix in TrainArgs.dataset_prefix]

    train_dataloader, val_dataloader, stats, _ = load_data_combined(
        dataset_dirs=dataset_dirs,
        num_cameras=TrainArgs.img_memory_size,
        batch_size_train=TrainArgs.batch_size_train,
        batch_size_val=TrainArgs.batch_size_val,
    )

    logger.save_pkl(stats, "dataset_stats.pkl")

    best_ckpt_info = train_bc(train_dataloader, val_dataloader)
    return best_ckpt_info


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(image_data, qpos_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader):
    from ml_logger import logger

    logger.job_started(TrainArgs=vars(TrainArgs))

    # fmt: off
    logger.log_text("""
            charts:
            - yKeys: ["train/loss", "eval/loss"]
              xKey: epoch
              yDomain: [0, 1]
            - yKeys: ["train/l1", "eval/l1"]
              xKey: epoch
              yDomain: [0, 1]
            - yKeys: ["train/l2", "eval/l2"]
              xKey: epoch
              yDomain: [1, 3]
            - yKeys: ["train/kl", "eval/kl"]
              xKey: epoch
              yDomain: [0, 1]
            """, dedent=True, filename=".charts.yml", overwrite=True)
    # fmt: on

    logger.upload_file(__file__)

    print("Training started", logger.get_dash_url())

    set_seed(TrainArgs.seed)
    policy = get_n_act_policy(TrainArgs.img_memory_size)
    
    if TrainArgs.load_checkpoint is not None:
        load_fn = torch.load if TrainArgs.local_load else logger.load_torch
        policy.load_state_dict(load_fn(TrainArgs.load_checkpoint))

    policy.cuda()
    optimizer = policy.configure_optimizers()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(TrainArgs.num_epochs)):
        print(f"\nEpoch {epoch}")
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        logger.print(summary_string)
        val_metrics = {f"eval/{k}": v.item() for k, v in epoch_summary.items()}

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)])
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        logger.print(summary_string)
        train_metrics = {f"train/{k}": v.item() for k, v in epoch_summary.items()}

        logger.log_metrics({**train_metrics, **val_metrics})
        logger.log_metrics_summary(key_values={"epoch": epoch})

        if TrainArgs.checkpoint_interval is not None and epoch % TrainArgs.checkpoint_interval == 0:
            logger.torch_save(policy.state_dict(), f"checkpoints/policy_epoch_{epoch}_seed_{TrainArgs.seed}.pt")

    logger.torch_save(policy.state_dict(), "checkpoints/policy_last.pt")

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    logger.torch_save(best_state_dict, f"checkpoints/policy_best_epoch_{best_epoch}.pt")
    logger.print(f"Training finished:\nSeed {TrainArgs.seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")

    return best_ckpt_info


if __name__ == "__main__":
    main()
