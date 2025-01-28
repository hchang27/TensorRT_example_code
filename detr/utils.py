import glob
import os

import IPython
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

e = IPython.embed


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, *episodic_datasets):
        super(CombinedDataset).__init__()
        self.episodic_datasets = episodic_datasets
        self.dataset_sizes = [len(d) for d in self.episodic_datasets]

        # map from global index to dataset index and local index
        self.global_to_local = []
        for i, dataset_size in enumerate(self.dataset_sizes):
            self.global_to_local.extend([(i, j) for j in range(dataset_size)])

    def __len__(self):
        return sum(self.dataset_sizes)

    def __getitem__(self, index):
        dataset_idx, local_idx = self.global_to_local[index]
        return self.episodic_datasets[dataset_idx][local_idx]


class EpisodicDataset(torch.utils.data.Dataset):
    max_action_size = 600

    def __init__(self, episode_ids, dataset_dir, num_cameras, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.num_cameras = num_cameras
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs["sim"]
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/prop"][start_ts]

            all_cam_images = []
            img_idxs = np.arange(start_ts - self.num_cameras + 1, start_ts + 1)
            img_idxs.clip(0, episode_len - 1)
            for i in img_idxs:
                all_cam_images.append(root["/observations/images/ego"][i])
            

            # for cam_name in self.camera_names:
            #     image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]

            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][max(0, start_ts - 1) :]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        self.is_sim = is_sim

        padded_action = np.zeros((self.max_action_size, 12), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_action_size)
        is_pad[action_len:] = 1

        # new axis for different cameras
        # all_cam_images = []
        # for cam_name in self.camera_names:
        #     all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, episode_ids):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in episode_ids:
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/prop"][()]
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)[None, ...]
    action_std = all_action_data.std(dim=0, keepdim=True)[None, ...]
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)[None, ...]
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)[None, ...]
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    return stats


def get_norm_stats_combined(dataset_dirs):
    """
    Gets the stats for ALL trajs in the dataset_dirs
    """
    all_qpos_data = []
    all_action_data = []
    for dataset_dir in dataset_dirs:
        all_files = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
        print(f"Found {len(all_files)} episodes")
        all_episodes = [int(f.split("/")[-1].split("_")[1].split(".")[0]) for f in all_files]
        for episode_idx in all_episodes:
            dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/prop"][()]
                action = root["/action"][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)[None, ...]
    action_std = all_action_data.std(dim=0, keepdim=True)[None, ...]
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)[None, ...]
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)[None, ...]
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, **kwargs):
    import glob

    print(f"\nData from: {dataset_dir}\n")

    all_files = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
    print(f"Found {len(all_files)} episodes")

    missing = set(range(num_episodes)) - set([int(f.split("/")[-1].split("_")[1].split(".")[0]) for f in all_files])
    all_episodes = set(range(num_episodes)) - missing

    train_ratio = 0.8
    shuffled_indices = np.random.permutation(list(all_episodes))
    train_indices = shuffled_indices[: int(train_ratio * len(shuffled_indices))]
    val_indices = shuffled_indices[int(train_ratio * len(shuffled_indices)) :]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, all_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def load_data_combined(dataset_dirs, num_cameras, batch_size_train, batch_size_val, **kwargs):
    import glob

    # obtain normalization stats for qpos and action
    # qpos here is the observation (, 753)
    norm_stats = get_norm_stats_combined(dataset_dirs)

    trainsets = []
    valsets = []

    for dataset_dir in dataset_dirs:
        all_files = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
        print(f"Found {len(all_files)} episodes")

        all_episodes = [int(f.split("/")[-1].split("_")[1].split(".")[0]) for f in all_files]

        train_ratio = 0.8
        shuffled_indices = np.random.permutation(list(all_episodes))
        train_indices = shuffled_indices[: int(train_ratio * len(shuffled_indices))]
        val_indices = shuffled_indices[int(train_ratio * len(shuffled_indices)) :]

        # construct dataset and dataloader
        train_dataset = EpisodicDataset(train_indices, dataset_dir, num_cameras, norm_stats)
        val_dataset = EpisodicDataset(val_indices, dataset_dir, num_cameras, norm_stats)

        trainsets.append(train_dataset)
        valsets.append(val_dataset)

    combined_train_dataset = CombinedDataset(*trainsets)
    combined_val_dataset = CombinedDataset(*valsets)

    train_dataloader = DataLoader(
        combined_train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1
    )
    val_dataloader = DataLoader(
        combined_val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)