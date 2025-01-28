import h5py
import numpy as np
import os
import pickle
import torch
from PIL import Image
from ml_logger import logger
from tqdm.contrib.concurrent import thread_map

from cxx.modules.parkour_actor import get_parkour_teacher_policy

# dataset = "/home/exx/datasets/lucidsim/lucidsim/lucidsim/corl/baseline_datasets/depth_v1/extensions_gaps_many_v3/datasets/dagger_0"
dataset = "/home/exx/datasets/lucidsim/lucidsim/lucidsim/corl/baseline_datasets/depth_v1/extensions_gaps_many_act_no_scandots_v1/datasets/dagger_2"
num_episodes = 1000
start_ep = 0
camera_names = ["ego"]

from cxx.modules.parkour_actor import PolicyArgs

# note: super important to set this to False, o/w the policy will assume vision mode
PolicyArgs.use_camera = False

sd = logger.torch_load(
    "/lucid-sim/lucid-sim/baselines/launch_gains/2024-03-20/04.03.35/go1/300/20/0.5/checkpoints/model_last.pt", map_location="cuda"
)
teacher = get_parkour_teacher_policy()
teacher.load_state_dict(sd)
teacher.to("cuda")
teacher.eval()


def process_episode(episode_idx):
    data_dict = {
        "/observations/prop": [],
        "/action": [],
    }
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    traj_file = os.path.join(dataset, f"{episode_idx:04d}_trajectory.pkl")
    try:
        with open(traj_file, "rb") as f:
            trajectory = pickle.load(f)
            for i, step in enumerate(trajectory["obs"]):
                prop_obs = step[0, :]
                data_dict["/observations/prop"].append(prop_obs)
                with torch.no_grad():
                    action = teacher(None, torch.from_numpy(step).cuda())[0].cpu().numpy()
                data_dict["/action"].append(action[0])
                img_path = f"{dataset}/{episode_idx:04d}_ego_views/render_depth/frame_{i:05d}_4x.png"
                try:
                    img = Image.open(img_path)
                    img = np.array(img)
                    img = np.stack([img, img, img], axis=-1)
                    for cam_name in camera_names:
                        data_dict[f"/observations/images/{cam_name}"].append(img)
                except:
                    print(f"Failed to load {img_path}")
                    return None  # Fail this episode
    except Exception as e:
        print(f"Error processing trajectory {traj_file}: {e}")
        return None

    max_timesteps = len(trajectory["obs"])
    dataset_path = os.path.join(dataset, f"episode_{episode_idx}")
    try:
        with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
            root.attrs["sim"] = True
            obs = root.create_group("observations")
            image = obs.create_group("images")
            for cam_name in camera_names:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, 180, 320, 3),
                    dtype="uint8",
                    chunks=(1, 180, 320, 3),
                )
            prop = obs.create_dataset("prop", (max_timesteps, 753))
            action = root.create_dataset("action", (max_timesteps, 12))

            for name, array in data_dict.items():
                root[name][...] = array
        return True
    except Exception as e:
        print(f"Error saving HDF5 for episode {episode_idx}: {e}")
        return None


# results = thread_map(process_episode, range(start_ep, num_episodes), max_workers=16, desc="Processing Episodes")

# failed_episodes = [i for i, result in enumerate(results, start=start_ep) if not result]
# if failed_episodes:
#     print(f"Failed episodes: {failed_episodes}")
# else:
#     print("All episodes processed successfully.")
