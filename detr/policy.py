import IPython
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from params_proto import PrefixProto
from torch.nn import functional as F

from detr.models.backbone import build_backbone
from detr.models.detr_vae import build_encoder, DETRVAE
from detr.models.transformer import build_transformer

act_args_preset = {
    "lr": 1e-04,
    "num_queries": 100,  # chunk size
    "kl_weight": 10,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "lr_backbone": 1e-05,
    "backbone": "resnet18",
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "num_cameras": 1,
}


def get_default_act_policy():
    ACTArgs._update(**act_args_preset)
    return ACTPolicy()


def get_n_act_policy(n=10):
    assert type(n) == int and n > 0, f"num_cameras must be a positive integer, got {n}"
    ACTArgs._update(**act_args_preset)
    ACTArgs.num_cameras = n
    return ACTPolicy()


class ACTArgs(PrefixProto):
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    batch_size: int = 2
    weight_decay: float = 1e-4
    epochs: int = 300
    lr_drop: int = 200
    clip_max_norm: float = 0.1
    backbone: str = "resnet18"
    dilation: bool = False
    position_embedding: str = "sine"
    num_cameras: int = 1
    enc_layers: int = 4
    dec_layers: int = 6
    dim_feedforward: int = 2048
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 400
    pre_norm: bool = False
    masks: bool = False
    eval: bool = False
    onscreen_render: bool = False
    ckpt_dir: str = None
    policy_class: str = None
    task_name: str = None
    seed: int = None
    num_epochs: int = None
    kl_weight: float = None
    chunk_size: int = None
    temporal_agg: bool = False


e = IPython.embed


class ACTPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        state_dim = 12  # TODO hardcode

        # From state
        # backbone = None # from state for now, no need for conv nets
        # From image
        backbones = []
        backbone = build_backbone(
            hidden_dim=ACTArgs.hidden_dim,
            position_embedding=ACTArgs.position_embedding,
            lr_backbone=ACTArgs.lr_backbone,
            backbone=ACTArgs.backbone,
            dilation=ACTArgs.dilation,
            masks=ACTArgs.masks,
        )

        backbones.append(backbone)

        transformer = build_transformer(
            hidden_dim=ACTArgs.hidden_dim,
            dropout=ACTArgs.dropout,
            nheads=ACTArgs.nheads,
            dim_feedforward=ACTArgs.dim_feedforward,
            pre_norm=ACTArgs.pre_norm,
            enc_layers=ACTArgs.enc_layers,
            dec_layers=ACTArgs.dec_layers,
        )

        encoder = build_encoder(
            hidden_dim=ACTArgs.hidden_dim,
            dropout=ACTArgs.dropout,
            nheads=ACTArgs.nheads,
            dim_feedforward=ACTArgs.dim_feedforward,
            pre_norm=ACTArgs.pre_norm,
            enc_layers=ACTArgs.enc_layers,
        )

        model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            num_queries=ACTArgs.num_queries,
            num_cameras=ACTArgs.num_cameras,
        )

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters / 1e6,))

        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": ACTArgs.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=ACTArgs.lr, weight_decay=ACTArgs.weight_decay)

        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = ACTArgs.kl_weight
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, image, qpos, actions=None, is_pad=None, **_):
        # note: at this point qpos contains privileged info to be used by the encoder. will be masked out before input to the decoder
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            assert False
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

            flattened_actions = actions.reshape(-1, 12)
            flattened_a_hat = a_hat.reshape(-1, 12)
            flattened_is_pad = is_pad.reshape(-1)

            err = (flattened_actions - flattened_a_hat).norm(p=2, dim=1)
            err = err[~flattened_is_pad]

            max_err = err.max()
            l2 = err.mean()

            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["l2"] = l2
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            loss_dict["max_l2"] = max_err

            # max_l2
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld