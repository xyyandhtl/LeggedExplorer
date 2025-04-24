import torch
from .base import POLICY_WEIGHTS_PATH
from torch import distributions as torchd
from torch.nn import functional as F


class WMPData:
    def __init__(self, num_envs, device):
        self.device = device
        self.num_envs = num_envs
        self.use_camera = True

        self.prop_dim = 33
        self.num_actions = 12

        self.history_length = 5
        self.depth_resized = tuple((64, 64))  # cfg["depth_resized"]  # 因解析为字符，而暂未起作用

        self.wm_update_interval = 5
        self.wm_feature_dim = 512
        self.wm_num_actions = self.num_actions * self.wm_update_interval

        self.trajectory_history = torch.zeros(size=(num_envs, self.history_length, self.prop_dim + self.num_actions - 3), device=device)
        self.obs_without_command = None
        self.wm_latent = self.wm_action = None

        self.wm_obs = {}
        self.wm_is_first = torch.ones(num_envs, device=device)

        self.wm_action_history = torch.zeros(size=(num_envs, self.wm_update_interval, self.num_actions), device=device)
        self.wm_feature = torch.zeros((num_envs, self.wm_feature_dim), device=self.device)
        self.wm_W = torch.nn.Parameter(torch.zeros((1, 512), device=torch.device(self.device)),requires_grad=True,)
        self.global_counter = 1


wmp_data = None

def init_wmp_data(num_envs, device):
    global wmp_data
    wmp_data = WMPData(num_envs, device)

def load_policy_wmp(robot_name, device='cuda:0'):
    if wmp_data is None:
        init_wmp_data(num_envs=1, device=device)

    path1 = f'{POLICY_WEIGHTS_PATH}/wmp_loco/{robot_name}/policy.jit'
    path2 = f'{POLICY_WEIGHTS_PATH}/wmp_loco/{robot_name}/world_model.jit'
    policy_model = torch.jit.load(path1, map_location=device)
    world_model = torch.jit.load(path2, map_location=device)
    # print(f"[policy_model_jit] : {policy_model}")
    # print(f"[world_model_jit] : {world_model}")
    wmp_data.WM = world_model

    def act_forward(command, history, wm_feature):
        latent_vector = policy_model.history_encoder(history)
        wm_latent_vector = policy_model.wm_feature_encoder(wm_feature)

        concat_observations = torch.concat((latent_vector, command, wm_latent_vector), dim=-1)
        actions_mean = policy_model.actor(concat_observations)
        return actions_mean

    def policy(obs):
        command = obs[:, 6:9]
        history = wmp_data.trajectory_history.flatten(1).to(device)  # (num_envs, 5 * 42)
        actions = act_forward(command.detach(), history.detach(), wmp_data.wm_feature.detach())

        wmp_data.global_counter += 1
        return actions

    return policy


def world_model_data_init(obs):
    """
    Args:
        obs: (num_envs, 45)
        wmp_data: WMPData
    """
    wmp_data.obs_without_command = torch.concat((obs[:, :6], obs[:, 9:]), dim=1)  # (num_envs, 42)
    wmp_data.trajectory_history = torch.concat((wmp_data.trajectory_history[:, 1:], wmp_data.obs_without_command.unsqueeze(1)), dim=1)

    wmp_data.wm_obs = {
        "prop": obs[:, : wmp_data.prop_dim],
        "is_first": wmp_data.wm_is_first,
    }
    if wmp_data.use_camera:
        wmp_data.wm_obs["image"] = torch.zeros(((wmp_data.num_envs,) + wmp_data.depth_resized + (1,)), device=wmp_data.device)

def world_model_encoder_forward(obs):
    outputs = []
    if wmp_data.use_camera:
        outputs.append(wmp_data.WM.encoder._cnn(obs["image"]))
    else:
        outputs.append(torch.zeros((wmp_data.num_envs, 4096), device=wmp_data.device))

    outputs.append(wmp_data.WM.encoder._mlp(obs["prop"]))
    outputs = torch.cat(outputs, -1)
    return outputs

def world_model_get_dist(state, dtype=None):
    logit = state["logit"]  # (num_envs, 32, 32)
    # tools.OneHotDist(): 创建一个基于logits的one-hot分类分布, 每个32x32矩阵表示32个随机状态维度，每个维度有32个类别的分类分布
    # torchd.independent.Independent(..., 1): 将最后1个维度（即32个类别）视为事件维度, 使得最终分布有32个独立事件（对应32个随机状态维度）
    dist = torchd.independent.Independent(
        OneHotDist(logit, unimix_ratio=0.01), 1
    )
    return dist

def world_model_rssm_suff_stats_layer(name, x):
    if name == "ims":
        x = wmp_data.WM.dynamics._imgs_stat_layer(x)
    elif name == "obs":
        x = wmp_data.WM.dynamics._obs_stat_layer(x)
    else:
        raise NotImplementedError
    logit = x.reshape(list(x.shape[:-1]) + [32, 32])
    return {"logit": logit}

def world_model_get_stoch(deter):
    x = wmp_data.WM.dynamics._img_out_layers(deter)
    stats = world_model_rssm_suff_stats_layer("ims", x)  # {"logit": (num_envs, 32, 32)}
    dist = world_model_get_dist(stats)
    return dist.mode()

def world_model_initial(batch_size):
    deter = torch.zeros(batch_size, 512).to(wmp_data.device)
    state = dict(
        logit=torch.zeros((batch_size, 32, 32)).to(wmp_data.device),
        stoch=torch.zeros((batch_size, 32, 32)).to(wmp_data.device),
        deter=deter,
    )

    state["deter"] = torch.tanh(wmp_data.wm_W).repeat(batch_size, 1)
    state["stoch"] = world_model_get_stoch(state["deter"])
    return state

def world_model_rssm_cell(inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = wmp_data.WM.dynamics._cell.layers(torch.cat([inputs, state], -1))
    reset, cand, update = torch.split(parts, [512] * 3, -1)
    reset = torch.sigmoid(reset)
    cand = torch.tanh(reset * cand)
    update = torch.sigmoid(update + -1)
    output = update * cand + (1 - update) * state
    return output, [output]

def world_model_img_step(prev_state, prev_action, sample=True):
    """
    根据 前一时间步的 状态 和 action历史 预测 先验状态（下一时间步的 随机状态、确定性状态、其他统计量）

    Args:
        prev_state (dict): 世界模型前一时间步潜在状态，包含:
            - stoch: 随机状态 (num_envs, 32, 32)
            - deter: 确定性状态 (num_envs, 512)
            - logit: 状态分布参数 (num_envs, 32, 32)
        prev_action (tensor): 前5个时间步的 action (num_envs, 5*12)
    """
    prev_stoch = prev_state["stoch"]  # (num_envs, 32, 32)
    shape = list(prev_stoch.shape[:-2]) + [32 * 32]  # (num_envs, 32*32)
    prev_stoch = prev_stoch.reshape(shape)  # (num_envs, stoch, discrete_num) ==> (num_envs, stoch * discrete_num)

    x = torch.cat([prev_stoch, prev_action], -1)  # (num_envs, 32*32 + 5*12)

    # 经输入层处理（输入 前一时间步的随机状态 + 前5个时间步的action）
    x = wmp_data.WM.dynamics._img_in_layers(x)  # (num_envs, 512)

    # 经 循环网络（GRU单元）处理（输入 输入层编码特征 + 前一时间步的确定性状态）
    for _ in range(1):  # rec depth is not correctly implemented
        deter = prev_state["deter"]  # (num_envs, 512)
        # (num_envs, hidden), (num_envs, deter) -> (num_envs, deter), (num_envs, deter)
        x, deter = world_model_rssm_cell(x, [deter])  # 经GRU单元更新后的 输入层编码特征 x 和  确定性状态 deter
        deter = deter[0]  # Keras wraps the state in a list.

    # 经输出层处理
    x = wmp_data.WM.dynamics._img_out_layers(x)  # (num_envs, 512) -> (num_envs, 512)

    # 获取 状态分布 logit
    stats = world_model_rssm_suff_stats_layer("ims", x)  # {"logit": (num_envs, 32, 32)}
    # 从 状态分布中 获取 随机状态
    if sample:  # 采样
        stoch = world_model_get_dist(stats).sample()
    else:  # 众数
        stoch = world_model_get_dist(stats).mode()

    # 构建并返回先验状态
    prior = {"stoch": stoch, "deter": deter, **stats}
    return prior

def world_model_rssm_forward(prev_state, prev_action, embed, is_first, sample=True):
    # 1. 处理 世界模型 状态 和 action
    if prev_state == None or torch.sum(is_first) == len(
            is_first):  # 如果是全新 episode，则初始化 prev_state、prev_action 为 0 tensor
        prev_state = world_model_initial(len(is_first))
        prev_action = torch.zeros((len(is_first), wmp_data.wm_num_actions)).to(wmp_data.device)
    elif torch.sum(is_first) > 0:  # 不是，则保留 不是第一次的env的 action 和 状态
        is_first = is_first[:, None]  # (num_envs, 1)
        # (1) 仅重置 是第一次 的 env 的 prev_action
        prev_action *= 1.0 - is_first
        # (2) 仅重置 是第一次 的 env 的 prev_state
        init_state = world_model_initial(len(is_first))  # 重置所有 env 的 init_state
        for key, val in prev_state.items():
            # 调整is_first的维度以匹配状态维度 (num_envs, 1, 1)
            is_first_r = torch.reshape(
                is_first,
                is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
            )
            prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
            )

    # 2. 通过 世界模型的前一时间步的 状态 和 action历史 预测 先验状态（当前时间步的 随机状态、确定性状态、logit）
    prior = world_model_img_step(prev_state, prev_action)

    # 3. 结合 先验确定性状态 + 观测编码特征 更新 后验随机性状态
    x = torch.cat([prior["deter"], embed], -1)  # (num_envs, 512+5120)
    # 经 观测输出层处理
    x = wmp_data.WM.dynamics._obs_out_layers(x)  # (num_envs, 512)
    # 获取 状态分布 logit
    stats = world_model_rssm_suff_stats_layer("obs", x)  # {"logit": (num_envs, 32, 32)}
    if sample:
        stoch = world_model_get_dist(stats).sample()
    else:
        stoch = world_model_get_dist(stats).mode()

    # 4. 构建 后验状态（后验随机状态、先验确定性状态、logit）
    post = {"stoch": stoch, "deter": prior["deter"], **stats}
    return post, prior

def update_wm(actions, obs, depth, reset_env_ids=[]):
    if wmp_data.global_counter % wmp_data.wm_update_interval == 0:
        if wmp_data.use_camera:
            wmp_data.wm_obs["image"][0] = depth.unsqueeze(-1).to(wmp_data.device)
            # print(f'[depth] : {depth}')

        # encoder
        wm_embed = world_model_encoder_forward(wmp_data.wm_obs)
        # print(f"[update_wm | wm_embed shape] : {wm_embed.shape}")
        # obs_step
        wmp_data.wm_latent, _ = world_model_rssm_forward(wmp_data.wm_latent, wmp_data.wm_action, wm_embed, wmp_data.wm_obs["is_first"],
                                                 sample=True)
        wmp_data.wm_feature = wmp_data.wm_latent["deter"]
        # print(f"[update_wm | wm_feature shape] : {wmp_data.wm_feature.shape}")
        wmp_data.wm_is_first[:] = 0

    # 更新世界模型的输入
    wmp_data.wm_action_history = torch.concat((wmp_data.wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
    wmp_data.wm_obs = {
        "prop": obs[:, :wmp_data.prop_dim],
        "is_first": wmp_data.wm_is_first,
    }
    if wmp_data.use_camera:
        wmp_data.wm_obs["image"] = torch.zeros(((wmp_data.num_envs,) + wmp_data.depth_resized + (1,)), device=wmp_data.device)

    # when env need to be reset
    if len(reset_env_ids) > 0:
        wmp_data.wm_action_history[reset_env_ids, :] = 0
        wmp_data.wm_is_first[reset_env_ids] = 1

    wmp_data.wm_action = wmp_data.wm_action_history.flatten(1)

    # process trajectory history
    wmp_data.obs_without_command = torch.concat((obs[:, :6], obs[:, 9:]), dim=1)
    wmp_data.trajectory_history = torch.concat((wmp_data.trajectory_history[:, 1:], wmp_data.obs_without_command.unsqueeze(1)), dim=1)



class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample