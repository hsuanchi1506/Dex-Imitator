from __future__ import annotations

import os
import random
from enum import Enum
from itertools import cycle
from time import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from ...utils import torch_jit_utils as torch_jit_utils
from bps_torch.bps import bps_torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import normalize_angle, quat_conjugate, quat_mul
from copy import deepcopy
import math
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory

# from main.dataset.favor_dataset_dexhand import FavorDatasetDexHand
from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH
from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
from main.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass
from main.dataset.transform import aa_to_quat, aa_to_rotmat, quat_to_rotmat, rotmat_to_aa, rotmat_to_quat, rot6d_to_aa
from torch import Tensor
from tqdm import tqdm
from ...asset_root import ASSET_ROOT


from ..core.config import ROBOT_HEIGHT, config
from ...envs.core.sim_config import sim_config
from ...envs.core.vec_task import VecTask
from ...utils.pose_utils import get_mat

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)

def _save_single_rgb(rgb_tensor, path: str):
    """把 Isaac Gym 取回的 COLOR tensor 存成 PNG。"""
    from PIL import Image
    rgb = rgb_tensor.detach().cpu()
    # 轉成 (H,W,3) uint8
    if rgb.ndim == 3 and rgb.shape[-1] == 4:      # (H,W,4)
        rgb = rgb[..., :3]
    elif rgb.ndim == 3 and rgb.shape[0] == 4:     # (4,H,W)
        rgb = rgb.permute(1, 2, 0)[..., :3]
    if rgb.dtype != torch.uint8:
        rgb = rgb.clamp(0, 255).to(torch.uint8)
    Image.fromarray(rgb.numpy(), mode="RGB").save(path)




class DexHandManipBiHEnv(VecTask):

    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
    ):
        self._record = record
        self.cfg = cfg

        self.debug = False

        self.camera_handlers = []   # Trigger _create_envs() to build camera branches
        self.cameras = []           # Store the camera handle for each env
        self.last_depths = None     # Store the most recent depth (list of HxW tensors)



        self.cameras_top = []       # top-down camera，每個 env 一支
        self.last_depths = None     # front depth (HxW tensors)
        self.last_segs = None       # front seg
        self.last_rgb_images = None # front rgb

        # 新增 top 的快取
        self.last_depths_top = None
        self.last_segs_top = None
        self.last_rgb_images_top = None

        # Calibration switches for per-env camera extrinsics: whether to add env origin
        # Keys: view -> env_id -> Optional[bool]; True means add origin, False means not
        self._cam_origin_mode: Dict[str, Dict[int, Optional[bool]]] = {"front": {}, "top": {}}
        self._cam_origin_mode_printed: Dict[str, Dict[int, bool]] = {"front": {}, "top": {}}

        use_quat_rot = self.use_quat_rot = self.cfg["env"]["useQuatRot"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        # self.dexhand_rh_dof_noise = self.cfg["env"]["dexhand_rDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.training = self.cfg["env"]["training"]
        self.dexhand_rh = DexHandFactory.create_hand(self.cfg["env"]["dexhand"], "right")
        self.dexhand_lh = DexHandFactory.create_hand(self.cfg["env"]["dexhand"], "left")

        self.use_pid_control = self.cfg["env"]["usePIDControl"]
        if self.use_pid_control:
            self.Kp_rot = self.dexhand_rh.Kp_rot
            self.Ki_rot = self.dexhand_rh.Ki_rot
            self.Kd_rot = self.dexhand_rh.Kd_rot

            self.Kp_pos = self.dexhand_rh.Kp_pos
            self.Ki_pos = self.dexhand_rh.Ki_pos
            self.Kd_pos = self.dexhand_rh.Kd_pos

        self.cfg["env"]["numActions"] = (
            (1 + 6 + self.dexhand_lh.n_dofs) if use_quat_rot else (6 + self.dexhand_lh.n_dofs)
        ) * (2 if self.cfg["env"]["bimanual_mode"] == "united" else 1)
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.translation_scale = self.cfg["env"]["translationScale"]
        self.orientation_scale = self.cfg["env"]["orientationScale"]

        # a dict containing prop obs name to dump and their dimensions
        # used for distillation
        self._prop_dump_info = self.cfg["env"]["propDumpInfo"]

        # Values to be filled in at runtime
        self.rh_states = {}
        self.lh_states = {}
        self.dexhand_rh_handles = {}  # will be dict mapping names to relevant sim handles
        self.dexhand_lh_handles = {}  # will be dict mapping names to relevant sim handles
        self.objs_handles = {}  # for obj handlers
        self.objs_assets = {}
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed

        self.dataIndices = self.cfg["env"]["dataIndices"]
        # self.dataIndices = [tuple([int(i) for i in idx.split("@")]) for idx in self.dataIndices]
        self.obs_future_length = self.cfg["env"]["obsFutureLength"]
        self.rollout_state_init = self.cfg["env"]["rolloutStateInit"]
        self.random_state_init = self.cfg["env"]["randomStateInit"]

        self.tighten_method = self.cfg["env"]["tightenMethod"]
        self.tighten_factor = self.cfg["env"]["tightenFactor"]
        self.tighten_steps = self.cfg["env"]["tightenSteps"]

        self.rollout_len = self.cfg["env"].get("rolloutLen", None)
        self.rollout_begin = self.cfg["env"].get("rolloutBegin", None)

        assert len(self.dataIndices) == 1 or self.rollout_len is None, "rolloutLen only works with one data"
        assert len(self.dataIndices) == 1 or self.rollout_begin is None, "rolloutBegin only works with one data"

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self.net_cf = None  # contact force
        self._eef_state = None  # end effector state (at grasping point)
        self._ftip_center_state = None  # center of fingertips
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._dexhand_rh_effort_limits = None  # Actuator effort limits for dexhand_r
        self._dexhand_rh_dof_speed_limits = None  # Actuator speed limits for dexhand_r
        self._global_dexhand_rh_indices = None  # Unique indices corresponding to all envs in flattened array

        self.sim_device = torch.device(sim_device)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )
        TARGET_OBS_DIM = (
            128
            + 5
            + (
                3
                + 3
                + 3
                + 4
                + 4
                + 3
                + 3
                + (self.dexhand_rh.n_bodies - 1) * 9
                + 3
                + 3
                + 3
                + 4
                + 4
                + 3
                + 3
                + self.dexhand_rh.n_bodies
            )
            * self.obs_future_length
        ) * 2
        self.obs_dict.update(
            {
                "target": torch.zeros((self.num_envs, TARGET_OBS_DIM), device=self.device),
            }
        )
        obs_space = self.obs_space.spaces
        obs_space["target"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TARGET_OBS_DIM,),
        )
        self.obs_space = spaces.Dict(obs_space)

        # dexhand_r defaults
        # TODO hack here
        # default_pose = self.cfg["env"].get("dexhand_rDefaultDofPos", None)
        default_pose = torch.ones(self.dexhand_rh.n_dofs, device=self.device) * np.pi / 12
        if self.cfg["env"]["dexhand"] == "inspire":
            default_pose[8] = 0.3
            default_pose[9] = 0.01
        self.dexhand_rh_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)
        self.dexhand_lh_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)  # ? TODO check this
        # self.dexhand_rh_default_dof_pos = torch.tensor([-3.5322e-01,  -0.100e-01,  3.2278e-01, -2.51e+00,  1.6036e-01,
        #   2.564e+00, 0.5,  0.10,  0.10], device=self.sim_device)

        # load BPS model
        self.bps_feat_type = "dists"
        self.bps_layer = bps_torch(
            bps_type="grid_sphere", n_bps_points=128, radius=0.2, randomize=False, device=self.device
        )

        obj_verts_rh = self.demo_data_rh["obj_verts"]
        self.obj_bps_rh = self.bps_layer.encode(obj_verts_rh, feature_type=self.bps_feat_type)[self.bps_feat_type]
        obj_verts_lh = self.demo_data_lh["obj_verts"]
        self.obj_bps_lh = self.bps_layer.encode(obj_verts_lh, feature_type=self.bps_feat_type)[self.bps_feat_type]

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def _debug_dump_seg_stats(self, max_envs: int = 1024):
        if not hasattr(self, "last_segs") or self.last_segs is None:
            print("[seg] no seg tensors")
            return
        num = min(len(self.last_segs), max_envs)
        for i in range(num):
            seg = self.last_segs[i]
            uniq = torch.unique(seg)
            print(f"[seg] env {i}: dtype={seg.dtype}, shape={tuple(seg.shape)}, unique={uniq[:50].tolist()} (count={uniq.numel()})")


    def _save_depth_preview_png(
        self,
        out_dir: str,
        step: int,
        max_envs: int = 1024,
        gamma: float = 0.6,

        # Windowing controls (same as before)
        focus_table_and_objects: bool = True,
        start_band_width: float = 0.18,
        max_band_width: float = 0.50,
        coverage_target: float = 0.65,
        coverage_step: float = 0.06,
        pct_low: float = 2.0,
        pct_high: float = 98.0,
        fallback_keep_nearest_pct: float = 0.80,
        fallback_low: float = 2.0,
        fallback_high: float = 98.0,

        # ★ New: remove the influence of the table to avoid eating up the color range
        remove_table: bool = True,
        bottom_rows_frac: float = 0.22,       # Mask the bottom 22% of the image (usually the table given your view)
        # ★ New: use depth to shade
        shade_from_depth: bool = True,
        horizontal_fov_deg: float = 69.4,     # Your camera setting
        min_ambient: float = 0.35,            # Minimum ambient light (avoid pure black)
        light_dir: tuple = (0.4, -0.3, 0.85), # Fake light direction (camera coordinate frame)
    ):
        """
        Save a colored depth preview (near = bright, far = dark) + (optional) fake shading from normals estimated by depth,
        so that the object has a 3D gradient.
        Also optionally mask the bottom (usually the table) first, to avoid the table dominating the color window.
        """
        import os, numpy as np, torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _closest_peak(arr, bins=256):
            hist, edges = np.histogram(arr, bins=bins, range=(arr.min(), arr.max()))
            centers = 0.5 * (edges[:-1] + edges[1:])
            return centers[int(hist.argmax())] if hist.max() > 0 else None

        def _compute_normals_from_depth(depth_m, fovx_deg):
            """
            depth_m: (H, W) in meters, NaN indicates invalid
            Return nmap: (H, W, 3) unit normals (camera coordinates)
            Method: first back-project each pixel to 3D, then take spatial gradients on XYZ, and compute normals via t_u × t_v
            """
            H, W = depth_m.shape
            # From fovx + width/height compute fx, fy
            fovx = np.deg2rad(fovx_deg)
            fx = W / (2.0 * np.tan(fovx / 2.0))
            fovy = 2.0 * np.arctan(np.tan(fovx / 2.0) * (H / W))
            fy = H / (2.0 * np.tan(fovy / 2.0))
            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

            z = depth_m
            # Use NaN for invalid values to avoid contaminating gradients
            z = np.where(np.isfinite(z) & (z > 0), z, np.nan)

            # Build pixel grids
            us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                                np.arange(H, dtype=np.float32))
            X = (us - cx) / fx * z
            Y = (vs - cy) / fy * z
            Z = z

            # Spatial gradients for XYZ (pixel directions)
            dXu, dXv = np.gradient(X)
            dYu, dYv = np.gradient(Y)
            dZu, dZv = np.gradient(Z)

            # Two tangent vectors
            t_u = np.stack([dXu, dYu, dZu], axis=-1)
            t_v = np.stack([dXv, dYv, dZv], axis=-1)

            # Cross product to get normals
            n = np.cross(t_u, t_v)
            n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
            n = np.where(n_norm > 1e-8, n / n_norm, np.array([0, 0, 1], dtype=np.float32))
            # Fill default normals for NaN regions
            n = np.where(np.isfinite(n), n, 0)
            return n.astype(np.float32)

        os.makedirs(out_dir, exist_ok=True)
        # NEW: ensure correct render order for multi-env seg
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.sim_device)

        num = min(len(self.last_depths) if self.last_depths is not None else 0, max_envs)
        for i in range(num):
            d = self.last_depths[i].detach().float().cpu()           # (H, W) meters
            valid = torch.isfinite(d) & (d > 0)
            if not valid.any():
                continue

            H, W = d.shape
            d_np = d.numpy()
            finite = valid.numpy()

            # === Optionally first mask the bottom (table)
            mask = finite.copy()
            if remove_table and H >= 8:
                cut = int(H * (1.0 - float(bottom_rows_frac)))
                mask[cut:, :] = False

            vals_for_window = d_np[mask]
            if vals_for_window.size < 10:
                vals_for_window = d_np[finite]  # In case the object is too small

            # === Windowing (reuse your main-peak/bandwidth + fallback)
            vnear, vfar = None, None
            if focus_table_and_objects and vals_for_window.size > 100:
                peak = _closest_peak(vals_for_window, bins=256)
                if peak is not None and np.isfinite(peak):
                    band = float(start_band_width)
                    total = float(vals_for_window.size)
                    while True:
                        lo = max(vals_for_window.min(), peak - band)
                        hi = min(vals_for_window.max(), peak + band)
                        cov = float(((vals_for_window >= lo) & (vals_for_window <= hi)).sum()) / total
                        if (cov >= coverage_target) or (band >= max_band_width):
                            in_band = vals_for_window[(vals_for_window >= lo) & (vals_for_window <= hi)]
                            if in_band.size >= 50:
                                p_lo = np.percentile(in_band, pct_low)
                                p_hi = np.percentile(in_band, pct_high)
                                vnear = float(min(p_lo, p_hi))
                                vfar  = float(max(p_lo, p_hi))
                            break
                        band += coverage_step

            if (vnear is None) or (vfar is None) or (vfar - vnear) < 1e-6:
                k = max(1, int(vals_for_window.size * float(fallback_keep_nearest_pct)))
                nearest = np.partition(vals_for_window, k - 1)[:k]
                p_lo = np.percentile(nearest, fallback_low)
                p_hi = np.percentile(nearest, fallback_high)
                vnear = float(min(p_lo, p_hi))
                vfar  = float(max(p_lo, p_hi))

            if not np.isfinite(vnear) or not np.isfinite(vfar) or (vfar - vnear) < 1e-6:
                allv = d_np[finite]
                vnear, vfar = float(allv.min()), float(allv.max()) + 1e-3

            # === Depth -> 0~1 (near bright, far dark)
            vis = d_np.copy()
            vis[~finite] = np.nan
            vis = np.clip(vis, vnear, vfar)
            d01 = (vfar - vis) / (vfar - vnear)
            d01 = np.clip(np.nan_to_num(d01, nan=0.0), 0.0, 1.0)
            if gamma is not None and gamma > 0:
                d01 = np.power(d01, 1.0 / gamma)

            # === (Optional) estimate normals from depth + fake lighting, multiply onto color to form 3D gradient
            if shade_from_depth:
                nmap = _compute_normals_from_depth(d_np, horizontal_fov_deg)   # (H, W, 3)
                L = np.asarray(light_dir, dtype=np.float32)
                L = L / (np.linalg.norm(L) + 1e-8)
                # Lambertian: clamp(n·L)
                shade = np.clip((nmap[..., 0] * L[0] + nmap[..., 1] * L[1] + nmap[..., 2] * L[2]), 0.0, 1.0)
                shade = np.where(finite, shade, 0.0)
                shade = min_ambient + (1.0 - min_ambient) * shade  # Add some ambient to avoid full black
            else:
                shade = 1.0

            # === Apply colormap + multiply shading
            cmap = plt.get_cmap("viridis")
            rgb = cmap(d01)[..., :3]                          # (H, W, 3) 0~1
            if np.isscalar(shade):
                rgb_shaded = rgb
            else:
                rgb_shaded = rgb * shade[..., None]
            rgb_shaded = np.clip(rgb_shaded, 0.0, 1.0)

            # (Optional) fill the masked bottom region (remove_table) with a darker tone of the same hue, to avoid abrupt edges
            if remove_table:
                cut = int(H * (1.0 - float(bottom_rows_frac)))
                if cut < H:
                    # Use the far-end color, multiplied by a lower brightness
                    far_color = cmap(0.0)[0:3]  # color at d01=0 (farthest)
                    rgb_shaded[cut:, :, :] = (np.asarray(far_color)[None, None, :] * 0.7)

            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"depth_env{i:02d}_step{step:06d}.png")
            plt.imsave(out_path, rgb_shaded)

            print(
                f"[depth] env {i} step {step}: "
                f"valid={valid.sum().item()}/{d.numel()}, "
                f"min/max(all)={float(d[valid].min()):.3f}/{float(d[valid].max()):.3f} m, "
                f"window=[{vnear:.3f},{vfar:.3f}], "
                f"shading={'on' if shade_from_depth else 'off'}, "
                f"remove_table={'on' if remove_table else 'off'} -> {out_path}"
            )

    def _get_allowed_object_ids_for_env(self, env_id: int):
        """
        Build a robust allowed-ids set so that it works whether IMAGE_SEGMENTATION returns:
        - rigid-body segmentation IDs (set via set_rigid_body_segmentation_id), or
        - actor instance IDs (per-sim or per-env; sometimes off-by-one in some builds).

        Also intersect with what's actually present in this env's segmentation image.
       """
        seg_off = env_id * 1000

        candidates: list[int] = []

        # 1) Our explicit rigid-body segmentation IDs (with and without offset)
        candidates += [201 + seg_off, 202 + seg_off, 201, 202]

        # 2) Actor instance IDs – SIM domain (existing fallback)
        try:
            rid_sim = int(self._global_manip_obj_rh_indices[env_id, 0].item())
            lid_sim = int(self._global_manip_obj_lh_indices[env_id, 0].item())
            candidates += [rid_sim, lid_sim, rid_sim + 1, lid_sim + 1]
        except Exception:
            pass

        # 3) Actor instance IDs – ENV domain (many Isaac Gym builds use these in IMAGE_SEGMENTATION)
        try:
            rid_env = int(self.gym.find_actor_index(self.envs[env_id], "manip_obj_rh", gymapi.DOMAIN_ENV))
            lid_env = int(self.gym.find_actor_index(self.envs[env_id], "manip_obj_lh", gymapi.DOMAIN_ENV))
            candidates += [rid_env, lid_env, rid_env + 1, lid_env + 1]
        except Exception:
            pass

        # Deduplicate -> tensor on sim device
        uniq = sorted(set(int(x) for x in candidates))
        cand_t = torch.tensor(uniq, device=self.sim_device, dtype=torch.int32)

        # 4) If we already have seg for this env, only keep IDs that actually appear there
        try:
            if hasattr(self, "last_segs") and self.last_segs is not None and env_id < len(self.last_segs):
                seg_unique = torch.unique(self.last_segs[env_id].to(torch.int32)).to(self.sim_device)
                present = cand_t[torch.isin(cand_t, seg_unique)]
                if present.numel() > 0:
                    return present
        except Exception:
            # If anything fails, fall back to the full candidate set
            pass

        # 5) Fallback: return the full candidate set; downstream will AND with seg anyway
        return cand_t


    def _depth_to_pointcloud_from_seg(
        self,
        depth_tensor: torch.Tensor,
        seg_tensor: torch.Tensor,
        allowed_ids: torch.Tensor,
        horizontal_fov_deg: float = 69.4,
        max_points: int | None = 120000,
        border_dilate_px: int = 1,
        z_min: float | None = 0.35,   # ★ New: minimum depth (meters)
        z_max: float | None = 1.50,   # ★ New: maximum depth (meters)
    ):
        """
        Convert only pixels whose segmentation is in allowed_ids into 3D point cloud (camera coordinates).
        ★ Fix: first dilate the *pure segmentation mask*, then AND with the depth condition; optionally window by depth to cut distant background.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        if depth_tensor is None or seg_tensor is None:
            return None

        H, W = depth_tensor.shape
        device = depth_tensor.device

        # === 0) Camera intrinsics from FOV ===
        fovx = np.deg2rad(horizontal_fov_deg)
        fx = W / (2.0 * np.tan(fovx / 2.0))
        fovy = 2.0 * np.arctan(np.tan(fovx / 2.0) * (H / W))
        fy = H / (2.0 * np.tan(fovy / 2.0))
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

        # === 1) Build the *pure seg mask* ===
        seg_i32 = seg_tensor.to(torch.int32)
        m_seg = torch.isin(seg_i32, allowed_ids)  # Keep only desired objects (e.g., 201+off, 202+off)

        # (Optional) boundary dilation: only apply to the *pure seg* to avoid swallowing large background regions
        if border_dilate_px and border_dilate_px > 0:
            k = 2 * border_dilate_px + 1
            m = m_seg[None, None].float()
            m = F.max_pool2d(m, kernel_size=k, stride=1, padding=border_dilate_px)
            m_seg = (m[0, 0] > 0.5)

        # === 2) AND with depth validity + (optional) depth window ===
        depth_ok = torch.isfinite(depth_tensor) & (depth_tensor > 0)
        if z_min is not None:
            depth_ok = depth_ok & (depth_tensor >= float(z_min))
        if z_max is not None:
            depth_ok = depth_ok & (depth_tensor <= float(z_max))

        mask = m_seg & depth_ok
        if not mask.any():
            return None

        # === 3) Pixels → 3D ===
        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')

        v_valid = v_grid[mask]
        u_valid = u_grid[mask]
        z_valid = depth_tensor[mask]

        x = (u_valid - cx) / fx * z_valid
        y = (v_valid - cy) / fy * z_valid

        # Your camera coordinate convention: X right, Y up (thus take -y)
        pts_cam = torch.stack([x, -y, z_valid], dim=1)  # (N,3)

        # === 4) Simple outlier removal (remove >98th percentile by distance from median)
        if pts_cam.shape[0] > 2000:
            med = torch.median(pts_cam, dim=0)[0]
            dist = torch.norm(pts_cam - med, dim=1)
            thr = torch.quantile(dist, 0.98)
            pts_cam = pts_cam[dist <= thr]

        # === 5) Downsampling (uniform stepping)
        if (max_points is not None) and (pts_cam.shape[0] > max_points):
            n = pts_cam.shape[0]
            step = max(1, n // max_points)
            idx = torch.arange(0, n, step, device=device)[:max_points]
            pts_cam = pts_cam[idx]

        return pts_cam



    def _save_depth_16bit_png(self, out_dir: str, step: int, max_envs: int = 1024, scale: float = 1000.0):
        """
        Save 16-bit grayscale PNG (store in integer millimeters; background=0). Read back with: depth_m = image_uint16.astype(np.float32) / scale
        """
        import os, numpy as np, torch
        from PIL import Image

        os.makedirs(out_dir, exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.sim_device)

        num = min(len(self.last_depths) if self.last_depths is not None else 0, max_envs)
        for i in range(num):
            d = self.last_depths[i].detach().float().cpu()  # (H,W) meters
            valid = torch.isfinite(d) & (d > 0)
            if not valid.any():
                continue
            d_m = torch.zeros_like(d)
            d_m[valid] = d[valid]
            d16 = (d_m * scale).round().clamp(0, 65535).to(torch.int32).numpy().astype(np.uint16)
            Image.fromarray(d16, mode="I;16").save(os.path.join(out_dir, f"depth16_env{i:02d}_step{step:06d}.png"))


    def _depth_to_pointcloud(self, depth_tensor, horizontal_fov_deg=69.4, 
                            max_points=None, min_depth=0.3, max_depth=1.5,
                            downsample_factor=1,  # ★ No downsampling
                            remove_table_bottom=True,
                            bottom_rows_frac=0.4):
        """
        Improved depth-to-pointcloud function
        """
        import torch
        import numpy as np
        
        if depth_tensor is None:
            return None
            
        H, W = depth_tensor.shape
        device = depth_tensor.device
        
        print(f"Original depth shape: {H}x{W}, range: {depth_tensor.min():.3f}-{depth_tensor.max():.3f}")
        
        # ===== 1. Compute camera intrinsics =====
        fovx = np.deg2rad(horizontal_fov_deg)
        fx = W / (2.0 * np.tan(fovx / 2.0))
        fovy = 2.0 * np.arctan(np.tan(fovx / 2.0) * (H / W))
        fy = H / (2.0 * np.tan(fovy / 2.0))
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        
        print(f"Camera params: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        
        # ===== 2. Build pixel coordinate grid (no or light downsampling) =====
        if downsample_factor > 1:
            # Light downsampling
            v_coords = torch.arange(0, H, downsample_factor, device=device, dtype=torch.float32)
            u_coords = torch.arange(0, W, downsample_factor, device=device, dtype=torch.float32)
            v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')
            
            # Corresponding depth values
            v_indices = (v_coords.long()).clamp(0, H-1)
            u_indices = (u_coords.long()).clamp(0, W-1)
            depth_sampled = depth_tensor[v_indices][:, u_indices]
        else:
            # Use all pixels
            v_coords = torch.arange(H, device=device, dtype=torch.float32)
            u_coords = torch.arange(W, device=device, dtype=torch.float32)
            v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')
            depth_sampled = depth_tensor
        
        # ===== 3. Remove table region =====
        if remove_table_bottom:
            table_threshold = H * (1.0 - bottom_rows_frac)
            table_mask = v_grid < table_threshold
        else:
            table_mask = torch.ones_like(v_grid, dtype=torch.bool)
        
        # ===== 4. Filter valid depth values =====
        valid_mask = (
            (depth_sampled > min_depth) & 
            (depth_sampled < max_depth) & 
            torch.isfinite(depth_sampled) &
            table_mask
        )
        
        print(f"Valid points: {valid_mask.sum()} / {valid_mask.numel()}")
        
        if not valid_mask.any():
            return None
            
        # Extract valid coords and depth
        v_valid = v_grid[valid_mask]
        u_valid = u_grid[valid_mask]
        z_valid = depth_sampled[valid_mask]
        
        # ===== 5. Pixel -> 3D =====
        # Note: coordinate system here
        x_valid = (u_valid - cx) / fx * z_valid  # right is positive
        y_valid = (v_valid - cy) / fy * z_valid  # down is positive
        
        # Assemble point cloud - note coordinate conversion
        points_3d = torch.stack([x_valid, -y_valid, z_valid], dim=1)  # ★ flip y-axis
        
        print(f"Point cloud range: X[{x_valid.min():.3f}, {x_valid.max():.3f}], "
            f"Y[{y_valid.min():.3f}, {y_valid.max():.3f}], "
            f"Z[{z_valid.min():.3f}, {z_valid.max():.3f}]")
        
        # ===== 6. Remove outliers =====
        if points_3d.shape[0] > 100:  # Only do outlier removal when there are enough points
            # Remove outliers based on Z
            z_median = torch.median(z_valid)
            z_mad = torch.median(torch.abs(z_valid - z_median))
            z_threshold = z_median + 3 * z_mad
            
            inlier_mask = torch.abs(z_valid - z_median) < 3 * z_mad
            points_3d = points_3d[inlier_mask]
            print(f"After outlier removal: {points_3d.shape[0]} points")
        
        # ===== 7. Smart downsampling (if too many points) =====
        if max_points is not None and points_3d.shape[0] > max_points:
            # Stratified stepping instead of random sampling
            n_points = points_3d.shape[0]
            step = n_points // max_points
            indices = torch.arange(0, n_points, step, device=device)[:max_points]
            points_3d = points_3d[indices]
            print(f"Downsampled to: {points_3d.shape[0]} points")
        
        return points_3d

    def _save_object_pointcloud_png(self, out_dir: str, step: int, max_envs: int = 1024, 
                               method="simple",  # "simple" or "world_coord"
                               views: tuple = ((20, -60),)):
        """
        Save object point cloud visualization
        """
        import os
        from main.dataset.base import save_point_cloud_images
        
        if self.last_depths is None:
            print("No depth data available")
            return
            
        os.makedirs(out_dir, exist_ok=True)
        
        num = min(len(self.last_depths), max_envs)
        
        for i in range(num):
            depth_tensor = self.last_depths[i]
            
            if depth_tensor is None:
                continue
                
            if self.debug:
                print(f"\n=== Processing object pointcloud for env {i} ===")
            
            # Choose different extraction functions based on method
            if method == "simple":
                points_3d = self._depth_to_object_pointcloud_simple(
                    depth_tensor,
                    object_depth_range=(0.6, 1.2),  # Adjust this range to focus on the object
                    center_crop_ratio=0.4,  # Only consider the central 40% region
                    max_points=200000
                )
            else:
                points_3d = self._depth_to_object_pointcloud(
                    depth_tensor, 
                    env_id=i,
                    object_radius=0.12,  # 12 cm around the object
                    max_points=200000
                )
            
            if points_3d is None or points_3d.shape[0] == 0:
                print(f"No object points for env {i}")
                continue
                
            # Save object point cloud
            out_path = os.path.join(out_dir, f"object_pointcloud_env{i:02d}_step{step:06d}.png")
            
            print(f"Saving object pointcloud env {i}: {points_3d.shape[0]} points")
            
            save_point_cloud_images(
                points_xyz=points_3d,
                out_path=out_path,
                views=views,
                s=10.0,  # Larger points for visibility
                show_world_axes=True,
                # Focus the view range around the object
                xlim=(-0.15, 0.15),
                ylim=(-0.15, 0.15), 
                zlim=(0.5, 1.3)
            )

    def _save_colored_object_pointcloud_png(self, out_dir: str, step: int, max_envs: int = 1024, 
                                       views: tuple = ((20, -60),)):
        """
        Save colored object point cloud visualization
        """
        import os
        
        if self.last_depths is None:
            print("No depth data available")
            return
            
        # ★ Fetch RGB images at the same time
        self.last_rgb_images = self._grab_rgb_images(device="gpu")
        
        if self.last_rgb_images is None:
            print("No RGB data available")
            return
            
        os.makedirs(out_dir, exist_ok=True)
        
        num = min(len(self.last_depths), len(self.last_rgb_images), max_envs)
        
        for i in range(num):
            depth_tensor = self.last_depths[i]
            rgb_tensor = self.last_rgb_images[i]
            
            if depth_tensor is None or rgb_tensor is None:
                continue

            if self.debug: 
                print(f"\n=== Processing colored object pointcloud for env {i} ===")
                print(f"Depth shape: {depth_tensor.shape}, RGB shape: {rgb_tensor.shape}")
            
            # Convert to colored point cloud
            points_3d, colors = self._depth_to_colored_object_pointcloud_simple(
                depth_tensor,
                rgb_tensor,
                object_depth_range=(0.6, 1.2),
                center_crop_ratio=0.4,
                max_points=80000
            )
            
            if points_3d is None or points_3d.shape[0] == 0:
                print(f"No colored object points for env {i}")
                continue
                
            # Save colored object point cloud
            out_path = os.path.join(out_dir, f"colored_object_pointcloud_env{i:02d}_step{step:06d}.png")
            
            if self.debug:
                print(f"Saving colored object pointcloud env {i}: {points_3d.shape[0]} points")
            
            # Use the improved colored point cloud visualization function
            self._save_colored_pointcloud_visualization(
                points_3d, colors, out_path, views
            )

    def _save_no_rgb_object_pointcloud_png(self, out_dir: str, step: int, max_envs: int = 1024, 
                                     views: tuple = ((20, -60),),
                                     color_method: str = 'depth'):
        """
        Save object point cloud without using RGB
        
        Args:
            color_method: 'depth', 'height', 'distance', 'uniform', 'coordinate'
        """
        import os
        
        if self.last_depths is None:
            print("No depth data available")
            return
            
        os.makedirs(out_dir, exist_ok=True)
        
        num = min(len(self.last_depths), max_envs)
        
        for i in range(num):
            depth_tensor = self.last_depths[i]
            
            if depth_tensor is None:
                continue
                
            print(f"\n=== Processing NO-RGB object pointcloud for env {i} ===")
            
            # Convert to point cloud (without using RGB)
            points_3d, depth_colors = self._depth_to_object_pointcloud_no_rgb(
                depth_tensor,
                object_depth_range=(0.6, 1.2),
                center_crop_ratio=0.7,
                max_points=150000
            )
            
            if points_3d is None or points_3d.shape[0] == 0:
                print(f"No object points for env {i}")
                continue
                
            out_path = os.path.join(out_dir, f"no_rgb_object_pointcloud_env{i:02d}_step{step:06d}.png")
            
            print(f"Saving NO-RGB object pointcloud env {i}: {points_3d.shape[0]} points")
            
            # Save point cloud according to the specified method
            if color_method == 'depth':
                self._save_depth_colored_pointcloud_visualization(
                    points_3d, depth_colors, out_path, views
                )
            else:
                self._save_alternative_colored_pointcloud_visualization(
                    points_3d, out_path, views, color_method
                )

    def _save_depth_colored_pointcloud_visualization(self, points_3d, depth_colors, out_path, views):
        """
        Visualization with colors assigned by depth values
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        pts = points_3d.detach().cpu().numpy()
        depths = depth_colors.detach().cpu().numpy()
        
        # High-resolution settings
        fig = plt.figure(figsize=(12 * len(views), 10), dpi=300)
        
        for i, (elev, azim) in enumerate(views, 1):
            ax = fig.add_subplot(1, len(views), i, projection='3d')
            
            # ★ Color the point cloud using depth values
            scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                            s=10,               # point size
                            c=depths,           # color by depth
                            cmap='viridis',     # colormap (near bright, far dark)
                            alpha=0.9,
                            edgecolors='none')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Depth (m)', fontsize=10)
            
            ax.view_init(elev=70, azim=-90)
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_zlabel('Z (m)', fontsize=12)
            
            ax.grid(True, alpha=0.3)
            
            # Focus on the object region
            ax.set_xlim(-0.1, 0.2)
            ax.set_ylim(-0.2, 0)
            ax.set_zlim(0, 1.2)
            
            ax.set_title(f'Depth-Colored Object Point Cloud\n({pts.shape[0]} points, elev={elev}, azim={azim})', 
                        fontsize=14)
        
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
        plt.close(fig)
        print(f"Saved depth-colored pointcloud to: {out_path}")


    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # * >>> import table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True

        table_width_offset = 0.2
        table_asset = self.gym.create_box(self.sim, 0.8 + table_width_offset, 1.6, 0.03, table_asset_options)

        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        self.dexhand_rh_pose = gymapi.Transform()
        table_half_height = 0.015
        table_half_width = 0.4
        self._table_surface_z = table_surface_z = table_pos.z + table_half_height
        self.dexhand_rh_pose.p = gymapi.Vec3(-table_half_width, 0, table_surface_z + ROBOT_HEIGHT)
        self.dexhand_rh_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, 0)
        self.dexhand_lh_pose = deepcopy(self.dexhand_rh_pose)

        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        dataset_list = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices]))

        self.demo_dataset_lh_dict = {}
        self.demo_dataset_rh_dict = {}

        for dataset_type in dataset_list:
            self.demo_dataset_lh_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side="left",
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_episode_length,
                dexhand=self.dexhand_lh,
                embodiment=self.cfg["env"]["dexhand"],
            )
            self.demo_dataset_rh_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side="right",
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_episode_length,
                dexhand=self.dexhand_rh,
                embodiment=self.cfg["env"]["dexhand"],
            )

        dexhand_rh_asset_file = self.dexhand_rh.urdf_path
        dexhand_lh_asset_file = self.dexhand_lh.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.angular_damping = 20
        asset_options.linear_damping = 20
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        dexhand_rh_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_rh_asset_file), asset_options)
        dexhand_lh_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_lh_asset_file), asset_options)
        dexhand_rh_dof_stiffness = torch.tensor(
            [500] * self.dexhand_rh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_rh_dof_damping = torch.tensor(
            [30] * self.dexhand_rh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_lh_dof_stiffness = torch.tensor(
            [500] * self.dexhand_lh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_lh_dof_damping = torch.tensor(
            [30] * self.dexhand_lh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_rh_asset)
        asset_lh_dof_props = self.gym.get_asset_dof_properties(dexhand_lh_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }
        self.limit_info["lh"] = {
            "lower": np.asarray(asset_lh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_lh_dof_props["upper"]).copy().astype(np.float32),
        }

        rigid_shape_rh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_rh_asset)
        for element in rigid_shape_rh_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(dexhand_rh_asset, rigid_shape_rh_props_asset)

        rigid_shape_lh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_lh_asset)
        for element in rigid_shape_lh_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(dexhand_lh_asset, rigid_shape_lh_props_asset)

        self.num_dexhand_rh_bodies = self.gym.get_asset_rigid_body_count(dexhand_rh_asset)
        self.num_dexhand_rh_dofs = self.gym.get_asset_dof_count(dexhand_rh_asset)
        self.num_dexhand_lh_bodies = self.gym.get_asset_rigid_body_count(dexhand_lh_asset)
        self.num_dexhand_lh_dofs = self.gym.get_asset_dof_count(dexhand_lh_asset)

        print(f"Num dexhand_r Bodies: {self.num_dexhand_rh_bodies}")
        print(f"Num dexhand_r DOFs: {self.num_dexhand_rh_dofs}")
        print(f"Num dexhand_l Bodies: {self.num_dexhand_lh_bodies}")
        print(f"Num dexhand_l DOFs: {self.num_dexhand_lh_dofs}")

        dexhand_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_rh_asset)
        self.dexhand_rh_dof_lower_limits = []
        self.dexhand_rh_dof_upper_limits = []
        self._dexhand_rh_effort_limits = []
        self._dexhand_rh_dof_speed_limits = []
        for i in range(self.num_dexhand_rh_dofs):
            dexhand_rh_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_rh_dof_props["stiffness"][i] = dexhand_rh_dof_stiffness[i]
            dexhand_rh_dof_props["damping"][i] = dexhand_rh_dof_damping[i]

            self.dexhand_rh_dof_lower_limits.append(dexhand_rh_dof_props["lower"][i])
            self.dexhand_rh_dof_upper_limits.append(dexhand_rh_dof_props["upper"][i])
            self._dexhand_rh_effort_limits.append(dexhand_rh_dof_props["effort"][i])
            self._dexhand_rh_dof_speed_limits.append(dexhand_rh_dof_props["velocity"][i])

        self.dexhand_rh_dof_lower_limits = torch.tensor(self.dexhand_rh_dof_lower_limits, device=self.sim_device)
        self.dexhand_rh_dof_upper_limits = torch.tensor(self.dexhand_rh_dof_upper_limits, device=self.sim_device)
        self._dexhand_rh_effort_limits = torch.tensor(self._dexhand_rh_effort_limits, device=self.sim_device)
        self._dexhand_rh_dof_speed_limits = torch.tensor(self._dexhand_rh_dof_speed_limits, device=self.sim_device)

        # set dexhand_l dof properties
        dexhand_lh_dof_props = self.gym.get_asset_dof_properties(dexhand_lh_asset)
        self.dexhand_lh_dof_lower_limits = []
        self.dexhand_lh_dof_upper_limits = []
        self._dexhand_lh_effort_limits = []
        self._dexhand_lh_dof_speed_limits = []
        for i in range(self.num_dexhand_lh_dofs):
            dexhand_lh_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_lh_dof_props["stiffness"][i] = dexhand_lh_dof_stiffness[i]
            dexhand_lh_dof_props["damping"][i] = dexhand_lh_dof_damping[i]

            self.dexhand_lh_dof_lower_limits.append(dexhand_lh_dof_props["lower"][i])
            self.dexhand_lh_dof_upper_limits.append(dexhand_lh_dof_props["upper"][i])
            self._dexhand_lh_effort_limits.append(dexhand_lh_dof_props["effort"][i])
            self._dexhand_lh_dof_speed_limits.append(dexhand_lh_dof_props["velocity"][i])

        self.dexhand_lh_dof_lower_limits = torch.tensor(self.dexhand_lh_dof_lower_limits, device=self.sim_device)
        self.dexhand_lh_dof_upper_limits = torch.tensor(self.dexhand_lh_dof_upper_limits, device=self.sim_device)
        self._dexhand_lh_effort_limits = torch.tensor(self._dexhand_lh_effort_limits, device=self.sim_device)
        self._dexhand_lh_dof_speed_limits = torch.tensor(self._dexhand_lh_dof_speed_limits, device=self.sim_device)

        # compute aggregate size
        num_dexhand_rh_bodies = self.gym.get_asset_rigid_body_count(dexhand_rh_asset)
        num_dexhand_rh_shapes = self.gym.get_asset_rigid_shape_count(dexhand_rh_asset)
        num_dexhand_lh_bodies = self.gym.get_asset_rigid_body_count(dexhand_lh_asset)
        num_dexhand_lh_shapes = self.gym.get_asset_rigid_shape_count(dexhand_lh_asset)

        self.dexhand_rs = []
        self.dexhand_ls = []
        self.envs = []

        assert len(self.dataIndices) == 1 or not self.rollout_state_init, "rollout_state_init only works with one data"

        def segment_data(k, data_dict):
            todo_list = self.dataIndices
            idx = todo_list[k % len(todo_list)]
            return data_dict[ManipDataFactory.dataset_type(idx)][idx]

        self.demo_data_lh = [segment_data(i, self.demo_dataset_lh_dict) for i in tqdm(range(self.num_envs))]
        self.demo_data_lh = self.pack_data(self.demo_data_lh, side="lh")
        self.demo_data_rh = [segment_data(i, self.demo_dataset_rh_dict) for i in tqdm(range(self.num_envs))]
        self.demo_data_rh = self.pack_data(self.demo_data_rh, side="rh")

        # Create environments
        self.manip_obj_rh_mass = []
        self.manip_obj_rh_com = []
        self.manip_obj_lh_mass = []
        self.manip_obj_lh_com = []
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            rh_current_asset, rh_sum_rigid_body_count, rh_sum_rigid_shape_count, rh_obj_scale, rh_obj_mass = (
                self._create_obj_assets(i, side="rh")
            )
            lh_current_asset, lh_sum_rigid_body_count, lh_sum_rigid_shape_count, lh_obj_scale, lh_obj_mass = (
                self._create_obj_assets(i, side="lh")
            )

            max_agg_bodies = (
                num_dexhand_rh_bodies
                + num_dexhand_lh_bodies
                + 1
                + rh_sum_rigid_body_count
                + lh_sum_rigid_body_count
                + (0 + (0 + self.dexhand_lh.n_bodies * 2 if not self.headless else 0))
            )  # 1 for table
            max_agg_shapes = (
                num_dexhand_rh_shapes
                + num_dexhand_lh_shapes
                + 1
                + rh_sum_rigid_shape_count
                + lh_sum_rigid_shape_count
                + (0 + (0 + self.dexhand_lh.n_bodies * 2 if not self.headless else 0))
                + (1 if self._record else 0)
            )
            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: dexhand_r should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # camera handler for view rendering
            # if self.camera_handlers is not None:
            #     self.camera_handlers.append(
            #         self.create_camera(
            #             env=env_ptr,
            #             isaac_gym=self.gym,
            #         )
            #     )
            cam = self.create_camera(env=env_ptr, isaac_gym=self.gym)
            # self.camera_handlers.append(cam)
            self.cameras.append(cam)

            # Create dexhand_r
            dexhand_rh_actor = self.gym.create_actor(
                env_ptr,
                dexhand_rh_asset,
                self.dexhand_rh_pose,
                "dexhand_r",
                i,
                (1 if self.dexhand_rh.self_collision else 0),
            )
            dexhand_lh_actor = self.gym.create_actor(
                env_ptr,
                dexhand_lh_asset,
                self.dexhand_lh_pose,
                "dexhand_l",
                i,
                (1 if self.dexhand_lh.self_collision else 0),
            )
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_rh_actor)
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_lh_actor)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_rh_actor, dexhand_rh_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_lh_actor, dexhand_lh_dof_props)

            # Create table and obstacles
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
            table_props[0].friction = 0.1  # ? only one table shape in each env
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            # set table's color to be dark gray
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

            self.obj_rh_handle, _ = self._create_obj_actor(
                env_ptr, i, rh_current_asset, side="rh"
            )  # the handle is all the same for all envs
            self.obj_lh_handle, _ = self._create_obj_actor(env_ptr, i, lh_current_asset, side="lh")
            self.gym.set_actor_scale(env_ptr, self.obj_rh_handle, rh_obj_scale)
            self.gym.set_actor_scale(env_ptr, self.obj_lh_handle, lh_obj_scale)
            obj_props_rh = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_rh_handle)
            obj_props_lh = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_lh_handle)
            obj_props_rh[0].mass = min(0.5, obj_props_rh[0].mass)  # * we only consider the mass less than 500g
            obj_props_lh[0].mass = min(0.5, obj_props_lh[0].mass)  # * we only consider the mass less than 500g

            if rh_obj_mass is not None:
                obj_props_rh[0].mass = rh_obj_mass
            if lh_obj_mass is not None:
                obj_props_lh[0].mass = lh_obj_mass

            # ! Updating the mass and scale might slightly alter the inertia tensor;
            # ! however, because the magnitude of our modifications is minimal, we temporarily neglect this effect.
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_rh_handle, obj_props_rh)
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_lh_handle, obj_props_lh)
            self.manip_obj_rh_mass.append(obj_props_rh[0].mass)
            self.manip_obj_rh_com.append(
                torch.tensor([obj_props_rh[0].com.x, obj_props_rh[0].com.y, obj_props_rh[0].com.z])
            )
            self.manip_obj_lh_mass.append(obj_props_lh[0].mass)
            self.manip_obj_lh_com.append(
                torch.tensor([obj_props_lh[0].com.x, obj_props_lh[0].com.y, obj_props_lh[0].com.z])
            )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.dexhand_rs.append(dexhand_rh_actor)
            self.dexhand_ls.append(dexhand_lh_actor)


            # =======================
            # segmentation id
            # =======================
            seg_off = i * 1000

            # right hand
            self._set_actor_bodies_seg_id(env_ptr, dexhand_rh_actor, 101 + seg_off)

            # left hand
            self._set_actor_bodies_seg_id(env_ptr, dexhand_lh_actor, 102 + seg_off)

            # right/left object
            self._set_actor_bodies_seg_id(env_ptr, self.obj_rh_handle, 201 + seg_off)
            self._set_actor_bodies_seg_id(env_ptr, self.obj_lh_handle, 202 + seg_off)

            # Desktop
            self._set_actor_bodies_seg_id(env_ptr, table_actor, 301 + seg_off)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

        self.manip_obj_rh_mass = torch.tensor(self.manip_obj_rh_mass, device=self.device)
        self.manip_obj_rh_com = torch.stack(self.manip_obj_rh_com, dim=0).to(self.device)
        self.manip_obj_lh_mass = torch.tensor(self.manip_obj_lh_mass, device=self.device)
        self.manip_obj_lh_com = torch.stack(self.manip_obj_lh_com, dim=0).to(self.device)

        # Setup data
        self.init_data()

        self._debug_print_hand_names_and_indices()

        # import ipdb; ipdb.set_trace()

    def _build_hand_edges(self, names: list[str], add_palm_links: bool = True):
        """
        Automatically build "joint connections" based on the rigid body names you provide.
        Node order = the order of `names`; returns edges = List[(i, j)] for drawing lines.
        Rules:
        - base_link → each finger's proximal
        - index/middle/ring/pinky: proximal → intermediate → tip
        - thumb: proximal_base → proximal → intermediate → distal → tip, and base_link → thumb_proximal_base
        - (optional) palm-side links: index_proximal—middle_proximal—ring_proximal—pinky_proximal
        """
        idx = {n: i for i, n in enumerate(names)}
        # Automatically determine side prefix "R_" or "L_"
        prefix = names[0].split('_', 1)[0] + '_'  # "R_" or "L_"
        base_name = next(n for n in names if n.endswith("hand_base_link"))
        base = idx[base_name]

        def get(local):  # e.g., get("index_proximal")
            return idx.get(prefix + local, None)

        edges = []

        # Four fingers: index/middle/ring/pinky (prox → inter → tip)
        for finger in ["index", "middle", "ring", "pinky"]:
            p = get(f"{finger}_proximal")
            m = get(f"{finger}_intermediate")
            t = get(f"{finger}_tip")
            if p is not None:
                edges.append((base, p))
            if p is not None and m is not None:
                edges.append((p, m))
            if m is not None and t is not None:
                edges.append((m, t))

        # Thumb: proximal_base → proximal → intermediate → distal → tip (and link base → proximal_base)
        tb = get("thumb_proximal_base")
        tp = get("thumb_proximal")
        ti = get("thumb_intermediate")
        td = get("thumb_distal")
        tt = get("thumb_tip")
        if tb is not None:
            edges.append((base, tb))
            if tp is not None:
                edges.append((tb, tp))
                if ti is not None:
                    edges.append((tp, ti))
                    if td is not None:
                        edges.append((ti, td))
                        if tt is not None:
                            edges.append((td, tt))

        # Palm-side (optional): connect the proximal links to form a "palm edge"
        if add_palm_links:
            palm = [get(f"{f}_proximal") for f in ["index", "middle", "ring", "pinky"]]
            palm = [i for i in palm if i is not None]
            for a, b in zip(palm[:-1], palm[1:]):
                edges.append((a, b))

        return edges


    def _set_actor_bodies_seg_id(self, env_ptr, actor_handle: int, seg_id: int):
        n_bodies = self.gym.get_actor_rigid_body_count(env_ptr, actor_handle)
        for b in range(n_bodies):
            # Pass "index" instead of handle
            self.gym.set_rigid_body_segmentation_id(env_ptr, actor_handle, b, seg_id)

        # (Optional) Read back immediately to verify
        check = [self.gym.get_rigid_body_segmentation_id(env_ptr, actor_handle, b) for b in range(n_bodies)]
        print(f"[seg] actor {actor_handle}: ids per body = {check[:8]}{'...' if n_bodies>8 else ''}")


    def _render_all_cameras(self):
        # Ensure physics results and graphics are updated before reading camera tensors
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)


    def _grab_depth_seg_rgb_once(self, cameras=None):
        cams = self.cameras if cameras is None else cameras
        depths, segs, rgbs = [], [], []
        # 一次同步 + 一次 render + 一次 access
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        try:
            for env_ptr, cam in zip(self.envs, cams):
                # depth
                d = -gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam, gymapi.IMAGE_DEPTH)
                ).to(self.sim_device).to(torch.float32)
                d = torch.where((d > 0) & torch.isfinite(d), d, torch.zeros_like(d))
                depths.append(d)

                # seg
                s = gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam, gymapi.IMAGE_SEGMENTATION)
                ).to(self.sim_device).to(torch.int32)
                segs.append(s)

                # rgb（用得到再存）
                rgb = gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam, gymapi.IMAGE_COLOR)
                )
                if rgb.ndim == 3 and rgb.shape[0] == 4:
                    rgb = rgb.permute(1, 2, 0)
                rgbs.append(rgb.to(self.sim_device))
        finally:
            self.gym.end_access_image_tensors(self.sim)

        return depths, segs, rgbs

    def _grab_depth_images(self, device="gpu", cameras=None):
        depths = []
        cams = self.cameras if cameras is None else cameras
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        try:
            for env_ptr, cam in zip(self.envs, cams):
                if device == "gpu":
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam, gymapi.IMAGE_DEPTH)
                    d = -gymtorch.wrap_tensor(depth_tensor).to(self.sim_device).to(torch.float32)
                    d_valid = torch.where((d > 0) & torch.isfinite(d), d, torch.zeros_like(d))
                    depths.append(d_valid)
                else:
                    d_np = self.gym.get_camera_image(self.sim, env_ptr, cam, gymapi.IMAGE_DEPTH).astype(np.float32)
                    d = -torch.from_numpy(d_np)
                    d_valid = torch.where((d > 0) & torch.isfinite(d), d, torch.zeros_like(d))
                    depths.append(d_valid.to(self.sim_device))
        finally:
            self.gym.end_access_image_tensors(self.sim)
        return depths

    def _grab_seg_images(self, device="gpu", cameras=None):
        segs = []
        cams = self.cameras if cameras is None else cameras

        # NEW: ensure correct render order for multi-env seg
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)


        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        try:
            for env_ptr, cam in zip(self.envs, cams):
                if device == "gpu":
                    seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam, gymapi.IMAGE_SEGMENTATION)
                    seg = gymtorch.wrap_tensor(seg_tensor).to(self.sim_device).to(torch.int32)
                else:
                    seg_np = self.gym.get_camera_image(self.sim, env_ptr, cam, gymapi.IMAGE_SEGMENTATION)
                    seg = torch.from_numpy(seg_np.astype(np.int32)).to(self.sim_device)
                segs.append(seg)
        finally:
            self.gym.end_access_image_tensors(self.sim)
        return segs

    def _grab_rgb_images(self, device="gpu", cameras=None):
        rgb_images = []
        cams = self.cameras if cameras is None else cameras
        # NEW: ensure correct render order for multi-env seg
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        try:
            for env_ptr, cam in zip(self.envs, cams):
                if device == "gpu":
                    rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam, gymapi.IMAGE_COLOR)
                    rgb = gymtorch.wrap_tensor(rgb_tensor)  # (H,W,4) or (4,H,W)
                    if rgb.ndim == 3 and rgb.shape[-1] == 4:
                        pass  # (H,W,4)
                    elif rgb.ndim == 3 and rgb.shape[0] == 4:
                        rgb = rgb.permute(1, 2, 0)
                    rgb_images.append(rgb.to(self.sim_device))
                else:
                    rgb_np = self.gym.get_camera_image(self.sim, env_ptr, cam, gymapi.IMAGE_COLOR)
                    rgb = torch.from_numpy(rgb_np.astype(np.uint8))
                    rgb_images.append(rgb.to(self.sim_device))
        finally:
            self.gym.end_access_image_tensors(self.sim)
        return rgb_images


    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        dexhand_rh_handle = self.gym.find_actor_handle(env_ptr, "dexhand_r")
        dexhand_lh_handle = self.gym.find_actor_handle(env_ptr, "dexhand_l")
        self.dexhand_rh_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_rh_handle, k) for k in self.dexhand_rh.body_names
        }
        self.dexhand_lh_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_lh_handle, k) for k in self.dexhand_lh.body_names
        }
        self.dexhand_rh_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand_rh.body_names
        }
        self.dexhand_lh_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand_lh.body_names
        }
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._rh_base_state = self._root_state[:, 0, :]
        self._lh_base_state = self._root_state[:, 1, :]

        # ? >>> for visualization
        if not self.headless:

            self.mano_joint_rh_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"rh_mano_joint_{i}"), :]
                for i in range(self.dexhand_rh.n_bodies)
            ]
            self.mano_joint_lh_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"lh_mano_joint_{i}"), :]
                for i in range(self.dexhand_lh.n_bodies)
            ]
        # ? <<<

        self._manip_obj_rh_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_rh")
        self._manip_obj_rh_root_state = self._root_state[:, self._manip_obj_rh_handle, :]
        self._manip_obj_lh_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_lh")
        self._manip_obj_lh_root_state = self._root_state[:, self._manip_obj_lh_handle, :]
        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)
        self._manip_obj_rh_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_rh_handle, "base"
        )
        self._manip_obj_lh_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_lh_handle, "base"
        )
        self._manip_obj_rh_cf = self.net_cf[:, self._manip_obj_rh_rigid_body_handle, :]
        self._manip_obj_lh_cf = self.net_cf[:, self._manip_obj_lh_rigid_body_handle, :]

        self.dexhand_rh_root_state = self._root_state[:, dexhand_rh_handle, :]
        self.dexhand_lh_root_state = self._root_state[:, dexhand_lh_handle, :]

        self.apply_forces = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.curr_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        if self.use_pid_control:
            self.rh_prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rh_prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rh_pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rh_rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        self._global_dexhand_rh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand_r", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_dexhand_lh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand_l", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        self._global_manip_obj_rh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj_rh", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_manip_obj_lh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj_lh", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        CONTACT_HISTORY_LEN = 3
        self.rh_tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()
        self.lh_tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()

    def pack_data(self, data, side="rh"):
        packed_data = {}
        packed_data["seq_len"] = torch.tensor([len(d["obj_trajectory"]) for d in data], device=self.device)
        max_len = packed_data["seq_len"].max()
        assert max_len <= self.max_episode_length, "max_len should be less than max_episode_length"

        def fill_data(stack_data):
            for i in range(len(stack_data)):
                if len(stack_data[i]) < max_len:
                    stack_data[i] = torch.cat(
                        [
                            stack_data[i],
                            stack_data[i][-1]
                            .unsqueeze(0)
                            .repeat(max_len - len(stack_data[i]), *[1 for _ in stack_data[i].shape[1:]]),
                        ],
                        dim=0,
                    )
            return torch.stack(stack_data)

        for k in data[0].keys():
            if k == "mano_joints" or k == "mano_joints_velocity":
                mano_joints = []
                for d in data:
                    if side == "rh":
                        mano_joints.append(
                            torch.concat(
                                [
                                    d[k][self.dexhand_rh.to_hand(j_name)[0]]
                                    for j_name in self.dexhand_rh.body_names
                                    if self.dexhand_rh.to_hand(j_name)[0] != "wrist"
                                ],
                                dim=-1,
                            )
                        )
                    else:
                        mano_joints.append(
                            torch.concat(
                                [
                                    d[k][self.dexhand_lh.to_hand(j_name)[0]]
                                    for j_name in self.dexhand_lh.body_names
                                    if self.dexhand_lh.to_hand(j_name)[0] != "wrist"
                                ],
                                dim=-1,
                            )
                        )
                packed_data[k] = fill_data(mano_joints)
            elif type(data[0][k]) == torch.Tensor:
                stack_data = [d[k] for d in data]
                if k != "obj_verts":
                    packed_data[k] = fill_data(stack_data)
                else:
                    packed_data[k] = torch.stack(stack_data).squeeze()
            elif type(data[0][k]) == np.ndarray:
                raise RuntimeError("Using np is very slow.")
            else:
                packed_data[k] = [d[k] for d in data]
        return packed_data

    def allocate_buffers(self):
        # will also allocate extra buffers for data dumping, used for distillation
        super().allocate_buffers()

        # basic prop fields
        if not self.training:
            self.dump_fileds = {
                k: torch.zeros(
                    (self.num_envs, v),
                    device=self.device,
                    dtype=torch.float,
                )
                for k, v in self._prop_dump_info.items()
            }

    def _create_obj_assets(self, i, side="rh"):
        if side == "rh":
            obj_id = self.demo_data_rh["obj_id"][i]
        else:
            obj_id = self.demo_data_lh["obj_id"][i]

        if obj_id in self.objs_assets:
            current_asset = self.objs_assets[obj_id]
        else:
            asset_options = gymapi.AssetOptions()
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.thickness = 0.001
            asset_options.max_linear_velocity = 50
            asset_options.max_angular_velocity = 100
            asset_options.fix_base_link = False
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 200000
            asset_options.density = 200  # * the average density of low-fill-rate 3D-printed models
            if side == "rh":
                obj_urdf_path = self.demo_data_rh["obj_urdf_path"][i]
            else:
                obj_urdf_path = self.demo_data_lh["obj_urdf_path"][i]
            current_asset = self.gym.load_asset(self.sim, *os.path.split(obj_urdf_path), asset_options)

            rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
            for element in rigid_shape_props_asset:
                element.friction = 2.0  # * We increase the friction coefficient to compensate for missing skin deformation friction in simulation. See the Appx for details.
                element.rolling_friction = 0.05
                element.torsion_friction = 0.05
            self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)
            self.objs_assets[obj_id] = current_asset

        # * load assigned scale and mass for the object if available
        if obj_id in oakink2_obj_scale:
            scale = oakink2_obj_scale[obj_id]
        else:
            scale = 1.0

        if obj_id in oakink2_obj_mass:
            mass = oakink2_obj_mass[obj_id]
        else:
            mass = None

        sum_rigid_body_count = self.gym.get_asset_rigid_body_count(current_asset)
        sum_rigid_shape_count = self.gym.get_asset_rigid_shape_count(current_asset)
        return current_asset, sum_rigid_body_count, sum_rigid_shape_count, scale, mass

    def _create_obj_actor(self, env_ptr, i, current_asset, side="rh"):

        if side == "rh":
            obj_transf = self.demo_data_rh["obj_trajectory"][i][0]
        else:
            obj_transf = self.demo_data_lh["obj_trajectory"][i][0]

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(obj_transf[0, 3], obj_transf[1, 3], obj_transf[2, 3])
        obj_aa = rotmat_to_aa(obj_transf[:3, :3])
        obj_aa_angle = torch.norm(obj_aa)
        obj_aa_axis = obj_aa / obj_aa_angle
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle)

        # ? object actor filter bit is always 1
        if side == "rh":
            obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj_rh", i, 0)
        else:
            obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj_lh", i, 0)
        obj_index = self.gym.get_actor_index(env_ptr, obj_actor, gymapi.DOMAIN_SIM)

        if side == "rh":
            scene_objs = self.demo_data_rh["scene_objs"][i]
        else:
            scene_objs = self.demo_data_lh["scene_objs"][i]
        scene_asset_options = gymapi.AssetOptions()
        scene_asset_options.fix_base_link = True

        for so_id, scene_obj in enumerate(scene_objs):
            scene_obj_type = scene_obj["obj"].type
            scene_obj_size = scene_obj["obj"].size
            scene_obj_pose = scene_obj["pose"]
            if scene_obj_type == "cube":
                scene_asset = self.gym.create_box(
                    self.sim,
                    scene_obj_size[0],
                    scene_obj_size[1],
                    scene_obj_size[2],
                    scene_asset_options,
                )
                offset = np.eye(4)
                offset[:3, 3] = np.array(scene_obj_size) / 2
                scene_obj_pose = scene_obj_pose @ offset
            elif scene_obj_type == "cylinder":
                scene_asset = self.gym.create_box(
                    self.sim,
                    scene_obj_size[0] * 2,
                    scene_obj_size[0] * 2,
                    scene_obj_size[1],
                    scene_asset_options,
                )
            else:
                raise NotImplementedError
            scene_obj_pose = self.mujoco2gym_transf @ torch.tensor(
                scene_obj_pose, device=self.sim_device, dtype=torch.float32
            )
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(scene_obj_pose[0, 3], scene_obj_pose[1, 3], scene_obj_pose[2, 3])
            obj_aa = rotmat_to_aa(scene_obj_pose[:3, :3])
            obj_aa_angle = torch.norm(obj_aa)
            obj_aa_axis = obj_aa / obj_aa_angle
            pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle
            )
            self.gym.create_actor(env_ptr, scene_asset, pose, f"scene_obj_{so_id}", i, 0)
        # add dummy scene object
        MAX_SCENE_OBJS = 0 + (0 if not self.headless else 0)
        for so_id in range(MAX_SCENE_OBJS - len(scene_objs)):
            scene_asset = self.gym.create_box(self.sim, 0.02, 0.04, 0.06, scene_asset_options)
            # ? collision filter bit is always 0b11111111, never collide with anything (except the ground)
            a = self.gym.create_actor(
                env_ptr,
                scene_asset,
                gymapi.Transform(),
                f"scene_obj_{so_id +  len(scene_objs)}",
                self.num_envs + 1,
                0b1,
            )
            c = [
                gymapi.Vec3(1, 1, 0.5),
                gymapi.Vec3(0.5, 1, 1),
                gymapi.Vec3(1, 0, 1),
                gymapi.Vec3(1, 1, 0),
                gymapi.Vec3(0, 1, 1),
                gymapi.Vec3(0, 0, 1),
                gymapi.Vec3(0, 1, 0),
                gymapi.Vec3(1, 0, 0),
            ][so_id + len(scene_objs)]
            self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)

        # * just for visualization purposes, add a small sphere at the finger positions
        if not self.headless:
            dexhand_template = self.dexhand_rh if side == "rh" else self.dexhand_lh
            for joint_vis_id, joint_name in enumerate(dexhand_template.body_names):
                joint_name = dexhand_template.to_hand(joint_name)[0]
                joint_point = self.gym.create_sphere(self.sim, 0.005, scene_asset_options)
                a = self.gym.create_actor(
                    env_ptr,
                    joint_point,
                    gymapi.Transform(),
                    f"{side}_mano_joint_{joint_vis_id}",
                    self.num_envs + 1,
                    0b1,
                )
                if "index" in joint_name:
                    inter_c = 70
                elif "middle" in joint_name:
                    inter_c = 130
                elif "ring" in joint_name:
                    inter_c = 190
                elif "pinky" in joint_name:
                    inter_c = 250
                elif "thumb" in joint_name:
                    inter_c = 10
                else:
                    inter_c = 0
                if "tip" in joint_name:
                    c = gymapi.Vec3(inter_c / 255, 200 / 255, 200 / 255)
                elif "proximal" in joint_name:
                    c = gymapi.Vec3(200 / 255, inter_c / 255, 200 / 255)
                elif "intermediate" in joint_name:
                    c = gymapi.Vec3(200 / 255, 200 / 255, inter_c / 255)
                else:
                    c = gymapi.Vec3(100 / 255, 150 / 255, 200 / 255)
                self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)

        return obj_actor, obj_index


    def _update_states(self):
        self.rh_states.update(
            {
                "q": self._q[:, : self.num_dexhand_rh_dofs],
                "cos_q": torch.cos(self._q[:, : self.num_dexhand_rh_dofs]),
                "sin_q": torch.sin(self._q[:, : self.num_dexhand_rh_dofs]),
                "dq": self._qd[:, : self.num_dexhand_rh_dofs],
                "base_state": self._rh_base_state[:, :],
            }
        )

        self.rh_states["joints_state"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_rh_handles[k], :][:, :10] for k in self.dexhand_rh.body_names],
            dim=1,
        )

        tip_names = [
            "R_thumb_tip",
            "R_index_tip",
            "R_middle_tip",
            "R_ring_tip",
            "R_pinky_tip",
        ]

        tip_indices = [self.dexhand_rh.body_names.index(name) for name in tip_names]
        self.rh_states["joints_tip_state"] = self.rh_states["joints_state"][:, tip_indices, :]

        self.rh_states.update(
            {
                "manip_obj_pos": self._manip_obj_rh_root_state[:, :3],
                "manip_obj_quat": self._manip_obj_rh_root_state[:, 3:7],
                "manip_obj_vel": self._manip_obj_rh_root_state[:, 7:10],
                "manip_obj_ang_vel": self._manip_obj_rh_root_state[:, 10:],
            }
        )

        self.lh_states.update(
            {
                "q": self._q[:, self.num_dexhand_rh_dofs :],
                "cos_q": torch.cos(self._q[:, self.num_dexhand_rh_dofs :]),
                "sin_q": torch.sin(self._q[:, self.num_dexhand_rh_dofs :]),
                "dq": self._qd[:, self.num_dexhand_rh_dofs :],
                "base_state": self._lh_base_state[:, :],
            }
        )
        self.lh_states["joints_state"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_lh_handles[k], :][:, :10] for k in self.dexhand_lh.body_names],
            dim=1,
        )

        tip_names = [
            "L_thumb_tip",
            "L_index_tip",
            "L_middle_tip",
            "L_ring_tip",
            "L_pinky_tip",
        ]

        tip_indices = [self.dexhand_lh.body_names.index(name) for name in tip_names]
        self.lh_states["joints_tip_state"] = self.lh_states["joints_state"][:, tip_indices, :]

        self.lh_states.update(
            {
                "manip_obj_pos": self._manip_obj_lh_root_state[:, :3],
                "manip_obj_quat": self._manip_obj_lh_root_state[:, 3:7],
                "manip_obj_vel": self._manip_obj_lh_root_state[:, 7:10],
                "manip_obj_ang_vel": self._manip_obj_lh_root_state[:, 10:],
            }
        )

    def _refresh(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        lh_rew_buf, lh_reset_buf, lh_success_buf, lh_failure_buf, lh_reward_dict, lh_error_buf = (
            self.compute_reward_side(actions, side="lh")
        )
        rh_rew_buf, rh_reset_buf, rh_success_buf, rh_failure_buf, rh_reward_dict, rh_error_buf = (
            self.compute_reward_side(actions, side="rh")
        )
        self.rew_buf = rh_rew_buf + lh_rew_buf
        self.reset_buf = rh_reset_buf | lh_reset_buf
        self.success_buf = rh_success_buf & lh_success_buf
        self.failure_buf = rh_failure_buf | lh_failure_buf
        self.error_buf = rh_error_buf | lh_error_buf
        self.reward_dict = {
            **{"rh_" + k: v for k, v in rh_reward_dict.items()},
            **{"lh_" + k: v for k, v in lh_reward_dict.items()},
        }

    def compute_reward_cp(self, actions):
        lh_rew_buf, lh_reset_buf, lh_success_buf, lh_failure_buf, lh_reward_dict, lh_error_buf = (
            self.compute_reward_side_cp(actions, side="lh")
        )
        rh_rew_buf, rh_reset_buf, rh_success_buf, rh_failure_buf, rh_reward_dict, rh_error_buf = (
            self.compute_reward_side_cp(actions, side="rh")
        )
        self.rew_buf = rh_rew_buf + lh_rew_buf
        self.reset_buf = rh_reset_buf | lh_reset_buf
        self.success_buf = rh_success_buf & lh_success_buf
        self.failure_buf = rh_failure_buf | lh_failure_buf
        self.error_buf = rh_error_buf | lh_error_buf
        self.reward_dict = {
            **{"rh_" + k: v for k, v in rh_reward_dict.items()},
            **{"lh_" + k: v for k, v in lh_reward_dict.items()},
        }

    def compute_reward_side(self, actions, side="rh"):
        side_demo_data = self.demo_data_rh if side == "rh" else self.demo_data_lh
        target_state = {}
        max_length = torch.clip(side_demo_data["seq_len"], 0, self.max_episode_length).float()
        cur_idx = self.progress_buf
        cur_wrist_pos = side_demo_data["wrist_pos"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_pos"] = cur_wrist_pos
        cur_wrist_rot = side_demo_data["wrist_rot"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_quat"] = aa_to_quat(cur_wrist_rot)[:, [1, 2, 3, 0]]

        target_state["wrist_vel"] = side_demo_data["wrist_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_ang_vel"] = side_demo_data["wrist_angular_velocity"][torch.arange(self.num_envs), cur_idx]

        target_state["tips_distance"] = side_demo_data["tips_distance"][torch.arange(self.num_envs), cur_idx]

        cur_joints_pos = side_demo_data["mano_joints"][torch.arange(self.num_envs), cur_idx]
        target_state["joints_pos"] = cur_joints_pos.reshape(self.num_envs, -1, 3)
        target_state["joints_vel"] = side_demo_data["mano_joints_velocity"][
            torch.arange(self.num_envs), cur_idx
        ].reshape(self.num_envs, -1, 3)

        cur_obj_transf = side_demo_data["obj_trajectory"][torch.arange(self.num_envs), cur_idx]
        target_state["manip_obj_pos"] = cur_obj_transf[:, :3, 3]
        target_state["manip_obj_quat"] = rotmat_to_quat(cur_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]

        target_state["manip_obj_vel"] = side_demo_data["obj_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["manip_obj_ang_vel"] = side_demo_data["obj_angular_velocity"][torch.arange(self.num_envs), cur_idx]

        target_state["tip_force"] = torch.stack(
            [
                self.net_cf[:, getattr(self, f"dexhand_{side}_handles")[k], :]
                for k in (self.dexhand_rh.contact_body_names if side == "rh" else self.dexhand_lh.contact_body_names)
            ],
            axis=1,
        )
        setattr(
            self,
            f"{side}_tips_contact_history",
            torch.concat(
                [
                    getattr(self, f"{side}_tips_contact_history")[:, 1:],
                    (torch.norm(target_state["tip_force"], dim=-1) > 0)[:, None],
                ],
                dim=1,
            ),
        )
        target_state["tip_contact_state"] = getattr(self, f"{side}_tips_contact_history")

        side_states = getattr(self, f"{side}_states")
        if side == "rh":
            power = torch.abs(torch.multiply(self.dof_force[:, : self.dexhand_rh.n_dofs], side_states["dq"])).sum(
                dim=-1
            )
        else:
            power = torch.abs(torch.multiply(self.dof_force[:, self.dexhand_rh.n_dofs :], side_states["dq"])).sum(
                dim=-1
            )
        target_state["power"] = power

        base_handle = getattr(self, f"dexhand_{side}_handles")[
            self.dexhand_rh.to_dex("wrist")[0] if side == "rh" else self.dexhand_lh.to_dex("wrist")[0]
        ]

        wrist_power = torch.abs(
            torch.sum(
                self.apply_forces[:, base_handle, :] * side_states["base_state"][:, 7:10],
                dim=-1,
            )
        )  # ? linear force * linear velocity
        wrist_power += torch.abs(
            torch.sum(
                self.apply_torque[:, base_handle, :] * side_states["base_state"][:, 10:],
                dim=-1,
            )
        )  # ? torque * angular velocity
        target_state["wrist_power"] = wrist_power

        if self.training:
            last_step = self.gym.get_frame_count(self.sim)
            if self.tighten_method == "None":
                scale_factor = 1.0
            elif self.tighten_method == "const":
                scale_factor = self.tighten_factor
            elif self.tighten_method == "linear_decay":
                scale_factor = 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last_step, self.tighten_steps)
            elif self.tighten_method == "exp_decay":
                scale_factor = (np.e * 2) ** (-1 * last_step / self.tighten_steps) * (
                    1 - self.tighten_factor
                ) + self.tighten_factor
            elif self.tighten_method == "cos":
                scale_factor = (self.tighten_factor) + np.abs(
                    -1 * (1 - self.tighten_factor) * np.cos(last_step / self.tighten_steps * np.pi)
                ) * (2 ** (-1 * last_step / self.tighten_steps))
            else:
                raise NotImplementedError
        else:
            scale_factor = 1.0

        assert not self.headless or isinstance(compute_imitation_reward, torch.jit.ScriptFunction)

        if self.rollout_len is not None:
            max_length = torch.clamp(max_length, 0, self.rollout_len + self.rollout_begin + 3 + 1)

        rew_buf, reset_buf, success_buf, failure_buf, reward_dict, error_buf = compute_imitation_reward(
            self.reset_buf,
            self.progress_buf,
            self.running_progress_buf,
            self.actions,
            side_states,
            target_state,
            max_length,
            scale_factor,
            (self.dexhand_rh if side == "rh" else self.dexhand_lh).weight_idx,
        )
        self.total_rew_buf += rew_buf
        return rew_buf, reset_buf, success_buf, failure_buf, reward_dict, error_buf

    def compute_reward_side_cp(self, actions, side="rh"):
        side_demo_data = self.demo_data_rh if side == "rh" else self.demo_data_lh
        target_state = {}
        max_length = torch.clip(side_demo_data["seq_len"], 0, self.max_episode_length).float()
        cur_idx = self.progress_buf
        

        side_states = getattr(self, f"{side}_states")
        target_state["contact_point"] = side_demo_data["contact_point_tips"][torch.arange(self.num_envs), cur_idx]

        base_handle = getattr(self, f"dexhand_{side}_handles")[
            self.dexhand_rh.to_dex("wrist")[0] if side == "rh" else self.dexhand_lh.to_dex("wrist")[0]
        ]

        wrist_power = torch.abs(
            torch.sum(
                self.apply_forces[:, base_handle, :] * side_states["base_state"][:, 7:10],
                dim=-1,
            )
        )  # ? linear force * linear velocity
        wrist_power += torch.abs(
            torch.sum(
                self.apply_torque[:, base_handle, :] * side_states["base_state"][:, 10:],
                dim=-1,
            )
        )  # ? torque * angular velocity

        if self.training:
            last_step = self.gym.get_frame_count(self.sim)
            if self.tighten_method == "None":
                scale_factor = 1.0
            elif self.tighten_method == "const":
                scale_factor = self.tighten_factor
            elif self.tighten_method == "linear_decay":
                scale_factor = 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last_step, self.tighten_steps)
            elif self.tighten_method == "exp_decay":
                scale_factor = (np.e * 2) ** (-1 * last_step / self.tighten_steps) * (
                    1 - self.tighten_factor
                ) + self.tighten_factor
            elif self.tighten_method == "cos":
                scale_factor = (self.tighten_factor) + np.abs(
                    -1 * (1 - self.tighten_factor) * np.cos(last_step / self.tighten_steps * np.pi)
                ) * (2 ** (-1 * last_step / self.tighten_steps))
            else:
                raise NotImplementedError
        else:
            scale_factor = 1.0

        assert not self.headless or isinstance(compute_imitation_reward_cp, torch.jit.ScriptFunction)

        if self.rollout_len is not None:
            max_length = torch.clamp(max_length, 0, self.rollout_len + self.rollout_begin + 3 + 1)


        rew_buf, reset_buf, success_buf, failure_buf, reward_dict, error_buf = compute_imitation_reward_cp(
            self.reset_buf,
            self.progress_buf,
            self.running_progress_buf,
            self.actions,
            side_states,
            target_state,
            max_length,
            scale_factor,
            (self.dexhand_rh if side == "rh" else self.dexhand_lh).weight_idx,
        )
        self.total_rew_buf += rew_buf
        return rew_buf, reset_buf, success_buf, failure_buf, reward_dict, error_buf

    def _save_segmentation_quick(self, out_dir: str, step: int, max_envs: int = 1024):
        """
        Will output two files:
        - seg_color_envXX_stepXXXXXX.png : preview image that maps actor id to color
        - seg_raw_envXX_stepXXXXXX.png   : store original actor id as 32-bit integer (can be read back for debugging)
        """
        os.makedirs(out_dir, exist_ok=True)
        if not hasattr(self, "last_segs") or self.last_segs is None:
            print("No segmentation tensors yet.")
            return
        num = min(len(self.last_segs), max_envs)
        for i in range(num):
            seg = self.last_segs[i].detach().cpu().numpy().astype(np.int32)  # (H,W) actor instance id, background is often -1

            # --- (a) Color visualization ---
            ids = np.unique(seg)
            # Build a continuous index mapping id -> 0..K-1 (to avoid very large ids)
            id_to_small = {int(k): idx for idx, k in enumerate(ids)}
            small_map = np.vectorize(lambda x: id_to_small[int(x)], otypes=[np.int32])(seg)

            # Use tab20 or hsv to generate colors (K color slots)
            K = len(ids)
            cmap = plt.get_cmap("tab20", K) if K <= 20 else plt.get_cmap("hsv", K)
            rgb = (cmap(small_map % K)[..., :3] * 255).astype(np.uint8)  # (H,W,3)

            # Background (usually -1) can be set to dark gray
            if -1 in id_to_small:
                bg_idx = id_to_small[-1]
                rgb[small_map == bg_idx] = np.array([30,30,30], dtype=np.uint8)

            color_path = os.path.join(out_dir, f"seg_color_env{i:02d}_step{step:06d}.png")
            Image.fromarray(rgb, mode="RGB").save(color_path)

            # --- (b) Store original IDs in 32-bit (for precise inspection) ---
            raw_path = os.path.join(out_dir, f"seg_raw_env{i:02d}_step{step:06d}.png")
            Image.fromarray(seg, mode="I").save(raw_path)  # 'I' = 32-bit signed int PNG

            print(f"[seg] env {i} -> {color_path} / {raw_path}")

    def _save_rgb_preview_png(self, out_dir: str, step: int, max_envs: int = 4, which: str = "front"):
        import os
        from PIL import Image
        imgs = self.last_rgb_images if which == "front" else self.last_rgb_images_top
        if imgs is None:
            return
        os.makedirs(out_dir, exist_ok=True)
        num = min(len(imgs), max_envs)
        for i in range(num):
            rgb = imgs[i].detach().cpu()
            # 轉成 (H,W,3) uint8
            if rgb.dtype != torch.uint8:
                rgb = rgb.clamp(0, 255).to(torch.uint8)
            if rgb.ndim == 3 and rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            elif rgb.ndim == 3 and rgb.shape[0] == 4:
                rgb = rgb.permute(1, 2, 0)[..., :3]
            Image.fromarray(rgb.numpy(), mode="RGB").save(
                os.path.join(out_dir, f"rgb_{which}_env{i:02d}_step{step:06d}.png")
            )


    def compute_observations(self):
        self._refresh()
        obs_rh = self.compute_observations_side("rh")
        obs_lh = self.compute_observations_side("lh")
        for k in obs_rh.keys():
            self.obs_dict[k] = torch.cat([obs_rh[k], obs_lh[k]], dim=-1)

        # === Render + grab depth images ===
        self._render_all_cameras()
        # self.last_depths = self._grab_depth_images(device="gpu")
        # self.last_segs   = self._grab_seg_images(device="gpu")

        # 既有（前視）
        # self.last_depths = self._grab_depth_images(device="gpu", cameras=self.cameras)
        # self.last_segs   = self._grab_seg_images(device="gpu", cameras=self.cameras)
        # self.last_rgb_images = self._grab_rgb_images(device="gpu", cameras=self.cameras)

        # # 新增（俯視）
        # self.last_depths_top = self._grab_depth_images(device="gpu", cameras=self.cameras_top)
        # self.last_segs_top   = self._grab_seg_images(device="gpu", cameras=self.cameras_top)
        # self.last_rgb_images_top = self._grab_rgb_images(device="gpu", cameras=self.cameras_top)

        self.last_depths, self.last_segs, self.last_rgb_images = self._grab_depth_seg_rgb_once(self.cameras)
        self.last_depths_top, self.last_segs_top, self.last_rgb_images_top = self._grab_depth_seg_rgb_once(self.cameras_top)

        # After images are available, calibrate per-env camera origin mode once
        if self.last_depths is not None and self.last_segs is not None:
            for env_id in range(min(len(self.last_depths), self.num_envs)):
                if self._cam_origin_mode["front"].get(env_id, None) is None:
                    self._calibrate_camera_origin_mode_for_env(env_id, view="front")
        if self.last_depths_top is not None and self.last_segs_top is not None:
            for env_id in range(min(len(self.last_depths_top), self.num_envs)):
                if self._cam_origin_mode["top"].get(env_id, None) is None:
                    self._calibrate_camera_origin_mode_for_env(env_id, view="top")

        
        step = int(self.gym.get_frame_count(self.sim))
        out_dir = "cam_debug"

        if self.debug:
            self._save_rgb_preview_png(out_dir, step, max_envs=4, which="top")

            self._debug_dump_seg_stats(max_envs=4)
        # self._save_segmentation_quick(out_dir, step, max_envs=4)
        # import ipdb; ipdb.set_trace()

        # Print info only once at the first step
        if not hasattr(self, "_printed_depth_info"):
            self._printed_depth_info = False
        if (self.last_depths is not None) and (not self._printed_depth_info):
            d0 = self.last_depths[0]
            finite = torch.isfinite(d0)
            if finite.any():
                v = d0[finite]
                print("depth shape:", tuple(d0.shape), "min/max:", float(v.min()), float(v.max()))
            else:
                print("depth has no finite values")
            self._printed_depth_info = True

        self.save_contacts(
            out_dir, step, max_envs=1024,
            object_depth_range=(0.6, 1.2),
            center_crop_ratio=0.6,
            shade="cam_depth"  
        )

        # === Save every 50 steps (adjustable) ===
        step = int(self.gym.get_frame_count(self.sim))
        if self.debug and ((step % 10) == 0 and self.last_depths is not None):
            out_dir = "depth_debug"  # your desired output folder
            out_dir = "depth_debug"  # your desired output folder
            # Save color preview
            self._save_depth_preview_png(out_dir, step, max_envs=4)
            # Also save 16-bit depth (optional)
            self._save_depth_16bit_png(out_dir, step, max_envs=4, scale=1000.0)  # millimeters
            # ★ New: save point cloud visualization
            self._save_no_rgb_object_pointcloud_png(
                out_dir, step, max_envs=4,
                views=((20, -60),),
                color_method='depth'  # or 'height', 'uniform', 'distance', 'coordinate'
            )
            self._find_contacts_and_save_cam(
                out_dir, step, max_envs=4,
                object_depth_range=(0.6, 1.2),
                center_crop_ratio=0.6,
                shade="cam_depth"   # or "height"
            )

    def _depth_to_object_pointcloud_simple(self, depth_tensor, 
                                     horizontal_fov_deg=69.4,
                                     object_depth_range=(0.6, 1.0),  # object depth range
                                     center_crop_ratio=0.6,  # center crop ratio
                                     max_points=15000):
        """
        Simplified version: extract object point cloud based on depth range and central area
        """
        import torch
        import numpy as np
        
        if depth_tensor is None:
            return None
            
        H, W = depth_tensor.shape
        device = depth_tensor.device
        
        # ===== 1. Compute camera intrinsics =====
        fovx = np.deg2rad(horizontal_fov_deg)
        fx = W / (2.0 * np.tan(fovx / 2.0))
        fovy = 2.0 * np.arctan(np.tan(fovx / 2.0) * (H / W))
        fy = H / (2.0 * np.tan(fovy / 2.0))
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        
        # ===== 2. Center-region crop =====
        h_start = int(H * (1 - center_crop_ratio) / 2)
        h_end = int(H * (1 + center_crop_ratio) / 2)
        w_start = int(W * (1 - center_crop_ratio) / 2)
        w_end = int(W * (1 + center_crop_ratio) / 2)
        
        # Create center-region mask
        center_mask = torch.zeros_like(depth_tensor, dtype=torch.bool)
        center_mask[h_start:h_end, w_start:w_end] = True
        
        # ===== 3. Object depth-range filtering =====
        object_depth_mask = (
            (depth_tensor >= object_depth_range[0]) & 
            (depth_tensor <= object_depth_range[1])
        )
        
        # ===== 4. Combine all filters =====
        valid_mask = (
            torch.isfinite(depth_tensor) &
            center_mask &
            object_depth_mask
        )
        
        if not valid_mask.any():
            print("No valid object points found")
            return None
        
        # ===== 5. Build coordinate grid and extract valid points =====
        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')
        
        v_valid = v_grid[valid_mask]
        u_valid = u_grid[valid_mask]
        z_valid = depth_tensor[valid_mask]
        
        # ===== 6. Convert to 3D coordinates =====
        x_valid = (u_valid - cx) / fx * z_valid
        y_valid = (v_valid - cy) / fy * z_valid
        
        points_3d = torch.stack([x_valid, -y_valid, z_valid], dim=1)
        
        print(f"Found {points_3d.shape[0]} object points in depth range {object_depth_range}")
        
        # ===== 7. Clustering denoise (optional) =====
        if points_3d.shape[0] > 100:
            # Remove outliers
            center = torch.median(points_3d, dim=0)[0]
            distances = torch.norm(points_3d - center, dim=1)
            threshold = torch.quantile(distances, 0.9)  # keep 90% of points
            inlier_mask = distances <= threshold
            points_3d = points_3d[inlier_mask]
            print(f"After outlier removal: {points_3d.shape[0]} points")
        
        # ===== 8. Downsample =====
        if max_points is not None and points_3d.shape[0] > max_points:
            indices = torch.randperm(points_3d.shape[0], device=device)[:max_points]
            points_3d = points_3d[indices]
        
        return points_3d

    
    # === Camera extrinsics: infer each env's world pose from create_camera()'s cam_pos/cam_target + env origin ===
    def _get_env_camera_pose_world(self, env_id: int):
        """
        Return (R_wc, t_wc)
        R_wc: [3,3], its column vectors (not rows) are [right, up, forward]
        t_wc: [3]   camera position in world coordinates
        """
        # The world's origin of the env (Isaac places each env on a grid)
        # o = self.gym.get_env_origin(self.envs[env_id])
        # origin_w = torch.tensor([o.x, o.y, o.z], device=self.device, dtype=torch.float32)

        # Retrieve the "env-local" position and target saved when set_camera_location was called
        cam_pos_local = self._cam_pos_local[env_id]
        cam_tgt_local = self._cam_target_local[env_id]

        add_origin = self._cam_origin_mode.get("front", {}).get(env_id, False)
        if add_origin:
            o = self.gym.get_env_origin(self.envs[env_id])
            origin_w = torch.tensor([o.x, o.y, o.z], device=self.device, dtype=torch.float32)
            t_wc = origin_w + cam_pos_local
            tgt_w = origin_w + cam_tgt_local
        else:
            t_wc = cam_pos_local
            tgt_w = cam_tgt_local
        # Use the world's z as an up guess, then build an orthonormal basis (right-handed coordinates)
        up_guess = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        forward = tgt_w - t_wc
        forward = forward / (torch.norm(forward) + 1e-8)
        right = torch.cross(forward, up_guess)
        right = right / (torch.norm(right) + 1e-8)
        up = torch.cross(right, forward)
        up = up / (torch.norm(up) + 1e-8)

        # The columns of R_wc are the camera axes in world coordinates (consistent with point cloud convention: z=forward)
        R_wc = torch.stack([right, up, forward], dim=1)  # shape [3,3]
        return R_wc, t_wc

    def _nn_contacts_for_hand_world(self, joints_world: torch.Tensor, pc_world: torch.Tensor):
        D = torch.cdist(joints_world, pc_world)  # [J, N]
        idx = torch.argmin(D, dim=1)
        cpts = pc_world[idx]
        d = D[torch.arange(D.shape[0], device=D.device), idx]
        return idx, cpts, d


    def _nn_contacts_for_hand(self, joints_cam: torch.Tensor, pc_cam: torch.Tensor):
        """
        joints_cam: [J,3] joints in camera coordinates (recommend dropping the wrist, keep J=movable hand joints)
        pc_cam:     [N,3] object point cloud in camera coordinates
        Return:
        idx: [J]   index of the nearest point for each joint
        cpts: [J,3] nearest point coordinates (camera coordinates)
        d: [J]     nearest distances
        """
        # torch.cdist is fast on GPU; J in tens and N in tens of thousands is fine (if N is huge, use FPS/voxel downsample first)
        D = torch.cdist(joints_cam, pc_cam)          # [J,N]
        idx = torch.argmin(D, dim=1)                 # [J]
        cpts = pc_cam[idx]                           # [J,3]
        d    = D[torch.arange(D.shape[0], device=D.device), idx]
        return idx, cpts, d


    def _pointcloud_shading_values(self, pc_world: torch.Tensor, env_id: int, mode: str = "cam_depth"):
        """
        Return a scalar for each point for coloring:
        - "cam_depth": +Z in camera coordinates (depth along the camera forward)
        - "cam_dist":  Euclidean distance from camera to the point
        - "height":    Z in world coordinates (height)
        """
        if pc_world is None or pc_world.shape[0] == 0:
            return None

        R_wc, t_wc = self._get_env_camera_pose_world(env_id)   # 3x3, 3
        forward = R_wc[:, 2]                                   # camera forward (world coordinates)
        pts = pc_world

        if mode == "cam_depth":
            # Project onto camera forward: depth = (p - t) · forward
            vals = torch.sum((pts - t_wc[None, :]) * forward[None, :], dim=1)
        elif mode == "cam_dist":
            vals = torch.norm(pts - t_wc[None, :], dim=1)
        elif mode == "height":
            vals = pts[:, 2]
        else:
            vals = torch.sum((pts - t_wc[None, :]) * forward[None, :], dim=1)  # default cam_depth

        return vals

    def _camera_to_world(self, pts_cam: torch.Tensor, env_id: int):
        R_wc, t_wc = self._get_env_camera_pose_world(env_id)
        return pts_cam @ R_wc.t() + t_wc[None, :]

    def _world_to_camera(self, pts_world: torch.Tensor, env_id: int):
        R_wc, t_wc = self._get_env_camera_pose_world(env_id)
        return (pts_world - t_wc[None, :]) @ R_wc.t()

    def _build_hand_skeleton_edges(self, body_names: List[str]):
        """
        Automatically build skeleton edges based on body_names.
        Return: List[Tuple[int,int]] representing parent_idx -> child_idx
        """
        name_to_idx = {n: i for i, n in enumerate(body_names)}
        def find_idx(key):
            return name_to_idx[key] if key in name_to_idx else None

        edges = []
        wrist = find_idx("wrist")

        def chain(finger: str, segments: List[str]):
            # Connect from wrist to the first segment (if wrist exists and the first segment exists)
            prev = wrist
            for seg in segments:
                idx = find_idx(f"{finger}_{seg}")
                if idx is not None:
                    if prev is not None:
                        edges.append((prev, idx))
                    prev = idx

        # Five fingers
        chain("thumb",  ["proximal", "distal", "tip"])           # thumb usually has only two segments
        chain("index",  ["proximal", "intermediate", "distal", "tip"])
        chain("middle", ["proximal", "intermediate", "distal", "tip"])
        chain("ring",   ["proximal", "intermediate", "distal", "tip"])
        chain("pinky",  ["proximal", "intermediate", "distal", "tip"])

        # Palm outline (connect the proximal of each finger)
        prox = [find_idx(f"{f}_proximal") for f in ["thumb","index","middle","ring","pinky"]]
        prox = [p for p in prox if p is not None]
        for a, b in zip(prox, prox[1:]):
            edges.append((a, b))

        return edges





    def _nn_contacts_for_hand_cam(self, joints_cam: torch.Tensor, pc_cam: torch.Tensor):
        """
        joints_cam: [J,3] hand joints in camera coordinates (usually drop the wrist)
        pc_cam:     [N,3] object point cloud in camera coordinates
        Return: (idx[J], cpts[J,3], d[J])
        """
        D = torch.cdist(joints_cam, pc_cam)  # [J,N]
        idx = torch.argmin(D, dim=1)
        cpts = pc_cam[idx]
        d = D[torch.arange(D.shape[0], device=D.device), idx]
        return idx, cpts, d




    def _save_contacts_visualization_cam(
        self, out_path: str,
        pc_cam: torch.Tensor,
        rh_j_all_cam: torch.Tensor, lh_j_all_cam: torch.Tensor,
        rh_edges: list[tuple[int,int]], lh_edges: list[tuple[int,int]],
        rh_j_vis_cam: torch.Tensor, rh_c_vis_cam: torch.Tensor,
        lh_j_vis_cam: torch.Tensor, lh_c_vis_cam: torch.Tensor,
        views: tuple = ((20, -60),),
        shade: str = "cam_depth",
        joint_size: float = 130.0,
        contact_size: float = 200.0,
        line_width: float = 2,

        # ★ New: control of depth gradient
        depth_colormap: str = "viridis",
        depth_percentile_low: float = 2.0,
        depth_percentile_high: float = 98.0,
        depth_gamma: float = 0.8,          # 0.6~0.9 gives a “slight gradient” feel
        size_near: float = 5.0,            # slightly larger for nearer points
        size_far: float = 3.0              # slightly smaller for farther points
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import Normalize

        pc = pc_cam.detach().cpu().numpy()

        rj_all = rh_j_all_cam.detach().cpu().numpy()
        lj_all = lh_j_all_cam.detach().cpu().numpy()
        rj = rh_j_vis_cam.detach().cpu().numpy()
        rc = rh_c_vis_cam.detach().cpu().numpy()
        lj = lh_j_vis_cam.detach().cpu().numpy()
        lc = lh_c_vis_cam.detach().cpu().numpy()

        # === (A) Prepare point cloud colors first (apply depth gradient only to the object point cloud) ===
        pc_colors = None
        pc_sizes  = None
        if pc.size > 0 and shade is not None:
            # Choose scalar source: cam_depth -> Z_cam; height -> Y_cam (your original definition)
            if shade == "cam_depth":
                vals = pc[:, 2].astype(np.float32)
            elif shade == "height":
                vals = pc[:, 1].astype(np.float32)
            else:
                vals = pc[:, 2].astype(np.float32)

            # Sort by depth so that far points are drawn first and near points overplot (more 3D feel)
            order = np.argsort(vals)       # far -> near
            pc = pc[order]
            vals = vals[order]

            # Percentile window + gamma to get “some” contrast without being harsh
            vmin = np.percentile(vals, depth_percentile_low)
            vmax = np.percentile(vals, depth_percentile_high)
            vmin, vmax = float(vmin), float(vmax + 1e-8)
            vals01 = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
            if depth_gamma is not None and depth_gamma > 0:
                # For cam_depth where Z is positive forward, gamma<1 increases near-end contrast
                vals01 = np.power(vals01, depth_gamma)

            cmap = plt.get_cmap(depth_colormap)
            pc_colors = cmap(vals01)[..., :3]  # (N,3) 0~1

            # Make nearer points slightly bigger (adds depth layering)
            if size_near is not None and size_far is not None:
                pc_sizes = (size_far + (size_near - size_far) * (1.0 - vals01))  # near -> bigger

        fig = plt.figure(figsize=(12 * len(views), 10), dpi=300)
        for i, (elev, azim) in enumerate(views, 1):
            ax = fig.add_subplot(1, len(views), i, projection='3d')

            # (1) Object point cloud (base layer): use depth-gradient color and size
            if pc.size > 0:
                if pc_colors is None:
                    ax.scatter(pc[:,0], pc[:,1], pc[:,2],
                            s=30, c="#BBBBBB", alpha=0.98, edgecolors='none',
                            rasterized=True, zorder=1)
                else:
                    # if pc_sizes is None:
                    ax.scatter(pc[:,0], pc[:,1], pc[:,2],
                            s=30, c=pc_colors, alpha=0.98, edgecolors='none',
                            rasterized=True, zorder=1)
                    # else:
                    #     ax.scatter(pc[:,0], pc[:,1], pc[:,2],
                    #             s=pc_sizes, c=pc_colors, alpha=0.98, edgecolors='none',
                    #             rasterized=True, zorder=1)

            # (2) Skeleton edges (middle layer)
            def draw_skeleton(j_all, edges, color="#34495E"):
                for a, b in edges:
                    pa, pb = j_all[a], j_all[b]
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                            color=color, linewidth=line_width, alpha=0.95, zorder=5)

            if rj_all.shape[0] > 0 and len(rh_edges) > 0:
                draw_skeleton(rj_all, rh_edges, color="#2E86DE")
            if lj_all.shape[0] > 0 and len(lh_edges) > 0:
                draw_skeleton(lj_all, lh_edges, color="#1ABC9C")

            # (3) Joint points (top layer)
            if rj_all.shape[0] > 0:
                ax.scatter(rj_all[:,0], rj_all[:,1], rj_all[:,2],
                        s=joint_size, c="#2E86DE", alpha=1.0, edgecolors='none', zorder=6, label="RH joints")
            if lj_all.shape[0] > 0:
                ax.scatter(lj_all[:,0], lj_all[:,1], lj_all[:,2],
                        s=joint_size, c="#1ABC9C", alpha=1.0, edgecolors='none', zorder=6, label="LH joints")

            # (4) Contact points (very top)
            if rj.size > 0:
                ax.scatter(rc[:,0], rc[:,1], rc[:,2], s=contact_size, c="#E74C3C",
                        alpha=1.0, edgecolors='none', zorder=7, label="Contacts")
                for p, q in zip(rj, rc):
                    ax.plot([p[0], q[0]],[p[1], q[1]],[p[2], q[2]],
                            linestyle="--", linewidth=0.9, color="#888888", alpha=0.9, zorder=6.5)
            if lj.size > 0:
                ax.scatter(lc[:,0], lc[:,1], lc[:,2], s=contact_size, c="#E74C3C",
                        alpha=1.0, edgecolors='none', zorder=7)
                for p, q in zip(lj, lc):
                    ax.plot([p[0], q[0]],[p[1], q[1]],[p[2], q[2]],
                            linestyle="--", linewidth=0.9, color="#888888", alpha=0.9, zorder=6.5)

            ax.view_init(elev=70, azim=-90)
            ax.set_xlabel("X_cam (m)"); ax.set_ylabel("Y_cam (m)"); ax.set_zlabel("Z_cam (m)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left")

        plt.tight_layout(pad=0.5)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)



    def _debug_print_hand_names_and_indices(self):
        print("\n[Right hand rigid bodies (order used by joints_state)]")
        for i, n in enumerate(self.dexhand_rh.body_names):
            h = self.dexhand_rh_handles[n]
            print(f"{i:02d}  name={n:30s}  handle(index)={h}")

        print("\n[Left hand rigid bodies (order used by joints_state)]")
        for i, n in enumerate(self.dexhand_lh.body_names):
            h = self.dexhand_lh_handles[n]
            print(f"{i:02d}  name={n:30s}  handle(index)={h}")



    def _find_contacts_and_save_cam(self, out_dir: str, step: int, max_envs: int = 1024,
                                object_depth_range=(0.6, 1.2),
                                center_crop_ratio=0.6,
                                shade: str = "cam_depth",
                                contact_thresh: float = 0.01):
        import os
        from PIL import Image
        os.makedirs(out_dir, exist_ok=True)
        if self.last_depths is None:
            return

        # todo: its only works for env_id = 0
        num = min(len(self.last_depths), max_envs)
        for env_id in range(num):

            # ---------- (A) 讀前視相機影像 ----------
            depth_front = self.last_depths[env_id]
            if depth_front is None:
                continue
            seg_front = self.last_segs[env_id]

            allowed_ids = self._get_allowed_object_ids_for_env(env_id)

            # 由前視相機建立點雲 (在 "front" 相機座標系)
            pc_cam_front = self._depth_to_pointcloud_from_seg(
                depth_tensor=depth_front,
                seg_tensor=seg_front,
                allowed_ids=allowed_ids,
                horizontal_fov_deg=69.4,
                max_points=200_000,
                border_dilate_px=0,
                # z_min=0.45,   # 依你相機與桌面的距離微調
                # z_max=1.30
            )

            if pc_cam_front is None or pc_cam_front.shape[0] == 0:
                print(f"[contacts/cam] env {env_id}: no OBJECT point cloud from FRONT (seg empty).")
                # 就算前視沒點，仍嘗試用 top（但視覺化會比較空）
                pc_cam_front = None

            # ---------- (B) 讀俯視相機影像 & 轉到前視相機座標 ----------
            pc_cam_top_in_front = None
            have_top = (getattr(self, "last_depths_top", None) is not None) and \
                    (getattr(self, "last_segs_top", None) is not None)
            if have_top and env_id < len(self.last_depths_top):
                depth_top = self.last_depths_top[env_id]
                seg_top   = self.last_segs_top[env_id]
                if (depth_top is not None) and (seg_top is not None):
                    # 先在 "top" 相機座標系建立點雲
                    pc_cam_top = self._depth_to_pointcloud_from_seg(
                        depth_tensor=depth_top,
                        seg_tensor=seg_top,
                        allowed_ids=allowed_ids,
                        horizontal_fov_deg=69.4,
                        max_points=200_000,
                        border_dilate_px=0,
                        # 俯視通常較遠，放寬 z 範圍（視你的高度調整）
                        # z_min=0.10,
                        # z_max=2.00
                    )
                    if (pc_cam_top is not None) and (pc_cam_top.shape[0] > 0):
                        # top(cam) → world → front(cam)
                        pc_world_from_top   = self._camera_to_world_view(pc_cam_top, env_id, view="top")
                        pc_cam_top_in_front = self._world_to_camera_view(pc_world_from_top, env_id, view="front")

            # ---------- (C) 合併到「前視相機座標」的點雲 ----------
            pc_cam_merged = None
            pcs = []
            if pc_cam_front is not None and pc_cam_front.shape[0] > 0:
                pcs.append(pc_cam_front)
            if pc_cam_top_in_front is not None and pc_cam_top_in_front.shape[0] > 0:
                pcs.append(pc_cam_top_in_front)

            if len(pcs) == 0:
                print(f"[contacts/cam] env {env_id}: no OBJECT point cloud from FRONT nor TOP.")
                # 仍儲存 top 的 RGB 以利排錯
                if getattr(self, "last_rgb_images_top", None) is not None and env_id < len(self.last_rgb_images_top):
                    rgb_top = self.last_rgb_images_top[env_id]
                    _save_single_rgb(rgb_top, os.path.join(out_dir, f"rgb_top_env{env_id:02d}_step{step:06d}.png"))
                continue

            pc_cam_merged = torch.cat(pcs, dim=0)

            # 避免點太多（可視情況下採）
            if pc_cam_merged.shape[0] > 200_000:
                idx = torch.randperm(pc_cam_merged.shape[0], device=pc_cam_merged.device)[:200_000]
                pc_cam_merged = pc_cam_merged[idx]

            # ---------- (D) joints 轉到「前視相機座標」 ----------
            # joints world（你的狀態裡本來就有）
            rh_world_all = self.rh_states["joints_state"][env_id, :, :3]   # [18,3]
            lh_world_all = self.lh_states["joints_state"][env_id, :, :3]

            # 做完整骨架視覺化用（含 base/palm）
            rh_cam_all = self._world_to_camera_view(rh_world_all, env_id, view="front")
            lh_cam_all = self._world_to_camera_view(lh_world_all, env_id, view="front")

            # 計算接觸只用可動關節（維持你原本作法：丟掉 index 0 的 base）
            rh_world = rh_world_all[1:, :]
            lh_world = lh_world_all[1:, :]
            rh_cam   = self._world_to_camera_view(rh_world, env_id, view="front")
            lh_cam   = self._world_to_camera_view(lh_world, env_id, view="front")

            # ---------- (E) 最近鄰接觸（在「前視相機座標」進行」） ----------
            _, rh_cpts_cam_all, rh_d_all = self._nn_contacts_for_hand_cam(rh_cam, pc_cam_merged)
            _, lh_cpts_cam_all, lh_d_all = self._nn_contacts_for_hand_cam(lh_cam, pc_cam_merged)

            rh_mask = rh_d_all < contact_thresh
            lh_mask = lh_d_all < contact_thresh
            rh_cam_vis = rh_cam[rh_mask]
            rh_cpts_cam = rh_cpts_cam_all[rh_mask]
            lh_cam_vis = lh_cam[lh_mask]
            lh_cpts_cam = lh_cpts_cam_all[lh_mask]

            # ---------- (F) 骨架連線 & 繪圖 ----------
            rh_edges = self._build_hand_edges(self.dexhand_rh.body_names, add_palm_links=True)
            lh_edges = self._build_hand_edges(self.dexhand_lh.body_names, add_palm_links=True)

            out_path = os.path.join(out_dir, f"contacts_cam_env{env_id:02d}_step{step:06d}.png")
            self._save_contacts_visualization_cam(
                out_path=out_path,
                pc_cam=pc_cam_merged,  # ← 用合併後的點雲
                rh_j_all_cam=rh_cam_all, lh_j_all_cam=lh_cam_all,
                rh_edges=rh_edges, lh_edges=lh_edges,
                rh_j_vis_cam=rh_cam_vis, rh_c_vis_cam=rh_cpts_cam,
                lh_j_vis_cam=lh_cam_vis, lh_c_vis_cam=lh_cpts_cam,
                views=((0, 0),), shade="cam_depth" if shade is None else shade
            )

            # ---------- (G) 另外把「俯視相機 RGB」存檔（debug） ----------
            if getattr(self, "last_rgb_images_top", None) is not None and env_id < len(self.last_rgb_images_top):
                rgb_top = self.last_rgb_images_top[env_id]
                _save_single_rgb(rgb_top, os.path.join(out_dir, f"rgb_top_env{env_id:02d}_step{step:06d}.png"))

            print(f"[contacts/cam] env {env_id}: RH/LH contacts(<{contact_thresh:.3f}m) "
                f"rh={int(rh_mask.sum().item())}, lh={int(lh_mask.sum().item())})")


    def save_contacts(self, out_dir: str, step: int, max_envs: int = 1024,
                                object_depth_range=(0.6, 1.2),
                                center_crop_ratio=0.6,
                                shade: str = "cam_depth",
                                contact_thresh: float = 0.01):
        import os, torch
        os.makedirs(out_dir, exist_ok=True)
        if self.last_depths is None:
            return
        num = min(len(self.last_depths), max_envs)
        nan = float('nan') 

        if "joint_pos_tip" not in self.rh_states or self.rh_states["joint_pos_tip"].shape != (num, 5, 3):
            self.rh_states["joint_pos_tip"] = torch.full((num, 5, 3), nan, device=self.device, dtype=torch.float32)
        if "joint_pos_tip" not in self.lh_states or self.lh_states["joint_pos_tip"].shape != (num, 5, 3):
            self.lh_states["joint_pos_tip"] = torch.full((num, 5, 3), nan, device=self.device, dtype=torch.float32)
        if "contact_point_tip" not in self.rh_states or \
            self.rh_states["contact_point_tip"].shape != (num, 5, 3):
                self.rh_states["contact_point_tip"] = torch.full(
                    (num, 5, 3), nan, device=self.device, dtype=torch.float32
                )
        if "contact_point_tip" not in self.lh_states or \
            self.lh_states["contact_point_tip"].shape != (num, 5, 3):
                self.lh_states["contact_point_tip"] = torch.full(
                    (num, 5, 3), nan, device=self.device, dtype=torch.float32
                )

        buf = self.obs_dict["extra"]  
        pad = torch.zeros((num, 138), device=buf.device, dtype=buf.dtype)
        


        # todo: its only works for env_id = 0
        for env_id in range(num):
            depth_tensor = self.last_depths[env_id]
            if depth_tensor is None:
                continue

            seg_tensor = self.last_segs[env_id]
            allowed_ids = self._get_allowed_object_ids_for_env(env_id)
            pc_cam = self._depth_to_pointcloud_from_seg(
                depth_tensor=depth_tensor,
                seg_tensor=seg_tensor,
                allowed_ids=allowed_ids,
                horizontal_fov_deg=69.4,
                max_points=200000,
                border_dilate_px=1,
                z_min=0.45,     # ← adjust according to the distance between your camera and the table
                z_max=1.30      # ← avoid capturing far background (the yellow block will not appear
            )
            if pc_cam is None or pc_cam.shape[0] == 0:
                print(f"[contacts/cam] env {env_id}: no OBJECT point cloud (seg-mask empty).")
                continue

            # === joints world → camera (this time include the base in order to connect to the palm)
            rh_world_all = self.rh_states["joints_state"][env_id, :, :3]   # [18,3]
            lh_world_all = self.lh_states["joints_state"][env_id, :, :3]
            rh_cam_all = self._world_to_camera(rh_world_all, env_id)
            lh_cam_all = self._world_to_camera(lh_world_all, env_id)

            rh_world_all_tip = self.rh_states["joints_tip_state"][env_id, :, :3]   # [18,3]
            lh_world_all_tip = self.lh_states["joints_tip_state"][env_id, :, :3]
            rh_cam_all_tip = self._world_to_camera(rh_world_all_tip, env_id)
            lh_cam_all_tip = self._world_to_camera(rh_world_all_tip, env_id)

            

            # In your original nearest-neighbor computation you dropped the base; keep it that way (compute contacts using only movable joints)
            rh_world = rh_world_all[1:, :]
            lh_world = lh_world_all[1:, :]
            rh_cam = self._world_to_camera(rh_world, env_id)
            lh_cam = self._world_to_camera(lh_world, env_id)

            rh_world_tip = rh_world_all_tip[1:, :]
            lh_world_tip = lh_world_all_tip[1:, :]
            rh_cam_tip = self._world_to_camera(rh_world_all_tip, env_id)
            lh_cam_tip = self._world_to_camera(lh_world_all_tip, env_id)

            # Nearest neighbors
            _, rh_cpts_cam_all, rh_d_all = self._nn_contacts_for_hand_cam(rh_cam, pc_cam)
            _, lh_cpts_cam_all, lh_d_all = self._nn_contacts_for_hand_cam(lh_cam, pc_cam)

            # Nearest neighbors
            _, rh_cpts_cam_all_tip, rh_d_all_tip = self._nn_contacts_for_hand_cam(rh_cam_tip, pc_cam)
            _, lh_cpts_cam_all_tip, lh_d_all_tip = self._nn_contacts_for_hand_cam(lh_cam_tip, pc_cam)

            # Keep only contacts with distance < 0.01 m
            contact_thresh = 0.01  # your intended threshold
            rh_mask = rh_d_all < contact_thresh
            lh_mask = lh_d_all < contact_thresh
            rh_cam_vis = rh_cam[rh_mask]
            rh_cpts_cam = rh_cpts_cam_all[rh_mask]
            lh_cam_vis = lh_cam[lh_mask]
            lh_cpts_cam = lh_cpts_cam_all[lh_mask]

            rh_mask_tip = rh_d_all_tip < contact_thresh
            lh_mask_tip = lh_d_all_tip < contact_thresh
            rh_cam_vis_tip = rh_cam_tip[rh_mask_tip]
            rh_cpts_cam_tip = rh_cpts_cam_all_tip[rh_mask_tip]
            lh_cam_vis_tip = lh_cam_tip[lh_mask_tip]
            lh_cpts_cam_tip = lh_cpts_cam_all_tip[lh_mask_tip]


            import torch

            rh_joints = rh_cam_all            
            lh_joints = lh_cam_all           
            rh_contacts = torch.zeros_like(rh_joints)  
            lh_contacts = torch.zeros_like(lh_joints)  
            rh_contacts[1:][rh_mask] = rh_cpts_cam_all[rh_mask]
            lh_contacts[1:][lh_mask] = lh_cpts_cam_all[lh_mask]

            rh_joints_tip = rh_cam_all_tip            
            lh_joints_tip = lh_cam_all_tip       
            rh_contacts_tip = torch.full_like(rh_joints_tip, nan)
            lh_contacts_tip = torch.full_like(rh_joints_tip, nan) 
            rh_contacts_tip[rh_mask_tip] = rh_cpts_cam_all_tip[rh_mask_tip]
            lh_contacts_tip[lh_mask_tip] = lh_cpts_cam_all_tip[lh_mask_tip]

            self.rh_states["contact_point_tip"][env_id] = rh_contacts_tip
            self.lh_states["contact_point_tip"][env_id] = lh_contacts_tip
            self.rh_states["joint_pos_tip"][env_id] = rh_joints_tip
            self.lh_states["joint_pos_tip"][env_id] = rh_joints_tip


            vec = torch.cat([
                lh_joints.reshape(-1),       
                rh_joints.reshape(-1),       
                lh_contacts_tip.reshape(-1),     
                rh_contacts_tip.reshape(-1),     
            ], dim=0).unsqueeze(0) 

            vec138 = vec.squeeze(0).to(dtype=torch.float32, device=rh_joints.device) 
            pad[env_id] = vec138.to(buf.dtype)

        self.obs_dict["extra"] = torch.cat([self.obs_dict["extra"], pad], dim=1)


    

    def _depth_to_colored_object_pointcloud_simple(self, depth_tensor, rgb_tensor,
                                                horizontal_fov_deg=69.4,
                                                object_depth_range=(0.6, 1.0),
                                                center_crop_ratio=0.6,
                                                max_points=50000):
        """
        Convert a depth map and RGB image into a colored point cloud (revised)
        """
        import torch
        import numpy as np
        
        if depth_tensor is None or rgb_tensor is None:
            return None, None
            
        H, W = depth_tensor.shape
        device = depth_tensor.device
        
        print(f"Input shapes - Depth: {depth_tensor.shape}, RGB: {rgb_tensor.shape}")
        
        # ===== Fix RGB tensor shape =====
        # The RGB format from Isaac Gym may be (H, W, 4), (H, 4, W), or other layouts
        if len(rgb_tensor.shape) == 3:
            if rgb_tensor.shape[2] == 4:  # (H, W, 4) - correct format
                rgb_hw4 = rgb_tensor
            elif rgb_tensor.shape[1] == 4:  # (H, 4, W) - needs permutation
                rgb_hw4 = rgb_tensor.permute(0, 2, 1)  # -> (H, W, 4)
            elif rgb_tensor.shape[0] == 4:  # (4, H, W) - needs permutation
                rgb_hw4 = rgb_tensor.permute(1, 2, 0)  # -> (H, W, 4)
            else:
                print(f"Unexpected RGB tensor shape: {rgb_tensor.shape}")
                return None, None
        else:
            print(f"Unexpected RGB tensor dimensions: {rgb_tensor.shape}")
            return None, None
        
        # Ensure the RGB and depth map sizes match
        if rgb_hw4.shape[:2] != (H, W):
            print(f"Size mismatch after reshape: depth {depth_tensor.shape}, rgb {rgb_hw4.shape}")
            return None, None
        
        print(f"Corrected RGB shape: {rgb_hw4.shape}")
        
        # ===== 1. Compute camera intrinsics =====
        fovx = np.deg2rad(horizontal_fov_deg)
        fx = W / (2.0 * np.tan(fovx / 2.0))
        fovy = 2.0 * np.arctan(np.tan(fovx / 2.0) * (H / W))
        fy = H / (2.0 * np.tan(fovy / 2.0))
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        
        # ===== 2. Center-region cropping =====
        h_start = int(H * (1 - center_crop_ratio) / 2)
        h_end = int(H * (1 + center_crop_ratio) / 2)
        w_start = int(W * (1 - center_crop_ratio) / 2)
        w_end = int(W * (1 + center_crop_ratio) / 2)
        
        center_mask = torch.zeros_like(depth_tensor, dtype=torch.bool)
        center_mask[h_start:h_end, w_start:w_end] = True
        
        # ===== 3. Object depth-range filtering =====
        object_depth_mask = (
            (depth_tensor >= object_depth_range[0]) & 
            (depth_tensor <= object_depth_range[1])
        )
        
        # ===== 4. Combine all filtering conditions =====
        valid_mask = (
            torch.isfinite(depth_tensor) &
            center_mask &
            object_depth_mask
        )
        
        print(f"Valid mask shape: {valid_mask.shape}, valid points: {valid_mask.sum()}")
        
        if not valid_mask.any():
            print("No valid object points found")
            return None, None
        
        # ===== 5. Build coordinate grids and gather valid points =====
        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')
        
        v_valid = v_grid[valid_mask]
        u_valid = u_grid[valid_mask]
        z_valid = depth_tensor[valid_mask]
        
        # ===== 6. Extract corresponding RGB colors (revised) =====
        # Use the correct indexing method
        rgb_colors = rgb_hw4[valid_mask][:, :3].float() / 255.0  # take RGB only, ignore A channel
        
        print(f"Extracted colors shape: {rgb_colors.shape}")
        
        # ===== 7. Convert to 3D coordinates =====
        x_valid = (u_valid - cx) / fx * z_valid
        y_valid = (v_valid - cy) / fy * z_valid
        
        points_3d = torch.stack([x_valid, -y_valid, z_valid], dim=1)
        
        print(f"Found {points_3d.shape[0]} colored object points in depth range {object_depth_range}")
        
        # ===== 8. Denoising via simple clustering =====
        if points_3d.shape[0] > 100:
            center = torch.median(points_3d, dim=0)[0]
            distances = torch.norm(points_3d - center, dim=1)
            threshold = torch.quantile(distances, 0.9)
            inlier_mask = distances <= threshold
            points_3d = points_3d[inlier_mask]
            rgb_colors = rgb_colors[inlier_mask]
            print(f"After outlier removal: {points_3d.shape[0]} points")
        
        # ===== 9. Downsampling =====
        if max_points is not None and points_3d.shape[0] > max_points:
            indices = torch.randperm(points_3d.shape[0], device=device)[:max_points]
            points_3d = points_3d[indices]
            rgb_colors = rgb_colors[indices]
        
        return points_3d, rgb_colors


    def _depth_to_object_pointcloud_no_rgb(self, depth_tensor,
                                    horizontal_fov_deg=69.4,
                                    object_depth_range=(0.6, 1.0),
                                    center_crop_ratio=0.8,
                                    max_points=100000):
        """
        Convert a depth map into a point cloud (without RGB; colorized by depth values)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (points_3d, depth_colors)
            - points_3d: shape=(N, 3), point cloud coordinates
            - depth_colors: shape=(N,), depth values used for coloring
        """
        import torch
        import numpy as np
        
        if depth_tensor is None:
            return None, None
            
        H, W = depth_tensor.shape
        device = depth_tensor.device
        
        print(f"Processing depth-only pointcloud - Shape: {depth_tensor.shape}")
        
        # ===== 1. Compute camera intrinsics =====
        fovx = np.deg2rad(horizontal_fov_deg)
        fx = W / (2.0 * np.tan(fovx / 2.0))
        fovy = 2.0 * np.arctan(np.tan(fovx / 2.0) * (H / W))
        fy = H / (2.0 * np.tan(fovy / 2.0))
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        
        # ===== 2. Center-region cropping =====
        h_start = int(H * (1 - center_crop_ratio) / 2)
        h_end = int(H * (1 + center_crop_ratio) / 2)
        w_start = int(W * (1 - center_crop_ratio) / 2)
        w_end = int(W * (1 + center_crop_ratio) / 2)
        
        center_mask = torch.zeros_like(depth_tensor, dtype=torch.bool)
        center_mask[h_start:h_end, w_start:w_end] = True
        
        # ===== 3. Depth filtering =====
        object_depth_mask = (
            (depth_tensor >= object_depth_range[0]) & 
            (depth_tensor <= object_depth_range[1])
        )
        
        # ===== 4. Combine filtering conditions =====
        valid_mask = (
            torch.isfinite(depth_tensor) &
            center_mask &
            object_depth_mask
        )
        
        print(f"Valid points: {valid_mask.sum()} / {valid_mask.numel()}")
        
        if not valid_mask.any():
            print("No valid object points found")
            return None, None
        
        # ===== 5. Gather valid points =====
        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')
        
        v_valid = v_grid[valid_mask]
        u_valid = u_grid[valid_mask]
        z_valid = depth_tensor[valid_mask]
        
        # ===== 6. Convert to 3D coordinates =====
        x_valid = (u_valid - cx) / fx * z_valid
        y_valid = (v_valid - cy) / fy * z_valid
        
        points_3d = torch.stack([x_valid, -y_valid, z_valid], dim=1)
        
        # ===== 7. Use depth values as color information =====
        depth_colors = z_valid.clone()  # directly use the depth values
        
        print(f"Generated {points_3d.shape[0]} points with depth coloring")
        
        # ===== 8. Outlier removal =====
        if points_3d.shape[0] > 1000:
            center = torch.median(points_3d, dim=0)[0]
            distances = torch.norm(points_3d - center, dim=1)
            threshold = torch.quantile(distances, 0.95)
            inlier_mask = distances <= threshold
            points_3d = points_3d[inlier_mask]
            depth_colors = depth_colors[inlier_mask]
            print(f"After outlier removal: {points_3d.shape[0]} points")
        
        # ===== 9. Downsampling =====
        if max_points is not None and points_3d.shape[0] > max_points:
            indices = torch.randperm(points_3d.shape[0], device=device)[:max_points]
            points_3d = points_3d[indices]
            depth_colors = depth_colors[indices]
            print(f"Downsampled to: {points_3d.shape[0]} points")
        
        return points_3d, depth_colors


    def compute_observations_side(self, side="rh"):
        # obs_keys: q, cos_q, sin_q, base_state
        side_states = getattr(self, f"{side}_states")
        side_demo_data = getattr(self, f"demo_data_{side}")

        obs_dict = {}

        obs_values = []
        for ob in self._obs_keys:
            if ob == "base_state":
                obs_values.append(
                    torch.cat([torch.zeros_like(side_states[ob][:, :3]), side_states[ob][:, 3:]], dim=-1)
                )  # ! ignore base position
            else:
                obs_values.append(side_states[ob])
        obs_dict["proprioception"] = torch.cat(obs_values, dim=-1)
        # privileged_obs_keys: dq, manip_obj_pos, manip_obj_quat, manip_obj_vel, manip_obj_ang_vel
        if len(self._privileged_obs_keys) > 0:
            pri_obs_values = []
            for ob in self._privileged_obs_keys:
                if ob == "manip_obj_pos":
                    pri_obs_values.append(side_states[ob] - side_states["base_state"][:, :3])
                elif ob == "manip_obj_com":
                    cur_com_pos = (
                        quat_to_rotmat(side_states["manip_obj_quat"][:, [1, 2, 3, 0]])
                        @ getattr(self, f"manip_obj_{side}_com").unsqueeze(-1)
                    ).squeeze(-1) + side_states["manip_obj_pos"]
                    pri_obs_values.append(cur_com_pos - side_states["base_state"][:, :3])
                elif ob == "manip_obj_weight":
                    prop = self.gym.get_sim_params(self.sim)
                    pri_obs_values.append((getattr(self, f"manip_obj_{side}_mass") * -1 * prop.gravity.z).unsqueeze(-1))
                elif ob == "tip_force":
                    tip_force = torch.stack(
                        [
                            self.net_cf[:, getattr(self, f"dexhand_{side}_handles")[k], :]
                            for k in (
                                self.dexhand_rh.contact_body_names
                                if side == "rh"
                                else self.dexhand_lh.contact_body_names
                            )
                        ],
                        axis=1,
                    )
                    tip_force = torch.cat(
                        [tip_force, torch.norm(tip_force, dim=-1, keepdim=True)], dim=-1
                    )  # add force magnitude
                    pri_obs_values.append(tip_force.reshape(self.num_envs, -1))
                else:
                    pri_obs_values.append(side_states[ob])
            obs_dict["privileged"] = torch.cat(pri_obs_values, dim=-1)

        next_target_state = {}

        cur_idx = self.progress_buf + 1
        cur_idx = torch.clamp(cur_idx, torch.zeros_like(side_demo_data["seq_len"]), side_demo_data["seq_len"] - 1)

        cur_idx = torch.stack(
            [cur_idx + t for t in range(self.obs_future_length)], dim=-1
        )  # [B, K], K = obs_future_length
        nE, nT = side_demo_data["wrist_pos"].shape[:2]
        nF = self.obs_future_length

        def indicing(data, idx):
            assert data.shape[0] == nE and data.shape[1] == nT
            remaining_shape = data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded_idx)

        target_wrist_pos = indicing(side_demo_data["wrist_pos"], cur_idx)  # [B, K, 3]
        cur_wrist_pos = side_states["base_state"][:, :3]  # [B, 3]
        next_target_state["delta_wrist_pos"] = (target_wrist_pos - cur_wrist_pos[:, None]).reshape(nE, -1)

        target_wrist_vel = indicing(side_demo_data["wrist_velocity"], cur_idx)
        cur_wrist_vel = side_states["base_state"][:, 7:10]
        next_target_state["wrist_vel"] = target_wrist_vel.reshape(nE, -1)
        next_target_state["delta_wrist_vel"] = (target_wrist_vel - cur_wrist_vel[:, None]).reshape(nE, -1)

        target_wrist_rot = indicing(side_demo_data["wrist_rot"], cur_idx)
        cur_wrist_rot = side_states["base_state"][:, 3:7]

        next_target_state["wrist_quat"] = aa_to_quat(target_wrist_rot.reshape(nE * nF, -1))[:, [1, 2, 3, 0]]
        next_target_state["delta_wrist_quat"] = quat_mul(
            cur_wrist_rot[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["wrist_quat"]),
        ).reshape(nE, -1)
        next_target_state["wrist_quat"] = next_target_state["wrist_quat"].reshape(nE, -1)

        target_wrist_ang_vel = indicing(side_demo_data["wrist_angular_velocity"], cur_idx)
        cur_wrist_ang_vel = side_states["base_state"][:, 10:13]
        next_target_state["wrist_ang_vel"] = target_wrist_ang_vel.reshape(nE, -1)
        next_target_state["delta_wrist_ang_vel"] = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)

        target_joints_pos = indicing(side_demo_data["mano_joints"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_pos = side_states["joints_state"][:, 1:, :3]  # skip the base joint
        next_target_state["delta_joints_pos"] = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

        target_joints_vel = indicing(side_demo_data["mano_joints_velocity"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_vel = side_states["joints_state"][:, 1:, 7:10]  # skip the base joint
        next_target_state["joints_vel"] = target_joints_vel.reshape(self.num_envs, -1)
        next_target_state["delta_joints_vel"] = (target_joints_vel - cur_joint_vel[:, None]).reshape(self.num_envs, -1)

        target_obj_transf = indicing(side_demo_data["obj_trajectory"], cur_idx)
        target_obj_transf = target_obj_transf.reshape(nE * nF, 4, 4)
        next_target_state["delta_manip_obj_pos"] = (
            target_obj_transf[:, :3, 3].reshape(nE, nF, -1) - side_states["manip_obj_pos"][:, None]
        ).reshape(nE, -1)

        target_obj_vel = indicing(side_demo_data["obj_velocity"], cur_idx)
        cur_obj_vel = side_states["manip_obj_vel"]
        next_target_state["manip_obj_vel"] = target_obj_vel.reshape(nE, -1)
        next_target_state["delta_manip_obj_vel"] = (target_obj_vel - cur_obj_vel[:, None]).reshape(nE, -1)

        next_target_state["manip_obj_quat"] = rotmat_to_quat(target_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]
        next_target_state["delta_manip_obj_quat"] = quat_mul(
            side_states["manip_obj_quat"][:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["manip_obj_quat"]),
        ).reshape(nE, -1)
        next_target_state["manip_obj_quat"] = next_target_state["manip_obj_quat"].reshape(nE, -1)

        target_obj_ang_vel = indicing(side_demo_data["obj_angular_velocity"], cur_idx)
        cur_obj_ang_vel = side_states["manip_obj_ang_vel"]
        next_target_state["manip_obj_ang_vel"] = target_obj_ang_vel.reshape(nE, -1)
        next_target_state["delta_manip_obj_ang_vel"] = (target_obj_ang_vel - cur_obj_ang_vel[:, None]).reshape(nE, -1)

        next_target_state["obj_to_joints"] = torch.norm(
            side_states["manip_obj_pos"][:, None] - side_states["joints_state"][:, :, :3], dim=-1
        ).reshape(self.num_envs, -1)

        next_target_state["gt_tips_distance"] = indicing(side_demo_data["tips_distance"], cur_idx).reshape(nE, -1)

        next_target_state["bps"] = getattr(self, f"obj_bps_{side}")
        next_target_state["contact_point_tips"] = indicing(side_demo_data["contact_point_tips"], cur_idx).reshape(nE, -1)
        obs_dict["target"] = torch.cat(
            [
                next_target_state[ob]
                for ob in [  # ! must be in the same order as the following
                    "delta_wrist_pos",
                    "wrist_vel",
                    "delta_wrist_vel",
                    "wrist_quat",
                    "delta_wrist_quat",
                    "wrist_ang_vel",
                    "delta_wrist_ang_vel",
                    "delta_joints_pos",
                    "joints_vel",
                    "delta_joints_vel",
                    "delta_manip_obj_pos",
                    "manip_obj_vel",
                    "delta_manip_obj_vel",
                    "manip_obj_quat",
                    "delta_manip_obj_quat",
                    "manip_obj_ang_vel",
                    "delta_manip_obj_ang_vel",
                    "obj_to_joints",
                    "gt_tips_distance",
                    "bps",
                ]
            ],
            dim=-1,
        )

        obs_dict["extra"] = next_target_state["contact_point_tips"]

        if not self.training:
            manip_obj_root_state = getattr(self, f"_manip_obj_{side}_root_state")
            dexhand_handles = getattr(self, f"dexhand_{side}_handles")
            for prop_name in self._prop_dump_info.keys():
                if prop_name == "state_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["base_state"]
                elif prop_name == "state_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["base_state"]
                elif prop_name == "state_manip_obj_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = manip_obj_root_state
                elif prop_name == "state_manip_obj_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = manip_obj_root_state
                elif prop_name == "joint_state_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = torch.stack(
                        [self._rigid_body_state[:, dexhand_handles[k], :] for k in self.dexhand_rh.body_names],
                        dim=1,
                    ).reshape(self.num_envs, -1)
                elif prop_name == "joint_state_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = torch.stack(
                        [self._rigid_body_state[:, dexhand_handles[k], :] for k in self.dexhand_lh.body_names],
                        dim=1,
                    ).reshape(self.num_envs, -1)
                elif prop_name == "q_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["q"]
                elif prop_name == "q_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["q"]
                elif prop_name == "dq_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["dq"]
                elif prop_name == "dq_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["dq"]
                elif prop_name == "tip_force_rh" and side == "rh":
                    tip_force = torch.stack(
                        [self.net_cf[:, dexhand_handles[k], :] for k in self.dexhand_rh.contact_body_names],
                        axis=1,
                    )
                    self.dump_fileds[prop_name][:] = tip_force.reshape(self.num_envs, -1)
                elif prop_name == "tip_force_lh" and side == "lh":
                    tip_force = torch.stack(
                        [self.net_cf[:, dexhand_handles[k], :] for k in self.dexhand_lh.contact_body_names],
                        axis=1,
                    )
                    self.dump_fileds[prop_name][:] = tip_force.reshape(self.num_envs, -1)
                elif prop_name == "reward":
                    self.dump_fileds[prop_name][:] = self.rew_buf.reshape(self.num_envs, -1).detach()
                else:
                    pass
        return obs_dict

    def _reset_default(self, env_ids):
        if self.random_state_init:
            if self.rollout_begin is not None:
                seq_idx = (
                    torch.floor(
                        self.rollout_len * 0.98 * torch.rand_like(self.demo_data_rh["seq_len"][env_ids].float())
                    ).long()
                    + self.rollout_begin
                )
                seq_idx = torch.clamp(
                    seq_idx,
                    torch.zeros(1, device=self.device).long(),
                    torch.floor(self.demo_data_rh["seq_len"][env_ids] * 0.98).long(),
                )
            else:
                seq_idx = torch.floor(
                    self.demo_data_rh["seq_len"][env_ids]
                    * 0.98
                    * torch.rand_like(self.demo_data_rh["seq_len"][env_ids].float())
                ).long()
        else:
            if self.rollout_begin is not None:
                seq_idx = self.rollout_begin * torch.ones_like(self.demo_data_rh["seq_len"][env_ids].long())
            else:
                seq_idx = torch.zeros_like(self.demo_data_rh["seq_len"][env_ids].long())

        self._reset_default_side(env_ids, seq_idx, side="lh")
        self._reset_default_side(env_ids, seq_idx, side="rh")

        dexhand_multi_env_ids_int32 = torch.concat(
            [
                self._global_dexhand_rh_indices[env_ids].flatten(),
                self._global_dexhand_lh_indices[env_ids].flatten(),
            ]
        )
        manip_obj_multi_env_ids_int32 = torch.concat(
            [self._global_manip_obj_rh_indices[env_ids].flatten(), self._global_manip_obj_lh_indices[env_ids].flatten()]
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
            len(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )

        self.progress_buf[env_ids] = seq_idx
        self.running_progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0
        self.error_buf[env_ids] = 0
        self.total_rew_buf[env_ids] = 0
        self.apply_forces[env_ids] = 0
        self.apply_torque[env_ids] = 0
        self.curr_targets[env_ids] = 0
        self.prev_targets[env_ids] = 0

        if self.use_pid_control:
            self.rh_prev_pos_error[env_ids] = 0
            self.rh_prev_rot_error[env_ids] = 0
            self.rh_pos_error_integral[env_ids] = 0
            self.rh_rot_error_integral[env_ids] = 0
            self.lh_prev_pos_error[env_ids] = 0
            self.lh_prev_rot_error[env_ids] = 0
            self.lh_pos_error_integral[env_ids] = 0
            self.lh_rot_error_integral[env_ids] = 0

        self.lh_tips_contact_history[env_ids] = torch.ones_like(self.lh_tips_contact_history[env_ids]).bool()
        self.rh_tips_contact_history[env_ids] = torch.ones_like(self.rh_tips_contact_history[env_ids]).bool()

    def _reset_default_side(self, env_ids, seq_idx, side="rh"):

        side_demo_data = getattr(self, f"demo_data_{side}")

        dof_pos = side_demo_data["opt_dof_pos"][env_ids, seq_idx]
        dof_pos = torch_jit_utils.tensor_clamp(
            dof_pos,
            getattr(self, f"dexhand_{side}_dof_lower_limits").unsqueeze(0),
            getattr(self, f"dexhand_{side}_dof_upper_limits").unsqueeze(0),
        )
        dof_vel = side_demo_data["opt_dof_velocity"][env_ids, seq_idx]
        dof_vel = torch_jit_utils.tensor_clamp(
            dof_vel,
            -1 * getattr(self, f"_dexhand_{side}_dof_speed_limits").unsqueeze(0),
            getattr(self, f"_dexhand_{side}_dof_speed_limits").unsqueeze(0),
        )

        opt_wrist_pos = side_demo_data["opt_wrist_pos"][env_ids, seq_idx]
        opt_wrist_rot = aa_to_quat(side_demo_data["opt_wrist_rot"][env_ids, seq_idx])
        opt_wrist_rot = opt_wrist_rot[:, [1, 2, 3, 0]]

        opt_wrist_vel = side_demo_data["opt_wrist_velocity"][env_ids, seq_idx]
        opt_wrist_ang_vel = side_demo_data["opt_wrist_angular_velocity"][env_ids, seq_idx]

        opt_hand_pose_vel = torch.concat([opt_wrist_pos, opt_wrist_rot, opt_wrist_vel, opt_wrist_ang_vel], dim=-1)

        getattr(self, f"_{side}_base_state")[env_ids, :] = opt_hand_pose_vel

        if side == "rh":
            self._q[env_ids, : self.num_dexhand_rh_dofs] = dof_pos
            self._qd[env_ids, : self.num_dexhand_rh_dofs] = dof_vel
            self._pos_control[env_ids, : self.num_dexhand_rh_dofs] = dof_pos
        else:
            self._q[env_ids, self.num_dexhand_rh_dofs :] = dof_pos
            self._qd[env_ids, self.num_dexhand_rh_dofs :] = dof_vel
            self._pos_control[env_ids, self.num_dexhand_rh_dofs :] = dof_pos

        # reset manip obj
        obj_pos_init = side_demo_data["obj_trajectory"][env_ids, seq_idx, :3, 3]
        obj_rot_init = side_demo_data["obj_trajectory"][env_ids, seq_idx, :3, :3]
        obj_rot_init = rotmat_to_quat(obj_rot_init)
        # [w, x, y, z] to [x, y, z, w]
        obj_rot_init = obj_rot_init[:, [1, 2, 3, 0]]

        obj_vel = side_demo_data["obj_velocity"][env_ids, seq_idx]
        obj_ang_vel = side_demo_data["obj_angular_velocity"][env_ids, seq_idx]

        manip_obj_root_state = getattr(self, f"_manip_obj_{side}_root_state")

        manip_obj_root_state[env_ids, :3] = obj_pos_init
        manip_obj_root_state[env_ids, 3:7] = obj_rot_init
        manip_obj_root_state[env_ids, 7:10] = obj_vel
        manip_obj_root_state[env_ids, 10:13] = obj_ang_vel

    def reset_idx(self, env_ids):
        self._refresh()
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        last_step = self.gym.get_frame_count(self.sim)
        if self.training and len(self.dataIndices) == 1 and last_step >= self.tighten_steps:
            running_steps = self.running_progress_buf[env_ids] - 1
            max_running_steps, max_running_idx = running_steps.max(dim=0)
            max_running_env_id = env_ids[max_running_idx]
            if max_running_steps > self.best_rollout_len:
                self.best_rollout_len = max_running_steps
                self.best_rollout_begin = self.progress_buf[max_running_env_id] - 1 - max_running_steps

        self._reset_default(env_ids)

    def reset_done(self):
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)
            self.compute_observations()

        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        info["reward_dict"] = self.reward_dict
        info["total_rewards"] = self.total_rew_buf
        info["total_steps"] = self.progress_buf
        return obs, rew, done, info

    def pre_physics_step(self, actions):

        # ? >>> for visualization
        if not self.headless:

            cur_idx = self.progress_buf

            self.gym.clear_lines(self.viewer)

            def set_side_joint(cur_idx, side="rh"):
                cur_wrist_pos = getattr(self, f"demo_data_{side}")["wrist_pos"][torch.arange(self.num_envs), cur_idx]
                cur_mano_joint_pos = getattr(self, f"demo_data_{side}")["mano_joints"][
                    torch.arange(self.num_envs), cur_idx
                ].reshape(self.num_envs, -1, 3)
                cur_mano_joint_pos = torch.concat([cur_wrist_pos[:, None], cur_mano_joint_pos], dim=1)
                for k in range(len(getattr(self, f"mano_joint_{side}_points"))):
                    getattr(self, f"mano_joint_{side}_points")[k][:, :3] = cur_mano_joint_pos[:, k]
                for env_id, env_ptr in enumerate(self.envs):
                    for rh_k, k in zip(
                        self.dexhand_rh.body_names,
                        (self.dexhand_rh.body_names if side == "rh" else self.dexhand_lh.body_names),
                    ):
                        self.set_force_vis(
                            env_ptr,
                            rh_k,
                            torch.norm(self.net_cf[env_id, getattr(self, f"dexhand_{side}_handles")[k]], dim=-1) != 0,
                            side,
                        )

                    def add_lines(viewer, env_ptr, hand_joints, color):
                        assert hand_joints.shape[0] == self.dexhand_rh.n_bodies and hand_joints.shape[1] == 3
                        hand_joints = hand_joints.cpu().numpy()
                        lines = np.array([[hand_joints[b[0]], hand_joints[b[1]]] for b in self.dexhand_rh.bone_links])
                        for line in lines:
                            self.gym.add_lines(viewer, env_ptr, 1, line, color)

                    color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
                    add_lines(self.viewer, env_ptr, cur_mano_joint_pos[env_id].cpu(), color)

            set_side_joint(cur_idx, "lh")
            set_side_joint(cur_idx, "rh")

        # ? <<< for visualization

        root_control_dim = 9 if self.use_pid_control else 6
        res_split_idx = (
            actions.shape[1] // 2
            if not self.use_pid_control
            else ((actions.shape[1] - 2 * (root_control_dim - 6)) // 2) + 2 * (root_control_dim - 6)
        )

        base_action = actions[:, :res_split_idx]  # ? in the range of [-1, 1]
        residual_action = actions[:, res_split_idx:] * 2  # ? the delta action is theoritically in the range of [-2, 2]

        rh_dof_pos = (
            1.0 * base_action[:, root_control_dim : root_control_dim + self.num_dexhand_rh_dofs]
            + residual_action[:, 6 : 6 + self.num_dexhand_rh_dofs]
        )
        rh_dof_pos = torch.clamp(rh_dof_pos, -1, 1)

        lh_dof_pos = (
            1.0 * base_action[:, root_control_dim + root_control_dim + self.num_dexhand_rh_dofs :]
            + residual_action[:, 6 + 6 + self.num_dexhand_rh_dofs :]
        )
        lh_dof_pos = torch.clamp(lh_dof_pos, -1, 1)

        curr_act_moving_average = self.act_moving_average

        self.rh_curr_targets = torch_jit_utils.scale(
            rh_dof_pos,  # ! actions must in [-1, 1]
            self.dexhand_rh_dof_lower_limits,
            self.dexhand_rh_dof_upper_limits,
        )
        self.rh_curr_targets = (
            curr_act_moving_average * self.rh_curr_targets
            + (1.0 - curr_act_moving_average) * self.prev_targets[:, : self.num_dexhand_rh_dofs]
        )
        self.rh_curr_targets = torch_jit_utils.tensor_clamp(
            self.rh_curr_targets,
            self.dexhand_rh_dof_lower_limits,
            self.dexhand_rh_dof_upper_limits,
        )
        self.prev_targets[:, : self.num_dexhand_rh_dofs] = self.rh_curr_targets[:]

        self.lh_curr_targets = torch_jit_utils.scale(
            lh_dof_pos,
            self.dexhand_lh_dof_lower_limits,
            self.dexhand_lh_dof_upper_limits,
        )
        self.lh_curr_targets = (
            curr_act_moving_average * self.lh_curr_targets
            + (1.0 - curr_act_moving_average) * self.prev_targets[:, self.num_dexhand_rh_dofs :]
        )
        self.lh_curr_targets = torch_jit_utils.tensor_clamp(
            self.lh_curr_targets,
            self.dexhand_lh_dof_lower_limits,
            self.dexhand_lh_dof_upper_limits,
        )
        self.prev_targets[:, self.num_dexhand_rh_dofs :] = self.lh_curr_targets[:]

        if self.use_pid_control:
            rh_position_error = base_action[:, 0:3]
            self.rh_pos_error_integral += rh_position_error * self.dt
            self.rh_pos_error_integral = torch.clamp(self.rh_pos_error_integral, -1, 1)
            rh_pos_derivative = (rh_position_error - self.rh_prev_pos_error) / self.dt
            rh_force = (
                self.Kp_pos * rh_position_error
                + self.Ki_pos * self.rh_pos_error_integral
                + self.Kd_pos * rh_pos_derivative
            )
            self.rh_prev_pos_error = rh_position_error

            rh_force = rh_force + residual_action[:, 0:3] * self.dt * self.translation_scale * 500
            self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )

            lh_position_error = base_action[
                :, root_control_dim + self.num_dexhand_rh_dofs : root_control_dim + self.num_dexhand_rh_dofs + 3
            ]
            self.lh_pos_error_integral += lh_position_error * self.dt
            self.lh_pos_error_integral = torch.clamp(self.lh_pos_error_integral, -1, 1)
            lh_pos_derivative = (lh_position_error - self.lh_prev_pos_error) / self.dt
            lh_force = (
                self.Kp_pos * lh_position_error
                + self.Ki_pos * self.lh_pos_error_integral
                + self.Kd_pos * lh_pos_derivative
            )
            self.lh_prev_pos_error = lh_position_error

            lh_force = (
                lh_force
                + residual_action[:, 6 + self.num_dexhand_rh_dofs : 6 + self.num_dexhand_rh_dofs + 3]
                * self.dt
                * self.translation_scale
                * 500
            )
            self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )

            rh_rotation_error = base_action[:, 3:root_control_dim]
            rh_rotation_error = rot6d_to_aa(rh_rotation_error)
            self.rh_rot_error_integral += rh_rotation_error * self.dt
            self.rh_rot_error_integral = torch.clamp(self.rh_rot_error_integral, -1, 1)
            rh_rot_derivative = (rh_rotation_error - self.rh_prev_rot_error) / self.dt
            rh_torque = (
                self.Kp_rot * rh_rotation_error
                + self.Ki_rot * self.rh_rot_error_integral
                + self.Kd_rot * rh_rot_derivative
            )
            self.rh_prev_rot_error = rh_rotation_error

            rh_torque = rh_torque + residual_action[:, 3:6] * self.dt * self.orientation_scale * 200
            self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )

            lh_rotation_error = base_action[
                :,
                root_control_dim
                + self.num_dexhand_rh_dofs
                + 3 : root_control_dim
                + self.num_dexhand_rh_dofs
                + root_control_dim,
            ]
            lh_rotation_error = rot6d_to_aa(lh_rotation_error)
            self.lh_rot_error_integral += lh_rotation_error * self.dt
            self.lh_rot_error_integral = torch.clamp(self.lh_rot_error_integral, -1, 1)
            lh_rot_derivative = (lh_rotation_error - self.lh_prev_rot_error) / self.dt
            lh_torque = (
                self.Kp_rot * lh_rotation_error
                + self.Ki_rot * self.lh_rot_error_integral
                + self.Kd_rot * lh_rot_derivative
            )
            self.lh_prev_rot_error = lh_rotation_error

            lh_torque = (
                lh_torque
                + residual_action[:, 6 + self.num_dexhand_rh_dofs + 3 : 6 + self.num_dexhand_rh_dofs + 6]
                * self.dt
                * self.orientation_scale
                * 200
            )
            self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )
        else:
            rh_force = 1.0 * (base_action[:, 0:3] * self.dt * self.translation_scale * 500) + (
                residual_action[:, 0:3] * self.dt * self.translation_scale * 500
            )
            rh_torque = 1.0 * (base_action[:, 3:6] * self.dt * self.orientation_scale * 200) + (
                residual_action[:, 3:6] * self.dt * self.orientation_scale * 200
            )
            lh_force = 1.0 * (
                base_action[
                    :, root_control_dim + self.num_dexhand_rh_dofs : root_control_dim + self.num_dexhand_rh_dofs + 3
                ]
                * self.dt
                * self.translation_scale
                * 500
            ) + (
                residual_action[:, 6 + self.num_dexhand_rh_dofs : 6 + self.num_dexhand_rh_dofs + 3]
                * self.dt
                * self.translation_scale
                * 500
            )
            lh_torque = 1.0 * (
                base_action[
                    :, root_control_dim + self.num_dexhand_rh_dofs + 3 : root_control_dim + self.num_dexhand_rh_dofs + 6
                ]
                * self.dt
                * self.orientation_scale
                * 200
            ) + (
                residual_action[:, 6 + self.num_dexhand_rh_dofs + 3 : 6 + self.num_dexhand_rh_dofs + 6]
                * self.dt
                * self.orientation_scale
                * 200
            )

            self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )
            self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )

            self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )
            self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.apply_forces),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )

        self._pos_control[:] = self.prev_targets[:]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def post_physics_step(self):
        self.compute_observations()
        self.compute_reward_cp(self.actions)

        self.progress_buf += 1
        self.running_progress_buf += 1
        self.randomize_buf += 1

    def _get_env_camera_pose_world_view(self, env_id: int, view: str = "front"):
        """
        回傳 (R_wc, t_wc)，依 view 使用前視或俯視的相機位姿。
        會考慮每個 env 的自動校正結果（是否需要加上 env origin）。
        """
        # env-local camera pose recorded at creation
        if view == "top":
            cam_pos_local = self._cam_pos_local_top[env_id]
            cam_tgt_local = self._cam_target_local_top[env_id]
        else:
            cam_pos_local = self._cam_pos_local[env_id]
            cam_tgt_local = self._cam_target_local[env_id]

        add_origin: Optional[bool] = self._cam_origin_mode.get(view, {}).get(env_id, None)
        # default to False (no origin) until calibration decides
        if add_origin is None:
            add_origin = False

        if add_origin:
            o = self.gym.get_env_origin(self.envs[env_id])
            origin_w = torch.tensor([o.x, o.y, o.z], device=self.device, dtype=torch.float32)
            t_wc = origin_w + cam_pos_local
            tgt_w = origin_w + cam_tgt_local
        else:
            t_wc = cam_pos_local
            tgt_w = cam_tgt_local

        up_guess = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        forward = tgt_w - t_wc
        forward = forward / (torch.norm(forward) + 1e-8)
        right = torch.cross(forward, up_guess); right = right / (torch.norm(right) + 1e-8)
        up = torch.cross(right, forward);       up = up / (torch.norm(up) + 1e-8)
        R_wc = torch.stack([right, up, forward], dim=1)  # columns = axes
        return R_wc, t_wc

    def _world_to_camera_view(self, pts_world: torch.Tensor, env_id: int, view: str = "front"):
        R_wc, t_wc = self._get_env_camera_pose_world_view(env_id, view=view)
        return (pts_world - t_wc[None, :]) @ R_wc  # world -> cam

    def _camera_to_world_view(self, pts_cam: torch.Tensor, env_id: int, view: str = "front"):
        R_wc, t_wc = self._get_env_camera_pose_world_view(env_id, view=view)
        return pts_cam @ R_wc.t() + t_wc[None, :]  # cam -> world


    def _calibrate_camera_origin_mode_for_env(self, env_id: int, view: str = "front"):
        """
        使用當前 depth+seg 估計物體點雲中心（相機座標），
        將物體 root 世界座標投到相機座標，比較「加 origin / 不加 origin」兩種外參與點雲中心的距離，
        選擇較小者，並記錄結果供後續使用。
        """
        try:
            if view == "top":
                if self.last_depths_top is None or self.last_segs_top is None:
                    return
                depth = self.last_depths_top[env_id]
                seg = self.last_segs_top[env_id]
            else:
                if self.last_depths is None or self.last_segs is None:
                    return
                depth = self.last_depths[env_id]
                seg = self.last_segs[env_id]

            if depth is None or seg is None:
                return

            allowed = self._get_allowed_object_ids_for_env(env_id)
            pc_cam = self._depth_to_pointcloud_from_seg(
                depth_tensor=depth,
                seg_tensor=seg,
                allowed_ids=allowed,
                horizontal_fov_deg=69.4,
                max_points=60000,
                border_dilate_px=0,
            )
            if pc_cam is None or pc_cam.shape[0] < 30:
                return

            pc_center = torch.median(pc_cam, dim=0)[0]

            if view == "top":
                cam_pos_local = self._cam_pos_local_top[env_id]
                cam_tgt_local = self._cam_target_local_top[env_id]
            else:
                cam_pos_local = self._cam_pos_local[env_id]
                cam_tgt_local = self._cam_target_local[env_id]

            o = self.gym.get_env_origin(self.envs[env_id])
            origin_w = torch.tensor([o.x, o.y, o.z], device=self.device, dtype=torch.float32)

            def build_RT(add_origin: bool):
                if add_origin:
                    t_wc = origin_w + cam_pos_local
                    tgt_w = origin_w + cam_tgt_local
                else:
                    t_wc = cam_pos_local
                    tgt_w = cam_tgt_local
                up_guess = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                forward = (tgt_w - t_wc); forward = forward / (torch.norm(forward) + 1e-8)
                right = torch.cross(forward, up_guess); right = right / (torch.norm(right) + 1e-8)
                up = torch.cross(right, forward); up = up / (torch.norm(up) + 1e-8)
                R_wc = torch.stack([right, up, forward], dim=1)
                return R_wc, t_wc

            R_wc_A, t_wc_A = build_RT(True)
            R_wc_B, t_wc_B = build_RT(False)

            rh_w = self._manip_obj_rh_root_state[env_id, :3]
            lh_w = self._manip_obj_lh_root_state[env_id, :3]
            rh_A = (rh_w - t_wc_A) @ R_wc_A
            lh_A = (lh_w - t_wc_A) @ R_wc_A
            rh_B = (rh_w - t_wc_B) @ R_wc_B
            lh_B = (lh_w - t_wc_B) @ R_wc_B

            dA = torch.min(torch.norm(rh_A - pc_center), torch.norm(lh_A - pc_center))
            dB = torch.min(torch.norm(rh_B - pc_center), torch.norm(lh_B - pc_center))

            chosen = bool(dA < dB)
            self._cam_origin_mode[view][env_id] = chosen
            if not self._cam_origin_mode_printed[view].get(env_id, False):
                how = "+origin" if chosen else "no-origin"
                print(f"[cam-calib] env {env_id} view={view}: chosen {how} (dA={float(dA):.4f}, dB={float(dB):.4f})")
                self._cam_origin_mode_printed[view][env_id] = True
        except Exception:
            pass



    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
        """
        Only create front camera for view purpose
        """
        if self._record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 320
            camera_cfg.height = 180
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)

        if not hasattr(self, "_cam_pos_local"):
            self._cam_pos_local = []
            self._cam_target_local = []
        self._cam_pos_local.append(torch.tensor([cam_pos.x, cam_pos.y, cam_pos.z], device=self.device, dtype=torch.float32))
        self._cam_target_local.append(torch.tensor([cam_target.x, cam_target.y, cam_target.z], device=self.device, dtype=torch.float32))

        # === 新增：俯視（top-down）===
        camera_cfg_top = gymapi.CameraProperties()
        camera_cfg_top.enable_tensors = True
        camera_cfg_top.width  = camera_cfg.width
        camera_cfg_top.height = camera_cfg.height
        camera_cfg_top.horizontal_fov = 69.4

        camera_top = isaac_gym.create_camera_sensor(env, camera_cfg_top)
        # 以桌面中心上方為基準（可再微調）
        cam_pos_top    = gymapi.Vec3(-1.2, 0, 0.74)
        # cam_pos_top    = gymapi.Vec3(0, 0, 1.5)
        cam_target_top = gymapi.Vec3(0.05, 0, 0.3)
        isaac_gym.set_camera_location(camera_top, env, cam_pos_top, cam_target_top)
        self.cameras_top.append(camera_top)

        # 紀錄「俯視相機」的 env-local 位姿
        if not hasattr(self, "_cam_pos_local_top"):
            self._cam_pos_local_top = []
            self._cam_target_local_top = []
        self._cam_pos_local_top.append(torch.tensor([cam_pos_top.x, cam_pos_top.y, cam_pos_top.z], device=self.device, dtype=torch.float32))
        self._cam_target_local_top.append(torch.tensor([cam_target_top.x, cam_target_top.y, cam_target_top.z], device=self.device, dtype=torch.float32))

        return camera  
    def set_force_vis(self, env_ptr, part_k, has_force, side):
        self.gym.set_rigid_body_color(
            env_ptr,
            self.gym.find_actor_handle(env_ptr, "dexhand_l" if side == "lh" else "dexhand_r"),
            getattr(self, f"dexhand_rh_handles")[part_k],  # tricks here, because the handle is the same
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    0.6,
                    0.6,
                )
                if has_force
                else gymapi.Vec3(1.0, 1.0, 1.0)
            ),
        )


@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def compute_imitation_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    scale_factor: float,
    dexhand_weight_idx: Dict[str, List[int]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    # type: (Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor], Tensor, float,  Dict[str, List[int]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor]

    # end effector pose reward
    current_eef_pos = states["base_state"][:, :3]
    current_eef_quat = states["base_state"][:, 3:7]

    target_eef_pos = target_states["wrist_pos"]
    target_eef_quat = target_states["wrist_quat"]
    diff_eef_pos = target_eef_pos - current_eef_pos
    diff_eef_pos_dist = torch.norm(diff_eef_pos, dim=-1)

    current_eef_vel = states["base_state"][:, 7:10]
    current_eef_ang_vel = states["base_state"][:, 10:13]
    target_eef_vel = target_states["wrist_vel"]
    target_eef_ang_vel = target_states["wrist_ang_vel"]

    diff_eef_vel = target_eef_vel - current_eef_vel
    diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

    joints_pos = states["joints_state"][:, 1:, :3]
    target_joints_pos = target_states["joints_pos"]
    diff_joints_pos = target_joints_pos - joints_pos
    diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

    # ? assign different weights to different joints
    # assert diff_joints_pos_dist.shape[1] == 17  # ignore the base joint
    diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
    diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
    diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
    diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
    diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
    diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)

    joints_vel = states["joints_state"][:, 1:, 7:10]
    target_joints_vel = target_states["joints_vel"]
    diff_joints_vel = target_joints_vel - joints_vel

    reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
    reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
    reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
    reward_middle_tip_pos = torch.exp(-80 * diff_middle_tip_pos_dist)
    reward_pinky_tip_pos = torch.exp(-60 * diff_pinky_tip_pos_dist)
    reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
    reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
    reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

    reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
    reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
    reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))

    current_dof_vel = states["dq"]

    diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
    diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
    reward_eef_rot = torch.exp(-1 * (diff_eef_rot_angle).abs())

    # object pose reward
    current_obj_pos = states["manip_obj_pos"]
    current_obj_quat = states["manip_obj_quat"]

    target_obj_pos = target_states["manip_obj_pos"]
    target_obj_quat = target_states["manip_obj_quat"]
    diff_obj_pos = target_obj_pos - current_obj_pos
    diff_obj_pos_dist = torch.norm(diff_obj_pos, dim=-1)

    reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

    diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
    diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0]
    reward_obj_rot = torch.exp(-3 * (diff_obj_rot_angle).abs())

    current_obj_vel = states["manip_obj_vel"]
    target_obj_vel = target_states["manip_obj_vel"]
    diff_obj_vel = target_obj_vel - current_obj_vel
    reward_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))

    current_obj_ang_vel = states["manip_obj_ang_vel"]
    target_obj_ang_vel = target_states["manip_obj_ang_vel"]
    diff_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel
    reward_obj_ang_vel = torch.exp(-1 * diff_obj_ang_vel.abs().mean(dim=-1))

    reward_power = torch.exp(-10 * target_states["power"])
    reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])

    finger_tip_force = target_states["tip_force"]
    finger_tip_distance = target_states["tips_distance"]
    contact_range = [0.02, 0.03]
    finger_tip_weight = torch.clamp(
        (contact_range[1] - finger_tip_distance) / (contact_range[1] - contact_range[0]), 0, 1
    )
    finger_tip_force_masked = finger_tip_force * finger_tip_weight[:, :, None]

    reward_finger_tip_force = torch.exp(-1 * (1 / (torch.norm(finger_tip_force_masked, dim=-1).sum(-1) + 1e-5)))

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
        | (torch.norm(current_obj_vel, dim=-1) > 100)
        | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
    )  # sanity check

    failed_execute = (
        (
            (diff_obj_pos_dist > 0.02 / 0.343 * scale_factor**3)  # TODO
            | (diff_thumb_tip_pos_dist > 0.04 / 0.7 * scale_factor)
            | (diff_index_tip_pos_dist > 0.045 / 0.7 * scale_factor)
            | (diff_middle_tip_pos_dist > 0.05 / 0.7 * scale_factor)
            | (diff_pinky_tip_pos_dist > 0.06 / 0.7 * scale_factor)
            | (diff_ring_tip_pos_dist > 0.06 / 0.7 * scale_factor)
            | (diff_level_1_pos_dist > 0.07 / 0.7 * scale_factor)
            | (diff_level_2_pos_dist > 0.08 / 0.7 * scale_factor)
            | (diff_obj_rot_angle.abs() / np.pi * 180 > 30 / 0.343 * scale_factor**3)  # TODO
            | torch.any((finger_tip_distance < 0.005) & ~(target_states["tip_contact_state"].any(1)), dim=-1)
        )
        & (running_progress_buf >= 8)
    ) | error_buf
    reward_execute = (
        0.1 * reward_eef_pos
        + 0.6 * reward_eef_rot
        + 0.9 * reward_thumb_tip_pos
        + 0.8 * reward_index_tip_pos
        + 0.75 * reward_middle_tip_pos
        + 0.6 * reward_pinky_tip_pos
        + 0.6 * reward_ring_tip_pos
        + 0.5 * reward_level_1_pos
        + 0.3 * reward_level_2_pos
        + 5.0 * reward_obj_pos
        + 1.0 * reward_obj_rot
        + 0.1 * reward_eef_vel
        + 0.05 * reward_eef_ang_vel
        + 0.1 * reward_joints_vel
        + 0.1 * reward_obj_vel
        + 0.1 * reward_obj_ang_vel
        + 1.0 * reward_finger_tip_force
        + 0.5 * reward_power
        + 0.5 * reward_wrist_power
    )

    succeeded = (
        progress_buf + 1 + 3 >= max_length
    ) & ~failed_execute  # reached the end of the trajectory, +3 for max future 3 steps
    reset_buf = torch.where(
        succeeded | failed_execute,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    reward_dict = {
        "reward_eef_pos": reward_eef_pos,
        "reward_eef_rot": reward_eef_rot,
        "reward_eef_vel": reward_eef_vel,
        "reward_eef_ang_vel": reward_eef_ang_vel,
        "reward_joints_vel": reward_joints_vel,
        "reward_obj_pos": reward_obj_pos,
        "reward_obj_rot": reward_obj_rot,
        "reward_obj_vel": reward_obj_vel,
        "reward_obj_ang_vel": reward_obj_ang_vel,
        "reward_joints_pos": (
            reward_thumb_tip_pos
            + reward_index_tip_pos
            + reward_middle_tip_pos
            + reward_pinky_tip_pos
            + reward_ring_tip_pos
            + reward_level_1_pos
            + reward_level_2_pos
        ),
        "reward_power": reward_power,
        "reward_wrist_power": reward_wrist_power,
        "reward_finger_tip_force": reward_finger_tip_force,
    }

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict, error_buf

@torch.jit.script
def compute_imitation_reward_cp(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    scale_factor: float,
    dexhand_weight_idx: Dict[str, List[int]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    # type: (Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor], Tensor, float,  Dict[str, List[int]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor]

    joints_vel = states["joints_state"][:, 1:, 7:10]
    current_dof_vel = states["dq"]
    current_obj_vel = states["manip_obj_vel"]
    current_obj_ang_vel = states["manip_obj_ang_vel"]
    
    current_eef_pos = states["base_state"][:, :3]
    current_eef_quat = states["base_state"][:, 3:7]
    current_eef_vel = states["base_state"][:, 7:10]
    current_eef_ang_vel = states["base_state"][:, 10:13]
    current_joints_pos = states["joints_state"][:, 1:, :3]
    current_contact_point = states["contact_point_tip"]
    current_joint_tips_pos = states["joint_pos_tip"]
    


    target_contact_point = target_states["contact_point"]

    diff_contact_point = current_contact_point - target_contact_point
    dis_contact_point = current_joint_tips_pos - target_contact_point

    # contact point pos diff reward
    valid_mask = torch.isfinite(diff_contact_point).all(dim=-1)     
    err = diff_contact_point.abs().mean(dim=-1)                      
    score_valid = 2.0 * torch.exp(-2 * err)                   
    penalty_invalid = -0.3
    group_reward = torch.where(valid_mask, score_valid, torch.full_like(score_valid, penalty_invalid))
    reward_execute = group_reward.sum(dim=-1)   

    

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
        | (torch.norm(current_obj_vel, dim=-1) > 100)
        | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
    )

    # cur joint & contact point diff failure    
    B = dis_contact_point.shape[0]
    thr_xy = 0.15
    thr_z  = 0.7

    abs_disp = dis_contact_point.abs()
    too_far_any = (abs_disp[..., 0] > thr_xy) | (abs_disp[..., 1] > thr_xy) | (abs_disp[..., 2] > thr_z)  

    finite_per_finger = torch.isfinite(dis_contact_point).all(dim=-1) 
    failed_mask = (too_far_any.sum(dim=-1) >= 3)                   

    failed_execute = torch.zeros(B, dtype=torch.bool, device="cuda:0")
    failed_execute.copy_(failed_mask)


    succeeded = (
        progress_buf + 1 + 3 >= max_length
    ) & ~failed_execute 
    
    
    reset_buf = torch.where(
        succeeded | failed_execute,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    
    reward_dict: Dict[str, torch.Tensor] = {
        "reward" : reward_execute
    }

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict, error_buf


def dump_gym_seg_related(gym_obj):
    import inspect, re
    names = dir(gym_obj)
    pat = re.compile(r"(seg|segment|label|id)", re.IGNORECASE)
    hits = [n for n in names if pat.search(n)]
    print("[gym] methods/attrs possibly related to segmentation:")
    for n in sorted(hits):
        try:
            obj = getattr(gym_obj, n)
            sig = ""
            if inspect.ismethod(obj) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
                try:
                    sig = str(inspect.signature(obj))
                except Exception:
                    pass
            print(f"  - {n}{sig}")
        except Exception as e:
            print(f"  - {n} (error: {e})")