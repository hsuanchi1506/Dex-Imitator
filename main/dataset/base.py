from abc import ABC, abstractmethod
import os
from scipy.ndimage import gaussian_filter1d
import numpy as np
import torch
from main.dataset.transform import aa_to_rotmat, caculate_align_mat, rotmat_to_aa
from torch.utils.data import Dataset
from pytorch3d.ops import sample_points_from_meshes
from termcolor import cprint
import pickle
from typing import Optional, List, Tuple, Union

# Matplotlib (safe for headless environments)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _set_axes_equal_and_same_ticks(ax, nticks: int = 5):
    """
    Make X/Y/Z axes:
    1) share the same displayed range (equal aspect ratio)
    2) use the same tick spacing to improve readability
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_mid = float(np.mean(x_limits))
    y_mid = float(np.mean(y_limits))
    z_mid = float(np.mean(z_limits))

    x_range = float(abs(x_limits[1] - x_limits[0]))
    y_range = float(abs(y_limits[1] - y_limits[0]))
    z_range = float(abs(z_limits[1] - z_limits[0]))

    max_range = max(x_range, y_range, z_range)
    if max_range < 1e-12:
        max_range = 1.0  # prevent zero range

    half = max_range / 2.0
    x_lo, x_hi = x_mid - half, x_mid + half
    y_lo, y_hi = y_mid - half, y_mid + half
    z_lo, z_hi = z_mid - half, z_mid + half

    ax.set_xlim3d([x_lo, x_hi])
    ax.set_ylim3d([y_lo, y_hi])
    ax.set_zlim3d([z_lo, z_hi])

    # Use identical tick spacing on all three axes
    ticks = np.linspace(-half, half, nticks)
    ax.set_xticks(ticks + x_mid)
    ax.set_yticks(ticks + y_mid)
    ax.set_zticks(ticks + z_mid)

def _set_fixed_axes(ax,
                    xlim: Tuple[float, float],
                    ylim: Tuple[float, float],
                    zlim: Tuple[float, float],
                    nticks: int = 5):
    """Lock all three axes to fixed absolute ranges and use consistent ticks."""
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)
    if nticks is not None and nticks > 0:
        ax.set_xticks(np.linspace(xlim[0], xlim[1], nticks))
        ax.set_yticks(np.linspace(ylim[0], ylim[1], nticks))
        ax.set_zticks(np.linspace(zlim[0], zlim[1], nticks))


def _draw_frame(ax,
                origin: np.ndarray,
                R: np.ndarray,
                length: float,
                lw: float = 2.0,
                alpha: float = 0.9,
                label: Optional[str] = None):
    """
    Draw a coordinate frame (X=red, Y=green, Z=blue) on a 3D Axes.
    origin: (3,); R: (3,3) rotation matrix; length: axis length.
    """
    o = origin.reshape(3)
    axes = [R[:, 0], R[:, 1], R[:, 2]]
    colors = ['r', 'g', 'b']
    for a, c in zip(axes, colors):
        p = o
        q = o + a * length
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], c=c, linewidth=lw, alpha=alpha)
    if label is not None:
        ax.text(o[0], o[1], o[2], label, fontsize=8, color='k')

def save_point_cloud_images(points_xyz: torch.Tensor,
                            out_path: str = "pc.png",
                            views: Tuple[Tuple[float, float], ...] = ((20, -60), (0, 0), (90, 0)),
                            s: float = 1.5,
                            xlim: Optional[Tuple[float, float]] = None,
                            ylim: Optional[Tuple[float, float]] = None,
                            zlim: Optional[Tuple[float, float]] = None,
                            show_world_axes: bool = True):
    """
    Save multi-view images. If xlim/ylim/zlim are provided, draw within fixed absolute ranges.
    """
    pts = points_xyz.detach().cpu().numpy()  # (N,3)

    # Axis length: use fixed ranges if provided; otherwise estimate from data
    if xlim and ylim and zlim:
        span = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
    else:
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        span = float(np.max(maxs - mins))
    axis_len = 0.1 * span if span > 1e-8 else 0.1

    fig = plt.figure(figsize=(4 * len(views), 4), dpi=150)
    for i, (elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, len(views), i, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=s, c=pts[:, 2], cmap='viridis', depthshade=False)
        ax.view_init(elev=elev, azim=azim)

        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.grid(True)

        if show_world_axes:
            _draw_frame(ax, origin=np.zeros(3), R=np.eye(3), length=axis_len, lw=2.0, label='World')

        if xlim and ylim and zlim:
            _set_fixed_axes(ax, xlim, ylim, zlim, nticks=5)
        else:
            _set_axes_equal_and_same_ticks(ax, nticks=5)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def draw_joint_pc_with_skeleton(ax,
                                points_xyz: torch.Tensor,
                                joints_xyz: torch.Tensor,
                                joint_names: List[str],
                                wrist_xyz: Optional[torch.Tensor] = None,
                                show_world_axes: bool = True,
                                obj_T: Optional[Union[torch.Tensor, np.ndarray]] = None,
                                xlim: Optional[Tuple[float, float]] = None,
                                ylim: Optional[Tuple[float, float]] = None,
                                zlim: Optional[Tuple[float, float]] = None,
                                title: Optional[str] = None,
                                contact_thresh: float = 0.05,
                                contact_color: str = "gold",
                                contact_size: float = 42.0,
                                normal_joint_color: str = "red",
                                normal_joint_size: float = 30.0):
    """
    Draw on an existing 3D Axes: object point cloud + hand joints + finger skeleton
    + (optional) world/object frames. Any joint whose nearest distance to the object
    point cloud is < contact_thresh will be highlighted with contact_color.
    """
    pts = points_xyz.detach().cpu().numpy()   # (N,3)
    jts = joints_xyz.detach().cpu().numpy()   # (J,3)
    name2idx = {n: i for i, n in enumerate(joint_names)}

    finger_chains = {
        "thumb":  ["thumb_proximal",  "thumb_intermediate",  "thumb_distal",  "thumb_tip"],
        "index":  ["index_proximal",  "index_intermediate",  "index_distal",  "index_tip"],
        "middle": ["middle_proximal", "middle_intermediate", "middle_distal", "middle_tip"],
        "ring":   ["ring_proximal",   "ring_intermediate",   "ring_distal",   "ring_tip"],
        "pinky":  ["pinky_proximal",  "pinky_intermediate",  "pinky_distal",  "pinky_tip"],
    }
    chains_idx = {f: [name2idx[n] for n in chain] for f, chain in finger_chains.items()
                  if all(n in name2idx for n in chain)}

    if xlim and ylim and zlim:
        span = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
    else:
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        span = float(np.max(maxs - mins))
    axis_len = 0.1 * span if span > 1e-8 else 0.1


    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c='gray', alpha=0.55, label="object")

    # === Contact detect ===
    dists = np.linalg.norm(jts[:, None, :] - pts[None, :, :], axis=2)  # (J,N)
    min_dist = dists.min(axis=1)                                       # (J,)
    contact_mask = (min_dist < float(contact_thresh))
    non_contact_mask = ~contact_mask

    if non_contact_mask.any():
        j_nc = jts[non_contact_mask]
        ax.scatter(j_nc[:, 0], j_nc[:, 1], j_nc[:, 2],
                   s=normal_joint_size, c=normal_joint_color, alpha=1.0, label="joints")
    if contact_mask.any():
        j_c = jts[contact_mask]
        ax.scatter(j_c[:, 0], j_c[:, 1], j_c[:, 2],
                   s=contact_size, c=contact_color, edgecolors='k', linewidths=0.5,
                   alpha=1.0, label=f"contact < {contact_thresh:.2f} m")

    color_map = {"thumb":"orange","index":"blue","middle":"green","ring":"purple","pinky":"brown"}
    for f, idxs in chains_idx.items():
        for a, b in zip(idxs[:-1], idxs[1:]):
            pa, pb = jts[a], jts[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                    linewidth=2.0, c=color_map.get(f, "black"))

    if wrist_xyz is not None:
        w = wrist_xyz.detach().cpu().numpy().reshape(1, 3)
        ax.scatter(w[:,0], w[:,1], w[:,2], s=40, c='black', label="wrist")
        for pn in ["thumb_proximal","index_proximal","middle_proximal","ring_proximal","pinky_proximal"]:
            if pn in name2idx:
                j = jts[name2idx[pn]]
                ax.plot([w[0,0], j[0]], [w[0,1], j[1]], [w[0,2], j[2]],
                        linewidth=1.5, c='black', alpha=0.8)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.grid(True)
    if show_world_axes:
        _draw_frame(ax, origin=np.zeros(3), R=np.eye(3), length=axis_len, lw=2.0, label='World')
    if obj_T is not None:
        T = obj_T.detach().cpu().numpy() if isinstance(obj_T, torch.Tensor) else np.asarray(obj_T)
        if T.shape == (4, 4):
            Rm, t = T[:3, :3], T[:3, 3]
            _draw_frame(ax, origin=t, R=Rm, length=axis_len*0.75, lw=1.8, label='Object')

    if xlim and ylim and zlim:
        _set_fixed_axes(ax, xlim, ylim, zlim, nticks=5)
    else:
        _set_axes_equal_and_same_ticks(ax, nticks=5)

    if title:
        ax.set_title(title, fontsize=10)


def save_joint_pc_with_skeleton_grid(frames: List[dict],
                                     out_path: str,
                                     layout: Tuple[float, float] = (2, 2),
                                     xlim: Optional[Tuple[float, float]] = None,
                                     ylim: Optional[Tuple[float, float]] = None,
                                     zlim: Optional[Tuple[float, float]] = None,
                                     show_world_axes: bool = True,
                                     panel_inches: float = 6.5,
                                     dpi: int = 180,
                                     outer_margin: float = 0.08,
                                     inner_wspace: float = 0.45,
                                     inner_hspace: float = 0.52,
                                     # ↓ 新增：接觸著色參數（預設 0.05 m）
                                     contact_thresh: float = 0.05,
                                     contact_color: str = "gold",
                                     contact_size: float = 42.0,
                                     normal_joint_color: str = "red",
                                     normal_joint_size: float = 30.0):
    rows, cols = layout
    nslots = rows * cols

    fig_w = cols * panel_inches
    fig_h = rows * panel_inches
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    gs = fig.add_gridspec(
        rows, cols,
        left=outer_margin, right=1.0 - outer_margin,
        bottom=outer_margin, top=1.0 - outer_margin,
        wspace=inner_wspace, hspace=inner_hspace
    )

    for i in range(nslots):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c], projection='3d')
        if i < len(frames):
            fr = frames[i]
            draw_joint_pc_with_skeleton(
                ax,
                fr["points"], fr["joints"], fr["joint_names"],
                wrist_xyz=fr.get("wrist", None),
                show_world_axes=show_world_axes,
                obj_T=fr.get("obj_T", None),
                xlim=xlim, ylim=ylim, zlim=zlim,
                title=fr.get("title", None),
                contact_thresh=contact_thresh,
                contact_color=contact_color,
                contact_size=contact_size,
                normal_joint_color=normal_joint_color,
                normal_joint_size=normal_joint_size,
            )
            ax.xaxis.labelpad = 6; ax.yaxis.labelpad = 6; ax.zaxis.labelpad = 6
            ax.tick_params(pad=2)
            if ax.legend_ is not None:
                ax.legend_.remove() 
        else:
            ax.axis('off')

    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)


class ManipData(Dataset, ABC):
    def __init__(
        self,
        *,
        data_dir: str,
        split: str = "all",
        skip: int = 2,
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        verbose=True,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.split = split
        self.skip = skip
        self.data_pathes = None

        self.dexhand = dexhand
        self.device = device

        self.verbose = verbose

        # Modify this depending on the origin point
        self.transf_offset = torch.eye(4, dtype=torch.float32, device=mujoco2gym_transf.device)

        self.mujoco2gym_transf = mujoco2gym_transf
        self.max_seq_len = max_seq_len

        # Contact distance (Chamfer)
        import chamfer_distance as chd
        self.ch_dist = chd.ChamferDistance()

    def __len__(self):
        return len(self.data_pathes)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @staticmethod
    def compute_velocity(p, time_delta, guassian_filter=True):
        # p: [T, K, 3]
        velocity = np.gradient(p.detach().cpu().numpy(), axis=0) / time_delta
        if guassian_filter:
            velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(velocity).to(p)

    @staticmethod
    def compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        # r: [T, K, 3, 3]
        diff_r = r[1:] @ r[:-1].transpose(-1, -2)  # [T-1, K, 3, 3]
        diff_aa = rotmat_to_aa(diff_r).detach().cpu().numpy()  # [T-1, K, 3]
        diff_angle = np.linalg.norm(diff_aa, axis=-1)  # [T-1, K]
        diff_axis = diff_aa / (diff_angle[:, :, None] + 1e-8)  # [T-1, K, 3]
        angular_velocity = diff_axis * diff_angle[:, :, None] / time_delta  # [T-1, K, 3]
        angular_velocity = np.concatenate([angular_velocity, angular_velocity[-1:]], axis=0)  # [T, K, 3]
        if guassian_filter:
            angular_velocity = gaussian_filter1d(angular_velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(angular_velocity).to(r)

    @staticmethod
    def compute_dof_velocity(dof, time_delta, guassian_filter=True):
        # dof: [T, K]
        velocity = np.gradient(dof.detach().cpu().numpy(), axis=0) / time_delta
        if guassian_filter:
            velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(velocity).to(dof)

    def random_sampling_pc(self, mesh, save_img_path: Optional[str] = None):
        """
        Sample 1000 points uniformly from a mesh surface and (optionally) save a multi-view image.
        """
        numpy_random_state = np.random.get_state()
        torch_random_state = torch.random.get_rng_state()
        torch_random_state_cuda = torch.cuda.get_rng_state()
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rs_verts_obj = sample_points_from_meshes(mesh, 1000, return_normals=False).to(self.device).squeeze(0)

        if save_img_path is not None:
            save_point_cloud_images(
                rs_verts_obj, out_path=save_img_path,
                views=((20, -60), (0, 0), (90, 0)), s=1.2
            )

        # Reset random states
        np.random.set_state(numpy_random_state)
        torch.random.set_rng_state(torch_random_state)
        torch.cuda.set_rng_state(torch_random_state_cuda)

        return rs_verts_obj

    def process_data(self, data, idx, rs_verts_obj):
        """
        - Transform object trajectory, wrist, and MANO joints into gym frame.
        - Build per-frame object point cloud (obj_verts_transf).
        - Compute fingertip distances (tips_distance).
        - Compute, for ALL joints, the nearest object point and the relative vector.
        - Compute velocities/ angular velocities (unchanged logic).
        """
        # Coordinate transform (MuJoCo -> gym)
        data["obj_trajectory"] = self.mujoco2gym_transf @ data["obj_trajectory"]
        data["wrist_pos"] = (self.mujoco2gym_transf[:3, :3] @ data["wrist_pos"].T).T + self.mujoco2gym_transf[:3, 3]
        data["wrist_rot"] = rotmat_to_aa(self.mujoco2gym_transf[:3, :3] @ data["wrist_rot"])
        for k in data["mano_joints"].keys():
            data["mano_joints"][k] = (
                self.mujoco2gym_transf[:3, :3] @ data["mano_joints"][k].T
            ).T + self.mujoco2gym_transf[:3, 3]

        # Per-frame object point cloud in world(gym) frame: (T, N, 3)
        obj_verts_transf = (data["obj_trajectory"][:, :3, :3] @ rs_verts_obj.T[None]).transpose(-1, -2) + \
                           data["obj_trajectory"][:, :3, 3][:, None]
        data["obj_verts_transf"] = obj_verts_transf

        # Fingertip distances via Chamfer (sqrt because ch_dist returns squared)
        tip_list = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
        tips = torch.cat([data["mano_joints"][t_k][:, None] for t_k in tip_list], dim=1)  # (T,5,3)
        tips_near, _, _, _ = self.ch_dist(tips, obj_verts_transf)
        data["tips_distance"] = torch.sqrt(tips_near)  # (T,5)

        # All joints -> nearest object points (robust to joint-list order)
        joint_names_sorted = sorted(list(data["mano_joints"].keys()))
        joints_all = torch.stack([data["mano_joints"][k] for k in joint_names_sorted], dim=1)  # (T, J, 3)

        # Distance matrix (T, J, N)
        dmat = torch.cdist(joints_all, obj_verts_transf, p=2)
        nearest_idx = torch.argmin(dmat, dim=-1)  # (T, J)
        # Gather nearest points (T, J, 3)
        nearest_pts = obj_verts_transf.gather(1, nearest_idx[..., None].expand(-1, -1, 3))
        rel_vec = nearest_pts - joints_all  # (T, J, 3)

        data["joints_all_names"] = joint_names_sorted
        data["joints_all"] = joints_all
        data["nearest_pc_idx"] = nearest_idx
        data["nearest_pc"] = nearest_pts
        data["joint_to_pc_vec"] = rel_vec

        # Velocities (unchanged)
        data["obj_velocity"] = self.compute_velocity(
            data["obj_trajectory"][:, None, :3, 3], 1 / (120 / self.skip), guassian_filter=True
        ).squeeze(1)
        data["obj_angular_velocity"] = self.compute_angular_velocity(
            data["obj_trajectory"][:, None, :3, :3], 1 / (120 / self.skip), guassian_filter=True
        ).squeeze(1)
        data["wrist_velocity"] = self.compute_velocity(
            data["wrist_pos"][:, None], 1 / (120 / self.skip), guassian_filter=True
        ).squeeze(1)
        data["wrist_angular_velocity"] = self.compute_angular_velocity(
            aa_to_rotmat(data["wrist_rot"][:, None]), 1 / (120 / self.skip), guassian_filter=True
        ).squeeze(1)
        data["mano_joints_velocity"] = {}
        for k in data["mano_joints"].keys():
            data["mano_joints_velocity"][k] = self.compute_velocity(
                data["mano_joints"][k], 1 / (120 / self.skip), guassian_filter=True
            )

        # Truncate long sequences (unchanged) and also truncate new fields
        if len(data["obj_trajectory"]) > self.max_seq_len:
            cprint(
                f"WARN: {self.data_pathes[idx]} is too long : {len(data['obj_trajectory'])}, cut to {self.max_seq_len}",
                "yellow",
            )
            data["obj_trajectory"] = data["obj_trajectory"][: self.max_seq_len]
            data["obj_velocity"] = data["obj_velocity"][: self.max_seq_len]
            data["obj_angular_velocity"] = data["obj_angular_velocity"][: self.max_seq_len]
            data["wrist_pos"] = data["wrist_pos"][: self.max_seq_len]
            data["wrist_rot"] = data["wrist_rot"][: self.max_seq_len]
            for k in data["mano_joints"].keys():
                data["mano_joints"][k] = data["mano_joints"][k][: self.max_seq_len]
            data["wrist_velocity"] = data["wrist_velocity"][: self.max_seq_len]
            data["wrist_angular_velocity"] = data["wrist_angular_velocity"][: self.max_seq_len]
            for k in data["mano_joints_velocity"].keys():
                data["mano_joints_velocity"][k] = data["mano_joints_velocity"][k][: self.max_seq_len]
            data["tips_distance"] = data["tips_distance"][: self.max_seq_len]
            # Truncate newly added fields
            T = self.max_seq_len
            data["obj_verts_transf"] = data["obj_verts_transf"][:T]
            data["joints_all"] = data["joints_all"][:T]
            data["nearest_pc_idx"] = data["nearest_pc_idx"][:T]
            data["nearest_pc"] = data["nearest_pc"][:T]
            data["joint_to_pc_vec"] = data["joint_to_pc_vec"][:T]

    def load_retargeted_data(self, data, retargeted_data_path):
        if not os.path.exists(retargeted_data_path):
            if self.verbose:
                cprint(f"\nWARNING: {retargeted_data_path} does not exist.", "red")
                cprint(f"WARNING: This may lead to a slower transfer process or even failure to converge.", "red")
                cprint(
                    f"WARNING: It is recommended to first execute the retargeting code to obtain initial values.\n",
                    "red",
                )
            data.update(
                {
                    "opt_wrist_pos": data["wrist_pos"],
                    "opt_wrist_rot": data["wrist_rot"],
                    "opt_dof_pos": torch.zeros([data["wrist_pos"].shape[0], self.dexhand.n_dofs], device=self.device),
                }
            )
        else:
            opt_params = pickle.load(open(retargeted_data_path, "rb"))
            data.update(
                {
                    "opt_wrist_pos": torch.tensor(opt_params["opt_wrist_pos"], device=self.device),
                    "opt_wrist_rot": torch.tensor(opt_params["opt_wrist_rot"], device=self.device),
                    "opt_dof_pos": torch.tensor(opt_params["opt_dof_pos"], device=self.device),
                    # "opt_joints_pos": torch.tensor(opt_params["opt_joints_pos"], device=self.device), # only for ablation
                }
            )
        data["opt_wrist_velocity"] = self.compute_velocity(
            data["opt_wrist_pos"][:, None], 1 / (120 / self.skip), guassian_filter=True
        ).squeeze(1)
        data["opt_wrist_angular_velocity"] = self.compute_angular_velocity(
            aa_to_rotmat(data["opt_wrist_rot"][:, None]), 1 / (120 / self.skip), guassian_filter=True
        ).squeeze(1)
        data["opt_dof_velocity"] = self.compute_dof_velocity(
            data["opt_dof_pos"], 1 / (120 / self.skip), guassian_filter=True
        )
        # if len(opt_joints_pos) exists, you could compute velocities similarly

        if len(data["opt_wrist_pos"]) > self.max_seq_len:
            data["opt_wrist_pos"] = data["opt_wrist_pos"][: self.max_seq_len]
            data["opt_wrist_rot"] = data["opt_wrist_rot"][: self.max_seq_len]
            data["opt_wrist_velocity"] = data["opt_wrist_velocity"][: self.max_seq_len]
            data["opt_wrist_angular_velocity"] = data["opt_wrist_angular_velocity"][: self.max_seq_len]
            data["opt_dof_pos"] = data["opt_dof_pos"][: self.max_seq_len]
            data["opt_dof_velocity"] = data["opt_dof_velocity"][: self.max_seq_len]

        assert len(data["opt_wrist_pos"]) == len(data["obj_trajectory"])
