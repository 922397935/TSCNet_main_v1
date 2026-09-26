# models/scnn_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from omegaconf import DictConfig

# 你工程里已有的模块
from .ptdec import DEC
from .components import InterpretableTransformerEncoder
from ..base import BaseModel


# -----------------------------
# 1) 从相似度/相关矩阵构图并构造 Hodge 拉普拉斯
# -----------------------------
def build_graph_from_similarity(sim: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    sim: [N, N] 相似度(如 |pearson| )，对称，diag 可忽略
    threshold: 阈值，>= threshold 的边被保留
    return: 邻接矩阵 A，二值或权重图皆可；这里给二值（去自环）
    """
    A = (sim >= threshold).to(sim.dtype)
    A = torch.triu(A, diagonal=1)  # 上三角，避免重复边
    A = A + A.T
    A.fill_diagonal_(0)
    return A


def incidence_from_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    根据无向图邻接矩阵 A 构造节点-边关联矩阵 B1 (0/1 带方向约定)。
    约定：对每条边 (i, j), i < j，则该边方向 i -> j
    B1 形状: [num_nodes, num_edges]，每列恰有两个非零：B[i,e]=+1, B[j,e]=-1
    """
    N = A.shape[0]
    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    idx_i = idx_i.to(A.device)
    idx_j = idx_j.to(A.device)
    mask = (A[idx_i, idx_j] > 0)
    i = idx_i[mask]
    j = idx_j[mask]
    E = i.numel()
    if E == 0:
        # 没有边时，返回空关联矩阵（避免后续 matmul 报错）
        return torch.zeros((N, 0), dtype=A.dtype, device=A.device)
    B = torch.zeros((N, E), dtype=A.dtype, device=A.device)
    B[i, torch.arange(E, device=A.device)] = 1.0
    B[j, torch.arange(E, device=A.device)] = -1.0
    return B


def build_hodge_laplacians_from_similarity(sim: torch.Tensor,
                                           threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从相似度矩阵构造 0/1 阶 Hodge 拉普拉斯:
      L0 = B1 * B1^T  (Graph Laplacian 的一种形式)
      L1 = B1^T * B1
    注意：这里未加入 2-单形（三角面）项，等价于 clique 复形截断到 1 阶。
    """
    A = build_graph_from_similarity(sim, threshold)
    B1 = incidence_from_adjacency(A)
    L0 = B1 @ B1.t()                 # [N, N]
    L1 = B1.t() @ B1 if B1.numel() > 0 else torch.zeros((0, 0), dtype=sim.dtype, device=sim.device)  # [E, E]
    return L0, L1


# -----------------------------
# 2) 单纯复形卷积层（简化版 SCNN）
# -----------------------------
class SimplicialConv(nn.Module):
    """
    简化版单纯复形卷积层：
      - 对 0-单形特征 x0（节点特征）做 L0 传播
      - 将节点特征投影到边空间（B1^T x0），在 L1 上传播，再拉回节点空间（B1 *)
    最终输出：节点空间特征（与输入形状一致）
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin_node = nn.Linear(in_channels, out_channels, bias=bias)  # L0 通道
        self.lin_edge = nn.Linear(in_channels, out_channels, bias=bias)  # L1 通道
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lin_node.weight, a=0.2)
        nn.init.kaiming_uniform_(self.lin_edge.weight, a=0.2)
        if self.lin_node.bias is not None:
            nn.init.zeros_(self.lin_node.bias)
        if self.lin_edge.bias is not None:
            nn.init.zeros_(self.lin_edge.bias)

    def forward(self,
                x_node: torch.Tensor,     # [B, N, C_in]
                L0: torch.Tensor,         # [N, N]
                B1: torch.Tensor,         # [N, E]
                L1: Optional[torch.Tensor]  # [E, E] 或 0x0
                ) -> torch.Tensor:        # [B, N, C_out]

        B, N, C_in = x_node.shape
        # 0-单形（节点）扩散：L0 @ x
        x0 = torch.matmul(L0, x_node)                 # [B, N, C_in]，广播 L0 @ 每个 batch
        x0 = self.lin_node(x0)

        # 1-单形（边）通道：B1^T x -> L1 扩散 -> B1 * 回到节点
        if B1.numel() == 0:
            x1_back = torch.zeros((B, N, self.lin_edge.out_features), device=x_node.device, dtype=x_node.dtype)
        else:
            # 投影到边空间
            x_edge = torch.matmul(B1.t(), x_node)     # [B, E, C_in]
            # L1 扩散
            if L1.numel() > 0:
                x_edge = torch.matmul(L1, x_edge)     # [B, E, C_in]
            # 拉回节点空间
            x1_back = torch.matmul(B1, x_edge)        # [B, N, C_in]
            x1_back = self.lin_edge(x1_back)

        out = F.leaky_relu(x0 + x1_back)
        return out


# -----------------------------
# 3) SCNN + Transformer + DEC Pooling 编码器
# -----------------------------
class TransSimplicialEncoder(nn.Module):
    """
    单纯复形卷积 + Transformer +（可选）DEC Pooling
    输入:  x: [B, N, F]
    输出:  x': [B, N_out, F]，以及 assignment（若启用 pooling）
    """
    def __init__(self,
                 input_feature_size: int,
                 input_node_num: int,
                 hidden_size: int,
                 output_node_num: int,
                 pooling: bool = True,
                 orthogonal: bool = True,
                 freeze_center: bool = False,
                 project_assignment: bool = True,
                 # 复形相关
                 complex_threshold: float = 0.7):
        super().__init__()

        # SCNN: 用于每个前向步骤时按 batch 生成/使用 L0/L1/B1
        self.scnn = SimplicialConv(in_channels=input_feature_size,
                                   out_channels=input_feature_size)

        # Transformer
        self.transformer = InterpretableTransformerEncoder(
            d_model=input_feature_size,
            nhead=4,
            dim_feedforward=hidden_size,
            batch_first=True
        )

        # DEC Pooling
        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num,
                           hidden_dimension=input_feature_size,
                           encoder=self.encoder,
                           orthogonal=orthogonal,
                           freeze_center=freeze_center,
                           project_assignment=project_assignment)

        # 复形构造阈值（基于相似度/相关矩阵）
        self.complex_threshold = complex_threshold

    def is_pooling_enabled(self):
        return self.pooling

    @torch.no_grad()
    def _build_batch_laplacians(self, sim_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从一批相似度矩阵构造批 L0/B1/L1
        sim_batch: [B, N, N]，例如 |pearson| 或其他相似度
        return:
          L0_batch: [B, N, N]
          B1_batch: list of [N, E_b] 张量（每个 batch 的边数可能不同）
          L1_batch: list of [E_b, E_b] 张量
        注意：不同 batch 的边数不同，无法打包成统一张量；后续逐样本计算。
        """
        B, N, _ = sim_batch.shape
        L0_list, B1_list, L1_list = [], [], []
        for b in range(B):
            sim = sim_batch[b]
            L0, L1 = build_hodge_laplacians_from_similarity(sim, self.complex_threshold)
            # 同时保留 B1 以便边空间往返
            A = build_graph_from_similarity(sim, self.complex_threshold)
            B1 = incidence_from_adjacency(A)
            L0_list.append(L0)
            B1_list.append(B1)
            L1_list.append(L1)
        # 堆叠 L0（N 固定），B1/L1 以列表返回
        L0_batch = torch.stack(L0_list, dim=0)
        return L0_batch, B1_list, L1_list

    def forward(self,
                x: torch.Tensor,           # [B, N, F] 节点特征
                sim: Optional[torch.Tensor] = None  # [B, N, N] 相似度（如 |pearson|），若 None 则用 x 的相关近似
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, N, F = x.shape

        # 若未提供相似度矩阵，用 x 估一个（对 batch 中每个样本做通道平均后算相关）
        if sim is None:
            with torch.no_grad():
                # 简单近似：对特征维度取均值 => [B, N]
                x_bar = x.mean(dim=-1, keepdim=False)  # [B, N]
                # 归一化后相关近似
                x_bar = (x_bar - x_bar.mean(dim=1, keepdim=True)) / (x_bar.std(dim=1, keepdim=True) + 1e-6)
                sim = torch.einsum('bi,bj->bij', x_bar, x_bar) / N
                sim = sim.abs()

        # 构造批 Hodge 拉普拉斯
        L0_batch, B1_list, L1_list = self._build_batch_laplacians(sim)

        # 逐样本做 SCNN（因每个样本的 B1/L1 大小不同）
        x_sc = []
        for b in range(B):
            x_b = x[b:b+1]                        # [1, N, F]
            L0 = L0_batch[b]
            B1 = B1_list[b]
            L1 = L1_list[b]
            x_b = self.scnn(x_b, L0, B1, L1)      # [1, N, F]
            x_sc.append(x_b)
        x = torch.cat(x_sc, dim=0)                # [B, N, F]

        # Transformer
        x = self.transformer(x)

        # Pooling（DEC）
        if self.pooling:
            x, assignment = self.dec(x)           # x: [B, N, F]（DEC 内部保持形状一致）
            return x, assignment
        else:
            return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment) if self.pooling else None


# -----------------------------
# 4) 顶层模型：SCNN + Transformer（多层可堆叠）+ 分类头
# -----------------------------
class BrainSCNNTransformer(BaseModel):
    """
    用 TransSimplicialEncoder 替换原来的 TransPoolingEncoder。
    仍支持可选位置编码、DEC 多层 pooling，以及最终的 MLP 分类头。
    """
    def __init__(self, config: DictConfig):
        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        # 位置编码（与你原版保持一致）
        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(
                torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim),
                requires_grad=True
            )
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        # 层级尺寸与池化设置
        sizes = list(config.model.sizes)  # 拷贝
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling

        complex_threshold = getattr(config.dataset, "complex_threshold", 0.7)

        # 堆叠多层 TransSimplicialEncoder
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransSimplicialEncoder(
                    input_feature_size=forward_dim,
                    input_node_num=in_sizes[index],
                    hidden_size=1024,
                    output_node_num=size,
                    pooling=do_pooling[index],
                    orthogonal=config.model.orthogonal,
                    freeze_center=config.model.freeze_center,
                    project_assignment=config.model.project_assignment,
                    complex_threshold=complex_threshold
                )
            )

        # 降维 + 分类头（与你原版一致）
        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self,
                time_seires: torch.Tensor,   # 占位保持签名一致（可不使用）
                node_feature: torch.Tensor,  # [B, N, F]
                pearson: Optional[torch.Tensor] = None  # [B, N, N] 可选，若提供将用于复形构造
                ) -> torch.Tensor:

        bz, N, F = node_feature.shape

        # 位置编码
        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)  # [B, N, P]
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)           # [B, N, F+P]

        assignments = []

        # 逐层：SCNN + Transformer +（可选）DEC
        x = node_feature
        sim = pearson.abs() if pearson is not None else None
        for encoder in self.attention_list:
            x, assignment = encoder(x, sim=sim)
            assignments.append(assignment)

        # 降维 + 分类
        x = self.dim_reduction(x)          # [B, N, 8]
        x = x.reshape((bz, -1))            # [B, 8*N_out]
        out = self.fc(x)                   # [B, num_classes]
        return out

    def get_attention_weights(self):
        return [enc.get_attention_weights() for enc in self.attention_list]

    def get_cluster_centers(self) -> List[torch.Tensor]:
        """
        返回每一层（若启用 pooling）的 DEC 聚类中心。
        """
        centers = []
        for enc in self.attention_list:
            if hasattr(enc, 'dec') and enc.pooling:
                centers.append(enc.dec.get_cluster_centers())
        return centers

    def loss(self, assignments: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        """
        聚合所有启用 pooling 的层的 DEC KL 损失。
        assignments: 来自 forward 中收集的 assignment 列表
        """
        decs = [enc for enc in self.attention_list if enc.is_pooling_enabled()]
        assignments = [a for a in assignments if a is not None]
        if not assignments:
            return None
        loss_all = None
        for idx, assignment in enumerate(assignments):
            cur = decs[idx].loss(assignment)
            loss_all = cur if loss_all is None else loss_all + cur
        return loss_all
