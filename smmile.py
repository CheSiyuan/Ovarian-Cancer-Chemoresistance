import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, auc
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import json
import warnings
warnings.filterwarnings('ignore')
from matplotlib import rcParams

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据加载模块 ====================
class H5WSIDataset(Dataset):
    """从H5文件读取WSI特征的数据集类"""
    
    def __init__(self, h5_paths: List[str], labels: List[int], 
                 feature_key: str = 'features', 
                 metadata_key: str = 'metadata',
                 transform=None):
        """
        Args:
            h5_paths: H5文件路径列表，每个文件对应一个WSI
            labels: 每个WSI的标签 (0:敏感, 1:耐药)
            feature_key: H5文件中存储特征的键名
            metadata_key: H5文件中存储元数据的键名
            transform: 数据增强变换
        """
        self.h5_paths = h5_paths
        self.labels = labels
        self.feature_key = feature_key
        self.metadata_key = metadata_key
        self.transform = transform
        self._validate_inputs()
    
    def _validate_inputs(self):
        """验证输入数据"""
        assert len(self.h5_paths) == len(self.labels), \
            "H5文件路径和标签长度不一致"
        
        # 检查H5文件是否存在且可读
        for h5_path in self.h5_paths:
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"H5文件不存在: {h5_path}")
            
            # 快速检查文件结构
            try:
                with h5py.File(h5_path, 'r') as f:
                    if self.feature_key not in f:
                        raise KeyError(f"特征键 '{self.feature_key}' 在文件 {h5_path} 中不存在")
            except Exception as e:
                raise ValueError(f"无法读取H5文件 {h5_path}: {str(e)}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        h5_path = self.h5_paths[idx]
        label = self.labels[idx]
        
        try:
            # 读取H5文件
            with h5py.File(h5_path, 'r') as f:
                # 读取特征数据
                features = f[self.feature_key][:]
                
                # 读取元数据（如果存在）
                metadata = {}
                if self.metadata_key in f:
                    metadata_group = f[self.metadata_key]
                    for key in metadata_group.attrs:
                        metadata[key] = metadata_group.attrs[key]
                
                # 读取坐标信息（如果存在）
                coords = None
                if 'coords' in f:
                    coords = f['coords'][:]
                
        except Exception as e:
            raise RuntimeError(f"读取H5文件失败 {h5_path}: {str(e)}")
        
        # 转换为Tensor
        features_tensor = torch.FloatTensor(features)
        
        # 确保标签是标量张量
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 应用数据增强
        if self.transform:
            features_tensor = self.transform(features_tensor)
        
        # 获取WSI名称
        wsi_name = os.path.splitext(os.path.basename(h5_path))[0]
        
        return features_tensor, label_tensor, wsi_name, metadata
    
    def get_feature_statistics(self) -> Dict:
        """获取数据集中所有WSI的特征统计信息"""
        stats = {
            'total_wsis': len(self),
            'feature_dim': None,
            'instance_counts': [],
            'mean_features_per_wsi': 0
        }
        
        instance_counts = []
        for i in range(len(self)):
            h5_path = self.h5_paths[i]
            with h5py.File(h5_path, 'r') as f:
                features = f[self.feature_key][:]
                instance_counts.append(features.shape[0])
                if stats['feature_dim'] is None:
                    stats['feature_dim'] = features.shape[1]
        
        if instance_counts:
            stats['instance_counts'] = instance_counts
            stats['mean_features_per_wsi'] = np.mean(instance_counts)
            stats['min_features'] = np.min(instance_counts)
            stats['max_features'] = np.max(instance_counts)
        else:
            stats['instance_counts'] = []
            stats['mean_features_per_wsi'] = 0
            stats['min_features'] = 0
            stats['max_features'] = 0
        
        return stats

# ==================== 2. 模型架构模块 ====================
class SpatialAttentionModule(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 512):
        super(SpatialAttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 注意力机制
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 参数自由实例丢弃
        self.dropout_threshold = 0.1
    
    def forward(self, instances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            instances: [n_instances, feature_dim]
        Returns:
            weighted_features: 加权后的特征 [feature_dim]
            attention_weights: 注意力权重 [n_instances]
        """
        # 计算注意力权重
        attention_scores = self.attention_net(instances)  # [n_instances, 1]
        
        # 应用参数自由实例丢弃
        with torch.no_grad():
            max_score = attention_scores.max()
            if max_score > 0.8:  # 抑制过度激活
                attention_scores = attention_scores * (0.8 / max_score)
        
        # 归一化注意力权重
        attention_weights = attention_scores / (attention_scores.sum() + 1e-8)
        
        # 加权聚合
        weighted_features = (instances * attention_weights).sum(dim=0)
        
        return weighted_features, attention_weights.squeeze()

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    
    def __init__(self, input_dim: int = 1024, hidden_dims: List[int] = [512, 256, 128]):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        self.scale_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for hidden_dim in hidden_dims
        ])
        
        # 特征融合层
        total_hidden_dim = sum(hidden_dims)
        self.fusion_net = nn.Linear(total_hidden_dim, 512)
    
    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        """提取多尺度特征"""
        scale_features = []
        
        for net in self.scale_nets:
            instance_features = net(instances)  # [n_instances, hidden_dim]
            scale_feature = instance_features.mean(dim=0)  # [hidden_dim]
            scale_features.append(scale_feature)
        
        # 融合多尺度特征
        fused_features = torch.cat(scale_features, dim=0)  # [total_hidden_dim]
        final_features = self.fusion_net(fused_features.unsqueeze(0))  # [1, 512]
        
        return final_features.squeeze()  # [512]

class OvarianCancerSMMILe(nn.Module):
    """卵巢癌耐药敏感诊断的SMMILe模型"""
    
    def __init__(self, input_dim: int = 1024, num_classes: int = 2,
                 hidden_dims: List[int] = [512, 256, 128]):
        super(OvarianCancerSMMILe, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # SMMILe核心模块
        self.spatial_attention = SpatialAttentionModule(input_dim, 512)
        self.multi_scale_extractor = MultiScaleFeatureExtractor(input_dim, hidden_dims)
        
        # 维度适配层
        self.dimension_adapter = nn.Linear(1024, 512)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # 耐药相关特征增强
        self.drug_resistance_encoder = nn.Linear(512, 512)
        
    def forward(self, instances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            instances: [n_instances, 1024] 一个WSI的实例特征
        Returns:
            logits: 分类logits [num_classes]
            attention_map: 注意力图用于解释性 [n_instances]
        """
        # 1. 空间注意力加权
        weighted_features, attention_weights = self.spatial_attention(instances)
        
        # 2. 多尺度特征提取
        multi_scale_features = self.multi_scale_extractor(instances)
        
        # 3. 维度适配
        weighted_features_512 = self.dimension_adapter(weighted_features.unsqueeze(0)).squeeze()
        
        # 4. 结合注意力加权特征和多尺度特征
        combined_features = 0.6 * weighted_features_512 + 0.4 * multi_scale_features
        
        # 5. 耐药相关特征增强
        resistance_enhanced = self.drug_resistance_encoder(combined_features)
        final_features = combined_features + 0.3 * resistance_enhanced
        
        # 6. 分类
        logits = self.classifier(final_features.unsqueeze(0))  # [1, num_classes]
        
        return logits.squeeze(), attention_weights

# ==================== 3. 数据处理与划分模块 ====================
def load_and_match_data(h5_directory: str, labels_csv: str) -> Tuple[List[str], List[int], Dict]:
    """
    加载H5文件和标签，并进行匹配
    
    Returns:
        h5_paths: 所有H5文件路径
        labels: 所有标签
        wsi_mapping: WSI ID到索引的映射
    """
    # 读取标签CSV文件
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"标签CSV文件不存在: {labels_csv}")
    
    labels_df = pd.read_csv(labels_csv)
    
    # 检查是否包含必要的列
    required_columns = ['wsi_id', 'label']
    for col in required_columns:
        if col not in labels_df.columns:
            # 尝试自动修正列名
            for actual_col in labels_df.columns:
                if col.lower() == actual_col.lower().strip():
                    labels_df.rename(columns={actual_col: col}, inplace=True)
                    print(f"已将列名 '{actual_col}' 重命名为 '{col}'")
                    break
    
    # 获取所有H5文件
    all_h5_files = [f for f in os.listdir(h5_directory) if f.endswith('.h5')]
    print(f"在目录 '{h5_directory}' 中找到 {len(all_h5_files)} 个H5文件。")
    
    # 创建H5路径和标签的完整列表
    h5_paths = []
    h5_labels = []
    wsi_mapping = {}  # wsi_id -> index
    
    for idx, row in labels_df.iterrows():
        wsi_id = str(row['wsi_id']).strip()
        try: 
            label = int(row['label'])
        except ValueError:
            print(f"行 {idx}: 标签值无效 '{row['label']}'，跳过")
            continue
        
        # 获取前12位作为前缀
        wsi_id_prefix = wsi_id[:12] if len(wsi_id) >= 12 else wsi_id
        matched_files = []
        
        # 查找匹配的H5文件
        for h5_file in all_h5_files:
            file_name_without_ext = os.path.splitext(h5_file)[0]
            file_prefix = file_name_without_ext[:12] if len(file_name_without_ext) >= 12 else file_name_without_ext
            
            if file_prefix == wsi_id_prefix:
                h5_path = os.path.join(h5_directory, h5_file)
                matched_files.append(h5_path)
        
        if len(matched_files) == 1:
            h5_paths.append(matched_files[0])
            h5_labels.append(label)
            wsi_mapping[wsi_id] = len(h5_paths) - 1
        elif len(matched_files) > 1:
            h5_paths.append(matched_files[0])
            h5_labels.append(label)
            wsi_mapping[wsi_id] = len(h5_paths) - 1
            print(f"[警告] 行 {idx}: ID '{wsi_id}' 匹配到多个H5文件，取第一个")
    
    if not h5_paths:
        raise ValueError("未找到任何有效的H5文件，请检查文件路径和命名规则")
    
    print(f"\n成功匹配到 {len(h5_paths)} 个WSI数据。")
    
    return h5_paths, h5_labels, wsi_mapping

def create_fixed_external_validation_datasets(
    h5_directory: str, 
    labels_csv: str, 
    fixed_val_indices: List[int] = None,
    n_external_val: int = 76,
    n_train_per_split: int = 178,
    n_internal_val: int = 44,
    n_splits: int = 10,
    random_seed: int = 42
) -> Tuple[List[Dict], List[str], List[int], List[int]]:
    """
    创建外部固定验证集的数据集
    
    Args:
        h5_directory: H5文件目录路径
        labels_csv: 标签CSV文件路径
        fixed_val_indices: 固定的外部验证集索引列表
        n_external_val: 外部验证集样本数
        n_train_per_split: 每个训练组合的样本数
        n_internal_val: 内部验证集样本数
        n_splits: 训练组合数量
        random_seed: 随机种子
        
    Returns:
        dataset_splits: 所有数据集划分
        all_h5_paths: 所有H5文件路径
        all_labels: 所有标签
        fixed_val_indices: 固定的外部验证集索引
    """
    
    print("="*60)
    print("创建外部固定验证集的数据集划分")
    print("="*60)
    
    # 1. 加载数据
    all_h5_paths, all_labels, wsi_mapping = load_and_match_data(h5_directory, labels_csv)
    total_samples = len(all_h5_paths)
    
    print(f"\n总样本数: {total_samples}")
    print(f"期望外部验证集样本数: {n_external_val}")
    print(f"期望每个训练组合样本数: {n_train_per_split}")
    print(f"期望内部验证集样本数: {n_internal_val}")
    
    # 检查样本数是否足够
    if total_samples < n_external_val + n_train_per_split:
        raise ValueError(f"总样本数({total_samples})不足以创建外部验证集({n_external_val})和训练组合({n_train_per_split})")
    
    # 2. 创建或使用指定的固定外部验证集
    all_indices = list(range(total_samples))
    
    if fixed_val_indices is None:
        # 随机选择固定验证集
        np.random.seed(random_seed)
        fixed_val_indices = np.random.choice(
            all_indices, 
            size=n_external_val, 
            replace=False
        ).tolist()
        print(f"随机选择了 {len(fixed_val_indices)} 个样本作为固定外部验证集")
    else:
        print(f"使用指定的 {len(fixed_val_indices)} 个样本作为固定外部验证集")
    
    # 验证集样本
    val_h5_paths = [all_h5_paths[i] for i in fixed_val_indices]
    val_labels = [all_labels[i] for i in fixed_val_indices]
    
    # 剩余样本作为训练池
    train_pool_indices = [i for i in all_indices if i not in fixed_val_indices]
    
    print(f"\n固定外部验证集: {len(val_h5_paths)} 个样本")
    print(f"  敏感(0): {val_labels.count(0)} 个")
    print(f"  耐药(1): {val_labels.count(1)} 个")
    print(f"训练池剩余样本: {len(train_pool_indices)} 个样本")
    
    # 创建固定外部验证集的数据集
    val_external_dataset = H5WSIDataset(val_h5_paths, val_labels)
    val_external_loader = DataLoader(val_external_dataset, batch_size=1, shuffle=False)
    
    # 3. 生成N种不同的训练组合
    dataset_splits = []
    
    for split_idx in range(n_splits):
        print(f"\n生成第{split_idx+1}/{n_splits}个训练组合...")
        
        # 设置随机种子（确保每个组合不同）
        np.random.seed(random_seed + split_idx)
        
        # 从训练池中随机抽取训练样本
        n_available = len(train_pool_indices)
        if n_train_per_split > n_available:
            print(f"警告: 训练池样本不足，使用所有可用样本")
            selected_indices = train_pool_indices.copy()
        else:
            selected_indices = np.random.choice(
                train_pool_indices, 
                size=n_train_per_split, 
                replace=False
            ).tolist()
        
        # 获取选中的H5路径和标签
        train_h5_paths = [all_h5_paths[i] for i in selected_indices]
        train_labels = [all_labels[i] for i in selected_indices]
        
        # 内部划分：训练集 + 内部验证集
        n_total_selected = len(selected_indices)
        n_train_internal = n_total_selected - n_internal_val
        
        if n_internal_val > n_total_selected:
            print(f"警告: 内部验证集样本数超过可用样本，调整为{n_total_selected-1}")
            n_train_internal = 1
            n_internal_val = n_total_selected - 1
        
        # 随机划分内部训练集和验证集
        internal_indices = list(range(n_total_selected))
        np.random.shuffle(internal_indices)
        
        train_internal_idx = internal_indices[:n_train_internal]
        val_internal_idx = internal_indices[n_train_internal:n_train_internal + n_internal_val]
        
        # 内部训练集
        train_internal_h5 = [train_h5_paths[i] for i in train_internal_idx]
        train_internal_labels = [train_labels[i] for i in train_internal_idx]
        
        # 内部验证集
        val_internal_h5 = [train_h5_paths[i] for i in val_internal_idx]
        val_internal_labels = [train_labels[i] for i in val_internal_idx]
        
        # 创建数据集
        train_dataset = H5WSIDataset(train_internal_h5, train_internal_labels)
        val_internal_dataset = H5WSIDataset(val_internal_h5, val_internal_labels)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_internal_loader = DataLoader(val_internal_dataset, batch_size=1, shuffle=False)
        
        # 统计信息
        print(f"  内部训练集: {len(train_dataset)} 个样本")
        print(f"    敏感(0): {train_internal_labels.count(0)} 个")
        print(f"    耐药(1): {train_internal_labels.count(1)} 个")
        print(f"  内部验证集: {len(val_internal_dataset)} 个样本")
        print(f"    敏感(0): {val_internal_labels.count(0)} 个")
        print(f"    耐药(1): {val_internal_labels.count(1)} 个")
        
        # 保存划分信息
        dataset_splits.append({
            'split_idx': split_idx,
            'train_loader': train_loader,
            'val_internal_loader': val_internal_loader,
            'val_external_loader': val_external_loader,  # 所有组合共享
            'train_indices': [selected_indices[i] for i in train_internal_idx],
            'val_internal_indices': [selected_indices[i] for i in val_internal_idx],
            'val_external_indices': fixed_val_indices,  # 固定不变
            'train_h5_paths': train_internal_h5,
            'val_internal_h5_paths': val_internal_h5,
            'val_external_h5_paths': val_h5_paths,
            'train_labels': train_internal_labels,
            'val_internal_labels': val_internal_labels,
            'val_external_labels': val_labels
        })
    
    return dataset_splits, all_h5_paths, all_labels, fixed_val_indices

# ==================== 4. 训练模块 ====================
def train_smmile_model(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=100, 
    learning_rate=1e-4, 
    device='cuda',
    early_stopping_patience=10
):
    """训练SMMILe模型"""
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_auc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for batch_idx, (features, labels, wsi_name, metadata) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 处理每个WSI
            logits, _ = model(features.squeeze(0))
            
            # 修正维度
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            if labels.dim() > 1:
                labels = labels.squeeze()
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            # 计算损失
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_preds.append(pred.item())
            train_targets.append(labels.item())
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds, val_targets, val_probs = [], [], []
        
        with torch.no_grad():
            for features, labels, wsi_name, metadata in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                logits, _ = model(features.squeeze(0))
                
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                if labels.dim() > 1:
                    labels = labels.squeeze()
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)
                
                val_preds.append(pred.item())
                val_targets.append(labels.item())
                val_probs.append(probs.cpu().numpy())
        
        # 计算指标
        train_acc = accuracy_score(train_targets, train_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        
        # 计算AUC
        if len(set(val_targets)) >= 2:
            val_auc = roc_auc_score(val_targets, [p[0][1] for p in val_probs])
        else:
            val_auc = 0.0
        
        # 保存损失和准确率
        train_losses.append(train_loss / max(1, len(train_loader)))
        val_losses.append(val_loss / max(1, len(val_loader)))
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 更新学习率
        scheduler.step(val_auc)
        
        # 早停机制
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # 打印训练信息
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch+1:3d}/{num_epochs}: '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}, '
                  f'Train Acc: {train_acc:.4f}, '
                  f'Val Acc: {val_acc:.4f}, '
                  f'Val AUC: {val_auc:.4f}')
        
        # 检查早停
        if epochs_no_improve >= early_stopping_patience:
            print(f'早停: 连续{early_stopping_patience}个epoch AUC无提升')
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_auc': best_val_auc,
        'final_epoch': epoch + 1
    }

def train_multiple_models_with_fixed_validation(
    dataset_splits, 
    num_epochs=100, 
    learning_rate=1e-4, 
    device='cuda'
):
    """训练多个模型（使用内部验证集训练，外部验证集评估）"""
    
    all_models = []
    all_histories = []
    all_metrics = []
    
    for split_idx, split_data in enumerate(dataset_splits):
        print(f"\n{'='*60}")
        print(f"训练第{split_idx+1}/{len(dataset_splits)}个模型")
        print(f"{'='*60}")
        
        train_loader = split_data['train_loader']
        val_internal_loader = split_data['val_internal_loader']
        val_external_loader = split_data['val_external_loader']
        
        # 初始化模型
        model = OvarianCancerSMMILe(input_dim=1024, num_classes=2)
        
        # 训练模型（使用内部验证集进行早停）
        trained_model, history = train_smmile_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_internal_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device
        )
        
        # 在外部验证集上评估
        print(f"在固定外部验证集上评估模型...")
        external_metrics = evaluate_model(trained_model, val_external_loader, device=device)
        
        # 记录结果
        all_models.append(trained_model)
        all_histories.append(history)
        
        all_metrics.append({
            'split_idx': split_idx,
            'internal_val_metrics': {
                'best_auc': history['best_val_auc'],
                'final_auc': history['val_accs'][-1] if history['val_accs'] else 0,
                'final_loss': history['val_losses'][-1] if history['val_losses'] else 0
            },
            'external_val_metrics': external_metrics,
            'train_indices': split_data['train_indices'],
            'val_internal_indices': split_data['val_internal_indices'],
            'val_external_indices': split_data['val_external_indices']
        })
        
        print(f"\n模型 #{split_idx+1} 性能总结:")
        print(f"  内部验证集最佳AUC: {history['best_val_auc']:.4f}")
        print(f"  固定外部验证集AUC: {external_metrics['auc']:.4f}")
        print(f"  固定外部验证集准确率: {external_metrics['accuracy']:.4f}")
        print(f"  训练epoch数: {history.get('final_epoch', num_epochs)}")
    
    return all_models, all_histories, all_metrics

# ==================== 5. 评估模块 ====================
def evaluate_model_with_roc(model, test_loader, device='cuda'):
    """评估模型性能并计算ROC曲线数据"""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    attention_maps = []
    
    with torch.no_grad():
        for features, labels, wsi_name, metadata in test_loader:
            features, labels = features.to(device), labels.to(device)
            logits, attention_weights = model(features.squeeze(0))
            
            # 维度修正
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            if labels.dim() > 1:
                labels = labels.squeeze()
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            
            all_preds.append(pred.item())
            all_targets.append(labels.item())
            all_probs.append(probs.cpu().numpy())
            attention_maps.append({
                'wsi_name': wsi_name[0],
                'attention': attention_weights.cpu().numpy()
            })
    
    # 计算指标
    accuracy = accuracy_score(all_targets, all_preds)
    
    if len(set(all_targets)) >= 2:
        # 获取正类概率
        positive_probs = [p[0][1] for p in all_probs]
        auc = roc_auc_score(all_targets, positive_probs)
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(all_targets, positive_probs)
        roc_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': auc
        }
    else:
        auc = 0.0
        roc_data = None
        print("警告: 测试集中只有一个类别，无法计算AUC和ROC曲线")
    
    f1 = f1_score(all_targets, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
        'attention_maps': attention_maps,
        'roc_data': roc_data
    }
    
    return metrics

def evaluate_model(model, test_loader, device='cuda'):
    """评估模型性能（兼容旧接口）"""
    return evaluate_model_with_roc(model, test_loader, device)

def plot_roc_curve(roc_data, model_name="模型", save_path=None):
    """绘制ROC曲线并保存
    
    Args:
        roc_data: ROC曲线数据，包含fpr, tpr, auc
        model_name: 模型名称
        save_path: 保存路径，如果为None则显示但不保存
    """
    if roc_data is None:
        print("警告: 无ROC数据可用")
        return
    
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    auc_score = roc_data['auc']
    
    plt.figure(figsize=(8, 6))
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'{model_name} (AUC = {auc_score:.4f})')
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器 (AUC = 0.5)')
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC曲线 - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 添加AUC值标注
    plt.text(0.6, 0.2, f'AUC = {auc_score:.4f}', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # 设置刻度
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存到: {save_path}")
    
    plt.show()
    
    return plt.gcf()

def plot_detailed_roc_analysis(roc_data, model_name="模型", save_path=None):
    """绘制详细的ROC分析图，包括曲线、最佳阈值和性能指标"""
    if roc_data is None:
        print("警告: 无ROC数据可用")
        return
    
    fpr = np.array(roc_data['fpr'])
    tpr = np.array(roc_data['tpr'])
    thresholds = np.array(roc_data['thresholds'])
    auc_score = roc_data['auc']
    
    # 计算Youden指数 (J = sensitivity + specificity - 1 = tpr - fpr)
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1: 标准ROC曲线
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC曲线 (AUC = {auc_score:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', alpha=0.7, label='随机分类器')
    
    # 标记最佳阈值点
    axes[0].scatter(best_fpr, best_tpr, color='red', s=100, zorder=5, 
                    label=f'最佳阈值点 (阈值={best_threshold:.3f})')
    
    # 添加最佳阈值点到坐标轴的连线
    axes[0].axvline(x=best_fpr, color='red', linestyle=':', alpha=0.5, lw=1)
    axes[0].axhline(y=best_tpr, color='red', linestyle=':', alpha=0.5, lw=1)
    
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('假阳性率 (False Positive Rate)', fontsize=12)
    axes[0].set_ylabel('真阳性率 (True Positive Rate)', fontsize=12)
    axes[0].set_title(f'ROC曲线 - {model_name}', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.6, 0.2, f'AUC = {auc_score:.4f}\n'
                          f'最佳阈值 = {best_threshold:.3f}\n'
                          f'TPR = {best_tpr:.3f}, FPR = {best_fpr:.3f}',
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # 子图2: 阈值与性能指标的关系
    axes[1].plot(thresholds, fpr, 'b-', label='假阳性率 (FPR)', alpha=0.7)
    axes[1].plot(thresholds, tpr, 'g-', label='真阳性率 (TPR)', alpha=0.7)
    axes[1].plot(thresholds, youden_index, 'r-', label='Youden指数 (TPR-FPR)', alpha=0.7)
    
    # 标记最佳阈值
    axes[1].axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7, 
                    label=f'最佳阈值 = {best_threshold:.3f}')
    axes[1].scatter([best_threshold], [best_fpr], color='blue', s=50, zorder=5)
    axes[1].scatter([best_threshold], [best_tpr], color='green', s=50, zorder=5)
    axes[1].scatter([best_threshold], [youden_index[best_idx]], color='red', s=50, zorder=5)
    
    axes[1].set_xlabel('分类阈值', fontsize=12)
    axes[1].set_ylabel('性能指标值', fontsize=12)
    axes[1].set_title('阈值与性能指标关系', fontsize=14, fontweight='bold')
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([-0.05, 1.05])
    
    # 在子图2中添加性能指标表格
    metrics_text = f'最佳阈值性能:\n' \
                  f'阈值 = {best_threshold:.3f}\n' \
                  f'TPR = {best_tpr:.3f}\n' \
                  f'FPR = {best_fpr:.3f}\n' \
                  f'特异度 = {1-best_fpr:.3f}\n' \
                  f'Youden指数 = {youden_index[best_idx]:.3f}'
    
    axes[1].text(0.02, 0.02, metrics_text, transform=axes[1].transAxes, 
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'ROC曲线详细分析 - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"详细ROC分析图已保存到: {save_path}")
    
    plt.show()
    
    # 返回性能指标
    performance_metrics = {
        'best_threshold': float(best_threshold),
        'best_tpr': float(best_tpr),
        'best_fpr': float(best_fpr),
        'specificity': float(1 - best_fpr),
        'youden_index': float(youden_index[best_idx]),
        'auc': float(auc_score)
    }
    
    return fig, performance_metrics

def compare_all_models(all_metrics):
    """比较所有模型的性能"""
    print("\n" + "="*60)
    print("所有模型性能比较")
    print("="*60)
    
    # 提取外部验证集AUC
    auc_scores = [m['external_val_metrics']['auc'] for m in all_metrics]
    acc_scores = [m['external_val_metrics']['accuracy'] for m in all_metrics]
    
    # 统计信息
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)
    
    best_idx = np.argmax(auc_scores)
    
    print(f"\n固定外部验证集性能汇总 (76个样本):")
    for i, (auc, acc) in enumerate(zip(auc_scores, acc_scores)):
        print(f"  模型 #{i+1}: AUC = {auc:.4f}, 准确率 = {acc:.4f}")
    
    print(f"\n统计结果:")
    print(f"  AUC: 均值 = {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  准确率: 均值 = {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  最佳模型: 模型 #{best_idx+1} (AUC = {auc_scores[best_idx]:.4f})")
    print(f"  模型数量: {len(auc_scores)}")
    
    return best_idx, auc_scores[best_idx], mean_auc, std_auc

# ==================== 6. 可视化模块 ====================
def plot_training_history(history, title="Training History"):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失曲线
    axes[0].plot(history['train_losses'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_losses'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    axes[1].plot(history['train_accs'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_accs'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_auc_comparison(all_metrics):
    """绘制AUC比较图"""
    split_indices = [m['split_idx'] for m in all_metrics]
    internal_auc = [m['internal_val_metrics']['best_auc'] for m in all_metrics]
    external_auc = [m['external_val_metrics']['auc'] for m in all_metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(split_indices))
    width = 0.35
    
    ax.bar(x - width/2, internal_auc, width, label='内部验证集最佳AUC', color='lightblue', alpha=0.8)
    ax.bar(x + width/2, external_auc, width, label='固定外部验证集AUC', color='lightcoral', alpha=0.8)
    
    ax.set_xlabel('模型编号')
    ax.set_ylabel('AUC Score')
    ax.set_title('不同模型的AUC比较（内部vs外部验证集）')
    ax.set_xticks(x)
    ax.set_xticklabels([f'模型{i+1}' for i in split_indices])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 标记最佳外部AUC
    best_idx = np.argmax(external_auc)
    ax.annotate(f'最佳外部AUC: {external_auc[best_idx]:.4f}',
                xy=(best_idx, external_auc[best_idx]),
                xytext=(best_idx, external_auc[best_idx] + 0.05),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center',
                fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return best_idx

def plot_attention_maps(attention_maps, sample_indices=None, n_samples=3, title="Attention Maps"):
    """绘制注意力图"""
    if sample_indices is None:
        sample_indices = range(min(n_samples, len(attention_maps)))
    
    fig, axes = plt.subplots(1, len(sample_indices), figsize=(15, 4))
    
    if len(sample_indices) == 1:
        axes = [axes]
    
    for idx, ax in zip(sample_indices, axes):
        attention_data = attention_maps[idx]
        attention = attention_data['attention']
        wsi_name = attention_data['wsi_name']
        
        # 对注意力权重进行排序
        sorted_attention = np.sort(attention)[::-1]
        
        # 绘制注意力分布
        ax.bar(range(len(sorted_attention[:50])), sorted_attention[:50], alpha=0.7)
        ax.set_xlabel('实例排序')
        ax.set_ylabel('注意力权重')
        ax.set_title(f'WSI: {wsi_name[:20]}...\nTop 50 注意力权重')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# ==================== 7. 结果保存模块 ====================
def save_training_results(all_models, all_histories, all_metrics, dataset_splits, best_idx):
    """保存训练结果，包括最佳模型的ROC曲线"""
    
    # 保存最佳模型
    best_model = all_models[best_idx]
    best_split_data = dataset_splits[best_idx]
    best_external_metrics = all_metrics[best_idx]['external_val_metrics']
    
    # 1. 绘制并保存最佳模型的ROC曲线
    print(f"\n生成最佳模型（模型 #{best_idx+1}）的ROC曲线...")
    
    # 简单ROC曲线
    roc_fig_simple = plot_roc_curve(
        roc_data=best_external_metrics.get('roc_data'),
        model_name=f"最佳模型 (#{best_idx+1})",
        save_path="best_model_roc_curve.png"
    )
    
    # 详细ROC分析
    if best_external_metrics.get('roc_data'):
        roc_fig_detailed, roc_performance = plot_detailed_roc_analysis(
            roc_data=best_external_metrics['roc_data'],
            model_name=f"最佳模型 (#{best_idx+1})",
            save_path="best_model_roc_detailed_analysis.png"
        )
    else:
        roc_performance = None
        print("警告: 无法生成详细ROC分析，ROC数据缺失")
    
    # 2. 保存模型参数
    model_save_dict = {
        'model_state_dict': best_model.state_dict(),
        'best_split_idx': best_idx,
        'best_auc': all_metrics[best_idx]['external_val_metrics']['auc'],
        'best_accuracy': all_metrics[best_idx]['external_val_metrics']['accuracy'],
        'best_f1_score': all_metrics[best_idx]['external_val_metrics']['f1_score'],
        'train_indices': best_split_data['train_indices'],
        'val_internal_indices': best_split_data['val_internal_indices'],
        'val_external_indices': best_split_data['val_external_indices'],
        'train_h5_paths': best_split_data['train_h5_paths'],
        'val_internal_h5_paths': best_split_data['val_internal_h5_paths'],
        'val_external_h5_paths': best_split_data['val_external_h5_paths'],
        'train_labels': best_split_data['train_labels'],
        'val_internal_labels': best_split_data['val_internal_labels'],
        'val_external_labels': best_split_data['val_external_labels'],
        'roc_performance': roc_performance
    }
    
    torch.save(model_save_dict, 'best_ovarian_cancer_smmile_model_fixed_val.pth')
    print(f"最佳模型已保存为 'best_ovarian_cancer_smmile_model_fixed_val.pth'")
    
    # 3. 保存ROC数据到JSON文件
    if best_external_metrics.get('roc_data'):
        roc_save_data = {
            'model_info': {
                'model_index': best_idx,
                'model_name': f"最佳模型 (#{best_idx+1})",
                'auc': float(best_external_metrics['auc']),
                'accuracy': float(best_external_metrics['accuracy']),
                'f1_score': float(best_external_metrics['f1_score'])
            },
            'roc_curve_data': best_external_metrics['roc_data'],
            'performance_metrics': roc_performance
        }
        
        with open('best_model_roc_data.json', 'w', encoding='utf-8') as f:
            json.dump(roc_save_data, f, indent=4, ensure_ascii=False)
        
        print("ROC曲线数据已保存为 'best_model_roc_data.json'")
    
    # 4. 保存结果汇总
    results_summary = {
        'all_external_auc_scores': [float(m['external_val_metrics']['auc']) for m in all_metrics],
        'all_external_accuracy_scores': [float(m['external_val_metrics']['accuracy']) for m in all_metrics],
        'all_internal_best_auc': [float(m['internal_val_metrics']['best_auc']) for m in all_metrics],
        'best_split_idx': int(best_idx),
        'best_external_auc': float(all_metrics[best_idx]['external_val_metrics']['auc']),
        'best_external_accuracy': float(all_metrics[best_idx]['external_val_metrics']['accuracy']),
        'best_model_roc_performance': roc_performance,
        'config': {
            'n_external_val': 76,
            'n_train_per_split': 178,
            'n_internal_val': 44,
            'n_splits': 10,
            'num_epochs': 100,
            'learning_rate': 1e-4
        },
        'statistics': {
            'mean_external_auc': float(np.mean([m['external_val_metrics']['auc'] for m in all_metrics])),
            'std_external_auc': float(np.std([m['external_val_metrics']['auc'] for m in all_metrics])),
            'mean_external_accuracy': float(np.mean([m['external_val_metrics']['accuracy'] for m in all_metrics])),
            'std_external_accuracy': float(np.std([m['external_val_metrics']['accuracy'] for m in all_metrics]))
        }
    }
    
    with open('training_results_summary_fixed_val.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)
    
    print("训练结果汇总已保存为 'training_results_summary_fixed_val.json'")
    
    return results_summary

# ==================== 8. 主程序 ====================
def main():
    """主程序入口"""
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 配置路径
    h5_directory = r"E:\fh\TCGA-OV\h5_files"  # 替换为你的H5文件所在文件夹路径
    labels_csv = r"E:/fh/ov/ovtrain.csv"  # 替换为你的标签CSV文件路径
    
    # 检查路径是否存在
    if not os.path.exists(h5_directory):
        print(f"错误: H5文件夹不存在: {h5_directory}")
        exit(1)
    
    if not os.path.exists(labels_csv):
        print(f"错误: CSV文件不存在: {labels_csv}")
        exit(1)
    
    try:
        # 1. 创建固定外部验证集的数据集
        print("\n" + "="*60)
        print("步骤1: 创建固定外部验证集的数据集划分")
        print("="*60)
        
        dataset_splits, all_h5_paths, all_labels, fixed_val_indices = create_fixed_external_validation_datasets(
            h5_directory=h5_directory,
            labels_csv=labels_csv,
            fixed_val_indices=None,  # 随机选择76个样本作为固定验证集
            n_external_val=76,       # 外部验证集76个样本
            n_train_per_split=178,   # 每个训练组合178个样本
            n_internal_val=44,       # 每个组合内部验证集44个样本
            n_splits=10,             # 10个训练组合
            random_seed=42
        )
        
        # 2. 训练多个模型
        print("\n" + "="*60)
        print("步骤2: 训练多个模型")
        print("="*60)
        
        all_models, all_histories, all_metrics = train_multiple_models_with_fixed_validation(
            dataset_splits=dataset_splits,
            num_epochs=100,          # 每个模型最多训练100个epoch
            learning_rate=1e-4,
            device=device
        )
        
        # 3. 比较所有模型的性能
        print("\n" + "="*60)
        print("步骤3: 比较所有模型的性能")
        print("="*60)
        
        best_idx, best_auc, mean_auc, std_auc = compare_all_models(all_metrics)
        
        # 4. 可视化结果
        print("\n" + "="*60)
        print("步骤4: 可视化结果")
        print("="*60)
        
        # 绘制AUC比较图
        plot_auc_comparison(all_metrics)
        
        # 绘制最佳模型的训练历史
        print(f"\n绘制最佳模型（模型 #{best_idx+1}）的训练历史...")
        plot_training_history(all_histories[best_idx], title=f"最佳模型(#{best_idx+1})训练历史")
        
        # 绘制最佳模型的注意力图
        print(f"绘制最佳模型的注意力图...")
        best_val_loader = dataset_splits[best_idx]['val_external_loader']
        best_metrics = evaluate_model(all_models[best_idx], best_val_loader, device=device)
        plot_attention_maps(best_metrics['attention_maps'], n_samples=3, 
                           title=f"最佳模型(#{best_idx+1})在固定验证集上的注意力图")
        
        # 5. 保存结果，包括ROC曲线
        print("\n" + "="*60)
        print("步骤5: 保存结果（包括ROC曲线）")
        print("="*60)
        
        save_training_results(all_models, all_histories, all_metrics, dataset_splits, best_idx)
        
        print("\n" + "="*60)
        print("训练完成！")
        print("="*60)
        print("生成的文件:")
        print("  1. best_model_roc_curve.png - 最佳模型的ROC曲线")
        print("  2. best_model_roc_detailed_analysis.png - 详细的ROC分析图")
        print("  3. best_model_roc_data.json - ROC曲线数据")
        print("  4. best_ovarian_cancer_smmile_model_fixed_val.pth - 最佳模型参数")
        print("  5. training_results_summary_fixed_val.json - 训练结果汇总")
        
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()