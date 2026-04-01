"""
complete_contrastive_mil_enhanced_fixed.py
增强版对比学习多实例学习模型 - 修复外部验证集AUC为0的问题
通过统一数据加载和确定性划分，确保标签匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import random
from itertools import combinations
import json
from datetime import datetime

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ====================================================
# 1. 数据集类（保持不变）
# ====================================================
class WSIContrastiveDataset(Dataset):
    """从H5文件加载WSI特征的数据集类，支持对比学习"""
    
    def __init__(self, h5_paths, labels, feature_key='features', 
                 max_instances=None, augmentation=False):
        """
        Args:
            h5_paths: H5文件路径列表
            labels: 对应的标签列表 (0:敏感, 1:耐药)
            feature_key: H5文件中特征数据的键名
            max_instances: 最大实例数（用于统一大小）
            augmentation: 是否使用数据增强
        """
        self.h5_paths = h5_paths
        self.labels = labels
        self.feature_key = feature_key
        self.max_instances = max_instances
        self.augmentation = augmentation
        
        # 缓存数据
        self.features_cache = {}
        self._preload_data()
    
    def _preload_data(self):
        """预加载所有数据到内存"""
        print("正在加载WSI数据...")
        for idx, h5_path in enumerate(tqdm(self.h5_paths, desc="加载H5文件")):
            try:
                with h5py.File(h5_path, 'r') as f:
                    features = f[self.feature_key]['data'][:]
                    
                    # 如果设置了最大实例数，进行采样或填充
                    if self.max_instances is not None:
                        features = self._adjust_instances(features)
                    
                    self.features_cache[idx] = {
                        'features': features.astype(np.float32),
                        'label': self.labels[idx],
                        'path': h5_path
                    }
                    
            except Exception as e:
                print(f"警告: 无法加载文件 {h5_path}: {str(e)}")
                # 创建一个空特征作为占位符
                dummy_features = np.zeros((100, 1024), dtype=np.float32)
                self.features_cache[idx] = {
                    'features': dummy_features,
                    'label': self.labels[idx],
                    'path': h5_path
                }
        
        print(f"成功加载 {len(self.features_cache)} 个WSI样本")
    
    def _adjust_instances(self, features):
        """调整实例数量到固定大小"""
        n_instances = features.shape[0]
        
        if n_instances >= self.max_instances:
            # 如果实例太多，随机采样
            indices = np.random.choice(n_instances, self.max_instances, replace=False)
            return features[indices]
        else:
            # 如果实例太少，随机重复填充
            n_repeats = self.max_instances // n_instances + 1
            repeated = np.tile(features, (n_repeats, 1))
            indices = np.random.choice(repeated.shape[0], self.max_instances, replace=False)
            return repeated[indices]
    
    def __len__(self):
        return len(self.h5_paths)
    
    def __getitem__(self, idx):
        data = self.features_cache[idx]
        features = data['features']
        label = data['label']
        
        # 数据增强（可选）
        if self.augmentation and random.random() > 0.5:
            # 随机缩放特征
            scale_factor = random.uniform(0.9, 1.1)
            features = features * scale_factor
            
            # 添加随机噪声
            noise = np.random.normal(0, 0.01, features.shape).astype(np.float32)
            features = features + noise
        
        # 转换为PyTorch张量
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return {
            'features': features_tensor,
            'label': label_tensor,
            'wsi_id': os.path.basename(self.h5_paths[idx]),
            'num_instances': features.shape[0]
        }

# ====================================================
# 2. 批处理函数（保持不变）
# ====================================================
def collate_contrastive_batch(batch):
    """处理变长序列的批处理函数"""
    batch_size = len(batch)
    
    # 获取最大实例数
    max_instances = max([item['num_instances'] for item in batch])
    feature_dim = batch[0]['features'].shape[1]
    
    # 初始化批处理张量
    batch_features = torch.zeros(batch_size, max_instances, feature_dim)
    batch_labels = torch.zeros(batch_size, dtype=torch.long)
    batch_masks = torch.zeros(batch_size, max_instances, dtype=torch.bool)
    batch_wsi_ids = []
    
    for i, item in enumerate(batch):
        n_instances = item['num_instances']
        features = item['features']
        
        # 填充特征
        batch_features[i, :n_instances] = features
        
        # 创建掩码（有效位置为True）
        batch_masks[i, :n_instances] = True
        
        # 标签和ID
        batch_labels[i] = item['label']
        batch_wsi_ids.append(item['wsi_id'])
    
    return {
        'features': batch_features,
        'label': batch_labels,
        'mask': batch_masks,
        'wsi_id': batch_wsi_ids,
        'max_instances': max_instances
    }

# ====================================================
# 3. ContrastiveMIL模型（保持不变）
# ====================================================
class ContrastiveMIL(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, n_classes=2, tau=0.1, dropout=0.2):
        super().__init__()
        self.tau = tau  # 温度系数
        self.hidden_dim = hidden_dim
        
        # 特征编码器（共享）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 注意力分支
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    # 对比损失计算（实例级）
    def contrastive_loss(self, h, mask):
        """改进的对比损失，考虑批次和掩码"""
        batch_size, max_instances, hidden_dim = h.shape
        
        # 展平并应用掩码
        h_flat = h.reshape(-1, hidden_dim)
        mask_flat = mask.reshape(-1)
        
        # 只对有效实例计算对比损失
        valid_indices = torch.where(mask_flat)[0]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=h.device)
        
        h_valid = h_flat[valid_indices]
        
        # 归一化
        h_norm = F.normalize(h_valid, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(h_norm, h_norm.T) / self.tau
        
        # 创建掩码（排除自身）
        n_valid = len(h_valid)
        pos_mask = torch.eye(n_valid, device=h.device).bool()
        neg_mask = ~pos_mask
        
        # 计算InfoNCE损失
        pos_sim = sim_matrix[pos_mask].unsqueeze(1)
        neg_sim = sim_matrix[neg_mask].reshape(n_valid, -1)
        
        # 限制负样本数量以防止内存爆炸
        if neg_sim.shape[1] > 100:
            neg_indices = torch.randperm(neg_sim.shape[1])[:100]
            neg_sim = neg_sim[:, neg_indices]
        
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        targets = torch.zeros(n_valid, device=h.device, dtype=torch.long)
        
        loss = F.cross_entropy(logits, targets)
        return loss
    
    def forward(self, x, mask, contrastive_loss_weight=0.1, training=False):
        """
        Args:
            x: [batch_size, max_instances, input_dim]
            mask: [batch_size, max_instances]
            contrastive_loss_weight: 对比损失权重
            training: 是否在训练模式
        """
        batch_size, max_instances, input_dim = x.shape
        
        # 1. 特征编码
        x_flat = x.reshape(-1, input_dim)
        h_encoded_flat = self.encoder(x_flat)
        h_encoded = h_encoded_flat.reshape(batch_size, max_instances, self.hidden_dim)
        
        # 2. 计算对比损失（仅在训练时）
        contrast_loss = torch.tensor(0.0, device=x.device)
        if training and contrastive_loss_weight > 0:
            contrast_loss = self.contrastive_loss(h_encoded, mask)
        
        # 3. 注意力加权聚合
        attention_scores = []
        bag_features = []
        
        for i in range(batch_size):
            # 获取有效实例
            valid_mask = mask[i]
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                # 如果没有有效实例，使用零向量
                bag_feature = torch.zeros(self.hidden_dim, device=x.device)
                attention_score = torch.zeros(max_instances, device=x.device)
            else:
                valid_features = h_encoded[i, valid_indices]
                
                # 计算注意力分数
                scores = self.attention(valid_features)
                A = F.softmax(scores, dim=0)
                
                # 加权聚合
                bag_feature = torch.mm(A.T, valid_features).squeeze(0)
                
                # 将注意力分数映射回原始大小
                attention_score = torch.zeros(max_instances, device=x.device)
                attention_score[valid_indices] = A.squeeze()
            
            bag_features.append(bag_feature)
            attention_scores.append(attention_score)
        
        # 堆叠结果
        bag_features = torch.stack(bag_features, dim=0)  # [batch_size, hidden_dim]
        attention_scores = torch.stack(attention_scores, dim=0)  # [batch_size, max_instances]
        
        # 4. 分类
        logits = self.classifier(bag_features)  # [batch_size, n_classes]
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        # 5. 总损失
        total_contrast_loss = contrast_loss * contrastive_loss_weight
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions,
            'attention_scores': attention_scores,
            'bag_features': bag_features,
            'contrast_loss': total_contrast_loss
        }

# ====================================================
# 4. 增强版训练器（支持多种训练组合）（保持不变）
# ====================================================
class EnhancedContrastiveMILTrainer:
    """增强版训练器，支持多种训练组合和外部验证"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # 训练历史记录
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': [],
            'contrast_loss': []
        }
    
    def train_epoch(self, train_loader, contrastive_weight=0.1):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        total_contrast_loss = 0.0
        
        pbar = tqdm(train_loader, desc="训练", leave=False)
        for batch in pbar:
            # 将数据移到设备
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(features, masks, contrastive_weight, training=True)
            
            # 计算损失
            classification_loss = self.criterion(outputs['logits'], labels)
            contrast_loss = outputs['contrast_loss']
            loss = classification_loss + contrast_loss
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计信息
            total_loss += loss.item()
            total_contrast_loss += contrast_loss.item() if contrast_loss > 0 else 0
            
            # 收集预测结果
            all_preds.extend(outputs['predictions'].cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(outputs['probabilities'][:, 1].detach().cpu().numpy())  # 类别1的概率
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls_loss': f'{classification_loss.item():.4f}',
                'contrast': f'{contrast_loss.item():.4f}' if contrast_loss > 0 else '0.0'
            })
        
        # 计算指标
        avg_loss = total_loss / len(train_loader)
        avg_contrast_loss = total_contrast_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        
        # 计算AUC
        if len(set(all_targets)) >= 2:
            auc = roc_auc_score(all_targets, all_probs)
        else:
            auc = 0.0
        
        return avg_loss, accuracy, auc, avg_contrast_loss
    
    def validate(self, val_loader, contrastive_weight=0.0):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        all_attention = []
        all_wsi_ids = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证", leave=False):
                # 将数据移到设备
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                masks = batch['mask'].to(self.device)
                wsi_ids = batch['wsi_id']
                
                # 前向传播
                outputs = self.model(features, masks, contrastive_weight, training=False)
                
                # 计算损失
                loss = self.criterion(outputs['logits'], labels)
                total_loss += loss.item()
                
                # 收集结果
                all_preds.extend(outputs['predictions'].cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(outputs['probabilities'].cpu().numpy())
                all_attention.extend(outputs['attention_scores'].cpu().numpy())
                all_wsi_ids.extend(wsi_ids)
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        
        # 计算AUC
        if len(set(all_targets)) >= 2:
            auc = roc_auc_score(all_targets, [p[1] for p in all_probs])
        else:
            auc = 0.0
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1_score(all_targets, all_preds),
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs,
            'attention': all_attention,
            'wsi_ids': all_wsi_ids
        }
        
        return results
    
    def train(self, train_loader, val_loader, num_epochs=50, 
              contrastive_weight=0.1, early_stopping_patience=10):
        """完整训练过程"""
        
        print(f"开始训练，共 {num_epochs} 个epochs")
        print(f"对比损失权重: {contrastive_weight}")
        
        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('='*60)
            
            # 训练
            train_loss, train_acc, train_auc, train_contrast = self.train_epoch(
                train_loader, contrastive_weight
            )
            
            # 验证
            val_results = self.validate(val_loader, contrastive_weight=0.0)
            
            # 更新学习率
            self.scheduler.step(val_results['auc'])
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_auc'].append(train_auc)
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_acc'].append(val_results['accuracy'])
            self.history['val_auc'].append(val_results['auc'])
            self.history['contrast_loss'].append(train_contrast)
            
            # 打印结果
            print(f"\n训练结果:")
            print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | 训练AUC: {train_auc:.4f}")
            print(f"  对比损失: {train_contrast:.4f}")
            print(f"\n验证结果:")
            print(f"  验证损失: {val_results['loss']:.4f} | 验证准确率: {val_results['accuracy']:.4f} | 验证AUC: {val_results['auc']:.4f}")
            
            # 早停检查
            if val_results['auc'] > best_val_auc:
                best_val_auc = val_results['auc']
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"  新的最佳AUC: {best_val_auc:.4f}")
                
                # 保存最佳模型
                model_save_path = f'best_model_epoch_{epoch+1}_auc_{best_val_auc:.4f}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': best_val_auc,
                    'history': self.history
                }, model_save_path)
                
            else:
                patience_counter += 1
                print(f"  AUC未提升，早停计数: {patience_counter}/{early_stopping_patience}")
            
            # 早停触发
            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发！已连续 {early_stopping_patience} 个epoch未改善。")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n恢复最佳模型，验证AUC: {best_val_auc:.4f}")
        
        return self.history, best_val_auc

# ====================================================
# 5. 增强版数据准备函数（支持手动划分训练/验证集）- 修改版本
# ====================================================
def prepare_data_enhanced_fixed(h5_directory, labels_csv, fixed_test_size=76, n_training_combinations=10, 
                                max_instances=2000, batch_size=4, match_prefix_length=12, random_seed=42):
    """修复版：增强版数据准备函数，支持固定验证集和多种训练组合
    
    Args:
        h5_directory: H5文件目录
        labels_csv: 标签CSV文件路径
        fixed_test_size: 固定的外部验证集大小
        n_training_combinations: 训练组合数量
        max_instances: 最大实例数
        batch_size: 批大小
        match_prefix_length: 匹配前缀长度
        random_seed: 随机种子
    """
    
    # 设置随机种子
    set_seed(random_seed)
    
    # 1. 读取CSV标签文件
    print("步骤1: 读取标签文件...")
    labels_df = pd.read_csv(labels_csv)
    
    # 检查列名
    required_cols = ['wsi_id', 'label']
    for col in required_cols:
        if col not in labels_df.columns:
            # 尝试自动检测
            for actual_col in labels_df.columns:
                if col in actual_col.lower():
                    labels_df = labels_df.rename(columns={actual_col: col})
                    break
    
    # 确保列存在
    if 'wsi_id' not in labels_df.columns or 'label' not in labels_df.columns:
        raise ValueError("CSV文件中必须包含 'wsi_id' 和 'label' 列")
    
    # 2. 获取所有H5文件
    print("\n步骤2: 获取所有H5文件...")
    all_h5_files = []
    for file in os.listdir(h5_directory):
        if file.endswith('.h5'):
            file_path = os.path.join(h5_directory, file)
            all_h5_files.append(file_path)
    print(f"在目录中发现 {len(all_h5_files)} 个H5文件")
    
    # 3. 构建H5文件名到路径的映射
    h5_name_to_path = {}
    for h5_path in all_h5_files:
        h5_name = os.path.splitext(os.path.basename(h5_path))[0]
        h5_name_to_path[h5_name] = h5_path
    
    # 4. 统一匹配CSV标签与H5文件
    print(f"\n步骤3: 统一匹配CSV标签与H5文件 (匹配前缀长度={match_prefix_length})...")
    matched_pairs = []  # 存储(文件路径, 标签, WSI_ID)的三元组
    
    for idx, row in labels_df.iterrows():
        wsi_id = str(row['wsi_id']).strip()
        try: 
            label = int(row['label'])
        except ValueError:
            print(f"警告: 跳过无效标签的样本: WSI_ID={wsi_id}, label={row['label']}")
            continue
        
        matched = False
        for h5_name, h5_path in h5_name_to_path.items():
            # 使用前缀匹配
            if wsi_id[:match_prefix_length] == h5_name[:match_prefix_length]:
                matched_pairs.append((h5_path, label, wsi_id))
                matched = True
                # 匹配成功后从映射中移除，避免重复匹配
                del h5_name_to_path[h5_name]
                break
        
        if not matched:
            print(f"警告: 未匹配到H5文件的WSI ID: {wsi_id}")
    
    print(f"成功匹配 {len(matched_pairs)} 个样本")
    
    # 5. 检查数据量
    total_samples = len(matched_pairs)
    if total_samples < 178 + fixed_test_size:
        print(f"错误: 总样本数({total_samples})小于所需的训练集(178) + 固定验证集({fixed_test_size}) = {178+fixed_test_size}")
        return None
    
    # 6. 固定验证集划分
    print(f"\n步骤4: 划分固定验证集 ({fixed_test_size} 个样本)...")
    # 按标签分层抽样以确保类别平衡
    matched_paths = [p[0] for p in matched_pairs]
    matched_labels = [p[1] for p in matched_pairs]
    matched_ids = [p[2] for p in matched_pairs]
    
    # 使用分层抽样划分固定验证集
    try:
        # 注意：train_test_split返回的剩余数据即为训练/验证池
        _, test_paths, _, test_labels, _, test_ids = train_test_split(
            matched_paths, matched_labels, matched_ids,
            test_size=fixed_test_size,
            stratify=matched_labels if len(set(matched_labels)) > 1 else None,
            random_state=random_seed
        )
    except:
        # 如果分层抽样失败，使用随机抽样
        from sklearn.utils import resample
        indices = np.arange(len(matched_pairs))
        test_indices = np.random.choice(indices, size=fixed_test_size, replace=False)
        
        test_paths = [matched_paths[i] for i in test_indices]
        test_labels = [matched_labels[i] for i in test_indices]
        test_ids = [matched_ids[i] for i in test_indices]
    
    # 剩余数据作为训练/验证池
    train_val_indices = [i for i in range(len(matched_pairs)) if matched_paths[i] not in test_paths]
    train_val_paths = [matched_paths[i] for i in train_val_indices]
    train_val_labels = [matched_labels[i] for i in train_val_indices]
    train_val_ids = [matched_ids[i] for i in train_val_indices]
    
    print(f"固定验证集大小: {len(test_paths)} (敏感:{test_labels.count(0)} 耐药:{test_labels.count(1)})")
    print(f"训练/验证池大小: {len(train_val_paths)} (敏感:{train_val_labels.count(0)} 耐药:{train_val_labels.count(1)})")
    
    # 7. 生成多种训练组合
    print(f"\n步骤5: 生成 {n_training_combinations} 种训练组合...")
    
    all_training_combinations = []
    
    for comb_idx in range(n_training_combinations):
        # 从训练/验证池中随机选择178个样本作为本次组合的训练集
        if len(train_val_paths) < 178:
            print(f"错误: 训练/验证池只有{len(train_val_paths)}个样本，无法选择178个")
            return None
        
        # 随机选择178个样本
        selected_indices = np.random.choice(len(train_val_paths), 178, replace=False)
        train_comb_paths = [train_val_paths[i] for i in selected_indices]
        train_comb_labels = [train_val_labels[i] for i in selected_indices]
        train_comb_ids = [train_val_ids[i] for i in selected_indices]
        
        # 从训练组合中分割验证集（用于训练过程中的验证）
        try:
            train_paths_split, val_paths_split, train_labels_split, val_labels_split, train_ids_split, val_ids_split = train_test_split(
                train_comb_paths, train_comb_labels, train_comb_ids,
                test_size=0.2,
                stratify=train_comb_labels if len(set(train_comb_labels)) > 1 else None,
                random_state=random_seed + comb_idx
            )
        except:
            train_paths_split, val_paths_split, train_labels_split, val_labels_split, train_ids_split, val_ids_split = train_test_split(
                train_comb_paths, train_comb_labels, train_comb_ids,
                test_size=0.2,
                random_state=random_seed + comb_idx
            )
        
        # 创建数据集
        train_dataset = WSIContrastiveDataset(
            train_paths_split, train_labels_split,
            max_instances=max_instances,
            augmentation=True
        )
        
        val_dataset = WSIContrastiveDataset(
            val_paths_split, val_labels_split,
            max_instances=max_instances,
            augmentation=False
        )
        
        external_test_dataset = WSIContrastiveDataset(
            test_paths, test_labels,
            max_instances=max_instances,
            augmentation=False
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, collate_fn=collate_contrastive_batch,
            num_workers=0, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, collate_fn=collate_contrastive_batch,
            num_workers=0, pin_memory=True
        )
        
        external_test_loader = DataLoader(
            external_test_dataset, batch_size=batch_size,
            shuffle=False, collate_fn=collate_contrastive_batch,
            num_workers=0, pin_memory=True
        )
        
        # 存储组合信息
        combination_info = {
            'combination_id': comb_idx + 1,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'external_test_loader': external_test_loader,
            'train_paths': train_paths_split,
            'train_labels': train_labels_split,
            'train_wsi_ids': train_ids_split,
            'val_paths': val_paths_split,
            'val_labels': val_labels_split,
            'val_wsi_ids': val_ids_split,
            'external_test_paths': test_paths,
            'external_test_labels': test_labels,
            'external_test_wsi_ids': test_ids,
            'train_stats': {
                'sensitive': train_labels_split.count(0),
                'resistant': train_labels_split.count(1),
                'total': len(train_labels_split)
            },
            'val_stats': {
                'sensitive': val_labels_split.count(0),
                'resistant': val_labels_split.count(1),
                'total': len(val_labels_split)
            },
            'external_test_stats': {
                'sensitive': test_labels.count(0),
                'resistant': test_labels.count(1),
                'total': len(test_labels)
            }
        }
        
        all_training_combinations.append(combination_info)
        
        print(f"组合 {comb_idx+1}: 训练集={len(train_labels_split)} (敏感:{train_labels_split.count(0)} 耐药:{train_labels_split.count(1)}), "
              f"验证集={len(val_labels_split)} (敏感:{val_labels_split.count(0)} 耐药:{val_labels_split.count(1)}), "
              f"外部验证集={len(test_labels)} (敏感:{test_labels.count(0)} 耐药:{test_labels.count(1)})")
    
    return all_training_combinations

# ====================================================
# 6. 多组合训练和评估（保持不变）
# ====================================================
def train_multiple_combinations(all_combinations, config):
    """训练多种训练组合，并在外部验证集上评估"""
    
    all_results = []
    best_external_auc = 0.0
    best_combination_id = 0
    best_model = None
    
    print("=" * 80)
    print("开始多组合训练")
    print("=" * 80)
    
    for i, comb_info in enumerate(all_combinations):
        print(f"\n{'='*60}")
        print(f"训练组合 {i+1}/{len(all_combinations)}")
        print(f"{'='*60}")
        
        # 打印组合统计信息
        train_stats = comb_info['train_stats']
        val_stats = comb_info['val_stats']
        test_stats = comb_info['external_test_stats']
        
        print(f"训练集: {train_stats['total']} 个样本 (敏感:{train_stats['sensitive']}, 耐药:{train_stats['resistant']})")
        print(f"验证集: {val_stats['total']} 个样本 (敏感:{val_stats['sensitive']}, 耐药:{val_stats['resistant']})")
        print(f"外部验证集: {test_stats['total']} 个样本 (敏感:{test_stats['sensitive']}, 耐药:{test_stats['resistant']})")
        
        # 初始化新模型
        model = ContrastiveMIL(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            n_classes=config['n_classes'],
            tau=0.1,
            dropout=0.2
        )
        
        # 初始化训练器
        trainer = EnhancedContrastiveMILTrainer(model, device=config['device'])
        
        # 训练模型
        history, best_val_auc = trainer.train(
            train_loader=comb_info['train_loader'],
            val_loader=comb_info['val_loader'],
            num_epochs=config['num_epochs'],
            contrastive_weight=config['contrastive_weight'],
            early_stopping_patience=config['early_stopping_patience']
        )
        
        # 在外部验证集上评估
        print(f"\n在外部验证集上评估组合 {i+1}...")
        external_test_results = trainer.validate(
            comb_info['external_test_loader'],
            contrastive_weight=0.0
        )
        
        external_auc = external_test_results['auc']
        
        print(f"组合 {i+1} 结果:")
        print(f"  最佳验证AUC: {best_val_auc:.4f}")
        print(f"  外部验证集AUC: {external_auc:.4f}")
        print(f"  外部验证集准确率: {external_test_results['accuracy']:.4f}")
        print(f"  外部验证集F1分数: {external_test_results['f1_score']:.4f}")
        
        # 保存结果
        result = {
            'combination_id': i + 1,
            'best_val_auc': best_val_auc,
            'external_test_auc': external_auc,
            'external_test_accuracy': external_test_results['accuracy'],
            'external_test_f1': external_test_results['f1_score'],
            'external_test_predictions': external_test_results['predictions'],
            'external_test_targets': external_test_results['targets'],
            'external_test_probabilities': external_test_results['probabilities'],
            'external_test_wsi_ids': external_test_results['wsi_ids'],
            'train_stats': train_stats,
            'val_stats': val_stats,
            'external_test_stats': test_stats,
            'model_state_dict': model.state_dict(),
            'history': history
        }
        
        all_results.append(result)
        
        # 检查是否为最佳模型
        if external_auc > best_external_auc:
            best_external_auc = external_auc
            best_combination_id = i + 1
            best_model = model
            print(f"  → 新的最佳外部验证AUC: {best_external_auc:.4f}")
        
        # 保存当前组合的结果
        save_combination_result(result, i+1)
        
        # 绘制当前组合的训练历史
        plot_training_history(history, f"training_history_combination_{i+1}.png")
        
        # 绘制当前组合的外部验证集ROC曲线
        plot_roc_curve(
            external_test_results['targets'],
            external_test_results['probabilities'],
            f"roc_curve_external_combination_{i+1}.png"
        )
    
    return all_results, best_external_auc, best_combination_id, best_model

def save_combination_result(result, comb_id):
    """保存单个组合的结果"""
    # 保存预测结果
    results_df = pd.DataFrame({
        'wsi_id': result['external_test_wsi_ids'],
        'true_label': result['external_test_targets'],
        'predicted_label': result['external_test_predictions'],
        'probability_sensitive': [p[0] for p in result['external_test_probabilities']],
        'probability_resistant': [p[1] for p in result['external_test_probabilities']]
    })
    results_df.to_csv(f'combination_{comb_id}_predictions.csv', index=False, encoding='utf-8-sig')
    
    # 保存模型
    torch.save({
        'combination_id': comb_id,
        'model_state_dict': result['model_state_dict'],
        'best_val_auc': result['best_val_auc'],
        'external_test_auc': result['external_test_auc'],
        'train_stats': result['train_stats'],
        'external_test_stats': result['external_test_stats']
    }, f'combination_{comb_id}_model.pth')
    
    # 保存训练历史
    history_df = pd.DataFrame(result['history'])
    history_df.to_csv(f'combination_{comb_id}_history.csv', index=False, encoding='utf-8-sig')
    
    print(f"组合 {comb_id} 的结果已保存")

# ====================================================
# 7. 结果分析和可视化（保持不变）
# ====================================================
def analyze_all_results(all_results, best_combination_id, best_external_auc):
    """分析所有组合的结果"""
    
    print("\n" + "="*80)
    print("所有组合结果分析")
    print("="*80)
    
    # 收集所有组合的结果
    combination_ids = []
    val_aucs = []
    external_aucs = []
    external_accuracies = []
    external_f1s = []
    
    for result in all_results:
        combination_ids.append(result['combination_id'])
        val_aucs.append(result['best_val_auc'])
        external_aucs.append(result['external_test_auc'])
        external_accuracies.append(result['external_test_accuracy'])
        external_f1s.append(result['external_test_f1'])
    
    # 创建结果汇总DataFrame
    summary_df = pd.DataFrame({
        '组合ID': combination_ids,
        '最佳验证AUC': val_aucs,
        '外部验证AUC': external_aucs,
        '外部验证准确率': external_accuracies,
        '外部验证F1分数': external_f1s
    })
    
    # 排序并显示
    summary_df = summary_df.sort_values('外部验证AUC', ascending=False)
    print("\n所有组合结果汇总（按外部验证AUC降序排列）:")
    print(summary_df.to_string(float_format=lambda x: f'{x:.4f}'))
    
    # 保存汇总结果
    summary_df.to_csv('all_combinations_summary.csv', index=False, encoding='utf-8-sig')
    
    # 找出最佳组合
    best_result = None
    for result in all_results:
        if result['combination_id'] == best_combination_id:
            best_result = result
            break
    
    if best_result:
        print(f"\n{'='*60}")
        print(f"最佳组合: 组合 {best_combination_id}")
        print(f"{'='*60}")
        print(f"最佳验证AUC: {best_result['best_val_auc']:.4f}")
        print(f"外部验证AUC: {best_result['external_test_auc']:.4f}")
        print(f"外部验证准确率: {best_result['external_test_accuracy']:.4f}")
        print(f"外部验证F1分数: {best_result['external_test_f1']:.4f}")
        print(f"训练集统计: 敏感:{best_result['train_stats']['sensitive']} 耐药:{best_result['train_stats']['resistant']} 总数:{best_result['train_stats']['total']}")
        print(f"验证集统计: 敏感:{best_result['val_stats']['sensitive']} 耐药:{best_result['val_stats']['resistant']} 总数:{best_result['val_stats']['total']}")
        print(f"外部验证集统计: 敏感:{best_result['external_test_stats']['sensitive']} 耐药:{best_result['external_test_stats']['resistant']} 总数:{best_result['external_test_stats']['total']}")
    
    # 绘制所有组合的AUC对比图
    plot_all_combinations_auc(combination_ids, external_aucs, best_combination_id)
    
    return summary_df, best_result

def plot_all_combinations_auc(combination_ids, external_aucs, best_combination_id):
    """绘制所有组合的外部验证AUC对比图"""
    plt.figure(figsize=(12, 6))
    
    # 创建颜色列表，最佳组合用红色
    colors = ['blue' if cid != best_combination_id else 'red' for cid in combination_ids]
    
    # 绘制柱状图
    bars = plt.bar(range(len(combination_ids)), external_aucs, color=colors, alpha=0.7)
    
    # 添加数值标签
    for i, (bar, auc_val) in enumerate(zip(bars, external_aucs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('组合ID')
    plt.ylabel('外部验证AUC')
    plt.title('所有训练组合的外部验证AUC对比')
    plt.xticks(range(len(combination_ids)), combination_ids)
    plt.axhline(y=np.mean(external_aucs), color='green', linestyle='--', 
                label=f'平均AUC: {np.mean(external_aucs):.3f}')
    plt.axhline(y=np.median(external_aucs), color='orange', linestyle='--', 
                label=f'中位数AUC: {np.median(external_aucs):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('all_combinations_auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制箱线图
    plt.figure(figsize=(8, 6))
    plt.boxplot(external_aucs)
    plt.xlabel('所有组合')
    plt.ylabel('外部验证AUC')
    plt.title('外部验证AUC分布')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('all_combinations_auc_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='训练损失')
    axes[0, 0].plot(history['val_loss'], label='验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(history['train_acc'], label='训练准确率')
    axes[0, 1].plot(history['val_acc'], label='验证准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC曲线
    axes[0, 2].plot(history['train_auc'], label='训练AUC')
    axes[0, 2].plot(history['val_auc'], label='验证AUC')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('AUC')
    axes[0, 2].set_title('AUC曲线')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 对比损失曲线
    axes[1, 0].plot(history['contrast_loss'], label='对比损失', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('对比损失')
    axes[1, 0].set_title('对比损失曲线')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 验证集预测分布
    if 'val_targets' in history and 'val_probabilities' in history:
        val_probs_class1 = [p[1] for p in history['val_probabilities']]
        axes[1, 1].hist(val_probs_class1, bins=30, alpha=0.7, color='blue')
        axes[1, 1].set_xlabel('耐药概率')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('验证集预测概率分布')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 留空一个位置用于其他可视化
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(targets, probabilities, save_path='roc_curve.png'):
    """绘制ROC曲线"""
    fpr, tpr, thresholds = roc_curve(targets, [p[1] for p in probabilities])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('接收者操作特征曲线 (ROC Curve)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ====================================================
# 8. 主程序 - 修改版本
# ====================================================
def main():
    """主函数：修复版增强训练和评估流程"""
    
    # 配置参数
    CONFIG = {
        'h5_directory': r"E:\fh\1-原始数据\TCGA\h5_files",      # H5文件文件夹路径
        'labels_csv': r"E:\fh\1-原始数据\TCGA\trainov2.csv",      # 标签CSV文件路径
        'input_dim': 1024,                       # 输入特征维度
        'hidden_dim': 256,                       # 隐藏层维度
        'n_classes': 2,                          # 类别数
        'max_instances': 2000,                   # 最大实例数
        'batch_size': 4,                         # 批大小
        'num_epochs': 50,                        # 训练轮数
        'contrastive_weight': 0.1,               # 对比损失权重
        'early_stopping_patience': 10,           # 早停耐心值
        'n_training_combinations': 10,          # 训练组合数量
        'fixed_test_size': 76,                   # 固定验证集大小
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'match_prefix_length': 12,               # 匹配前缀长度
        'random_seed': 42                        # 随机种子
    }
    
    print("=" * 80)
    print("卵巢癌耐药性预测 - 修复版ContrastiveMIL模型")
    print("=" * 80)
    print(f"使用设备: {CONFIG['device']}")
    print(f"配置参数: {CONFIG}")
    
    # 检查路径
    if not os.path.exists(CONFIG['h5_directory']):
        print(f"错误: H5文件夹不存在: {CONFIG['h5_directory']}")
        return
    
    if not os.path.exists(CONFIG['labels_csv']):
        print(f"错误: CSV文件不存在: {CONFIG['labels_csv']}")
        return
    
    try:
        # 步骤1: 准备数据（修复版函数，统一管理数据匹配和划分）
        print("\n" + "="*60)
        print("步骤1: 准备数据（修复标签匹配问题）")
        print("="*60)
        
        all_combinations = prepare_data_enhanced_fixed(
            h5_directory=CONFIG['h5_directory'],
            labels_csv=CONFIG['labels_csv'],
            fixed_test_size=CONFIG['fixed_test_size'],
            n_training_combinations=CONFIG['n_training_combinations'],
            max_instances=CONFIG['max_instances'],
            batch_size=CONFIG['batch_size'],
            match_prefix_length=CONFIG['match_prefix_length'],
            random_seed=CONFIG['random_seed']
        )
        
        if all_combinations is None or len(all_combinations) == 0:
            print("错误: 无法准备训练数据")
            return
        
        # 步骤2: 训练多种组合并评估
        print("\n" + "="*60)
        print("步骤2: 训练多种组合并评估")
        print("="*60)
        
        all_results, best_external_auc, best_combination_id, best_model = train_multiple_combinations(
            all_combinations, CONFIG
        )
        
        # 步骤3: 分析结果
        print("\n" + "="*60)
        print("步骤3: 分析所有组合结果")
        print("="*60)
        
        summary_df, best_result = analyze_all_results(all_results, best_combination_id, best_external_auc)
        
        # 步骤4: 保存最佳模型和最终结果
        print("\n" + "="*60)
        print("步骤4: 保存最终结果")
        print("="*60)
        
        if best_model is not None and best_result is not None:
            # 保存最佳模型
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'model_config': {
                    'input_dim': CONFIG['input_dim'],
                    'hidden_dim': CONFIG['hidden_dim'],
                    'n_classes': CONFIG['n_classes']
                },
                'best_result': best_result,
                'all_results_summary': summary_df.to_dict('records'),
                'config': CONFIG
            }, 'best_final_model.pth')
            
            # 保存最佳组合的详细预测结果
            best_predictions_df = pd.DataFrame({
                'wsi_id': best_result['external_test_wsi_ids'],
                'true_label': best_result['external_test_targets'],
                'predicted_label': best_result['external_test_predictions'],
                'probability_sensitive': [p[0] for p in best_result['external_test_probabilities']],
                'probability_resistant': [p[1] for p in best_result['external_test_probabilities']]
            })
            best_predictions_df.to_csv('best_combination_predictions.csv', index=False, encoding='utf-8-sig')
            
            # 绘制最佳组合的ROC曲线
            plot_roc_curve(
                best_result['external_test_targets'],
                best_result['external_test_probabilities'],
                'best_combination_roc_curve.png'
            )
            
            # 保存最终报告
            with open('final_report.txt', 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("卵巢癌耐药性预测 - 最终报告（修复版）\n")
                f.write("="*60 + "\n\n")
                f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"最佳组合ID: {best_combination_id}\n")
                f.write(f"最佳外部验证AUC: {best_external_auc:.4f}\n")
                f.write(f"最佳外部验证准确率: {best_result['external_test_accuracy']:.4f}\n")
                f.write(f"最佳外部验证F1分数: {best_result['external_test_f1']:.4f}\n\n")
                f.write("所有组合结果汇总:\n")
                f.write(summary_df.to_string(float_format=lambda x: f'{x:.4f}'))
                f.write("\n\n")
                f.write("最佳组合详细信息:\n")
                f.write(f"  训练集: {best_result['train_stats']['total']} 个样本 "
                       f"(敏感:{best_result['train_stats']['sensitive']} 耐药:{best_result['train_stats']['resistant']})\n")
                f.write(f"  验证集: {best_result['val_stats']['total']} 个样本 "
                       f"(敏感:{best_result['val_stats']['sensitive']} 耐药:{best_result['val_stats']['resistant']})\n")
                f.write(f"  外部验证集: {best_result['external_test_stats']['total']} 个样本 "
                       f"(敏感:{best_result['external_test_stats']['sensitive']} 耐药:{best_result['external_test_stats']['resistant']})\n")
            
            print(f"\n最佳组合已保存:")
            print(f"  最佳组合ID: {best_combination_id}")
            print(f"  最佳外部验证AUC: {best_external_auc:.4f}")
            print(f"  模型已保存为: best_final_model.pth")
            print(f"  预测结果已保存为: best_combination_predictions.csv")
            print(f"  所有组合汇总已保存为: all_combinations_summary.csv")
            print(f"  最终报告已保存为: final_report.txt")
        
        # 打印最终统计
        print("\n" + "="*80)
        print("训练完成!")
        print("="*80)
        print(f"总组合数: {len(all_results)}")
        print(f"外部验证AUC范围: {min([r['external_test_auc'] for r in all_results]):.4f} - "
              f"{max([r['external_test_auc'] for r in all_results]):.4f}")
        print(f"外部验证AUC平均值: {np.mean([r['external_test_auc'] for r in all_results]):.4f}")
        print(f"外部验证AUC标准差: {np.std([r['external_test_auc'] for r in all_results]):.4f}")
        
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
