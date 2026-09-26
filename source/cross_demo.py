from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from torch.utils.data import DataLoader, Subset

def run_cross_validation(all_dataset, labels, site_ids, cfg, logger):
    # 方案 A: 分层 5 折交叉验证 (兼顾诊断比例和站点比例)
    # 我们将 Site 和 Label 结合，确保每一折里各站点和各疾病状态分布均匀
    combined_groups = [f"{l}_{s}" for l, s in zip(labels, site_ids)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 方案 B: 留一站点验证 (更严谨，验证泛化性)
    # logo = LeaveOneGroupOut()
    # cv_split = logo.split(all_dataset, labels, groups=site_ids)

    all_fold_results = []

    for fold, (train_idx, test_val_idx) in enumerate(skf.split(all_dataset, combined_groups)):
        logger.info(f"#"*10 + f" FOLD {fold} " + f"#"*10)
        
        # 进一步将 test_val 划分为 Val 和 Test (例如 1:1)
        val_idx = test_val_idx[:len(test_val_idx)//2]
        test_idx = test_val_idx[len(test_val_idx)//2:]
        
        # 创建 DataLoader
        train_loader = DataLoader(Subset(all_dataset, train_idx), batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(all_dataset, val_idx), batch_size=cfg.batch_size)
        test_loader = DataLoader(Subset(all_dataset, test_idx), batch_size=cfg.batch_size)
        
        dataloaders = [train_loader, val_loader, test_loader]
        
        # --- 实例化你的模型、优化器和 Scheduler ---
        model = YourTSCNetModel(...).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = LRScheduler(optimizer, cfg)
        
        # --- 调用你现有的 Train 类 ---
        # 注意：需要修改你的 unique_id 以便区分不同 fold 的模型保存
        cfg.unique_id = f"experiment_fold_{fold}"
        trainer = Train(cfg, model, [optimizer], [scheduler], dataloaders, logger)
        
        # 开始训练
        trainer.train()
        
        # 记录该 Fold 的最终测试性能
        # 你可以修改 trainer.train() 返回最后的 test_result