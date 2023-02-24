python train_fairAC_GNN_report.py \
        --seed=42 \
        --epochs=3000 \
        --model=GCN \
        --dataset=pokec_z \
        --num-hidden=128 \
        --acc=0.65 \
        --roc=0.69 \
        --lambda1=1.0 \
        --lambda2=1.0 \
        --label_number=1000 \
        --feat_drop_rate=0.3
