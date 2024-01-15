python train_fairGNN.py \
        --seed=42 \
        --epochs=3000 \
        --model=GCN \
        --dataset=nba \
        --num-hidden=128 \
        --acc=0.70 \
        --roc=0.72 \
        --alpha=10 \
        --beta=1 \
        --feat_drop_rate=0.3