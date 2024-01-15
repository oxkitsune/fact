for alpha in 0.1 0.3 0.5 0.8
do
    echo "[FACT21] Running experiment with alpha=${alpha} and default beta"
    python train_fairGNN.py \
        --seed=42 \
        --epochs=3000 \
        --model=GCN \
        --dataset=pokec_z \
        --num-hidden=128 \
        --acc=0.65 \
        --roc=0.69 \
        --alpha=10 \
        --beta=1 \
        --label_number=1000 \
        --save_dir=./logs/alpha_experiments_fairGNN_3/pokec_z/ \
        --feat_drop_rate=${alpha}
done