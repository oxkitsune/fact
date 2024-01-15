for alpha in 0.1 0.3 0.5 0.8
do
    echo "[FACT21] Running experiment with alpha=${alpha} and beta=0 for baseAC"
    python train_fairAC_GNN_report.py  \
        --seed=42 \
        --epochs=3000 \
        --model=GCN \
        --dataset=pokec_z \
        --num-hidden=128 \
        --acc=0.65 \
        --roc=0.69 \
        --lambda1=1.0 \
        --lambda2=0 \
        --label_number=1000 \
        --feat_drop_rate="${alpha}" \
        --save_dir=./logs/alpha_experiments3/alpha_experiment_baseAC_pokec_z
done