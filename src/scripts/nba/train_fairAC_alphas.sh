for alpha in 0.1 0.3 0.5 0.8
do
    echo "[FACT21] Running experiment with alpha=${alpha} and default beta"
    python train_fairAC_GNN_report.py \
            --seed=42 \
            --epochs=3000 \
            --model=GCN \
            --dataset=nba \
            --num-hidden=128 \
            --acc=0.40 \
            --roc=0.42 \
            --lambda1=1.0 \
            --lambda2=0.7   \
            --label_number=1000  \
            --feat_drop_rate="${alpha}" \
            --save_dir=./logs/alpha_experiments2/alpha_experiment_fairAC_nba
done