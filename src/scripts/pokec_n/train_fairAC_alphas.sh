for alpha in 0.1 0.3 0.5 0.8
do
    echo "[FACT21] Running experiment with alpha=${alpha} and default beta"
    python train_fairAC_GNN_report.py  \
            --seed=42 \
            --epochs=3000 \
            --model=GCN \
            --dataset=pokec_n \
            --num-hidden=128 \
            --acc=0.66 \
            --roc=0.69 \
            --lambda1=1.0 \
            --lambda2=0.5 \
            --label_number=1000 \
            --feat_drop_rate="${alpha}" \
            --save_dir=./logs/alpha_experiments/alpha_experiment_fairAC_pokec_n

done