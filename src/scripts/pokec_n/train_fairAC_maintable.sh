for i in {1..3}
do
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
                --feat_drop_rate=0.3 \
                --save_dir=./logs/main_pokec_n 
done