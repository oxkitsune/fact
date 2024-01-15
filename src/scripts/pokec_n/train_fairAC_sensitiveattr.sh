        
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
        --save_dir=./logs/sens_attr/pokec_n/gender \
        --sens_attr_pokec=gender