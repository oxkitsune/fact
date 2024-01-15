alpha="0.3"

for beta in 0 0.2 0.4 0.7 0.8 
do
	echo "[FACT21] THIRD RUN Running experiment with beta=${beta} and alpha=${alpha}"
	python train_fairAC_GNN_report.py \
        	--seed=42 \
       		--epochs=3000 \
        	--model=GCN \
        	--dataset=pokec_z \
        	--num-hidden=128 \
        	--acc=0.65 \
        	--roc=0.69 \
					--lambda2="${beta}" \
		      --feat_drop_rate="${alpha}" \
        	--lambda1=1.0 \
        	--label_number=1000 \
        	--feat_drop_rate=0.3 \
	        --save_dir="./logs/beta_experiments3"
done

