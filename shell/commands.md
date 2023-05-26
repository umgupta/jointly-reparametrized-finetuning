To change the seed or any other hyperparameter, see the corresponding shell script.


## GLUE Experiments (Table 2 and 3) 
### SLaSh
`bash shell/roberta-slash/run_roberta_task.sh <z_init> <w_init> <lr> <epochs> <d or z_size> <task> <roberta-base/roberta-large>`

### JR-WARP
`bash shell/roberta-jr-warp/run_roberta_task.sh <z_init> <w_init> <lr> <epochs> <d or z_size> <task> <roberta-base/roberta-large>`

### Adapter
`bash shell/roberta-adapter/run_roberta_task.sh <lr> <epochs> <task> <roberta-base/roberta-large>`

### LoRA
`bash shell/roberta-lora/run_roberta_task.sh <lr> <epochs> <task> <roberta-base/roberta-large>`

### BitFit
`bash shell/roberta-bitfit/run_roberta_task.sh <lr> <epochs> <task> <roberta-base/roberta-large>`

## NER Experiments (Table 4)

### Full-finetune
`bash shell/ner/ner.sh bert-base-cased <lr> <epochs> <batch size>`

### SLaSh 
`bash shell/ner/ner_slash.sh bert-base-cased <lr> <epochs> <batch size> <z_size/d> <z/w init>`

### JR-WARP 
`bash shell/ner/ner_jr_warp.sh bert-base-cased <lr> <epochs> <batch size> <z_size/d> <z/w init>`

### Linear/Adapter/LoRA/BitFit
`bash shell/ner/ner_<method>.sh bert-base-cased <lr> <epochs> <batch size>`

## Profiling (Table 5)
See `inference_time.sh` and `training_time.sh` for reproducing results for Roberta-base. Similar commands will work for Roberta-large.

## DP Training Experiments (Table 6)

- For private training, compute noise multiplier or noise std using `src/compute_noise.py`
- DP training often requires bigger batch size, and so we use acc. step parameter (`--gradient_accumulation_steps`) to simulate larger batch sizes. One iteration batch size is 16, so use acc. step 128 to get effective batch size of 2048.

`bash shell/roberta-jr-warp-private.sh <noise std> <per-sample-grad-norm> <lr> <acc. steps> <z_size> <z/w init> <task> <model>`