gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

#CHUNKS=${#GPULIST[@]}
CHUNKS=3

CKPT="llava-v1.5-13b"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/shared/nas2/sstoica2/LLaVA/playground/data/eval/gqa"
#${GPULIST[$IDX]}
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m llava.eval.model_vqa_loader \
        --model-path /shared/nas2/sstoica2/MoL/checkpoints/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-token-router_learnable-gate_dino_patch_layer_corrected_k=4 \
        --question-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder /shared/nas2/sstoica2/LLaVA/playground/data/eval/gqa/images \
        --answers-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --attn_level "patch_layer" \
        --benchmark gqa \
        --use_router True \
        --mm_use_text_enc \
        --attn_level "patch_layer" \
        --num_layers_to_choose 4 \
        --force_reserve_layer True \
        --conv-mode vicuna_v1 & 

done

wait

output_file=/shared/nas2/sstoica2/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /shared/nas2/sstoica2/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python /shared/nas2/sstoica2/LLaVA/playground/data/eval/gqa/eval/eval.py --tier testdev_balanced
python /shared/nas/data/m1/sstoica2/PAPO_copy/debug_gpu.py --gpus 0,1,2,3 --memory 70