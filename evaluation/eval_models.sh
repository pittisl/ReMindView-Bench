FILES=("balanced_1000_dense_object_centric_object_object_qa.csv", "balanced_1000_dense_object_centric_view_object_qa.csv", "balanced_1000_dense_object_centric_view_view_qa.csv", "balanced_1000_processed_sparse_object_centric_object_object_qa.csv", "balanced_1000_processed_sparse_object_centric_view_object_qa.csv", "balanced_1000_processed_sparse_object_centric_view_view_qa.csv", "balanced_1000_dense_view_centric_object_object_qa.csv", "balanced_1000_dense_view_centric_view_object_qa.csv", "balanced_1000_processed_sparse_view_centric_object_object_qa.csv", "balanced_1000_processed_sparse_view_centric_view_object_qa.csv")

# FILES=("balanced_1000_dense_object_centric_object_object_qa.csv")


for CSV_FILE in "${FILES[@]}"; do
#     python eval_qwen_vl.py \
#         --csv "$CSV_FILE" \               
#         --dtype bf16 \
#         --model_name qwen_vl-3b \
#         --model Qwen/Qwen2.5-VL-3B-Instruct \
#         --device cuda:0 \
#         --compile
#         --max-new-tokens 600

#     python eval_qwen_vl.py \
#         --csv "$CSV_FILE" \
#         --dtype bf16 \
#         --model Qwen/Qwen2.5-VL-7B-Instruct \
#         --model_name qwen_vl-7b \
#         --device cuda:0 \
#         --compile \
#         --max-new-tokens 600

#      python eval_qwen_vl.py \
#         --csv "$CSV_FILE" \
#         --model Qwen/Qwen2.5-VL-32B-Instruct \
#         --model_name qwen_vl-32b \
#         --dtype bf16 \
#         --device cuda:0 \
#         --compile \
#         --max-new-tokens 600

#      python eval_internvl35.py \
#         --csv "$CSV_FILE" \
#         --model OpenGVLab/InternVL3_5-8B-HF \
#         --model_name intervl3_5-8b \
#         --device cuda:0 \
#         --dtype bf16 \
#         --max-new-tokens 300

#      python eval_internvl35.py \
#         --csv "$CSV_FILE" \
#         --model OpenGVLab/InternVL3_5-2B-HF \
#         --model_name intervl3_5-2b \
#         --device cuda:0 \
#         --dtype bf16 \
#         --max-new-tokens 300

#      python eval_internvl35.py \
#         --csv "$CSV_FILE" \
#         --model OpenGVLab/InternVL3_5-38B-HF \
#         --model_name intervl3_5-38b \
#         --device cuda:0 \
#         --dtype bf16 \
#         --max-new-tokens 300

     python eval_llava_onevision.py \
        --csv "$CSV_FILE" \
        --model lmms-lab/llava-onevision-qwen2-0.5b-ov \
        --model_name llava-onevision-qwen2-0.5b-ov \
        --device cuda:0 \
        --dtype bf16 \
        --max-new-tokens 300


done