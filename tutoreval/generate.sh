#!/bin/bash

export OPENAI_API_KEY=""                                    #your api keys go here

model=${MOD:-"princeton-nlp/Llemma-7B-32K-MathMix"}         #model to evaluate
hf_chat_template=${CHATTEMPLATE:-false}
output_dir=${OUT:-"tutoreval/generations"}                  #directory to save outputs
batch_size=${BATCH:-1}                                      #batch size during generation
ddp_worldsize=${DDP:-1}                                    #data parallel 
closedbook=${CLOSEDBOOK:-false}                             #TutorEval-ClosedBook evaluation
bnb4bit=${QUANT:-true}                                      #4bit quantization




############## 
# generate 
header="python -m tutoreval.generate"
args=(
    --model ${model}
    --output_dir ${output_dir}
    --batch_size ${batch_size}
    --ddp_worldsize ${ddp_worldsize}
    $@
)

if [ $closedbook == true ]; then
    args+=(--closedbook)
fi

if [ $hf_chat_template == true ]; then
    args+=(--hf_chat_template)
fi

if [ $bnb4bit == true ]; then
    args+=(--bnb4bit)
fi

if [ ${ddp_worldsize} == 1 ]; then
    echo "${header} "${args[@]}""
    ${header} "${args[@]}"
else 
    for ((rank=0; rank<=ddp_worldsize-1; rank++)) ; do 
        ranked_args=(${args[@]} --ddp_rank $rank)
        echo "${header} "${ranked_args[@]}""
       export CUDA_VISIBLE_DEVICES=$rank ; ${header} "${ranked_args[@]}" &
    done
    wait
fi

# The current script handles data-parallel and model-sharding separately: setting ddp_worldsize=1 with multiple GPUs will shard the model using device_map="auto". 
# When ddp_worldsize is greater than 1, this script automatically assigns a single GPU to each data fragment. 
# If you want to use both data-parallel and model sharding, edit CUDA_VISIBLE_DEVICES to fit your situation


# merge files
header="python -m tutoreval.merge_generations"
merge_args=(
    --model ${model} 
    --dir ${output_dir} 
    --ddp_worldsize ${ddp_worldsize}
)

if [ $closedbook == true ]; then
    merge_args+=(--closedbook)
fi


echo "${header} "${merge_args[@]}""
${header} "${merge_args[@]}"