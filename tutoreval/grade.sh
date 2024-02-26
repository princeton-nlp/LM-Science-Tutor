#!/bin/bash


export OPENAI_API_KEY=""                                   # your api key goes here

model=${MOD:-"princeton-nlp/Llemma-7B-32K-MathMix"}
closedbook=${CLOSEDBOOK:-false}
grader=${GRADER:-"gpt-4-1106-preview"}
dir=${DIR:-"generations"}
ddp_worldsize=${DDP:-1}                                    #data parallel uses the splits created during generation. Split the generations files if you want to grade faster.


header="python -m tutoreval.grade"
args=(
    --model ${model}
    --grader ${grader}
    --dir ${dir}
    --ddp_worldsize ${ddp_worldsize}
    $@
)

if [ $closedbook == true ]; then
    args+=(--closedbook)
fi

if [ ${ddp_worldsize} == 1 ]; then
    echo "${header} "${args[@]}""
    ${header} "${args[@]}"
else 
    for ((rank=0; rank<=ddp_worldsize-1; rank++)) ; do 
        ranked_args=(${args[@]} --ddp_rank $rank)
        echo "${header} "${ranked_args[@]}""
        ${header} "${ranked_args[@]}" &
    done
    wait
    # merge graded files
    header="python -m tutoreval.merge_generations"
    merge_args=(
        --model ${model} 
        --output_dir ${output_dir} 
        --ddp_worldsize ${ddp_worldsize}
    )

    if [ $closedbook == true ]; then
        merge_args+=(--closedbook)
    fi

    echo "${header} "${merge_args[@]}""
    ${header} "${merge_args[@]}"  
fi
