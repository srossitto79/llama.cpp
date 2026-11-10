# End-to-end MoE pruning check on a real MoE GGUF: analyze (soft-pruned perplexity per ratio),
# hard-prune one profile with soft/hard validation, then run inference on the pruned model.
#
# Run with cmake -P and these definitions:
#   PRUNE        path to llama-prune
#   COMPLETION   path to llama-completion
#   MODEL        MoE model GGUF
#   WORK_DIR     scratch directory, recreated on every run
#   NGL          -ngl value for every step

foreach(var PRUNE COMPLETION MODEL WORK_DIR NGL)
    if (NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is not defined")
    endif()
endforeach()

if (NOT EXISTS "${MODEL}")
    message(FATAL_ERROR "model not found: ${MODEL}")
endif()

# hard and soft pruning remove the same experts, their perplexities differ only by numerics
set(MAX_SOFT_HARD_DELTA_PERCENT 2)

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

function(run_step name)
    execute_process(
        COMMAND ${ARGN}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE  out
    )
    file(WRITE "${WORK_DIR}/${name}.log" "${out}")
    if (NOT rc EQUAL 0)
        message("${out}")
        message(FATAL_ERROR "${name} failed with exit code ${rc}, log: ${WORK_DIR}/${name}.log")
    endif()
    set(STEP_OUTPUT "${out}" PARENT_SCOPE)
endfunction()

# calibration conversations with varied topics, so the routing spreads over the experts
set(DATASET "${WORK_DIR}/calibration.jsonl")
file(WRITE "${DATASET}" "")
foreach(pair
        "What is the boiling point of water at sea level?|Water boils at 100 degrees Celsius at sea level."
        "Write a haiku about autumn.|Red leaves drift and fall, the cold wind hums through bare trees, the year breathes out slow."
        "How do I reverse a list in Python?|Use my_list.reverse() to reverse it in place, or my_list[::-1] for a reversed copy."
        "Translate 'good morning' to Italian.|Good morning in Italian is buongiorno."
        "What is 17 times 23?|17 times 23 is 391."
        "Name three planets of the solar system.|Mercury, Venus and Mars are three planets of the solar system."
        "Explain photosynthesis briefly.|Plants use sunlight to turn carbon dioxide and water into glucose and oxygen."
        "Give me a synonym for happy.|A synonym for happy is cheerful."
        "What does an HTTP 404 status mean?|It means the server could not find the requested resource."
        "Who wrote Romeo and Juliet?|William Shakespeare wrote Romeo and Juliet.")
    string(REPLACE "|" ";" pair "${pair}")
    list(GET pair 0 question)
    list(GET pair 1 answer)
    file(APPEND "${DATASET}"
        "{\"messages\":[{\"role\":\"user\",\"content\":\"${question}\"},{\"role\":\"assistant\",\"content\":\"${answer}\"}]}\n")
endforeach()

set(COMMON_ARGS --ctx-size 512 -ngl ${NGL} --ppl-mask all)

# 1. importance statistics and soft-pruned perplexity; llama-prune fails if a disabled expert is routed
run_step(analyze "${PRUNE}" analyze -m "${MODEL}" --dataset "${DATASET}" --ratios 0.125,0.25
    --max-layer-ratio 0.5 --output-dir "${WORK_DIR}/analysis" ${COMMON_ARGS})
foreach(profile profile-013.json profile-025.json analysis.json importance-cache.json)
    if (NOT EXISTS "${WORK_DIR}/analysis/${profile}")
        message(FATAL_ERROR "analyze did not write ${profile}")
    endif()
endforeach()

# 2. hard-prune 25% of the experts, compared with the soft profile on the same data
set(PRUNED "${WORK_DIR}/pruned.gguf")
run_step(hard "${PRUNE}" hard -m "${MODEL}" --profile "${WORK_DIR}/analysis/profile-025.json" --output "${PRUNED}"
    --dataset "${DATASET}" --validate --max-ppl-delta-percent ${MAX_SOFT_HARD_DELTA_PERCENT} ${COMMON_ARGS})
if (NOT STEP_OUTPUT MATCHES "hard-pruned model validated")
    message(FATAL_ERROR "hard pruning did not validate")
endif()
file(SIZE "${MODEL}" model_size)
file(SIZE "${PRUNED}" pruned_size)
if (NOT pruned_size LESS model_size)
    message(FATAL_ERROR "the pruned model is not smaller: ${pruned_size} >= ${model_size} bytes")
endif()
message(STATUS "pruned model: ${pruned_size} bytes, source ${model_size} bytes")

# 3. inference with the pruned model
run_step(completion-pruned "${COMPLETION}" -m "${PRUNED}" -p "The capital of France is" -n 8 -no-cnv --temp 0 -ngl ${NGL})

file(REMOVE_RECURSE "${WORK_DIR}")
