# End-to-end QLoRA check on a generated dummy model: train a LoRA adapter with
# llama-finetune-qlora, run inference with it, merge it with llama-export-lora and
# run inference with the merged model.
#
# With more than one NGL value the training runs once per value, with the same seed,
# and the per-epoch losses must agree: this compares the GPU backward pass with the CPU.
#
# Run with cmake -P and these definitions:
#   FINETUNE     path to llama-finetune-qlora
#   COMPLETION   path to llama-completion
#   EXPORT_LORA  path to llama-export-lora
#   QUANTIZE     path to llama-quantize (used only with QUANT)
#   MODEL        dummy model GGUF (from test-llama-archs -o)
#   QUANT        optional quantization type applied to MODEL first, e.g. q4_0
#   WORK_DIR     scratch directory, recreated on every run
#   NGL          comma-separated -ngl values, e.g. 0,99 (0 is the CPU, 99 the default GPU backend)

foreach(var FINETUNE COMPLETION EXPORT_LORA MODEL WORK_DIR NGL)
    if (NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is not defined")
    endif()
endforeach()

if (NOT EXISTS "${MODEL}")
    message(FATAL_ERROR "model not found: ${MODEL} (run test-generate-models first)")
endif()

set(EPOCHS 3)
# largest per-epoch loss difference between backends, in units of 1e-4
set(MAX_BACKEND_LOSS_DIFF 50)

string(REPLACE "," ";" NGL_LIST "${NGL}")

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

if (DEFINED QUANT AND NOT QUANT STREQUAL "")
    if (NOT DEFINED QUANTIZE)
        message(FATAL_ERROR "QUANT requires QUANTIZE")
    endif()
    set(QUANT_MODEL "${WORK_DIR}/model-${QUANT}.gguf")
    run_step(quantize "${QUANTIZE}" "${MODEL}" "${QUANT_MODEL}" ${QUANT})
    set(MODEL "${QUANT_MODEL}")
endif()

# The same answer repeated across prompts, so the loss has something to learn.
set(TRAIN_FILE "${WORK_DIR}/train.jsonl")
file(WRITE "${TRAIN_FILE}" "")
foreach(i RANGE 1 8)
    file(APPEND "${TRAIN_FILE}"
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the code word number ${i}.\"},"
        "{\"role\":\"assistant\",\"content\":\"The code word is blue mango under bridge forty seven.\"}]}\n")
endforeach()

# 1. train once per NGL value
set(reference_losses "")
foreach(ngl IN LISTS NGL_LIST)
    set(adapter "${WORK_DIR}/adapter-ngl${ngl}.gguf")
    run_step(finetune-ngl${ngl} "${FINETUNE}"
        -m "${MODEL}" --train-file "${TRAIN_FILE}" --lora-out "${adapter}"
        --lora-rank 4 -c 64 -b 64 -ub 64 --epochs ${EPOCHS} -lr 1e-2
        -ngl ${ngl} -t 4 --seed 42)

    string(REGEX MATCHALL "epoch [0-9]+/[0-9]+: train_loss=[^ ]+" loss_lines "${STEP_OUTPUT}")
    list(LENGTH loss_lines n_loss)
    if (NOT n_loss EQUAL EPOCHS)
        message(FATAL_ERROR "ngl=${ngl}: expected ${EPOCHS} train_loss lines, found ${n_loss}")
    endif()

    # the losses are printed with 4 decimals: keep them as integers in units of 1e-4
    set(losses "")
    foreach(line IN LISTS loss_lines)
        string(REGEX REPLACE ".*train_loss=" "" loss "${line}")
        if (NOT loss MATCHES "^[0-9]+\\.[0-9][0-9][0-9][0-9]$")
            message(FATAL_ERROR "ngl=${ngl}: train_loss is not a finite number: ${line}")
        endif()
        string(REPLACE "." "" loss "${loss}")
        string(REGEX REPLACE "^0+([0-9])" "\\1" loss "${loss}")
        list(APPEND losses "${loss}")
    endforeach()
    message(STATUS "ngl=${ngl}: train_loss x1e4 per epoch: ${losses}")

    list(GET losses 0  first_loss)
    list(GET losses -1 last_loss)
    if (NOT last_loss LESS first_loss)
        message(FATAL_ERROR "ngl=${ngl}: train_loss did not decrease: ${losses}")
    endif()
    if (NOT EXISTS "${adapter}")
        message(FATAL_ERROR "ngl=${ngl}: adapter was not written: ${adapter}")
    endif()

    if (reference_losses STREQUAL "")
        set(reference_losses "${losses}")
        set(reference_ngl ${ngl})
        set(ADAPTER "${adapter}")
    else()
        foreach(epoch RANGE 0 ${EPOCHS})
            if (epoch EQUAL EPOCHS)
                break()
            endif()
            list(GET reference_losses ${epoch} a)
            list(GET losses           ${epoch} b)
            math(EXPR diff "${a} - ${b}")
            if (diff LESS 0)
                math(EXPR diff "-${diff}")
            endif()
            if (diff GREATER MAX_BACKEND_LOSS_DIFF)
                message(FATAL_ERROR "train_loss differs between ngl=${reference_ngl} (${reference_losses}) "
                                    "and ngl=${ngl} (${losses}) at epoch ${epoch}")
            endif()
        endforeach()
    endif()
endforeach()

list(GET NGL_LIST 0 NGL_INFER)

# 2. inference with the adapter
run_step(completion-lora "${COMPLETION}"
    -m "${MODEL}" --lora "${ADAPTER}" -p "hello" -n 4 -no-cnv --temp 0 -ngl ${NGL_INFER})

# 3. merge the adapter into the model
set(MERGED "${WORK_DIR}/merged.gguf")
run_step(export-lora "${EXPORT_LORA}" -m "${MODEL}" --lora "${ADAPTER}" -o "${MERGED}")
if (NOT STEP_OUTPUT MATCHES "merged [1-9][0-9]* tensors with lora adapters")
    message(FATAL_ERROR "export-lora merged no tensors")
endif()

# 4. inference with the merged model
run_step(completion-merged "${COMPLETION}"
    -m "${MERGED}" -p "hello" -n 4 -no-cnv --temp 0 -ngl ${NGL_INFER})

file(REMOVE_RECURSE "${WORK_DIR}")
