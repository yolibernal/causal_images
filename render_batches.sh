#!/bin/bash

set -e

CONF_DIRS=( "configs/2dcubes_equiscale_perspective_equi_child_noise" )
OUT_DIR="outputs"


INTERVENTIONS_DIR="same_mechanism"
MATERIAL_LIBRARY_PATH='./scenes/scene.blend'

TAGS=( "train" "test" "val" "dci_train" )
# TAGS=( "train" )

# Number of samples per intervention and tag
NUM_SAMPLES_PER_TAG=( 25000 5000 5000 10000 )
# NUM_SAMPLES_PER_TAG=( 10000 )
# NUM_SAMPLES_PER_TAG=( 100 )

SKIP_RENDER=false

SEQUENCE_LENGTH=5

BATCH_OFFSETS=( 0 )

# Intervention probabilities (_empty_intervention + each intervention (alphabetically))
# INTERVENTION_PROBABILITIES=( 0.1 0.1 0.1 0.25 0.1 0.1 0.25 )
INTERVENTION_PROBABILITIES=( -1 )

BATCHSIZE=300

for CONF_DIR in "${CONF_DIRS[@]}"; do
  EXP_NAME=$(basename ${CONF_DIR})
  if [ "$SKIP_RENDER" = true ]; then
    EXP_NAME="${EXP_NAME}_skip_render"
  fi

  # if sequence length is not 1, add it to the experiment name
  if [ $SEQUENCE_LENGTH -ne 1 ]; then
    EXP_NAME="${EXP_NAME}_${SEQUENCE_LENGTH}seq"
  fi

  echo "Rendering ${EXP_NAME}..."
  for i in "${!TAGS[@]}"; do
    TAG=${TAGS[$i]}
    NUM_SAMPLES=${NUM_SAMPLES_PER_TAG[$i]}

      # Calculate the number of batches
    NUM_BATCHES=$((NUM_SAMPLES / BATCHSIZE))

    # Check if there are any remaining items not evenly divisible by BATCHSIZE
    if [ $((NUM_SAMPLES % BATCHSIZE)) -ne 0 ]; then
        NUM_BATCHES=$((NUM_BATCHES + 1))
    fi

    batch_offset=${BATCH_OFFSETS[$i]}

    first_batch=$((batch_offset + 1))

    # Loop through each batch
    for ((batch = first_batch; batch <= NUM_BATCHES; batch++)); do
        # Calculate the range for the current batch
        START=$(( (batch - 1) * BATCHSIZE + 1 ))
        END=$(( batch * BATCHSIZE ))

        # Adjust the end of the last batch if it exceeds NUM_SAMPLES
        if [ $END -gt $NUM_SAMPLES ]; then
            END=$NUM_SAMPLES
        fi

        # Run your command with the current batch range
        echo "Running batch $batch: Items $START to $END"
        BATCH_NUM_SAMPLES=$((END - START + 1))

        command="blenderproc run main.py sample-weakly-supervised \
          --num_samples=${BATCH_NUM_SAMPLES} \
          --sampling_config ${CONF_DIR}/original/sampling.json \
          --fixed_config ${CONF_DIR}/original/config.json \
          --interventions_dir ${CONF_DIR}/same_mechanism \
          --output_dir $OUT_DIR/data/${EXP_NAME}/${TAG}/${batch} \
          --output_image_dir $OUT_DIR/images/${EXP_NAME}/${TAG}/${batch} \
          --sequence_length ${SEQUENCE_LENGTH} \
          --to_image --image_format PNG"

        # Need to pass intervention probabilities one by one (not as list)
        for j in "${!INTERVENTION_PROBABILITIES[@]}"; do
          command="${command} --intervention_probabilities ${INTERVENTION_PROBABILITIES[$j]}"
        done

        # skip rendering
        if [ "$SKIP_RENDER" = true ]; then
            command="${command} --skip_render"
        fi

        if [ -n "${MATERIAL_LIBRARY_PATH}" ]; then
            command="${command} --material_library_path ${MATERIAL_LIBRARY_PATH}"
        fi

        echo "${command}"

        # Execute the command
        eval "${command}"


        # Sleep or add any other desired operations between batches
        # sleep 1
    done
  done
done
