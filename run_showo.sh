# #!/bin/bash
# GPUS="0,1,2,3"
# # Generate commands for all combinations of parameters
# for DECODE in cfg vanilla; do
#     for TEACHER_FORCE in true false; do
#         if [ "$TEACHER_FORCE" == "true" ]; then
#             UPTO=0.2
#         else
#             UPTO=1.0
#         fi
#         for NONMYOPIC in true false; do
#             for IDX in 100 150 200 250 300; do
#                 for CFG_SCALE in 5.0 20.0; do
#                     # Print the command
#                     echo "python3 gt_test.py -m decode=cfg teacher_force=${TEACHER_FORCE} teacher_force_upto=${UPTO} nonmyopic=${NONMYOPIC} prompt_idx=${IDX} cfg_scale=${CFG_SCALE}"
#                 done
#             done
#         done
#     done
# done | simple_gpu_scheduler --gpus ${GPUS}
