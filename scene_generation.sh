# Exit immediately on error
set -e

# Define seeds and room types to iterate over
SEEDS=({0..9})

SCENE_ROOM_TYPES=("DiningRoom" "Bedroom" "Kitchen" "LivingRoom" "Bathroom")
VIEW_ROOM_TYPES=("dining-room" "bedroom" "kitchen" "living-room" "bathroom")

TERRAIN_SETTINGS="False"

# Generate visual data
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    for i in "${!SCENE_ROOM_TYPES[@]}"; do
        ROOM=${SCENE_ROOM_TYPES[$i]}
        TERRAIN=${TERRAIN_SETTINGS}
        SCENE_OUTPUT_FOLDER="outputs/indoors/${ROOM}_${SEED}"
        OBJECT_CENTRIC_VIEW_OUTPUT_FOLDER="object_centric_view_frame_outputs/${ROOM}/${ROOM}_${SEED}"
        VIEW_CENTRIC_VIEW_OUTPUT_FOLDER="view_centric_view_frame_outputs/${ROOM}/${ROOM}_${SEED}"

        echo "Running seed=$SEED room=$ROOM"

        python -m infinigen_examples.generate_indoors \
            --seed "$SEED" \
            --task coarse \
            --output_folder "$SCENE_OUTPUT_FOLDER" \
            -g fast_solve.gin singleroom.gin \
            -p compose_indoors.terrain_enabled=$TERRAIN \
               restrict_solving.restrict_parent_rooms="[\"$ROOM\"]"

        python object_centric_view_frame_generation.py \
            --config object_centric_view_frame_generation_config.json \
            --output_dir "$OBJECT_CENTRIC_VIEW_OUTPUT_FOLDER" \
            --room_type "${VIEW_ROOM_TYPES[$i]}" \
            --scene_path "$SCENE_OUTPUT_FOLDER/scene.blend"

        python view_centric_view_frame_generation.py \
            --config view_centric_view_frame_generation_config.json \
            --output_dir "$VIEW_CENTRIC_VIEW_OUTPUT_FOLDER" \
            --room_type "${VIEW_ROOM_TYPES[$i]}" \
            --scene_path "$SCENE_OUTPUT_FOLDER/scene.blend"
    done
done

# Generate object visibility list
for i in "${!SCENE_ROOM_TYPES[@]}"; do
    python -m generate_object_list \
    --input_dir "view_centric_view_frame_outputs" \
    --room_dir "${SCENE_ROOM_TYPES[$i]}" 
    
    python -m generate_object_list \
    --input_dir "object_centric_view_frame_outputs" \
    --room_dir "${SCENE_ROOM_TYPES[$i]}" 
done

# Clean visual data
python clean_visual_data.py \
    --dir_path object_centric_view_frame_outputs 
python clean_visual_data.py \
    --dir_path view_centric_view_frame_outputs 