#!/bin/bash
# Fix mujoco_playground XMLs to use local assets instead of mujoco_menagerie

set -e

cd ~/newevoll

echo "=== Fixing MuJoCo XML paths ==="

# Copy menagerie assets into playground
echo "Copying Go1 assets into mujoco_playground..."
mkdir -p mujoco_playground/mujoco_playground/_src/locomotion/go1/assets
cp -r ~/mujoco_menagerie/unitree_go1/assets/* \
   mujoco_playground/mujoco_playground/_src/locomotion/go1/assets/

# Fix XML meshdir to point to local assets folder
cd mujoco_playground/mujoco_playground/_src/locomotion/go1

echo "Updating XML meshdir paths..."
for xml in *.xml; do
    if [ -f "$xml" ]; then
        # Change meshdir from "../../../../../../mujoco_menagerie/unitree_go1/assets" to "./assets"
        sed -i 's|meshdir="../../../../../../mujoco_menagerie/unitree_go1/assets"|meshdir="./assets"|g' "$xml"
        echo "  Fixed: $xml"
    fi
done

echo ""
echo "✓ XMLs fixed successfully!"
echo ""
echo "Verifying fix:"
grep "meshdir" *.xml | head -5

cd ~/newevoll
