model_path=$1

# set of 7 vid
bash scripts/render_result.sh relephant0009  $model_path --catemodel --cnnpp 
bash scripts/render_result.sh relephant0010  $model_path --catemodel --cnnpp 
bash scripts/render_result.sh relephant0014  $model_path --catemodel --cnnpp 
bash scripts/render_result.sh relephant0038  $model_path --catemodel --cnnpp 
bash scripts/render_result.sh relephant0058  $model_path --catemodel --cnnpp 
bash scripts/render_result.sh relephant0075  $model_path --catemodel --cnnpp 
bash scripts/render_result.sh relephant-walk $model_path --catemodel --cnnpp 
