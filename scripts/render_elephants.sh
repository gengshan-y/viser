model_path=$1
nbones=$2

# set of 7 vid
bash scripts/render_result.sh elephant0009  $model_path $nbones --catemodel --cnnpp 
bash scripts/render_result.sh elephant0010  $model_path $nbones --catemodel --cnnpp 
bash scripts/render_result.sh elephant0014  $model_path $nbones --catemodel --cnnpp 
bash scripts/render_result.sh elephant0038  $model_path $nbones --catemodel --cnnpp 
bash scripts/render_result.sh elephant0058  $model_path $nbones --catemodel --cnnpp 
bash scripts/render_result.sh elephant0075  $model_path $nbones --catemodel --cnnpp 
bash scripts/render_result.sh elephant-walk $model_path $nbones --catemodel --cnnpp 
