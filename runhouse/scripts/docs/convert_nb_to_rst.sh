# Note: please run this script from inside runhouse git directory

input_path=$1
output_path=${input_path/.ipynb/.rst}
dest_path=${output_path/notebooks/tutorials}

echo $output_path
echo $dest_path

jupyter nbconvert --to rst $1
mv $output_path $dest_path
python runhouse/scripts/docs/update_rst.py --files $dest_path --link-colab
