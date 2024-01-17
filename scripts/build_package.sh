# Delete dist directory if exists
rm -r dist
# Run from base directory of runhouse project
python3 -m build --sdist --wheel
#twine upload --repository testpypi dist/*
twine upload dist/*
