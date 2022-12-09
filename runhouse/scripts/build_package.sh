pip install twine
python3 ../../setup.py sdist -d ../../dist
twine upload --repository-url http://3.83.200.25:8080 ../../dist/*

