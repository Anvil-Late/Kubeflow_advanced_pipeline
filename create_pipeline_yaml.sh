echo TEST
echo TEST2
cd -- "$(dirname "$BASH_SOURCE")"
dsl-compile --py pipeline/pipeline.py --output pipeline/pipeline.yaml
