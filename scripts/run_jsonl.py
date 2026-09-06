import argparse, json
from umirage.adapters import MockMultiImageAdapter, TransformersVLMAdapter
from umirage.experiment import run

p=argparse.ArgumentParser(description="Run U-MIRAGE on a canonical JSONL dataset")
p.add_argument("--input",required=True)
p.add_argument("--output",required=True)
p.add_argument("--model-id")
p.add_argument("--revision")
p.add_argument("--mock",action="store_true")
a=p.parse_args()
if not a.mock and not a.model_id: p.error("--model-id is required unless --mock is used")
adapter=MockMultiImageAdapter() if a.mock else TransformersVLMAdapter(a.model_id,revision=a.revision)
print(json.dumps(run(a.input,adapter,a.output,{"model_id":a.model_id,"revision":a.revision,"mock":a.mock}),indent=2))
