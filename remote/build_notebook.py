"""Build upload code only. Never uploads or downloads anything.

Remote execution on this competition is pending sponsor clarification of the
no-LLM-copying rule; do not push merely because the artifact is buildable.
"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'evaluate.py').read_text()
source_sha256 = hashlib.sha256(source.encode()).hexdigest()
runner = '''
# Private portfolio audit; NOT a course submission. Permission must be resolved before use.
# Only numeric classes and aggregate output; no head(), comments, predictions or model artifacts.
from pathlib import Path
import contextlib
import hashlib
import io
import json

namespace = {"__name__": "remote_baseline"}
exec(SOURCE, namespace)
path = Path("/kaggle/input/comment-category-prediction-challenge/train.csv")
if not path.is_file():
    raise RuntimeError("Expected remote competition training file is unavailable")
# This code runs in Kaggle; dataset bytes never transit to the Mac.
digest = hashlib.sha256()
with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
# Strict no-row-output discipline: capture incidental fit output and emit aggregates only.
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    df = namespace["pd"].read_csv(path, usecols=["comment", "post_id", "label"])
    metrics = namespace["evaluate"](df, "/kaggle/working/aggregate-baseline", digest.hexdigest(),
        "Private remote evaluation of original Kaggle competition 127742, bundle 15295088; terms must be resolved", aggregate_only=True)
metrics["evaluator_source_sha256"] = SOURCE_SHA256
Path("/kaggle/working/aggregate-baseline/metrics.json").write_text(json.dumps(metrics, indent=2) + "\\n")
print("AGGREGATE_METRICS_JSON=" + json.dumps(metrics, sort_keys=True))
'''
code = 'SOURCE = ' + repr(source) + '\nSOURCE_SHA256 = ' + repr(source_sha256) + '\n' + runner
(root / 'remote' / 'private-baseline.py').write_text(code)
metadata = {
    'id': 'jbanmol9/comment-evaluation-private-audit',
    'title': 'Comment Evaluation Private Audit',
    'code_file': 'private-baseline.py', 'language': 'python', 'kernel_type': 'script',
    'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
    'competition_sources': ['comment-category-prediction-challenge'],
    'dataset_sources': [], 'kernel_sources': [], 'model_sources': [],
}
(root / 'remote' / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('Built private code-only remote artifact; execution permission remains unresolved.')
