# FMEH Report

## Aggregate Metrics
task,model_id,prompt_version,metric_parse_valid,metric_accuracy,metric_macro_f1_single,metric_precision,metric_recall,metric_f1,metric_exact_match,latency_sec
classification,mock_json,v1,1.0,0.0,0.0,,,,,3.0830269679427147e-06
extraction,mock_json,v1,1.0,,,0.0,0.0,0.0,0.0,8.330098353326321e-07
summarization,mock_json,v1,,,,,,,,0.0


## Parse Errors
parse_error,count
,1


## Top Failures
### classification | mock_json | v1 | pubmedqa:26348845:classification
Parse error: 
```text
{"label": "maybe", "rationale": "insufficient explicit evidence"}
```
### extraction | mock_json | v1 | synthetic:1:extraction
Parse error: 
```text
{"diseases": ["cancer"], "chemicals": ["aspirin"]}
```
### summarization | mock_json | v1 | pubmedqa:26348845:summarization
Parse error: 
```text

```