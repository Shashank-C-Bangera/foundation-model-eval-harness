# FMEH Report

## Aggregate Metrics
task,model_id,prompt_version,metric_parse_valid,metric_accuracy,metric_rougeL,metric_bertscore_f1,metric_pred_length,metric_compression_ratio,metric_unsupported_claim_proxy,metric_precision,metric_recall,metric_f1,metric_exact_match,latency_sec
classification,flan_t5_small,v2,1.0,0.51,,,,,,,,,,0.12880910657870118
classification,t5_small,v2,0.18,0.11,,,,,,,,,,0.6335823941417038
classification,tiny_gpt2,v2,0.0,0.04,,,,,,,,,,0.037458177949301896
extraction,flan_t5_small,v2,0.66,,,,,,,0.1,0.05,0.062380952380952384,0.0,0.21951614252175206
extraction,t5_small,v2,0.38,,,,,,,0.03,0.021666666666666664,0.024666666666666667,0.0,0.534697287514573
extraction,tiny_gpt2,v2,0.0,,,,,,,0.0,0.0,0.0,0.0,0.03477426998084411
summarization,flan_t5_small,v2,0.94,,0.1151361976177026,0.7265718477837583,12.34,0.050531777337211946,0.8,,,,,0.3398456090968102
summarization,t5_small,v2,1.0,,0.07323803593675318,0.6917600893974304,25.06,0.11082065977375154,1.0,,,,,0.529609104100382
summarization,tiny_gpt2,v2,1.0,,0.00037383177570093456,0.5892678844928741,64.0,0.28285857447487867,1.0,,,,,0.03708172337850556


## Parse Errors
parse_error,count
Expecting value: line 1 column 1 (char 0),272
empty output,3
Extra data: line 1 column 2 (char 1),2
Extra data: line 1 column 16 (char 15),1
Extra data: line 1 column 50 (char 49),1
Extra data: line 1 column 160 (char 159),1
Extra data: line 1 column 5 (char 4),1
Extra data: line 1 column 143 (char 142),1
Extra data: line 1 column 3 (char 2),1


## Top Failures
### classification | t5_small | v2 | pubmedqa:26348845:classification
Parse error: Expecting value: line 1 column 1 (char 0)
```text
Pap smears with glandular cell abnormalities: Are they detected by rapid prescreening? Abstract: ["BACKGROUND", "METHODS", "RESULTS"] ["Adenocarcinoma on histologic follow-up
```
### classification | t5_small | v2 | pubmedqa:26163474:classification
Parse error: Expecting value: line 1 column 1 (char 0)
```text
, p0.002, CI 95 %). Sublingual varices indicate hypertension with a positive predictive value of 0.5 and a negative predictive value of 0.80. ["BACKGROUND", "METHODS", "BACKGROUND",
```
### classification | t5_small | v2 | pubmedqa:24374414:classification
Parse error: Expecting value: line 1 column 1 (char 0)
```text
: Does health information exchange reduce redundant imaging? Abstract: ["BACKGROUND", "OBJECT", "OBJECT", "OBJECT", "BACKGROUND", "OBJECT", "OBJECT",
```
### classification | t5_small | v2 | pubmedqa:18507507:classification
Parse error: Expecting value: line 1 column 1 (char 0)
```text
:: - Return exactly one JSON object. - Do not echo the prompt or schema. - Do not echo the prompt or schema. "BACKGROUND", "OBJECTIVE", "SUMMARY"] ["Chronic Disease",
```
### classification | t5_small | v2 | pubmedqa:23495128:classification
Parse error: Expecting value: line 1 column 1 (char 0)
```text
: The colour of pain: can patients use colour to describe osteoarthritis pain? Abstract: ["The aim of the present study was to explore patients" views on the acceptability and feasibility of using colour to describe pain.
```
### classification | t5_small | v2 | pubmedqa:12163782:classification
Parse error: Expecting value: line 1 column 1 (char 0)
```text
compared to the ARDS patients.: migratory activity after major trauma was a factor in the etiology of acute respiratory distress syndrome? Abstract: ["Neutrophil infiltration of the lung is characteristic of early posttraumatic acute respiratory distress syndrome (ARDS)
```
### classification | t5_small | v2 | pubmedqa:9582182:classification
Parse error: Expecting value: line 1 column 1 (char 0)
```text
: Does the SCL 90-R obsessive dimension identify cognitive impairments? Abstract: ["To investigate the relevance of the Symptom Checklist 90-R Obsessive-Compulsive subscale to cognition in individuals with brain tumor"
```
### classification | t5_small | v2 | pubmedqa:25588461:classification
Parse error: Expecting value: line 1 column 1 (char 0)
```text
: "Do not output prose outside JSON."] ["PURPOSE", "METHODS", "RESULT"] ["Adult", "Aged", "Consciousness Disorders", "Per
```
### classification | t5_small | v2 | pubmedqa:23453079:classification
Parse error: Expecting value: line 1 column 1 (char 0)
```text
: Does prostate morphology affect outcomes after holmium laser enucleation? Abstract: [To determine whether prostate morphology or technique used has any effect on outcomes after holmium laser enucleation?"
```
### classification | t5_small | v2 | pubmedqa:21084567:classification
Parse error: Expecting value: line 1 column 1 (char 0)
```text
:: Are stroke patients" reports of home blood pressure readings reliable? Abstract: ["Home blood pressure monitoring is gaining increasing popularity among patients" Abstract: ["To assess the reliability of home BP monitoring in hypertensive patients who had suffered a recent stroke or transient ischa
```