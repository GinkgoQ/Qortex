
##### running 0_import_config #####

=== PROJECT 0: installed package and runtime configuration ===

=== Observed output ===
qortex version         : 0.1.0
configured cache       : /tmp/tmp4vjq3zft/cache
download workers       : 3
metadata timeout       : 7.5
dataset facade id      : ds000001
public search callable : True
RESULT: project 0 passed

##### running 1_manifest_models #####

=== PROJECT 1: real OpenNeuro manifest and semantic recording graph ===
dataset            : ds000001
snapshot           : 1.0.0
doi                : 10.18112/openneuro.ds000001.v1.0.0
files              : 136
subjects           : 16
sessions           : 0
tasks              : balloonanalogrisktask
modalities         : behavior, fmri, mri
logical recordings : 80

=== First real logical recordings ===
primary                                                          | modality | subject | task                  | events | channels | bytes   
-----------------------------------------------------------------+----------+---------+-----------------------+--------+----------+---------
sub-16/func/sub-16_task-balloonanalogrisktask_run-01_bold.nii.gz | fmri     | 16      | balloonanalogrisktask | True   | False    | 50498319
sub-16/func/sub-16_task-balloonanalogrisktask_run-02_bold.nii.gz | fmri     | 16      | balloonanalogrisktask | True   | False    | 50496033
sub-16/func/sub-16_task-balloonanalogrisktask_run-03_bold.nii.gz | fmri     | 16      | balloonanalogrisktask | True   | False    | 50498545
sub-16/anat/sub-16_T1w.nii.gz                                    | mri      | 16      | None                  | False  | False    | 5801162 
sub-16/anat/sub-16_inplaneT2.nii.gz                              | mri      | 16      | None                  | False  | False    | 701074  
sub-15/func/sub-15_task-balloonanalogrisktask_run-01_bold.nii.gz | fmri     | 15      | balloonanalogrisktask | True   | False    | 46573956
sub-15/func/sub-15_task-balloonanalogrisktask_run-02_bold.nii.gz | fmri     | 15      | balloonanalogrisktask | True   | False    | 46585785
sub-15/func/sub-15_task-balloonanalogrisktask_run-03_bold.nii.gz | fmri     | 15      | balloonanalogrisktask | True   | False    | 46613596
sub-15/anat/sub-15_T1w.nii.gz                                    | mri      | 15      | None                  | False  | False    | 5511442 
sub-15/anat/sub-15_inplaneT2.nii.gz                              | mri      | 15      | None                  | False  | False    | 665711  
sub-14/func/sub-14_task-balloonanalogrisktask_run-01_bold.nii.gz | fmri     | 14      | balloonanalogrisktask | True   | False    | 47613439
sub-14/func/sub-14_task-balloonanalogrisktask_run-02_bold.nii.gz | fmri     | 14      | balloonanalogrisktask | True   | False    | 47695653
RESULT: real manifest project passed

##### running 2_selection_planning #####

=== PROJECT 2: real primary-file selection with companion closure ===
selected primary : sub-16/func/sub-16_task-balloonanalogrisktask_run-01_bold.nii.gz
selected files   : 7
estimated bytes  : 50498319
warnings         : 0
Dataset : ds000001  (snapshot 1.0.0)
Target  : /tmp/tmp40_yz2qa/ds000001
Files   : 7
Size    : 0.05 GB (estimated)

=== Real selected files and reasons ===
path                                                             | size     | reason                                                                     
-----------------------------------------------------------------+----------+----------------------------------------------------------------------------
CHANGES                                                          | 286      | essential BIDS/OpenNeuro metadata; required companion or inherited metadata
README                                                           | 1175     | essential BIDS/OpenNeuro metadata; required companion or inherited metadata
dataset_description.json                                         | 615      | essential BIDS/OpenNeuro metadata; required companion or inherited metadata
participants.tsv                                                 | 216      | essential BIDS/OpenNeuro metadata; required companion or inherited metadata
sub-16/func/sub-16_task-balloonanalogrisktask_run-01_bold.nii.gz | 50489238 | matched selection filters                                                  
sub-16/func/sub-16_task-balloonanalogrisktask_run-01_events.tsv  | 6716     | required companion or inherited metadata                                   
task-balloonanalogrisktask_bold.json                             | 73       | required companion or inherited metadata                                   
RESULT: real selection project passed

##### running 3_preview_local #####

=== PROJECT 3: real remote metadata preview without full download ===
dataset           : ds000001
snapshot          : 1.0.0
table             : participants.tsv
table source      : remote
table bytes read  : 216
table columns     : participant_id, sex, age
description bytes : 615

=== participants.tsv first rows ===
participant_id | sex | age
---------------+-----+----
sub-01         | F   | 26 
sub-02         | M   | 24 
sub-03         | F   | 27 
sub-04         | F   | 20 
sub-05         | M,  | 22 
dataset_description.json preview:
{
  "Authors": [
    "Tom Schonberg",
    "Christopher Trepel",
    "Craig Fox",
    "Russell A. Poldrack"
  ],
  "BIDSVersion": "1.0.0",
  "DatasetDOI": "10.18112/openneuro.ds000001.v1.0.0",
  "License": "CC0",
  "Name": "Balloon Analog Risk-taking Task",
  "ReferencesAndLinks": [
    "Schonberg TS, Fox CR, Mumford JA, Congdon E, Trepel C, Poldrack RA (2012). Decreasing ventromedial prefrontal cortex activity during sequential risk-taking: An fMRI investigation of the Balloon Analogue Risk Task. Frontiers in Decision Neuroscience, 6:80 doi: 10.3389/fnins.2012.00080"
  ]
}
RESULT: real preview project passed

##### running 4_download_specific_parts_project #####

=== PROJECT 4: real metadata-only and exact-path download plans ===
dataset                  : ds000001
metadata-only files      : 53
metadata-only bytes      : 421311
exact-path closure files : 7
exact-path bytes         : 50498319

=== Real metadata-only plan ===
path                                                            | size | extension
----------------------------------------------------------------+------+----------
CHANGES                                                         | 286  |          
README                                                          | 1175 |          
dataset_description.json                                        | 615  | .json    
participants.tsv                                                | 216  | .tsv     
task-balloonanalogrisktask_bold.json                            | 73   | .json    
sub-16/func/sub-16_task-balloonanalogrisktask_run-01_events.tsv | 6716 | .tsv     
sub-16/func/sub-16_task-balloonanalogrisktask_run-02_events.tsv | 7587 | .tsv     
sub-16/func/sub-16_task-balloonanalogrisktask_run-03_events.tsv | 8568 | .tsv     
sub-15/func/sub-15_task-balloonanalogrisktask_run-01_events.tsv | 8093 | .tsv     
sub-15/func/sub-15_task-balloonanalogrisktask_run-02_events.tsv | 6987 | .tsv     
sub-15/func/sub-15_task-balloonanalogrisktask_run-03_events.tsv | 7184 | .tsv     
sub-14/func/sub-14_task-balloonanalogrisktask_run-01_events.tsv | 9187 | .tsv     
sub-14/func/sub-14_task-balloonanalogrisktask_run-02_events.tsv | 9290 | .tsv     
sub-14/func/sub-14_task-balloonanalogrisktask_run-03_events.tsv | 8672 | .tsv     
sub-13/func/sub-13_task-balloonanalogrisktask_run-01_events.tsv | 8111 | .tsv     
sub-13/func/sub-13_task-balloonanalogrisktask_run-02_events.tsv | 9485 | .tsv     

=== Real exact path plus companions ===
path                                                             | size    
-----------------------------------------------------------------+---------
CHANGES                                                          | 286     
README                                                           | 1175    
dataset_description.json                                         | 615     
participants.tsv                                                 | 216     
sub-16/func/sub-16_task-balloonanalogrisktask_run-01_bold.nii.gz | 50489238
sub-16/func/sub-16_task-balloonanalogrisktask_run-01_events.tsv  | 6716    
task-balloonanalogrisktask_bold.json                             | 73      
RESULT: real specific-download project passed

##### running 5_eda_events #####
📥 Downloading ds000001 (snapshot 1.0.0): 53 files, ~0.00 GB
✅ Finished ds000001. 53 files, 0.4 MB in 4.9s.

=== PROJECT 5: real downloaded OpenNeuro metadata EDA ===
dataset                : ds000001
snapshot               : 1.0.0
local metadata root    : /tmp/tmpqc0n3k7y/ds000001
event files summarized : 48
quality bids score     : 90.0
quality ml score       : 78.2
html report bytes      : 31896

=== Real event-label distributions ===
path                                                            | rows | label_column | classes | counts                                                                                     | imbalance
----------------------------------------------------------------+------+--------------+---------+--------------------------------------------------------------------------------------------+----------
sub-16/func/sub-16_task-balloonanalogrisktask_run-01_events.tsv | 121  | trial_type   | 4       | {"cash_demean": 7, "control_pumps_demean": 52, "explode_demean": 8, "pumps_demean": 54}    | 7.714    
sub-16/func/sub-16_task-balloonanalogrisktask_run-02_events.tsv | 141  | trial_type   | 4       | {"cash_demean": 11, "control_pumps_demean": 34, "explode_demean": 8, "pumps_demean": 88}   | 11.0     
sub-16/func/sub-16_task-balloonanalogrisktask_run-03_events.tsv | 157  | trial_type   | 4       | {"cash_demean": 11, "control_pumps_demean": 54, "explode_demean": 9, "pumps_demean": 83}   | 9.222    
sub-15/func/sub-15_task-balloonanalogrisktask_run-01_events.tsv | 149  | trial_type   | 4       | {"cash_demean": 15, "control_pumps_demean": 47, "explode_demean": 13, "pumps_demean": 74}  | 5.692    
sub-15/func/sub-15_task-balloonanalogrisktask_run-02_events.tsv | 127  | trial_type   | 4       | {"cash_demean": 17, "control_pumps_demean": 49, "explode_demean": 7, "pumps_demean": 54}   | 7.714    
sub-15/func/sub-15_task-balloonanalogrisktask_run-03_events.tsv | 135  | trial_type   | 4       | {"cash_demean": 28, "control_pumps_demean": 25, "explode_demean": 2, "pumps_demean": 80}   | 40.0     
sub-14/func/sub-14_task-balloonanalogrisktask_run-01_events.tsv | 168  | trial_type   | 4       | {"cash_demean": 11, "control_pumps_demean": 60, "explode_demean": 11, "pumps_demean": 86}  | 7.818    
sub-14/func/sub-14_task-balloonanalogrisktask_run-02_events.tsv | 173  | trial_type   | 4       | {"cash_demean": 16, "control_pumps_demean": 42, "explode_demean": 14, "pumps_demean": 101} | 7.214    
sub-14/func/sub-14_task-balloonanalogrisktask_run-03_events.tsv | 162  | trial_type   | 4       | {"cash_demean": 20, "control_pumps_demean": 37, "explode_demean": 9, "pumps_demean": 96}   | 10.667   
sub-13/func/sub-13_task-balloonanalogrisktask_run-01_events.tsv | 151  | trial_type   | 4       | {"cash_demean": 11, "control_pumps_demean": 35, "explode_demean": 12, "pumps_demean": 93}  | 8.455    
sub-13/func/sub-13_task-balloonanalogrisktask_run-02_events.tsv | 172  | trial_type   | 4       | {"cash_demean": 15, "control_pumps_demean": 73, "explode_demean": 8, "pumps_demean": 76}   | 9.5      
sub-13/func/sub-13_task-balloonanalogrisktask_run-03_events.tsv | 162  | trial_type   | 4       | {"cash_demean": 15, "control_pumps_demean": 59, "explode_demean": 8, "pumps_demean": 80}   | 10.0     
RESULT: real EDA project passed

##### running 6_conversion_artifact #####
📥 Downloading ds000001 (snapshot 1.0.0): 53 files, ~0.00 GB
✅ Finished ds000001. 53 files, 0.4 MB in 5.0s.

=== PROJECT 6: real metadata-to-Parquet conversion artifact ===
dataset       : ds000001
snapshot      : 1.0.0
output format : parquet
samples       : 7723
subjects      : 16
splits        : {'train': 5459, 'val': 838, 'test': 1426}
artifact id   : dbfa93e8bce17d70

=== Real artifact directory ===
file                  
----------------------
_SUCCESS              
artifact_manifest.json
qortex_provenance.json
shard_00000.parquet   
shard_00001.parquet   
shard_00002.parquet   
shard_00003.parquet   
shard_00004.parquet   
shard_00005.parquet   
shard_00006.parquet   
shard_00007.parquet   
shard_00008.parquet   
shard_00009.parquet   
shard_00010.parquet   
shard_00011.parquet   
shard_00012.parquet   
shard_00013.parquet   
shard_00014.parquet   
shard_00015.parquet   
shard_00016.parquet   

=== Real artifact summary ===
artifact_id      | dataset_id | snapshot | format  | n_samples | n_subjects | splits                                   
-----------------+------------+----------+---------+-----------+------------+------------------------------------------
dbfa93e8bce17d70 | ds000001   | 1.0.0    | parquet | 7723      | 16         | {"test": 1426, "train": 5459, "val": 838}
RESULT: real conversion project passed

##### running 7_readiness_report_project #####
📥 Downloading ds000001 (snapshot 1.0.0): 53 files, ~0.00 GB
✅ Finished ds000001. 53 files, 0.4 MB in 3.8s.

=== PROJECT 7: real dataset readiness analysis ===
dataset        : ds000001
snapshot       : 1.0.0
score          : 84.0
recordings     : 80
loadable       : 80
event complete : 48/80
label ready    : 48/80
can download   : True
can convert    : True
Dataset : ds000001 (snapshot 1.0.0)
Score   : 84.0/100
Records : 80 logical recording(s)
Events  : 48/80 event-complete
Labels  : 48/80 label-ready

=== Real readiness findings ===
(no rows)
RESULT: real readiness project passed

##### running 8_behavior_loader_project #####
📥 Downloading ds000001 (snapshot 1.0.0): 53 files, ~0.00 GB
✅ Finished ds000001. 53 files, 0.4 MB in 4.7s.

=== PROJECT 8: real BIDS events loader ===
dataset         : ds000001
event file      : sub-01/func/sub-01_task-balloonanalogrisktask_run-01_events.tsv
can load        : True
rows            : 158
columns         : onset, duration, trial_type, cash_demean, control_pumps_demean, explode_demean, pumps_demean, response_time
label column    : trial_type
label preview   : ['cash_demean', 'explode_demean', 'pumps_demean']
samples emitted : 158

=== Real event samples ===
label | label_name     | onset  | duration | response_time
------+----------------+--------+----------+--------------
3     | pumps_demean   | 0.061  | 0.772    | 2.42         
3     | pumps_demean   | 4.958  | 0.772    | 0.578        
3     | pumps_demean   | 7.179  | 0.772    | 0.766        
3     | pumps_demean   | 10.416 | 0.772    | 0.84         
3     | pumps_demean   | 13.419 | 0.772    | 1.462        
2     | explode_demean | 16.754 | 0.772    | None         
3     | pumps_demean   | 24.905 | 0.772    | 1.295        
3     | pumps_demean   | 27.454 | 0.772    | 1.083        
0     | cash_demean    | 30.111 | 0.772    | 1.498        
3     | pumps_demean   | 38.449 | 0.772    | 0.656        
3     | pumps_demean   | 41.028 | 0.772    | 0.652        
3     | pumps_demean   | 44.529 | 0.772    | 0.864        
RESULT: real behavior-loader project passed

##### running 9_window_split_project #####
📥 Downloading ds000001 (snapshot 1.0.0): 53 files, ~0.00 GB
✅ Finished ds000001. 53 files, 0.4 MB in 4.5s.

=== PROJECT 9: real event samples and subject-safe split assignment ===
dataset       : ds000001
event samples : 7723
train         : 5459
val           : 838
test          : 1426
subjects      : 16

=== Real split preview ===
subject | split | task                  | label_name     | onset 
--------+-------+-----------------------+----------------+-------
01      | test  | balloonanalogrisktask | pumps_demean   | 0.061 
01      | test  | balloonanalogrisktask | pumps_demean   | 4.958 
01      | test  | balloonanalogrisktask | pumps_demean   | 7.179 
01      | test  | balloonanalogrisktask | pumps_demean   | 10.416
01      | test  | balloonanalogrisktask | pumps_demean   | 13.419
01      | test  | balloonanalogrisktask | explode_demean | 16.754
01      | test  | balloonanalogrisktask | pumps_demean   | 24.905
01      | test  | balloonanalogrisktask | pumps_demean   | 27.454
01      | test  | balloonanalogrisktask | cash_demean    | 30.111
01      | test  | balloonanalogrisktask | pumps_demean   | 38.449
01      | test  | balloonanalogrisktask | pumps_demean   | 41.028
01      | test  | balloonanalogrisktask | pumps_demean   | 44.529
01      | test  | balloonanalogrisktask | pumps_demean   | 47.692
01      | test  | balloonanalogrisktask | cash_demean    | 51.102
01      | test  | balloonanalogrisktask | pumps_demean   | 59.031
01      | test  | balloonanalogrisktask | pumps_demean   | 61.85 
RESULT: real split project passed

##### running 10_local_index_validation_cache_project #####
📥 Downloading ds000001 (snapshot 1.0.0): 53 files, ~0.00 GB
✅ Finished ds000001. 53 files, 0.4 MB in 4.2s.

=== PROJECT 10: real local metadata index and validation report artifacts ===
dataset          : ds000001
indexed files    : 53
missing remote   : 83
extra local      : 0
cached valid     : False
validation score : 97.0
resolved issues  : 1
Dataset path     : /tmp/tmprgiq1nf3/ds000001
Indexed files    : 53
Indexed dirs     : 0
Missing remote   : 83
Extra local      : 0
Size mismatches  : 0
Consistent       : False
Dataset : /tmp/tmprgiq1nf3/ds000001
Valid   : False
Score   : 97.0/100
Errors  : 0
Warnings: 1
Ignored : 0
WARNING: METADATA_ONLY_LOCAL_TREE [/tmp/tmprgiq1nf3/ds000001]: This local tree intentionally contains only downloaded OpenNeuro metadata files.

=== Real validation report exports ===
path                                  | bytes
--------------------------------------+------
/tmp/tmprgiq1nf3/real_validation.json | 576  
/tmp/tmprgiq1nf3/real_validation.md   | 367  
/tmp/tmprgiq1nf3/real_validation.html | 1113 
RESULT: real index/validation project passed

##### running 11_catalog_project #####

=== PROJECT 11: real OpenNeuro catalog refresh and search ===
indexed this run : 25
catalog count    : 25
result count     : 10
catalog path     : /tmp/tmp2hsd1zjq/cache/catalog/catalog.duckdb

=== Real catalog results ===
dataset_id | name                                                                                       | subjects | modalities     | snapshot
-----------+--------------------------------------------------------------------------------------------+----------+----------------+---------
ds007857   | Michigan Neural Distinctiveness (MiND) Study                                               | 289      | ["mri"]        | 1.0.1   
ds007990   | Subjective tinnitus severity correlates with objective changes in human auditory cortex as | 63       | ["nirs"]       | 1.0.0   
ds007993   | pupilomiletics                                                                             | 61       | ["beh"]        | 1.0.0   
ds007864   | Neuroepo multisession Phase II and III                                                     | 45       | ["eeg"]        | 1.0.0   
ds007987   | Raw resting-state EEG dataset with alternating eyes-open and eyes-closed recordings in hea | 43       | ["eeg"]        | 1.0.0   
ds007968   | Narrative Reward                                                                           | 40       | ["eeg", "beh"] | 1.0.0   
ds007952   | Visual Aesthetic Value under Continuous Flash Suppression                                  | 37       | ["beh"]        | 1.0.0   
ds007859   | A sex-balanced longitudinal developmental resting-state fMRI rat dataset                   | 36       | ["mri"]        | 1.0.4   
ds007908   | Multi-scale, multi-modal imaging assessment of trajectories of cognitive impairment in Mul | 28       | ["mri"]        | 1.0.0   
ds007865   | Placebo Neuroepo multisession Phase II and III                                             | 24       | ["eeg"]        | 1.0.0   
RESULT: real catalog project passed

##### running 12_cli_project #####

=== PROJECT 12: installed CLI against real OpenNeuro metadata ===
command               : /home/arman/.venv/bin/qortex
help return code      : 0
metadata return code  : 0
metadata output lines : 8
qortex metadata output:
CHANGES  0.3 KB
README  1.2 KB
dataset_description.json  0.6 KB
participants.tsv  0.2 KB
task-balloonanalogrisktask_bold.json  0.1 KB
sub-16/func/sub-16_task-balloonanalogrisktask_run-01_events.tsv  6.7 KB
sub-16/func/sub-16_task-balloonanalogrisktask_run-02_events.tsv  7.6 KB
sub-16/func/sub-16_task-balloonanalogrisktask_run-03_events.tsv  8.6 KB

RESULT: real CLI project passed

##### running 13_dataset_facade_project #####

=== PROJECT 13: high-level Dataset facade on a real OpenNeuro dataset ===
dataset           : ds000001
snapshot          : 1.0.0
info files        : 136
info subjects     : 16
metadata files    : 53
filtered modality : behavior
filtered files    : 48
first rows        : 5
metadata previews : 5

=== Dataset.info output ===
dataset_id | snapshot | doi                                | n_files | n_subjects | n_sessions | n_tasks | total_size_gb | modalities                  | has_events | has_derivatives
-----------+----------+------------------------------------+---------+------------+------------+---------+---------------+-----------------------------+------------+----------------
ds000001   | 1.0.0    | 10.18112/openneuro.ds000001.v1.0.0 | 136     | 16         | 0          | 1       | 2.416         | ["behavior", "fmri", "mri"] | True       | False          

=== Dataset.metadata_files output ===
path                                                            | size
----------------------------------------------------------------+-----
CHANGES                                                         | 286 
README                                                          | 1175
dataset_description.json                                        | 615 
participants.tsv                                                | 216 
task-balloonanalogrisktask_bold.json                            | 73  
sub-16/func/sub-16_task-balloonanalogrisktask_run-01_events.tsv | 6716
sub-16/func/sub-16_task-balloonanalogrisktask_run-02_events.tsv | 7587
sub-16/func/sub-16_task-balloonanalogrisktask_run-03_events.tsv | 8568
sub-15/func/sub-15_task-balloonanalogrisktask_run-01_events.tsv | 8093
sub-15/func/sub-15_task-balloonanalogrisktask_run-02_events.tsv | 6987
sub-15/func/sub-15_task-balloonanalogrisktask_run-03_events.tsv | 7184
sub-14/func/sub-14_task-balloonanalogrisktask_run-01_events.tsv | 9187

=== Dataset.first_rows('participants.tsv') ===
participant_id | sex | age
---------------+-----+----
sub-01         | F   | 26 
sub-02         | M   | 24 
sub-03         | F   | 27 
sub-04         | F   | 20 
sub-05         | M,  | 22 
RESULT: real Dataset facade project passed

##### running 14_live_openneuro_metadata_project #####

=== PROJECT 14: real live OpenNeuro metadata smoke ===
dataset            : ds000001
snapshot           : 1.0.0
files              : 136
subjects           : 16
table              : participants.tsv
table rows         : 3
description source : remote
description bytes  : 615

=== participants.tsv first rows ===
participant_id | sex | age
---------------+-----+----
sub-01         | F   | 26 
sub-02         | M   | 24 
sub-03         | F   | 27 
RESULT: real live OpenNeuro project passed

all 15 staged Qortex scenario projects passed
