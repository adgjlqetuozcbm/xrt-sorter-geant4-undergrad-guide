# v8A v7 hard-negative diagnostics

Generated: 2026-05-12T14:38:52+00:00

- Decision: `v7_hard_negative_diagnostic_completed_next_step_targeted_hm_ilmenite_robustness_design`
- Diagnostic completed: `true`
- Main finding: `combined_stress_high creates a Hematite-Ilmenite hard-negative failure in the 20-material scout`

## Nearest Train Centroids

query_split,query_profile,query_material,train_profile_filter,candidate_split,candidate_profile,candidate_material,candidate_n,euclidean_distance_zscore
stress_holdout,combined_stress_high,Hematite,__all_train_profiles__,train,combined_train_moderate,Ilmenite,6,4.278744765464851
stress_holdout,combined_stress_high,Hematite,__all_train_profiles__,train,combined_train_moderate,Hematite,6,5.246351639154637
stress_holdout,combined_stress_high,Hematite,__all_train_profiles__,train,combined_train_moderate,Siderite,6,5.513871706658044
stress_holdout,combined_stress_high,Hematite,__all_train_profiles__,train,resolution_blur_train_moderate,Ilmenite,6,5.621787689989686
stress_holdout,combined_stress_high,Hematite,__all_train_profiles__,train,resolution_blur_train_moderate,Hematite,6,6.394942498960324
stress_holdout,combined_stress_high,Hematite,__all_train_profiles__,train,combined_train_moderate,Pyrite,6,6.630090359555052
stress_holdout,combined_stress_high,Hematite,__all_train_profiles__,train,resolution_blur_train_moderate,Siderite,6,6.698247681300673
stress_holdout,combined_stress_high,Hematite,__all_train_profiles__,train,resolution_blur_train_moderate,Pyrite,6,7.383944432532149
stress_holdout,combined_stress_high,Hematite,__all_train_profiles__,train,combined_train_moderate,Rutile,6,7.602112745529282
stress_holdout,combined_stress_high,Hematite,__all_train_profiles__,train,combined_train_moderate,Magnetite,6,8.26939328289489
stress_holdout,combined_stress_high,Hematite,__all_train_profiles__,train,combined_train_moderate,Dolomite,6,8.709916017530592
stress_holdout,combined_stress_high,Hematite,__all_train_profiles__,train,combined_train_moderate,Magnesite,6,8.821200197352438
stress_holdout,combined_stress_high,Hematite,combined_train_moderate,train,combined_train_moderate,Ilmenite,6,4.278744765464851
stress_holdout,combined_stress_high,Hematite,combined_train_moderate,train,combined_train_moderate,Hematite,6,5.246351639154637
stress_holdout,combined_stress_high,Hematite,combined_train_moderate,train,combined_train_moderate,Siderite,6,5.513871706658044
stress_holdout,combined_stress_high,Hematite,combined_train_moderate,train,combined_train_moderate,Pyrite,6,6.630090359555052

## Stress Pair Distances

split,physical_perturbation_profile,left_material,right_material,left_n,right_n,euclidean_distance_zscore
stress_holdout,combined_stress_high,Hematite,Ilmenite,4,4,2.0563585737530565
stress_holdout,combined_stress_high,Ilmenite,Siderite,4,4,4.171357269341888
stress_holdout,combined_stress_high,Hematite,Magnetite,4,4,4.303964775568698
stress_holdout,combined_stress_high,Hematite,Siderite,4,4,5.168422100529535
stress_holdout,combined_stress_high,Ilmenite,Rutile,4,4,5.2950988915080766
stress_holdout,combined_stress_high,Hematite,Rutile,4,4,5.375050473072143
stress_holdout,combined_stress_high,Rutile,Siderite,4,4,5.51441042368959
stress_holdout,combined_stress_high,Ilmenite,Magnetite,4,4,5.705327407818541
stress_holdout,combined_stress_high,Magnetite,Rutile,4,4,7.781799929461571
stress_holdout,combined_stress_high,Goethite,Siderite,4,4,8.229776501214438
stress_holdout,combined_stress_high,Magnetite,Siderite,4,4,8.405147535366488
stress_holdout,combined_stress_high,Goethite,Ilmenite,4,4,9.407805125020545
stress_holdout,combined_stress_high,Goethite,Hematite,4,4,9.726952188010314
stress_holdout,combined_stress_high,Goethite,Rutile,4,4,10.477162086040362
stress_holdout,combined_stress_high,Goethite,Magnetite,4,4,10.802095105808721

## Peak Sum Focus

split,physical_perturbation_profile,seed_block,material,peak_sum_hematite,peak_sum_magnetite,peak_sum_ilmenite,peak_sum_goethite,peak_sum_siderite,peak_sum_rutile,diffraction_window_hematite_unique_sum,diffraction_window_magnetite_unique_sum,diffraction_window_all_peaks_sum,hematite_minus_ilmenite_peak_sum,magnetite_minus_ilmenite_peak_sum
stress_holdout,combined_stress_high,ho_scale_001,Hematite,0.250375,0.21400000000000002,0.254625,0.312125,0.24212499999999998,0.21875,0.0,0.003875,0.5285,-0.004249999999999976,-0.04062499999999997
stress_holdout,combined_stress_high,ho_scale_001,Ilmenite,0.248875,0.2075,0.24975,0.300875,0.247,0.21687500000000004,0.0,0.002375,0.53325,-0.0008749999999999869,-0.04225000000000001
stress_holdout,combined_stress_high,ho_scale_001,Magnetite,0.244375,0.23850000000000002,0.244375,0.303125,0.22175,0.23775000000000002,0.0,0.0052499999999999995,0.512875,0.0,-0.005874999999999991
stress_holdout,combined_stress_high,ho_scale_002,Hematite,0.254625,0.226625,0.24849999999999997,0.31775000000000003,0.23575000000000002,0.233625,0.0,0.003,0.539625,0.006125000000000019,-0.021874999999999978
stress_holdout,combined_stress_high,ho_scale_002,Ilmenite,0.243375,0.22262500000000002,0.24612499999999998,0.317125,0.24475000000000002,0.22424999999999998,0.0,0.0035,0.543875,-0.0027499999999999747,-0.023499999999999965
stress_holdout,combined_stress_high,ho_scale_002,Magnetite,0.25325,0.24500000000000002,0.246375,0.31074999999999997,0.227375,0.25775,0.0,0.005625,0.5237499999999999,0.0068749999999999645,-0.0013749999999999873
train,combined_train_moderate,tr_scale_001,Hematite,0.3122499999999999,0.26499999999999996,0.3026249999999999,0.384,0.279625,0.27825,0.0,0.005,0.588625,0.009624999999999995,-0.037624999999999964
train,combined_train_moderate,tr_scale_001,Ilmenite,0.300125,0.27424999999999994,0.3071249999999999,0.36487500000000006,0.28525,0.257625,0.0,0.00425,0.594625,-0.006999999999999951,-0.03287499999999999
train,combined_train_moderate,tr_scale_001,Magnetite,0.28800000000000003,0.281875,0.275625,0.365375,0.243875,0.29187499999999994,0.0,0.009375,0.573375,0.012375000000000025,0.006249999999999978
train,combined_train_moderate,tr_scale_002,Hematite,0.30987499999999996,0.27149999999999996,0.30424999999999996,0.39175000000000004,0.277875,0.26712499999999995,0.0,0.007,0.5994999999999999,0.005624999999999991,-0.03275
train,combined_train_moderate,tr_scale_002,Ilmenite,0.2985,0.252625,0.29999999999999993,0.36600000000000005,0.29624999999999996,0.25487499999999996,0.0,0.0033750000000000004,0.604,-0.0014999999999999458,-0.047374999999999945
train,combined_train_moderate,tr_scale_002,Magnetite,0.3,0.3016249999999999,0.283875,0.391,0.23762499999999998,0.29337499999999994,0.0,0.0075,0.58225,0.016125,0.017749999999999932
train,combined_train_moderate,tr_scale_003,Hematite,0.29699999999999993,0.2845,0.31399999999999995,0.39375000000000004,0.27312499999999995,0.26825,0.0,0.0043749999999999995,0.5876250000000001,-0.017000000000000015,-0.02949999999999997
train,combined_train_moderate,tr_scale_003,Ilmenite,0.30124999999999996,0.278875,0.29474999999999996,0.39537500000000003,0.26724999999999993,0.2617499999999999,0.0,0.00375,0.592625,0.006500000000000006,-0.015874999999999972
train,combined_train_moderate,tr_scale_003,Magnetite,0.292875,0.29074999999999995,0.2915,0.385625,0.23750000000000004,0.2916249999999999,0.0,0.008,0.573375,0.001375000000000015,-0.0007500000000000284
validation,combined_validation_mid,va_scale_001,Hematite,0.28787499999999994,0.251625,0.2815,0.35625,0.27925,0.26712499999999995,0.0,0.004875,0.5706249999999999,0.006374999999999964,-0.029874999999999985
validation,combined_validation_mid,va_scale_001,Ilmenite,0.27862499999999996,0.24762499999999998,0.27912499999999996,0.346125,0.28037499999999993,0.253,0.0,0.0022500000000000003,0.574875,-0.0005000000000000004,-0.03149999999999997
validation,combined_validation_mid,va_scale_001,Magnetite,0.276875,0.26699999999999996,0.263625,0.360625,0.23612500000000003,0.28862499999999996,0.0,0.00625,0.5567500000000001,0.013249999999999984,0.0033749999999999614
validation,combined_validation_mid,va_scale_002,Hematite,0.2896249999999999,0.246625,0.28725,0.35275,0.280375,0.273,0.0,0.00575,0.5765,0.002374999999999905,-0.040624999999999994
validation,combined_validation_mid,va_scale_002,Ilmenite,0.281875,0.23262500000000003,0.286375,0.345125,0.289375,0.259,0.0,0.004625,0.581875,-0.004500000000000004,-0.053749999999999964
validation,combined_validation_mid,va_scale_002,Magnetite,0.280125,0.268375,0.26374999999999993,0.34950000000000003,0.22925,0.28037500000000004,0.0,0.007,0.5562499999999999,0.016375000000000084,0.004625000000000046
