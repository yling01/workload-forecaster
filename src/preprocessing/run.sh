#!/bin/bash

# Copy and decompress the sample data file
# curl -O http://www.cs.cmu.edu/~malin199/data/tiramisu-sample/tiramisu-sample.tar.gz
# tar -xvzf tiramisu-sample.tar.gz

# Generate and combine query templates
# python ./pre-processor/templatizer.py tiramisu --dir tiramisu-sample/ --output templates
# python ./pre-processor/csv-combiner.py --input_dir templates/ --output_dir tiramisu-combined-csv

# Run through clustering algorithm
# python ./clusterer/online_clustering.py --dir tiramisu-combined-csv/ --rho 0.8
# python ./clusterer/generate-cluster-coverage.py --project tiramisu --assignment online-clustering-results/None-0.8-assignments.pickle --output_csv_dir online-clusters/ --output_dir cluster-coverage/

# # Run forecasting models
# ./forecaster/run_sample.sh

# # Generate ENSEMBLE and HYBRID results
# ./forecaster/generate_ensemble_hybrid.py prediction-results/agg-60/horizon-4320/ar/ prediction-results/agg-60/horizon-4320/noencoder-rnn/ prediction-results/agg-60/horizon-4320/ensemble False
# ./forecaster/generate_ensemble_hybrid.py  prediction-results/agg-60/horizon-4320/ensemble prediction-results/agg-60/horizon-4320/kr prediction-results/agg-60/horizon-4320/hybrid True

####### Admission ########
python ./pre-processor/templatizer.py admissions --dir admission_data/ --output admission_results/templates_admission
python ./pre-processor/csv-combiner.py --input_dir admission_results/templates_admission/ --output_dir admission_results/admission-combined-csv

python ./clusterer/online_clustering.py --dir admission_results/admission-combined-csv/ --rho 0.8
python ./clusterer/generate-cluster-coverage.py --project admissions --assignment admission_results/online-clustering-results/None-0.8-assignments.pickle --output_csv_dir admission_results/online-clusters/ --output_dir admission_results/cluster-coverage/
