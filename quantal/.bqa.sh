#!/bin/bash

baseline_noise_standard_deviation=2.5
resolution_for_probability_of_release=128
resolution_for_coefficent_of_variation=128
resolution_for_heterogenous_model_alpha=1
resolution_for_number_of_release_sites=96
maximum_number_of_release_sites=191
no_prompt=1

python2 ~/.code/python/quantal/bqamain.py \
$baseline_noise_standard_deviation \
$resolution_for_probability_of_release \
$resolution_for_coefficent_of_variation \
$resolution_for_heterogenous_model_alpha \
$resolution_for_number_of_release_sites \
$maximum_number_of_release_sites \
$no_prompt \

