#!/usr/bin/env bash

NUM_WORKERS=1
MAX_SEQUENCE_LENGTH=500

bert-serving-start -model_dir bert-model/uncased_L-12_H-768_A-12/ -num_worker=$NUM_WORKERS -max_seq_len=$MAX_SEQUENCE_LENGTH -http_port 8081
