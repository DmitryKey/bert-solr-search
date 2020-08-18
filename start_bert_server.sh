#!/usr/bin/env bash

bert-serving-start -model_dir bert-model/uncased_L-12_H-768_A-12/ -num_worker=4 -max_seq_len=500 -http_port 8081
