## Description
Implementation of the work presented in **"CNN based Query by Example Spoken Term Detection"**.

We have included some example groundtruth files for training as well as development set.

The posteriors features are extracted using the setup presented in the following paper:
**"High-performance query-by-example spoken term detection on the SWS 2013 evaluation"**.

The input feature files for training/evaluation are in pytorch readable format which are 
saved as python dictionaries. The keys are the names of the files in *'groundtruth files'*
and values are the features in matrix format.

## Training

```
python query_detection_dtw_cnn.py -optim adam -learning_rate 0.0001 -input_size 152 -batch_size 50 -layers 9 -depth 30 -dropout 0.2 -loss_threshold 0.1 -n_valid 50 -max_batch_dev 250 -max_batch_train 1000
```

## Evaluation

```
python query_detection_dtw_cnn_evaluation.py -input_size 152 -depth 30 -load_model -modelpath cnn_qbe_std_model.pt -outdir outpath -query_list dev_queries_sample_list.txt -search_list search_utterances_sample_list.txt
```

## Reference
```
@inproceedings{ram2018cnn,
  title={CNN based Query by Example Spoken Term Detection},
  author={Ram, Dhananjay and Miculicich, Lesly and Bourlard, Herv{\'e}},
  booktitle={Nineteenth Annual Conference of the International Speech Communication Association (INTERSPEECH)},
  year={2018}
}
```

## Contact:

dhananjay.ram@idiap.ch
