# Report for Coding Assignment 2

## Overview

I implemented a CBOW model and tested it with context window sizes of 2 and 4. 
Model and word_vec data is stored in the "outputs" folder, while plots of accuracy and loss for the different context sizes tested are in the "graphs" folder, along with the text output file of the downstream evaluations run. 


## Running the Code
I added some additional CLI flags for running the code, notably --context_size, which 
you can use to set the context window size. Additionally, --embedding_dim is also a flag you can set. 

## Implementation choices

I used a CBOW model with an Embedding Layer and Linear Layer to keep things simple. For the optimizer, I chose Adam to not overcomplicate the model and cross entropy loss as the criterion. 

I tested batch sizes ranging from 32-512, which all had about the same accuracy/loss. In the end, I went with 32 because it seemed to be the fastest. 

I chose 128 for embedding_dim in my tests, since I didn't want it to be too small or too large either.

For vocab_size, I used 3000 since that seemed to be the best that my machine could do. 

As for splitting the train/test data, I looped over every word between `start` and `end` tokens for that line, so `start` and `end` could be included in context but not as target words. I did not include any padding tokens as target words because that had caused it to be a lot slower, but if a word was at the edge (i.e. for window size 4), then 0 was used as context. 

Each epoch took roughly 20 minutes to complete, so for preliminary testing purposes I had used 5 epochs. For reference I am using a Macbook Air (not M1) laptop. For larger epoch sizes that I tested (15-30), I ran it overnight.

## Analyzing the code

The in vivo task being evaluated is splitting the data into train and test sets randomly using a dataset of popular books, with model iteration on train and testing on the test set. This is measuring the accuracy of predicting the target word from a given context (list of words). 

The in vitro task being evaluated is the analogies from the analogies_v3000_1309.json file, with 2 root words being and a grammatical part-of-speech transformation on them. 

As for the evaluation criteria, "exact" is the proportion of words that were predicted exactly correct. On the other hand, "MR" (mean rank) predicts the average distance from the top guess by the model the actual word is, i.e. if predictions are ordered by probability, and the nth rank prediction is the actual word, then our rank would be n. "MRR" is the reciprocal of this. 

Since the downstream dataset is pretty specific in terms of parts of speech, I believe the model was not as effective on it because the common vocab words differed significantly compared to the words typically present in books. If the vocab_size were to be increased, we would probably have better downstream results since there would be less UNK tokens.

## Results

My best results in vitro were obtained with the window_size 4 model in 5 epochs and best in vivo were obtained with a window_size 2 and 30 epochs.

All the other experiments aside from the 30-epoch one yielded scores of 0 for exact, probably because the hadn't done enough iterations to actually train well enough to be applied to another task. 

For both context_size graphs, validation accuracy was much more volatile while training accuracy seemed to plateau. Similar result with loss, but less variance in validation and training also plateaued less. Both would probably converge eventually if I had used more epochs. 

### Context Window Size 2
Most of my testing was done on a context window of size 2, mainly because that was quicker to run. The results graph and downstream eval of this in the graphs folder is from a test run with 15 epochs.

For the in vitro tasks, the results can be seen in the graphs, but accuracy was around.81 and loss around .66. 

My best results in vivo were attained when I rain it with 30 epochs (which I only tested on context_size 2). It took about 10 hours to finish running this, but I accidentally overwrote the file in my next run and can't rerun it in time as it takes 10 hours :( 
The screenshot of the results is here though:
![Alt text](./IMG_4606.png?raw=true "Results")

The results were .0008 for exact, .0031 for MMR and 318 for MR, which is pretty bad but still better than random guessing. 

You can also replicate this using the following command: 
`--analogies_fn analogies_v3000_1309.json --data_dir books/ --num_epochs 30 --word_vector_fn word_vec.txt --output_dir outputs --batch_size 32 --embedding_dim 128 --context_size 2 `
followed by the downstream_eval run command.



### Comparing Context Window Size 4

My best in vitro results were obtained with the model with a context window size 4. It
 had noticeably better results with accuracy hovering around .85 for both validation and training, with 5 epochs of training. It also had a lower loss value, which converged to about .44.

These results make sense, because having more context words adds specificity to the model compared to the 2-context model, leading to overall better performance.  

While I would have liked to see the results with more epochs, with the limited resources of my computer this wasn't feasible. This model took longer per-epoch than the window size 2 model, so training it on 30 epochs to see comparable downstream results would have taken way too long. However, I would presume it performs slightly better than the context_window 2 due to its higher initial accuracy and better MR scores for 5 epochs than 15 epochs for a window size 2.