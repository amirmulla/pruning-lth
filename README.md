The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks:
Jonathan Frankle, Michael Carbin

Paper Abstract:

Neural network pruning techniques can reduce the parameter counts of trained networks by over 90%, decreasing storage requirements and improving computational performance of inference without compromising accuracy. However, contemporary experience is that the sparse architectures produced by pruning are difficult to train from the start, which would similarly improve training performance.

We find that a standard pruning technique naturally uncovers subnetworks whose initializations made them capable of training effectively. Based on these results, we articulate the "lottery ticket hypothesis:" dense, randomly-initialized, feed-forward networks contain subnetworks ("winning tickets") that - when trained in isolation - reach test accuracy comparable to the original network in a similar number of iterations. The winning tickets we find have won the initialization lottery: their connections have initial weights that make training particularly effective.

We present an algorithm to identify winning tickets and a series of experiments that support the lottery ticket hypothesis and the importance of these fortuitous initializations. We consistently find winning tickets that are less than 10-20% of the size of several fully-connected and convolutional feed-forward architectures for MNIST and CIFAR10. Above this size, the winning tickets that we find learn faster than the original network and reach higher test accuracy.

Code Description:

This code implements The Lottery Ticket Hypothesis paper by Jonathan Frankle and Michael Carbin. Code allows to run 3 choosen models from the paper - VGG19, Conv-4 and Lenet with various options as described below. 

python main.py --help

usage: main.py [-h] 
               
               [--model_type MODEL_TYPE] [--batch_size BATCH_SIZE]

               [--prune_approach PRUNE_APPROACH] [--prune_method PRUNE_METHOD]
               
               [--train_epochs TRAIN_EPOCHS] [--prune_ratio PRUNE_RATIO]
               
               [--prune_output_layer PRUNE_OUTPUT_LAYER]
               
               [--prune_rounds PRUNE_ROUNDS]
               
               [--winning_ticket_reinit WINNING_TICKET_REINIT]
               
               [--dp_ratio DP_RATIO] [--prune_ratio_conv PRUNE_RATIO_CONV]
               
               [--find_matching_tickets FIND_MATCHING_TICKETS]
               
               [--stabilize_epochs STABILIZE_EPOCHS]
               
               [--use_lr_scheduler USE_LR_SCHEDULER]
               

optional arguments:
  
  --model_type MODEL_TYPE
  
    lenet | conv4 | vgg19
                        
  --batch_size BATCH_SIZE
  
  --prune_approach PRUNE_APPROACH
  
    iterative | oneshot | random
                        
  --prune_method PRUNE_METHOD
  
    local | global
                        
  --train_epochs TRAIN_EPOCHS
  
  --prune_ratio PRUNE_RATIO
  
     Initial pruning ratio (0-100)
                        
  --prune_output_layer PRUNE_OUTPUT_LAYER
  
     Apply pruning to output layer
                        
  --prune_rounds PRUNE_ROUNDS
  
    Number of pruning rounds
                        
  --winning_ticket_reinit WINNING_TICKET_REINIT
  
    Random reinitialization of winning tickets (from pruning with rewind)
                        
  --dp_ratio DP_RATIO   Dropout ratio (0-100)
  
  --prune_ratio_conv PRUNE_RATIO_CONV
  
    Initial pruning ratio (0-100) for Convolution layers
                        
  --find_matching_tickets FIND_MATCHING_TICKETS
  
    apply pre pruning training to stabilize model
                        
  --stabilize_epochs STABILIZE_EPOCHS
  
    pre pruning training epochs to stabilize model
                        
  --use_lr_scheduler USE_LR_SCHEDULER
  
    Use learning rate scheduler
                        

VGG Example:

    python main.py --model_type vgg19 --prune_approach iterative --prune_method global --train_epochs 100 --prune_ratio 20 --prune_ratio_conv 20 --prune_rounds 8  prune_output_layer 0 --dp_ratio 0 --winning_ticket_reinit 1 --use_lr_scheduler 1
