import numpy as np

class Callback:
    def __init__(self):
        pass

    def on_epoch_begin(self):
        return NotImplemented
    
    def on_epcoh_end(self):
        return NotImplemented
    
    def on_batch_begin(self):
        return NotImplemented
    
    def on_batch_end(self):
        return NotImplemented
    

class ReduceLronPlateau(Callback):
    def __init__(self, 
                 patience=10, 
                 factor=0.1, 
                 min_delta=1e-5, 
                 min_lr=0):
        '''
        | Callback should be called on the begining of each epoch, 
        | check if the monitored metric (loss) doesn't get optimized for number of epochs

        Args: 
            patience:   number of epochs to wait before updating learning rate.
            factor:     factor of updating learning rate -> newLR = oldLR * factor
            min_delta:  minimum delta in loss before considering there is no optimizing in our metric
            min_lr:     minimum learning rate that can't get below.

        '''
        self.patience = patience
        self.factor = factor
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.last_loss = 1e18
        self.last_p = 0
        super().__init__()

    def on_epoch_begin(self, loss:float, lr:float) -> float:
        '''
        On Start of Epoch: 
            
        Args: 
            loss:   loss of last epoch.
            lr:     learning rate of last epcoh.

        Returns: 
            lr:     learning rate algorithm should use in current epoch.
        '''

        if lr==self.min_lr:
            return lr
        
        if np.abs(loss - self.last_loss)  <= self.min_delta:
            self.last_loss = loss
            if self.last_p < self.patience:
                self.last_p+=1
                return lr
            else:
                new_lr = lr * self.factor
                if new_lr < self.min_lr:
                    return self.min_lr
                else:
                    self.last_p=0
                    return new_lr
        else:
            self.last_p = 0
            self.last_loss = loss
            return lr

class EarlyStopping(Callback):
    def __init__(self, 
                 min_delta, 
                 patience):
        
        '''
        | EarlyStopping: a callback should be called at the begining of each epoch
        | it monitors the loss of model and if loss doesn't get optimized for number of epochs it terminates training process.

        Args: 
            min_delta:  minimum delta in loss before considering there is no optimizing in our metric.
            patience:   number of epochs to wait before terminating the process of learning.
        '''
        self.min_delta = min_delta
        self.patience = patience
        self.last_p = 0
        self.last_l = 1e18

        super().__init__()

    def on_epoch_begin(self, loss:float) -> int:
        '''
        On Start of Epoch: 
            
        Args: 
            loss:   loss of last epoch.

        Returns: 
            flag:
                0: don't terminate learning.
                1: terminate learning.
        '''
        if np.abs(loss - self.last_l) <= self.min_delta:
            self.last_l = loss
            if self.last_p < self.patience:
                self.last_p += 1
                return 0
            else: 
                return 1
        else:
            self.last_l = loss
            self.last_p = 0
            return 0
        

class LearningRateScheduler(Callback):
    def __init__(self, 
                 schedule):
        '''
        | Learning Rate Scheduler: a callback should be called at the begining of each epoch
        | it schedule learning rate upon a function of epochs.

        Args: 
            schedule: Function that controls the learning rate.
        '''

        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch:int, lr:float) -> float:
        '''
        Args: 
            epoch:  Number of the begining epoch.
            lr:     Learning rate to update.

        Returns: 
            lr:     The new learning rate.    
        '''
        return self.schedule(epoch, lr)