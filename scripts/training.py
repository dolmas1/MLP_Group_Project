# libraries
import torch
import logging
from sklearn.metrics import  confusion_matrix
import os

# scripts

from checkpoints import save_checkpoint
from load_safe_metrics import save_metrics


# Cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model,
          optimizer,
          train_loader,
          valid_loader,
          num_epochs,
          file_path,
          best_valid_loss = float("Inf"), 
          early_stopping=10):
    """Train method for the classifier model.

    Args:
        model: the initialized model
        optimizer: the optimizer
        train_loader: iterator for the train set
        valid_loader: iterator for the validation set
        num_epochs (int): number of maximum epochs
        file_path (str): path where to store the results
        best_valid_loss (float): float with the current best validation loss
        early_stopping (int): after how many epochs to use early stopping
    """
    
    # initialize running values

    eval_every = len(train_loader) // 2

    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0

    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    early_stopping_acc = [0] * (early_stopping+1)
    early_stopping_loss = [float("inf")] * (early_stopping+1)

    finished = [False, False]
    best_acc = 0.0

    # training loop
    model.train()
    for epoch in range(num_epochs):
        
        #train_loader.create_batches()
        #for batch in train_loader.batches:
        for batch, batch_labels in train_loader:
           
            labels = batch_labels.to(device)
            text = batch['input_ids'].squeeze(1).to(device)      
            output = model(text, labels)
            #logits = output.logits
            #probs = F.softmax(logits, dim=1)
            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                y_pred = []
                y_true = []
                model.eval()
                with torch.no_grad():                    
                    # validation loop
                    for batch, batch_labels in valid_loader:
                    
                        labels = batch_labels.to(device)
                        text = batch['input_ids'].squeeze(1).to(device)      
                        output = model(text, labels)

                        logits = output.logits
                        y_pred.extend(torch.argmax(logits, 1).tolist())
                        y_true.extend(labels.tolist())

                        loss = output.loss
                        valid_running_loss += loss.item()


                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
                tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=[1,0]).ravel()
                acc = (tp + tn) / (tp + tn + fp + fn)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                logging.info('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Validation Acc: {:.2f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss, acc))
                
                # checkpoint
                if best_valid_loss > average_valid_loss and finished[0] == False:
                    best_valid_loss = average_valid_loss
                    model_path = os.path.join(file_path, "model_best_loss.pt")
                    save_checkpoint(model_path, model, best_valid_loss)
                    
                if best_acc < acc and finished[1] == False:
                    best_acc = acc
                    model_path = os.path.join(file_path, "model_best_acc.pt")
                    save_checkpoint(model_path, model, best_valid_loss)
                
                early_stopping_acc.pop(0)
                early_stopping_acc.append(acc)
                early_stopping_loss.pop(0)
                early_stopping_loss.append(average_valid_loss)

                if all(early_stopping_acc[0] >= val for val in early_stopping_acc[1:]) and finished[0] == False:
                    logging.info("Finished Training for Accuracy!")
                    finished[0] = True

                if all(early_stopping_loss[0] <= val for val in early_stopping_loss[1:]) and finished[1] == False:
                    logging.info("Finished Training for Loss!")
                    finished[1] = True

                if all(finished):
                    metrics_path = os.path.join(file_path, "metrics.pt")
                    save_metrics(metrics_path, train_loss_list, valid_loss_list, global_steps_list)
                    return

    metrics_path = os.path.join(file_path, "metrics.pt")
    save_metrics(metrics_path, train_loss_list, valid_loss_list, global_steps_list)
    
