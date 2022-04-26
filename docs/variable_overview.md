# Variable Overview

## Logging

This provides an overview of the meaning of the variables that are logged by default using Tensorboard.

### Splits

-   `{split}`: Is used to indicated the split. Can be `val`, `train` or `test`.

### Scalars

| Variable Name                      | Description                                                                                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| epoch                              | the epoch                                                                                                                                               |
| {split}\_loss/kl_div               | The KL divergence of the latent space                                                                                                                   |
| {split}\_loss/next_FIELD_NAME      | For each element in the sequences we predict the value of FIELD_NAME of the next element in the sequence. This is the according loss.                   |
| {split}\_loss/churn                | The churn loss.                                                                                                                                         |
| {split}\_loss/composite            | The sum of all losses for the attributes of events and profiles.                                                                                        |
| {split}\_loss/dt                   | Synonym for `loss_tau`                                                                                                                                  |
| {split}\_loss/model_fit            | The loss associated with $p(y\|x,z)$. This is the model fit. It is $\text{loss\_model\_fit}=\text{loss\_churn}+\text{loss\_tau}+\text{loss\_composite}$ |
| {split}\_loss/total                | The overall loss: $\text{loss\_total}=w*\text{loss\_kl\_div}+\text{loss\_model\_fit}$                                                                   |
| {split}\_loss/tau                  | The loss for estimating time frequency. Here the $\tau$ of the exponential distribution.                                                                |
| p_churn/mean_last_element\_{split} | The mean of the predicted churn probabilities of the last event for each customer. See `Histogramm.last_p_churn_predicted`.                             |
| p_churn/mean\_{split}              | The mean of the predicted churn probabilities of all events for each custumer. See `Histogramm.p_churn_predicted`.                                      |

### Histograms

| Variable Name          | Description                                                                                                                                                                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| lambda_True            | The distribution of the true even rate parameter, lambda, per customer in the BTYD dataset. Should look like a Gamma Mixture distribution with the parameters as defined.                                                                          |
| last_p_churn_predicted | The distribution of predicted churn probabilities for the last event in each customer's sequence. This does not under-estimate the churn probability due to the class bias induced by the varying sequence length driven by the churn probability. |
| p_churn_True           | The distribution of the true churn probabilities per customer in the BTYD dataset. Should look like a Beta Mixture distribution with the parameters as defined.                                                                                    |
| p_churn_predicted      | The distribution of predicted churn probabilities for each event. As users with many events have lower churn probability, this will under-estimate the average churn probability per customer.                                                     |
| t_to_next_Pred         | The distribution of times between events in the dataset.                                                                                                                                                                                           |
| t_to_next_True         | The distribution of predicted times between events.                                                                                                                                                                                                |
| log_t_to_next_Pred     | log time of `t_to_next_Pred`                                                                                                                                                                                                                       |
| log_t_to_next_True     | log time of `t_to_next_True`                                                                                                                                                                                                                       |
| t_to_next_lambda       | The estimate for the lambda of the generative process: `1/t_to_next_Pred`                                                                                                                                                                          |

### Projections

| Variable Name | Description                                     |
| ------------- | ----------------------------------------------- |
| label         | mode index of the BTYD multimodal dataset model |

## Batch Dictionary

The `SequenceLoader` outputs a batch of multiple variables. Here is an overview of their meaning and dimensions.

`N` number of customers

`K_n` number of events for customer `n`

`K` total number of event. (sum of `K_n`'s)

| Key                         | dtype              | Dimensions                                              | Description                                                                                    |
| --------------------------- | ------------------ | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| t                           | np.ndarray[object] | `N` + `K`                                               | Sequence of events. Each datapoints sequence starts with <START_TOKEN>                         |
| dt                          | torch.float32      | `N` + `K`                                               | dt_i = t_i - t\_{i-1}. dt_i = 0 if t\_{i-1} does not exist, or t_i or t\_{i-1} = <START_TOKEN> |
| <CONTINUOUS_VARIABLE>       | torch.float32      | x                                                       | N/A                                                                                            |
| <DISCRETE_VARIABLE>         | np.ndarray[str]    | x                                                       | N/A                                                                                            |
| offsets                     | torch.int64        | `N+1` ?                                                 | Indicates the index of the <START_TOKEN> in each input sequence                                |
| t_to_now                    | torch.float32      | `N`                                                     | Time from the last event to now (the models asof_time)                                         |
| next_dt                     | .                  | `K` (+ `N` start tokens - `N` first elements from `dt`) | next_dt_i = dt\_{i-1}                                                                          |
| next\_<CONTINUOUS_VARIABLE> | torch.float32      | `K`                                                     | next\_<CONTINUOUS_VARIABLE>\_i = <CONTINUOUS_VARIABLE>\_{i-1}                                  |
| next\_<DISCRETE_VARIABLE>   | torch.int64        | `K`                                                     | next\_<DISCRETE_VARIABLE>\_i = <DISCRETE_VARIABLE>\_{i-1}                                      |

N 109
K_n = [22,28,8,3,26,2,1,5...]
962 events

## Sequence Layout

The variables in the batch outputted by the `SequenceLoader` and by the model have different lengths for memory-efficiency. Here we outline the data.

### Example Data

N customers A,B and C with sequences:

A1, A2, A3

B1, B2

C1, C2

K total events

T is start token

### Sequence Lengths

| Dictionary          | Key     | Length | Value                               |
| ------------------- | ------- | ------ | ----------------------------------- |
| batch               | t       | N + K  | T, A1, A2, A3, T, B1, B2, T, C1, C2 |
| batch               | dt      | N + K  | T, A1, A2, A3, T, B1, B2, T, C1, C2 |
| batch               | next_dt | K      | T, A2, A3, T, B2, T, C2             |
| model_output        | next_dt | N + K  | ?, A2, A3, A4, ?, B2, B3, ?, C2, C3 |
| y_pred (loss input) | next_dt |        | ?, A2, A3, ?, B2, ?, C2             |

y_pred (loss input) is after self.preprosessing in Composite Loss
